#[path = "support/reference.rs"]
mod reference;
use anyhow::Result;
use dinov3_hrx::*;
#[test]
fn rejects_options_before_loading() {
    assert!(
        DINOv3::load(
            "/nonexistent",
            Options {
                device: 0,
                max_batch: 0
            }
        )
        .is_err()
    );
    assert!(
        DINOv3::load(
            "/nonexistent",
            Options {
                device: 0,
                max_batch: 65
            }
        )
        .is_err()
    );
}

fn overflowing_weights<M: ModelSpec>() -> Result<()> {
    use half::bf16;
    use safetensors::{Dtype, tensor::TensorView};

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("model.safetensors");
    for dtype in [Dtype::F32, Dtype::BF16] {
        for value in [100_000f32, -100_000f32] {
            let mut data = vec![0; M::HIDDEN * 3 * 16 * 16 * (dtype.bitsize() / 8)];
            match dtype {
                Dtype::F32 => data[..4].copy_from_slice(&value.to_le_bytes()),
                Dtype::BF16 => data[..2].copy_from_slice(&bf16::from_f32(value).to_le_bytes()),
                _ => unreachable!(),
            }
            let tensor = TensorView::new(dtype, vec![M::HIDDEN, 3, 16, 16], &data)?;
            let bytes =
                safetensors::serialize([("embeddings.patch_embeddings.weight", tensor)], None)?;
            std::fs::write(&path, bytes)?;
            // The first tensor must fail packing, before reading any other
            // tensors or trying to initialize a GPU.
            let error = match DINOv3Model::<M>::load(&path, Options::default()) {
                Ok(_) => panic!("accepted overflowing {dtype:?} weight {value}"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("patch_w[0]"), "{error}");
            assert!(error.contains("finite float16"), "{error}");
        }
    }
    Ok(())
}

fn full_reference_and_changing_batch<M: ModelSpec>() -> Result<()> {
    let path = model_path::<M>()?;
    let model = DINOv3Model::<M>::load(
        &path,
        Options {
            device: 0,
            max_batch: 4,
        },
    )?;
    let images: Vec<f32> = (0..4 * IMAGE_ELEMENTS)
        .map(|i| ((i as f64 * 0.037).sin() * 0.6) as f32)
        .collect();
    let batched = model.forward(&images)?;
    for (i, img) in images.chunks(IMAGE_ELEMENTS).enumerate() {
        let got = model.forward(img)?;
        let want = reference::forward(&path, img, M::NAME)?;
        let compare = |a: &[f32], b: &[f64]| {
            let dot = a.iter().zip(b).map(|(a, b)| *a as f64 * b).sum::<f64>();
            let aa = a.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
            let bb = b.iter().map(|b| b * b).sum::<f64>();
            dot / (aa * bb).sqrt()
        };
        assert!(compare(&got, &want) > 0.9999, "full image {i}");
        assert!(
            compare(&got[..M::HIDDEN], &want[..M::HIDDEN]) > 0.9999,
            "CLS image {i}"
        );
        assert!(
            compare(
                &batched[i * TOKENS * M::HIDDEN..(i + 1) * TOKENS * M::HIDDEN],
                &got.iter().map(|v| *v as f64).collect::<Vec<_>>()
            ) > 0.99999,
            "batch image {i}"
        );
    }
    assert_eq!(model.forward(&images)?, batched);
    assert!(model.forward(&[0.; 3]).is_err());
    assert!(model.forward(&vec![f32::NAN; IMAGE_ELEMENTS]).is_err());
    Ok(())
}

fn compact_rgb_descriptors_preserve_pooling_contract_and_transfer_only_descriptors<M: ModelSpec>()
-> Result<()> {
    let model = DINOv3Model::<M>::load(
        model_path::<M>()?,
        Options {
            device: 0,
            max_batch: 2,
        },
    )?;
    let rgb = (0..2 * IMAGE_ELEMENTS)
        .map(|i| ((i * 13 + i / 379) % 256) as u8)
        .collect::<Vec<_>>();
    let mut input = vec![0f32; rgb.len()];
    for b in 0..2 {
        for c in 0..3 {
            for p in 0..224 * 224 {
                input[b * IMAGE_ELEMENTS + c * 224 * 224 + p] =
                    (rgb[b * IMAGE_ELEMENTS + p * 3 + c] as f32 / 255. - [0.485, 0.456, 0.406][c])
                        / [0.229, 0.224, 0.225][c];
            }
        }
    }
    let masks = (0..392)
        .map(|i| u8::from(i < 196 && i % 14 < 9))
        .collect::<Vec<_>>();
    let tokens = model.forward(&input)?;
    let mut expected = Vec::new();
    for b in 0..2 {
        let mut cls = tokens[b * TOKENS * M::HIDDEN..b * TOKENS * M::HIDDEN + M::HIDDEN].to_vec();
        let mut mean = vec![0f32; M::HIDDEN];
        let mut kept = 0;
        for p in 0..196 {
            if masks[b * 196 + p] != 0 {
                kept += 1;
                for h in 0..M::HIDDEN {
                    mean[h] += tokens[(b * TOKENS + 5 + p) * M::HIDDEN + h];
                }
            }
        }
        if kept > 0 {
            for v in &mut mean {
                *v /= kept as f32;
            }
        }
        for row in [&mut cls, &mut mean] {
            let sum = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            let norm = if sum > 0. { sum.sqrt() as f32 } else { 1. };
            for v in row.iter_mut() {
                *v /= norm;
            }
            expected.extend_from_slice(row);
        }
    }
    let before = model.context().runtime().statistics().downloaded_bytes;
    let actual = model.describe_rgb(&rgb, &masks)?;
    let after = model.context().runtime().statistics().downloaded_bytes;
    assert_eq!(after - before, 0, "descriptor outputs are host-visible");
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "compact descriptor maximum error: {max_error}; downloaded {} bytes",
        after - before
    );
    assert!(max_error < 2e-4, "descriptor maximum error {max_error}");
    assert_eq!(&actual[3 * M::HIDDEN..], &vec![0f32; M::HIDDEN]);
    let before = model.context().runtime().statistics();
    assert_eq!(model.describe_rgb(&rgb, &masks)?, actual);
    let after = model.context().runtime().statistics();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
    assert_eq!(after.device_copied_bytes, before.device_copied_bytes);
    assert_eq!(after.submissions - before.submissions, 1);
    assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
    Ok(())
}

fn descriptor_tiles_preserve_batch_tails_and_warm_replay<M: ModelSpec>() -> Result<()> {
    let model = DINOv3Model::<M>::load(
        model_path::<M>()?,
        Options {
            device: 0,
            max_batch: 64,
        },
    )?;
    let images: Vec<Vec<u8>> = (0..3)
        .map(|seed| {
            (0..IMAGE_ELEMENTS)
                .map(|i| ((i * 13 + i / 379 + seed * 47) % 256) as u8)
                .collect()
        })
        .collect();
    let masks: Vec<Vec<u8>> = (0..3)
        .map(|seed| (0..196).map(|i| u8::from((i + seed) % 7 != 0)).collect())
        .collect();
    let references = images
        .iter()
        .zip(&masks)
        .map(|(image, mask)| model.describe_rgb(image, mask))
        .collect::<Result<Vec<_>>>()?;
    // 201 token rows per image exercise partial and exact 64-row tiles.
    for batch in [1, 2, 3, 7, 16, 32, 64] {
        let order: Vec<_> = (0..batch).map(|i| (i * 2 + batch) % 3).collect();
        let rgb: Vec<_> = order
            .iter()
            .flat_map(|&i| images[i].iter().copied())
            .collect();
        let mask: Vec<_> = order
            .iter()
            .flat_map(|&i| masks[i].iter().copied())
            .collect();
        let expected = model.describe_rgb(&rgb, &mask)?;
        for (actual, &index) in expected.chunks_exact(2 * M::HIDDEN).zip(&order) {
            for (a, b) in actual
                .chunks_exact(M::HIDDEN)
                .zip(references[index].chunks_exact(M::HIDDEN))
            {
                let dot = a
                    .iter()
                    .zip(b)
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum::<f64>();
                let norm = |row: &[f32]| row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
                // Batch one uses split-K; preserve the existing full-model
                // single/batch cosine gate without requiring identical rounding.
                assert!(dot / (norm(a) * norm(b)).sqrt() > 0.99999, "batch {batch}");
            }
        }
        let before = model.context().runtime().statistics();
        assert_eq!(model.describe_rgb(&rgb, &mask)?, expected);
        let after = model.context().runtime().statistics();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
        assert_eq!(after.copied_bytes, before.copied_bytes);
        assert_eq!(after.submissions - before.submissions, 1);
    }
    Ok(())
}

fn model_path<M: ModelSpec>() -> Result<std::path::PathBuf> {
    match std::env::var_os(match M::NAME {
        "vits16plus" => "DINOV3_MODEL",
        "vitb16" => "DINOV3_VITB_MODEL",
        "vits16" => "DINOV3_VITS_MODEL",
        "vitl16" => "DINOV3_VITL_MODEL",
        "vith16plus" => "DINOV3_VITH_MODEL",
        "vit7b16" => "DINOV3_VIT7B_MODEL",
        _ => unreachable!(),
    }) {
        Some(path) => Ok(path.into()),
        None => dinov3_hrx::hub::weights_for::<M>(false),
    }
}

fn raw_summaries_keep_values_and_download_only_requested_rows<M: ModelSpec>() -> Result<()> {
    let model = DINOv3Model::<M>::load(
        model_path::<M>()?,
        Options {
            device: 0,
            max_batch: 2,
        },
    )?;
    // Three different images also exercise the short final chunk.
    let input = (0..3 * IMAGE_ELEMENTS)
        .map(|i| ((i * 17 + i / 379) % 251) as f32 / 127. - 1.)
        .collect::<Vec<_>>();
    let tokens = model.forward(&input)?;
    for mean in [false, true] {
        let expected = tokens
            .chunks(TOKENS * M::HIDDEN)
            .flat_map(|image| {
                if !mean {
                    return image[..M::HIDDEN].to_vec();
                }
                let mut result = vec![0.; M::HIDDEN];
                for patch in image[5 * M::HIDDEN..].chunks(M::HIDDEN) {
                    for (sum, &v) in result.iter_mut().zip(patch) {
                        *sum += v / 196.;
                    }
                }
                result
            })
            .collect::<Vec<_>>();
        for _ in 0..2 {
            let before = model.context().runtime().statistics().downloaded_bytes;
            let actual = if mean {
                model.patch_mean(&input)?
            } else {
                model.cls(&input)?
            };
            let after = model.context().runtime().statistics().downloaded_bytes;
            assert!(
                after - before <= (3 * M::HIDDEN * 4) as u64,
                "only requested rows may be transferred"
            );
            assert_eq!(
                bytemuck::cast_slice::<M::Row, f32>(&actual),
                expected,
                "mean={mean}"
            );
        }
    }
    assert!(model.cls(&[])?.is_empty());
    assert!(model.patch_mean(&[])?.is_empty());
    assert!(model.cls(&input[..3]).is_err());
    let mut invalid = input[..IMAGE_ELEMENTS].to_vec();
    invalid[17] = f32::NAN;
    assert!(model.patch_mean(&invalid).is_err());
    Ok(())
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn full_reference_and_changing_batch_vits() -> Result<()> {
    full_reference_and_changing_batch::<ViTS16Plus>()
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn full_reference_and_changing_batch_vitb() -> Result<()> {
    full_reference_and_changing_batch::<ViTB16>()
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn compact_rgb_descriptors_preserve_pooling_contract_and_transfer_only_descriptors_vits()
-> Result<()> {
    compact_rgb_descriptors_preserve_pooling_contract_and_transfer_only_descriptors::<ViTS16Plus>()
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn compact_rgb_descriptors_preserve_pooling_contract_and_transfer_only_descriptors_vitb()
-> Result<()> {
    compact_rgb_descriptors_preserve_pooling_contract_and_transfer_only_descriptors::<ViTB16>()
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn descriptor_tiles_preserve_batch_tails_and_warm_replay_vits() -> Result<()> {
    descriptor_tiles_preserve_batch_tails_and_warm_replay::<ViTS16Plus>()
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn descriptor_tiles_preserve_batch_tails_and_warm_replay_vitb() -> Result<()> {
    descriptor_tiles_preserve_batch_tails_and_warm_replay::<ViTB16>()
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn raw_summaries_keep_values_and_download_only_requested_rows_vits() -> Result<()> {
    raw_summaries_keep_values_and_download_only_requested_rows::<ViTS16Plus>()
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn raw_summaries_keep_values_and_download_only_requested_rows_vitb() -> Result<()> {
    raw_summaries_keep_values_and_download_only_requested_rows::<ViTB16>()
}

#[test]
fn model_types_preserve_array_widths() {
    type SmallSummary = fn(&DINOv3, &[f32]) -> Result<Vec<[f32; 384]>>;
    type BaseSummary = fn(&DINOv3ViTB, &[f32]) -> Result<Vec<[f32; 768]>>;
    let _: SmallSummary = DINOv3::cls;
    let _: SmallSummary = DINOv3::patch_mean;
    let _: BaseSummary = DINOv3ViTB::cls;
    let _: BaseSummary = DINOv3ViTB::patch_mean;
    type LargeSummary = fn(&DINOv3ViTL, &[f32]) -> Result<Vec<[f32; 1024]>>;
    type HugeSummary = fn(&DINOv3ViTH, &[f32]) -> Result<Vec<[f32; 1280]>>;
    type GiantSummary = fn(&DINOv3ViT7B, &[f32]) -> Result<Vec<[f32; 4096]>>;
    type PlainSmallSummary = fn(&DINOv3ViTS, &[f32]) -> Result<Vec<[f32; 384]>>;
    let _: PlainSmallSummary = DINOv3ViTS::cls;
    let _: PlainSmallSummary = DINOv3ViTS::patch_mean;
    let _: LargeSummary = DINOv3ViTL::cls;
    let _: HugeSummary = DINOv3ViTH::cls;
    let _: GiantSummary = DINOv3ViT7B::cls;
    let _: LargeSummary = DINOv3ViTL::patch_mean;
    let _: HugeSummary = DINOv3ViTH::patch_mean;
    let _: GiantSummary = DINOv3ViT7B::patch_mean;
    assert_eq!(DINOv3::HIDDEN, HIDDEN);
    assert_eq!(DINOv3ViTB::HIDDEN, 768);
}

#[test]
#[ignore = "requires both checkpoints and gfx1151"]
fn mixed_models_share_context_and_compose_graphs() -> Result<()> {
    use hrx::{
        inference::ModelContext,
        tensor::{DType, Layout, TensorDesc},
    };
    let context = ModelContext::new(Default::default())?;
    let small = DINOv3::load_in(model_path::<ViTS16Plus>()?, &context, 2)?;
    let base = DINOv3ViTB::load_in(model_path::<ViTB16>()?, &context, 2)?;
    let pixels: Vec<_> = (0..IMAGE_ELEMENTS).map(|i| (i % 71) as f32 / 71.).collect();
    let input = context.upload(
        TensorDesc::new(DType::F32, vec![1, 3, 224, 224])?.with_layout(Layout::Nchw)?,
        bytemuck::cast_slice(&pixels),
    )?;
    let mask_values = vec![1u8; 196];
    let masks = context.upload(TensorDesc::new(DType::U8, vec![1, 196])?, &mask_values)?;
    let mut graph = context.runtime().graph();
    let small_tokens = small.record(&mut graph, &input)?;
    let base_tokens = base.record(&mut graph, &input)?;
    let small_descriptors = small.record_descriptors(&mut graph, &input, &masks)?;
    let base_descriptors = base.record_descriptors(&mut graph, &input, &masks)?;
    let graph = graph.prepare()?;
    let expected_small = small.forward(&pixels)?;
    let expected_base = base.forward(&pixels)?;
    let expected_sd = small.descriptors(&pixels, &mask_values)?;
    let expected_bd = base.descriptors(&pixels, &mask_values)?;
    for _ in 0..2 {
        graph.submit()?.wait()?;
        for (output, expected) in [
            (&small_tokens, &expected_small),
            (&base_tokens, &expected_base),
            (&small_descriptors, &expected_sd),
            (&base_descriptors, &expected_bd),
        ] {
            assert_eq!(
                context.download(output)?.wait()?,
                bytemuck::cast_slice::<f32, u8>(expected)
            );
        }
    }
    // Device submission and independent pooling retain the model-specific widths.
    let small_inference = small.submit(&input)?;
    let base_inference = base.submit(&input)?;
    assert_eq!(
        small_inference.outputs()[0].desc().shape(),
        [1, TOKENS, 384]
    );
    assert_eq!(base_inference.outputs()[0].desc().shape(), [1, TOKENS, 768]);
    let pooled = base.pool_descriptors(&base_inference.outputs()[0], &masks)?;
    assert_eq!(
        pooled.download()?.wait()?[0],
        bytemuck::cast_slice::<f32, u8>(&expected_bd)
    );
    assert!(
        small
            .pool_descriptors(&base_inference.outputs()[0], &masks)
            .is_err()
    );
    assert!(
        base.pool_descriptors(&small_inference.outputs()[0], &masks)
            .is_err()
    );
    Ok(())
}

#[test]
fn rejects_overflowing_weights_before_gpu_initialization() -> Result<()> {
    overflowing_weights::<ViTS16Plus>()?;
    overflowing_weights::<ViTB16>()
}

#[test]
fn rejects_checkpoint_architecture_mismatch_before_gpu_initialization() -> Result<()> {
    use safetensors::{Dtype, tensor::TensorView};
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("model.safetensors");
    for width in [384, 768] {
        let data = vec![0u8; width * 768 * 4];
        let tensor = TensorView::new(Dtype::F32, vec![width, 3, 16, 16], &data)?;
        std::fs::write(
            &path,
            safetensors::serialize([("embeddings.patch_embeddings.weight", tensor)], None)?,
        )?;
        let error = if width == 384 {
            DINOv3ViTB::load(&path, Options::default()).err().unwrap()
        } else {
            DINOv3::load(&path, Options::default()).err().unwrap()
        };
        assert!(
            error
                .to_string()
                .contains("embeddings.patch_embeddings.weight: expected"),
            "{error}"
        );
    }
    Ok(())
}

fn family_reference_and_outputs<M: ModelSpec>() -> Result<()> {
    let path = model_path::<M>()?;
    let model = DINOv3Model::<M>::load(
        &path,
        Options {
            device: 0,
            max_batch: 2,
        },
    )?;
    let pixels: Vec<_> = (0..3 * IMAGE_ELEMENTS)
        .map(|i| ((i as f64 * 0.037).sin() * 0.6) as f32)
        .collect();
    let first = &pixels[..IMAGE_ELEMENTS];
    let got = model.forward(first)?;
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{} produced non-finite output",
        M::NAME
    );
    eprintln!("{} GPU output is finite; computing F64 reference", M::NAME);
    let want = reference::forward(&path, first, M::NAME)?;
    fn cosine(a: &[f32], b: &[f64]) -> f64 {
        let dot = a.iter().zip(b).map(|(&x, &y)| x as f64 * y).sum::<f64>();
        dot / (a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
            * b.iter().map(|x| x * x).sum::<f64>())
        .sqrt()
    }
    eprintln!(
        "{} GPU finite {}/{} max {} reference finite {}/{} max {}",
        M::NAME,
        got.iter().filter(|x| x.is_finite()).count(),
        got.len(),
        got.iter()
            .copied()
            .filter(|x| x.is_finite())
            .map(f32::abs)
            .fold(0f32, f32::max),
        want.iter().filter(|x| x.is_finite()).count(),
        want.len(),
        want.iter()
            .copied()
            .filter(|x| x.is_finite())
            .map(f64::abs)
            .fold(0f64, f64::max)
    );
    let all = cosine(&got, &want);
    let cls = cosine(&got[..M::HIDDEN], &want[..M::HIDDEN]);
    eprintln!("{} reference: all={all}, CLS={cls}", M::NAME);
    assert!(all > 0.9999 && cls > 0.9999);
    let batched = model.forward(&pixels)?;
    assert_eq!(batched.len(), 3 * TOKENS * M::HIDDEN);
    assert!(
        cosine(
            &batched[..TOKENS * M::HIDDEN],
            &got.iter().map(|&x| x as f64).collect::<Vec<_>>()
        ) > 0.99999
    );
    assert_eq!(model.forward(&pixels)?, batched);
    assert_eq!(model.cls(first)?[0].as_ref(), &got[..M::HIDDEN]);
    let means = model.patch_mean(first)?;
    let mut mean = vec![0f32; M::HIDDEN];
    for row in got[5 * M::HIDDEN..].chunks(M::HIDDEN) {
        for (sum, &value) in mean.iter_mut().zip(row) {
            *sum += value / 196.;
        }
    }
    assert_eq!(means[0].as_ref(), mean);
    let masks = vec![0; 3 * 196];
    let descriptors = model.descriptors(&pixels, &masks)?;
    assert_eq!(descriptors.len(), 3 * 2 * M::HIDDEN);
    for row in descriptors.chunks(2 * M::HIDDEN) {
        assert!(row[M::HIDDEN..].iter().all(|&x| x == 0.));
    }
    assert!(model.forward(&[])?.is_empty());
    assert!(model.descriptors(&[], &[])?.is_empty());
    assert!(model.describe_rgb(&[], &[])?.is_empty());
    Ok(())
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn family_reference_vits() -> Result<()> {
    family_reference_and_outputs::<ViTS16>()
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn family_reference_vitl() -> Result<()> {
    family_reference_and_outputs::<ViTL16>()
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn family_reference_vith() -> Result<()> {
    family_reference_and_outputs::<ViTH16Plus>()
}
#[test]
#[ignore = "requires 7B checkpoint, substantial RAM and gfx1151"]
fn family_reference_vit7b() -> Result<()> {
    family_reference_and_outputs::<ViT7B16>()
}

#[test]
fn rejects_gated_small_checkpoint_for_plain_small_model() -> Result<()> {
    use safetensors::{Dtype, tensor::TensorView};
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("model.safetensors");
    let data = [0u8; 4];
    let gate = TensorView::new(Dtype::F32, vec![1], &data)?;
    std::fs::write(
        &path,
        safetensors::serialize([("layer.0.mlp.gate_proj.weight", gate)], None)?,
    )?;
    let error = DINOv3ViTS::load(&path, Options::default()).err().unwrap();
    assert!(
        error
            .to_string()
            .contains("gated MLP, but vits16 expects GELU"),
        "{error}"
    );
    Ok(())
}
