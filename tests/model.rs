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

#[test]
fn rejects_overflowing_weights_before_gpu_initialization() -> Result<()> {
    use half::bf16;
    use safetensors::{Dtype, tensor::TensorView};

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("model.safetensors");
    for dtype in [Dtype::F32, Dtype::BF16] {
        for value in [100_000f32, -100_000f32] {
            let mut data = vec![0; 384 * 3 * 16 * 16 * (dtype.bitsize() / 8)];
            match dtype {
                Dtype::F32 => data[..4].copy_from_slice(&value.to_le_bytes()),
                Dtype::BF16 => data[..2].copy_from_slice(&bf16::from_f32(value).to_le_bytes()),
                _ => unreachable!(),
            }
            let tensor = TensorView::new(dtype, vec![384, 3, 16, 16], &data)?;
            let bytes =
                safetensors::serialize([("embeddings.patch_embeddings.weight", tensor)], None)?;
            std::fs::write(&path, bytes)?;
            // The first tensor must fail packing, before reading any other
            // tensors or trying to initialize a GPU.
            let error = match DINOv3::load(&path, Options::default()) {
                Ok(_) => panic!("accepted overflowing {dtype:?} weight {value}"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains("patch_w[0]"), "{error}");
            assert!(error.contains("finite float16"), "{error}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn full_reference_and_changing_batch() -> Result<()> {
    let path = model_path()?;
    let model = DINOv3::load(
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
        let want = reference::forward(std::path::Path::new(&path), img)?;
        let compare = |a: &[f32], b: &[f64]| {
            let dot = a.iter().zip(b).map(|(a, b)| *a as f64 * b).sum::<f64>();
            let aa = a.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
            let bb = b.iter().map(|b| b * b).sum::<f64>();
            dot / (aa * bb).sqrt()
        };
        assert!(compare(&got, &want) > 0.9999, "full image {i}");
        assert!(compare(&got[..384], &want[..384]) > 0.9999, "CLS image {i}");
        assert!(
            compare(
                &batched[i * TOKENS * HIDDEN..(i + 1) * TOKENS * HIDDEN],
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

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn compact_rgb_descriptors_preserve_pooling_contract_and_transfer_only_descriptors() -> Result<()> {
    let model = DINOv3::load(
        model_path()?,
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
        let mut cls = tokens[b * TOKENS * HIDDEN..b * TOKENS * HIDDEN + HIDDEN].to_vec();
        let mut mean = vec![0f32; HIDDEN];
        let mut kept = 0;
        for p in 0..196 {
            if masks[b * 196 + p] != 0 {
                kept += 1;
                for h in 0..HIDDEN {
                    mean[h] += tokens[(b * TOKENS + 5 + p) * HIDDEN + h];
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
    assert_eq!(after - before, (2 * 2 * HIDDEN * 4) as u64);
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
    assert_eq!(&actual[3 * HIDDEN..], &vec![0f32; HIDDEN]);
    Ok(())
}

fn model_path() -> Result<std::path::PathBuf> {
    match std::env::var_os("DINOV3_MODEL") {
        Some(path) => Ok(path.into()),
        None => dinov3_hrx::hub::weights(false),
    }
}
