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
    let mut model = DINOv3::load(
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

fn model_path() -> Result<std::path::PathBuf> {
    match std::env::var_os("DINOV3_MODEL") {
        Some(path) => Ok(path.into()),
        None => dinov3_hrx::hub::weights(false),
    }
}
