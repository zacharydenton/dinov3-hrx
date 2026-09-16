//! Compare token/descriptor API latency and readback volume, excluding preparation.
use anyhow::{Result, ensure};
use dinov3_hrx::{DINOv3, IMAGE_ELEMENTS, Options};
use std::time::Instant;
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let mode = args.get(1).map(String::as_str).unwrap_or("rgb");
    let batch: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let samples: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(100);
    ensure!(
        batch > 0 && batch <= 64 && samples > 0,
        "invalid batch/samples"
    );
    let model = DINOv3::from_pretrained(Options {
        max_batch: batch,
        ..Default::default()
    })?;
    let rgb = (0..batch * IMAGE_ELEMENTS)
        .map(|i| ((i * 13 + i / 379) % 256) as u8)
        .collect::<Vec<_>>();
    let mut input = vec![0.; rgb.len()];
    for b in 0..batch {
        for c in 0..3 {
            for p in 0..224 * 224 {
                input[b * IMAGE_ELEMENTS + c * 224 * 224 + p] =
                    (rgb[b * IMAGE_ELEMENTS + p * 3 + c] as f32 / 255. - [0.485, 0.456, 0.406][c])
                        / [0.229, 0.224, 0.225][c];
            }
        }
    }
    let masks = vec![1; batch * 196];
    let run = || -> Result<Vec<f32>> {
        Ok(match mode {
            "tokens" => model.forward(&input)?,
            "cls" => model.cls(&input)?.into_iter().flatten().collect(),
            "mean" => model.patch_mean(&input)?.into_iter().flatten().collect(),
            "describe" => model.descriptors(&input, &masks)?,
            "rgb" => model.describe_rgb(&rgb, &masks)?,
            _ => anyhow::bail!("expected tokens, cls, mean, describe or rgb"),
        })
    };
    for _ in 0..5 {
        std::hint::black_box(run()?);
    }
    let before = model.context().runtime().statistics();
    let mut times = Vec::new();
    for _ in 0..samples {
        let start = Instant::now();
        std::hint::black_box(run()?);
        times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let after = model.context().runtime().statistics();
    let output = run()?;
    if let Some(path) = args.get(4) {
        std::fs::write(path, bytemuck::cast_slice(&output))?;
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"mode":mode,"batch":batch,"samples":samples,
        "median_ms":times[samples/2],"p95_ms":times[(samples*95).div_ceil(100)-1],
        "download_bytes_per_call":(after.downloaded_bytes-before.downloaded_bytes)/samples as u64,
        "warm_ms":times})
    );
    Ok(())
}
