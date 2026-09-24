//! Compare token/descriptor latency, throughput, and readback, excluding preparation.
use anyhow::{Result, ensure};
use clap::Parser;
use dinov3_hrx::{
    DINOv3Model, IMAGE_ELEMENTS, ModelSpec, Options, ViT7B16, ViTB16, ViTH16Plus, ViTL16, ViTS16,
    ViTS16Plus,
};
use std::time::Instant;
#[derive(Parser)]
struct Args {
    #[arg(default_value = "rgb")]
    mode: String,
    #[arg(default_value_t = 1)]
    batch: usize,
    #[arg(default_value_t = 100)]
    samples: usize,
    output: Option<std::path::PathBuf>,
    #[arg(long, default_value = "vits16plus", value_parser = ["vits16plus", "vitb16", "vits16", "vitl16", "vith16plus", "vit7b16"])]
    variant: String,
}
fn main() -> Result<()> {
    let args = Args::parse();
    match args.variant.as_str() {
        "vits16" => run::<ViTS16>(args),
        "vitl16" => run::<ViTL16>(args),
        "vith16plus" => run::<ViTH16Plus>(args),
        "vit7b16" => run::<ViT7B16>(args),
        "vitb16" => run::<ViTB16>(args),
        _ => run::<ViTS16Plus>(args),
    }
}
fn run<M: ModelSpec>(args: Args) -> Result<()> {
    let mode = args.mode.as_str();
    let batch = args.batch;
    let samples = args.samples;
    ensure!(
        batch > 0 && batch <= 64 && samples > 0,
        "invalid batch/samples"
    );
    let model = DINOv3Model::<M>::from_pretrained(Options {
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
            "cls" => bytemuck::cast_slice(&model.cls(&input)?).to_vec(),
            "mean" => bytemuck::cast_slice(&model.patch_mean(&input)?).to_vec(),
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
    ensure!(output.iter().all(|v| v.is_finite()), "nonfinite output");
    if let Some(path) = args.output {
        std::fs::write(path, bytemuck::cast_slice(&output))?;
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"variant":M::NAME,"mode":mode,"batch":batch,"samples":samples,
        "median_ms":times[samples/2],"images_per_second":(batch*samples) as f64*1000./times.iter().sum::<f64>(),
        "ms_per_image":times[samples/2]/batch as f64,"p95_ms":times[(samples*95).div_ceil(100)-1],
        "download_bytes_per_call":(after.downloaded_bytes-before.downloaded_bytes)/samples as u64,
        "live_bytes":after.live_bytes,"peak_bytes":after.peak_bytes,
        "warm_allocations":after.allocations-before.allocations,
        "warm_ms":times})
    );
    Ok(())
}
