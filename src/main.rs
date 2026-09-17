use anyhow::{Result, ensure};
use clap::Parser;
use dinov3_hrx::*;
use std::{path::PathBuf, time::Instant};
/// Run inference or measure warm end-to-end inference, including transfers.
#[derive(Parser)]
struct Args {
    /// Architecture to specialize; local weights must match this variant.
    #[arg(long, default_value = "vits16plus", value_parser = ["vits16plus", "vitb16", "vits16", "vitl16", "vith16plus", "vit7b16"])]
    variant: String,
    /// Local model file; otherwise fetch the pinned weights from Hugging Face.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Use only cached model weights.
    #[arg(long)]
    offline: bool,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 16)]
    max_batch: usize,
    /// Warm benchmark iterations; zero performs one inference.
    #[arg(long, default_value_t = 0)]
    benchmark: usize,
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
    ensure!(
        (1..=64).contains(&args.max_batch),
        "max_batch must be 1..=64"
    );
    let input = std::fs::read(&args.input)?;
    let setup = Instant::now();
    let model_path = match args.model {
        Some(path) => path,
        None => hub::weights_for::<M>(args.offline)?,
    };
    ensure!(
        input.len().is_multiple_of(4),
        "input must be little-endian float32 NCHW RGB"
    );
    let input: Vec<f32> = input
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let model = DINOv3Model::<M>::load(
        &model_path,
        Options {
            device: args.device,
            max_batch: args.max_batch,
        },
    )?;
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    if args.benchmark > 0 {
        let mut report = serde_json::to_value(model.benchmark(&input, args.benchmark)?)?;
        report["variant"] = M::NAME.into();
        eprintln!("{}", serde_json::to_string(&report)?);
    }
    let run = || model.forward(&input);
    let output = run()?;
    if let Some(path) = args.output {
        std::fs::write(path, bytemuck::cast_slice(&output))?;
    } else if args.benchmark == 0 {
        use std::io::Write;
        std::io::stdout()
            .lock()
            .write_all(bytemuck::cast_slice(&output))?;
    }
    if args.benchmark > 0 {
        ensure!(args.benchmark >= 10, "use at least 10 benchmark iterations");
        for _ in 0..10 {
            std::hint::black_box(run()?);
        }
        let mut times = Vec::with_capacity(args.benchmark);
        for _ in 0..args.benchmark {
            let start = Instant::now();
            std::hint::black_box(run()?);
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"variant":M::NAME,"scope":"warm end-to-end, including transfers","setup_ms":setup_ms,"samples":times.len(),"median_ms":times[times.len()/2],"p95_ms":times[(times.len()*95).div_ceil(100)-1]})
        );
    }
    Ok(())
}
