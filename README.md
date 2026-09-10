# dinov3-hrx

DINOv3 ViT-S+/16 inference in Rust for AMD GPUs with
[HRX](https://github.com/zacharydenton/hrx-rs) and Loom. Includes a library and CLI
with resident weights, buffers and reusable execution graphs.

Requires Rust 1.91+, Linux x86-64, glibc 2.43+ and a Radeon 8060S (`gfx1151`).
HRX provisions its runtime and compiler bundle on first use.

```bash
cargo build --release
cargo run --release -- --input normalized-nchw.f32 --output tokens.f32
```

## Library

```rust
use dinov3_hrx::{DINOv3, Options};

fn main() -> anyhow::Result<()> {
    let mut model = DINOv3::from_pretrained(Options::default())?;
    let pixels = vec![0.0; 3 * 224 * 224];
    let tokens = model.forward(&pixels)?; // flattened [batch, 201, 384]
    Ok(())
}
```

Inputs are float32 NCHW RGB images resized to 224×224. Normalize RGB values in
[0, 1] with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. Outputs contain CLS at token 0, four registers at
tokens 1–4 and 196 patch tokens at tokens 5–200. `cls` and `patch_mean` each run
inference and return one `[f32; 384]` per image.

`Options` selects the device and resident batch capacity (default 32, range 1–64).
Larger batches are chunked; empty batches return empty results. Incomplete images
and non-finite inputs return errors. Each model owns a stream and requires mutable
access; use separate models for concurrent callers. GPU failures invalidate the session.

## Weights

The default model is
[`facebook/dinov3-vits16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m/tree/c93d816fc9e567563bc068f01475bec89cc634a6),
file `model.safetensors`, pinned to revision `c93d816fc9e567563bc068f01475bec89cc634a6`.
Weights are validated and packed in memory. Downloads require access to the gated
model repository and `HF_TOKEN` or a cached Hugging Face login.

- **Cache:** the shared Hugging Face cache, configured by `HF_HOME` or `HF_HUB_CACHE`.
- **Local file:** `DINOv3::load(path, options)` or CLI `--model model.safetensors`.
- **Offline:** `HF_HUB_OFFLINE=1`, CLI `--offline` or `hub::weights(true)` uses cached
  weights only. Set `HRX_OFFLINE=1` to also disable runtime bundle downloads.

## CLI and benchmarks

Input and output files use contiguous little-endian float32 values in the library
layouts above. The CLI defaults to `--max-batch 16`.

Add `--benchmark 100` to measure resident graph/direct execution and warm
end-to-end inference. Benchmark input must fit one resident batch. Timings use
synchronized host clocks; see [measurements](docs/optimization-2026-09-10.md).

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

CPU tests need no GPU or weights. Ignored tests require `gfx1151` and fetch the
pinned weights (`DINOV3_MODEL` overrides the path). They check graph replay,
changing batches and numerical agreement with a float64 Rust reference; all-token
and CLS cosine similarity must exceed 0.9999.

## License

Code: [Apache-2.0](LICENSE). Model weights are downloaded separately under the
[DINOv3 terms](THIRD_PARTY_NOTICES.md).
