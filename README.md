# dinov3-hrx

DINOv3 image embeddings and patch features in Rust for AMD Strix Halo.
GPU inference uses [Loom](https://github.com/ROCm/hrx-system) kernels and
[HRX](https://github.com/zacharydenton/hrx-rs), with resident weights and reusable graphs.

| Model | CLI variant | Feature width | Recorded images/s |
| --- | --- | ---: | ---: |
| ViT-S | `vits16` | 384 | 1,736 |
| ViT-S+ (default) | `vits16plus` | 384 | 1,492 |
| ViT-B | `vitb16` | 768 | 521 |
| ViT-L | `vitl16` | 1024 | 144 |
| ViT-H+ | `vith16plus` | 1280 | 62 |
| ViT-7B | `vit7b16` | 4096 | 8.92 |

Measured September 18, 2026 on Radeon 8060S with HRX 0.7.1: warm RGB descriptor
inference, batch 16, synthetic 224×224 inputs. Loading, compilation and graph
preparation are excluded. [Methodology and numerical checks](docs/memory-layout-2026-09-18.md).

## Build and run

Requires Rust 1.91+, Linux x86-64, glibc 2.43+ and `gfx1151`.
The Meta checkpoints are gated: accept their terms and set `HF_TOKEN` or use a
cached Hugging Face login. From the checkout:

```sh
cargo run --release -- --variant vits16plus \
  --input normalized-nchw.f32 --output tokens.f32
```

Input is little-endian float32 NCHW, resized to 224×224 and ImageNet-normalized.
Output contains 201 tokens per image: CLS, four registers and 196 patch tokens.
Weights and the HRX runtime download on first use. Use `--model PATH` for a local
checkpoint or `--offline` for cached weights. ViT-7B downloads about 27 GB.

## Rust API

```rust
use dinov3_hrx::{DINOv3, Options};

fn main() -> anyhow::Result<()> {
    let model = DINOv3::from_pretrained(Options::default())?;
    let pixels = vec![0.0; 3 * 224 * 224];
    let tokens = model.forward(&pixels)?; // [batch, 201, 384]
    Ok(())
}
```

Typed aliases select other models, such as `DINOv3ViTB` and `DINOv3ViT7B`.
`cls` and `patch_mean` return unnormalized features; `descriptors` and
`describe_rgb` return normalized CLS and masked patch-mean descriptors.

The library also exposes a generic `Encoder<S>` for embedded tokens, device
submission and composition in caller-owned HRX graphs.
[Model types, input layouts and encoder API](docs/api.md).

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
cargo run --release --example bench_descriptors -- rgb 16 100 --variant vitb16
```

Ignored tests require the GPU and selected checkpoints. They compare against
an independent float64 reference and exercise graph reuse and model composition.
[Tests and model-path overrides](docs/api.md#development) ·
[Model files](docs/api.md#weights)

## License

[Apache-2.0](LICENSE). Weights are downloaded separately under the
[DINOv3 terms](THIRD_PARTY_NOTICES.md).
