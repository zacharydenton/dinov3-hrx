# dinov3-hrx

DINOv3 ViT-S/16, ViT-S+/16, ViT-B/16, ViT-L/16, ViT-H+/16 and ViT-7B/16
inference in Rust for AMD GPUs with
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
    let model = DINOv3::from_pretrained(Options::default())?;
    let pixels = vec![0.0; 3 * 224 * 224];
    let tokens = model.forward(&pixels)?; // flattened [batch, 201, 384]
    Ok(())
}
```

Inputs are float32 NCHW RGB images resized to 224×224. Normalize RGB values in
[0, 1] with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. Outputs contain CLS at token 0, four registers at
tokens 1–4 and 196 patch tokens at tokens 5–200. `cls` and `patch_mean` each run
inference and return one `[f32; 384]` per image for the default model.

Select ViT-B at compile time with `DINOv3ViTB`:

```rust
use dinov3_hrx::{DINOv3ViTB, Options};

fn main() -> anyhow::Result<()> {
    let model = DINOv3ViTB::from_pretrained(Options::default())?;
    let pixels = vec![0.0; 3 * 224 * 224];
    let tokens = model.forward(&pixels)?; // flattened [batch, 201, 768]
    let cls: Vec<[f32; 768]> = model.cls(&pixels)?;
    Ok(())
}
```

Choose an architecture through its Rust type or the CLI's `--variant`:

| Model | Rust alias | CLI variant | Features | Layers | Heads | MLP |
| --- | --- | --- | ---: | ---: | ---: | --- |
| ViT-S | `DINOv3ViTS` | `vits16` | 384 | 12 | 6 | GELU |
| ViT-S+ (default) | `DINOv3` | `vits16plus` | 384 | 12 | 6 | SwiGLU |
| ViT-B | `DINOv3ViTB` | `vitb16` | 768 | 12 | 12 | GELU |
| ViT-L | `DINOv3ViTL` | `vitl16` | 1024 | 24 | 16 | GELU |
| ViT-H+ | `DINOv3ViTH` | `vith16plus` | 1280 | 32 | 20 | SwiGLU |
| ViT-7B | `DINOv3ViT7B` | `vit7b16` | 4096 | 40 | 32 | SwiGLU |

The aliases specialize `DINOv3Model<M>` with the sealed `ModelSpec` marker types
`ViTS16`, `ViTS16Plus`, `ViTB16`, `ViTL16`, `ViTH16Plus`, and `ViT7B16`.
Rust specializes host code by model type; Loom compiles GPU kernels at
initialization with fixed architecture dimensions. Each alias exposes its own
`HIDDEN` constant; the module-level `HIDDEN` remains 384 for compatibility.
`cls` and `patch_mean` return arrays with the model's feature width.

ViT-L, ViT-H+, and ViT-7B keep the residual stream in F32 because register-token
outliers can exceed F16's finite range. Matrix inputs and weights remain F16 with
F32 accumulation. ViT-7B uses 128-channel heads; the other models use 64.

All models support host inference, resident device submission, caller-owned
graphs, and descriptor pooling. `descriptors` and `describe_rgb` return flattened
`[batch,2,HIDDEN]` normalized CLS and masked patch-mean rows; an empty mask gives a
zero mean row. `cls` and `patch_mean` return unnormalized fixed-size arrays.
Each typed model owns its graph caches, so different models can share a `ModelContext` through
`load_in`.

`Options` selects the device and resident batch capacity (default 32, range 1–64).
Larger batches are chunked; empty batches return empty results. Incomplete images
and non-finite inputs return errors. Inference methods use shared access with
bounded reusable execution slots.
GPU failures invalidate the session.

## Weights

Pretrained constructors use Meta's LVD-1689M checkpoints, pinned independently:

| Variant | Hugging Face repository | Revision |
| --- | --- | --- |
| `vits16plus` | [facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m/tree/c93d816fc9e567563bc068f01475bec89cc634a6) | `c93d816fc9e567563bc068f01475bec89cc634a6` |
| `vits16` | [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/114c1379950215c8b35dfcd4e90a5c251dde0d32) | `114c1379950215c8b35dfcd4e90a5c251dde0d32` |
| `vitb16` | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/tree/5931719e67bbdb9737e363e781fb0c67687896bc) | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| `vitl16` | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51) | `ea8dc2863c51be0a264bab82070e3e8836b02d51` |
| `vith16plus` | [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/tree/c807c9eeea853df70aec4069e6f56b28ddc82acc) | `c807c9eeea853df70aec4069e6f56b28ddc82acc` |
| `vit7b16` | [facebook/dinov3-vit7b16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m/tree/b80367753773648a6793235ab9c65cdbb029506f) | `b80367753773648a6793235ab9c65cdbb029506f` |

The first five models use `model.safetensors`. ViT-7B uses
`model.safetensors.index.json` and six shards. `hub::weights_for::<ViT7B16>`
resolves every shard and returns the index path; offline mode requires all shards
in the cache. The loader buffers one source shard at a time.

ViT-7B downloads about 27 GB and uses about 13.5 GB for F16 GPU weights, plus
temporary packing storage and inference buffers. Start with `max_batch: 1` when
using it alongside other models.

Weights are validated against the selected architecture and packed in memory.
Downloads require access to the gated model repository and `HF_TOKEN` or a cached Hugging Face login.

- **Cache:** the shared Hugging Face cache, configured by `HF_HOME` or `HF_HUB_CACHE`.
- **Local checkpoint:** each typed model's `load(path, options)` accepts a
  SafeTensors file, shard index, or directory containing either. For example,
  CLI `--variant vit7b16 --model /models/vit7b/model.safetensors.index.json`.
- **Offline:** `HF_HUB_OFFLINE=1`, CLI `--offline` or `hub::weights(true)` uses cached
  weights only. For ViT-B use `hub::weights_for::<ViTB16>(true)`.
  Set `HRX_OFFLINE=1` to also disable runtime bundle downloads.

## CLI and benchmarks

Input and output files use contiguous little-endian float32 values in the library
layouts above. The CLI defaults to `--variant vits16plus --max-batch 16`.

```bash
cargo run --release -- --variant vitb16 --input normalized-nchw.f32 --output tokens.f32
cargo run --release --example bench_descriptors -- rgb 4 100 --variant vitb16
```

`--variant` selects a compiled model type once at startup; local weights must
match it. Benchmark JSON includes the selected variant.

Add `--benchmark 100` to measure warm end-to-end inference. Benchmark input must
fit one resident batch. Timings use synchronized host clocks; see the
[full-family validation and measurements](docs/vit-families-2026-09-17.md),
[ViT-B comparison](docs/vitb-2026-09-17.md), and
[earlier measurements](docs/optimization-2026-09-10.md).

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

CPU tests need no GPU or weights. Ignored tests require `gfx1151` and fetch the
pinned weights (`DINOV3_MODEL`, `DINOV3_VITS_MODEL`, `DINOV3_VITB_MODEL`,
`DINOV3_VITL_MODEL`, `DINOV3_VITH_MODEL`, and `DINOV3_VIT7B_MODEL` override paths).
They check graph replay, changing batches and numerical agreement with a float64
Rust reference; all-token and CLS cosine similarity must exceed 0.9999 for all six architectures. Focused GPU
tests cover wide LayerNorm with F32 outliers, erf-based GELU, 128-channel attention
and RoPE, partial tiles, mixed-model graph composition, and descriptor masks.
ViT-S+/ViT-B tests exercise warm replay through batch 64. ViT-7B tests need the
large checkpoint and enough host/device memory for the model and F64 reference.

## License

Code: [Apache-2.0](LICENSE). Model weights are downloaded separately under the
[DINOv3 terms](THIRD_PARTY_NOTICES.md).
