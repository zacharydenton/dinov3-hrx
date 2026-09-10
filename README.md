# dinov3-hrx

DINOv3 ViT-S+/16 inference in Rust, using [hrx-rs](https://github.com/zacharydenton/hrx-rs)
for GPU execution and Loom compilation. One Cargo package provides the library
and CLI. Weights, kernels, activation buffers and I/O storage stay resident;
HRX graphs are recorded once per encountered batch size and replayed.

Requires Rust 1.91+, Linux x86-64 and a Radeon 8060S (`gfx1151`). HRX 0.4.0
provisions its verified runtime and compiler bundle; its native Linux bundle
requires glibc 2.43 or newer. Model weights are fetched through Hugging Face Hub when needed.

```bash
cargo build --release
```

## Library

`from_pretrained` loads the original `model.safetensors` from
`facebook/dinov3-vits16plus-pretrain-lvd1689m`. The validated revision is
`c93d816fc9e567563bc068f01475bec89cc634a6`. Loading validates tensor names,
shapes and dtypes, then packs weights in memory.

```rust,no_run
use dinov3_hrx::{DINOv3, Options};

# fn main() -> anyhow::Result<()> {
let mut model = DINOv3::from_pretrained(Options::default())?;
let pixels = vec![0.0; 3 * 224 * 224];
let tokens = model.forward(&pixels)?; // flattened [batch, 201, 384]
# Ok(())
# }
```

Inputs are contiguous float32 NCHW RGB, already resized to 224×224 and
normalized using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]` on values in [0, 1]. The library performs patchification.
Token 0 is CLS, tokens 1–4 are registers, and tokens 5–200 are the 14×14 patches.
`cls` and `patch_mean` each run inference and return one `[f32; 384]` per image.

`Options` selects a device index and maximum resident batch (default 32, range
1–64). Larger inputs are chunked; an empty batch returns an empty result.
Incomplete images and non-finite inputs return errors.

## Model weights

`from_pretrained` and the CLI without `--model` use
[`facebook/dinov3-vits16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m/tree/c93d816fc9e567563bc068f01475bec89cc634a6),
file `model.safetensors`, pinned to revision `c93d816fc9e567563bc068f01475bec89cc634a6`. The `hf-hub` 1.0 client reuses the shared
Hugging Face cache before downloading.
`HF_HOME` and `HF_HUB_CACHE` control its location; `HF_TOKEN` or a cached
Hugging Face login supplies authentication.

Use `DINOv3::load(path, options)` or `--model model.safetensors` for a local file.
An explicit local path never falls back to a download. `hub::weights(true)`
returns only cached weights. The CLI's `--offline` and `HF_HUB_OFFLINE=1`
also disable model downloads; `HRX_OFFLINE=1` separately disables runtime
bundle downloads.

The first download requires access to the gated DINOv3 repository. Request
access on its model page and authenticate before fetching uncached weights.

## CLI

```bash
cargo run --release -- --input normalized-nchw.f32 --output tokens.f32
```

Input and output files contain contiguous little-endian float32 values in the
library layouts. Add `--benchmark 100` for timings; benchmark input must fit
one resident batch.

## Execution and validation

Inference requires `&mut` access to the model. A model owns its stream; use
separate models for independent concurrent callers. GPU failures return errors
and make the session unusable. Drop releases owned resources through HRX.
The fixed production kernels are validated only for `gfx1151`.

Warm calls reuse compiled kernels, device allocations and graphs. Uploads are
queued; readback copies share the inference stream and complete before host
access. Graph dependencies preserve launch order and activation-buffer reuse.
`benchmark` reports alternating graph/direct forward timings, excluding transfers;
the CLI also reports warm end-to-end timing. Both are synchronized host timings,
not hardware timestamp measurements. See [current measurements](docs/optimization-2026-09-10.md).

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

CPU tests run without a GPU or model files. Ignored tests fetch the pinned
weights when needed (`DINOV3_MODEL` overrides the path) and require hardware; they cover numerical agreement, changing inputs, partial batches and
graph replay. The full-model reference evaluates the original weights in float64 Rust.
All-token and CLS cosine similarity must exceed 0.9999.

## License

Project code is Apache-2.0. Model weights have separate terms and are not
bundled with this crate. Downloads remain subject to those terms. See [third-party notices](THIRD_PARTY_NOTICES.md)
for model terms and retained source attribution.
