# DINOv3 ViT family validation — 2026-09-17

All six LVD-1689M ViT architectures are implemented as compile-time model types.
Checkpoint identities and revisions are listed in the [README](../README.md).
Validation used `gfx1151` and HRX 0.7.0.

## Accuracy

The independent F64 reference reads original F32 checkpoint tensors and buffers
one layer at a time. Its architecture table is separate from production ModelSpec.
The existing acceptance gate remains >0.9999 all-token and CLS cosine.

| New model | All-token cosine | CLS cosine |
| --- | ---: | ---: |
| `vits16` | 0.9999987722 | 0.9999985630 |
| `vitl16` | 0.9999981485 | 0.9999996765 |
| `vith16plus` | 0.9999980697 | 0.9999997985 |
| `vit7b16` | 0.9999942409 | 0.9999999216 |

ViT-S+ and ViT-B also pass their existing multi-image reference tests. New model
tests cover changing batch sizes, a final partial chunk, graph replay, typed CLS
and patch-mean rows, descriptor output sizes, and empty masks/inputs.

Large models need F32 residuals: the ViT-L reference reaches approximately 155,693
after its first block, and ViT-H+ reaches approximately 87,501 after block 23 on
the deterministic test input. F16 residuals overflow on these values. The new
F32 residual kernels preserve them while matrix inputs and weights remain F16.

Focused GPU tests validate wide LayerNorm with values beyond the F16 range,
128-channel attention across every head and image/tile boundaries, and F32 RoPE
before narrowing (including unrotated prefix/V channels). Masked pooling is
checked for every model width. CPU tests cover shard switching, missing tensors,
invalid shard paths, dtype packing, and S versus S+ architecture mismatches.

## Checks

All 32 tests passed across the CPU suite, release GPU suite, and separate ViT-7B
reference run. The 7B test took about 149 seconds on this machine, including model
loading, the F64 reference, and output/chunking checks.

```bash
cargo test
cargo test --release -- --include-ignored --skip family_reference_vit7b --test-threads=1
cargo test --release --test model family_reference_vit7b -- --ignored --nocapture
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Warm latency

Batch one, synthetic 224×224 RGB input, full patch mask, five warmup calls and ten
timed calls per process. Model loading and compilation are excluded. These are
smoke benchmark measurements rather than a controlled performance comparison.

| Variant | Median (ms) | p95 (ms) |
| --- | ---: | ---: |
| `vits16plus` | 1.701 | 1.713 |
| `vitb16` | 3.965 | 3.996 |
| `vits16` | 1.596 | 1.660 |
| `vitl16` | 20.811 | 20.867 |
| `vith16plus` | 31.894 | 32.173 |
| `vit7b16` | 702.464 | 715.613 |

Every output was checked for its expected size and finite values. Download counters
remained zero because descriptor results use mapped host-visible storage.

```bash
cargo run --release --example bench_descriptors -- rgb 1 10 --variant vit7b16
```

[Raw measurements](vit-families-2026-09-17.json) contain all samples.
