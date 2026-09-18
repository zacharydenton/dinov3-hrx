# Large DINOv3 throughput and encoder extraction — 2026-09-18

Radeon 8060S (`gfx1151`), published HRX 0.7.1. Baseline: released `v0.1.0`
at `3db8b09`; both builds use the same published runtime/compiler. The target is
a **2× improvement in warm RGB descriptor throughput at batch 16**.
The optimized build includes the extracted `Encoder<S>` used by DINOv3.

## Batch-16 results

The complete family on the release implementation:

| Model | Images/s | Batch median (ms) |
| --- | ---: | ---: |
| ViT-S | 1749.31 | 9.12 |
| ViT-S+ | 1479.04 | 10.80 |
| ViT-B | 508.55 | 31.44 |
| ViT-L | 107.74 | 148.72 |
| ViT-H+ | 61.48 | 260.13 |
| ViT-7B | 5.00 | 3199.04 |

S, S+, and B were measured during 0.2.0 release preparation with the same
batch size, fixture, warmup count, and sample count. Their runs were sequential;
package verification ran concurrently on the CPU. The large-model results below
were retained from the preceding measurements, rather than rerun in this pass.
[Small-model raw samples](small-models-batch16-2026-09-18.json).

Large-model comparison with 0.1.0:

| Model | Baseline batch median (ms) | Optimized batch median (ms) | Baseline images/s | Optimized images/s | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vitl16` | 151.97 | 148.72 | 105.53 | 107.74 | 1.02× |
| `vith16plus` | 322.59 | 260.13 | 49.64 | 61.48 | 1.24× |
| `vit7b16` | 6900.56 | 3199.04 | 2.30 | 5.00 | 2.17× |

Throughput is total timed images divided by total timed seconds, including slow
samples; it is not the reciprocal of median latency. The 2× target is met for
**ViT-7B only**. H+ improves modestly; L retains its released projection path.
Every descriptor output is finite and byte-identical to its baseline, including
the expected byte count. [Raw samples and hashes](large-models-batch16-2026-09-18.json).

## Measurement protocol

The release baseline was measured once per model. Optimized measurements were
refreshed after the workspace-capacity fix; all GPU runs were sequential. Each process uses the
same `bench_descriptors` fixture: synthetic 224×224 RGB, full patch mask, five
warmups and twenty synchronized host-clock samples. Checkpoint packing,
compilation and graph preparation are excluded. No other GPU benchmark or test
was launched concurrently by this experiment. Other host/desktop workloads were
not controlled. These are local warm-throughput measurements, not tail-latency
or cross-hardware guarantees.

```bash
cargo build --release --locked --example bench_descriptors
# Save baseline and optimized executables from their respective revisions.
HF_HUB_OFFLINE=1 HRX_OFFLINE=1 /path/to/bench_descriptors rgb 16 20 output.f32 --variant vit7b16
```

## Implementation

H+ uses a 128×128 output tile and 7B a 256×64 tile, replacing 64×64 projections.
Eight output fragments per wave reuse matrix inputs across more multiply-adds.
The 64-element reduction tile, individual WMMA accumulation order, F16 matrix
inputs/weights and F32 residuals are preserved. Row tails are staged as zero and
never published. One kernel specializes tile shape, F32 projection, erf-GELU,
residual, and split-K partial epilogues.

QKV is rotated in F32 before narrowing. H+/7B gate/up projections finish SwiGLU
in a separate pass, narrowing after the F32 activation and multiplication.
QKV and gated MLP projections reuse one F32 scratch region. At batch 16 this
region is 131,727,360 bytes for H+ and 210,763,776 bytes for 7B. Each prepared
inference slot owns its workspace. L and the small models retain their released
projection kernels.

The unsealed `EncoderSpec` holds architecture constants. `Encoder<S>` owns the
shared transformer stack, while DINOv3 retains image embedding, checkpoint
metadata and descriptor pooling. Caller graphs may use explicit weights,
1–1024-token sequences, and optional caller-supplied rotary tables. Public
recording preserves its input; DINOv3 reuses its fresh embedding buffer internally
so existing copy-free warm replay is retained. See the README for contracts and
kernel dimension limits and configurable workspace-byte budgets.

## Validation

All 39 unit/integration tests, including ignored GPU/checkpoint tests, and the
compiled API doctest passed. The >0.9999 all-token and CLS cosine thresholds
against independent F64 references are unchanged. DINOv3 checks include all six
checkpoints, batch tails, graph composition, descriptor masks, and warm replay
without additional allocation, graph preparation, or copies.

Downstream-only encoder implementations exercise GELU/SwiGLU, F16/F32 residuals,
64/128-wide heads, sequences of 1/17/33/257 tokens, custom rotary tables,
the four-layer taste shape (1024×32×384), malformed weights, input preservation, independent graphs, and encoder lifetime.
Kernel tests cover both projection tile shapes, all epilogues, split-K, partial
and maximum-row tails, output canaries, F32 outliers, RoPE, and SwiGLU.

```bash
HF_HUB_OFFLINE=1 HRX_OFFLINE=1 cargo test --release --locked -- --include-ignored --test-threads=1
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check
cargo package --allow-dirty --locked
```

## Earlier screening

A previous candidate used 256×64 projections for all three large models. Paired
batch-one runs measured 21.04→17.09 ms for L (1.23×), 31.03→28.61 ms for H+
(1.08×), and 706.11→243.45 ms for 7B (2.90×). Those numbers describe that earlier
candidate, not the final implementation above. L had noisy tail latency.
[Historical batch-one samples and hashes](large-models-2026-09-18.json).

At batch 16 that candidate regressed L and H+, so it was not retained for them.
Other local screens varied tile sizes, wave counts, LDS padding, register
prefetching, and fused SwiGLU. The final selection follows end-to-end batch-16
measurements rather than isolated projection timings.
