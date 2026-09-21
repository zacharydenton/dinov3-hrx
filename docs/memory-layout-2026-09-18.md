# Memory-layout and projection optimization — 2026-09-18

Radeon 8060S (`gfx1151`), published HRX 0.7.1. Baseline: released `v0.2.0`
(`87db58b`). Target: warm RGB descriptor throughput at batch 16, with unchanged
numerical accuracy and the existing downstream encoder API.

## Results

| Model | 0.2.0 images/s | Optimized images/s | Speedup |
| --- | ---: | ---: | ---: |
| ViT-S | 1740.34 | 1736.17 | 1.00× |
| ViT-S+ | 1506.51 | 1492.14 | 0.99× |
| ViT-B | 516.50 | 521.27 | 1.01× |
| ViT-L | 103.53 | 144.02 | 1.39× |
| ViT-H+ | 62.15 | 61.97 | 1.00× |
| ViT-7B | 4.86 | 8.92 | 1.83× |

L improves **1.39×** and 7B **1.83×** relative to 0.2.0. S, S+, B, and H+
remain within about 1% of baseline. Every measured descriptor output is finite
and byte-identical to its corresponding baseline, including batch-one checks.

The shorter batch-one sanity check measured L at 44.16→76.13 images/s (1.72×),
H+ at 37.56→37.75 (1.01×), and 7B at 4.17→5.21 (1.25×). These five-sample
checks are secondary to the paired batch-16 measurements.

Final paired measurements are recorded in
[memory-layout-batch16-2026-09-18.json](memory-layout-batch16-2026-09-18.json).

## Why these changes

An isolated synthetic-block profile showed that attention was much more expensive
than its arithmetic count suggested. Changing only the QKV row stride by 64 F16
elements reduced its median synchronized latency from 1.21 to 0.46 ms for L and
7.97 to 2.78 ms for 7B. These are diagnostic host timings, including submission
overhead, not GPU timestamps or additive estimates of graph time. The sensitivity
to stride is consistent with memory-access conflicts; no hardware counters were
used to distinguish cache-set or memory-bank effects.

The same padding helps power-of-two projection weight rows. The public weight
format remains contiguous row-major F16: the encoder pads qualifying rows once
at upload. Kernels receive the physical stride separately from the logical K.
Only F32-residual models pad weight rows, and only when K is a power of two at
least 1024. QKV rows gain 64 F16 elements when HIDDEN is a power of two at least
1024. Biases and position tables keep their original layout.

The new wide projections use 128×128 output tiles, with the next 64-column reduction
slice prefetched into registers while the current slice is multiplied. Groups
of four token-row tiles improve cache reuse; the final incomplete group uses
ordinary ordering, without extra workgroups. Split-K keeps ordinary ordering.
The reduction tile and WMMA accumulation order are unchanged. L retains its
existing projection geometry, with padded physical weight and QKV strides.
H+ keeps the released projection schedule after prefetch regressed its paired
batch-16 throughput by about 4%. The attention arithmetic itself is unchanged.

These changes add 27 MiB of resident weight storage for L and 180 MiB for 7B.
At batch 16 each adds 0.3945 MiB of QKV scratch, including the existing 16-row
headroom. Workspace validation includes the extra QKV bytes. The taste
1024×32×384 shape has no added padding and retains its existing byte budget.
Checkpoint files and the encoder's public input/output layouts are unchanged.

## Measurement protocol

Each batch-16 model runs in baseline/optimized/optimized/baseline order, with
five warmups and ten timed calls per process: twenty timed calls per revision.
The fixture is synthetic 224×224 RGB with a full patch mask. Every call is
synchronized; loading, compilation, and graph preparation are excluded.
Throughput is total timed images divided by total timed seconds, including all
samples. All GPU runs are sequential, after the regression suite; unrelated
host/desktop activity is not controlled. A separate five-sample batch-one check
per revision screens the three large models for regressions.

H+ was rerun after restoring its released projection schedule. Other models
retain the exact kernel sources and specialization values used in their paired
measurements; the prefetch source was moved into its own file without changing
its exported ABI. The JSON retains the rejected H+ prefetch measurements.

The raw JSON includes every sample, executable hashes, and descriptor output
hashes and lengths. Each run checks finite outputs and exact byte equality with
its baseline. These are local measurements, not cross-hardware guarantees.

```bash
cargo build --release --locked --example bench_descriptors
# Save binaries built from v0.2.0 and this implementation as baseline/optimized.
variant=vit7b16
for build in baseline optimized optimized baseline; do
  HF_HUB_OFFLINE=1 HRX_OFFLINE=1 /path/to/$build rgb 16 10 --variant "$variant"
done
```

## Validation

All 39 unit/integration tests, including the ignored GPU/checkpoint tests, plus
the API doctest pass. All six DINO checkpoints retain the existing independent
F64 reference thresholds. Downstream graph composition, input preservation,
warm replay, and the four-layer taste model at batch 1024 pass.

Projection checks cover one-row inputs, full and partial tiles, grouped-row
tails, padded weights containing NaN canaries, each epilogue, split-K, output
canaries, and 32,768 rows. RoPE checks verify values and untouched row padding;
attention checks span heads and image boundaries. A downstream F32 gated
encoder additionally exercises one-token sequences.

Formatting, Clippy with warnings denied, rustdoc with warnings denied, and
package verification pass.
