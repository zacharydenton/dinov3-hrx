# Inference optimization — 2026-09-10

Radeon 8060S (`gfx1151`), published HRX 0.4.0 and its pinned compiler.
Three paired rounds reverse executable order each round; each run uses
10 warmups and 300 samples, with `max_batch=16`. The table reports the median
of three run medians for complete inference, including patchification and I/O.
Model loading and compilation are excluded. Batch four repeats the batch-one
normalized NCHW input. [Raw results](optimization-2026-09-10.json) retain every
run and its p95. These are local host timings, not hardware timestamps.

| Batch | Shipped median ms | Unrolled median ms | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 2.027 | 1.978 | 2.4% |
| 4 | 4.052 | 3.983 | 1.7% |

## Changes

Five WMMA kernels explicitly unroll their two-iteration staging and publication
loops. The long K loops stay rolled. This uses Loom's existing `scf.for ... unroll`
primitive and keeps the same matrix tiles, arithmetic and synchronization.
The improvement is modest but appeared in all six paired comparisons.

The residual allocation now matches its float16 storage, saving 154,368 bytes
per reserved image (4.71 MiB at the library's default capacity of 32). The timing
comparison above isolates loop unrolling; it does not attribute a speedup to
the allocation correction. Kernel comments now describe their actual dtypes.
Full float64 reference and changing-batch tests pass with unchanged tolerances.

## Experiments not retained

Coherent shared inputs and outputs helped the face models but slowed DINOv3:
three paired batch-one runs had median latencies of 2.008 ms for the shipped
path and 2.038 ms for shared I/O. DINOv3 retains device-local inputs and outputs
with queued transfers.

The inspected `hrx-system` checkout was at `ecaaf7376f7d` with local compiler
edits; those edits were not built or used. Its vector dialect exposes matrix
fragment load/store and repack operations. A direct residual epilogue prototype
hit `fragment_memory.view_stride` and `fragment_memory.payload_form` diagnostics
in the published compiler for broadcast views and converting result loads.
A scalar register-publication alternative compiled but was slower in a screening
run (about 2.055 ms end-to-end), so neither prototype remains in the crate.

Native event elapsed time currently subtracts host record timestamps
(`libhrx/src/libhrx/event.c`); it cannot supply GPU-only kernel timings.
The [earlier compiler regression](benchmark-2026-09-10.md) remains unresolved.
Retesting fragment epilogues against a verified newer compiler is the next
useful kernel experiment; this change keeps the published bundle pin.
