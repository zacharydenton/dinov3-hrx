# HRX migration measurements — 2026-09-10

Radeon 8060S (`gfx1151`), HRX 0.4.0, batch 1. Each measurement uses
10 warmups and 100 samples. Forward measurements alternate graph replay and
direct dispatch and synchronize each sample. End-to-end measurements include
preprocessing, upload, inference and download; image decoding and model setup
are excluded. These are local measurements, not cross-machine performance claims.

| Path | Median ms |
| --- | ---: |
| HRX graph forward | 1.893 |
| HRX direct forward | 1.930 |
| Rust end-to-end | 1.995 |
| Former Python API end-to-end, old compiler | 1.670 |

Full distributions and setup timings: [JSON](benchmark-2026-09-10.json).
Earlier benchmark files describe the former implementation and toolchain.

The approximately 0.3 ms regression from the old baseline follows the compiler:
the former HIP host also measures approximately 2.0 ms with kernels emitted by
HRX's pinned compiler. The Rust migration matches that control. The newer
compiler emits different GPU instructions; this migration does not substitute
unverified old compiler binaries to reproduce the earlier number.
