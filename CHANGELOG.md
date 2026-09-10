# Changelog

## Unreleased

- Fetch pinned pretrained weights through the shared Hugging Face cache by default; retain local-file and offline loading.
- Require Rust 1.91 for the HF Hub 1.0 dependency stack.

- Unroll fixed staging and publication loops in five WMMA kernels.
- Size the float16 residual allocation correctly, saving 154,368 bytes per reserved image.

## 0.1.0 — Rust migration

- Renamed `dinov3-loom` to `dinov3-hrx`.
- Replaced the Python package and C ABI with a Rust library and CLI using HRX 0.4.0.
- Load original model files directly; weight conversion and graph scheduling run in Rust.
- Cache resident GPU graphs by batch size and reuse activation and readback allocations.
- Removed standalone HIP hosts, build scripts, Python tooling and obsolete experiments.

This replaces the former APIs; there are no deprecated compatibility wrappers.
