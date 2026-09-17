# Changelog

## Unreleased

- Add compile-time DINOv3 ViT-B/16 support through `DINOv3ViTB`, retaining the
  existing `DINOv3` default and fixed-size summary arrays. Specialize weight
  packing, kernels, graph composition, and descriptor pooling for each model.
- Add 768-channel LayerNorm and fused erf-GELU projection kernels, pinned ViT-B
  downloads, and CLI/benchmark `--variant vitb16` selection.
- Validate local weights and batch options before GPU context initialization.

- Widen matrix staging packets and double reduction tiles while preserving
  accumulation order; cover batch tails and allocation-free replay through 64.
- Use HRX 0.7 graph composition and directly mapped input/output storage.
- Pool CLS and patch-mean summaries on the GPU with compact readback, and fuse
  RGB normalization with patch packing without changing output contracts.
- Reuse SwiGLU staging memory for result tiles, reducing shared memory per
  workgroup from 31 KiB to 16 KiB without changing arithmetic.

- Use HRX 0.5's shared Hugging Face resolver, mapped SafeTensors loader,
  specialization builders, and top-level `hrx::model` API.
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
