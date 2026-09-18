# Changelog

## 0.2.0 — 2026-09-18

- Split the unsealed architecture-only `EncoderSpec` from DINOv3 checkpoint
  metadata in `ModelSpec`. Direct architecture-constant access on marker types
  now requires importing `EncoderSpec`.
- Expose `Encoder<S>` with typed weights, sequence/RoPE options, and recording
  into caller-owned HRX graphs. DINOv3 uses the same stack and exposes it through
  `model.encoder()`; existing image inference and output types are retained.
- Use 128×128 projection tiles for ViT-H+ and 256×64 for ViT-7B, sharing F32
  projection scratch between QKV/RoPE and gated MLPs. Keep ViT-L's released
  projection path after batch-16 measurements found no useful gain.
- Extend numerical checks to downstream encoder specifications, graph input
  preservation/lifetimes, projection epilogues, split-K tails, RoPE, and SwiGLU.
- Bound encoder capacity by checked workspace bytes instead of DINOv3 token
  counts; accept up to 1024 sequences, including taste’s 1024×32×384 shape.
- Report images/second and milliseconds/image in descriptor benchmarks.

## 0.1.0 — 2026-09-18

- Require published HRX 0.7.1 for reusable private graph scratch.
- Add compile-time ViT-S/16, ViT-L/16, ViT-H+/16, and ViT-7B/16 models, with
  pinned LVD-1689M checkpoints and matching CLI/benchmark variants.
- Load sharded SafeTensors checkpoints by index or directory while buffering one
  source shard at a time. Handle ViT-7B's bias-free Q/V projections and 128-wide
  heads with F32 RoPE and specialized attention.
- Preserve large-model register-token outliers using F32 residual streams and
  wide LayerNorm, while retaining F16 matrix inputs and weights.

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

### Rust migration

- Renamed `dinov3-loom` to `dinov3-hrx`.
- Replaced the Python package and C ABI with a Rust library and CLI using HRX 0.4.0.
- Load original model files directly; weight conversion and graph scheduling run in Rust.
- Cache resident GPU graphs by batch size and reuse activation and readback allocations.
- Removed standalone HIP hosts, build scripts, Python tooling and obsolete experiments.

This replaces the former APIs; there are no deprecated compatibility wrappers.
