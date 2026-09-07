# dinov3-loom

DINOv3 ViT-S+/16 inference in [Loom](https://github.com/ROCm/hrx-system) for the
AMD Radeon 8060S (gfx1151). Takes normalised 224×224 RGB images and returns
201 tokens of 384 features per image: one CLS token, four registers and 196 patches.

The implementation uses f16 weights and activations with f32 accumulation.
Weights, kernels and GPU buffers stay resident between calls. The Python API
accepts NumPy arrays or PyTorch tensors and returns a NumPy array.

## Performance and validation

Recorded on 2026-09-04 on a Radeon 8060S, with torch 2.13.0 + ROCm. Results are
the best of three interleaved rounds with other CPU work running.

| Runtime | Batch | Images/s |
| --- | ---: | ---: |
| Loom fp16 | 1 | 645.9 |
| torch max-autotune fp16 | 1 | 599.4 |
| Loom fp16 | 32 | 1431.0 |
| Loom fp16 | 64 | 1324.9 |
| torch max-autotune fp16 | 64 | 1280.7 |
| torch compile fp16 | 64 | 1081.1 |
| torch eager fp16 | 64 | 736.6 |

These are **warm GPU forward timings**, excluding model setup, patchification,
upload and download. They do not measure Python API throughput. At matched
batch sizes, Loom is 1.08× faster at batch 1 and 1.03× at batch 64. Its batch-32
result is 1.12× the best measured torch configuration, which uses batch 64.
See [raw results](docs/benchmark-2026-09-04-release.txt) and
[`tools/benchmark.py`](tools/benchmark.py).

Validation compares all output tokens with Hugging Face Transformers on three
deterministic synthetic inputs, with minimum cosine similarity **0.9999807**.
A separate test checks four distinct inputs in one batch. Kernel tests compare
with a float64 NumPy reference. These checks measure numerical agreement;
evaluate retrieval, clustering or other downstream tasks on your own data.

## Build

Requires Linux x86-64, a gfx1151 GPU, ROCm, Python 3.11+, the Loom compiler and
access to Meta's `dinov3-vits16plus-pretrain-lvd1689m` model.

Follow the [build guide](docs/building.md) for the pinned public compiler revision
and model access. Then run from this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
source scripts/env.sh
python tools/export_weights.py
./scripts/build_kernels.sh
./scripts/build_host.sh
./scripts/test.sh
```

`test.sh` exports weights, rebuilds the native assets and runs the full suite.
`--quick` skips the Transformers comparisons; it still needs the GPU and model
weights. The kernel and API tests do not need torch.

The Python wheel contains the loader only. Outside a checkout, set
`DINOV3_LOOM_WEIGHTS`, `DINOV3_LOOM_KERNELS` and `DINOV3_LOOM_LIBRARY` to the
exported weights, HSACOs and `libdinov3.so`. See the
[build guide](docs/building.md#paths) for all overrides.

## Python API

```python
from dinov3_loom import DINOv3Loom

# pixel_values: (B, 3, 224, 224), resized and normalised RGB
with DINOv3Loom(max_batch=32) as model:
    tokens = model(pixel_values)                # (B, 201, 384), float32
    cls = tokens[:, 0]                          # (B, 384)
    patches = tokens[:, 5:].reshape(-1, 14, 14, 384)
```

Use the model's image processor configured for 224×224, or supply your own
preprocessing. RGB values in [0, 1] use ImageNet mean `[0.485, 0.456, 0.406]` and
standard deviation `[0.229, 0.224, 0.225]`. This library performs patchification;
it does not resize or normalise the image.

`model.cls(pixel_values)` returns CLS features; `model.patch_mean(pixel_values)`
returns the mean of the 196 patch tokens. Both run inference. Inputs larger than
`max_batch` are split automatically; the default is 32 and the maximum is 64.

Calls on one model are serialized and thread-safe. Use `close()` or a context
manager to release GPU resources. Create models after forking. The output is a
NumPy array corresponding to `last_hidden_state`, not a Transformers output object.
Only ViT-S+/16 at 224×224 and gfx1151 is validated.

## Implementation

Eleven kernel configurations implement patch embedding, LayerNorm, QKV with
RoPE, attention, SwiGLU and residual projections. Batch 1 uses split-K for the
down projection. The host executes the fixed 12-layer network through a C ABI.

- [`kernels/`](kernels/): production Loom kernels.
- [`host/`](host/): resident C ABI, inference CLI and kernel test runner.
- [`tools/`](tools/): weight export, reference, validation and benchmarks.
- [`experiments/`](experiments/): alternative kernels and earlier implementations.
- [Engineering notes](docs/notes.md): implementation history and measurements.

## License

Project code is [Apache-2.0](LICENSE). Built with DINOv3.

Model files and exported weights are excluded from the repository and Python
packages. They remain subject to Meta's separate
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
See [third-party notices](THIRD_PARTY_NOTICES.md) and the
[model page](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
for access and terms.
