# Building dinov3-loom

The source checkout builds the native library and kernels. The Python wheel
contains only the portable loader; model weights and GPU assets remain external.

## Requirements

- Linux x86-64, an AMD Radeon 8060S (gfx1151), the amdgpu driver, and access to
  `/dev/kfd` and the render node under `/dev/dri`.
- ROCm with HIP headers, `hipcc` and ROCm clang. Release checks use HIP
  7.2.53211 and clang 22 from `/opt/rocm`, with the system HIP runtime.
  See [AMD's installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
- Python 3.11+, Git, CMake 3.26+, Ninja and a C/C++ toolchain.
- Access to `facebook/dinov3-vits16plus-pretrain-lvd1689m` on Hugging Face.

## Loom compiler

Run these commands from the dinov3-loom checkout. The pinned public revision is
[`c9855b47e96e7eb1cbb5b81b1de973762982ae95`](https://github.com/ROCm/hrx-system/commit/c9855b47e96e7eb1cbb5b81b1de973762982ae95).
No local HRX patches or HRX HIP compatibility library are needed. CMake fetches
the source dependencies pinned by that revision, so configuration needs network
access.

```bash
mkdir -p build
git clone https://github.com/ROCm/hrx-system.git build/hrx-source
git -C build/hrx-source checkout --detach c9855b47e96e7eb1cbb5b81b1de973762982ae95

export ROCM_PATH=/opt/rocm
cmake -S build/hrx-source -B build/hrx-system -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DIREE_ROCM_PATH="$ROCM_PATH" \
  -DIREE_ROCM_DEPENDENCY_MODE=pinned \
  -DIREE_BUILD_TESTS=OFF \
  -DLIBHRX_BUILD=OFF \
  -DLOOM_BUILD=ON \
  -DIREE_HAL_DRIVER_AMDGPU=ON
cmake --build build/hrx-system --target loom-compile loom-format --parallel 8
source scripts/env.sh
```

The compiler and formatter are the only Loom executables used by this project's
test suite. GPU tests run through the HIP host programs built here.

## Python environment and model

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The model has gated access. Review and accept Meta's terms on the
[model page](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
using your own Hugging Face account, then authenticate locally:

```bash
hf auth login
hf download facebook/dinov3-vits16plus-pretrain-lvd1689m \
  --revision c93d816fc9e567563bc068f01475bec89cc634a6 \
  --local-dir build/model
export DINOV3_SNAPSHOT="$PWD/build/model"
```

`tools/reference.py` pins this model revision. `DINOV3_SNAPSHOT` can instead
point to an existing local snapshot containing `model.safetensors` and the model
configuration files. With no override, the tools resolve the pinned revision
through `huggingface_hub` and its normal cache. Model weights and exported blobs
retain Meta's [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
They are not covered by the project's Apache-2.0 license.

For kernel and API tests without Transformers, the Python dependencies are
NumPy, safetensors, huggingface_hub and pip. The full validation suite also needs
torch and transformers. CPU torch is enough for reference comparisons; ROCm
PyTorch is required for the GPU benchmark. NumPy 2.3 requires Python 3.11 or newer.

## Build and test

```bash
source scripts/env.sh
python tools/export_weights.py
./scripts/build_kernels.sh
./scripts/build_host.sh
./scripts/test.sh
```

The suite exports weights, rebuilds the native assets and checks kernel
formatting, generated files, the loader wheel, fork guards, GPU error recovery,
all production kernels and the resident Python API. The final tests compare
single and batched outputs against Transformers. `--quick` skips those final
comparisons but still needs the GPU and model weights. Test inputs are generated
within each invocation; no pre-existing benchmark files are required.

The full benchmark is `python tools/benchmark.py`. It measures warm GPU forwards,
excluding input/output copies and Python preprocessing, and requires torch with
ROCm plus a working `torch.compile` toolchain. Results in `docs/` are historical
measurements, not a guarantee for every compiler or runtime version.

## Paths

Set build overrides before sourcing `scripts/env.sh` in a fresh shell.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HRX_BUILD` | `<checkout>/build/hrx-system` | Existing Loom CMake build |
| `LOOM_TOOLS` | `$HRX_BUILD/loom/src/loom/tools` | Loom tools directory |
| `LOOM_COMPILE`, `LOOM_FORMAT` | Under `LOOM_TOOLS` | Individual compiler/formatter overrides |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation |
| `HIPCC` | `$ROCM_PATH/bin/hipcc` | Host compiler |
| `LOOM_TARGET` | `gfx1151` | Kernel target; other chips are unvalidated |
| `DINOV3_SNAPSHOT` | Hugging Face cache, pinned revision | Model source directory |
| `DINOV3_LOOM_RUNTIME_PATH` | Unset | Optional library directory prepended to `LD_LIBRARY_PATH` |
| `DINOV3_LOOM_WEIGHTS` | `<checkout>/build/weights` | Exported weights for the Python API |
| `DINOV3_LOOM_KERNELS` | `<checkout>/build/kernels` | HSACOs for the Python API |
| `DINOV3_LOOM_LIBRARY` | `<checkout>/build/libdinov3.so` | Native library for the Python API |

The default leaves `LD_LIBRARY_PATH` unchanged and uses the system HIP runtime.
There is no dependency on an extracted toolbox runtime. For ROCm installed
outside the loader search path, set `DINOV3_LOOM_RUNTIME_PATH` to its library
directory. Use a consistent ROCm runtime when importing torch and the native
library in the same process.

The loader's `weights=`, `kernels=` and `library=` arguments override their
environment variables. When using the wheel outside a checkout, supply absolute
paths for all three assets. The CLI uses `--weights` and `--kernels`, with defaults
relative to its working directory. Its input is patchified f32 data, shaped
`(B, 196, 768)`; `dinov3_loom.patchify` produces that layout.

## Python distribution

```bash
python -m pip install build
python -m build
```

The wheel and source distribution contain the Python loader, README and license
notices. Neither includes the native source tree, HSACOs or weights; use a
repository checkout to build those. The build backend requires setuptools
77.0.3 or newer.
