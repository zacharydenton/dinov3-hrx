#!/usr/bin/env bash
# The one test command. Builds the kernels, runs every unit test against a
# float64 NumPy reference on the real GPU, checks the runner's error paths, then
# validates the whole model against transformers.
#
#   scripts/test.sh          everything
#   scripts/test.sh --quick  skip the transformers comparison (no torch needed)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

quick=0
[ "${1:-}" = "--quick" ] && quick=1

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
export tmpdir

status=0
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then
    printf '  ok\n'
  else
    printf '  FAILED: %s\n' "$name"
    status=1
  fi
}

step "fork lifecycle checks survive python -O" python3 -O tools/test_lifecycle.py
step "graph and generator checks survive python -O" python3 -O tools/test_invariants.py

# Anything the tests compile has to build first.
step "loom sources are canonically formatted" bash -c '
  "$LOOM_FORMAT" --check kernels/*.loom experiments/*.loom experiments/superseded/*.loom'
step "wheel imports independently" python3 tools/test_package.py
step "export weights" python3 tools/export_weights.py
step "build kernels" ./scripts/build_kernels.sh
# Generate to a scratch path and diff, so the check can never rewrite the
# tracked file it is supposed to be verifying.
step "generated kernels match their generator" bash -c '
  python3 tools/gen_f16_variants.py --output "$tmpdir/cf16.loom" >/dev/null
  cmp -s "$tmpdir/cf16.loom" kernels/attention_online_f16_wmma_cf16.loom'

# Always rebuilt: stale binaries or libraries would test the previous source.
step "build host programs and shared library" ./scripts/build_host.sh
step "native GPU error recovery" python3 tools/test_native_errors.py

step "embed scatter"      python3 tools/test_embed_scatter.py
step "layernorm"          python3 tools/test_layernorm.py
step "gate/up + swiglu"   python3 tools/test_swiglu_matmul.py
step "matmul (WMMA)"      python3 tools/test_matmul_wmma.py
step "fused-epilogue matmuls" python3 tools/test_fused_matmuls.py
step "online attention"   python3 tools/test_attention_online.py
step "resident Python API" python3 tools/test_python_api.py

# Error paths: each of these must exit non-zero with a diagnostic rather than
# reading uninitialised memory or overflowing a buffer.
step "runner rejects bad input" bash -c '
  # rejects <expected-phrase> <command...>: the command must exit 64 (usage
  # error) AND print the phrase. A crash, a silent exit, or the wrong message
  # each fail on their own.
  rejects() {
    local want="$1"; shift
    local out; out=$("$@" 2>&1); local rc=$?
    if [ "$rc" != 64 ]; then echo "  expected exit 64, got $rc: $*"; return 1; fi
    if ! grep -q -- "$want" <<<"$out"; then echo "  missing \"$want\" in: $out"; return 1; fi
  }
  printf short > "$tmpdir/short.bin"
  python3 -c "import numpy as np; np.zeros((196, 768), np.float32).tofile(\"$tmpdir/one.bin\")"
  cat "$tmpdir/one.bin" "$tmpdir/one.bin" > "$tmpdir/two.bin"
  long=$(python3 -c "print(\"A\"*900)")
  args=(--hsaco x --kernel k); for i in $(seq 40); do args+=(--i32 1); done
  rejects "needs a value"     ./host/dinov3 --batch                                          &&
  rejects "must be 1"         ./host/dinov3 --input "$tmpdir/one.bin" --batch 0           &&
  rejects "must be 1"         ./host/dinov3 --input "$tmpdir/one.bin" --batch 999         &&
  rejects "must be an integer" ./host/dinov3 --input "$tmpdir/one.bin" --batch nope       &&
  rejects "at least 1"        ./host/dinov3 --input "$tmpdir/one.bin" --repeat 0          &&
  rejects "input is required" ./host/dinov3                                                  &&
  rejects "not a multiple"    ./host/dinov3 --input "$tmpdir/short.bin"                      &&
  rejects "holds 2 images"    ./host/dinov3 --input "$tmpdir/two.bin" --batch 3              &&
  rejects "path too long"     ./host/loomrun --hsaco x --kernel k --out "$long:16"           &&
  rejects "too many args"     ./host/loomrun "${args[@]}"'

if [ "$quick" = 0 ]; then
  step "end-to-end vs transformers" python3 tools/validate.py
  step "batched python API vs transformers" python3 tools/test_batch.py
fi

printf '\n'
[ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
