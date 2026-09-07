"""Inject failures after a real GPU launch; verify drain, retry and invalidation."""
from pathlib import Path
import ctypes as C
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def child():
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from dinov3_loom import DINOv3Loom, DINOv3Error
    probe = C.CDLL(None)
    probe.loom_test_arm.argtypes = [C.c_int]
    probe.loom_test_arm.restype = None
    probe.loom_test_count.argtypes = [C.c_int]
    probe.loom_test_count.restype = C.c_int
    submitted = lambda: (probe.loom_test_count(0), probe.loom_test_count(1))
    for drain_error in (0, 1):
        with DINOv3Loom(max_batch=1) as model:
            image = np.zeros((1, 3, 224, 224), np.float32)
            inp = np.zeros((1, 196, 768), np.float32)
            output = np.full((1, 201, 384), 123.0, np.float32)
            run = lambda: model(image)
            baseline = run()
            error = C.create_string_buffer(4096)
            probe.loom_test_arm(drain_error)
            status = model._native.dinov3_run(model._handle,
                inp.ctypes.data_as(C.POINTER(C.c_float)), inp.size,
                output.ctypes.data_as(C.POINTER(C.c_float)), output.size, 1, error, len(error))
            check(status == 1 and b"hipModuleLaunchKernel" in error.value, "original launch error was lost")
            check(submitted() == (1, 2), f"queued launch was not drained: {submitted()}")
            check(np.all(output == 123.0), "launch failure changed the caller's output")
            if drain_error:
                try:
                    run()
                except DINOv3Error as failure:
                    check("unusable" in str(failure), str(failure))
                else:
                    raise AssertionError("run accepted after failed recovery")
                check(submitted() == (1, 2), "unusable session submitted more GPU work")
                print("  PASS failed recovery disables the session")
            else:
                check(np.array_equal(run(), baseline), "retry differs after recovery")
                print("  PASS launch failure drains queued work, preserves output and permits retry")


def main():
    if sys.argv[1:] == ["--child"]:
        child()
        return
    hipcc = os.environ.get("HIPCC", str(Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"))
    with tempfile.TemporaryDirectory(prefix="dinov3-fault-test-") as directory:
        library = Path(directory) / "hip_fault_injection.so"
        subprocess.run([hipcc, "-shared", "-fPIC", "-O2", "-Wall", "-Werror",
                        str(ROOT / "tools/hip_fault_injection.cpp"), "-o", str(library), "-ldl"], check=True)
        env = os.environ.copy()
        env["LD_PRELOAD"] = str(library) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env=env, check=True)


if __name__ == "__main__":
    main()
