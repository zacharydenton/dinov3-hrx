"""Fork guards must reject inherited sessions before touching a copied lock."""
import ctypes
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dinov3_loom import DINOv3Loom, DINOv3Error


class MustNotLock:
    def __enter__(self):
        raise AssertionError("attempted to acquire a fork-inherited lock")

    def __exit__(self, *args):
        return False


def main():
    model = DINOv3Loom.__new__(DINOv3Loom)
    model._handle = ctypes.c_void_p(1)
    model._pid = -1
    model._lock = MustNotLock()
    model.max_batch = 1
    for call in (lambda: model(np.zeros((1, 3, 224, 224), np.float32)),):
        try:
            call()
        except DINOv3Error as failure:
            if "fork" not in str(failure):
                raise AssertionError(str(failure))
        else:
            raise AssertionError("inherited session accepted a call")
    model.close()
    if not model.closed:
        raise AssertionError("close did not invalidate the inherited session")
    print("  PASS fork guards reject calls and close without acquiring inherited locks")


if __name__ == "__main__":
    main()
