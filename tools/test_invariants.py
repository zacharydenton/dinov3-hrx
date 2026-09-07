"""Generator failures remain active when Python assertions are disabled."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import gen_f16_variants as G


def main():
    original = (G.KERNELS / G.SOURCE).read_text()
    with tempfile.TemporaryDirectory(prefix="dinov3-generator-test-") as directory:
        kernels = Path(directory)
        (kernels / G.SOURCE).write_text(original.replace(G.EDITS[0][0], "// anchor removed"))
        with patch.object(G, "KERNELS", kernels), patch("sys.argv", ["gen_f16_variants.py"]):
            try:
                G.main()
            except ValueError as failure:
                if "anchor moved" not in str(failure):
                    raise
            else:
                raise AssertionError("generator accepted a missing anchor")
        if (kernels / G.TARGET).exists():
            raise AssertionError("generator wrote an invalid kernel")
    print("  PASS missing generator anchors prevent output under python -O")


if __name__ == "__main__":
    main()
