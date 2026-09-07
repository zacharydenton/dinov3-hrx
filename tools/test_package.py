"""Build the wheel in isolation and prove its runtime modules and notices ship."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dinov3-wheel-") as directory:
        tmp = Path(directory)
        source, wheels, site, run = (tmp / name for name in ("source", "wheels", "site", "run"))
        for path in (source, wheels, site, run):
            path.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                     "dinov3_loom.py"):
            shutil.copy2(ROOT / name, source / name)

        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                        str(source), "-w", str(wheels)], check=True, capture_output=True, text=True)
        wheel, = wheels.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        assert "dinov3_loom.py" in names
        assert any(name.endswith("licenses/THIRD_PARTY_NOTICES.md") for name in names), names

        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(site),
                        str(wheel)], check=True, capture_output=True, text=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(site)
        imported = subprocess.run(
            [sys.executable, "-c", "import pathlib, dinov3_loom; print(pathlib.Path(dinov3_loom.__file__).parent)"],
            cwd=run, env=environment, check=True, capture_output=True, text=True,
        )
        assert Path(imported.stdout.strip()).resolve() == site.resolve(), imported.stdout
    print("  PASS isolated wheel imports and includes third-party notices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
