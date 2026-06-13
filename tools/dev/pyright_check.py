#!/usr/bin/env python3
"""Run pyright in strict mode on the installable package and tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    return subprocess.call(
        [sys.executable, "-m", "pyright", "ponyexl3", "tests"],
        cwd=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
