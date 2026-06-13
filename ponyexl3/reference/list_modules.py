#!/usr/bin/env python3
"""List EXL3 linear keys and/or exllamav3 forward-module keys for a checkpoint.

Does not require CUDA for ``--exl3-only``. Pass ``--load`` to also print the
module order used by ``Model.forward`` (needs GPU + checkpoint load).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    from _cuda_common import (
        list_exl3_module_keys,
        list_forward_module_keys,
        load_exllama_model,
    )
else:
    from ponyexl3.reference._cuda_common import (
        list_exl3_module_keys,
        list_forward_module_keys,
        load_exllama_model,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=str)
    ap.add_argument("--exl3-only", action="store_true", help="skip model load")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    exl3 = list_exl3_module_keys(args.model_dir)
    fwd: list[str] = []

    if not args.exl3_only:
        model, config = load_exllama_model(
            args.model_dir, seq_len=args.seq_len, progressbar=False
        )
        fwd = list_forward_module_keys(model)
        arch = type(config).__name__
    else:
        with open(Path(args.model_dir) / "config.json", encoding="utf-8") as f:
            arch = json.load(f).get("architectures", ["?"])[0]

    if args.json:
        print(json.dumps({"arch": arch, "exl3": exl3, "forward": fwd}, indent=2))
        return 0

    print(f"arch: {arch}")
    print(f"exl3 linear modules: {len(exl3)}")
    for key in exl3:
        print(f"  {key}")
    if fwd:
        print(f"forward modules: {len(fwd)}")
        for key in fwd:
            print(f"  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
