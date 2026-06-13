#!/usr/bin/env python3
"""Generate a small synthetic EXL3 layer for local testing (no model download)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ponyexl3.ref.trellis import pack_trellis_tile

_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_FIXTURES / "synthetic_layer.npz")
    p.add_argument("-k", type=int, default=4, help="bits per weight (1-8)")
    p.add_argument("--in-features", type=int, default=128)
    p.add_argument("--out-features", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    k = args.k
    in_tiles = args.in_features // 16
    out_tiles = args.out_features // 16
    mask = (1 << k) - 1

    trellis = np.empty((in_tiles, out_tiles, 256 * k // 16), dtype=np.uint16)
    for i in range(in_tiles):
        for j in range(out_tiles):
            encoded = (rng.integers(0, 65536, 256) & mask).astype(np.uint16)
            trellis[i, j] = pack_trellis_tile(encoded, k)

    suh = np.where(rng.random((in_tiles, 16)) > 0.5, -1.0, 1.0).astype(np.float16).reshape(-1)
    svh = np.where(rng.random((out_tiles, 16)) > 0.5, -1.0, 1.0).astype(np.float16).reshape(-1)
    bias = rng.standard_normal(args.out_features).astype(np.float16) * 0.01

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        key="synthetic.linear",
        in_features=args.in_features,
        out_features=args.out_features,
        k=k,
        trellis=trellis,
        suh=suh,
        svh=svh,
        bias=bias,
        mcg=np.int8(0),
        mul1=np.int8(0),
    )
    meta = {
        "key": "synthetic.linear",
        "in_features": args.in_features,
        "out_features": args.out_features,
        "k": k,
        "path": str(args.out),
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
