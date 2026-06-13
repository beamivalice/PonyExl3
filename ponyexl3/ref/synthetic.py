"""Synthetic EXL3 layers for tests and local benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .layer import EXL3Layer
from .trellis import pack_trellis_tile


def random_packed_tile(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = (1 << k) - 1
    encoded = (rng.integers(0, 2**16, size=256, dtype=np.uint32) & mask).astype(np.uint16)
    return pack_trellis_tile(encoded, k)


def make_trellis(
    k: int,
    in_features: int,
    out_features: int,
    seed: int,
) -> np.ndarray:
    in_tiles = in_features // 16
    out_tiles = out_features // 16
    return np.stack(
        [
            [random_packed_tile(k, seed * 1000 + i * 10 + j) for j in range(out_tiles)]
            for i in range(in_tiles)
        ],
        axis=0,
    )


def make_exl3_layer(
    *,
    k: int = 4,
    in_features: int = 128,
    out_features: int = 128,
    seed: int = 0,
    suh: np.ndarray | None | str = "random",
    svh: np.ndarray | None | str = "random",
    mcg: bool = False,
    mul1: bool = False,
    bias: bool = False,
) -> EXL3Layer:
    rng = np.random.default_rng(seed)
    trellis = make_trellis(k, in_features, out_features, seed)

    def _signs(n: int) -> np.ndarray:
        return np.where(rng.random(n) > 0.5, -1.0, 1.0).astype(np.float16)

    if suh == "random":
        suh_arr = _signs(in_features)
    elif suh is None:
        suh_arr = None
    else:
        suh_arr = suh if isinstance(suh, np.ndarray) else None
    if svh == "random":
        svh_arr = _signs(out_features)
    elif svh is None:
        svh_arr = None
    else:
        svh_arr = svh if isinstance(svh, np.ndarray) else None
    bias_arr = None
    if bias:
        bias_arr = rng.standard_normal(out_features).astype(np.float16) * 0.01

    return EXL3Layer(
        key="test.layer",
        in_features=in_features,
        out_features=out_features,
        k=k,
        trellis=trellis,
        suh=suh_arr,
        svh=svh_arr,
        bias=bias_arr,
        mcg=mcg,
        mul1=mul1,
    )


def load_synthetic_npz(path: str | Path) -> EXL3Layer:
    d = np.load(path, allow_pickle=False)
    return EXL3Layer(
        key=str(d["key"]),
        in_features=int(d["in_features"]),
        out_features=int(d["out_features"]),
        k=int(d["k"]),
        trellis=d["trellis"],
        suh=d["suh"],
        svh=d["svh"],
        bias=d["bias"],
        mcg=bool(int(d["mcg"])),
        mul1=bool(int(d["mul1"])),
    )
