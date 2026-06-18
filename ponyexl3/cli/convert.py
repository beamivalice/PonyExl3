"""HF -> EXL3 converter bring-up CLI.

The current implementation is the fast oracle-comparable tile pilot.  It
quantizes one 16x16 tile from a source checkpoint and compares it to the same
tile in an existing EXL3 oracle checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ponyexl3.convert.fixtures import run_tile_pilot, tile_pilot_summary


DEFAULT_PILOT_MODULE = "model.language_model.layers.0.linear_attn.in_proj_qkv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, type=Path, help="source HF checkpoint")
    parser.add_argument("--out-dir", type=Path, help="reserved for full conversion output")
    parser.add_argument("--work-dir", type=Path, help="reserved for resumable conversion state")
    parser.add_argument("--oracle-dir", required=True, type=Path, help="EXL3 oracle checkpoint")
    parser.add_argument("--bits", type=float, default=4.0, help="target decoder bpw")
    parser.add_argument("--head-bits", type=int, default=6, help="target head bpw")
    parser.add_argument(
        "--codebook",
        choices=("mcg", "mul1", "3inst"),
        default="mcg",
        help="target codebook; tile pilot uses the oracle layer's stored mode",
    )
    parser.add_argument("--only-layer", type=int, help="reserved for layer-scoped conversion")
    parser.add_argument(
        "--only-module",
        default=DEFAULT_PILOT_MODULE,
        help="module key to pilot",
    )
    parser.add_argument("--tile-k", type=int, default=0, help="input tile index")
    parser.add_argument("--tile-n", type=int, default=0, help="output tile index")
    parser.add_argument("--resume", action="store_true", help="reserved for full conversion")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    try:
        result = run_tile_pilot(
            args.in_dir,
            args.oracle_dir,
            args.only_module,
            tile_k=args.tile_k,
            tile_n=args.tile_n,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    summary = tile_pilot_summary(result)
    summary["requested"] = {
        "bits": args.bits,
        "head_bits": args.head_bits,
        "codebook": args.codebook,
        "out_dir": None if args.out_dir is None else str(args.out_dir),
        "work_dir": None if args.work_dir is None else str(args.work_dir),
        "only_layer": args.only_layer,
        "resume": bool(args.resume),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    stats = summary["stats"]
    print(f"module: {summary['module']}")
    print(f"tile: {summary['tile']}  K={summary['k']}  codebook={summary['codebook']}")
    print(
        "target MSE: "
        f"converted={stats['converted_target_mse']:.6e}  "
        f"oracle={stats['oracle_target_mse']:.6e}"
    )
    print(
        "rel RMS: "
        f"converted={stats['converted_target_rel_rms']:.6f}  "
        f"oracle={stats['oracle_target_rel_rms']:.6f}  "
        f"converted-vs-oracle={stats['converted_vs_oracle_rel_rms']:.6f}"
    )
    print(
        "pack roundtrip: "
        f"converted={stats['converted_pack_roundtrip']}  "
        f"oracle={stats['oracle_pack_roundtrip']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
