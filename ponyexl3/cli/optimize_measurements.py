"""Optimize a budgeted EXL3 bit plan from ponyexl3 measurement JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from ponyexl3.convert.measure import optimize_measurement_plan


def _load_measurement(path: Path) -> dict[str, Any]:
    text = sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("measurement JSON root must be an object")
    return data


def _print_plan(plan: dict[str, Any]) -> None:
    shrinkage = plan.get("hessian_shrinkage")
    shrinkage_s = "per-module" if shrinkage is None else f"{float(shrinkage):.3f}"
    print(
        "optimized measurement plan: "
        f"modules={plan['module_count']} "
        f"target_bpw={float(plan['target_bpw']):.6f} "
        f"average_bits={float(plan['average_bits']):.6f} "
        f"objective={float(plan['objective']):.6f} "
        f"shrinkage={shrinkage_s}"
    )
    if not bool(plan.get("feasible")):
        print(
            "warning: cheapest measured candidates exceed target bpw",
            file=sys.stderr,
            flush=True,
        )
    if shrinkage is not None:
        print(f"--hessian-shrinkage {float(shrinkage):.3f}")
    print("layer bit overrides:")
    layer_bits = plan.get("layer_bits")
    if isinstance(layer_bits, list):
        for spec in layer_bits:
            print(f"  --layer-bits {spec}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurement",
        type=Path,
        help="measurement JSON file from ponyexl3-convert --measure-candidates, or '-' for stdin",
    )
    parser.add_argument("--bits", type=float, required=True, help="target weighted bpw")
    parser.add_argument(
        "--score",
        help="measurement stats key to optimize; defaults to measurement score_metric",
    )
    parser.add_argument(
        "--hessian-shrinkage",
        type=float,
        help="optimize only candidates with this global shrinkage",
    )
    parser.add_argument(
        "--head-bits",
        type=int,
        help="force lm_head to this K when lm_head is present in the measurement",
    )
    parser.add_argument(
        "--per-module-shrinkage",
        action="store_true",
        help="allow each module to choose its best measured shrinkage; diagnostic only",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    try:
        if args.hessian_shrinkage is not None and not 0.0 <= args.hessian_shrinkage <= 1.0:
            raise ValueError("--hessian-shrinkage must be in [0, 1]")
        if args.per_module_shrinkage and args.hessian_shrinkage is not None:
            raise ValueError("--per-module-shrinkage and --hessian-shrinkage are mutually exclusive")
        if args.head_bits is not None and not 1 <= args.head_bits <= 8:
            raise ValueError("--head-bits must be in [1, 8]")
        measurement = _load_measurement(args.measurement)
        plan = optimize_measurement_plan(
            measurement,
            target_bpw=args.bits,
            score_metric=args.score,
            hessian_shrinkage=args.hessian_shrinkage,
            per_module_shrinkage=bool(args.per_module_shrinkage),
            fixed_bits=None if args.head_bits is None else {"lm_head": args.head_bits},
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        _print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
