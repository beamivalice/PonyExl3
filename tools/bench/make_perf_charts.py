#!/usr/bin/env python3
"""README decode-throughput charts — Apple M5 Max vs M1 Max (32 GB), per model.

Measured numbers are hardcoded (the bench jsonl is ephemeral): M5 Max from the
sandwiched campaign (tools/bench/perf_chart_bench.py), M1 Max from a 32 GB run.
All speculative numbers are verify-gated (token-identical to plain greedy).
Colour code: purple = M5 Max, blue = M1 Max.

    python tools/bench/make_perf_charts.py
"""

from __future__ import annotations
from pathlib import Path

# (label, {hw: tok_s}) — drafters not run on a given machine are absent.
DATA = {
    "27b4_15": {
        "title": "Qwen3.6-27B · 4.15 bpw", "subtitle": "dense",
        "prefill": {"M5 Max": 662, "M1 Max": 126}, "mem": 15.1,
        "rows": [
            ("Plain decode", {"M5 Max": 16.6, "M1 Max": 4.0}),
            ("+ EAGLE-3",    {"M5 Max": 24.2}),
            ("+ MTP",        {"M5 Max": 28.3, "M1 Max": 8.7}),
            ("+ DFlash",     {"M5 Max": 37.8, "M1 Max": 15.0}),
        ],
    },
    "moe": {
        "title": "Qwen3.6-35B-A3B", "subtitle": "MoE · 8 of 256 experts/token",
        "prefill": {"M5 Max": 2775, "M1 Max": 497}, "mem": 17.2,
        "rows": [
            ("Plain decode", {"M5 Max": 68.5, "M1 Max": 23.5}),
            ("+ EAGLE-3",    {"M5 Max": 79.8}),
        ],
    },
    "27b8": {
        "title": "Qwen3.6-27B · 8.00 bpw", "subtitle": "dense · near-lossless",
        "prefill": {"M5 Max": 642}, "mem": 26.3,
        "rows": [
            ("Plain decode", {"M5 Max": 15.6}),
            ("+ DFlash",     {"M5 Max": 31.6}),
        ],
    },
}

W, PAD, X0 = 760, 26, 200
BAR5, BAR1, GAP, GROUP = 26, 18, 6, 16
TOP = 126
CARD, INK, SUB, INK2 = "#0f172a", "#e2e8f0", "#94a3b8", "#cbd5e1"
M5, M1 = "#a855f7", "#3b82f6"   # purple = M5 Max, blue = M1 Max


def build(meta: dict) -> str:
    rows = meta["rows"]
    plain5 = rows[0][1]["M5 Max"]
    plain1 = rows[0][1].get("M1 Max")
    vmax = max(v for _, d in rows for v in d.values()) * 1.16
    barw = W - X0 - PAD - 96
    fastest = max(range(len(rows)), key=lambda i: rows[i][1].get("M5 Max", 0))
    n_m1 = sum(1 for _, d in rows if "M1 Max" in d)
    has_m1 = n_m1 > 0
    h = TOP + len(rows) * BAR5 + n_m1 * (BAR1 + 2) + len(rows) * (GAP + GROUP) + 22

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{M5}"/><stop offset="1" stop-color="#c084fc"/>'
        '</linearGradient></defs>',
        f'<rect x="0" y="0" width="{W}" height="{h}" rx="16" fill="{CARD}"/>',
        f'<text x="{PAD}" y="40" fill="{INK}" font-size="23" font-weight="700">{meta["title"]}</text>',
        f'<text x="{PAD}" y="62" fill="{SUB}" font-size="14">{meta["subtitle"]}</text>',
        f'<text x="{PAD}" y="82" fill="{SUB}" font-size="12.5">decode tok/s · greedy, verify-gated (token-identical to plain)</text>',
    ]
    # legend — own row below the caption, swatches match the bars
    legend = [("M5 Max", M5)] + ([("M1 Max", M1)] if has_m1 else [])
    lx = PAD
    for name, col in legend:
        out.append(f'<rect x="{lx}" y="96" width="12" height="12" rx="3" fill="{col}"/>')
        out.append(f'<text x="{lx+18}" y="106" fill="{INK}" font-size="12.5">{name}</text>')
        lx += 30 + int(len(name) * 7.2) + 16

    y = TOP
    for i, (label, d) in enumerate(rows):
        hm1 = "M1 Max" in d
        pair_h = BAR5 + (BAR1 + 2 if hm1 else 0)
        out.append(f'<text x="{X0-12}" y="{y+pair_h/2+5:.0f}" fill="{INK}" font-size="14" text-anchor="end">{label}</text>')
        # M5 Max bar (purple; fastest = gradient + tag)
        v5 = d["M5 Max"]; bw = max(2, barw * v5 / vmax)
        out.append(f'<rect x="{X0}" y="{y}" width="{bw:.1f}" height="{BAR5}" rx="6" fill="{"url(#g)" if i==fastest else M5}"/>')
        sp = f"  {v5/plain5:.2f}×" if label != "Plain decode" else ""
        out.append(f'<text x="{X0+bw+9:.1f}" y="{y+BAR5/2+5:.0f}" fill="{INK}" font-size="14.5" '
                   f'font-weight="{700 if i==fastest else 600}">{v5:.0f}'
                   f'<tspan fill="{SUB}" font-weight="400" font-size="12">{sp}</tspan></text>')
        if i == fastest:
            out.append(f'<text x="{X0+bw+9:.1f}" y="{y-3:.0f}" fill="{M5}" font-size="10" font-weight="700">FASTEST</text>')
        # M1 Max bar (blue)
        if hm1:
            v1 = d["M1 Max"]; bw1 = max(2, barw * v1 / vmax); yy = y + BAR5 + 2
            out.append(f'<rect x="{X0}" y="{yy}" width="{bw1:.1f}" height="{BAR1}" rx="5" fill="{M1}"/>')
            sp1 = f"  {v1/plain1:.2f}×" if (label != "Plain decode" and plain1) else ""
            out.append(f'<text x="{X0+bw1+9:.1f}" y="{yy+BAR1/2+4:.0f}" fill="{INK2}" font-size="13">{v1:.1f}'
                       f'<tspan fill="{SUB}" font-size="11">{sp1}</tspan></text>')
        y += pair_h + GAP + GROUP

    pre = meta["prefill"]
    foot = "prefill " + " / ".join(f"{hw.split()[0]} {v}" for hw, v in pre.items()) + f" tok/s  ·  {meta['mem']:.1f} GB resident"
    out.append(f'<text x="{PAD}" y="{h-15}" fill="{SUB}" font-size="12.5">{foot}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    out_dir = Path("docs/assets"); out_dir.mkdir(parents=True, exist_ok=True)
    for key, meta in DATA.items():
        (out_dir / f"perf_{key}.svg").write_text(build(meta))
        print(f"wrote docs/assets/perf_{key}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
