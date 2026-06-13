#!/usr/bin/env python3
"""Build README decode-throughput charts from perf_chart_bench results.

Reads a JSONL of RESULT rows (multiple sandwiched passes), keeps the PEAK
decode tok/s per config (thermally-clean reading), and emits one self-
contained SVG bar chart per model into docs/assets/.

    python tools/bench/make_perf_charts.py /tmp/perf_results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# label prefix -> (model title, subtitle)
MODELS = {
    "27b4.15": ("Qwen3.6-27B · 4.15 bpw", "dense"),
    "27b8": ("Qwen3.6-27B · 8.00 bpw", "dense · near-lossless"),
    "moe": ("Qwen3.6-35B-A3B", "MoE · 8 of 256 experts/token"),
}
# config token -> (display name, is_spec)
# n-gram lookup is intentionally omitted from the bars: it is draft-free and
# workload-dependent (wins big on copy/edit, ~neutral on novel text like this
# prompt). Showing it here would misrepresent it — it gets a prose note instead.
CONFIGS = [
    ("plain", "Plain decode", False),
    ("eagle3", "+ EAGLE-3", True),
    ("mtp", "+ MTP spec", True),
    ("dflash", "+ DFlash (k=7)", True),
]

W, PAD = 760, 26
BAR_H, GAP = 40, 16
ACCENT = "#7c3aed"      # violet — fastest bar
BAR = "#6366f1"         # indigo — spec bars
BASE = "#94a3b8"        # slate — plain baseline
CARD = "#0f172a"        # slate-900 card bg
INK = "#e2e8f0"         # text
SUB = "#94a3b8"         # muted text
GRID = "#1e293b"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg(title: str, subtitle: str, prefill: float | None, mem: float | None,
              bars: list[tuple[str, float, bool, float]]) -> str:
    """bars: (label, tok_s, is_fastest, speedup)."""
    top = 96
    h = top + len(bars) * (BAR_H + GAP) + 64
    vmax = max(v for _, v, _, _ in bars) * 1.16
    x0 = 200
    bar_w_max = W - x0 - PAD - 70

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        '<defs>'
        f'<linearGradient id="g" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{ACCENT}"/><stop offset="1" stop-color="#a855f7"/>'
        '</linearGradient>'
        f'<linearGradient id="b" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{BAR}"/><stop offset="1" stop-color="#818cf8"/>'
        '</linearGradient></defs>',
        f'<rect x="0" y="0" width="{W}" height="{h}" rx="16" fill="{CARD}"/>',
        f'<text x="{PAD}" y="40" fill="{INK}" font-size="23" font-weight="700">{_esc(title)}</text>',
        f'<text x="{PAD}" y="64" fill="{SUB}" font-size="14">{_esc(subtitle)}</text>',
    ]
    # caption: hardware + prefill + memory
    cap = "Apple M5 Max · decode tok/s · greedy, verify-gated (token-identical)"
    out.append(f'<text x="{PAD}" y="84" fill="{SUB}" font-size="12.5">{_esc(cap)}</text>')

    for i, (label, v, fastest, speedup) in enumerate(bars):
        y = top + i * (BAR_H + GAP)
        bw = max(2, bar_w_max * v / vmax)
        if fastest:
            fill = "url(#g)"
        elif label.startswith("+"):
            fill = "url(#b)"
        else:
            fill = BASE
        out.append(
            f'<text x="{x0 - 12}" y="{y + BAR_H/2 + 5}" fill="{INK}" font-size="14.5" '
            f'text-anchor="end">{_esc(label)}</text>'
        )
        out.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="{BAR_H}" rx="7" fill="{fill}"/>')
        vlabel = f"{v:.0f}"
        sp = f"  {speedup:.2f}×" if speedup and speedup > 1.001 else ""
        out.append(
            f'<text x="{x0 + bw + 10:.1f}" y="{y + BAR_H/2 + 5}" fill="{INK}" '
            f'font-size="15" font-weight="{700 if fastest else 600}">{vlabel}'
            f'<tspan fill="{SUB}" font-weight="500" font-size="13">{sp}</tspan></text>'
        )
        if fastest:
            out.append(
                f'<text x="{x0 + bw + 10:.1f}" y="{y - 3:.1f}" fill="{ACCENT}" '
                f'font-size="10.5" font-weight="700">FASTEST</text>'
            )

    foot = []
    if prefill:
        foot.append(f"prefill {prefill:.0f} tok/s")
    if mem:
        foot.append(f"{mem:.1f} GB resident")
    foot.append("4-bit weights stay resident — decoded in-kernel")
    out.append(
        f'<text x="{PAD}" y="{h - 22}" fill="{SUB}" font-size="13">{_esc("  ·  ".join(foot))}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/perf_results.jsonl")
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    # peak per (label) across passes
    decode: dict[str, float] = {}
    prefill: dict[str, float] = {}
    mem: dict[str, float] = {}
    speed_extra: dict[str, dict] = {}
    for r in rows:
        lab = r["label"]
        if r.get("mode") == "prefill" and r.get("prefill_tps"):
            prefill[lab] = max(prefill.get(lab, 0), r["prefill_tps"])
        if r.get("decode_tps"):
            if r["decode_tps"] > decode.get(lab, 0):
                decode[lab] = r["decode_tps"]
                speed_extra[lab] = {"accept": r.get("accept"), "tpc": r.get("tok_per_cycle")}
        if r.get("active_gb"):
            mem[lab] = min(mem.get(lab, 1e9), r["active_gb"])

    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for prefix, (title, subtitle) in MODELS.items():
        plain = decode.get(f"{prefix} plain")
        if not plain:
            continue
        bars = []
        for tok, disp, _spec in CONFIGS:
            v = decode.get(f"{prefix} {tok}")
            if v:
                bars.append([disp, v, False, v / plain])
        # mark fastest
        fi = max(range(len(bars)), key=lambda i: bars[i][1])
        bars[fi][2] = True
        # order: baseline first, then ascending tok/s
        base = [b for b in bars if b[0] == "Plain decode"]
        rest = sorted([b for b in bars if b[0] != "Plain decode"], key=lambda b: b[1])
        ordered = base + rest
        svg = build_svg(
            title, subtitle,
            prefill.get(f"{prefix} prefill"),
            mem.get(f"{prefix} plain"),
            [(b[0], b[1], b[2], b[3]) for b in ordered],
        )
        p = out_dir / f"perf_{prefix.replace('.', '_')}.svg"
        p.write_text(svg)
        written.append(str(p))
        print(f"wrote {p}  (plain {plain:.0f} → fastest {max(b[1] for b in bars):.0f} tok/s)")
    if not written:
        print("no decode results yet")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
