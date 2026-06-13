#!/usr/bin/env python3
"""Cross-platform decode comparison: Apple M5 Max (this repo) vs RTX 4090
(exllamav3), same EXL3 weights. 4090 numbers are plain TG (no drafter);
M5 Max drafters are verify-gated (token-identical). Decode is the
latency-bound regime — the 4090 wins prefill on raw compute (see README)."""

from __future__ import annotations
from pathlib import Path

# (model, [(platform, tok_s, note, kind)])  kind: m5 | m5spec | gpu
GROUPS = [
    ("Qwen3.6-27B · 4.15 bpw", [
        ("RTX 4090", 33.0, "plain", "gpu"),
        ("M5 Max", 16.6, "plain", "m5"),
        ("M5 Max", 38.0, "+ DFlash", "m5spec"),
    ]),
    ("Qwen3.6-35B-A3B · 4.00 bpw", [
        ("RTX 4090", 52.0, "plain", "gpu"),
        ("M5 Max", 68.5, "plain", "m5"),
        ("M5 Max", 80.0, "+ EAGLE-3", "m5spec"),
    ]),
]
COL = {"gpu": "#76b900", "m5": "#94a3b8", "m5spec": "#a855f7"}
CARD, INK, SUB = "#0f172a", "#e2e8f0", "#94a3b8"

W, PAD = 760, 26
X0 = 250
BAR_H, GAP, GROUP_GAP = 34, 9, 34
TOP = 104


def main() -> int:
    vmax = max(v for _, bars in GROUPS for _, v, _, _ in bars) * 1.18
    barw = W - X0 - PAD - 96
    n_bars = sum(len(b) for _, b in GROUPS)
    h = TOP + n_bars * (BAR_H + GAP) + len(GROUPS) * GROUP_GAP + 30

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect x="0" y="0" width="{W}" height="{h}" rx="16" fill="{CARD}"/>',
        f'<text x="{PAD}" y="40" fill="{INK}" font-size="23" font-weight="700">EXL3 decode: Apple M5 Max vs RTX 4090</text>',
        f'<text x="{PAD}" y="64" fill="{SUB}" font-size="13.5">tok/s, greedy · same EXL3 weights · 4090 = exllamav3 plain TG · M5 Max drafters verify-gated (token-identical)</text>',
        f'<text x="{PAD}" y="84" fill="{SUB}" font-size="12.5">M5 Max = 128 GB laptop SoC · RTX 4090 = 24 GB desktop · the 4090 wins prefill on raw compute (see Performance)</text>',
    ]
    # legend
    leg = [("RTX 4090 (plain)", "gpu"), ("M5 Max (plain)", "m5"), ("M5 Max (+drafter)", "m5spec")]
    lx = X0
    for name, k in leg:
        out.append(f'<rect x="{lx}" y="92" width="12" height="12" rx="3" fill="{COL[k]}"/>')
        out.append(f'<text x="{lx+17}" y="102" fill="{SUB}" font-size="12">{name}</text>')
        lx += 22 + len(name) * 7.0
    y = TOP
    for title, bars in GROUPS:
        out.append(f'<text x="{PAD}" y="{y+14}" fill="{INK}" font-size="14.5" font-weight="700">{title}</text>')
        y += 22
        for plat, v, note, kind in bars:
            bw = max(2, barw * v / vmax)
            label = f"{plat} {note}".strip()
            out.append(f'<text x="{X0-12}" y="{y+BAR_H/2+5:.0f}" fill="{INK}" font-size="13.5" text-anchor="end">{label}</text>')
            out.append(f'<rect x="{X0}" y="{y}" width="{bw:.1f}" height="{BAR_H}" rx="6" fill="{COL[kind]}"/>')
            bold = 700 if kind != "m5" else 600
            out.append(f'<text x="{X0+bw+10:.1f}" y="{y+BAR_H/2+5:.0f}" fill="{INK}" font-size="14.5" font-weight="{bold}">{v:.0f}</text>')
            y += BAR_H + GAP
        y += GROUP_GAP - GAP
    # takeaway
    out.append(f'<text x="{PAD}" y="{h-14}" fill="{SUB}" font-size="12.5">M5 Max wins MoE decode outright (68 vs 52); on dense it trails plain but its verify-gated drafters pull ahead.</text>')
    out.append("</svg>")
    p = Path("docs/assets/vs_rtx4090_decode.svg")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
