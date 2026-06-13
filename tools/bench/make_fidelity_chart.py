#!/usr/bin/env python3
"""CUDA↔MLX fidelity chart: per-module cosine similarity of the MLX Metal
EXL3 forward against the reference CUDA exllamav3 forward (same EXL3 weights,
Qwen3.6-35B-A3B, fixed-seed 512-token reference). Data from compare_trace.py
(docs/drifts_investigation.md). Shows that this is *real* EXL3 — it tracks the
canonical CUDA engine to the fp16 cross-platform floor through all 40 layers."""

from __future__ import annotations

from pathlib import Path

# (module label, cosine to CUDA) — measured 2026-06-13, compare_trace.py
COS = [
    ("emb", 1.0000000), ("0", 0.9999980), ("1", 0.9999323), ("2", 0.9999461),
    ("3", 0.9999356), ("4", 0.9999301), ("5", 0.9999089), ("6", 0.9999188),
    ("7", 0.9998977), ("8", 0.9998936), ("9", 0.9998764), ("10", 0.9998687),
    ("11", 0.9998495), ("12", 0.9998155), ("13", 0.9985848), ("14", 0.9984115),
    ("15", 0.9981697), ("16", 0.9984282), ("17", 0.9985958), ("18", 0.9987229),
    ("19", 0.9987639), ("20", 0.9989671), ("21", 0.9987211), ("22", 0.9982797),
    ("23", 0.9981500), ("24", 0.9980050), ("25", 0.9982279), ("26", 0.9983857),
    ("27", 0.9983543), ("28", 0.9986667), ("29", 0.9986653), ("30", 0.9988711),
    ("31", 0.9980710), ("32", 0.9977375), ("33", 0.9984429), ("34", 0.9990467),
    ("35", 0.9992438), ("36", 0.9989343), ("37", 0.9982232), ("38", 0.9990020),
    ("39", 0.9988636), ("norm", 0.9969304), ("logits", 0.9990685),
]

W, H = 760, 300
L, R, T, B = 56, 26, 96, 52        # plot margins
YLO, YHI = 0.995, 1.0
CARD, INK, SUB, GRID = "#0f172a", "#e2e8f0", "#94a3b8", "#1e293b"
LINE, FILL, BAND = "#a855f7", "#7c3aed", "#1e293b"


def _x(i: int, n: int) -> float:
    return L + (W - L - R) * i / (n - 1)


def _y(v: float) -> float:
    return T + (H - T - B) * (1 - (v - YLO) / (YHI - YLO))


def main() -> int:
    n = len(COS)
    pts = [(_x(i, n), _y(v)) for i, (_, v) in enumerate(COS)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{L},{_y(YLO):.1f} " + poly + f" {W-R},{_y(YLO):.1f}"

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<defs><linearGradient id="f" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{FILL}" stop-opacity="0.45"/>'
        f'<stop offset="1" stop-color="{FILL}" stop-opacity="0.03"/></linearGradient></defs>',
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="16" fill="{CARD}"/>',
        f'<text x="26" y="38" fill="{INK}" font-size="22" font-weight="700">Real EXL3: MLX Metal vs reference CUDA</text>',
        f'<text x="26" y="60" fill="{SUB}" font-size="13.5">Qwen3.6-35B-A3B · identical EXL3 weights · cosine similarity of each module\'s output</text>',
        f'<text x="26" y="80" fill="{SUB}" font-size="12.5">top-1 argmax matches · logit cosine 0.9991 · drift is the fp16 cross-platform floor, not a bug</text>',
    ]
    # y gridlines + labels
    for v in (0.995, 0.996, 0.997, 0.998, 0.999, 1.0):
        y = _y(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{L-8}" y="{y+4:.1f}" fill="{SUB}" font-size="11" text-anchor="end">{v:.3f}</text>')
    # area + line
    out.append(f'<polygon points="{area}" fill="url(#f)"/>')
    out.append(f'<polyline points="{poly}" fill="none" stroke="{LINE}" stroke-width="2.5" stroke-linejoin="round"/>')
    # bit-exact embed marker
    ex, ey = pts[0]
    out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{INK}"/>')
    out.append(f'<text x="{ex+8:.1f}" y="{ey+4:.1f}" fill="{INK}" font-size="11">embeddings bit-exact (1.000)</text>')
    # x labels (sparse)
    for i, (lab, _) in enumerate(COS):
        if lab in ("emb", "0", "10", "20", "30", "39", "norm", "logits"):
            x = _x(i, n)
            tag = "embed" if lab == "emb" else ("L" + lab if lab.isdigit() else lab)
            out.append(f'<text x="{x:.1f}" y="{H-26}" fill="{SUB}" font-size="11" text-anchor="middle">{tag}</text>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-8}" fill="{SUB}" font-size="12" text-anchor="middle">forward module (embed → 40 layers → norm → lm_head)</text>')
    out.append("</svg>")

    p = Path("docs/assets/fidelity_cuda.svg")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    lo = min(v for _, v in COS)
    print(f"wrote {p}  (min cosine {lo:.4f} at {[l for l,v in COS if v==lo][0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
