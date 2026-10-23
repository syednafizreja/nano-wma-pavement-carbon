"""Graphical abstract for the README: the durability-to-carbon chain and its key numbers.

    python docs/make_graphical_abstract.py     # reads results/results.json, writes docs/graphical_abstract.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text())
C_H, C_N, INK, MUTED = "#B03A2E", "#1F5C99", "#1d2733", "#6b7685"
STAGES = ["#e9eef5", "#e6efe8", "#f4ece1", "#efe7f2"]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})
fig = plt.figure(figsize=(12, 5.2), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(0, 52); ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(3, 48.3, "From laboratory durability to annualised carbon", fontsize=19, weight="bold", color=INK, va="center")
ax.text(3, 44.6, "Nano-silica warm mix asphalt (N-WMA) vs hot-mix asphalt (HMA) on a 120 mm flexible pavement, "
        "BZZ-100 dual wheel, Chengdu climate", fontsize=10.5, color=MUTED, va="center")

stages = [
    ("1  Material inputs", "stiffness ratio  R_E\nfatigue shift  β_f\nrutting factor  β_r\nTSR-driven moisture law"),
    ("2  Axisymmetric FE", "9-node elements, 10,530 dof\nverified vs Boussinesq (<0.4%)\n12 monthly temperature states\n132-solution response database"),
    ("3  Damage & life", "MEPDG fatigue & rutting\nMiner's rule, monthly steps\n4% traffic growth\nanchored at 15-yr HMA life"),
    ("4  Annualised carbon", "A1–A3 GWP per lane-km-yr\n61.31 vs 58.15 kg CO₂e/t\nOAT sensitivity\nMonte Carlo, n = 1,500"),
]
x0, w, gap, y, h = 3, 26.2, 2.4, 22.5, 18.5
for k, ((title, body), col) in enumerate(zip(stages, STAGES)):
    x = x0 + k * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.6", fc=col, ec="none"))
    ax.text(x + 1.6, y + h - 2.6, title, fontsize=12, weight="bold", color=INK, va="center")
    ax.text(x + 1.6, y + h - 5.2, body, fontsize=9.6, color=INK, va="top", linespacing=1.55)
    if k < 3:
        ax.annotate("", xy=(x + w + gap - 0.2, y + h / 2), xytext=(x + w + 0.4, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=14))

life_h, life_n = R["life_HMA"]["L"], R["life_N-WMA"]["L"]
c, mc = R["carbon"], R["mc"]
cards = [
    (f"{life_h:.1f} → {life_n:.1f} yr", f"service life (+{100 * (life_n / life_h - 1):.0f}%)", C_N),
    (f"{c['ann_HMA']:.2f} → {c['ann_NWMA']:.2f}", "t CO₂e per lane-km per year", C_N),
    (f"−{c['dG_pct']:.1f}%", f"annualised GWP\nMonte Carlo 5–95%: {mc['dG']['p05']:.0f}–{mc['dG']['p95']:.0f}%", C_N),
    (f"{100 * (1 - c['production_share']):.0f}%", "of the benefit comes from\ndurability, not the plant", C_H),
]
cw, cg = 26.2, 2.4
for k, (big, small, col) in enumerate(cards):
    x = x0 + k * (cw + cg)
    ax.add_patch(FancyBboxPatch((x, 4.0), cw, 13.5, boxstyle="round,pad=0.25,rounding_size=1.6",
                                fc="white", ec=col, lw=1.6))
    ax.text(x + cw / 2, 12.6, big, fontsize=21, weight="bold", color=col, ha="center", va="center")
    ax.text(x + cw / 2, 7.4, small, fontsize=9.4, color=INK, ha="center", va="center", linespacing=1.4)

ax.text(117, 1.4, f"Main uncertainty: intrinsic fatigue resistance β_f (Spearman ρ = {mc['spearman']['beta_f']:.2f}). "
        "Ratios, not absolute lives, are the transferable result.", fontsize=8.6, color=MUTED, ha="right", va="center")
fig.savefig(ROOT / "docs" / "graphical_abstract.png", facecolor="white")
print("written docs/graphical_abstract.png")
