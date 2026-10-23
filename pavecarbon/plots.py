"""plots.py -- Figures 1-6 of the paper (PNG at 600 dpi and vector PDF)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .pavement import MONTHS, build_mesh, A_LOAD, WHEEL_SPACING, LAYER_DEF

C_H, C_N, GREY = "#B03A2E", "#1F5C99", "#555555"


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 600,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, path):
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig1_structure(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a.grid(False); a.set_xlim(-450, 900); a.set_ylim(1000, -170); a.axis("off")
    fills = ["#3a3a3a", "#555555", "#707070", "#c9b28a", "#e1d3b0", "#efe8d8"]
    labels = ["AC upper sublayer, 40 mm", "AC middle sublayer, 40 mm", "AC lower sublayer, 40 mm",
              "Granular base, 350 mm, E = 400 MPa", "Granular subbase, 300 mm, E = 250 MPa",
              "Subgrade, E = 100 MPa"]
    hatches = [None, None, None, "..", "//", None]
    for (name, t, bot, nu, E), f, lab, h in zip(LAYER_DEF, fills, labels, hatches):
        bb = min(bot, 1000)
        a.add_patch(Rectangle((-450, t), 1350, bb - t, fc=f, ec="white", lw=0.6, hatch=h))
        tc = "white" if t < 120 else "black"
        a.text(880, (t + bb) / 2, lab, ha="right", va="center", fontsize=7, color=tc)
    for xc in (0.0, WHEEL_SPACING):
        a.add_patch(Rectangle((xc - A_LOAD, -60), 2 * A_LOAD, 45, fc="#222222", ec="none"))
        for xx in np.linspace(xc - A_LOAD, xc + A_LOAD, 5):
            a.annotate("", xy=(xx, -2), xytext=(xx, -14), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
    a.text(WHEEL_SPACING / 2, -95, "BZZ-100 dual wheel: 0.70 MPa, a = 106.5 mm,\ncentre spacing 319.5 mm",
           ha="center", va="bottom", fontsize=7)
    a.plot([0, WHEEL_SPACING / 2], [120, 120], color="#f1c40f", lw=2.2)
    a.text(-440, 145, "ε_t search line (bottom of AC)", fontsize=6.5, color="black", va="top")
    a.set_title("(a)", loc="left", fontsize=9)
    mesh = build_mesh(1)
    for r in mesh.r_edges[mesh.r_edges <= 650]:
        b.plot([r, r], [0, 900], color=GREY, lw=0.4)
    for z in mesh.z_edges:
        b.plot([0, 650], [z, z], color=GREY, lw=0.4)
    for z in (120.0, 470.0, 770.0):
        b.plot([0, 650], [z, z], color="black", lw=1.2)
        b.text(655, z, f"z = {z:g} mm", va="center", fontsize=6.5)
    b.plot([0, A_LOAD], [0, 0], color=C_H, lw=3)
    b.set_xlim(0, 760); b.set_ylim(900, -20); b.grid(False)
    b.set_xlabel("Radial distance r (mm)"); b.set_ylabel("Depth z (mm)")
    b.set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout(); save(fig, Path(out) / "fig1_structure_mesh")


def fig2_verification(curves, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(curves["cf"], curves["z"], color="black", lw=1.4, label="Boussinesq closed form")
    ax.plot(curves["fe"][::4], curves["z"][::4], "o", ms=3.5, mfc="white", mec=C_N, label="Axisymmetric FE (Q9)")
    ax.invert_yaxis(); ax.set_xlabel("Vertical stress on the load axis, σ_z (MPa)"); ax.set_ylabel("Depth z (mm)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, Path(out) / "fig2_verification")


def fig3_monthly_strain(ms, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    x = np.arange(12)
    for ax, yr, lab in zip(axs, (0, 15), ("(a) As built", "(b) After 15 years of moisture conditioning")):
        d = ms[ms.year == yr]
        ax.plot(x, d.HMA_ue, "-o", color=C_H, ms=3.5, label="HMA")
        ax.plot(x, d.NWMA_ue, "--s", color=C_N, ms=3.5, label="N-WMA")
        ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MONTHS])
        ax.set_title(lab, loc="left", fontsize=9)
    axs[0].set_ylabel("ε_t at bottom of asphalt (µε)")
    axs[0].legend(frameon=False)
    fig.tight_layout(); save(fig, Path(out) / "fig3_monthly_strain")


def fig4_damage(hist, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    h, n = hist["HMA"], hist["N-WMA"]
    a.plot(h["t"], h["Df"], color=C_H, label="HMA"); a.plot(n["t"], n["Df"], "--", color=C_N, label="N-WMA")
    a.axhline(1.0, color=GREY, ls=":", lw=1); a.set_xlim(0, 35); a.set_ylim(0, 2.2)
    a.set_xlabel("Age (years)"); a.set_ylabel("Cumulative fatigue damage D"); a.legend(frameon=False)
    a.set_title("(a)", loc="left", fontsize=9)
    b.plot(h["t"], h["rut"], color=C_H, label="HMA"); b.plot(n["t"], n["rut"], "--", color=C_N, label="N-WMA")
    b.axhline(10.0, color=GREY, ls=":", lw=1); b.set_xlim(0, 35); b.set_ylim(0, 16)
    b.set_xlabel("Age (years)"); b.set_ylabel("Asphalt-layer rut depth (mm)")
    b.set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout(); save(fig, Path(out) / "fig4_damage")


def fig5_carbon(res, mc, out):
    c = res["carbon"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[0.8, 1.2]))
    vals = [c["ann_HMA"], c["ann_NWMA"]]
    bars = a.bar(["HMA", "N-WMA"], vals, color=[C_H, C_N], width=0.55)
    bars[1].set_hatch("//")
    for x, v in zip([0, 1], vals):
        a.text(x, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)
    a.set_ylabel("Annualised A1–A3 GWP\n(t CO₂e per lane-km per year)"); a.set_ylim(0, 5.2)
    a.set_title("(a)", loc="left", fontsize=9)
    b.hist(mc.dG_pct, bins=30, color=C_N, alpha=0.75, edgecolor="white")
    for q, ls in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
        b.axvline(mc.dG_pct.quantile(q), color="black", ls=ls, lw=1)
    b.axvline(c["dG_pct"], color=C_H, ls="--", lw=1.2, label="Deterministic base case")
    b.set_xlabel("Reduction in annualised GWP (%)"); b.set_ylabel("Number of samples")
    b.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    b.set_title("(b)", loc="right", fontsize=9)
    fig.tight_layout(); save(fig, Path(out) / "fig5_carbon")


def fig6_tornado(oat, out):
    base = float(oat.loc[oat.case == "base case", "dG_pct"].iloc[0])
    pairs = [("β_f (1.0–1.8)", "beta_f = 1.00", "beta_f = 1.80"),
             ("Traffic growth (0–7%)", "traffic growth 7%", "traffic growth 0%"),
             ("HMA TSR (0.88–0.75)", "TSR (HMA) = 0.88", "TSR (HMA) = 0.75"),
             ("R_E (0.95–1.30)", "R_E = 0.95", "R_E = 1.30"),
             ("N-WMA production GWP (64.0–58.15)", "N-WMA production GWP = 64.0", "base case"),
             ("Traffic level (×0.5–×2)", "traffic x 0.5", "traffic x 2"),
             ("τ (8–3 years)", "tau = 8 yr", "tau = 3 yr"),
             ("N-WMA TSR (0.85–0.93)", "TSR (N-WMA) = 0.85", "TSR (N-WMA) = 0.93"),
             ("β_r (0.50–1.00)", "beta_r = 0.50", "beta_r = 1.00")]
    rows = []
    for lab, lo, hi in pairs:
        v1 = float(oat.loc[oat.case == lo, "dG_pct"].iloc[0]); v2 = float(oat.loc[oat.case == hi, "dG_pct"].iloc[0])
        rows.append((lab, min(v1, v2), max(v1, v2)))
    rows.sort(key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, (lab, lo, hi) in enumerate(rows):
        ax.barh(k, hi - base, left=base, color=C_N, height=0.6)
        ax.barh(k, lo - base, left=base, color=C_H, height=0.6, hatch="//")
    ax.axvline(base, color="black", lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel(f"Reduction in annualised GWP (%), base case {base:.1f}%")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, Path(out) / "fig6_tornado")


def make_all(R, out="results"):
    style()
    fig1_structure(out); fig2_verification(R["curves"], out); fig3_monthly_strain(R["ms"], out)
    fig4_damage(R["hist"], out); fig5_carbon(R["res"], R["mc"], out); fig6_tornado(R["oat"], out)
