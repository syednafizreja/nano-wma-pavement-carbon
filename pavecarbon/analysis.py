"""
analysis.py -- every number reported in the paper, written to results/.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

from . import lca
from .fe_axisym import AxisymModel, boussinesq_sigma_z_axis, boussinesq_w0
from .pavement import (MONTHS, CHENGDU_AIR_C, AC_SUB_MID, make_model, monthly_states, layer_moduli,
                       dual_wheel_response, single_wheel_eps_t, witczak_modulus_mpa)
from .damage import (HMA, NWMA, WMA_POOR, Mix, ResponseDB, Scenario, run_life, at_year,
                     anchor_scenario, moisture_factor)

MC_SEED = 2026
MC_N = 1500
RANGES = dict(R_E=(0.95, 1.30), beta_f=(1.0, 1.8), beta_r=(0.50, 1.00), TSR_N=(0.85, 0.93),
              TSR_H=(0.75, 0.88), tau=(3.0, 8.0))


def verification():
    rows = []
    cases = [("base mesh, 4 x 6 m", 1, 4000.0, 6000.0), ("2x mesh, 4 x 6 m", 2, 4000.0, 6000.0),
             ("base mesh, 8 x 12 m", 1, 8000.0, 12000.0)]
    curves = {}
    for label, ref, R, D in cases:
        m0 = make_model(ref, R=R, D=D)
        m = AxisymModel(m0.mesh, {L.name: 0.35 for L in m0.mesh.layers})
        sol = m.solve([100.0] * 6)
        for z in [50.0, 106.5, 200.0, 400.0]:
            fe = -sol.strain_stress(0.0, z)[1][1]
            cf = boussinesq_sigma_z_axis(0.70, 106.5, z)
            rows.append(dict(case=label, quantity=f"sigma_z at z={z:g} mm (MPa)", closed_form=cf, fe=fe,
                             error_pct=100 * (fe / cf - 1)))
        w0 = boussinesq_w0(0.70, 106.5, 100.0, 0.35)
        w = sol.surface_uz(0.0)
        rows.append(dict(case=label, quantity="surface deflection, centre (mm)", closed_form=w0, fe=w,
                         error_pct=100 * (w / w0 - 1)))
        if ref == 1 and R == 4000.0:
            zz = np.linspace(1.0, 1200.0, 120)
            curves = dict(z=zz, fe=np.array([-sol.strain_stress(0.0, z)[1][1] for z in zz]),
                          cf=boussinesq_sigma_z_axis(0.70, 106.5, zz))
            mesh_info = dict(elements=m.mesh.n_elem, dof=m.mesh.n_dof)
    return pd.DataFrame(rows), curves, mesh_info


def layered_checks(E):
    """Mesh convergence, Poisson's-ratio check and dual- vs single-wheel, April state."""
    i = 3
    base = make_model(1)
    fine = make_model(2)
    r1 = dual_wheel_response(base.solve(layer_moduli(E[i])))
    r2 = dual_wheel_response(fine.solve(layer_moduli(E[i])))
    single = single_wheel_eps_t(base.solve(layer_moduli(E[i])))
    nu = {}
    for v in (0.25, 0.45):
        rv = dual_wheel_response(make_model(1, nu_ac=v).solve(layer_moduli(E[i])))
        nu[v] = dict(eps_t_change_pct=100 * (rv["eps_t"] / r1["eps_t"] - 1),
                     eps_v3_change_pct=100 * (rv["eps_v3"] / r1["eps_v3"] - 1))
    nu_all = []
    for v in (0.25, 0.45):
        m = make_model(1, nu_ac=v)
        for mm in range(12):
            a = dual_wheel_response(m.solve(layer_moduli(E[mm])))["eps_t"]
            b = dual_wheel_response(base.solve(layer_moduli(E[mm])))["eps_t"]
            nu_all.append(abs(a / b - 1) * 100)
    return dict(mesh_eps_t_change_pct=100 * (r2["eps_t"] / r1["eps_t"] - 1),
                mesh_eps_c_change_pct=100 * (r2["eps_c"] / r1["eps_c"] - 1),
                dual_over_single_pct=100 * (r1["eps_t"] / single - 1),
                governing_x_mm=r1["x_t"], poisson=nu, poisson_max_eps_t_change_all_months_pct=max(nu_all),
                fine_mesh=dict(elements=fine.mesh.n_elem, dof=fine.mesh.n_dof))


def response_table(db):
    rows = []
    for label, mix, year in [("HMA", HMA, 0), ("N-WMA", NWMA, 0), ("HMA", HMA, 15), ("N-WMA", NWMA, 15)]:
        mult = mix.R_E * moisture_factor(year, mix.TSR, mix.tau) if year > 0 else mix.R_E
        for i in (0, 3, 6):
            rows.append(dict(case=label, year=year, month=MONTHS[i], multiplier=mult,
                             E_lower_MPa=db.E[i, 2] * mult,
                             eps_t_ue=1e6 * db.get("eps_t", i, mult), sig_t_MPa=db.get("sig_t", i, mult),
                             eps_c_ue=1e6 * db.get("eps_c", i, mult), eps_v3_ue=1e6 * db.get("eps_v3", i, mult),
                             defl_mm=db.get("defl", i, mult)))
    return pd.DataFrame(rows)


def monthly_strain(db):
    rows = []
    for year in (0, 15):
        for i in range(12):
            mh = HMA.R_E * (moisture_factor(year, HMA.TSR, HMA.tau) if year else 1.0)
            mn = NWMA.R_E * (moisture_factor(year, NWMA.TSR, NWMA.tau) if year else 1.0)
            h, n = db.get("eps_t", i, mh) * 1e6, db.get("eps_t", i, mn) * 1e6
            rows.append(dict(year=year, month=MONTHS[i], HMA_ue=h, NWMA_ue=n, reduction_pct=100 * (1 - n / h)))
    return pd.DataFrame(rows)


def life_and_carbon(L_h, L_n, g_h=HMA.gwp, g_n=NWMA.gwp):
    return dict(L_HMA=L_h, L_NWMA=L_n, life_ext_pct=100 * (L_n / L_h - 1),
                dG_pct=100 * lca.reduction(g_h, L_h, g_n, L_n))


def oat(db, sc):
    base_h = run_life(db, HMA, sc)["L"]
    base_n = run_life(db, NWMA, sc)["L"]
    cases = [("base case", HMA, NWMA, sc, None)]
    add = lambda lab, h=HMA, n=NWMA, s=sc, gn=None: cases.append((lab, h, n, s, gn))
    add("beta_f = 1.00", n=replace(NWMA, beta_f=1.0)); add("beta_f = 1.80", n=replace(NWMA, beta_f=1.8))
    add("R_E = 0.95", n=replace(NWMA, R_E=0.95)); add("R_E = 1.30", n=replace(NWMA, R_E=1.30))
    add("TSR (N-WMA) = 0.85", n=replace(NWMA, TSR=0.85)); add("TSR (N-WMA) = 0.93", n=replace(NWMA, TSR=0.93))
    add("TSR (HMA) = 0.75", h=replace(HMA, TSR=0.75)); add("TSR (HMA) = 0.88", h=replace(HMA, TSR=0.88))
    add("tau = 3 yr", h=replace(HMA, tau=3.0), n=replace(NWMA, tau=3.0))
    add("tau = 8 yr", h=replace(HMA, tau=8.0), n=replace(NWMA, tau=8.0))
    add("beta_r = 0.50", n=replace(NWMA, beta_r=0.50)); add("beta_r = 1.00", n=replace(NWMA, beta_r=1.00))
    add("traffic growth 0%", s=replace(sc, growth=0.0)); add("traffic growth 7%", s=replace(sc, growth=0.07))
    add("traffic x 0.5", s=replace(sc, n0=sc.n0 * 0.5, years=120)); add("traffic x 2", s=replace(sc, n0=sc.n0 * 2.0))
    add("AI (MS-1) fatigue model", s=replace(sc, fatigue_model="ai"))
    add("moisture also reduces fatigue resistance", s=replace(sc, moisture_on_fatigue=True))
    add("N-WMA production GWP = 61.31", gn=61.31); add("N-WMA production GWP = 64.0", gn=64.0)
    add("WMA without nano benefit", n=WMA_POOR)
    rows = []
    for lab, h, n, s, gn in cases:
        rh, rn = run_life(db, h, s), run_life(db, n, s)
        d = life_and_carbon(rh["L"], rn["L"], g_n=gn if gn else n.gwp)
        d.update(case=lab, gov_HMA=_gov(rh), gov_NWMA=_gov(rn))
        rows.append(d)
    return pd.DataFrame(rows)[["case", "L_HMA", "L_NWMA", "life_ext_pct", "dG_pct", "gov_HMA", "gov_NWMA"]]


def _gov(r):
    lives = dict(fatigue=r["L_fatigue"], rutting=r["L_rut"], subgrade=r["L_subgrade"])
    lives = {k: v for k, v in lives.items() if not np.isnan(v)}
    lo = min(lives.values())
    return "+".join(k for k, v in lives.items() if abs(v - lo) < 0.05)


def decomposition(db, sc):
    steps = [("HMA reference", HMA),
             ("+ stiffness, R_E = 1.10", replace(HMA, R_E=1.10)),
             ("+ moisture resistance, TSR 0.82 -> 0.91", replace(HMA, R_E=1.10, TSR=0.91)),
             ("+ intrinsic fatigue resistance, beta_f = 1.30", replace(HMA, R_E=1.10, TSR=0.91, beta_f=1.30)),
             ("+ rutting resistance, beta_r = 0.70 (full N-WMA)", NWMA)]
    rows, prev = [], None
    for lab, mix in steps:
        r = run_life(db, mix, sc)
        rows.append(dict(step=lab, L_fatigue=r["L_fatigue"], increment=(r["L_fatigue"] - prev) if prev else np.nan,
                         L_rut=r["L_rut"]))
        prev = r["L_fatigue"]
    return pd.DataFrame(rows)


def monte_carlo(db, sc, n=MC_N, seed=MC_SEED):
    rng = np.random.default_rng(seed)
    u = lambda k: rng.uniform(*RANGES[k], n)
    S = pd.DataFrame(dict(R_E=u("R_E"), beta_f=u("beta_f"), beta_r=u("beta_r"), TSR_N=u("TSR_N"),
                          TSR_H=u("TSR_H"), tau=u("tau"),
                          gwp_H=HMA.gwp * rng.uniform(0.97, 1.03, n), gwp_N=NWMA.gwp * rng.uniform(0.97, 1.03, n)))
    Lh, Ln = np.empty(n), np.empty(n)
    for q, row in S.iterrows():
        h = replace(HMA, TSR=row.TSR_H, tau=row.tau)
        nm = replace(NWMA, R_E=row.R_E, beta_f=row.beta_f, beta_r=row.beta_r, TSR=row.TSR_N, tau=row.tau)
        Lh[q], Ln[q] = run_life(db, h, sc)["L"], run_life(db, nm, sc)["L"]
    S["L_HMA"], S["L_NWMA"] = Lh, Ln
    S["life_ext_pct"] = 100 * (Ln / Lh - 1)
    S["dG_pct"] = 100 * (1 - (S.gwp_N / S.gwp_H) * (Lh / Ln))
    rho = {k: spearmanr(S[k], S.dG_pct).statistic for k in ["R_E", "beta_f", "beta_r", "TSR_N", "TSR_H", "tau", "gwp_H", "gwp_N"]}
    return S, rho


def seasonal_damage_share(db, sc, mix):
    h = run_life(db, mix, sc, keep_history=True)
    inc = np.diff(np.concatenate([[0.0], h["Df"]]))[:12 * 15]
    by_month = inc.reshape(15, 12).sum(axis=0)
    by_month /= by_month.sum()
    return by_month


def run_all(outdir="results"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    T, E = monthly_states()
    res = {}
    ver, curves, mesh_info = verification()
    ver.to_csv(out / "verification_boussinesq.csv", index=False)
    res["mesh"] = mesh_info
    res["layered_checks"] = layered_checks(E)
    res["E21_MPa"] = float(witczak_modulus_mpa(21.0))
    pd.DataFrame(dict(month=MONTHS, air_C=CHENGDU_AIR_C, T20=T[:, 0], T60=T[:, 1], T100=T[:, 2],
                      E20=E[:, 0], E60=E[:, 1], E100=E[:, 2])).to_csv(out / "monthly_states.csv", index=False)
    db = ResponseDB()
    res["n_solutions_db"] = db.n_solutions
    sc = anchor_scenario(db)
    res["scenario"] = asdict(sc)
    res["traffic_15yr_cum"] = sc.n0 * ((1 + sc.growth) ** 15 - 1) / sc.growth
    response_table(db).to_csv(out / "responses_table4.csv", index=False)
    ms = monthly_strain(db); ms.to_csv(out / "monthly_eps_t.csv", index=False)
    hist = {}
    for mix in (HMA, NWMA, WMA_POOR):
        h = run_life(db, mix, sc, keep_history=True)
        hist[mix.name] = h
        res[f"life_{mix.name}"] = {k: float(h[k]) for k in ("L_fatigue", "L_rut", "L_subgrade", "L")}
        res[f"life_{mix.name}"].update(D15=at_year(h, "Df", 15.0), rut15=at_year(h, "rut", 15.0),
                                       m15=float(mix.R_E * moisture_factor(15.0, mix.TSR, mix.tau)))
    pd.DataFrame(dict(t=hist["HMA"]["t"], Df_HMA=hist["HMA"]["Df"], Df_NWMA=hist["N-WMA"]["Df"],
                      rut_HMA=hist["HMA"]["rut"], rut_NWMA=hist["N-WMA"]["rut"])).to_csv(out / "damage_history.csv", index=False)
    share = seasonal_damage_share(db, sc, HMA); share_n = seasonal_damage_share(db, sc, NWMA)
    res["damage_share_May_Sep"] = dict(HMA=float(share[4:9].sum()), NWMA=float(share_n[4:9].sum()))
    res["damage_share_Jan"] = dict(HMA=float(share[0]), NWMA=float(share_n[0]))
    Lh, Ln = res["life_HMA"]["L"], res["life_N-WMA"]["L"]
    carbon = dict(
        mass_t=lca.MASS_T_PER_LANE_KM,
        embodied_HMA=lca.embodied_t(HMA.gwp), embodied_NWMA=lca.embodied_t(NWMA.gwp),
        ann_HMA=lca.annualised_t_per_year(HMA.gwp, Lh), ann_NWMA=lca.annualised_t_per_year(NWMA.gwp, Ln),
        ann_per_t_HMA=HMA.gwp / Lh, ann_per_t_NWMA=NWMA.gwp / Ln,
        dG_pct=100 * lca.reduction(HMA.gwp, Lh, NWMA.gwp, Ln),
        production_share=float(np.log(NWMA.gwp / HMA.gwp) / np.log((NWMA.gwp / HMA.gwp) * (Lh / Ln))),
        break_even_gwp=lca.break_even_gwp(HMA.gwp, Lh, Ln),
        h40_prorated_HMA=lca.horizon_total(HMA.gwp, Lh)[0], h40_prorated_NWMA=lca.horizon_total(NWMA.gwp, Ln)[0],
        h40_discrete_HMA=lca.horizon_total(HMA.gwp, Lh, discrete=True), h40_discrete_NWMA=lca.horizon_total(NWMA.gwp, Ln, discrete=True),
        dG_poor_pct=100 * lca.reduction(HMA.gwp, Lh, WMA_POOR.gwp, res["life_WMA without nano benefit"]["L"]))
    res["carbon"] = carbon
    o = oat(db, sc); o.to_csv(out / "sensitivity_oat.csv", index=False)
    dec = decomposition(db, sc); dec.to_csv(out / "decomposition.csv", index=False)
    S, rho = monte_carlo(db, sc); S.to_csv(out / "monte_carlo.csv", index=False)
    res["mc"] = dict(n=MC_N, seed=MC_SEED, spearman=rho,
                     life_ext=dict(p05=S.life_ext_pct.quantile(0.05), p50=S.life_ext_pct.median(), p95=S.life_ext_pct.quantile(0.95)),
                     dG=dict(p05=S.dG_pct.quantile(0.05), p50=S.dG_pct.median(), p95=S.dG_pct.quantile(0.95)),
                     n_increase=int((S.dG_pct < 0).sum()), mean_R_E=S.R_E.mean(), mean_beta_f=S.beta_f.mean())
    # thermal stress: fully restrained layer, closed form sigma = E alpha dT / (1 - nu)
    alpha, dT, nu = 2.0e-5, 15.0, 0.35
    res["thermal_MPa"] = dict(HMA=E[0, 0] * alpha * dT / (1 - nu), NWMA=E[0, 0] * NWMA.R_E * alpha * dT / (1 - nu))
    res["versions"] = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                           pandas=pd.__version__)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x)))
    return dict(res=res, ver=ver, curves=curves, db=db, sc=sc, hist=hist, ms=ms, oat=o, dec=dec, mc=S, T=T, E=E)
