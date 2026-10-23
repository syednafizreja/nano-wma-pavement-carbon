"""
damage.py -- response database, moisture-degradation law, M-E transfer
functions, Miner's rule and service life.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .pavement import (MPA_TO_PSI, H_AC, AC_SUB_MID, c_to_f, monthly_states,
                       layer_moduli, dual_wheel_response, make_model)

RESP_KEYS = ["eps_t", "sig_t", "eps_c", "eps_v1", "eps_v2", "eps_v3", "defl"]
M_GRID = np.geomspace(0.60, 1.60, 11)


@dataclass(frozen=True)
class Mix:
    """Material inputs for one mixture (see Table 1 of the paper)."""
    name: str
    R_E: float = 1.0        # stiffness ratio to HMA at the same temperature
    beta_f: float = 1.0     # fatigue shift factor
    beta_r: float = 1.0     # rutting factor (relative)
    TSR: float = 0.82       # tensile strength ratio
    tau: float = 5.0        # moisture time constant, years
    gwp: float = 61.31      # A1-A3 GWP, kg CO2e/t


HMA = Mix("HMA", 1.00, 1.00, 1.00, 0.82, 5.0, 61.31)
NWMA = Mix("N-WMA", 1.10, 1.30, 0.70, 0.91, 5.0, 58.15)
WMA_POOR = Mix("WMA without nano benefit", 0.92, 1.00, 1.15, 0.78, 5.0, 58.15)


def moisture_factor(t, TSR, tau):
    """s(t) = 1 - (1 - TSR)(1 - exp(-t / tau)); the modulus multiplier is m(t) = R_E * s(t)."""
    return 1.0 - (1.0 - TSR) * (1.0 - np.exp(-np.asarray(t) / tau))


class ResponseDB:
    """FE responses for 12 monthly temperature states x 11 modulus multipliers (132 solutions)."""

    def __init__(self, model=None, m_grid=M_GRID):
        self.T, self.E = monthly_states()
        self.model = model or make_model(1)
        self.m_grid = np.asarray(m_grid)
        self.data = {k: np.zeros((12, len(self.m_grid))) for k in RESP_KEYS}
        for i in range(12):
            for j, m in enumerate(self.m_grid):
                r = dual_wheel_response(self.model.solve(layer_moduli(self.E[i] * m)))
                for k in RESP_KEYS:
                    self.data[k][i, j] = r[k]
        self.n_solutions = 12 * len(self.m_grid)

    def get(self, key, month, m):
        """Log-log interpolation in m for one month index (vectorised over m)."""
        y = self.data[key][month]
        lm = np.log(np.clip(m, self.m_grid[0], self.m_grid[-1]))
        if np.all(y > 0):
            return np.exp(np.interp(lm, np.log(self.m_grid), np.log(y)))
        return np.interp(lm, np.log(self.m_grid), y)


# --- transfer functions ---------------------------------------------------------
def k1_prime(h_ac_in):
    return 1.0 / (0.000398 + 0.003602 / (1.0 + np.exp(11.02 - 3.49 * h_ac_in)))


def mepdg_fatigue(eps_t, e_mpa, beta_f, Va=5.0, Vbeff=11.0, h_ac_mm=H_AC):
    M = 4.84 * (Vbeff / (Va + Vbeff) - 0.69)
    C = 10.0 ** M
    return (0.00432 * k1_prime(h_ac_mm / 25.4) * beta_f * C
            * (1.0 / eps_t) ** 3.9492 * (1.0 / (e_mpa * MPA_TO_PSI)) ** 1.281)


def ai_fatigue(eps_t, e_mpa, beta_f):
    """Asphalt Institute (MS-1) fatigue criterion, E in psi (Huang 2004)."""
    return beta_f * 0.0796 * eps_t ** -3.291 * (e_mpa * MPA_TO_PSI) ** -0.854


def ai_subgrade(eps_c):
    return 1.365e-9 * eps_c ** -4.477


def kz_factor(depth_mm, h_ac_mm=H_AC):
    H, d = h_ac_mm / 25.4, depth_mm / 25.4
    C1 = -0.1039 * H ** 2 + 2.4868 * H - 17.342
    C2 = 0.0172 * H ** 2 - 1.7331 * H + 27.428
    return (C1 + C2 * d) * 0.328196 ** d


RUT_K = 10.0 ** -3.35412
RUT_T, RUT_N = 1.5606, 0.4791


@dataclass(frozen=True)
class Scenario:
    n0: float                      # BZZ-100 axles in year 1 (design lane)
    growth: float = 0.04
    fatigue_model: str = "mepdg"   # or "ai"
    moisture_on_fatigue: bool = False
    beta_r_local: float = 1.0
    years: int = 80
    rut_limit_mm: float = 10.0


def run_life(db: ResponseDB, mix: Mix, sc: Scenario, keep_history=False):
    """Month-by-month damage. Returns lives (years) for fatigue, rutting and subgrade."""
    years = sc.years
    t_mid = (np.arange(years * 12) + 0.5) / 12.0
    month = np.tile(np.arange(12), years)
    yr = np.repeat(np.arange(years), 12)
    s = moisture_factor(t_mid, mix.TSR, mix.tau)
    m = mix.R_E * s
    n = sc.n0 * (1.0 + sc.growth) ** yr / 12.0
    eps_t = np.empty_like(t_mid); e_low = np.empty_like(t_mid); eps_c = np.empty_like(t_mid)
    eps_v = np.empty((3, len(t_mid)))
    for i in range(12):
        idx = month == i
        eps_t[idx] = db.get("eps_t", i, m[idx])
        eps_c[idx] = db.get("eps_c", i, m[idx])
        e_low[idx] = db.E[i, 2] * m[idx]
        for k in range(3):
            eps_v[k, idx] = np.maximum(db.get(f"eps_v{k+1}", i, m[idx]), 0.0)
    bf = mix.beta_f * (s if sc.moisture_on_fatigue else 1.0)
    if sc.fatigue_model == "mepdg":
        Nf = mepdg_fatigue(eps_t, e_low, bf)
    else:
        Nf = ai_fatigue(eps_t, e_low, bf)
    Df = np.cumsum(n / Nf)
    Dd = np.cumsum(n / ai_subgrade(eps_c))
    # rutting: strain hardening, sublayer by sublayer
    T_f = c_to_f(db.T[month])                          # (N, 3)
    ep = np.zeros(3)
    rut = np.empty_like(t_mid)
    kz = kz_factor(AC_SUB_MID)
    for q in range(len(t_mid)):
        a = kz * mix.beta_r * sc.beta_r_local * RUT_K * T_f[q] ** RUT_T
        er = eps_v[:, q]
        with np.errstate(divide="ignore", invalid="ignore"):
            base = er * a
            neq = np.where(base > 0, (ep / np.where(base > 0, base, 1.0)) ** (1.0 / RUT_N), 0.0)
            ep = np.where(base > 0, base * (neq + n[q]) ** RUT_N, ep)
        rut[q] = np.sum(ep * 40.0)
    t_end = np.arange(1, years * 12 + 1) / 12.0

    def cross(series, level):
        k = np.argmax(series >= level)
        if series[k] < level:
            return np.nan
        prev = series[k - 1] if k > 0 else 0.0
        t0 = t_end[k - 1] if k > 0 else 0.0
        return t0 + (level - prev) / (series[k] - prev) / 12.0

    Lf, Lr, Ld = cross(Df, 1.0), cross(rut, sc.rut_limit_mm), cross(Dd, 1.0)
    out = dict(L_fatigue=Lf, L_rut=Lr, L_subgrade=Ld,
               L=np.nanmin([Lf, Lr, Ld]))
    if keep_history:
        out.update(t=t_end, Df=Df, rut=rut, Dd=Dd, m=m, eps_t=eps_t)
    return out


def at_year(hist, key, year):
    return float(np.interp(year, hist["t"], hist[key]))


def anchor_scenario(db, growth=0.04, design_years=15.0):
    """Traffic n0 so HMA fatigue life = design_years; then local rutting factor so HMA rut = 10 mm then."""
    sc = Scenario(n0=1.0, growth=growth)
    h = run_life(db, HMA, sc, keep_history=True)
    n0 = 1.0 / at_year(h, "Df", design_years)
    sc = replace(sc, n0=n0)
    h = run_life(db, HMA, sc, keep_history=True)
    br = 10.0 / at_year(h, "rut", design_years)
    return replace(sc, beta_r_local=br)
