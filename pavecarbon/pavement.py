"""
pavement.py -- reference structure, mesh, materials, climate and the BZZ-100
dual-wheel response evaluation.
"""
from __future__ import annotations

import numpy as np

from .fe_axisym import Layer, Mesh, AxisymModel, geometric_edges

MPA_TO_PSI = 145.0377

# --- BZZ-100 standard axle (JTG D50-2017) ---------------------------------
P_TYRE = 0.70            # MPa
A_LOAD = 106.5           # mm, equivalent contact radius
WHEEL_SPACING = 319.5    # mm, centre-to-centre of the dual wheels

# --- reference structure -----------------------------------------------------
H_AC = 120.0                            # mm, asphalt layer (three 40 mm sublayers)
AC_SUB_MID = np.array([20.0, 60.0, 100.0])
LAYER_DEF = [  # name, top, bottom, nu, E (None = asphalt, set per state)
    ("AC1", 0.0, 40.0, 0.35, None),
    ("AC2", 40.0, 80.0, 0.35, None),
    ("AC3", 80.0, 120.0, 0.35, None),
    ("base", 120.0, 470.0, 0.35, 400.0),
    ("subbase", 470.0, 770.0, 0.35, 250.0),
    ("subgrade", 770.0, 6000.0, 0.40, 100.0),
]
Z_SUBGRADE = 770.0

# --- Chengdu (Shuangliu) monthly mean air temperature, 1991-2020 normals (CMA) ---
CHENGDU_AIR_C = np.array([6.0, 8.5, 12.6, 17.6, 21.6, 24.2, 25.8, 25.5, 21.9, 17.5, 12.8, 7.5])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- Witczak (NCHRP 1-37A) mixture inputs, AC-20-type dense mixture, PG 64-22 ---
WITCZAK = dict(p200=5.0, p4=48.0, p38=28.0, p34=5.0, Va=5.0, Vbeff=11.0,
               A=10.980, VTS=-3.680, f=10.0)


def c_to_f(t):
    return np.asarray(t) * 9.0 / 5.0 + 32.0


def f_to_c(t):
    return (np.asarray(t) - 32.0) * 5.0 / 9.0


def pavement_temp_c(t_air_c, depth_mm):
    """Witczak mean pavement temperature at depth z (Huang 2004): Tp = Ta(1+1/(z+4)) - 34/(z+4) + 6 (degF, z in inches)."""
    z = np.asarray(depth_mm) / 25.4
    ta = c_to_f(t_air_c)
    return f_to_c(ta * (1.0 + 1.0 / (z + 4.0)) - 34.0 / (z + 4.0) + 6.0)


def witczak_modulus_mpa(t_c, w=WITCZAK):
    """Dynamic modulus |E*| (MPa) from the Witczak predictive equation (NCHRP 1-37A)."""
    t_r = c_to_f(t_c) + 459.67
    log_eta_cp = 10.0 ** (w["A"] + w["VTS"] * np.log10(t_r))
    eta = 10.0 ** log_eta_cp / 1.0e8                  # cP -> 10^6 poise
    p200, p4, p38, p34 = w["p200"], w["p4"], w["p38"], w["p34"]
    Va, Vb = w["Va"], w["Vbeff"]
    lhs = (3.750063 + 0.02932 * p200 - 0.001767 * p200 ** 2 - 0.002841 * p4
           - 0.058097 * Va - 0.802208 * Vb / (Vb + Va))
    num = 3.871977 - 0.0021 * p4 + 0.003958 * p38 - 0.000017 * p38 ** 2 + 0.005470 * p34
    den = 1.0 + np.exp(-0.603313 - 0.313351 * np.log10(w["f"]) - 0.393532 * np.log10(eta))
    e_psi = 10.0 ** (lhs + num / den)
    return e_psi / MPA_TO_PSI


def monthly_states():
    """Sublayer temperatures (C) and HMA moduli (MPa), shape (12, 3)."""
    T = np.array([[pavement_temp_c(ta, d) for d in AC_SUB_MID] for ta in CHENGDU_AIR_C])
    E = witczak_modulus_mpa(T)
    return T, E


def build_mesh(refine=1, R=4000.0, D=6000.0):
    r_edges = np.concatenate([
        np.linspace(0.0, A_LOAD, 8)[:-1],                       # 7 elements under the load
        np.linspace(A_LOAD, 450.0, 22)[:-1],                     # 21 elements to 450 mm
        geometric_edges(450.0, R, 12, 20.0),                      # 12 graded elements
    ])
    z_edges = np.concatenate([
        np.linspace(0.0, H_AC, 13)[:-1],                         # 12 x 10 mm in the asphalt
        geometric_edges(120.0, 470.0, 8, 15.0)[:-1],              # base
        geometric_edges(470.0, 770.0, 5, 45.0)[:-1],              # subbase
        geometric_edges(770.0, D, 7, 70.0),                       # subgrade
    ])
    layers = [Layer(n, t, b, nu) for n, t, b, nu, _ in LAYER_DEF]
    layers[-1] = Layer("subgrade", 770.0, D, 0.40)
    mesh = Mesh(r_edges, z_edges, layers)
    return mesh.refined(refine) if refine > 1 else mesh


def layer_moduli(e_ac3):
    """Full list of layer moduli from three asphalt sublayer moduli."""
    return list(e_ac3) + [L[4] for L in LAYER_DEF[3:]]


DUAL_X = np.linspace(0.0, WHEEL_SPACING / 2.0, 17)   # evaluation points, wheel centre -> midpoint


def dual_wheel_response(sol):
    """Critical responses under the BZZ-100 dual wheel by superposition of one axisymmetric solution.

    Points lie on the line joining the wheel centres; there the radial direction of each
    wheel coincides with that line, so x-strains add from eps_r and y-strains from eps_theta.
    """
    out = {}
    eps_t, sig_t, x_t = -np.inf, np.nan, np.nan
    eps_c, eps_v, defl = -np.inf, np.full(3, -np.inf), -np.inf
    for x in DUAL_X:
        r1, r2 = x, WHEEL_SPACING - x
        e1, s1 = sol.strain_stress(r1, H_AC, side="above")
        e2, s2 = sol.strain_stress(r2, H_AC, side="above")
        exx, eyy = e1[0] + e2[0], e1[2] + e2[2]
        sxx, syy = s1[0] + s2[0], s1[2] + s2[2]
        if max(exx, eyy) > eps_t:
            eps_t = max(exx, eyy)
            sig_t = sxx if exx >= eyy else syy
            x_t = x
        c1, _ = sol.strain_stress(r1, Z_SUBGRADE, side="below")
        c2, _ = sol.strain_stress(r2, Z_SUBGRADE, side="below")
        eps_c = max(eps_c, -(c1[1] + c2[1]))
        for k, zm in enumerate(AC_SUB_MID):
            v1, _ = sol.strain_stress(r1, zm)
            v2, _ = sol.strain_stress(r2, zm)
            eps_v[k] = max(eps_v[k], -(v1[1] + v2[1]))
        defl = max(defl, sol.surface_uz(r1) + sol.surface_uz(r2))
    out.update(eps_t=eps_t, sig_t=sig_t, x_t=x_t, eps_c=eps_c,
               eps_v1=eps_v[0], eps_v2=eps_v[1], eps_v3=eps_v[2], defl=defl)
    return out


def single_wheel_eps_t(sol):
    """Maximum bottom-of-asphalt horizontal strain under one wheel (for comparison)."""
    best = -np.inf
    for r in np.linspace(0.0, 300.0, 61):
        e, _ = sol.strain_stress(r, H_AC, side="above")
        best = max(best, e[0], e[2])
    return best


def make_model(refine=1, nu_ac=None, R=4000.0, D=6000.0):
    mesh = build_mesh(refine, R, D)
    over = None if nu_ac is None else {"AC1": nu_ac, "AC2": nu_ac, "AC3": nu_ac}
    return AxisymModel(mesh, over)
