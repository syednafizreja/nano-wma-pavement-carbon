"""
lca.py -- annualised A1-A3 carbon of the asphalt layer.

Functional unit: one lane-km (3.75 m wide) of the 120 mm asphalt layer,
per year of service. Production GWP per tonne comes from the author's earlier
screening LCA (61.31 and 58.15 kg CO2e/t).
"""
from __future__ import annotations

import math

LANE_WIDTH_M = 3.75
THICKNESS_M = 0.12
DENSITY_T_M3 = 2.40
MASS_T_PER_LANE_KM = LANE_WIDTH_M * 1000.0 * THICKNESS_M * DENSITY_T_M3   # 1,080 t


def embodied_t(gwp_kg_per_t, mass_t=MASS_T_PER_LANE_KM):
    return gwp_kg_per_t * mass_t / 1000.0


def annualised_t_per_year(gwp_kg_per_t, life_years, mass_t=MASS_T_PER_LANE_KM):
    return embodied_t(gwp_kg_per_t, mass_t) / life_years


def reduction(g_h, L_h, g_n, L_n):
    """Delta G = 1 - (g_N / g_H)(L_H / L_N)."""
    return 1.0 - (g_n / g_h) * (L_h / L_n)


def break_even_gwp(g_h, L_h, L_n):
    return g_h * L_n / L_h


def horizon_total(gwp_kg_per_t, life, horizon=40.0, discrete=False):
    n = math.ceil(horizon / life) if discrete else horizon / life
    return n * embodied_t(gwp_kg_per_t), n
