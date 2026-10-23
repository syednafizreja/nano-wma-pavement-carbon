# Model reference

The full derivation and justification are in the paper. This page lists the equations exactly as implemented, with the module that implements each one.

## 1. Asphalt modulus — `pavecarbon/pavement.py`

Witczak predictive equation (NCHRP 1-37A), |E*| in psi, f = 10 Hz:

```
log10|E*| = 3.750063 + 0.02932 ρ200 − 0.001767 ρ200² − 0.002841 ρ4 − 0.058097 Va − 0.802208 Vbeff/(Vbeff + Va)
          + (3.871977 − 0.0021 ρ4 + 0.003958 ρ38 − 0.000017 ρ38² + 0.005470 ρ34)
            / (1 + exp(−0.603313 − 0.313351 log10 f − 0.393532 log10 η))
log log η(cP) = A + VTS log T_R        (PG 64-22: A = 10.980, VTS = −3.680; T_R in °Rankine)
```

Mixture: Va = 5 %, Vbeff = 11 %, ρ34 = 5, ρ38 = 28, ρ4 = 48, ρ200 = 5. Result at 21 °C: 6,189 MPa.

Pavement temperature at mid-depth z (inches), Witczak relation as given by Huang (2004):

```
Tp = Ta (1 + 1/(z + 4)) − 34/(z + 4) + 6        (°F)
```

## 2. Finite elements — `pavecarbon/fe_axisym.py`

Axisymmetric linear elasticity, 9-node Lagrange quadrilaterals, 3 × 3 Gauss points.
Strain vector [ε_r, ε_z, ε_θ, γ_rz] with ε_θ = u_r / r. Element stiffness

```
K_e = ∫∫ Bᵀ D B 2π r dr dz
```

Boundary conditions: u_r = 0 on the axis and the outer radius (4 m); u_r = u_z = 0 at the base (6 m).
Load: uniform 0.70 MPa on a = 106.5 mm, consistent nodal forces (total 24.94 kN per wheel).
Dual wheel (319.5 mm spacing): superposition of one axisymmetric solution along the line joining the wheel centres.

## 3. Moisture, damage and life — `pavecarbon/damage.py`

Moisture-degradation law (asphalt modulus multiplier):

```
m(t) = R_E [1 − (1 − TSR)(1 − exp(−t/τ))]
```

MEPDG fatigue (E in psi, h_ac = 4.72 in):

```
N_f = 0.00432 k1' β_f C (1/ε_t)^3.9492 (1/E)^1.281,   C = 10^M,   M = 4.84 (Vbeff/(Va + Vbeff) − 0.69)
k1' = 1 / (0.000398 + 0.003602 / (1 + exp(11.02 − 3.49 h_ac)))
```

Rutting (NCHRP 1-40D global coefficients, strain hardening, three 40 mm sublayers):

```
ε_p / ε_r = k_z β_r β_r,local 10^−3.35412 T^1.5606 N^0.4791,   k_z = (C1 + C2 d) 0.328196^d
```

Subgrade (Asphalt Institute): `N_d = 1.365e−9 ε_c^−4.477`. Miner's rule on monthly increments; life = first time D = 1 (fatigue, subgrade) or rut = 10 mm.

Anchoring: initial traffic n0 is set so that HMA fatigue life is exactly 15.0 years (n0 = 1.684 × 10⁶ BZZ-100 axles/yr, 4 %/yr growth); β_r,local = 0.235 is set so that HMA rut depth is 10.0 mm at 15 years.

## 4. Carbon — `pavecarbon/lca.py`

```
G_ann = g M / L,     ΔG = 1 − (g_N / g_H)(L_H / L_N),     M = 3.75 m × 1 km × 0.12 m × 2.40 t/m³ = 1,080 t
```
