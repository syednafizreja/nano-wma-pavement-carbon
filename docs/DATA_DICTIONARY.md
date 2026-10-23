# Data dictionary (`results/`)

Units: strain in strain (×10⁶ = µε) unless the column name ends in `_ue`; stress MPa; deflection mm; life years; GWP kg CO₂e/t or t CO₂e/lane-km.

| File | Columns |
| --- | --- |
| `verification_boussinesq.csv` | `case` (mesh/domain), `quantity`, `closed_form`, `fe`, `error_pct` |
| `monthly_states.csv` | `month`, `air_C`, `T20`/`T60`/`T100` (°C at sublayer mid-depths), `E20`/`E60`/`E100` (HMA modulus, MPa) |
| `responses_table4.csv` | `case`, `year`, `month`, `multiplier` (m), `E_lower_MPa`, `eps_t_ue`, `sig_t_MPa`, `eps_c_ue`, `eps_v3_ue`, `defl_mm` |
| `monthly_eps_t.csv` | `year` (0 or 15), `month`, `HMA_ue`, `NWMA_ue`, `reduction_pct` |
| `damage_history.csv` | `t` (years, month ends), `Df_HMA`, `Df_NWMA` (fatigue damage), `rut_HMA`, `rut_NWMA` (mm) |
| `decomposition.csv` | `step`, `L_fatigue`, `increment`, `L_rut` |
| `sensitivity_oat.csv` | `case`, `L_HMA`, `L_NWMA`, `life_ext_pct`, `dG_pct`, `gov_HMA`, `gov_NWMA` (governing criterion) |
| `monte_carlo.csv` | sampled `R_E`, `beta_f`, `beta_r`, `TSR_N`, `TSR_H`, `tau`, `gwp_H`, `gwp_N`; outputs `L_HMA`, `L_NWMA`, `life_ext_pct`, `dG_pct` |
| `results.json` | mesh size, layered checks, anchored scenario, service lives, carbon results, Monte Carlo statistics and Spearman ρ, thermal check, package versions |
