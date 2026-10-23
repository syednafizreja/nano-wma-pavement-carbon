#!/usr/bin/env python3
"""
Reproduce every table, figure and number in the paper.

    python run_all.py                 # full run, writes results/ (about 1-2 minutes)
    python run_all.py --selftest      # quick verification of the FE solver
"""
import argparse
import sys

from pavecarbon.analysis import run_all
from pavecarbon.plots import make_all


def selftest():
    from pavecarbon.fe_axisym import AxisymModel, boussinesq_sigma_z_axis, boussinesq_w0
    from pavecarbon.pavement import make_model
    m0 = make_model(1)
    m = AxisymModel(m0.mesh, {L.name: 0.35 for L in m0.mesh.layers})
    s = m.solve([100.0] * 6)
    for z in (50.0, 106.5, 200.0, 400.0):
        fe, cf = -s.strain_stress(0.0, z)[1][1], boussinesq_sigma_z_axis(0.70, 106.5, z)
        assert abs(fe / cf - 1) < 0.01, (z, fe, cf)
    w, w0 = s.surface_uz(0.0), boussinesq_w0(0.70, 106.5, 100.0, 0.35)
    assert abs(w / w0 - 1) < 0.02, (w, w0)
    print("selftest passed: FE matches the Boussinesq half-space within 1% (stress) and 2% (deflection)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    R = run_all(a.outdir)
    make_all(R, a.outdir)
    print(f"done: results written to {a.outdir}/")
