"""Regression tests: the headline numbers of the paper are reproduced by the code."""
import pytest

from pavecarbon.damage import HMA, NWMA, ResponseDB, anchor_scenario, run_life
from pavecarbon.lca import reduction


@pytest.fixture(scope="module")
def setup():
    db = ResponseDB()
    return db, anchor_scenario(db)


def test_anchoring(setup):
    db, sc = setup
    assert sc.n0 == pytest.approx(1.684e6, rel=2e-3)
    assert sc.beta_r_local == pytest.approx(0.235, abs=1e-3)
    r = run_life(db, HMA, sc)
    assert r["L_fatigue"] == pytest.approx(15.0, abs=1e-6)
    assert r["L_rut"] == pytest.approx(15.0, abs=1e-6)


def test_headline_results(setup):
    db, sc = setup
    h, n = run_life(db, HMA, sc), run_life(db, NWMA, sc)
    assert n["L"] == pytest.approx(19.35, abs=0.01)
    assert n["L_rut"] == pytest.approx(32.5, abs=0.05)
    assert 100 * reduction(HMA.gwp, h["L"], NWMA.gwp, n["L"]) == pytest.approx(26.5, abs=0.05)
