import numpy as np
import pytest

from pavecarbon.fe_axisym import AxisymModel, boussinesq_sigma_z_axis, boussinesq_w0
from pavecarbon.pavement import make_model, witczak_modulus_mpa, pavement_temp_c
from pavecarbon.lca import MASS_T_PER_LANE_KM, reduction


@pytest.fixture(scope="module")
def halfspace():
    m0 = make_model(1)
    m = AxisymModel(m0.mesh, {L.name: 0.35 for L in m0.mesh.layers})
    return m.solve([100.0] * 6)


@pytest.mark.parametrize("z", [50.0, 106.5, 200.0, 400.0])
def test_boussinesq_axis_stress(halfspace, z):
    fe = -halfspace.strain_stress(0.0, z)[1][1]
    assert fe == pytest.approx(boussinesq_sigma_z_axis(0.70, 106.5, z), rel=0.01)


def test_surface_deflection(halfspace):
    assert halfspace.surface_uz(0.0) == pytest.approx(boussinesq_w0(0.70, 106.5, 100.0, 0.35), rel=0.02)


def test_total_load_equals_wheel_load():
    m = make_model(1)
    f = m.load_vector(0.70, 106.5)
    assert f.sum() == pytest.approx(0.70 * np.pi * 106.5 ** 2, rel=1e-10)


def test_witczak_reference_modulus():
    assert witczak_modulus_mpa(21.0) == pytest.approx(6189, rel=0.002)


def test_pavement_temperature_january_top():
    assert pavement_temp_c(6.0, 20.0) == pytest.approx(10.4, abs=0.05)


def test_lca_bookkeeping():
    assert MASS_T_PER_LANE_KM == pytest.approx(1080.0)
    assert reduction(61.31, 15.0, 58.15, 15.0) == pytest.approx(1 - 58.15 / 61.31)
