"""clm_ml.processes.canopy_water and plant_hydraulics (Phase 7.2, process 5), checked against
what they must conserve."""

from collections import namedtuple

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import canopy_water, plant_hydraulics  # noqa: E402
from clm_ml.processes.constants import dewmx, maximum_leaf_wetted_fraction, mmh2o  # noqa: E402

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 unused)
N_SOI = 4  # soil levels; legacy arrays have N_SOI + 1 entries
DTIME = 300.0
Params = namedtuple(
    "Params",
    "plant_conductance plant_capacitance root_radius root_density root_resistivity sla_top",
)
PARAMS = Params(4.0, 2500.0, 0.29e-3, 0.31e6, 14.0, 0.03)


def _dpai():
    v = np.zeros(N_SLOTS)
    v[1:3] = [1.2, 0.9]
    return jnp.asarray(v)


def test_interception_conserves_water():
    dpai = _dpai()
    bef = jnp.where(dpai > 0, 0.1, 0.0)
    r = canopy_water.interception(5e-3, 1e-3, 1.4, 0.7, dpai, bef, DTIME)
    drip = r.qflx_tflrain + r.qflx_tflsnow - (6e-3 - r.qflx_intr)  # beyond direct throughfall
    stored = float(jnp.sum(r.h2ocan - bef))
    np.testing.assert_allclose(stored + float(drip) * DTIME, float(r.qflx_intr) * DTIME, rtol=1e-12)
    assert np.all(np.asarray(r.h2ocan) <= np.asarray(dewmx * dpai) + 1e-15)  # capacity
    assert np.all(np.asarray(r.h2ocan)[np.asarray(dpai) == 0] == 0.0)


def test_evaporation_removes_what_the_leaves_evaporate():
    dpai, fracsun = _dpai(), jnp.full(N_SLOTS, 0.3)
    h2ocan = jnp.where(dpai > 0, 0.1, 0.05)
    ev = jnp.full((N_SLOTS, 2), 1e-5)
    new = canopy_water.evaporation(h2ocan, dpai, fracsun, ev, jnp.zeros((N_SLOTS, 2)), DTIME)
    expected = h2ocan - jnp.where(dpai > 0, 1e-5 * dpai * mmh2o * DTIME, 0.0)
    np.testing.assert_allclose(new, expected, rtol=1e-12)


def test_wetted_fraction_limits():
    dpai = _dpai()
    fwet, fdry = canopy_water.wetted_fraction(dewmx * dpai * 10.0, dpai, 0.5 * dpai)
    assert np.all(np.asarray(fwet)[np.asarray(dpai) > 0] == maximum_leaf_wetted_fraction)
    assert np.all(np.asarray(fdry)[np.asarray(dpai) == 0] == 0.0)


def test_soil_uptake_fractions_sum_to_one_above_bedrock():
    rng = np.random.default_rng(2)
    r = plant_hydraulics.soil_resistance(
        jnp.asarray(rng.uniform(1e-6, 1e-4, N_SOI)), jnp.asarray(rng.uniform(-3e4, -1e3, N_SOI)),
        jnp.full(N_SOI, 0.2), jnp.full(N_SOI, 0.25), jnp.zeros(N_SOI), 3, 1.4, 500.0,
        *PARAMS[2:5],
    )  # fmt: skip
    np.testing.assert_allclose(float(jnp.sum(r.soil_et_loss)), 1.0, rtol=1e-12)
    assert float(r.soil_et_loss[3]) == 0.0  # below bedrock


def test_leaf_water_potential_relaxes_to_steady_state():
    dpai = _dpai()
    lsc = plant_hydraulics.plant_resistance(dpai, 0.5, 4.0)
    tr = jnp.full((N_SLOTS, 2), 1e-3)
    zs = jnp.arange(N_SLOTS, dtype=float)
    lwp = plant_hydraulics.leaf_water_potential(
        dpai, zs, lsc, tr, jnp.full((N_SLOTS, 2), -0.5), -0.3, 2500.0, 1e9
    )
    head = plant_hydraulics.HEAD
    steady = -0.3 - head * zs[:, None] - 1000.0 * tr / jnp.maximum(lsc, 1e-30)[:, None]
    np.testing.assert_allclose(np.asarray(lwp)[1:3], np.asarray(steady)[1:3], rtol=1e-12)
    assert np.all(np.asarray(lwp)[np.asarray(dpai) == 0] == 0.0)
