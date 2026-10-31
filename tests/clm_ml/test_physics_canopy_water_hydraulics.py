"""clm_ml.physics.canopy_water, plant_hydraulics, leaf_heat_capacity and their legacy adapters
(Phase 7.2, process 5).

Each adapter runs on fake containers holding only the fields it declares
(``DECLARED`` in ``clm_ml.legacy.canopy_water`` and ``clm_ml.legacy.plant_hydraulics``):
reading an undeclared field fails, and exactly the declared writes must change. The physics
is checked against what it must conserve.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import canopy_water as water_adapters  # noqa: E402
from clm_ml.legacy import plant_hydraulics as hydraulics_adapters  # noqa: E402
from clm_ml.physics import canopy_water, plant_hydraulics  # noqa: E402
from clm_ml.physics.constants import dewmx, maximum_leaf_wetted_fraction, mmh2o  # noqa: E402

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 unused)
LEAFY = slice(2, 4)  # legacy slots with plant area
N_SOI = 4  # soil levels; legacy arrays have N_SOI + 1 entries
DTIME = 300.0
Opts = namedtuple("Opts", "dtime_ml")
Params = namedtuple(
    "Params",
    "plant_conductance plant_capacitance root_radius root_density root_resistivity sla_top",
)
PARAMS = Params(4.0, 2500.0, 0.29e-3, 0.31e6, 14.0, 0.03)


@pytest.fixture(autouse=True)
def _module_tables(monkeypatch):
    """The soil layer count and the patch -> column map the soil adapter reads."""
    from clm_src_main import clm_varpar

    monkeypatch.setattr(clm_varpar, "nlevsoi", N_SOI)
    monkeypatch.setattr(hydraulics_adapters, "patch", namedtuple("patch", "column")(jnp.array([0])))


def _value(field, rng):
    n = N_SLOTS + 1
    if field in ("mlcanopy.dpai_profile", "mlcanopy.dlai_profile"):
        v = np.zeros(n)
        v[LEAFY] = [1.2, 0.9] if field == "mlcanopy.dpai_profile" else [0.8, 0.6]
        return v[None]
    special = {
        "mlcanopy.fracsun_profile": rng.uniform(0.2, 0.8, (1, n)),
        "mlcanopy.h2ocan_profile": rng.uniform(0.0, 0.2, (1, n)),
        "mlcanopy.h2ocan_bef_profile": rng.uniform(0.0, 0.2, (1, n)),
        "mlcanopy.evleaf_leaf": rng.uniform(-1e-4, 1e-4, (1, n, 3)),
        "mlcanopy.trleaf_leaf": rng.uniform(0.0, 2e-3, (1, n, 3)),
        "mlcanopy.lwp_bef_leaf": rng.uniform(-1.5, -0.5, (1, n, 3)),
        "mlcanopy.lwp_leaf": rng.uniform(-1.0, -0.5, (1, n, 3)),
        "mlcanopy.zs_profile": np.arange(n, dtype=float)[None],
        "mlcanopy.qflx_rain_forcing": np.array([2e-3]),
        "mlcanopy.qflx_snow_forcing": np.array([5e-4]),
        "mlcanopy.lai_canopy": np.array([1.4]),
        "mlcanopy.sai_canopy": np.array([0.7]),
        "mlcanopy.rsoil_soil": np.array([0.5]),
        "mlcanopy.psis_soil": np.array([-0.3]),
        "mlcanopy.root_biomass_canopy": np.array([500.0]),
        "mlcanopy.soil_et_loss_soil": rng.uniform(-1.0, -0.5, (1, N_SOI + 1)),
        "site.dz_col": np.full((1, N_SOI + 1), 0.2),
        "site.nbedrock_col": np.array([3]),
        "soilstate.smp_l_col": rng.uniform(-3e4, -1e3, (1, N_SOI + 1)),
        "soilstate.hk_l_col": rng.uniform(1e-6, 1e-4, (1, N_SOI + 1)),
        "soilstate.rootfr_patch": np.full((1, N_SOI + 1), 0.25),
        "waterstatebulk.h2osoi_ice_col": np.zeros((1, N_SOI + 1)),
    }
    container = field.split(".")[1]
    if field in special:
        return special[field]
    return rng.uniform(-1.0, -0.5, (1, n) if container.endswith("_profile") else (1,))


def _fakes(name, rng):
    """Fake containers holding the fields adapter ``name`` declares."""
    declared = {**water_adapters.DECLARED, **hydraulics_adapters.DECLARED}[name]
    by_container = {}
    for key in (*declared[0], *declared[1]):
        container, field = key.split(".")
        by_container.setdefault(container, {})[field] = jnp.asarray(_value(key, rng))
    return {
        c: namedtuple(c, sorted(fs))(**{f: fs[f] for f in sorted(fs)})
        for c, fs in by_container.items()
    }


def _call(name, k):
    w, h, ml = water_adapters, hydraulics_adapters, k["mlcanopy"]
    opts = Opts(DTIME)
    calls = {
        "CanopyWettedFraction": lambda: w.CanopyWettedFraction(1, [0], ml),
        "CanopyInterception": lambda: w.CanopyInterception(1, [0], ml, opts=opts),
        "CanopyEvaporation": lambda: w.CanopyEvaporation(1, [0], ml, opts=opts),
        "PlantResistance": lambda: h.PlantResistance(1, [0], ml, PARAMS),
        "SoilResistance": lambda: h.SoilResistance(
            1, [0], k["soilstate"], k["waterstatebulk"], ml, PARAMS, site=k["site"]
        ),
        "LeafWaterPotential": lambda: h.LeafWaterPotential(1, [0], ml, PARAMS, opts=opts),
        "LeafHeatCapacity": lambda: h.LeafHeatCapacity(1, [0], ml, PARAMS),
    }
    return {"mlcanopy": calls[name]()}


ADAPTERS = (*water_adapters.DECLARED, *hydraulics_adapters.DECLARED)


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_reads_and_writes_what_it_declares(name):
    before = _fakes(name, np.random.default_rng(11))
    after = _call(name, before)
    declared = {**water_adapters.DECLARED, **hydraulics_adapters.DECLARED}[name]
    changed = set()
    for container, inst in after.items():
        for field in inst._fields:
            old = np.asarray(getattr(before[container], field))
            new = np.asarray(getattr(inst, field))
            assert old.shape == new.shape, field
            assert np.all(np.isfinite(new)), field
            if not np.array_equal(old, new):
                changed.add(f"{container}.{field}")
    assert changed == set(declared[1])


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
