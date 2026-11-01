"""clm_ml.physics.canopy and its legacy adapters (Phase 7.2, process 10).

Each adapter runs on fake containers holding only the fields it declares (``DECLARED`` in
``clm_ml.legacy.canopy``): reading an undeclared field fails, and exactly the declared writes
must change. The physics is checked against hand-computed canopy totals, the conservation of
plant area and the definition of the sub-step average and the sunlit/shaded merge.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import canopy as adapter  # noqa: E402
from clm_ml.physics import canopy  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 unused/ground)
N_CAN = 4  # layers in use (legacy slots 1..4; 0-based 0..3)
N_CHECKS = 13
_BAND_FIELDS = {"swskyb_forcing", "swskyd_forcing", "swsoi_soil", "swveg_canopy",
                "swvegsun_canopy", "swvegsha_canopy", "albcan_canopy"}  # fmt: skip
_LEVEL_BAND = {"swupw_profile", "swdwn_profile", "swbeam_profile"}
Opts = namedtuple("Opts", "flux_profile_type")


def _value(field, rng):
    container, name = field.split(".")
    n = N_SLOTS + 1
    special = {
        "ncan_canopy": np.array([N_CAN]),
        "ntop_canopy": np.array([3]),
        "nbot_canopy": np.array([1]),
        "balance_error": np.zeros((1, N_CHECKS)),
        "dz_profile": np.ones((1, n)),
        "zs_profile": np.arange(n, dtype=float)[None],
        "zw_profile": np.arange(n, dtype=float)[None],
        "dpai_profile": np.array([[0.0, 0.3, 0.5, 0.4, 0.0, 1e36, 1e36]]),
        "fracsun_profile": np.linspace(0.2, 0.7, n)[None],
        "fwet_profile": np.full((1, n), 0.1),
        "lai_canopy": np.array([0.9]),
        "sai_canopy": np.array([0.3]),
        "elai_patch": np.array([1.1]),
        "esai_patch": np.array([0.4]),
        "smp_l_col": np.full((1, 4), -1.0e4),
        "t_soisno_col": np.full((1, 4), 290.0),
        "tref_forcing": np.array([290.0]),
    }
    if name in special:
        return special[name]
    if container != "mlcanopy":
        return rng.uniform(1.0, 2.0, (1,))
    if name in _BAND_FIELDS:
        return rng.uniform(1.0, 2.0, (1, 3))
    if name in _LEVEL_BAND:
        return rng.uniform(1.0, 2.0, (1, n, 3))
    if name == "swleaf_leaf":
        return rng.uniform(1.0, 2.0, (1, n, 3, 3))
    if name.endswith("_leaf"):
        return rng.uniform(1.0, 2.0, (1, n, 3))
    if name.endswith("_profile"):
        return rng.uniform(0.1, 0.9, (1, n))
    return rng.uniform(1.0, 2.0, (1,))


def _fakes(declared, rng):
    fields = {}
    for key in (*declared[0], *declared[1]):
        container, name = key.split(".")
        fields.setdefault(container, {})[name] = jnp.asarray(_value(key, rng))
    return {c: namedtuple(c, sorted(f))(**{k: f[k] for k in sorted(f)}) for c, f in fields.items()}


def _means(seed=11):
    """Averages over the sub-steps (random values, every slot)."""
    return _random_fluxes(np.random.default_rng(seed))


CASES = {
    "PlantAreaProfiles": lambda c: adapter.PlantAreaProfiles(
        c["mlcanopy"], c["canopystate"], 0, 0.05
    ),
    "SoilSurfaceHumidity": lambda c: adapter.SoilSurfaceHumidity(
        c["mlcanopy"], c["soilstate"], c["temperature"], 0, 0
    ),
    "WriteSubstepMeans": lambda c: adapter.WriteSubstepMeans(c["mlcanopy"], 0, _means()),
    "CanopyFluxesDiagnostics": lambda c: adapter.CanopyFluxesDiagnostics(
        c["mlcanopy"], 0, _means(), opts=Opts(1)
    ),
    "CanopyFluxesDiagnostics0": lambda c: adapter.CanopyFluxesDiagnostics(
        c["mlcanopy"], 0, _means(), opts=Opts(0)
    ),
    "MergeSunShade": lambda c: adapter.MergeSunShade(c["mlcanopy"], 0),
}


@pytest.mark.parametrize("name", CASES)
def test_adapter_reads_and_writes_what_it_declares(name):
    declared = adapter.DECLARED[name.rstrip("0")]
    before = _fakes(declared, np.random.default_rng(5))
    after = CASES[name](before)
    changed = set()
    for field in after._fields:
        old = np.asarray(getattr(before["mlcanopy"], field))
        new = np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        if not np.array_equal(old, new):
            changed.add("mlcanopy." + field)
    assert changed == set(declared[1])


def _substep_container(rng):
    """A fake container holding the fields of ``canopy.Substep`` in the legacy layout."""
    n, stages = N_SLOTS + 1, 5
    shapes = {
        "scalar": (1,), "band": (1, 3), "layer": (1, n), "level": (1, n),
        "level_band": (1, n, 3), "leaf": (1, n, 3), "leaf_band": (1, n, 3, 3),
        "stage": (1, stages), "layer_stage": (1, n, stages), "leaf_stage": (1, n, 3, stages),
        "checks": (1, N_CHECKS),
    }  # fmt: skip
    fields = {
        f: jnp.asarray(rng.uniform(1.0, 2.0, shapes[kind]))
        for kind, names in canopy.SUBSTEP_KINDS.items()
        for f in names
    }
    return namedtuple("mlcanopy", sorted(fields))(**{f: fields[f] for f in sorted(fields)})


def test_substep_carry_round_trip_keeps_index_0():
    ml = _substep_container(np.random.default_rng(3))
    c = adapter.substep_from_container(ml, 0)
    assert np.asarray(c.tleaf_leaf).shape == (N_SLOTS, 2)
    assert np.asarray(c.swleaf_leaf).shape == (N_SLOTS, 2, 2)
    assert np.asarray(c.swupw_profile).shape == (N_SLOTS + 1, 2)  # levels: ground included
    assert np.asarray(c.dtleaf_leaf).shape == (N_SLOTS, 2, 5)
    np.testing.assert_array_equal(np.asarray(c.tair_profile), np.asarray(ml.tair_profile[0, 1:]))
    back = adapter.container_with_substep(ml, 0, c)
    for f in ml._fields:
        np.testing.assert_array_equal(np.asarray(getattr(back, f)), np.asarray(getattr(ml, f)))
    # new values go to the 0-based slots; the unused index 0 keeps its value
    c2 = jax.tree.map(lambda x: x + 1.0, c)
    back = adapter.container_with_substep(ml, 0, c2)
    np.testing.assert_array_equal(
        np.asarray(back.tleaf_leaf[0, 0]), np.asarray(ml.tleaf_leaf[0, 0])
    )
    np.testing.assert_array_equal(np.asarray(back.tleaf_leaf[0, 1:, 0]),
                                  np.asarray(ml.tleaf_leaf[0, 1:, 0]))  # fmt: skip
    np.testing.assert_array_equal(np.asarray(back.tleaf_leaf[0, 1:, 1:]),
                                  np.asarray(ml.tleaf_leaf[0, 1:, 1:]) + 1.0)  # fmt: skip
    np.testing.assert_array_equal(np.asarray(back.lwupw_profile[0]),
                                  np.asarray(ml.lwupw_profile[0]) + 1.0)  # fmt: skip


def test_substep_options_refuse_unsupported_variants():
    from multilayer_canopy.MLclm_varctl import DEFAULT_OPTIONS

    o = adapter.substep_options(DEFAULT_OPTIONS._replace(met_type=3))
    assert o.integrator == "rk4" and o.interpolation == "three_point"
    with pytest.raises(NotImplementedError):
        adapter.substep_options(DEFAULT_OPTIONS._replace(met_type=3, light_type=1))
    with pytest.raises(NotImplementedError):
        adapter.substep_options(DEFAULT_OPTIONS._replace(met_type=3, flux_profile_type=0))
    with pytest.raises(ValueError):
        adapter.substep_options(DEFAULT_OPTIONS._replace(met_type=2))


def test_n_stages():
    assert canopy.n_stages("euler") == 1
    assert canopy.n_stages("rk4") == 5


def _layers(n_layers=N_CAN, n_within=3, first_leafy=0):
    return PatchLayers(
        n_layers=jnp.asarray(n_layers),
        n_within=jnp.asarray(n_within),
        first_leafy=jnp.asarray(first_leafy),
        dz=jnp.ones(N_SLOTS),
        zs=jnp.arange(N_SLOTS, dtype=float),
        zw=jnp.arange(N_SLOTS + 1, dtype=float),
    )


def test_plant_area_conserves_lai_and_sai_and_keeps_the_floor():
    lay = _layers()
    dlai_frac = jnp.array([0.001, 0.6, 0.399, 0.0, 9.0, 9.0])  # padding holds junk
    dsai_frac = jnp.array([0.0, 0.5, 0.5, 0.0, 9.0, 9.0])
    r = canopy.plant_area(lay, dlai_frac, dsai_frac, 2.0, 0.5, 0.05)
    np.testing.assert_allclose(float(r.dpai[0]), 0.05)  # thin leafy layer raised to the floor
    np.testing.assert_allclose(float(r.sai_floor), 0.05 - 0.002)
    np.testing.assert_allclose(float(r.sai), 0.5 + 0.05 - 0.002)
    assert abs(float(r.error)) < 1e-15
    assert float(r.dpai[3]) == 0.0  # top layer (above the canopy) leafless
    # layers at or above the floor unchanged
    np.testing.assert_array_equal(np.asarray(r.dlai[1:3]), np.asarray(dlai_frac[1:3]) * 2.0)


def test_soil_surface_humidity():
    assert float(canopy.soil_surface_humidity(0.0, 290.0)) == 1.0
    assert 0.99 < float(canopy.soil_surface_humidity(-1.0e4, 290.0)) < 1.0


def _random_fluxes(rng, scale=1.0):
    def a(*shape):
        return jnp.asarray(rng.uniform(0.5, 1.5, shape) * scale)

    shapes = {
        "swsoi": (2,), "swskyb": (2,), "swskyd": (2,), "shair": (N_SLOTS,),
        "etair": (N_SLOTS,), "stair": (N_SLOTS,), "mflx": (N_SLOTS,), "kc_eddy": (N_SLOTS,),
        "gac": (N_SLOTS,), "swupw": (N_SLOTS + 1, 2), "swdwn": (N_SLOTS + 1, 2),
        "swbeam": (N_SLOTS + 1, 2), "lwupw": (N_SLOTS + 1,), "lwdwn": (N_SLOTS + 1,),
        "swleaf": (N_SLOTS, 2, 2),
    }  # fmt: skip
    return canopy.SubstepFluxes(
        **{
            n: a(*shapes.get(n, (N_SLOTS, 2) if n in _LEAF_NAMES else ()))
            for n in canopy.SubstepFluxes._fields
        }
    )


_LEAF_NAMES = {"lwleaf", "rnleaf", "shleaf", "lhleaf", "trleaf", "evleaf", "stleaf", "anet",
               "agross", "gs"}  # fmt: skip


def test_average_over_substeps():
    rng = np.random.default_rng(1)
    steps = [_random_fluxes(rng) for _ in range(6)]
    sums = canopy.zero_sums(steps[0])
    for s in steps:
        sums = canopy.accumulate(sums, s)
    mean = canopy.average(sums, 6)
    for name in canopy.SubstepFluxes._fields:
        expect = np.mean([np.asarray(getattr(s, name)) for s in steps], axis=0)
        np.testing.assert_allclose(np.asarray(getattr(mean, name)), expect, rtol=1e-14)


def test_canopy_totals_by_hand():
    rng = np.random.default_rng(2)
    lay = _layers()
    m = _random_fluxes(rng)
    dpai = jnp.array([0.3, 0.5, 0.4, 0.0, 1e36, 1e36])
    fracsun = jnp.linspace(0.2, 0.7, N_SLOTS)
    leaves = canopy.CanopyLeaves(
        dpai=dpai,
        fracsun=fracsun,
        fwet=jnp.full(N_SLOTS, 0.2),
        fdry=jnp.full(N_SLOTS, 0.6),
        wind=jnp.linspace(1.0, 3.0, N_SLOTS),
        tair=jnp.linspace(290.0, 292.0, N_SLOTS),
        vcmax25=jnp.full(N_SLOTS, 50.0),
        tleaf=jnp.stack([jnp.full(N_SLOTS, 295.0), jnp.full(N_SLOTS, 291.0)], axis=1),
        lwp=jnp.stack([jnp.full(N_SLOTS, -2.5), jnp.full(N_SLOTS, -1.0)], axis=1),
        vcmax25_leaf=jnp.full((N_SLOTS, 2), 50.0),
    )
    t = canopy.canopy_totals(lay, m, leaves, 0.9, 0.3, flux_profile_type=1)
    d = np.array([0.3, 0.5, 0.4])
    fs = np.asarray(fracsun)[:3]
    sh = np.asarray(m.shleaf)[:3]
    np.testing.assert_allclose(float(t.shveg), np.sum((sh[:, 0] * fs + sh[:, 1] * (1 - fs)) * d))
    np.testing.assert_allclose(float(t.shvegsun), np.sum(sh[:, 0] * fs * d))
    np.testing.assert_allclose(float(t.laisun + t.laisha), d.sum())
    np.testing.assert_allclose(float(t.vcmax25veg), 50.0 * d.sum())
    np.testing.assert_allclose(float(t.tlvegsun), 295.0)
    np.testing.assert_allclose(float(t.tlvegsha), 291.0)
    # fracgreen = fdry / (1 - fwet) = 0.75
    ag = np.asarray(m.agross)[:3]
    np.testing.assert_allclose(float(t.gppvegsun), np.sum(ag[:, 0] * fs * d * 0.75))
    # the canopy's fluxes at the top layer in use, its albedo at the canopy top (ntop)
    assert float(t.shflx) == float(m.shair[N_CAN - 1])
    assert float(t.lhflx) == float(m.lhtop)
    inc = np.asarray(m.swskyb) + np.asarray(m.swskyd)
    np.testing.assert_allclose(np.asarray(t.albcan), np.asarray(m.swupw)[3] / inc)
    # sunlit/shaded mean of the leaf water potential in leafy layers
    lp = -2.5 * fs + -1.0 * (1 - fs)
    np.testing.assert_allclose(np.asarray(t.lwp_mean)[:3], lp)
    np.testing.assert_allclose(float(t.fracminlwp), np.sum(np.where(lp <= -2.0, d, 0)) / 1.2)
    t0 = canopy.canopy_totals(lay, m, leaves, 0.9, 0.3, flux_profile_type=0)
    np.testing.assert_allclose(float(t0.shflx), float(t.shveg + m.shsoi))
    with pytest.raises(ValueError):
        canopy.canopy_totals(lay, m, leaves, 0.9, 0.3, flux_profile_type=2)


def test_merge_sun_shade():
    lay = _layers()
    dpai = jnp.array([0.3, 0.0, 0.4, 0.0, 1e36, 1e36])
    fracsun = jnp.full(N_SLOTS, 0.25)
    tleaf = jnp.stack([jnp.full(N_SLOTS, 300.0), jnp.full(N_SLOTS, 296.0)], axis=1)
    m = canopy.merge_sun_shade(lay, dpai, fracsun, tleaf, tleaf - 300.0)
    merged = np.asarray(m.tleaf)
    np.testing.assert_allclose(merged[[0, 2]], 297.0)  # layers with plant area
    np.testing.assert_array_equal(merged[[1, 3, 4, 5]], np.asarray(tleaf)[[1, 3, 4, 5]])
    np.testing.assert_array_equal(np.asarray(m.tleaf_hist), np.asarray(tleaf))
    np.testing.assert_allclose(np.asarray(m.lwp)[0], -3.0)
