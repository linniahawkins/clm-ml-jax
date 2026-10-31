"""clm_ml.physics.flux_profile, clm_ml.physics.water_vapor and the legacy adapter (Phase 7.2,
process 8).

The adapter runs on a fake multilayer canopy container holding only the fields it declares
(``clm_ml.legacy.flux_profile.DECLARED``): reading an undeclared field fails, and exactly the
declared writes must change. The physics is checked with Fortran's own checks of the
implicit solution (``ErrorCheck01``: the leaf and soil fluxes equal those of the implicit
solution; ``ErrorCheck02``: energy and water vapor are conserved in every layer, the canopy
and at the ground), and padding slots must not change the solution.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import flux_profile as adapter  # noqa: E402
from clm_ml.physics import flux_profile as fp  # noqa: E402
from clm_ml.physics import water_vapor as wv  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

# 7 slots: layer 0 without plant area, 1-3 leafy, 4-5 above the canopy, 6 padding
N_SLOTS, NCAN, NTOP, NBOT = 7, 6, 4, 2  # legacy 1-based counts
DPAI = np.array([0.0, 0.6, 0.9, 0.5, 0.0, 0.0, 0.0])
DTIME = 300.0
Grid = namedtuple("Grid", "p")
Opts = namedtuple("Opts", "flux_profile_type dtime_ml")
# tolerances of Fortran's checks (W/m2 for fluxes, K and Pa for temperatures and pressure)
TOLERANCE = dict(leaf_temperature=1e-6, soil_temperature=1e-6, soil_vapor_pressure=1e-6)


def _layers(n_layers=NCAN):
    zw = np.concatenate([[0.0], np.cumsum(np.full(N_SLOTS, 2.0))])
    dz = np.diff(zw)
    return PatchLayers(
        n_layers=jnp.asarray(n_layers), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.asarray(dz),
        zs=jnp.asarray(zw[:-1] + 0.5 * dz), zw=jnp.asarray(zw),
    )  # fmt: skip


def _inputs(rng):
    n = N_SLOTS
    leaf = lambda lo, hi: jnp.asarray(rng.uniform(lo, hi, (n, 2)))  # noqa: E731
    layer = lambda lo, hi: jnp.asarray(rng.uniform(lo, hi, n))  # noqa: E731
    canopy = fp.Canopy(
        dpai=jnp.asarray(DPAI), fwet=layer(0.0, 0.3), fdry=layer(0.5, 0.7),
        fracsun=layer(0.2, 0.8), cpleaf=layer(500.0, 900.0), gac=layer(1.0, 5.0),
        tair_bef=layer(290.0, 295.0), eair_bef=layer(1200.0, 1600.0), gbh=leaf(0.5, 2.0),
        gbv=leaf(0.5, 2.0), gs=leaf(0.05, 0.4), rnleaf=leaf(50.0, 400.0),
        tleaf_bef=leaf(290.0, 298.0),
    )  # fmt: skip
    ground = fp.Ground(*(jnp.asarray(v) for v in (293.0, 0.95, 120.0, 300.0, 291.0, 0.05,
                                                  1.2, 0.8)))  # fmt: skip
    ref = fp.Reference(*(jnp.asarray(v) for v in (101325.0, 41.6, 29.2, 292.0, 293.5,
                                                  1400.0)))  # fmt: skip
    return canopy, ground, ref


def test_implicit_solution_passes_fortran_checks():
    canopy, ground, ref = _inputs(np.random.default_rng(0))
    layers = _layers()
    f = fp.implicit(canopy, ground, ref, layers, DTIME)
    errors = fp.implicit_errors(f, canopy, ground, layers)
    for name, value in errors._asdict().items():
        assert float(value) < TOLERANCE.get(name, 1e-3) * 1e-3, (name, float(value))
    assert float(jnp.max(jnp.abs(f.leaf.error))) < 1e-9
    assert abs(float(f.soil.error)) < 1e-9
    # leafless layers: no leaf fluxes, leaf temperature = air temperature
    leafless = DPAI[:NCAN] == 0
    np.testing.assert_array_equal(np.asarray(f.leaf.shleaf)[:NCAN][leafless], 0.0)
    np.testing.assert_array_equal(
        np.asarray(f.leaf.tleaf)[:NCAN][leafless, 0], np.asarray(f.tair)[:NCAN][leafless]
    )


def test_padding_does_not_change_the_solution():
    canopy, ground, ref = _inputs(np.random.default_rng(1))
    a = fp.implicit(canopy, ground, ref, _layers(), DTIME)
    # different values in the padding slot
    junk = canopy._replace(gac=canopy.gac.at[NCAN:].set(1e3), dpai=canopy.dpai.at[NCAN:].set(9.0))
    b = fp.implicit(junk, ground, ref, _layers(), DTIME)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim:
            x, y = x[:NCAN], y[:NCAN]
        np.testing.assert_array_equal(x, y)
    for x in jax.tree.leaves(b):
        assert np.all(np.isfinite(np.asarray(x)))


def test_leaf_and_soil_fluxes_close_their_energy_balance():
    canopy, ground, ref = _inputs(np.random.default_rng(2))
    lf = fp.leaf_fluxes(
        canopy.tair_bef[:, None], canopy.eair_bef[:, None], canopy.cpleaf[:, None],
        canopy.fwet[:, None], canopy.fdry[:, None], canopy.gbh, canopy.gbv, canopy.gs,
        canopy.rnleaf, canopy.tleaf_bef, canopy.dpai[:, None], ref.pref, ref.cpair, ref.tref,
        DTIME,
    )  # fmt: skip
    leafy = DPAI > 0
    lh = np.asarray(lf.lhleaf)[leafy]
    balance = np.asarray(canopy.rnleaf)[leafy] - np.asarray(lf.shleaf)[leafy] - lh
    np.testing.assert_allclose(balance, np.asarray(lf.stleaf)[leafy], atol=1e-9)
    s = fp.soil_fluxes(292.0, 1400.0, ground, ref)
    np.testing.assert_allclose(s.shsoi + s.lhsoi + s.gsoi, ground.rnsoi, atol=1e-9)
    np.testing.assert_allclose(s.lhsoi, s.etsoi * wv.lat_vap(ref.tref), rtol=1e-14)


def _fake_mlcanopy(rng):
    reads, writes = adapter.DECLARED["FluxProfileSolution"]
    fields = sorted({k.split(".")[1] for k in (*reads, *writes)})
    n = N_SLOTS + 1
    canopy, ground, ref = _inputs(rng)
    values = {}
    for f in fields:
        if f.endswith("_leaf"):
            shape = (1, n, 3)
        elif f.endswith("_profile"):
            shape = (1, n)
        elif f == "balance_error":
            shape = (1, 13)
        else:
            shape = (1,)
        values[f] = jnp.asarray(rng.uniform(-1.0, -0.5, shape))

    def legacy(x):  # 0-based (layer, ...) -> legacy (1, layer + 1, ...), leaves at 1, 2
        x = np.asarray(x)
        if x.ndim == 2:
            x = np.concatenate([np.zeros((N_SLOTS, 1)), x], axis=1)
        return jnp.asarray(np.concatenate([np.zeros((1,) + x.shape[1:]), x])[None])

    for name, value in canopy._asdict().items():
        values[name + ("_leaf" if np.ndim(value) == 2 else "_profile")] = legacy(value)
    for name, value in ground._asdict().items():
        values[name + "_soil"] = jnp.asarray(value)[None]
    for name, value in ref._asdict().items():
        values[name + "_forcing"] = jnp.asarray(value)[None]
    lay = _layers()
    values.update(
        ncan_canopy=jnp.array([NCAN]), ntop_canopy=jnp.array([NTOP]),
        nbot_canopy=jnp.array([NBOT]), dz_profile=legacy(lay.dz), zs_profile=legacy(lay.zs),
        zw_profile=lay.zw[None], co2ref_forcing=jnp.array([383.0]),
        balance_error=-jnp.ones((1, 13)),
    )  # fmt: skip
    return namedtuple("mlcanopy", fields)(**values)


def test_adapter_reads_and_writes_what_it_declares():
    before = _fake_mlcanopy(np.random.default_rng(3))
    after = adapter.FluxProfileSolution(1, (0,), before, Grid(0), opts=Opts(1, DTIME))
    changed = set()
    for field in after._fields:
        old, new = np.asarray(getattr(before, field)), np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        assert np.all(np.isfinite(new)), field
        if not np.array_equal(old, new):
            changed.add(f"mlcanopy.{field}")
    assert changed == set(adapter.DECLARED["FluxProfileSolution"][1])
    # the padding slot kept; balance checks recorded and passed
    for f in ("tair_profile", "shair_profile", "tleaf_leaf"):
        np.testing.assert_array_equal(getattr(after, f)[0, NCAN + 1 :],
                                      getattr(before, f)[0, NCAN + 1 :])  # fmt: skip
    assert float(np.max(after.balance_error[0])) < 1e-9
    # same values as the physics (the adapter is compiled: round-off)
    canopy, ground, ref = _inputs(np.random.default_rng(3))
    f = fp.implicit(canopy, ground, ref, _layers(), DTIME)
    np.testing.assert_allclose(after.tair_profile[0, 1 : NCAN + 1], f.tair[:NCAN], rtol=1e-12)
    np.testing.assert_allclose(after.trleaf_leaf[0, 1 : NCAN + 1, 1:], f.leaf.trleaf[:NCAN],
                               rtol=1e-12, atol=1e-300)  # fmt: skip


def test_well_mixed_is_refused():
    with pytest.raises(NotImplementedError):
        fp.FLUX_PROFILE["well_mixed"]()
    assert fp.FLUX_PROFILE["implicit"] is fp.implicit


def test_water_vapor():
    t = jnp.array([250.0, 273.15, 300.0])
    es, desdt = wv.sat_vap(t)
    np.testing.assert_allclose(es[1], 611.213476, rtol=1e-12)  # a0 at 0 C, over water
    h = 1e-4  # derivative polynomial matches the vapor pressure polynomial
    fd = (wv.sat_vap(t + h)[0] - wv.sat_vap(t - h)[0]) / (2 * h)
    np.testing.assert_allclose(desdt[jnp.array([0, 2])], fd[jnp.array([0, 2])], rtol=1e-4)
    lam = wv.lat_vap(t)
    assert float(lam[0]) == float(lam[1]) > float(lam[2])  # sublimation at or below freezing
