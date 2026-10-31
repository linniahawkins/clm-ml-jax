"""clm_ml.physics.atm_forcing and its legacy adapter (Phase 7.2, process 3).

The adapter runs on a fake multilayer canopy container holding only the fields it declares
(``clm_ml.legacy.atm_forcing.DECLARED``): reading an undeclared field fails, and exactly the
declared writes must change. The physics is checked against the interpolation it describes.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import atm_forcing as adapter  # noqa: E402
from clm_ml.physics import atm_forcing  # noqa: E402
from clm_ml.physics.constants import wind_forc_min  # noqa: E402

N_PATCH, N_BAND = 1, 3  # legacy layout: one patch, band column 0 unused
Grid = namedtuple("Grid", "p")
Opts = namedtuple("Opts", "met_type")


def _legacy_record(rng, uref=3.0):
    return adapter.AtmForcingRecord(
        uref=np.full(N_PATCH, uref),
        tref=rng.uniform(280.0, 300.0, N_PATCH),
        qref=rng.uniform(0.004, 0.012, N_PATCH),
        pref=rng.uniform(9.8e4, 1.01e5, N_PATCH),
        co2ref=rng.uniform(380.0, 390.0, N_PATCH),
        swskyb=rng.uniform(0.0, 400.0, (N_PATCH, N_BAND)),
        swskyd=rng.uniform(0.0, 100.0, (N_PATCH, N_BAND)),
        lwsky=rng.uniform(300.0, 400.0, N_PATCH),
    )


def _window(rng):
    recs = [_legacy_record(rng, u) for u in (0.2, 2.0, 4.0)]
    return adapter.AtmForcingWindow(*(jax.tree_util.tree_map(jnp.asarray, r) for r in recs))


def _fake_mlcanopy(rng):
    reads, writes = adapter.DECLARED["GetAtmForcing"]
    fields = sorted({k.split(".")[1] for k in (*reads, *writes)})
    shape = {"swskyb_forcing": (N_PATCH, N_BAND), "swskyd_forcing": (N_PATCH, N_BAND)}
    values = {f: jnp.asarray(rng.uniform(-1.0, 0.0, shape.get(f, (N_PATCH,)))) for f in fields}
    values["zref_forcing"] = jnp.full(N_PATCH, 23.0)
    return namedtuple("mlcanopy", fields)(**values)


@pytest.mark.parametrize("met_type", sorted(adapter.MET_TYPE))
def test_adapter_reads_and_writes_what_it_declares(met_type):
    rng = np.random.default_rng(3)
    before = _fake_mlcanopy(rng)
    after = adapter.GetAtmForcing(
        10.0, 11.0, 12.0, 11.3, 1, None, before, _window(rng), Grid(0), opts=Opts(met_type)
    )
    changed = set()
    for field in after._fields:
        old, new = np.asarray(getattr(before, field)), np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        assert np.all(np.isfinite(new)), field
        if not np.array_equal(old, new):
            changed.add(f"mlcanopy.{field}")
    assert changed == set(adapter.DECLARED["GetAtmForcing"][1])
    # the unused band column is not written
    assert after.swskyb_forcing[0, 0] == before.swskyb_forcing[0, 0]


def test_adapter_rejects_two_point_interpolation():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="met_type 2"):
        adapter.GetAtmForcing(
            10.0, 11.0, 12.0, 11.3, 1, None, _fake_mlcanopy(rng), _window(rng), Grid(0),
            opts=Opts(2),
        )  # fmt: skip


def _record(x):
    return atm_forcing.ForcingRecord(
        x, x, x, x, x, jnp.array([x, 2 * x]), jnp.array([x, 3 * x]), x
    )  # fmt: skip


@pytest.mark.parametrize("t_ml, expected", [(1.0, 10.0), (2.0, 20.0), (3.0, 40.0), (1.5, 15.0),
                                            (2.5, 30.0)])  # fmt: skip
def test_three_point_interpolation_is_piecewise_linear(t_ml, expected):
    bef, cur, nxt = _record(10.0), _record(20.0), _record(40.0)
    f = atm_forcing.TIME_INTERPOLATION["three_point"](bef, cur, nxt, 1.0, 2.0, 3.0, t_ml)
    np.testing.assert_allclose(f.tref, expected, rtol=1e-14)
    np.testing.assert_allclose(f.swskyb, [expected, 2 * expected], rtol=1e-14)
    np.testing.assert_allclose(f.swskyd, [expected, 3 * expected], rtol=1e-14)


def test_derived_air_properties():
    rec = atm_forcing.ForcingRecord(
        jnp.asarray(0.1), jnp.asarray(293.15), jnp.asarray(0.01), jnp.asarray(1.0e5),
        jnp.asarray(383.0), jnp.zeros(2), jnp.zeros(2), jnp.asarray(350.0),
    )  # fmt: skip
    f = atm_forcing.atm_forcing(rec, rec, rec, 23.0, 0.0, 1.0, 2.0, 1.0, interpolation="none")
    assert float(f.uref) == wind_forc_min  # wind below the minimum is raised to it
    # ideal gas: rho = p M / (R T) with the moist-air molecular mass
    np.testing.assert_allclose(f.rhoair, 1.0e5 * f.mmair / (8.31446 * 293.15), rtol=1e-12)
    assert 0.0285 < float(f.mmair) < 0.02897  # moist air is lighter than dry air
    np.testing.assert_allclose(f.thref, 293.15 + 0.0098 * 23.0, rtol=1e-14)
