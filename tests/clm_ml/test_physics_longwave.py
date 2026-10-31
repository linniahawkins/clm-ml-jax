"""clm_ml.physics.longwave and its legacy adapter (Phase 7.2, process 5).

The adapter runs on a fake multilayer canopy container holding only the fields it declares
(``clm_ml.legacy.longwave.DECLARED``): reading an undeclared field fails, and exactly the
declared writes must change. The physics is checked against what it must conserve.
"""

from collections import namedtuple

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import longwave as adapter  # noqa: E402
from clm_ml.physics import longwave as lw  # noqa: E402
from clm_ml.physics.constants import sb  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 ground/unused)
NCAN, NTOP, NBOT = 5, 3, 2  # legacy 1-based counts: leafy layers 2..3, padding slot 6
EMLEAF = 0.98
Grid = namedtuple("Grid", "p")
Opts = namedtuple("Opts", "longwave_type")
Params = namedtuple("Params", "leaf_emissivity")


def _profiles():
    """Plant area and diffuse transmittance of the legacy layer slots (leafy layers only)."""
    dpai = np.zeros(N_SLOTS + 1)
    td = np.zeros(N_SLOTS + 1)
    dpai[NBOT : NTOP + 1] = [1.2, 1.5]
    td[NBOT : NTOP + 1] = [0.4, 0.3]
    return dpai, td


def _fake_mlcanopy(rng):
    reads, writes = adapter.DECLARED["LongwaveRadiation"]
    fields = sorted({k.split(".")[1] for k in (*reads, *writes)})
    n = N_SLOTS + 1
    shapes = {"tleaf_leaf": (1, n, 3), "lwleaf_leaf": (1, n, 3), "balance_error": (1, 13)}
    values = {}
    for f in fields:
        shape = shapes.get(f, (1, n) if f.endswith("_profile") else (1,))
        values[f] = jnp.asarray(rng.uniform(-1.0, -0.5, shape))
    dpai, td = _profiles()
    values.update(
        dpai_profile=jnp.asarray(dpai[None]),
        td_profile=jnp.asarray(td[None]),
        fracsun_profile=jnp.asarray(rng.uniform(0.2, 0.8, (1, n))),
        tleaf_leaf=jnp.asarray(rng.uniform(285.0, 300.0, (1, n, 3))),
        ncan_canopy=jnp.array([NCAN]),
        ntop_canopy=jnp.array([NTOP]),
        nbot_canopy=jnp.array([NBOT]),
        lwsky_forcing=jnp.array([330.0]),
        tg_soil=jnp.array([292.0]),
        balance_error=-jnp.ones((1, 13)),  # any recorded error (>= 0) changes it
    )
    return namedtuple("mlcanopy", fields)(**values)


def test_adapter_reads_and_writes_what_it_declares():
    before = _fake_mlcanopy(np.random.default_rng(3))
    after = adapter.LongwaveRadiation(None, 1, [0], before, Params(EMLEAF), Grid(0), opts=Opts(1))
    changed = set()
    for field in after._fields:
        old, new = np.asarray(getattr(before, field)), np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        assert np.all(np.isfinite(new)), field
        if not np.array_equal(old, new):
            changed.add(f"mlcanopy.{field}")
    assert changed == set(adapter.DECLARED["LongwaveRadiation"][1])
    # levels above the canopy top and the padding kept; below the leaves 0
    for f in ("lwupw_profile", "lwleaf_leaf"):
        np.testing.assert_array_equal(
            getattr(after, f)[0, NTOP + 1 :], getattr(before, f)[0, NTOP + 1 :]
        )
    assert float(after.lwupw_profile[0, NBOT - 1]) == 0.0
    assert float(np.max(after.balance_error[0])) < 1e-6  # conserves energy


def _layers():
    return PatchLayers(
        n_layers=jnp.asarray(NCAN), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.ones(N_SLOTS), zs=jnp.ones(N_SLOTS),
        zw=jnp.ones(N_SLOTS + 1),
    )  # fmt: skip


def _norman(tleaf, lwsky, tg):
    dpai, td = _profiles()
    fracsun = jnp.full(N_SLOTS, 0.4)
    return lw.norman(tleaf, fracsun, jnp.asarray(td[1:]), jnp.asarray(dpai[1:]), lwsky, tg,
                     _layers(), EMLEAF)  # fmt: skip


def test_norman_conserves_energy():
    tleaf = jnp.asarray(np.random.default_rng(1).uniform(285.0, 300.0, (N_SLOTS, 2)))
    f = _norman(tleaf, 330.0, 292.0)
    np.testing.assert_allclose(f.lwveg + f.lwsoi, 330.0 - f.lwup, rtol=1e-12)
    assert abs(float(f.error)) < 1e-9
    leafy = np.zeros(N_SLOTS, bool)
    leafy[NBOT - 1 : NTOP] = True
    assert np.all(np.asarray(f.lwleaf)[~leafy] == 0.0)
    np.testing.assert_array_equal(f.lwleaf[:, 0], f.lwleaf[:, 1])  # same for sun and shade


def test_isothermal_canopy_is_in_radiative_equilibrium():
    # Everything at T under a sky emitting sigma T^4: no net exchange (Kirchhoff's law)
    t = 290.0
    f = _norman(jnp.full((N_SLOTS, 2), t), sb * t**4, t)
    np.testing.assert_allclose(f.lwup, sb * t**4, rtol=1e-12)
    assert abs(float(f.lwveg)) < 1e-9 and abs(float(f.lwsoi)) < 1e-9
    np.testing.assert_allclose(np.asarray(f.lwleaf), 0.0, atol=1e-9)


def test_registry():
    assert lw.LONGWAVE_RADIATION == {"norman": lw.norman}
