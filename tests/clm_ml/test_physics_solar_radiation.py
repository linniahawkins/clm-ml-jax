"""clm_ml.physics.solar_radiation and its legacy adapter (Phase 7.2, process 4).

The adapter runs on a fake multilayer canopy container holding only the fields it declares
(``clm_ml.legacy.solar_radiation.DECLARED``): reading an undeclared field fails, and exactly
the declared writes must change. The physics is checked against what it must conserve.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import solar_radiation as adapter  # noqa: E402
from clm_ml.physics import solar_radiation as solar  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 ground/unused)
NCAN, NTOP, NBOT = 5, 3, 2  # legacy 1-based counts: leafy layers 2..3, padding slot 6
Grid = namedtuple("Grid", "p")
Opts = namedtuple("Opts", "leaf_optics_type light_type")
Params = namedtuple(
    "Params",
    "leaf_reflectance leaf_transmittance stem_reflectance stem_transmittance "
    "leaf_angle_index clumping_index",
)
PARAMS = Params(
    jnp.array([0.0, 0.06, 0.42]), jnp.array([0.0, 0.04, 0.43]), jnp.array([0.0, 0.21, 0.49]),
    jnp.array([0.0, 0.001, 0.001]), 0.53, 0.85,
)  # fmt: skip


def _profiles():
    """Leaf and stem area of the legacy layer slots (leafy layers only)."""
    dlai = np.zeros(N_SLOTS + 1)
    dsai = np.zeros(N_SLOTS + 1)
    dlai[NBOT : NTOP + 1] = [0.8, 1.2]
    dsai[NBOT : NTOP + 1] = [0.4, 0.3]
    return dlai, dsai


def _fake_mlcanopy(rng):
    reads, writes = adapter.DECLARED["SolarRadiation"]
    fields = sorted({k.split(".")[1] for k in (*reads, *writes)})
    n = N_SLOTS + 1
    shapes = {"swleaf_leaf": (1, n, 3, 3), "apar_leaf": (1, n, 3), "balance_error": (1, 13)}
    for f in ("swbeam_profile", "swdwn_profile", "swupw_profile"):
        shapes[f] = (1, n, 3)
    for f in ("swskyb_forcing", "swskyd_forcing", "albsoib_soil", "albsoid_soil", "swsoi_soil",
              "swveg_canopy", "swvegsun_canopy", "swvegsha_canopy", "albcan_canopy"):  # fmt: skip
        shapes[f] = (1, 3)
    values = {}
    for f in fields:
        shape = shapes.get(f, (1, n) if f.endswith("_profile") else (1,))
        values[f] = jnp.asarray(rng.uniform(-1.0, -0.5, shape))
    dlai, dsai = _profiles()
    values.update(
        dlai_profile=jnp.asarray(dlai[None]),
        dsai_profile=jnp.asarray(dsai[None]),
        dpai_profile=jnp.asarray((dlai + dsai)[None]),
        ncan_canopy=jnp.array([NCAN]),
        ntop_canopy=jnp.array([NTOP]),
        nbot_canopy=jnp.array([NBOT]),
        solar_zen_forcing=jnp.array([0.6]),
        swskyb_forcing=jnp.array([[0.0, 300.0, 280.0]]),
        swskyd_forcing=jnp.array([[0.0, 60.0, 40.0]]),
        albsoib_soil=jnp.array([[0.0, 0.1, 0.2]]),
        albsoid_soil=jnp.array([[0.0, 0.1, 0.2]]),
        balance_error=-jnp.ones((1, 13)),  # any recorded error (>= 0) changes it
    )
    return namedtuple("mlcanopy", fields)(**values)


def test_adapter_reads_and_writes_what_it_declares():
    before = _fake_mlcanopy(np.random.default_rng(5))
    after = adapter.SolarRadiation(None, 1, [0], before, PARAMS, Grid(0), opts=Opts(0, 2))
    changed = set()
    for field in after._fields:
        old, new = np.asarray(getattr(before, field)), np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        assert np.all(np.isfinite(new)), field
        if not np.array_equal(old, new):
            changed.add(f"mlcanopy.{field}")
    assert changed == set(adapter.DECLARED["SolarRadiation"][1])
    # padding slot (layer 6) kept, layers in use outside the leaves 0
    np.testing.assert_array_equal(after.kb_profile[0, NCAN + 1 :], before.kb_profile[0, NCAN + 1 :])
    assert float(after.kb_profile[0, NCAN]) == 0.0
    assert float(np.max(after.balance_error[0])) < 1e-6  # conserves energy


def _physics():
    dlai, dsai = _profiles()
    layers = PatchLayers(
        n_layers=jnp.asarray(NCAN), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.ones(N_SLOTS), zs=jnp.ones(N_SLOTS),
        zw=jnp.ones(N_SLOTS + 1),
    )  # fmt: skip
    dpai = jnp.asarray((dlai + dsai)[1:])
    optics = solar.canopy_optics(
        0.6, jnp.asarray(dlai[1:]), jnp.asarray(dsai[1:]), dpai, layers,
        *(x[1:] for x in PARAMS[:4]), PARAMS.leaf_angle_index, PARAMS.clumping_index,
    )  # fmt: skip
    sky_b, sky_d, alb = jnp.array([300.0, 280.0]), jnp.array([60.0, 40.0]), jnp.array([0.1, 0.2])
    fluxes = solar.two_stream(optics, dpai, sky_b, sky_d, alb, alb, layers)
    return optics, fluxes, dpai, sky_b, sky_d


def test_two_stream_conserves_energy():
    optics, f, dpai, sky_b, sky_d = _physics()
    assert float(optics.sunlit_fraction_error) == 0.0
    incoming = sky_b + sky_d
    np.testing.assert_allclose(f.swveg + f.swsoi + f.albcan * incoming, incoming, rtol=1e-12)
    np.testing.assert_allclose(f.swvegsun + f.swvegsha, f.swveg, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f.error), 0.0, atol=1e-9)


def test_direct_beam_onto_the_ground_is_beer_law():
    optics, f, dpai, sky_b, _ = _physics()
    k = float(optics.kb[NBOT - 1])  # same in every leafy layer
    expected = np.exp(-k * PARAMS.clumping_index * float(dpai.sum()))
    np.testing.assert_allclose(optics.tbi_ground, expected, rtol=1e-12)
    np.testing.assert_allclose(f.swbeam[0], sky_b * expected, rtol=1e-12)


def test_layers_without_leaves_get_zero():
    optics, f, *_ = _physics()
    leafy = np.zeros(N_SLOTS, bool)
    leafy[NBOT - 1 : NTOP] = True
    for v in (optics.kb, optics.fracsun, optics.tbi, f.swleaf, f.swbeam[1:]):
        assert np.all(np.asarray(v)[~leafy] == 0.0)
        assert np.all(np.asarray(v)[leafy] != 0.0)
    np.testing.assert_allclose(solar.absorbed_par(f.swleaf), f.swleaf[..., 0] * 4.6, rtol=0)


def test_norman_is_registered_but_not_supported():
    assert set(solar.SOLAR_RADIATION) == {"two_stream", "norman"}
    with pytest.raises(NotImplementedError, match="norman is not supported"):
        solar.SOLAR_RADIATION["norman"]()
