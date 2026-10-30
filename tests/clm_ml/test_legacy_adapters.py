"""The adapters between the legacy containers and the 0-based step state (Phase 7.1,
``clm_ml.legacy.state``) and the legacy view's field list, on small fake containers: 4 layer
slots, 3 soil levels, random values (so a block in the wrong place shows)."""

import functools
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clm_ml.legacy import state as A
from clm_ml.state import Diagnostics

N_SLOTS, N_SOIL, N_CHECKS = 4, 3, 13
#: diagnostics with a band axis and no layer axis: legacy (patch, 1 + band)
BAND_FIELDS = ("swskyb_forcing", "swskyd_forcing", "albcan_canopy", "swveg_canopy",
               "swvegsun_canopy", "swvegsha_canopy")  # fmt: skip


@pytest.fixture(autouse=True)
def _soil_levels(monkeypatch):
    from clm_src_main import clm_varpar

    monkeypatch.setattr(clm_varpar, "nlevgrnd", N_SOIL)


def _legacy_shape(layout) -> tuple:
    """The legacy shape (with the patch axis) of a field with ``layout``."""
    if isinstance(layout, A._SoilLevels):
        return (1, 1 + N_SOIL + 2)  # 1-based levels and unused snow/extra slots
    shapes = {
        A.SCALAR: (1,),
        A.LAYER: (1, N_SLOTS + 1),
        A.LAYER_LEAF: (1, N_SLOTS + 1, 3),
        A.LEVEL_BAND: (1, N_SLOTS + 1, 3),
    }
    return shapes[layout]


@functools.cache
def _container_type(name, fields):  # one class per container (jax.tree compares types)
    return namedtuple(name, fields)


Insts = namedtuple("Insts", ("mlcanopy", "temperature"))


def _containers(seed=0):
    """Fake ``ClmInstances`` with an ``mlcanopy`` and a ``temperature`` container."""
    rng = np.random.default_rng(seed)
    ml, temp = {}, {}
    for name in Diagnostics._fields:
        if name == "balance_error":
            shape = (1, N_CHECKS)
        elif name in BAND_FIELDS:
            shape = (1, 3)
        elif name in A._GROUND_LEVEL:
            shape = (1, N_SLOTS + 1)
        else:
            shape = _legacy_shape(A._diagnostic_layout(name, 1))
        ml[name] = rng.random(shape)
    for _, (inst, field, layout) in {**A.CANOPY_FIELDS, **A.SOIL_FIELDS}.items():
        (ml if inst == "mlcanopy" else temp)[field] = rng.random(_legacy_shape(layout))
    ml.update(
        ncan_canopy=np.array([3.0]),
        ntop_canopy=np.array([2.0]),
        nbot_canopy=np.array([2.0]),
        dz_profile=rng.random((1, N_SLOTS + 1)),
        zs_profile=rng.random((1, N_SLOTS + 1)),
        zw_profile=rng.random((1, N_SLOTS + 1)),
    )
    ML, T = _container_type("ML", tuple(sorted(ml))), _container_type("T", tuple(sorted(temp)))
    to_jax = lambda d, cls: cls(**{k: jnp.asarray(v) for k, v in d.items()})  # noqa: E731
    return Insts(to_jax(ml, ML), to_jax(temp, T))


def _assert_same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_shapes_are_0_based_without_patch_axis():
    step = A.step_state_from_instances(_containers())
    c = step.state.canopy
    assert c.air_temperature.shape == (N_SLOTS,)
    assert c.leaf_temperature.shape == (N_SLOTS, 2)
    assert c.ground_temperature.shape == ()
    assert step.state.soil.temperature.shape == (N_SOIL,)
    d = step.diagnostics
    assert d.lwdwn_profile.shape == (N_SLOTS + 1,)  # ground + layers
    assert d.swdwn_profile.shape == (N_SLOTS + 1, 2)
    assert d.swskyb_forcing.shape == (2,)
    assert d.anet_leaf.shape == (N_SLOTS, 2)
    assert d.balance_error.shape == (N_CHECKS,)


def test_blocks_are_the_legacy_entries():
    insts = _containers()
    c = A.state_from_instances(insts).canopy
    ml = insts.mlcanopy
    np.testing.assert_array_equal(c.air_temperature, ml.tair_profile[0, 1:])
    np.testing.assert_array_equal(c.leaf_temperature[:, 1], ml.tleaf_leaf[0, 1:, 2])  # shaded
    d = A.diagnostics_from_instances(insts)
    np.testing.assert_array_equal(d.swupw_profile[0, 0], ml.swupw_profile[0, 0, 1])  # ground, vis
    s = A.state_from_instances(insts).soil.temperature
    np.testing.assert_array_equal(s, insts.temperature.t_soisno_col[0, 1 : N_SOIL + 1])


def test_round_trip_reproduces_the_containers():
    insts = _containers()
    step = A.step_state_from_instances(insts)
    back = A.instances_with_step_state(insts, step, A.patch_layers(insts))
    _assert_same(back, insts)


def test_state_written_into_template_and_read_back():
    template, other = _containers(0), _containers(1)
    state = A.state_from_instances(other)
    insts = A.instances_with_state(template, state)
    _assert_same(A.state_from_instances(insts, state), state)
    # entries outside the blocks keep the template's values
    np.testing.assert_array_equal(
        insts.mlcanopy.tair_profile[0, 0], template.mlcanopy.tair_profile[0, 0]
    )
    np.testing.assert_array_equal(
        insts.mlcanopy.tleaf_leaf[0, :, 0], template.mlcanopy.tleaf_leaf[0, :, 0]
    )
    np.testing.assert_array_equal(
        insts.temperature.t_soisno_col[0, N_SOIL + 1 :],
        template.temperature.t_soisno_col[0, N_SOIL + 1 :],
    )


def test_batched_state_has_one_patch_axis():
    patches = [_containers(k) for k in range(3)]
    batched = jax.tree.map(lambda *x: jnp.stack(x), *patches)  # (patch, 1, ...)
    state = A.batched_state(batched)
    assert state.canopy.air_temperature.shape == (3, N_SLOTS)
    assert state.canopy.ground_temperature.shape == (3,)
    for k, insts in enumerate(patches):
        _assert_same(jax.tree.map(lambda x, k=k: x[k], state), A.state_from_instances(insts))


def test_put_rejects_a_block_of_the_wrong_shape():
    legacy = jnp.zeros((1, N_SLOTS + 1))
    with pytest.raises(ValueError, match="block of shape"):
        A.put(legacy, A.LAYER, jnp.ones(N_SLOTS + 1))  # a level array as a layer array


def test_patch_layer_masks():
    lay = A.patch_layers(_containers())  # ncan 3, ntop 2, nbot 2 (1-based)
    np.testing.assert_array_equal(lay.in_use, [True, True, True, False])
    np.testing.assert_array_equal(lay.in_canopy, [True, True, False, False])
    np.testing.assert_array_equal(lay.above_canopy, [False, False, True, False])
    np.testing.assert_array_equal(lay.leafy, [False, True, False, False])


def test_view_gives_every_reference_field():
    from clm_ml.legacy.view import FIELDS
    from tests.regression.record_run import REFERENCE_FILE

    with np.load(REFERENCE_FILE) as ref:
        recorded = {k for k in ref.files if not k.startswith("meta.")}
    assert recorded <= {f"{c}.{f}" for c, f in FIELDS}
