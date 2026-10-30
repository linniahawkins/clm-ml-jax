"""
Boundary adapters between the legacy containers and :mod:`clm_ml.state` (Phase 7.1).

The physics still runs on the legacy containers (1-based, leading patch axis; ``ClmInstances``)
until Phase 7.2 moves it. A time step rebuilds its containers from a fixed *template* (the
containers after the first step: layer grid, PFT tables, and fields that every step
overwrites) plus the carried :class:`State`, and returns its results as a :class:`StepState`.
The regression test checks bit for bit that this reproduces the reference, i.e. that
``State`` holds everything one step passes to the next and that the legacy view
(:mod:`clm_ml.legacy.view`) rebuilds every recorded field from ``StepState`` and
``PatchLayers``.

Every new field is a block of a legacy array (patch index 0, which ``vmap`` makes the patch's
own row): :data:`LAYOUTS` gives the block's first index on each axis after the patch axis.
Reading takes the block; writing puts it into the template with ``lax.dynamic_update_slice``
(constant start indices, not a scatter: scatters into the scan carry misbehaved on CPU,
numerics doc §6). Entries outside the blocks (index 0 of layer and leaf axes, unused soil
levels) are constant fill values, taken from the template.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from clm_ml.state import CanopyState, Diagnostics, PatchLayers, SoilState, State, StepState

# First index of the new array in the legacy array, per axis after the patch axis.
SCALAR = ()  # legacy (patch,)
LAYER = (1,)  # (patch, 1 + layer): index 0 unused
LEVEL = (0,)  # (patch, ground + layers): index 0 the ground
BAND = (1,)  # (patch, 1 + band)
LAYER_LEAF = (1, 1)  # (patch, 1 + layer, 1 + leaf)
LEVEL_BAND = (0, 1)  # (patch, ground + layers, 1 + band)


class _SoilLevels(tuple):
    """The soil layout: starts like ``LAYER`` but covers only the first ``nlevgrnd`` levels."""


SOIL = _SoilLevels((1,))  # (column, 1 + soil level + unused)
WHOLE = (0,)  # (patch, n): every entry

#: State field -> (ClmInstances field, container field, layout)
CANOPY_FIELDS = {
    "air_temperature": ("mlcanopy", "tair_profile", LAYER),
    "vapor_pressure": ("mlcanopy", "eair_profile", LAYER),
    "co2": ("mlcanopy", "cair_profile", LAYER),
    "leaf_temperature": ("mlcanopy", "tleaf_leaf", LAYER_LEAF),
    "leaf_water_potential": ("mlcanopy", "lwp_leaf", LAYER_LEAF),
    "intercepted_water": ("mlcanopy", "h2ocan_profile", LAYER),
    "ground_temperature": ("mlcanopy", "tg_soil", SCALAR),
    "acclimation_temperature": ("temperature", "t_a10_patch", SCALAR),
}
SOIL_FIELDS = {
    "temperature": ("temperature", "t_soisno_col", SOIL),
}

#: Diagnostics with a ground level (index 0 of the legacy layer axis is the ground)
_GROUND_LEVEL = {"lwdwn_profile", "lwupw_profile"}
_GROUND_LEVEL_BAND = {"swbeam_profile", "swdwn_profile", "swupw_profile"}


def _diagnostic_layout(name: str, ndim: int) -> tuple:
    """The layout of diagnostic ``name`` (an ``mlcanopy`` field of ``ndim`` legacy axes)."""
    if name == "balance_error":
        return WHOLE
    if name in _GROUND_LEVEL:
        return LEVEL
    if name in _GROUND_LEVEL_BAND:
        return LEVEL_BAND
    if name.endswith("_leaf"):
        return LAYER_LEAF
    if name.endswith("_profile"):
        return LAYER
    return {1: SCALAR, 2: BAND}[ndim]


def _size(legacy, layout: tuple) -> tuple:
    """The new array's shape: the rest of each axis after its start (soil: nlevgrnd)."""
    if isinstance(layout, _SoilLevels):
        from clm_src_main import clm_varpar

        return (clm_varpar.nlevgrnd,)
    return tuple(n - s for n, s in zip(legacy.shape[1:], layout))


def get(legacy, layout: tuple):
    """The new (0-based, no patch axis) array held in ``legacy`` at ``layout``."""
    block = tuple(slice(s, s + n) for s, n in zip(layout, _size(legacy, layout)))
    return legacy[(0,) + block]


def put(legacy, layout: tuple, value):
    """``legacy`` with the block at ``layout`` replaced by ``value``."""
    value = jnp.asarray(value, dtype=legacy.dtype)
    return lax.dynamic_update_slice(legacy, value[None], (0,) + tuple(layout))


def _get_fields(insts, fields: dict) -> dict:
    return {
        name: get(getattr(getattr(insts, inst), field), layout)
        for name, (inst, field, layout) in fields.items()
    }


def state_from_instances(insts, previous: State | None = None) -> State:
    """The prognostic variables in ``insts`` (a ``ClmInstances``). State-only fields (not in
    the containers: ``acclimation_steps``) are carried from ``previous``; without one (the
    first step) the running mean holds one step."""
    canopy = _get_fields(insts, CANOPY_FIELDS)
    if previous is not None:
        canopy["acclimation_steps"] = previous.canopy.acclimation_steps
    else:
        canopy["acclimation_steps"] = jnp.ones_like(canopy["acclimation_temperature"])
    return State(canopy=CanopyState(**canopy), soil=SoilState(**_get_fields(insts, SOIL_FIELDS)))


def batched_state(template) -> State:
    """The :class:`State` of a batched template (containers with a leading axis of patches,
    each with its legacy patch axis of length 1): every field with a leading patch axis."""
    import jax

    return jax.vmap(state_from_instances)(template)


def _with_fields(template, values: dict):
    """``template`` (a ``ClmInstances``) with ``{(inst, field): (layout, value)}`` put in."""
    changes: dict[str, dict] = {}
    for (inst, field), (layout, value) in values.items():
        container = getattr(template, inst)
        legacy = changes.get(inst, {}).get(field, getattr(container, field))
        changes.setdefault(inst, {})[field] = put(legacy, layout, value)
    return template._replace(
        **{inst: getattr(template, inst)._replace(**fields) for inst, fields in changes.items()}
    )


def _state_values(state: State) -> dict:
    values = {}
    for part, fields in ((state.canopy, CANOPY_FIELDS), (state.soil, SOIL_FIELDS)):
        for name, (inst, field, layout) in fields.items():  # not state-only fields
            values[(inst, field)] = (layout, getattr(part, name))
    return values


def instances_with_state(template, state: State):
    """``template`` (a ``ClmInstances``) with its prognostic variables taken from ``state``."""
    return _with_fields(template, _state_values(state))


def diagnostics_from_instances(insts) -> Diagnostics:
    """The :class:`Diagnostics` of a step's containers ``insts``."""
    ml = insts.mlcanopy
    return Diagnostics(
        **{
            name: get(getattr(ml, name), _diagnostic_layout(name, getattr(ml, name).ndim))
            for name in Diagnostics._fields
        }
    )


def step_state_from_instances(insts, previous: State | None = None) -> StepState:
    """A step's result (its containers ``insts``) as a :class:`StepState`."""
    return StepState(
        state=state_from_instances(insts, previous), diagnostics=diagnostics_from_instances(insts)
    )


def patch_layers(insts) -> PatchLayers:
    """The :class:`PatchLayers` of the containers ``insts`` (counts in the legacy dtype)."""
    ml = insts.mlcanopy
    return PatchLayers(
        n_layers=ml.ncan_canopy[0],
        n_within=ml.ntop_canopy[0],
        first_leafy=ml.nbot_canopy[0] - 1,
        dz=get(ml.dz_profile, LAYER),
        zs=get(ml.zs_profile, LAYER),
        zw=get(ml.zw_profile, LEVEL),
    )


def instances_with_step_state(template, step: StepState, layers: PatchLayers):
    """``template`` with everything ``step`` and ``layers`` hold put in: the containers the
    legacy view reads."""
    ml = template.mlcanopy
    values = _state_values(step.state)
    for name in Diagnostics._fields:
        layout = _diagnostic_layout(name, getattr(ml, name).ndim)
        values[("mlcanopy", name)] = (layout, getattr(step.diagnostics, name))
    for field, value in (
        ("ncan_canopy", layers.n_layers),
        ("ntop_canopy", layers.n_within),
        ("nbot_canopy", layers.first_leafy + 1),
    ):
        values[("mlcanopy", field)] = (SCALAR, value)
    for field, layout, value in (
        ("dz_profile", LAYER, layers.dz),
        ("zs_profile", LAYER, layers.zs),
        ("zw_profile", LEVEL, layers.zw),
    ):
        values[("mlcanopy", field)] = (layout, value)
    return _with_fields(template, values)
