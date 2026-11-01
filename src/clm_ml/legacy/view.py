"""The legacy view: a step's results under the reference's field names and 1-based shapes.

The end-to-end regression reference (``tests/regression/reference/*.npz``) stores raw legacy
fields (``mlcanopy.tair_profile``, ``temperature.t_soisno_col``, ...) with the legacy layout:
a leading patch axis of length 1 and 1-based layer, leaf, band and soil-level axes (index 0
unused or the ground). Since Phase 7.1 a step's results are a :class:`clm_ml.state.StepState`
(0-based, no patch axis, only what outputs and checks use) and the patch's layer grid a
:class:`clm_ml.state.PatchLayers`; this view maps them back, so every Phase 7 work package is
checked bit for bit against the Phase 6 reference (phase7_scope.md §3). The output variables
of ``clm_ml.outputs`` (``_Raw``), the validation stream variables (``mlcanopy.<field>``) and
the legacy ``.out`` files are computed from it, so nothing reads a step's containers.

The entries the new state does not hold (index 0 of layer, leaf and band axes, the legacy
snow levels and unused soil levels) hold the containers' fill value, ``spval`` (Phase 7.4:
checked to be the only value there; before, they were copied from the template). At the end
of Phase 7 the reference is re-recorded in the new names and this module goes.
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp

from clm_ml.legacy.state import (
    BAND,
    CANOPY_FIELDS,
    LAYER,
    SCALAR,
    SOIL_FIELDS,
    _diagnostic_layout,
    patch_layers,
    step_state_from_instances,
)
from clm_ml.state import Diagnostics, PatchLayers, StepState

#: The fields the view gives, ``(container, field)``: the diagnostics, the prognostic state
#: and the layer grid under their legacy names. They include every field the reference
#: records and the legacy ``.out`` writer reads, and are the only raw fields a validation
#: stream can name (``mlcanopy.<field>``).
FIELDS = (
    tuple(("mlcanopy", f) for f in Diagnostics._fields if f != "balance_error")
    + tuple((c, f) for c, f, _ in (*CANOPY_FIELDS.values(), *SOIL_FIELDS.values()))
    + tuple(
        ("mlcanopy", f)
        for f in ("ncan_canopy", "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile")
    )
    + (("mlcanopy", "zw_profile"),)
)

#: Legacy snow levels in front of the soil levels of ``t_soisno_col`` (index ``-nlevsno + 1
#: .. nlevgrnd``, stored from 0), by the number of ground levels (clm_varpar)
_SNOW_LEVELS = {25: 12, 15: 5}

_SPVAL = 1.0e36  # clm_varcon.spval: the legacy containers' fill value


def _legacy(value, layout: tuple):
    """``value`` (0-based, no patch axis) as a legacy array: ``layout[i]`` fill slots in front
    of axis ``i`` and a leading patch axis of length 1."""
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value[None]
    pads = [(s, 0) for s in layout] + [(0, 0)] * (value.ndim - len(layout))
    return jnp.pad(value, pads, constant_values=_SPVAL)[None]


def _layout(name: str, value) -> tuple:
    """The legacy layout of diagnostic ``name`` (``_diagnostic_layout`` from the 0-based
    array: scalars and band arrays are told apart by their rank)."""
    if name.endswith(("_leaf", "_profile")) or name == "balance_error":
        return _diagnostic_layout(name, 0)
    return SCALAR if jnp.ndim(value) == 0 else BAND


def legacy_view(step: StepState, layers: PatchLayers) -> Mapping[str, object]:
    """``{"mlcanopy.<field>": array, "temperature.<field>": array}`` of one step's results
    ``step`` for the patch with layer grid ``layers``, in the legacy layout (leading patch
    axis, 1-based profile axes, ``spval`` where the state holds nothing). Arrays are jax
    arrays."""
    out = {}
    for name in Diagnostics._fields:
        if name != "balance_error":
            value = getattr(step.diagnostics, name)
            out[f"mlcanopy.{name}"] = _legacy(value, _layout(name, value))
    for part, fields in ((step.state.canopy, CANOPY_FIELDS), (step.state.soil, SOIL_FIELDS)):
        for name, (inst, field, layout) in fields.items():
            value = getattr(part, name)
            if field == "t_soisno_col":  # soil level j at index j; snow slots after them
                n = value.shape[-1]
                snow = _SNOW_LEVELS[n]
                full = jnp.pad(value, (1, snow - 1), constant_values=_SPVAL)
                out[f"{inst}.{field}"] = full[None]
            else:
                out[f"{inst}.{field}"] = _legacy(value, layout)
    for field, value in (
        ("ncan_canopy", layers.n_layers),
        ("ntop_canopy", layers.n_within),
        ("nbot_canopy", layers.first_leafy + 1),
    ):
        out[f"mlcanopy.{field}"] = jnp.asarray(value, dtype=jnp.int32)[None]
    out["mlcanopy.dz_profile"] = _legacy(layers.dz, LAYER)
    out["mlcanopy.zs_profile"] = _legacy(layers.zs, LAYER)
    out["mlcanopy.zw_profile"] = jnp.asarray(layers.zw)[None]
    return {f"{c}.{f}": out[f"{c}.{f}"] for c, f in FIELDS}


def view_of_instances(insts) -> Mapping[str, object]:
    """The legacy view of a step's containers ``insts`` (one patch), through the step state:
    the first (initializing) step, and output shapes."""
    return legacy_view(step_state_from_instances(insts), patch_layers(insts))


def view_series(steps: StepState, layers: PatchLayers) -> Mapping[str, object]:
    """The legacy view of a chunk's steps (``steps`` with a leading step axis, as a
    one-patch chunk returns them): every field with a leading step axis."""
    import jax

    return jax.vmap(lambda s: legacy_view(s, layers))(steps)


def raw_fields(view: Mapping[str, object], xp) -> dict:
    """The view's ``mlcanopy`` fields with a leading time axis of length 1, as
    ``clm_ml.outputs._Raw`` takes them."""
    return {k: xp.asarray(v)[None] for k, v in view.items() if k.startswith("mlcanopy.")}
