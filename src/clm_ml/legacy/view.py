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

The entries the new state does not hold (index 0 of layer and leaf axes, unused soil levels:
constant fill values) come from the template (the containers after the first step). At the
end of Phase 7 the reference is re-recorded in the new names and this module goes.
"""

from __future__ import annotations

from typing import Mapping

from clm_ml.legacy.state import (
    instances_with_step_state,
    patch_layers,
    step_state_from_instances,
)
from clm_ml.state import Diagnostics, PatchLayers, StepState

#: The fields the view gives, ``(container, field)``: every field the reference records
#: (and the legacy ``.out`` writer reads).
FIELDS = (
    tuple(("mlcanopy", f) for f in Diagnostics._fields if f != "balance_error")
    + tuple(("mlcanopy", f) for f in ("tair_profile", "eair_profile"))  # State
    + tuple(  # PatchLayers
        ("mlcanopy", f)
        for f in ("ncan_canopy", "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile")
    )
    + (("mlcanopy", "zw_profile"), ("temperature", "t_soisno_col"))
)


def legacy_view(template, step: StepState, layers: PatchLayers) -> Mapping[str, object]:
    """``{"mlcanopy.<field>": array, "temperature.<field>": array}`` of one step's results
    ``step`` for the patch with layer grid ``layers``, in the legacy layout (leading patch
    axis, 1-based profile axes). ``template``: the patch's containers after the first step
    (``ClmInstances``), for the entries the state does not hold. Arrays are jax arrays."""
    insts = instances_with_step_state(template, step, layers)
    return {f"{c}.{f}": getattr(getattr(insts, c), f) for c, f in FIELDS}


def view_of_instances(insts) -> Mapping[str, object]:
    """The legacy view of a step's containers ``insts`` (one patch), through the step state:
    the first (initializing) step, and output shapes."""
    return legacy_view(insts, step_state_from_instances(insts), patch_layers(insts))


def view_series(template, steps: StepState) -> Mapping[str, object]:
    """The legacy view of a chunk's steps (``steps`` with a leading step axis, as a
    one-patch chunk returns them): every field with a leading step axis."""
    import jax

    layers = patch_layers(template)
    return jax.vmap(lambda s: legacy_view(template, s, layers))(steps)


def raw_fields(view: Mapping[str, object], xp) -> dict:
    """The view's ``mlcanopy`` fields with a leading time axis of length 1, as
    ``clm_ml.outputs._Raw`` takes them."""
    return {k: xp.asarray(v)[None] for k, v in view.items() if k.startswith("mlcanopy.")}
