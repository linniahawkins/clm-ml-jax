"""The legacy view: model state under the reference's field names and 1-based shapes.

The end-to-end regression reference (``tests/regression/reference/*.npz``) stores raw legacy
fields (``mlcanopy.tair_profile``, ``temperature.t_soisno_col``, ...) with the legacy layout:
a leading patch axis of length 1 and 1-based layer, leaf, band and soil-level axes (index 0
unused or the ground). Phase 7 changes the state's names and layout (0-based, no patch axis,
fewer diagnostics); this view maps the state back, so every Phase 7 work package is checked
bit for bit against the Phase 6 reference (phase7_scope.md §3). The output variables of
``clm_ml.outputs`` (``_Raw``) and the validation stream variables (``mlcanopy.<field>``) are
computed from it too.

Until the step state exists (package 7.1) the view is today's containers, field by field.
At the end of Phase 7 the reference is re-recorded in the new names and this module goes.
"""

from __future__ import annotations

from typing import Mapping

#: Containers whose fields the view exposes (as ``<container>.<field>``).
CONTAINERS = ("mlcanopy", "temperature")


def legacy_view(insts) -> Mapping[str, object]:
    """``{"mlcanopy.<field>": array, "temperature.<field>": array}`` of one step's state
    ``insts`` (``ClmInstances``), in the legacy layout: leading patch axis, 1-based profile
    axes. Arrays are returned as they are (numpy or jax)."""
    view = {}
    for name in CONTAINERS:
        container = getattr(insts, name)
        view.update({f"{name}.{f}": getattr(container, f) for f in container._fields})
    return view


def raw_fields(view: Mapping[str, object], xp) -> dict:
    """The view's ``mlcanopy`` fields with a leading time axis of length 1, as
    ``clm_ml.outputs._Raw`` takes them."""
    return {k: xp.asarray(v)[None] for k, v in view.items() if k.startswith("mlcanopy.")}
