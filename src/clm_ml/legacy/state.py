"""
Copy :class:`clm_ml.state.State` out of, and back into, the legacy state containers.

A time step rebuilds its containers from a fixed *template* (the containers after the first
step: layer grid, PFT tables, and fields that every step overwrites) plus the carried
``State``. The regression test checks that this reproduces the reference bit for bit, i.e.
that ``State`` holds everything one step passes to the next.
"""

from __future__ import annotations

from clm_ml.state import CanopyState, SoilState, State

# State field -> (ClmInstances field, container field)
CANOPY_FIELDS = {
    "air_temperature": ("mlcanopy", "tair_profile"),
    "vapor_pressure": ("mlcanopy", "eair_profile"),
    "co2": ("mlcanopy", "cair_profile"),
    "leaf_temperature": ("mlcanopy", "tleaf_leaf"),
    "leaf_water_potential": ("mlcanopy", "lwp_leaf"),
    "intercepted_water": ("mlcanopy", "h2ocan_profile"),
    "ground_temperature": ("mlcanopy", "tg_soil"),
    "acclimation_temperature": ("temperature", "t_a10_patch"),
}
SOIL_FIELDS = {
    "temperature": ("temperature", "t_soisno_col"),
}


def state_from_instances(insts, previous: State | None = None) -> State:
    """The prognostic variables in ``insts`` (a ``ClmInstances``). State-only fields (not in
    the containers: ``acclimation_steps``) are carried from ``previous``; without one (the
    first step) the running mean holds one step."""
    import jax.numpy as jnp

    def get(fields):
        return {
            name: getattr(getattr(insts, inst), field) for name, (inst, field) in fields.items()
        }

    canopy = get(CANOPY_FIELDS)
    if previous is not None:
        canopy["acclimation_steps"] = previous.canopy.acclimation_steps
    else:
        canopy["acclimation_steps"] = jnp.ones_like(canopy["acclimation_temperature"])
    return State(canopy=CanopyState(**canopy), soil=SoilState(**get(SOIL_FIELDS)))


def instances_with_state(template, state: State):
    """``template`` (a ``ClmInstances``) with its prognostic variables taken from ``state``."""
    changes: dict[str, dict] = {}
    for part, fields in (
        (state.canopy, CANOPY_FIELDS),
        (state.soil, SOIL_FIELDS),
    ):  # not state-only
        for name, (inst, field) in fields.items():
            changes.setdefault(inst, {})[field] = getattr(part, name)
    return template._replace(
        **{inst: getattr(template, inst)._replace(**fields) for inst, fields in changes.items()}
    )
