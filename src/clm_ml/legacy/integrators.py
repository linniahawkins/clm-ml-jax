"""The Runge-Kutta state update of the legacy time step as an adapter to
``clm_ml.physics.integrators`` (Phase 7.2, process 9; transition only: it goes when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (index 0 the ground), the leaf axis
with its unused index 0 (updated with the leaves, as before), increments with a trailing
stage axis. ``DECLARED`` lists the container fields the adapter reads and writes (checked by
``tests/clm_ml/test_physics_nitrogen_integrators.py``).
"""

from __future__ import annotations

from clm_ml.config import INTEGRATOR_CODES
from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import integrators as physics
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions

_ML = "mlcanopy."
#: ``physics.Integrated`` field -> (state, its value at the start of the sub-step, increments)
_FIELDS = {
    "tair": ("tair_profile", "tair_bef_profile", "dtair_profile"),
    "eair": ("eair_profile", "eair_bef_profile", "deair_profile"),
    "h2ocan": ("h2ocan_profile", "h2ocan_bef_profile", "dh2ocan_profile"),
    "tleaf": ("tleaf_leaf", "tleaf_bef_leaf", "dtleaf_leaf"),
    "lwp": ("lwp_leaf", "lwp_bef_leaf", "dlwp_leaf"),
    "tg": ("tg_soil", "tg_bef_soil", "dtg_soil"),
}
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "RungeKuttaUpdate": (
        tuple(_ML + f for f in (*(f for fs in _FIELDS.values() for f in fs), "ncan_canopy",
                                "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile",
                                "zw_profile")),
        tuple(_ML + f for fs in _FIELDS.values() for f in (fs[0], fs[2])),
    )
}  # fmt: skip
#: Fortran ``runge_kutta_type`` -> ``numerics.integrator``
INTEGRATOR = {code: name for name, code in INTEGRATOR_CODES.items()}


def integrator_tableau(opts: MLOptions) -> physics.Tableau:
    """The Butcher tableau of ``opts.runge_kutta_type``.

    Fortran reference: MLRungeKuttaMod.F90::RungeKuttaIni.
    """
    if opts.runge_kutta_type not in INTEGRATOR:
        raise ValueError(f"runge_kutta_type {opts.runge_kutta_type} not one of {tuple(INTEGRATOR)}")
    return physics.TABLEAUS[INTEGRATOR[opts.runge_kutta_type]]


def RungeKuttaUpdate(
    irk, rk: physics.Tableau, mlcanopy_inst: mlcanopy_type, grid: GridInfo
) -> mlcanopy_type:
    """The state of patch ``grid.p`` after Runge-Kutta stage ``irk`` (1-based)
    (``physics.runge_kutta_update``).

    Fortran reference: MLRungeKuttaMod.F90::RungeKuttaUpdate.
    """
    inst = mlcanopy_inst
    p = grid.p
    rows = (p, slice(1, None))  # layers 1..max_layers
    sl = {"tair": rows, "eair": rows, "h2ocan": rows, "tleaf": rows + (slice(None),),
          "lwp": rows + (slice(None),), "tg": (p,)}  # fmt: skip

    def _gather(i):
        return physics.Integrated(**{n: getattr(inst, fs[i])[sl[n]] for n, fs in _FIELDS.items()})

    new, increment, k = physics.runge_kutta_update(
        irk, rk, mlcanopy_layers(inst, p), _gather(1), _gather(0), _gather(2)
    )
    updates = {}
    for name, (state, _, delta) in _FIELDS.items():
        updates[state] = getattr(inst, state).at[sl[name]].set(getattr(new, name))
        updates[delta] = getattr(inst, delta).at[sl[name] + (k,)].set(getattr(increment, name))
    return inst._replace(**updates)
