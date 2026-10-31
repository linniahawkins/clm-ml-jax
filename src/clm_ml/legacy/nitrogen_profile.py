"""The canopy nitrogen profile of the legacy time step as an adapter to
``clm_ml.physics.nitrogen_profile`` (Phase 7.2, process 9; transition only: it goes when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (index 0 the ground), leaves
``isun = 1``/``isha = 2`` (index 0 unused). Every layer slot is written (0 outside the layers
with plant area), as before. ``DECLARED`` lists the container fields the adapter reads and
writes (checked by ``tests/clm_ml/test_physics_nitrogen_integrators.py``).
"""

from __future__ import annotations

from typing import Sequence

import jax

from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import nitrogen_profile as physics
from multilayer_canopy.MLCanopyFluxesType import NITROGEN, mlcanopy_type, record_balance_error
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun

_ML = "mlcanopy."
_WRITES = ("vcmax25_leaf", "jmax25_leaf", "rd25_leaf", "kp25_leaf", "vcmax25_profile",
           "jmax25_profile", "rd25_profile", "kp25_profile")  # fmt: skip
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "CanopyNitrogenProfile": (
        tuple(_ML + f for f in ("dpai_profile", "kb_profile", "tbi_profile", "fracsun_profile",
                                "lai_canopy", "sai_canopy", "tacclim_forcing", "ncan_canopy",
                                "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile",
                                "zw_profile", *_WRITES, "balance_error")),
        tuple(_ML + f for f in (*_WRITES, "balance_error")),
    )
}  # fmt: skip
_LEAVES = slice(isun, isha + 1)  # sunlit, shaded

# One program per call, as the legacy routine was (step-by-step runs)
_canopy_nitrogen_profile = jax.jit(
    physics.canopy_nitrogen_profile,
    static_argnames=("c3", "acclimation", "kn_val", "vary_with_height"),
)


def CanopyNitrogenProfile(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    params,
    grid: GridInfo,
    *,
    opts: MLOptions,
) -> mlcanopy_type:
    """Leaf rates at 25 C of patch ``grid.p``'s layers
    (``physics.canopy_nitrogen_profile``); the canopy integration error is recorded as the
    check ``NITROGEN``.

    Fortran reference: MLCanopyNitrogenProfileMod.F90::CanopyNitrogenProfile.
    """
    for name, code in (("acclim_type", opts.acclim_type),
                       ("leaf_optics_type", opts.leaf_optics_type)):  # fmt: skip
        if code not in (0, 1):
            raise ValueError(f"CanopyNitrogenProfile: {name} {code} not one of (0, 1)")
    inst = mlcanopy_inst
    for fp in range(num_filter):
        p = filter_patch[fp]
        r = _canopy_nitrogen_profile(
            mlcanopy_layers(inst, p),
            inst.dpai_profile[p, 1:],
            inst.kb_profile[p, 1:],
            inst.tbi_profile[p, 1:],
            inst.fracsun_profile[p, 1:],
            inst.lai_canopy[p],
            inst.sai_canopy[p],
            inst.tacclim_forcing[p],
            params.vcmax25,
            params.clumping_index,
            c3=bool(grid.c3),
            acclimation=bool(opts.acclim_type == 1),
            kn_val=float(opts.kn_val),
            vary_with_height=bool(opts.leaf_optics_type == 1),
        )
        updates = {}
        for field in _WRITES:
            arr = getattr(inst, field)
            sl = (p, slice(1, None), _LEAVES) if field.endswith("_leaf") else (p, slice(1, None))
            updates[field] = arr.at[sl].set(getattr(r, field))
        inst = record_balance_error(inst._replace(**updates), p, NITROGEN, r.integration_error)
    return inst
