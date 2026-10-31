"""The longwave radiation routine of the legacy time step as an adapter to
``clm_ml.physics.longwave`` (Phase 7.2, process 5; transition only: it goes when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (level 0 the ground), leaves
``isun = 1``/``isha = 2`` (index 0 unused). ``DECLARED`` lists the container fields the
adapter reads and writes (checked by ``tests/clm_ml/test_physics_longwave.py``).
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import longwave as physics
from clm_src_main.abortutils import endrun
from clm_src_main.decompMod import bounds_type
from multilayer_canopy.MLCanopyFluxesType import LONGWAVE, mlcanopy_type, record_balance_error
from multilayer_canopy.MLclm_varctl import GridInfo
from multilayer_canopy.MLclm_varpar import isha, isun

_ML = "mlcanopy."
_WRITES = ("lwupw_profile", "lwdwn_profile", "lwleaf_leaf", "lwsoi_soil", "lwveg_canopy",
           "lwup_canopy", "balance_error")  # fmt: skip
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "LongwaveRadiation": (
        tuple(_ML + f for f in (
            "tleaf_leaf", "fracsun_profile", "td_profile", "dpai_profile", "lwsky_forcing",
            "tg_soil", "ncan_canopy", "ntop_canopy", "nbot_canopy", "dz_profile",
            "zs_profile", "zw_profile", *_WRITES)),
        tuple(_ML + f for f in _WRITES),
    )
}  # fmt: skip


def LongwaveRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    params,
    grid: GridInfo,
    *,
    opts,
) -> mlcanopy_type:
    """Longwave radiation of the canopy layers and the ground (``physics.norman``; Fortran
    ``longwave_type = 1``, the only scheme).

    Fortran reference: MLLongwaveRadiationMod.F90::LongwaveRadiation. Levels 0..ntop are
    written (Fortran ``lwupw(0:ntop)`` etc.: 0 outside the leafy layers); the layers above
    the canopy and the padding keep their values.
    """
    if opts.longwave_type != 1:
        endrun(msg=" ERROR: LongwaveRadiation: longwave_type not valid")
    inst = mlcanopy_inst
    p = grid.p
    lay = mlcanopy_layers(inst, p)
    ls = slice(1, None)
    f = physics.norman(
        inst.tleaf_leaf[p, ls, isun : isha + 1],
        inst.fracsun_profile[p, ls],
        inst.td_profile[p, ls],
        inst.dpai_profile[p, ls],
        inst.lwsky_forcing[p],
        inst.tg_soil[p],
        lay,
        params.leaf_emissivity,
    )
    inst = record_balance_error(inst, p, LONGWAVE, f.error)

    upto_top = jnp.arange(inst.dpai_profile.shape[1]) <= lay.n_within  # levels 0..ntop

    def _set(arr, new):
        return arr.at[p].set(jnp.where(upto_top, new, arr[p]))

    lwleaf = inst.lwleaf_leaf
    for leaf, il in ((0, isun), (1, isha)):
        new = jnp.concatenate([jnp.zeros(1), f.lwleaf[:, leaf]])  # level 0: 0
        lwleaf = lwleaf.at[p, :, il].set(jnp.where(upto_top, new, lwleaf[p, :, il]))
    return inst._replace(
        lwup_canopy=inst.lwup_canopy.at[p].set(f.lwup),
        lwveg_canopy=inst.lwveg_canopy.at[p].set(f.lwveg),
        lwsoi_soil=inst.lwsoi_soil.at[p].set(f.lwsoi),
        lwleaf_leaf=lwleaf,
        lwupw_profile=_set(inst.lwupw_profile, f.lwupw),
        lwdwn_profile=_set(inst.lwdwn_profile, f.lwdwn),
    )
