"""The canopy water routines of the legacy time step as adapters to
``clm_ml.physics.canopy_water`` (Phase 7.2, process 5; transition only: they go when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (index 0 the ground), leaves
``isun = 1``/``isha = 2`` (index 0 unused). Every layer slot is written, as before.
``DECLARED`` lists the container fields each adapter reads and writes (checked by
``tests/clm_ml/test_physics_canopy_water_hydraulics.py``).
"""

from __future__ import annotations

from typing import Sequence

from clm_ml.physics import canopy_water as physics
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
from multilayer_canopy.MLclm_varpar import isha, isun

_ML = "mlcanopy."
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "CanopyWettedFraction": (
        tuple(_ML + f for f in ("h2ocan_profile", "dpai_profile", "dlai_profile",
                                "fwet_profile", "fdry_profile")),
        (_ML + "fwet_profile", _ML + "fdry_profile"),
    ),
    "CanopyInterception": (
        tuple(_ML + f for f in ("qflx_rain_forcing", "qflx_snow_forcing", "lai_canopy",
                                "sai_canopy", "dpai_profile", "h2ocan_bef_profile",
                                "h2ocan_profile", "qflx_intr_canopy", "qflx_tflrain_canopy",
                                "qflx_tflsnow_canopy")),
        tuple(_ML + f for f in ("h2ocan_profile", "qflx_intr_canopy", "qflx_tflrain_canopy",
                                "qflx_tflsnow_canopy")),
    ),
    "CanopyEvaporation": (
        tuple(_ML + f for f in ("h2ocan_profile", "dpai_profile", "fracsun_profile",
                                "evleaf_leaf", "trleaf_leaf")),
        (_ML + "h2ocan_profile",),
    ),
}  # fmt: skip
_LEAVES = slice(isun, isha + 1)  # sunlit, shaded


def CanopyWettedFraction(
    num_filter: int, filter_patch: Sequence[int], mlcanopy_inst: mlcanopy_type
) -> mlcanopy_type:
    """Wetted and dry fractions of each layer (``physics.wetted_fraction``).

    Fortran reference: MLCanopyWaterMod.F90::CanopyWettedFraction.
    """
    inst = mlcanopy_inst
    fwet, fdry = inst.fwet_profile, inst.fdry_profile
    for fp in range(num_filter):
        p = filter_patch[fp]
        fwet_p, fdry_p = physics.wetted_fraction(
            inst.h2ocan_profile[p, 1:], inst.dpai_profile[p, 1:], inst.dlai_profile[p, 1:]
        )
        fwet = fwet.at[p, 1:].set(fwet_p)
        fdry = fdry.at[p, 1:].set(fdry_p)
    return inst._replace(fwet_profile=fwet, fdry_profile=fdry)


def CanopyInterception(
    num_filter: int, filter_patch: Sequence[int], mlcanopy_inst: mlcanopy_type, *, opts
) -> mlcanopy_type:
    """Interception and throughfall (``physics.interception``).

    Fortran reference: MLCanopyWaterMod.F90::CanopyInterception.
    """
    inst = mlcanopy_inst
    h2ocan, intr = inst.h2ocan_profile, inst.qflx_intr_canopy
    tflrain, tflsnow = inst.qflx_tflrain_canopy, inst.qflx_tflsnow_canopy
    for fp in range(num_filter):
        p = filter_patch[fp]
        r = physics.interception(
            inst.qflx_rain_forcing[p],
            inst.qflx_snow_forcing[p],
            inst.lai_canopy[p],
            inst.sai_canopy[p],
            inst.dpai_profile[p, 1:],
            inst.h2ocan_bef_profile[p, 1:],
            opts.dtime_ml,
        )
        h2ocan = h2ocan.at[p, 1:].set(r.h2ocan)
        intr = intr.at[p].set(r.qflx_intr)
        tflrain = tflrain.at[p].set(r.qflx_tflrain)
        tflsnow = tflsnow.at[p].set(r.qflx_tflsnow)
    return inst._replace(
        h2ocan_profile=h2ocan,
        qflx_intr_canopy=intr,
        qflx_tflrain_canopy=tflrain,
        qflx_tflsnow_canopy=tflsnow,
    )


def CanopyEvaporation(
    num_filter: int, filter_patch: Sequence[int], mlcanopy_inst: mlcanopy_type, *, opts
) -> mlcanopy_type:
    """Intercepted water after dew and evaporation (``physics.evaporation``).

    Fortran reference: MLCanopyWaterMod.F90::CanopyEvaporation.
    """
    inst = mlcanopy_inst
    h2ocan = inst.h2ocan_profile
    for fp in range(num_filter):
        p = filter_patch[fp]
        new = physics.evaporation(
            h2ocan[p, 1:],
            inst.dpai_profile[p, 1:],
            inst.fracsun_profile[p, 1:],
            inst.evleaf_leaf[p, 1:, _LEAVES],
            inst.trleaf_leaf[p, 1:, _LEAVES],
            opts.dtime_ml,
        )
        h2ocan = h2ocan.at[p, 1:].set(new)
    return inst._replace(h2ocan_profile=h2ocan)
