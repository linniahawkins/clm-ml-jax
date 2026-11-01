"""The step-level canopy pieces of the legacy time step (``MLCanopyFluxes``) as adapters to
``clm_ml.physics.canopy`` (Phase 7.2, process 10; transition only: they go when the
containers leave the step).

Legacy layout: a patch axis, layer slots 1..max_layers (level 0 the ground), leaves
``isun = 1``/``isha = 2`` and bands ``ivis = 1``/``inir = 2`` (index 0 unused). ``DECLARED``
lists the container fields each adapter reads and writes (checked by
``tests/clm_ml/test_physics_canopy.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import canopy as physics
from clm_ml.physics.layers import keep
from multilayer_canopy.MLCanopyFluxesType import (
    CANOPY_ENERGY_1,
    PLANT_AREA,
    mlcanopy_type,
    record_balance_error,
)

_ML = "mlcanopy."
_LAYERS = ("ncan_canopy", "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile", "zw_profile")

#: SubstepFluxes entry -> (container field, kind); kinds give the legacy slots of the entry
_SCALAR, _BAND, _LAYER, _LEVEL, _LEVEL_BAND, _LEAF, _LEAF_BAND = range(7)
_SUBSTEP_FIELDS = {
    "ustar": ("ustar_canopy", _SCALAR),
    "beta": ("beta_canopy", _SCALAR),
    "obu": ("obu_canopy", _SCALAR),
    "z0m": ("z0m_canopy", _SCALAR),
    "zdisp": ("zdisp_canopy", _SCALAR),
    "lwup": ("lwup_canopy", _SCALAR),
    "swsoi": ("swsoi_soil", _BAND),
    "lwsoi": ("lwsoi_soil", _SCALAR),
    "rnsoi": ("rnsoi_soil", _SCALAR),
    "shsoi": ("shsoi_soil", _SCALAR),
    "lhsoi": ("lhsoi_soil", _SCALAR),
    "etsoi": ("etsoi_soil", _SCALAR),
    "gsoi": ("gsoi_soil", _SCALAR),
    "gac0": ("gac0_soil", _SCALAR),
    "qflx_intr": ("qflx_intr_canopy", _SCALAR),
    "qflx_tflrain": ("qflx_tflrain_canopy", _SCALAR),
    "qflx_tflsnow": ("qflx_tflsnow_canopy", _SCALAR),
    "swskyb": ("swskyb_forcing", _BAND),
    "swskyd": ("swskyd_forcing", _BAND),
    "lwsky": ("lwsky_forcing", _SCALAR),
    "lhtop": ("lhtop_canopy", _SCALAR),
    "shair": ("shair_profile", _LAYER),
    "etair": ("etair_profile", _LAYER),
    "stair": ("stair_profile", _LAYER),
    "mflx": ("mflx_profile", _LAYER),
    "kc_eddy": ("kc_eddy_profile", _LAYER),
    "gac": ("gac_profile", _LAYER),
    "swupw": ("swupw_profile", _LEVEL_BAND),
    "swdwn": ("swdwn_profile", _LEVEL_BAND),
    "swbeam": ("swbeam_profile", _LEVEL_BAND),
    "lwupw": ("lwupw_profile", _LEVEL),
    "lwdwn": ("lwdwn_profile", _LEVEL),
    "swleaf": ("swleaf_leaf", _LEAF_BAND),
    "lwleaf": ("lwleaf_leaf", _LEAF),
    "rnleaf": ("rnleaf_leaf", _LEAF),
    "shleaf": ("shleaf_leaf", _LEAF),
    "lhleaf": ("lhleaf_leaf", _LEAF),
    "trleaf": ("trleaf_leaf", _LEAF),
    "evleaf": ("evleaf_leaf", _LEAF),
    "stleaf": ("stleaf_leaf", _LEAF),
    "anet": ("anet_leaf", _LEAF),
    "agross": ("agross_leaf", _LEAF),
    "gs": ("gs_leaf", _LEAF),
}
#: legacy slots of each kind after the patch index. Leaf fields: every layer and leaf slot,
#: the unused index 0 included (the legacy averages wrote it, and the reference records it:
#: six copies of the 1e36 fill average to 9.999999999999999e35); :func:`zero_based` drops it.
_SLOTS = {
    _SCALAR: (),
    _BAND: (slice(1, None),),
    _LAYER: (slice(1, None),),
    _LEVEL: (slice(None),),
    _LEVEL_BAND: (slice(None), slice(1, None)),
    _LEAF: (slice(None), slice(None)),
    _LEAF_BAND: (slice(None), slice(None), slice(1, None)),
}
_SUBSTEP_READS = tuple(f for f, _ in _SUBSTEP_FIELDS.values() if f != "lhtop_canopy")

#: Canopy totals -> container fields (all ``*_canopy`` but the mean leaf water potential)
_TOTALS = {n: n + "_canopy" for n in physics.CanopyTotals._fields if n not in ("lwp_mean",
           "errors")}  # fmt: skip
_TOTALS_IN = {"dpai": "dpai_profile", "fracsun": "fracsun_profile", "fwet": "fwet_profile",
              "fdry": "fdry_profile", "wind": "wind_profile", "tair": "tair_profile",
              "vcmax25": "vcmax25_profile", "tleaf": "tleaf_leaf", "lwp": "lwp_leaf",
              "vcmax25_leaf": "vcmax25_leaf"}  # fmt: skip

_PLANT_AREA_WRITES = ("lai_canopy", "sai_canopy", "sai_floor_canopy", "dlai_profile",
                      "dsai_profile", "dpai_profile", "balance_error")  # fmt: skip
_MERGE_WRITES = ("tleaf_leaf", "lwp_leaf", "tleaf_hist_leaf", "lwp_hist_leaf")

#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "PlantAreaProfiles": (
        tuple(_ML + f for f in ("dlai_frac_profile", "dsai_frac_profile", *_LAYERS,
                                *_PLANT_AREA_WRITES))
        + ("canopystate.elai_patch", "canopystate.esai_patch"),
        tuple(_ML + f for f in _PLANT_AREA_WRITES),
    ),
    "SoilSurfaceHumidity": (
        (_ML + "rhg_soil", "soilstate.smp_l_col", "temperature.t_soisno_col"),
        (_ML + "rhg_soil",),
    ),
    "substep_fluxes": (tuple(_ML + f for f in (*_SUBSTEP_READS, "tref_forcing", *_LAYERS)),
                       ()),
    "WriteSubstepMeans": (
        tuple(_ML + f for f, _ in _SUBSTEP_FIELDS.values()) + tuple(_ML + f for f in _LAYERS),
        tuple(_ML + f for f, _ in _SUBSTEP_FIELDS.values()),
    ),
    "CanopyFluxesDiagnostics": (
        tuple(_ML + f for f in (*_TOTALS_IN.values(), "lai_canopy", "sai_canopy", *_LAYERS,
                                *_TOTALS.values(), "lwp_mean_profile", "balance_error")),
        tuple(_ML + f for f in (*_TOTALS.values(), "lwp_mean_profile", "balance_error")),
    ),
    "MergeSunShade": (
        tuple(_ML + f for f in ("dpai_profile", "fracsun_profile", *_LAYERS, *_MERGE_WRITES)),
        tuple(_ML + f for f in _MERGE_WRITES),
    ),
}  # fmt: skip


def PlantAreaProfiles(
    mlcanopy_inst: mlcanopy_type, canopystate_inst, p: int, layer_pai_floor: float
) -> mlcanopy_type:
    """Leaf, stem and plant area of patch ``p``'s layers for the step's LAI and SAI
    (``physics.plant_area``); the layers in use are written (padding kept) and the
    conservation error is recorded as the check ``PLANT_AREA``.

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 220-243).
    """
    inst = mlcanopy_inst
    lay = mlcanopy_layers(inst, p)
    r = physics.plant_area(
        lay,
        inst.dlai_frac_profile[p, 1:],
        inst.dsai_frac_profile[p, 1:],
        canopystate_inst.elai_patch[p],
        canopystate_inst.esai_patch[p],
        layer_pai_floor,
    )
    sai_floor = inst.sai_floor_canopy
    if sai_floor is None:  # a container from before the field existed
        sai_floor = jnp.zeros_like(inst.sai_canopy)

    def layer(arr, new):
        return arr.at[p, 1:].set(keep(lay.in_use, new, arr[p, 1:]))

    inst = inst._replace(
        lai_canopy=inst.lai_canopy.at[p].set(r.lai),
        sai_canopy=inst.sai_canopy.at[p].set(r.sai),
        sai_floor_canopy=sai_floor.at[p].set(r.sai_floor),
        dlai_profile=layer(inst.dlai_profile, r.dlai),
        dsai_profile=layer(inst.dsai_profile, r.dsai),
        dpai_profile=layer(inst.dpai_profile, r.dpai),
    )
    return record_balance_error(inst, p, PLANT_AREA, r.error)


def SoilSurfaceHumidity(
    mlcanopy_inst: mlcanopy_type, soilstate_inst, temperature_inst, p: int, c: int
) -> mlcanopy_type:
    """Relative humidity at the soil surface of patch ``p`` from its column ``c``'s first
    soil layer (``physics.soil_surface_humidity``).

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 259-262).
    """
    rhg = physics.soil_surface_humidity(
        soilstate_inst.smp_l_col[c, 1], temperature_inst.t_soisno_col[c, 1]
    )
    return mlcanopy_inst._replace(rhg_soil=mlcanopy_inst.rhg_soil.at[p].set(rhg))


def substep_fluxes(mlcanopy_inst: mlcanopy_type, p: int) -> physics.SubstepFluxes:
    """The values of patch ``p`` that a sub-step adds to the step's averages (every slot; leaf
    fields with the legacy index 0, see :data:`_SLOTS`)."""
    inst = mlcanopy_inst
    values = {}
    for name, (field, kind) in _SUBSTEP_FIELDS.items():
        if name == "lhtop":
            values[name] = physics.canopy_top_latent_heat(
                mlcanopy_layers(inst, p), inst.etair_profile[p, 1:], inst.tref_forcing[p]
            )
        else:
            values[name] = getattr(inst, field)[(p,) + _SLOTS[kind]]
    return physics.SubstepFluxes(**values)


def zero_based(fluxes: physics.SubstepFluxes) -> physics.SubstepFluxes:
    """``fluxes`` from :func:`substep_fluxes` (leaf fields with the legacy index 0) as the
    physics takes them (0-based layers and leaves)."""
    return fluxes._replace(
        **{
            name: getattr(fluxes, name)[1:, 1:]
            for name, (_, kind) in _SUBSTEP_FIELDS.items()
            if kind in (_LEAF, _LEAF_BAND)
        }
    )


def WriteSubstepMeans(
    mlcanopy_inst: mlcanopy_type, p: int, means: physics.SubstepFluxes
) -> mlcanopy_type:
    """The container with patch ``p``'s averages over the sub-steps written: layer fields in
    the layers in use, level fields at the ground and levels up to the top layer in use,
    leaf fields in every slot (padding kept elsewhere).

    Fortran reference: MLCanopyFluxesMod.F90::MLTimeStepFluxIntegration (``nstep_ml ==
    num_ml_steps``).
    """
    inst = mlcanopy_inst
    lay = mlcanopy_layers(inst, p)
    up_to_top = jnp.arange(lay.dz.shape[0] + 1) <= lay.n_layers  # levels 0..ncan
    changes = {}
    for name, (field, kind) in _SUBSTEP_FIELDS.items():
        arr, new = getattr(inst, field), getattr(means, name)
        if arr is None:  # lhtop_canopy: a container from before the field existed
            arr = jnp.zeros_like(inst.lwsky_forcing)
        sl = (p,) + _SLOTS[kind]
        if kind == _LAYER:
            new = keep(lay.in_use, new, arr[sl])
        elif kind in (_LEVEL, _LEVEL_BAND):
            new = keep(up_to_top, new, arr[sl])
        changes[field] = arr.at[sl].set(new)
    return inst._replace(**changes)


def CanopyFluxesDiagnostics(
    mlcanopy_inst: mlcanopy_type, p: int, means: physics.SubstepFluxes, *, opts
) -> mlcanopy_type:
    """Canopy totals of patch ``p`` (``physics.canopy_totals``) from the step's averages
    ``means`` and the end-of-step layers and leaves; the four energy checks are recorded.
    Only the totals outputs and checks read are written (D6).

    Fortran reference: MLCanopyFluxesMod.F90::CanopyFluxesDiagnostics.
    """
    inst = mlcanopy_inst
    lay = mlcanopy_layers(inst, p)
    leaves = physics.CanopyLeaves(
        **{
            name: getattr(inst, field)[(p, slice(1, None))
                                       + ((slice(1, None),) if field.endswith("_leaf") else ())]
            for name, field in _TOTALS_IN.items()
        }
    )  # fmt: skip
    # The averages through a barrier, as Fortran stores them before the totals (the
    # compiled chunk is closer to the legacy one with it: 4 fields 1 ulp off instead of 54
    # fields at round-off; C19)
    t = physics.canopy_totals(
        lay,
        jax.lax.optimization_barrier(zero_based(means)),
        leaves,
        inst.lai_canopy[p],
        inst.sai_canopy[p],
        flux_profile_type=opts.flux_profile_type,
    )
    changes = {}
    for name, field in _TOTALS.items():
        arr, new = getattr(inst, field), getattr(t, name)
        changes[field] = arr.at[(p, slice(1, None)) if arr.ndim == 2 else p].set(new)
    lwp_mean = inst.lwp_mean_profile
    changes["lwp_mean_profile"] = lwp_mean.at[p, 1:].set(
        keep(lay.in_use, t.lwp_mean, lwp_mean[p, 1:])
    )
    inst = inst._replace(**changes)
    for k in range(4):
        inst = record_balance_error(inst, p, CANOPY_ENERGY_1 + k, t.errors[k])
    return inst


def MergeSunShade(mlcanopy_inst: mlcanopy_type, p: int) -> mlcanopy_type:
    """Sunlit and shaded leaf temperature and water potential of patch ``p`` merged to their
    layer mean (``physics.merge_sun_shade``), the values before the merge kept as
    ``*_hist_leaf``; the layers in use are written.

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 355-375).
    """
    inst = mlcanopy_inst
    lay = mlcanopy_layers(inst, p)
    leaves = (p, slice(1, None), slice(1, None))
    m = physics.merge_sun_shade(
        lay,
        inst.dpai_profile[p, 1:],
        inst.fracsun_profile[p, 1:],
        inst.tleaf_leaf[leaves],
        inst.lwp_leaf[leaves],
    )
    changes = {}
    for name, field in zip(m._fields, _MERGE_WRITES):
        arr = getattr(inst, field)
        changes[field] = arr.at[leaves].set(keep(lay.in_use, getattr(m, name), arr[leaves]))
    # The unused leaf index 0 of the history fields takes the leaves' (as the legacy copy did;
    # the reference records it)
    for field, source in (("tleaf_hist_leaf", inst.tleaf_leaf), ("lwp_hist_leaf", inst.lwp_leaf)):
        arr, sl = changes[field], (p, slice(1, None), 0)
        changes[field] = arr.at[sl].set(keep(lay.in_use, source[sl], arr[sl]))
    return inst._replace(**changes)
