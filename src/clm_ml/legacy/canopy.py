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
#: legacy slots of each kind after the patch index (the unused index 0 of layer, leaf and
#: band axes is not written by the step since process 10b: C20)
_S1, _ALL = slice(1, None), slice(None)
_SLOTS = {
    _SCALAR: (),
    _BAND: (_S1,),
    _LAYER: (_S1,),
    _LEVEL: (_ALL,),
    _LEVEL_BAND: (_ALL, _S1),
    _LEAF: (_S1, _S1),
    _LEAF_BAND: (_S1, _S1, _S1),
}
#: the same for the kinds of ``physics.canopy.SUBSTEP_KINDS``
_KIND_SLOTS = {
    "scalar": (), "band": (_S1,), "layer": (_S1,), "level": (_ALL,), "level_band": (_ALL, _S1),
    "leaf": (_S1, _S1), "leaf_band": (_S1, _S1, _S1), "stage": (_ALL,),
    "layer_stage": (_S1, _ALL), "leaf_stage": (_S1, _S1, _ALL), "checks": (_ALL,),
}  # fmt: skip

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
    # The averages through a barrier, as Fortran stores them before the totals (C19)
    t = physics.canopy_totals(
        lay,
        jax.lax.optimization_barrier(means),
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
    return inst._replace(**changes)


# ---------------------------------------------------------------------------
# The sub-step (physics.canopy.substep) on the legacy containers
# ---------------------------------------------------------------------------

#: Fortran codes -> model.yaml names
_MET_TYPE = {0: "none", 3: "three_point"}
_GROUND = {1: "extend_hf", 2: "log_profile"}
_REGIME = {1: "laminar", 2: "turbulent", 3: "free_convection"}
_STOMATAL = {0: "medlyn", 1: "ball_berry", 2: "wue"}
_WUE_SOLVER = {1: "brent", 2: "bisection"}


def substep_options(opts) -> physics.SubstepOptions:
    """The sub-step's choices from the legacy options (``MLOptions``), with the legacy
    routines' checks: Norman solar radiation and the well-mixed flux-profile solution are
    registered but not supported (Phase 7 decision D8: ``clm_ml.run`` refuses them; the
    legacy translations stay in ``clm_ml.legacy.solar_radiation`` and
    ``clm_ml.legacy.flux_profile``), the other codes must be valid."""
    from clm_ml.legacy.integrators import INTEGRATOR

    if opts.light_type == 1:
        raise NotImplementedError("Norman solar radiation (light_type 1) is not supported")
    if opts.flux_profile_type in (0, -1):
        raise NotImplementedError("the well-mixed flux-profile solution is not supported")
    checks = (("met_type", opts.met_type, _MET_TYPE), ("light_type", opts.light_type, (2,)),
              ("leaf_optics_type", opts.leaf_optics_type, (0,)),
              ("acclim_type", opts.acclim_type, (0, 1)), ("longwave_type", opts.longwave_type, (1,)),
              ("turb_type", opts.turb_type, (1,)),
              ("HF_extension_type", opts.HF_extension_type, _GROUND),
              ("gb_type", opts.gb_type, _REGIME), ("gs_type", opts.gs_type, _STOMATAL),
              ("gs_solver", opts.gs_solver, _WUE_SOLVER), ("gspot_type", opts.gspot_type, (0, 1)),
              ("colim_type", opts.colim_type, (0, 1)),
              ("flux_profile_type", opts.flux_profile_type, (1,)),
              ("runge_kutta_type", opts.runge_kutta_type, INTEGRATOR))  # fmt: skip
    for name, code, valid in checks:
        if code not in valid:
            raise ValueError(f"{name} {code} not one of {tuple(valid)}")
    return physics.SubstepOptions(
        interpolation=_MET_TYPE[opts.met_type],
        acclimation=bool(opts.acclim_type == 1),
        kn_val=float(opts.kn_val),
        sparse_canopy=bool(opts.sparse_canopy_type == 1),
        ground_conductance=_GROUND[opts.HF_extension_type],
        regime=_REGIME[opts.gb_type],
        stomatal_conductance=_STOMATAL[opts.gs_type],
        water_stress=bool(opts.gspot_type == 1),
        colimitation=bool(opts.colim_type == 1),
        wue_solver=_WUE_SOLVER[opts.gs_solver],
        dtime=float(opts.dtime_ml),
        integrator=INTEGRATOR[opts.runge_kutta_type],
    )


def _kind_of(field: str) -> str:
    return next(k for k, fs in physics.SUBSTEP_KINDS.items() if field in fs)


def substep_from_container(mlcanopy_inst: mlcanopy_type, p: int) -> physics.Substep:
    """Patch ``p``'s values the sub-step carries, 0-based (every slot)."""
    return physics.Substep(
        **{f: getattr(mlcanopy_inst, f)[(p,) + _KIND_SLOTS[_kind_of(f)]]
           for f in physics.Substep._fields}
    )  # fmt: skip


def container_with_substep(
    mlcanopy_inst: mlcanopy_type, p: int, c: physics.Substep
) -> mlcanopy_type:
    """The container with patch ``p``'s sub-step values ``c`` written (the unused index 0
    of layer, leaf and band axes kept)."""
    return mlcanopy_inst._replace(
        **{f: getattr(mlcanopy_inst, f).at[(p,) + _KIND_SLOTS[_kind_of(f)]].set(getattr(c, f))
           for f in physics.Substep._fields}
    )  # fmt: skip


def substep_inputs(mlcanopy_inst: mlcanopy_type, p: int, window, times) -> physics.SubstepInputs:
    """Patch ``p``'s values fixed over the step's sub-steps; ``window`` the legacy forcing
    window (``AtmForcingWindow``), ``times`` the calendar days of its three records."""
    from clm_ml.legacy.atm_forcing import _patch_record

    inst = mlcanopy_inst

    def layer(f):
        return getattr(inst, f)[p, 1:]

    return physics.SubstepInputs(
        layers=mlcanopy_layers(inst, p),
        forcing_bef=_patch_record(window.bef, p),
        forcing_cur=_patch_record(window.cur, p),
        forcing_next=_patch_record(window.next, p),
        time_bef=times[0],
        time_cur=times[1],
        time_next=times[2],
        zref=inst.zref_forcing[p],
        solar_zen=inst.solar_zen_forcing[p],
        dlai=layer("dlai_profile"),
        dsai=layer("dsai_profile"),
        dpai=layer("dpai_profile"),
        albsoib=inst.albsoib_soil[p, 1:],
        albsoid=inst.albsoid_soil[p, 1:],
        lai=inst.lai_canopy[p],
        sai=inst.sai_canopy[p],
        tacclim=inst.tacclim_forcing[p],
        ztop=inst.ztop_canopy[p],
        o2ref=inst.o2ref_forcing[p],
        qflx_rain=inst.qflx_rain_forcing[p],
        qflx_snow=inst.qflx_snow_forcing[p],
        rhg=inst.rhg_soil[p],
        soilres=inst.soilres_soil[p],
        soil_t=inst.soil_t_soil[p],
        soil_dz=inst.soil_dz_soil[p],
        soil_tk=inst.soil_tk_soil[p],
        cpleaf=layer("cpleaf_profile"),
        lsc=layer("lsc_profile"),
        psis=inst.psis_soil[p],
    )
