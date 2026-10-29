"""
JAX translation of MLCanopyFluxesMod Fortran module.

Orchestrator for all multilayer canopy physics within a single CLM
timestep.  Manages the filter of vegetated patches, initialisation,
atmospheric-forcing interpolation, the Runge-Kutta loop, and the
pass-back to CLM output variables.

Public routine
--------------
- :func:`MLCanopyFluxes`: main driver.

Private helpers
---------------
- :func:`_GetCLMVar`: stub — copy CLM variables to mlcanopy container.
- :func:`_MLTimeStepFluxIntegration`: stub — accumulate ML-step fluxes.
- :func:`_CanopyFluxesDiagnostics`: stub — canopy-wide diagnostics.

Original Fortran module: MLCanopyFluxesMod
Fortran lines 1-400
"""

from __future__ import annotations

from functools import partial
from typing import Any, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_varcon import grav, spval  # noqa: F401
from clm_src_main.clm_varctl import iulog  # noqa: F401
from clm_src_main.clm_varpar import inir, ivis  # noqa: F401
from clm_src_main.PatchType import patch  # noqa: F401
from clm_src_utils.clm_time_manager import StepTime
from clm_src_utils.spmdMod import masterproc  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import (  # noqa: F401
    CANOPY_ENERGY_1,
    CANOPY_ENERGY_2,
    CANOPY_ENERGY_3,
    CANOPY_ENERGY_4,
    PLANT_AREA,
    mlcanopy_type,
    record_balance_error,
)

# Physics modules (translated elsewhere)
from multilayer_canopy.MLCanopyNitrogenProfileMod import CanopyNitrogenProfile  # noqa: F401
from multilayer_canopy.MLCanopyParamsMod import CanopyParams, patch_setup
from multilayer_canopy.MLCanopyTurbulenceMod import CanopyTurbulence  # noqa: F401
from multilayer_canopy.MLCanopyWaterMod import (  # noqa: F401
    CanopyEvaporation,
    CanopyInterception,
    CanopyWettedFraction,
)
from multilayer_canopy.MLclm_varcon import mmh2o, rgas  # noqa: F401
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun, nleaf  # noqa: F401
from multilayer_canopy.MLFluxProfileSolutionMod import FluxProfileSolution  # noqa: F401
from multilayer_canopy.MLGetAtmForcingMod import (  # noqa: F401
    AtmForcingRecord,
    AtmForcingWindow,
    GetAtmForcing,
    clm_forcing_to_ml,
)
from multilayer_canopy.MLinitVerticalMod import (  # noqa: F401
    getPADparameters,
    initVerticalProfiles,
    initVerticalStructure,
)
from multilayer_canopy.MLLayersMod import keep, layers, pad_with_top
from multilayer_canopy.MLLeafBoundaryLayerMod import (
    LeafBoundaryLayerBoth,
)  # noqa: F401
from multilayer_canopy.MLLeafHeatCapacityMod import LeafHeatCapacity  # noqa: F401
from multilayer_canopy.MLLeafPhotosynthesisMod import LeafPhotosynthesis  # noqa: F401
from multilayer_canopy.MLLongwaveRadiationMod import LongwaveRadiation  # noqa: F401
from multilayer_canopy.MLMathToolsMod import is_concrete
from multilayer_canopy.MLPlantHydraulicsMod import (  # noqa: F401
    LeafWaterPotential,
    PlantResistance,
    SoilResistance,
)
from multilayer_canopy.MLRungeKuttaMod import RungeKuttaIni, RungeKuttaUpdate  # noqa: F401
from multilayer_canopy.MLSolarRadiationMod import SolarRadiation  # noqa: F401
from multilayer_canopy.MLWaterVaporMod import LatVap  # noqa: F401

# ---------------------------------------------------------------------------
# JIT-compiled bef-state copy helper
# ---------------------------------------------------------------------------


def _float_if_concrete(x):
    """``float(x)`` in a normal run (Python-float arithmetic, as before); ``x`` itself when
    it is traced (a chunk of steps compiled with jax.jit)."""
    return float(x) if is_concrete(x) else x


@partial(jax.jit, static_argnums=(0,))
def _copy_bef_state(
    filter_mlcan: tuple,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """Copy current-state arrays to their ``*_bef`` counterparts, layers 1..ncan (data;
    padding kept).

    Mirrors Fortran lines 272-285 of ``MLCanopyFluxes``.
    """
    inst = mlcanopy_inst
    for p in filter_mlcan:
        use = layers(inst, p).in_use

        def _copy(bef, cur):
            return bef.at[p, 1:].set(keep(use, cur[p, 1:], bef[p, 1:]))

        inst = inst._replace(
            tg_bef_soil=inst.tg_bef_soil.at[p].set(inst.tg_soil[p]),
            tair_bef_profile=_copy(inst.tair_bef_profile, inst.tair_profile),
            eair_bef_profile=_copy(inst.eair_bef_profile, inst.eair_profile),
            cair_bef_profile=_copy(inst.cair_bef_profile, inst.cair_profile),
            h2ocan_bef_profile=_copy(inst.h2ocan_bef_profile, inst.h2ocan_profile),
            tleaf_bef_leaf=_copy(inst.tleaf_bef_leaf, inst.tleaf_leaf),
            lwp_bef_leaf=_copy(inst.lwp_bef_leaf, inst.lwp_leaf),
        )
    return inst


# ---------------------------------------------------------------------------
# Public: MLCanopyFluxes
# ---------------------------------------------------------------------------


def MLCanopyFluxes(
    bounds: Any,
    num_exposedvegp: int,
    filter_exposedvegp: Sequence[int],
    atm2lnd_inst: Any,
    canopystate_inst: Any,
    soilstate_inst: Any,
    temperature_inst: Any,
    waterstatebulk_inst: Any,
    waterfluxbulk_inst: Any,
    energyflux_inst: Any,
    frictionvel_inst: Any,
    surfalb_inst: Any,
    solarabs_inst: Any,
    mlcanopy_inst: mlcanopy_type,
    wateratm2lndbulk_inst: Any,
    waterdiagnosticbulk_inst: Any,
    step_time: StepTime,
    forcing_bef: AtmForcingRecord,
    forcing_next: AtmForcingRecord,
    initialize: bool,
    grid: "GridInfo | None" = None,
    params: "CanopyParams | None" = None,
    *,
    opts: MLOptions,
    site=None,
) -> mlcanopy_type:
    """
    Compute all multilayer canopy fluxes for one CLM timestep.

    Mirrors Fortran subroutine ``MLCanopyFluxes`` (lines 48-400).

    This is the top-level driver.  It:

    1. Resolves CLM time information (``nstep``, ``dtime_clm``,
       calendar days).
    2. Builds the ``filter_mlcan`` patch filter from the exposed-veg
       filter (Fortran lines 152-160).
    3. On the first CLM timestep (``initialize``), initialises PAD
       parameters, vertical structure, vertical profiles, and
       Runge-Kutta coefficients (Fortran lines 162-198).
    4. Copies CLM state into ``mlcanopy_inst`` via ``_GetCLMVar``
       (Fortran line 201).
    5. Forms the forcing window (previous, current, next CLM step) from
       ``forcing_bef``, the current forcing from ``_GetCLMVar`` and
       ``forcing_next`` (Fortran stored these as ``*_bef/_cur/_next_forcing``
       and seeded ``*_bef`` on the first timestep, lines 204-218).
    6. Updates the leaf/stem area profiles for current LAI/SAI and
       checks total PAI conservation (Fortran lines 220-243).
    7. Computes soil hydraulic resistance, plant resistance, leaf heat
       capacity, and soil relative humidity (Fortran lines 245-258).
    8. Loops over ``num_ml_steps`` sub-steps (Fortran lines 261-330):

       a. Saves previous-step state variables.
       b. Interpolates atmospheric forcing to the ML timestep.
       c. Calls physics in order:

          - :func:`SolarRadiation`
          - :func:`CanopyNitrogenProfile`
          - Inner Runge-Kutta loop (``irk = 1, nrk_steps+1``):

            - :func:`CanopyWettedFraction`
            - :func:`LongwaveRadiation`
            - Net radiation (inline, Fortran lines 301-307)
            - :func:`CanopyTurbulence`
            - :func:`LeafBoundaryLayer` (sun + shade)
            - :func:`LeafPhotosynthesis` (sun + shade)
            - :func:`FluxProfileSolution`
            - :func:`LeafWaterPotential` (sun + shade)
            - :func:`CanopyInterception` + :func:`CanopyEvaporation`
            - :func:`RungeKuttaUpdate` (for non-Euler steps)

       d. Accumulates fluxes over ML sub-steps.

    9. (Fortran lines 332-343 saved the current forcing as ``*_bef`` for the
       next CLM timestep; the caller now supplies ``forcing_bef`` instead.)
    10. Calls :func:`_CanopyFluxesDiagnostics`.
    11. Merges sun/shade leaf temperature and water potential to a
        layer-mean prognostic state (Fortran lines 355-375).
    12. Copies selected fields back to CLM output variables when
        ``mlcan_to_clm == 1`` (Fortran lines 378-396).

    **Atmospheric forcing interpolation** (Fortran lines 105-130):

    .. code-block:: none

        met_type == 0: no interpolation; calday = 0
        met_type == 2: 2-point (end of interval); NOT VALID in this
                       implementation (endrun)
        met_type == 3: 3-point (centred); calday_cur = 0.5*(end+beg)

    **Runge-Kutta sub-stepping** (Fortran lines 265-271):

    .. code-block:: none

        runge_kutta_type == 10: Euler, nrk_steps = 0
        runge_kutta_type >= 20: nrk_steps = runge_kutta_type // 10

    **Sun/shade merge** (Fortran lines 361-375):

    .. code-block:: none

        if dpai(ic) > 0:
            tleaf_sun_merged = tleaf_sun*fracsun + tleaf_sha*(1-fracsun)
            tleaf_sha        = tleaf_sun_merged
            lwp_sun_merged   = lwp_sun*fracsun  + lwp_sha*(1-fracsun)
            lwp_sha          = lwp_sun_merged

    Args:
        bounds: CLM decomposition bounds (``begp``, ``endp``).
        num_exposedvegp: Number of non-snow-covered veg patches.
        filter_exposedvegp: CLM patch filter (1-based values).
        atm2lnd_inst: CLM atmosphere-to-land instance.
        canopystate_inst: CLM canopy state instance (LAI/SAI).
        soilstate_inst: CLM soil state instance.
        temperature_inst: CLM temperature instance.
        waterstatebulk_inst: CLM bulk water state instance.
        waterfluxbulk_inst: CLM bulk water flux instance.
        energyflux_inst: CLM energy flux instance (output receiver).
        frictionvel_inst: CLM friction velocity instance (output receiver).
        surfalb_inst: CLM surface albedo instance.
        solarabs_inst: CLM solar absorbed instance (output receiver).
        mlcanopy_inst: Multilayer canopy container; updated in-place
            over the timestep and returned.
        wateratm2lndbulk_inst: CLM water forcing instance.
        waterdiagnosticbulk_inst: CLM water diagnostic instance
            (output receiver).
        step_time: Time information for this CLM time step (step number,
            step length, calendar day at the beginning and end of the step).
            Replaces the Fortran calls to the time manager (lines 155-165).
        forcing_bef: Forcing at the previous CLM time step (the current step at
            the first step), for time interpolation. Replaces the Fortran
            ``*_bef_forcing`` fields carried in ``mlcanopy_type``.
        forcing_next: Forcing at the next CLM time step (replaces
            ``*_next_forcing``, which Fortran ``TowerMetNext`` filled).
        initialize: True on the first time step of a run (``"only"``: initialize, then
            return without the step's physics, which a batched first step runs with
            ``initialize=False``; Phase 6 step 6): set up the PAD
            parameters and the vertical structure and profiles. Fortran
            detected this from an unset ``zref`` (``zref == spval``); the
            caller now says so explicitly (False when continuing from a
            restart or in differentiable mode).
        grid: Patch index and canopy layer indices as Python ints. Built from
            ``mlcanopy_inst`` when omitted; pass it when the state is traced
            (e.g. under ``jax.grad``), where the indices cannot be read.
        params: Parameters of the patch's PFT (:class:`CanopyParams`); built from the PFT
            tables when omitted (pass it to differentiate with respect to parameters).

    Returns:
        Updated :class:`mlcanopy_type` holding end-of-timestep state
        and CLM-timestep-mean fluxes.
    """
    nlevmlcan = mlcanopy_inst.dpai_profile.shape[1] - 1  # layer slots (model.yaml max_layers)
    met_type, mlcan_to_clm = opts.met_type, opts.mlcan_to_clm

    # ------------------------------------------------------------------
    # CLM time information — Fortran lines 155-165
    # (passed in as step_time instead of read from the time manager)
    # ------------------------------------------------------------------
    nstep = step_time.nstep
    dtime_clm = float(step_time.dtime)

    # Python floats in a normal run; traced values when a chunk of steps is compiled
    curr_calday_end = _float_if_concrete(step_time.calday)
    curr_calday_beg = _float_if_concrete(step_time.calday_beg)

    # Interpolation calendar days — Fortran lines 105-130
    if met_type == 0:
        calday_interp_cur = 0.0
        calday_interp_bef = 0.0
        calday_interp_next = 0.0
    elif met_type == 2:
        endrun(msg=" ERROR: met_type not valid")
        calday_interp_cur = curr_calday_end
        calday_interp_bef = calday_interp_cur - dtime_clm / 86400.0
        calday_interp_next = 0.0
    elif met_type == 3:
        calday_interp_cur = 0.5 * (curr_calday_end + curr_calday_beg)
        calday_interp_bef = calday_interp_cur - dtime_clm / 86400.0
        calday_interp_next = calday_interp_cur + dtime_clm / 86400.0
    else:
        endrun(msg=" ERROR: MLCanopyFluxes: met_type not valid")
        calday_interp_cur = calday_interp_bef = calday_interp_next = 0.0

    # Number of ML sub-steps — Fortran line 132
    num_ml_steps: int = int(dtime_clm / opts.dtime_ml)

    # ------------------------------------------------------------------
    # Build filter for patches to process — Fortran lines 152-160
    # ------------------------------------------------------------------
    num_mlcan: int = 0
    _filter_mlcan_list: List[int] = []
    for fp in range(1, num_exposedvegp + 1):
        p = int(filter_exposedvegp[fp - 1])
        num_mlcan += 1
        _filter_mlcan_list.append(p)
    filter_mlcan: tuple = tuple(_filter_mlcan_list)  # tuple required for jax.jit static_argnums

    # ------------------------------------------------------------------
    # First-timestep initialisation — Fortran lines 162-198
    # (Fortran triggered it when zref == spval; the caller passes `initialize`)
    # ------------------------------------------------------------------
    if initialize:

        if masterproc:
            print("Attempting to initialize multilayer canopy pad parameters .....")
        mlcanopy_inst = getPADparameters(num_mlcan, filter_mlcan, mlcanopy_inst)
        if masterproc:
            print("Successfuly initialized multilayer canopy pad parameters")

        if masterproc:
            print("Attempting to initialize multilayer canopy vertical structure .....")
        mlcanopy_inst = initVerticalStructure(
            bounds,
            num_mlcan,
            filter_mlcan,
            canopystate_inst,
            frictionvel_inst,
            mlcanopy_inst,
            opts=opts,
        )
        mlcanopy_inst = initVerticalProfiles(
            num_mlcan, filter_mlcan, atm2lnd_inst, wateratm2lndbulk_inst, mlcanopy_inst
        )
        if masterproc:
            print("Successfully initialized multilayer canopy vertical structure")
        if initialize == "only":  # the batched first step runs the physics (Phase 6 step 6)
            return mlcanopy_inst

    # Runge-Kutta coefficients — Fortran line 197 (computed on the first step
    # and kept in module state; they depend only on runge_kutta_type, so they are
    # computed here instead). Constants: evaluated eagerly even when this function is
    # traced (RungeKuttaUpdate uses them as numpy values).
    with jax.ensure_compile_time_eval():
        ark, brk, crk = RungeKuttaIni(opts)

    # ------------------------------------------------------------------
    # Copy CLM variables to mlcanopy container — Fortran line 201
    # ------------------------------------------------------------------
    mlcanopy_inst, forcing_cur = _GetCLMVar(
        nstep,
        dtime_clm,
        curr_calday_beg,
        num_mlcan,
        filter_mlcan,
        atm2lnd_inst,
        soilstate_inst,
        temperature_inst,
        surfalb_inst,
        wateratm2lndbulk_inst,
        mlcanopy_inst,
        site=site,
    )
    # ------------------------------------------------------------------
    # Forcing window for time interpolation — replaces the Fortran
    # *_bef/_cur/_next_forcing fields (and the *_bef seeding, lines 204-218)
    # ------------------------------------------------------------------
    forcing_window = AtmForcingWindow(bef=forcing_bef, cur=forcing_cur, next=forcing_next)

    # ------------------------------------------------------------------
    # Patch and canopy layer indices as Python ints (static under jax tracing).
    # The patch is the unit of computation; several patches are batched with
    # vmap (plan Phase 4), not looped over here.
    # ------------------------------------------------------------------
    if num_mlcan != 1:
        endrun(msg=" ERROR: MLCanopyFluxes: one patch per call (batch patches with vmap)")
    # Layer indices and C3/C4 flag (static) and the PFT's parameters (traced), unless the
    # caller passes them (it must when the state or the parameters are traced).
    if grid is None or params is None:
        _grid, _params = patch_setup(filter_mlcan[0], mlcanopy_inst)
        grid = _grid if grid is None else grid
        params = _params if params is None else params

    # ------------------------------------------------------------------
    # Update leaf/stem area profiles for current LAI/SAI — Fortran lines 220-243
    # ------------------------------------------------------------------
    lai = mlcanopy_inst.lai_canopy
    sai = mlcanopy_inst.sai_canopy
    sai_floor = mlcanopy_inst.sai_floor_canopy
    if sai_floor is None:  # a container from before the field existed
        sai_floor = jnp.zeros_like(sai)
    dlai = mlcanopy_inst.dlai_profile
    dsai = mlcanopy_inst.dsai_profile
    dpai = mlcanopy_inst.dpai_profile

    for p in filter_mlcan:
        lai_val = canopystate_inst.elai_patch[p]
        sai_val = canopystate_inst.esai_patch[p]

        # Layers 1..ncan (data; padding kept), vectorized over the layer slots
        lay = layers(mlcanopy_inst, p)
        _dlai_sl = pad_with_top(lay, mlcanopy_inst.dlai_frac_profile[p, 1:]) * lai_val
        _dsai_sl = pad_with_top(lay, mlcanopy_inst.dsai_frac_profile[p, 1:]) * sai_val
        # Not Fortran (Phase 6b, issue 14): every leafy layer keeps at least
        # opts.layer_pai_floor of plant area, the missing part added as stem area, and the
        # canopy's stem area includes it. CLM-ml's 9-angle diffuse transmittance (SolarRadiation)
        # exceeds 1 below ~0.005 m2/m2, which makes a layer's net longwave grow with its
        # temperature (Norman longwave) and the leaf temperature run away. Layers at or above
        # the floor are unchanged (x + 0.0), so CLM-ml v2 results are kept where no layer is
        # that thin (CHATS7). Value: CLM-ml v1's dpai_min (m2/m2).
        _add = jnp.where(
            lay.leafy, jnp.maximum(opts.layer_pai_floor - (_dlai_sl + _dsai_sl), 0.0), 0.0
        )
        _dsai_sl = _dsai_sl + _add
        _sai_floor = jnp.sum(_add)
        sai_val = sai_val + _sai_floor
        lai = lai.at[p].set(lai_val)
        sai = sai.at[p].set(sai_val)
        sai_floor = sai_floor.at[p].set(_sai_floor)
        dlai = dlai.at[p, 1:].set(keep(lay.in_use, _dlai_sl, dlai[p, 1:]))
        dsai = dsai.at[p, 1:].set(keep(lay.in_use, _dsai_sl, dsai[p, 1:]))
        dpai = dpai.at[p, 1:].set(keep(lay.in_use, _dlai_sl + _dsai_sl, dpai[p, 1:]))

        # PAI conservation check — Fortran lines 239-242; recorded, checked on the host
        totpai = jnp.sum(jnp.where(lay.in_use, dpai[p, 1:], 0.0))
        mlcanopy_inst = record_balance_error(
            mlcanopy_inst, p, PLANT_AREA, totpai - (lai_val + sai_val)
        )

    mlcanopy_inst = mlcanopy_inst._replace(
        lai_canopy=lai,
        sai_canopy=sai,
        sai_floor_canopy=sai_floor,
        dlai_profile=dlai,
        dsai_profile=dsai,
        dpai_profile=dpai,
    )

    # ------------------------------------------------------------------
    # Plant hydraulics and leaf heat capacity — Fortran lines 245-257
    # ------------------------------------------------------------------
    mlcanopy_inst = SoilResistance(
        num_mlcan,
        filter_mlcan,
        soilstate_inst,
        waterstatebulk_inst,
        mlcanopy_inst,
        params,
        site=site,
    )
    mlcanopy_inst = PlantResistance(num_mlcan, filter_mlcan, mlcanopy_inst, params)
    mlcanopy_inst = LeafHeatCapacity(num_mlcan, filter_mlcan, mlcanopy_inst, params)

    # Soil surface relative humidity — Fortran lines 259-262
    # Pre-materialise patch.column as numpy so int() is concrete even inside
    # jax.grad tracing (patch hierarchy is invariant for a fixed site).
    _patch_col_np = np.asarray(patch.column)
    rhg = mlcanopy_inst.rhg_soil
    for p in filter_mlcan:
        c = int(_patch_col_np[p])
        smp1 = soilstate_inst.smp_l_col[c, 1]  # mm
        tsoi1 = temperature_inst.t_soisno_col[c, 1]  # K
        rhg_val = jnp.exp(grav * mmh2o * smp1 * 1.0e-3 / (rgas * tsoi1))
        rhg = rhg.at[p].set(rhg_val)
    mlcanopy_inst = mlcanopy_inst._replace(rhg_soil=rhg)

    # ------------------------------------------------------------------
    # Flux accumulators — Fortran lines 57-64 (local arrays)
    # ------------------------------------------------------------------
    nvar1d = 23
    nvar2d = 14
    nvar3d = 12
    endp = bounds.endp
    flux_accumulators = (
        jnp.zeros((endp + 1, nvar1d)),
        jnp.zeros((endp + 1, nlevmlcan + 2, nvar2d)),
        jnp.zeros((endp + 1, nlevmlcan + 1, nleaf + 1, nvar3d)),
    )

    # ------------------------------------------------------------------
    # Runge-Kutta configuration — Fortran lines 265-271
    # ------------------------------------------------------------------
    _runge_kutta_type = opts.runge_kutta_type
    if _runge_kutta_type == 10:
        nrk_steps = 0
    elif _runge_kutta_type >= 20:
        nrk_steps = _runge_kutta_type // 10
    else:
        nrk_steps = 0

    if masterproc and is_concrete(nstep) and nstep == 1:
        print(
            f"MLCanopyFluxes: starting ML sub-steps (num_ml_steps={num_ml_steps}, "
            f"nrk_steps={nrk_steps}, num_mlcan={num_mlcan})",
            flush=True,
        )

    # Keep heartbeat output bounded while still proving forward progress.
    progress_stride = max(1, num_ml_steps // 10)

    # ==================================================================
    # Multilayer canopy time-stepping loop — Fortran lines 261-330
    # ==================================================================
    # One sub-step is one function, _substep (physics + flux accumulation). It is
    # driven by a Python loop in a normal run, and by lax.scan (with
    # jax.checkpoint, so backward memory does not grow with the number of
    # sub-steps) when the state is being traced, e.g. under jax.grad. A lax.scan in
    # a normal run would recompile at every CLM step (the forcing it closes over
    # changes); compiling a whole run is plan Phase 4.
    # ==================================================================

    # Current sub-step, for warning messages only (0 inside lax.scan)
    nstep_ml = 0

    def _physics_step_fn(inst, calday_ml):
        """Pure (inst, calday_ml) → inst for one ML sub-step."""
        # Save previous-step state — Fortran lines 272-285
        inst = _copy_bef_state(filter_mlcan, inst)
        # Atmospheric forcing — Fortran line 287
        # calday_ml is a JAX traced scalar inside lax.scan;
        # TimeInterpolation3 uses jnp.where so this is safe.
        inst = GetAtmForcing(
            calday_interp_bef,
            calday_interp_cur,
            calday_interp_next,
            calday_ml,
            num_mlcan,
            filter_mlcan,
            inst,
            forcing_window,
            grid=grid,
            opts=opts,
        )
        # Solar radiation — Fortran line 290
        inst = SolarRadiation(bounds, num_mlcan, filter_mlcan, inst, params, grid=grid, opts=opts)
        # Nitrogen profile — Fortran line 293
        inst = CanopyNitrogenProfile(num_mlcan, filter_mlcan, inst, params, grid, opts=opts)

        # Runge-Kutta stages — Fortran lines 310-328: nrk_steps stages, each followed by
        # a state update, and a final physics pass. One function, _rk_stage, is a stage:
        # a Python loop drives it in normal runs; lax.scan (one traced copy of the
        # physics instead of nrk_steps + 1) when traced, which keeps the gradient small
        # enough to compile.
        def _rk_stage(inst, irk):
            inst = CanopyWettedFraction(num_mlcan, filter_mlcan, inst)
            inst = LongwaveRadiation(
                bounds, num_mlcan, filter_mlcan, inst, params, grid=grid, opts=opts
            )
            inst = inst._replace(
                rnleaf_leaf=(
                    inst.swleaf_leaf[..., ivis] + inst.swleaf_leaf[..., inir] + inst.lwleaf_leaf
                ),
                rnsoi_soil=(inst.swsoi_soil[:, ivis] + inst.swsoi_soil[:, inir] + inst.lwsoi_soil),
            )
            inst = CanopyTurbulence(
                nstep_ml, num_mlcan, filter_mlcan, inst, grid=grid, nstep=nstep, opts=opts
            )
            # Both sun and shade in one fused GPU dispatch (2× fewer round-trips)
            inst = LeafBoundaryLayerBoth(num_mlcan, filter_mlcan, inst, params, opts=opts)
            inst = LeafPhotosynthesis(
                num_mlcan,
                filter_mlcan,
                isun,
                inst,
                grid=grid,
                params=params,
                opts=opts,
            )
            inst = LeafPhotosynthesis(
                num_mlcan,
                filter_mlcan,
                isha,
                inst,
                grid=grid,
                params=params,
                opts=opts,
            )
            inst = FluxProfileSolution(num_mlcan, filter_mlcan, inst, grid=grid, opts=opts)
            inst = LeafWaterPotential(num_mlcan, filter_mlcan, isun, inst, params, opts=opts)
            inst = LeafWaterPotential(num_mlcan, filter_mlcan, isha, inst, params, opts=opts)
            inst = CanopyInterception(num_mlcan, filter_mlcan, inst, opts=opts)
            inst = CanopyEvaporation(num_mlcan, filter_mlcan, inst, opts=opts)
            if nrk_steps > 0:  # no-op after the last stage (irk > nrk)
                inst = RungeKuttaUpdate(
                    irk,
                    ark,
                    brk,
                    crk,
                    num_mlcan,
                    filter_mlcan,
                    inst,
                    grid=grid,
                )
            return inst

        stages = range(1, nrk_steps + 2)
        if _traced:
            inst, _ = jax.lax.scan(
                lambda st, irk: (_rk_stage(st, irk), None), inst, jnp.arange(1, nrk_steps + 2)
            )
        else:
            for irk in stages:
                inst = _rk_stage(inst, irk)
        return inst

    def _substep(carry, calday_ml):
        """One ML sub-step: physics, then accumulate fluxes — Fortran line 372."""
        inst, accumulators = carry
        inst = _physics_step_fn(inst, calday_ml)
        accumulators = _MLAccumulateFluxes(num_mlcan, filter_mlcan, *accumulators, inst)
        return (inst, accumulators), None

    # Calendar day of each ML sub-step — Fortran lines 263-270
    _dtime_ml = opts.dtime_ml
    _offset = 0.0 if met_type in (0, 2) else 0.5  # met_type 3: mid-sub-step
    caldays_ml = [
        curr_calday_beg + (float(k) - _offset) * (_dtime_ml / 86400.0)
        for k in range(1, num_ml_steps + 1)
    ]

    carry = (mlcanopy_inst, flux_accumulators)
    _traced = not is_concrete(
        *jax.tree_util.tree_leaves((mlcanopy_inst, forcing_bef, forcing_cur, forcing_next))
    )
    if _traced:
        import os as _os

        _scan_body = _substep
        if _os.environ.get("CLM_ML_NO_CHECKPOINT", "0") != "1":
            _scan_body = jax.checkpoint(_substep)
        carry, _ = jax.lax.scan(_scan_body, carry, jnp.array(caldays_ml, dtype=jnp.float64))
    else:
        for nstep_ml, calday_ml in enumerate(caldays_ml, start=1):
            if (
                masterproc
                and is_concrete(nstep)
                and nstep == 1
                and (nstep_ml == 1 or nstep_ml == num_ml_steps or (nstep_ml % progress_stride == 0))
            ):
                print(f"MLCanopyFluxes: sub-step {nstep_ml}/{num_ml_steps}", flush=True)
            carry, _ = _substep(carry, calday_ml)
    mlcanopy_inst, flux_accumulators = carry

    # Scale accumulated sums by 1/num_ml_steps and write back to inst
    *_, mlcanopy_inst = _MLScaleAndWriteBack(
        num_ml_steps, num_mlcan, filter_mlcan, *flux_accumulators, mlcanopy_inst
    )

    # End multilayer time-stepping loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Canopy-level diagnostics — Fortran line 346
    # ------------------------------------------------------------------
    mlcanopy_inst = _CanopyFluxesDiagnostics(num_mlcan, filter_mlcan, mlcanopy_inst, opts)

    # ------------------------------------------------------------------
    # Merge sun/shade leaf temperature and water potential — Fortran lines 355-375
    # State is prognostic; sun/shade fraction changes between CLM steps,
    # so merge to a layer-mean before the next timestep.
    # ------------------------------------------------------------------
    tleaf = mlcanopy_inst.tleaf_leaf
    tleaf_hist = mlcanopy_inst.tleaf_hist_leaf
    lwp = mlcanopy_inst.lwp_leaf
    lwp_hist = mlcanopy_inst.lwp_hist_leaf

    for p in filter_mlcan:
        _sl = slice(1, None)  # every slot; layers above ncan kept below
        _use = layers(mlcanopy_inst, p).in_use

        # Save sun/shade values for history files (bulk copy, uses JAX .at[])
        tleaf_hist = tleaf_hist.at[p, _sl, :].set(
            keep(_use, tleaf[p, _sl, :], tleaf_hist[p, _sl, :])
        )
        lwp_hist = lwp_hist.at[p, _sl, :].set(keep(_use, lwp[p, _sl, :], lwp_hist[p, _sl, :]))

        # Vectorized merge to layer mean — Fortran lines 367-374
        # Pure JAX: no np.asarray() D->H sync; all ops stay on device.
        dpai_p = mlcanopy_inst.dpai_profile[p, _sl]  # shape (ncan,)
        fracsun_p = mlcanopy_inst.fracsun_profile[p, _sl]  # shape (ncan,)
        tleaf_p = tleaf[p, _sl, :]  # shape (ncan, nleaf+1)
        lwp_p = lwp[p, _sl, :]  # shape (ncan, nleaf+1)

        has_pai = (dpai_p > 0.0) & _use  # (slots,)

        t_sun = tleaf_p[:, isun]
        t_sha = tleaf_p[:, isha]
        t_mean = t_sun * fracsun_p + t_sha * (1.0 - fracsun_p)
        tleaf_p_new = tleaf_p.at[:, isun].set(jnp.where(has_pai, t_mean, t_sun))
        tleaf_p_new = tleaf_p_new.at[:, isha].set(jnp.where(has_pai, t_mean, t_sha))

        l_sun = lwp_p[:, isun]
        l_sha = lwp_p[:, isha]
        l_mean = l_sun * fracsun_p + l_sha * (1.0 - fracsun_p)
        lwp_p_new = lwp_p.at[:, isun].set(jnp.where(has_pai, l_mean, l_sun))
        lwp_p_new = lwp_p_new.at[:, isha].set(jnp.where(has_pai, l_mean, l_sha))

        tleaf = tleaf.at[p, _sl, :].set(keep(_use, tleaf_p_new, tleaf_p))
        lwp = lwp.at[p, _sl, :].set(keep(_use, lwp_p_new, lwp_p))

    mlcanopy_inst = mlcanopy_inst._replace(
        tleaf_leaf=tleaf,
        tleaf_hist_leaf=tleaf_hist,
        lwp_leaf=lwp,
        lwp_hist_leaf=lwp_hist,
    )

    # ------------------------------------------------------------------
    # Copy to CLM output variables (when mlcan_to_clm == 1) — Fortran lines 378-396
    # ------------------------------------------------------------------
    if mlcan_to_clm == 1:
        for p in filter_mlcan:
            for ib in (ivis, inir):
                surfalb_inst.albd_patch[p, ib] = 0.0
                surfalb_inst.albi_patch[p, ib] = 0.0
            energyflux_inst.taux_patch[p] = 0.0
            energyflux_inst.tauy_patch[p] = 0.0
            energyflux_inst.eflx_lh_tot_patch[p] = mlcanopy_inst.lhflx_canopy[p]
            energyflux_inst.eflx_sh_tot_patch[p] = mlcanopy_inst.shflx_canopy[p]
            energyflux_inst.eflx_lwrad_out_patch[p] = mlcanopy_inst.lwup_canopy[p]
            waterfluxbulk_inst.qflx_evap_tot_patch[p] = mlcanopy_inst.etflx_canopy[p] * mmh2o
            frictionvel_inst.fv_patch[p] = mlcanopy_inst.ustar_canopy[p]
            frictionvel_inst.u10_clm_patch[p] = 0.0
            temperature_inst.t_ref2m_patch[p] = 0.0
            waterdiagnosticbulk_inst.q_ref2m_patch[p] = 0.0
            solarabs_inst.fsa_patch[p] = (
                mlcanopy_inst.swveg_canopy[p, ivis]
                + mlcanopy_inst.swveg_canopy[p, inir]
                + mlcanopy_inst.swsoi_soil[p, ivis]
                + mlcanopy_inst.swsoi_soil[p, inir]
            )

    return mlcanopy_inst


def _GetCLMVar(
    nstep: int,
    dtime_clm: float,
    calday_beg: float,
    num_filter: int,
    filter: List[int],
    atm2lnd_inst: Any,
    soilstate_inst: Any,
    temperature_inst: Any,
    surfalb_inst: Any,
    wateratm2lndbulk_inst: Any,
    mlcanopy_inst: mlcanopy_type,
    site=None,
) -> "tuple[mlcanopy_type, AtmForcingRecord]":
    """
    Copy CLM variables into the multilayer canopy container.

    Mirrors Fortran subroutine ``GetCLMVar`` (lines 1-120).

    All CLM variables are copied verbatim into the corresponding
    ``mlcanopy_inst`` fields.  The only arithmetic beyond indexing is:

    - Wind speed: ``uref_cur(p) = sqrt(forc_u(g)^2 + forc_v(g)^2)``
      (resultant from east and north components).
    - CO2 unit conversion:
      ``co2ref_cur(p) = forc_pco2(g) / forc_pbot(c) * 1e6``
      (Pa → umol/mol).
    - O2 unit conversion:
      ``o2ref(p) = forc_po2(g) / forc_pbot(c) * 1e3``
      (Pa → mmol/mol).
    - First snow/soil layer depth:
      ``soil_dz(p) = z(c, snl(c)+1) - zi(c, snl(c))``
      (layer mid-point depth minus interface depth above it).

    Solar zenith angle (Fortran lines 92-110):

    The zenith angle is computed for the *beginning* of the CLM
    timestep, i.e. at calendar day ``caldaym1 = calday(nstep-1)``.
    The call chain is::

        caldaym1  → shr_orb_decl  → declinm1
        (caldaym1, lat, lon, declinm1) → shr_orb_cosz → coszen
        solar_zen(p) = acos(max(0.01, coszen))

    The ``max(0.01, coszen)`` floor prevents ``acos`` from returning
    values beyond ``pi/2`` (i.e. clamps the sun no lower than ~89.4°
    below horizontal), matching the Fortran exactly.

    .. warning::
        ``soil_t``, ``soil_dz``, and ``soil_tk`` (Fortran lines 84-88)
        are required for the soil heat flux calculation.  When run
        **coupled to CLM**, ``t_soisno`` is updated by CLM's own
        ``SoilTemperatureMod``, *not* by the multilayer canopy's
        ``MLSoilTemperatureMod``.  Therefore the soil temperature seen
        here is independent of the multilayer canopy soil energy
        balance.

    Args:
        nstep: Current CLM timestep number.
        dtime_clm: CLM timestep duration (s).
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
        atm2lnd_inst: CLM atmosphere-to-land instance.  Required
            fields:

            - ``forc_u_grc[g]``, ``forc_v_grc[g]`` — wind components (m/s)
            - ``forc_pco2_grc[g]`` — CO2 partial pressure (Pa)
            - ``forc_po2_grc[g]`` — O2 partial pressure (Pa)
            - ``forc_solai_grc[g, ib]`` — diffuse SW by waveband (W/m2)
            - ``forc_solad_downscaled_col[c, ib]`` — direct SW (W/m2)
            - ``forc_t_downscaled_col[c]`` — air temperature (K)
            - ``forc_pbot_downscaled_col[c]`` — air pressure (Pa)
            - ``forc_lwrad_downscaled_col[c]`` — longwave (W/m2)

        soilstate_inst: CLM soil state instance.  Required fields:

            - ``soilresis_col[c]`` — soil evaporative resistance (s/m)
            - ``thk_col[c, j]`` — thermal conductivity (W/m/K)

        temperature_inst: CLM temperature instance.  Required fields:

            - ``t_a10_patch[p]`` — 10-day mean 2-m air temperature (K)
            - ``t_soisno_col[c, j]`` — soil/snow temperature (K)

        surfalb_inst: CLM surface albedo instance.  Required fields:

            - ``albgrd_col[c, ib]`` — direct beam ground albedo (-)
            - ``albgri_col[c, ib]`` — diffuse ground albedo (-)

        wateratm2lndbulk_inst: CLM water forcing instance.  Required
            fields:

            - ``forc_q_downscaled_col[c]`` — specific humidity (kg/kg)
            - ``forc_rain_downscaled_col[c]`` — rainfall (mm/s)
            - ``forc_snow_downscaled_col[c]`` — snowfall (mm/s)

        mlcanopy_inst: Canopy container to update.

    Returns:
        ``(mlcanopy_inst, forcing_cur)``: the container with ``*_soil`` and
        ``solar_zen_forcing`` fields set for the current CLM timestep, and the
        current-step forcing (Fortran ``*_cur_forcing``) as an
        :class:`AtmForcingRecord`.
    """
    from clm_share.shr_orb_mod import shr_orb_cosz, shr_orb_decl  # noqa: F401
    from clm_src_main.clm_varcon import pi  # noqa: F401 (pi = rpi)
    from clm_src_main.clm_varpar import (  # noqa: F401
        inir,
        ivis,
        nlevsno,  # noqa: F401
    )
    from clm_src_main.ColumnType import col  # noqa: F401
    from clm_src_main.PatchType import patch  # noqa: F401
    from clm_src_main.SiteType import site_or_modules
    from clm_src_utils.clm_varorb import eccen, lambm0, mvelpp, obliqr  # noqa: F401

    # ------------------------------------------------------------------
    # Unpack CLM inputs (mirrors Fortran associate block)
    # ------------------------------------------------------------------
    forc_u = atm2lnd_inst.forc_u_grc
    forc_v = atm2lnd_inst.forc_v_grc
    forc_pco2 = atm2lnd_inst.forc_pco2_grc
    forc_po2 = atm2lnd_inst.forc_po2_grc
    forc_solad_col = atm2lnd_inst.forc_solad_downscaled_col
    forc_solai = atm2lnd_inst.forc_solai_grc
    forc_t = atm2lnd_inst.forc_t_downscaled_col
    forc_pbot = atm2lnd_inst.forc_pbot_downscaled_col
    forc_lwrad = atm2lnd_inst.forc_lwrad_downscaled_col
    forc_q = wateratm2lndbulk_inst.forc_q_downscaled_col
    forc_rain = wateratm2lndbulk_inst.forc_rain_downscaled_col
    forc_snow = wateratm2lndbulk_inst.forc_snow_downscaled_col
    albgrd = surfalb_inst.albgrd_col
    albgri = surfalb_inst.albgri_col
    soilresis = soilstate_inst.soilresis_col
    thk = soilstate_inst.thk_col
    t_a10_patch = temperature_inst.t_a10_patch
    t_soisno = temperature_inst.t_soisno_col
    snl = col.snl  # structure (always 0 here)
    site = site_or_modules(site)  # soil geometry and location: per-site data
    z = site.z_col
    zi = site.zi_col

    # Pre-materialise snl as numpy so int(snl[c]) is always concrete
    # (snl is a JAX array; tracing would make it abstract inside jit/grad).
    _snl_np = np.asarray(snl)

    # ------------------------------------------------------------------
    # Mutable copies of the mlcanopy arrays to update
    # ------------------------------------------------------------------
    # Current-step forcing (Fortran *_cur_forcing), returned as a record
    tref_cur = jnp.zeros_like(mlcanopy_inst.tref_forcing)
    qref_cur = jnp.zeros_like(mlcanopy_inst.qref_forcing)
    uref_cur = jnp.zeros_like(mlcanopy_inst.uref_forcing)
    pref_cur = jnp.zeros_like(mlcanopy_inst.pref_forcing)
    co2ref_cur = jnp.zeros_like(mlcanopy_inst.co2ref_forcing)
    o2ref = mlcanopy_inst.o2ref_forcing
    solar_zen = mlcanopy_inst.solar_zen_forcing
    swskyb_cur = jnp.zeros_like(mlcanopy_inst.swskyb_forcing)
    swskyd_cur = jnp.zeros_like(mlcanopy_inst.swskyd_forcing)
    lwsky_cur = jnp.zeros_like(mlcanopy_inst.lwsky_forcing)
    qflx_rain = mlcanopy_inst.qflx_rain_forcing
    qflx_snow = mlcanopy_inst.qflx_snow_forcing
    tacclim = mlcanopy_inst.tacclim_forcing
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    soilres = mlcanopy_inst.soilres_soil
    soil_t = mlcanopy_inst.soil_t_soil
    soil_dz = mlcanopy_inst.soil_dz_soil
    soil_tk = mlcanopy_inst.soil_tk_soil

    # ------------------------------------------------------------------
    # Copy CLM variables to multilayer canopy — Fortran lines 63-90
    # ------------------------------------------------------------------
    # Pre-materialise patch hierarchy indices as numpy ints once so that
    # int() calls below are always concrete, even inside jax.grad tracing.
    _patch_column_np = np.asarray(patch.column)
    _patch_gridcell_np = np.asarray(patch.gridcell)

    for fp in range(1, num_filter + 1):  # Fortran: do fp = 1, num_filter
        p = int(filter[fp - 1])
        c = int(_patch_column_np[p])
        g = int(_patch_gridcell_np[p])

        # Current-step forcing: wind resultant, CLM gridcell/column -> patch,
        # CO2 Pa -> umol/mol — Fortran lines 67-84 (see clm_forcing_to_ml)
        _u, _t, _q, _p, _co2, _solad, _solai, _lw = clm_forcing_to_ml(
            forc_u[g],
            forc_v[g],
            forc_t[c],
            forc_q[c],
            forc_pbot[c],
            forc_pco2[g],
            (forc_solad_col[c, ivis], forc_solad_col[c, inir]),
            (forc_solai[g, ivis], forc_solai[g, inir]),
            forc_lwrad[c],
        )
        uref_cur = uref_cur.at[p].set(_u)
        tref_cur = tref_cur.at[p].set(_t)
        qref_cur = qref_cur.at[p].set(_q)
        pref_cur = pref_cur.at[p].set(_p)
        co2ref_cur = co2ref_cur.at[p].set(_co2)
        swskyb_cur = swskyb_cur.at[p, ivis].set(_solad[0])
        swskyb_cur = swskyb_cur.at[p, inir].set(_solad[1])
        swskyd_cur = swskyd_cur.at[p, ivis].set(_solai[0])
        swskyd_cur = swskyd_cur.at[p, inir].set(_solai[1])
        lwsky_cur = lwsky_cur.at[p].set(_lw)
        qflx_rain = qflx_rain.at[p].set(forc_rain[c])
        qflx_snow = qflx_snow.at[p].set(forc_snow[c])

        # O2 unit conversion — Fortran line 84
        pbot_val = forc_pbot[c]
        o2ref = o2ref.at[p].set(forc_po2[g] / pbot_val * 1.0e3)  # Pa -> mmol/mol

        # Acclimation temperature — Fortran line 87
        tacclim = tacclim.at[p].set(t_a10_patch[p])

        # Ground albedos: column (c) → patch (p) — Fortran lines 89-90
        albsoib = albsoib.at[p, ivis].set(albgrd[c, ivis])
        albsoib = albsoib.at[p, inir].set(albgrd[c, inir])
        albsoid = albsoid.at[p, ivis].set(albgri[c, ivis])
        albsoid = albsoid.at[p, inir].set(albgri[c, inir])

        # Soil evaporative resistance — Fortran line 93
        soilres = soilres.at[p].set(soilresis[c])

        # First snow/soil layer properties — Fortran lines 99-101
        # snl(c) is the number of snow layers (negative value in CLM);
        # snl(c)+1 gives the Fortran index of the first snow or soil layer.
        # In standalone mode snl=0, so j=1 (first soil layer).
        # All arrays use direct j indexing (1:nlevgrnd), no nlevsno offset.
        j = int(_snl_np[c]) + 1  # Fortran index = Python index

        soil_t = soil_t.at[p].set(t_soisno[c, j])
        soil_dz = soil_dz.at[p].set(z[c, j] - zi[c, int(_snl_np[c])])
        soil_tk = soil_tk.at[p].set(thk[c, j])

    # ------------------------------------------------------------------
    # Solar zenith angle — Fortran lines 92-110
    #
    # Computed for the beginning of the CLM timestep, i.e. at
    # calday(nstep-1):  caldaym1 = get_curr_calday(offset=-dtime_clm),
    # passed in as calday_beg
    # ------------------------------------------------------------------
    caldaym1 = _float_if_concrete(calday_beg)

    # Orbital declination for caldaym1 — Fortran line 105
    declinm1, eccf = shr_orb_decl(caldaym1, eccen, mvelpp, lambm0, obliqr)

    # Latitude and longitude: per-site data (Phase 6 step 2), traced in compiled chunks
    latdeg = site.latdeg_grc
    londeg = site.londeg_grc

    for fp in range(1, num_filter + 1):  # Fortran: do fp = 1, num_filter
        p = int(filter[fp - 1])
        g = int(_patch_gridcell_np[p])  # use pre-materialised numpy copy

        # Latitude and longitude in radians — Fortran lines 107-108
        lat = latdeg[g] * pi / 180.0
        lon = londeg[g] * pi / 180.0

        # Cosine of solar zenith angle — Fortran line 109
        coszen = shr_orb_cosz(caldaym1, lat, lon, declinm1)

        # Zenith angle: floor coszen at 0.01 to avoid acos(≤0) — Fortran line 110
        solar_zen_val = jnp.arccos(jnp.maximum(jnp.asarray(0.01), jnp.asarray(coszen)))
        solar_zen = solar_zen.at[p].set(solar_zen_val)

    # ------------------------------------------------------------------
    # Return updated container
    # ------------------------------------------------------------------
    forcing_cur = AtmForcingRecord(
        uref=uref_cur,
        tref=tref_cur,
        qref=qref_cur,
        pref=pref_cur,
        co2ref=co2ref_cur,
        swskyb=swskyb_cur,
        swskyd=swskyd_cur,
        lwsky=lwsky_cur,
    )
    return (
        mlcanopy_inst._replace(
            o2ref_forcing=o2ref,
            solar_zen_forcing=solar_zen,
            qflx_rain_forcing=qflx_rain,
            qflx_snow_forcing=qflx_snow,
            tacclim_forcing=tacclim,
            albsoib_soil=albsoib,
            albsoid_soil=albsoid,
            soilres_soil=soilres,
            soil_t_soil=soil_t,
            soil_dz_soil=soil_dz,
            soil_tk_soil=soil_tk,
        ),
        forcing_cur,
    )


def _MLAccumulateFluxes(
    num_filter: int,
    filter: List[int],
    flux_accumulator: jnp.ndarray,
    flux_accumulator_profile: jnp.ndarray,
    flux_accumulator_leaf: jnp.ndarray,
    mlcanopy_inst: mlcanopy_type,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Accumulate ML-step fluxes into running-sum accumulators.

    Called once per ML sub-step.  The accumulators are initialised to
    zero before the loop (``jnp.zeros`` in MLCanopyFluxes), so the
    Fortran ``nstep_ml == 1`` zero-init branch is not needed here.

    Split from the original ``_MLTimeStepFluxIntegration`` so the
    accumulation body has no Python conditionals on ``nstep_ml`` —
    a prerequisite for future ``jax.lax.fori_loop`` usage once all
    physics functions are ported to XLA-traceable paths.

    Args:
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
            avoiding ``int(mlcanopy_inst.ncan_canopy[p])`` D→H syncs.
        flux_accumulator: JAX array ``(endp+1, nvar1d)`` — running sums
            of single-level fluxes.
        flux_accumulator_profile: JAX array ``(endp+1, nlevmlcan+2, nvar2d)``.
        flux_accumulator_leaf: JAX array
            ``(endp+1, nlevmlcan+1, nleaf+1, nvar3d)``.
        mlcanopy_inst: Canopy container (read-only in this function).

    Returns:
        Updated ``(flux_accumulator, flux_accumulator_profile,
        flux_accumulator_leaf)`` tuple.
    """
    from clm_src_main.clm_varpar import inir, ivis  # noqa: F401

    nvar1d = 23
    nvar2d = 14
    nvar3d = 12

    for fp in range(num_filter):
        p = int(filter[fp])
        # Every layer slot is accumulated; the write-back keeps layers 1..ncan only
        _ncan = mlcanopy_inst.dpai_profile.shape[1] - 1

        # ------------------------------------------------------------------
        # Accumulate single-level (1-D) fluxes — Fortran lines 65-87
        # ------------------------------------------------------------------
        i = -1

        def _acc1(val):
            nonlocal i, flux_accumulator
            i += 1
            flux_accumulator = flux_accumulator.at[p, i].add(val)

        _acc1(mlcanopy_inst.ustar_canopy[p])
        _acc1(mlcanopy_inst.beta_canopy[p])
        _acc1(mlcanopy_inst.obu_canopy[p])
        _acc1(mlcanopy_inst.z0m_canopy[p])
        _acc1(mlcanopy_inst.zdisp_canopy[p])
        _acc1(mlcanopy_inst.lwup_canopy[p])
        _acc1(mlcanopy_inst.swsoi_soil[p, ivis])
        _acc1(mlcanopy_inst.swsoi_soil[p, inir])
        _acc1(mlcanopy_inst.lwsoi_soil[p])
        _acc1(mlcanopy_inst.rnsoi_soil[p])
        _acc1(mlcanopy_inst.shsoi_soil[p])
        _acc1(mlcanopy_inst.lhsoi_soil[p])
        _acc1(mlcanopy_inst.etsoi_soil[p])
        _acc1(mlcanopy_inst.gsoi_soil[p])
        _acc1(mlcanopy_inst.gac0_soil[p])
        _acc1(mlcanopy_inst.qflx_intr_canopy[p])
        _acc1(mlcanopy_inst.qflx_tflrain_canopy[p])
        _acc1(mlcanopy_inst.qflx_tflsnow_canopy[p])
        _acc1(mlcanopy_inst.swskyb_forcing[p, ivis])
        _acc1(mlcanopy_inst.swskyb_forcing[p, inir])
        _acc1(mlcanopy_inst.swskyd_forcing[p, ivis])
        _acc1(mlcanopy_inst.swskyd_forcing[p, inir])
        _acc1(mlcanopy_inst.lwsky_forcing[p])

        assert (i + 1) == nvar1d, f"_MLAccumulateFluxes: nvar1d mismatch ({i+1} != {nvar1d})"

        # ------------------------------------------------------------------
        # Accumulate profile (2-D) fluxes — Fortran lines 89-102
        # ------------------------------------------------------------------
        j = -1

        def _acc2_layers(arr_slice):
            nonlocal j, flux_accumulator_profile
            j += 1
            flux_accumulator_profile = flux_accumulator_profile.at[p, 1 : _ncan + 1, j].add(
                arr_slice
            )

        def _acc2_rad(arr_slice):
            nonlocal j, flux_accumulator_profile
            j += 1
            flux_accumulator_profile = flux_accumulator_profile.at[p, 1 : _ncan + 2, j].add(
                arr_slice
            )

        _acc2_layers(mlcanopy_inst.shair_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.etair_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.stair_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.mflx_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.kc_eddy_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.gac_profile[p, 1 : _ncan + 1])
        _acc2_rad(mlcanopy_inst.swupw_profile[p, 0 : _ncan + 1, ivis])
        _acc2_rad(mlcanopy_inst.swupw_profile[p, 0 : _ncan + 1, inir])
        _acc2_rad(mlcanopy_inst.swdwn_profile[p, 0 : _ncan + 1, ivis])
        _acc2_rad(mlcanopy_inst.swdwn_profile[p, 0 : _ncan + 1, inir])
        _acc2_rad(mlcanopy_inst.swbeam_profile[p, 0 : _ncan + 1, ivis])
        _acc2_rad(mlcanopy_inst.swbeam_profile[p, 0 : _ncan + 1, inir])
        _acc2_rad(mlcanopy_inst.lwupw_profile[p, 0 : _ncan + 1])
        _acc2_rad(mlcanopy_inst.lwdwn_profile[p, 0 : _ncan + 1])

        assert (j + 1) == nvar2d, f"_MLAccumulateFluxes: nvar2d mismatch ({j+1} != {nvar2d})"

        # ------------------------------------------------------------------
        # Accumulate leaf (3-D) fluxes — Fortran lines 104-117
        # ------------------------------------------------------------------
        k = -1

        def _acc3(arr_slice):
            nonlocal k, flux_accumulator_leaf
            k += 1
            flux_accumulator_leaf = flux_accumulator_leaf.at[p, :, :, k].add(arr_slice)

        _acc3(mlcanopy_inst.swleaf_leaf[p, :, :, ivis])
        _acc3(mlcanopy_inst.swleaf_leaf[p, :, :, inir])
        _acc3(mlcanopy_inst.lwleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.rnleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.shleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.lhleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.trleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.evleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.stleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.anet_leaf[p, :, :])
        _acc3(mlcanopy_inst.agross_leaf[p, :, :])
        _acc3(mlcanopy_inst.gs_leaf[p, :, :])

        assert (k + 1) == nvar3d, f"_MLAccumulateFluxes: nvar3d mismatch ({k+1} != {nvar3d})"

    return flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf


def _MLScaleAndWriteBack(
    num_ml_steps: int,
    num_filter: int,
    filter: List[int],
    flux_accumulator: jnp.ndarray,
    flux_accumulator_profile: jnp.ndarray,
    flux_accumulator_leaf: jnp.ndarray,
    mlcanopy_inst: mlcanopy_type,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, mlcanopy_type]:
    """
    Scale accumulated flux sums by ``1/num_ml_steps`` and write the
    time-averaged values back into ``mlcanopy_inst``.

    Called once after the ML sub-step loop completes.  Split from
    ``_MLTimeStepFluxIntegration`` so the per-step accumulation body
    (``_MLAccumulateFluxes``) has no loop-index conditionals.

    Args:
        num_ml_steps: Total number of ML sub-steps (divisor for average).
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
        flux_accumulator: Running sums (single-level).
        flux_accumulator_profile: Running sums (profile).
        flux_accumulator_leaf: Running sums (leaf).
        mlcanopy_inst: Canopy container to update with averaged fluxes.

    Returns:
        ``(flux_accumulator, flux_accumulator_profile,
        flux_accumulator_leaf, mlcanopy_inst)`` with scaled accumulators
        and updated canopy container.
    """
    from clm_src_main.abortutils import endrun  # noqa: F401
    from clm_src_main.clm_varpar import inir, ivis  # noqa: F401

    nvar1d = 23
    nvar2d = 14
    nvar3d = 12

    scale = 1.0 / float(num_ml_steps)

    for fp in range(num_filter):
        p = int(filter[fp])
        flux_accumulator = flux_accumulator.at[p, :].mul(scale)
        flux_accumulator_profile = flux_accumulator_profile.at[p, :, :].mul(scale)
        flux_accumulator_leaf = flux_accumulator_leaf.at[p, :, :, :].mul(scale)

    # Read all arrays we need to update
    ustar = mlcanopy_inst.ustar_canopy
    beta = mlcanopy_inst.beta_canopy
    obu = mlcanopy_inst.obu_canopy
    z0m = mlcanopy_inst.z0m_canopy
    zdisp = mlcanopy_inst.zdisp_canopy
    lwup = mlcanopy_inst.lwup_canopy
    swsoi = mlcanopy_inst.swsoi_soil
    lwsoi = mlcanopy_inst.lwsoi_soil
    rnsoi = mlcanopy_inst.rnsoi_soil
    shsoi = mlcanopy_inst.shsoi_soil
    lhsoi = mlcanopy_inst.lhsoi_soil
    etsoi = mlcanopy_inst.etsoi_soil
    gsoi = mlcanopy_inst.gsoi_soil
    gac0 = mlcanopy_inst.gac0_soil
    qflx_intr = mlcanopy_inst.qflx_intr_canopy
    qflx_tflrain = mlcanopy_inst.qflx_tflrain_canopy
    qflx_tflsnow = mlcanopy_inst.qflx_tflsnow_canopy
    swskyb = mlcanopy_inst.swskyb_forcing
    swskyd = mlcanopy_inst.swskyd_forcing
    lwsky = mlcanopy_inst.lwsky_forcing
    shair = mlcanopy_inst.shair_profile
    etair = mlcanopy_inst.etair_profile
    stair = mlcanopy_inst.stair_profile
    mflx = mlcanopy_inst.mflx_profile
    kc_eddy = mlcanopy_inst.kc_eddy_profile
    gac = mlcanopy_inst.gac_profile
    swupw = mlcanopy_inst.swupw_profile
    swdwn = mlcanopy_inst.swdwn_profile
    swbeam = mlcanopy_inst.swbeam_profile
    lwupw = mlcanopy_inst.lwupw_profile
    lwdwn = mlcanopy_inst.lwdwn_profile
    swleaf = mlcanopy_inst.swleaf_leaf
    lwleaf = mlcanopy_inst.lwleaf_leaf
    rnleaf = mlcanopy_inst.rnleaf_leaf
    shleaf = mlcanopy_inst.shleaf_leaf
    lhleaf = mlcanopy_inst.lhleaf_leaf
    trleaf = mlcanopy_inst.trleaf_leaf
    evleaf = mlcanopy_inst.evleaf_leaf
    stleaf = mlcanopy_inst.stleaf_leaf
    anet = mlcanopy_inst.anet_leaf
    agross = mlcanopy_inst.agross_leaf
    gs = mlcanopy_inst.gs_leaf

    for fp in range(num_filter):
        p = int(filter[fp])
        _ncan = mlcanopy_inst.dpai_profile.shape[1] - 1  # every slot; written up to ncan
        _lay = layers(mlcanopy_inst, p)
        _upto = jnp.arange(_ncan + 1) <= _lay.ncan  # levels 0..ncan

        i = -1

        def _wb1():
            nonlocal i
            i += 1
            return flux_accumulator[p, i]

        ustar = ustar.at[p].set(_wb1())
        beta = beta.at[p].set(_wb1())
        obu = obu.at[p].set(_wb1())
        z0m = z0m.at[p].set(_wb1())
        zdisp = zdisp.at[p].set(_wb1())
        lwup = lwup.at[p].set(_wb1())
        swsoi = swsoi.at[p, ivis].set(_wb1())
        swsoi = swsoi.at[p, inir].set(_wb1())
        lwsoi = lwsoi.at[p].set(_wb1())
        rnsoi = rnsoi.at[p].set(_wb1())
        shsoi = shsoi.at[p].set(_wb1())
        lhsoi = lhsoi.at[p].set(_wb1())
        etsoi = etsoi.at[p].set(_wb1())
        gsoi = gsoi.at[p].set(_wb1())
        gac0 = gac0.at[p].set(_wb1())
        qflx_intr = qflx_intr.at[p].set(_wb1())
        qflx_tflrain = qflx_tflrain.at[p].set(_wb1())
        qflx_tflsnow = qflx_tflsnow.at[p].set(_wb1())
        swskyb = swskyb.at[p, ivis].set(_wb1())
        swskyb = swskyb.at[p, inir].set(_wb1())
        swskyd = swskyd.at[p, ivis].set(_wb1())
        swskyd = swskyd.at[p, inir].set(_wb1())
        lwsky = lwsky.at[p].set(_wb1())

        assert (i + 1) == nvar1d, "_MLScaleAndWriteBack: nvar1d mismatch"

        j = -1

        def _wb2_layers():
            nonlocal j
            j += 1
            return flux_accumulator_profile[p, 1 : _ncan + 1, j]

        def _wb2_rad():
            nonlocal j
            j += 1
            return flux_accumulator_profile[p, 1 : _ncan + 2, j]

        def _lay_set(arr):
            return arr.at[p, 1:].set(keep(_lay.in_use, _wb2_layers(), arr[p, 1:]))

        def _rad_set(arr, *band):
            sl = (p, slice(None)) + band
            return arr.at[sl].set(jnp.where(_upto, _wb2_rad(), arr[sl]))

        shair = _lay_set(shair)
        etair = _lay_set(etair)
        stair = _lay_set(stair)
        mflx = _lay_set(mflx)
        kc_eddy = _lay_set(kc_eddy)
        gac = _lay_set(gac)
        swupw = _rad_set(swupw, ivis)
        swupw = _rad_set(swupw, inir)
        swdwn = _rad_set(swdwn, ivis)
        swdwn = _rad_set(swdwn, inir)
        swbeam = _rad_set(swbeam, ivis)
        swbeam = _rad_set(swbeam, inir)
        lwupw = _rad_set(lwupw)
        lwdwn = _rad_set(lwdwn)

        assert (j + 1) == nvar2d, "_MLScaleAndWriteBack: nvar2d mismatch"

        k = -1

        def _wb3():
            nonlocal k
            k += 1
            return flux_accumulator_leaf[p, :, :, k]

        swleaf = swleaf.at[p, :, :, ivis].set(_wb3())
        swleaf = swleaf.at[p, :, :, inir].set(_wb3())
        lwleaf = lwleaf.at[p, :, :].set(_wb3())
        rnleaf = rnleaf.at[p, :, :].set(_wb3())
        shleaf = shleaf.at[p, :, :].set(_wb3())
        lhleaf = lhleaf.at[p, :, :].set(_wb3())
        trleaf = trleaf.at[p, :, :].set(_wb3())
        evleaf = evleaf.at[p, :, :].set(_wb3())
        stleaf = stleaf.at[p, :, :].set(_wb3())
        anet = anet.at[p, :, :].set(_wb3())
        agross = agross.at[p, :, :].set(_wb3())
        gs = gs.at[p, :, :].set(_wb3())

        assert (k + 1) == nvar3d, "_MLScaleAndWriteBack: nvar3d mismatch"

    mlcanopy_inst = mlcanopy_inst._replace(
        ustar_canopy=ustar,
        beta_canopy=beta,
        obu_canopy=obu,
        z0m_canopy=z0m,
        zdisp_canopy=zdisp,
        lwup_canopy=lwup,
        swsoi_soil=swsoi,
        lwsoi_soil=lwsoi,
        rnsoi_soil=rnsoi,
        shsoi_soil=shsoi,
        lhsoi_soil=lhsoi,
        etsoi_soil=etsoi,
        gsoi_soil=gsoi,
        gac0_soil=gac0,
        qflx_intr_canopy=qflx_intr,
        qflx_tflrain_canopy=qflx_tflrain,
        qflx_tflsnow_canopy=qflx_tflsnow,
        swskyb_forcing=swskyb,
        swskyd_forcing=swskyd,
        lwsky_forcing=lwsky,
        shair_profile=shair,
        etair_profile=etair,
        stair_profile=stair,
        mflx_profile=mflx,
        kc_eddy_profile=kc_eddy,
        gac_profile=gac,
        swupw_profile=swupw,
        swdwn_profile=swdwn,
        swbeam_profile=swbeam,
        lwupw_profile=lwupw,
        lwdwn_profile=lwdwn,
        swleaf_leaf=swleaf,
        lwleaf_leaf=lwleaf,
        rnleaf_leaf=rnleaf,
        shleaf_leaf=shleaf,
        lhleaf_leaf=lhleaf,
        trleaf_leaf=trleaf,
        evleaf_leaf=evleaf,
        stleaf_leaf=stleaf,
        anet_leaf=anet,
        agross_leaf=agross,
        gs_leaf=gs,
    )

    return flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf, mlcanopy_inst


# _MLTimeStepFluxIntegration has been replaced by the two-function split:
#   _MLAccumulateFluxes  — per-step accumulation (no conditionals on nstep_ml)
#   _MLScaleAndWriteBack — post-loop scale + write-back (called once)
# This separation is a prerequisite for future lax.fori_loop usage once all
# physics functions are ported to XLA-traceable paths.


def _CanopyFluxesDiagnostics(
    num_filter: int,
    filter: List[int],
    mlcanopy_inst: mlcanopy_type,
    opts: MLOptions,
) -> mlcanopy_type:
    """
    Compute canopy-integrated fluxes, diagnostics, and energy balance checks.

    Mirrors Fortran subroutine ``CanopyFluxesDiagnostics`` (lines 1-270).

    **Algorithm** (one pass per patch ``p``):

    1. **Layer-mean leaf fluxes** (Fortran lines 80-130):
       For each layer ``ic`` with ``dpai > 0`` compute sun/shade
       weighted means::

           X_mean(ic) = X(ic,isun)*fracsun(ic) + X(ic,isha)*(1-fracsun(ic))

       and convert to per-ground-area source terms::

           Xsrc(ic) = X_mean(ic) * dpai(ic)

       Layers with ``dpai == 0`` are zeroed.

    2. **Vegetation totals** (Fortran lines 132-158):
       Sum source terms over all layers to produce ``swveg``, ``lwveg``,
       ``shveg``, ``lhveg``, ``etveg``, ``trveg``, ``evveg``,
       ``gppveg``, ``vcmax25veg``, ``gsveg``, ``stflx_veg``.

    3. **Vegetation energy balance check** (Fortran lines 160-163):
       ``|swveg(vis) + swveg(nir) + lwveg - shveg - lhveg - stflx_veg| < 1e-3 W/m²``

    4. **Albedo** (Fortran lines 165-171):
       ``albcan(ib) = swupw(ntop, ib) / (swskyb(ib) + swskyd(ib))``
       (0 when incoming == 0).

    5. **Turbulent fluxes** (Fortran lines 173-182):
       ``flux_profile_type == 0 or -1``: ``shflx = shveg + shsoi``,
       ``etflx = etveg + etsoi``, ``lhflx = lhveg + lhsoi``.
       ``flux_profile_type == 1``: take top-layer air flux directly.

    6. **Air storage flux** (Fortran lines 184-187):
       ``stflx_air = sum(stair(1:ncan))``.

    7. **Overall energy balance checks** (Fortran lines 189-211):
       Three independent checks with tolerances 0.001, 0.001, and
       0.01 W/m².

    8. **Sun/shade canopy totals** (Fortran lines 213-250):
       Accumulate ``laisun``, ``laisha``, per-population fluxes
       ``swvegsun/sha``, ``lwvegsun/sha``, ``shvegsun/sha``,
       ``lhvegsun/sha``, ``etvegsun/sha``, ``gppvegsun/sha``,
       ``vcmax25sun/sha``, ``gsvegsun/sha``.

    9. **PAI-weighted temperatures and wind** (Fortran lines 252-268):
       ``windveg``, ``windvegsun``, ``windvegsha``, ``tlveg``,
       ``tlvegsun``, ``tlvegsha``, ``taveg``, ``tavegsun``,
       ``tavegsha``.

    10. **Water-stress fraction** (Fortran lines 270-278):
        ``fracminlwp = sum(dpai where lwp_mean <= -2 MPa) / (lai+sai)``

    Args:
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; all fields read; output fields
            written and returned.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    from clm_src_main.abortutils import endrun  # noqa: F401
    from clm_src_main.clm_varpar import inir, ivis, numrad  # noqa: F401

    flux_profile_type = opts.flux_profile_type
    from multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401

    # ------------------------------------------------------------------
    # Extract mutable arrays (all output fields)
    # ------------------------------------------------------------------
    rnet = mlcanopy_inst.rnet_canopy
    stflx_air = mlcanopy_inst.stflx_air_canopy
    stflx_veg = mlcanopy_inst.stflx_veg_canopy
    shflx = mlcanopy_inst.shflx_canopy
    lhflx = mlcanopy_inst.lhflx_canopy
    etflx = mlcanopy_inst.etflx_canopy
    albcan = mlcanopy_inst.albcan_canopy
    swveg = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    lwveg = mlcanopy_inst.lwveg_canopy
    lwvegsun = mlcanopy_inst.lwvegsun_canopy
    lwvegsha = mlcanopy_inst.lwvegsha_canopy
    shveg = mlcanopy_inst.shveg_canopy
    shvegsun = mlcanopy_inst.shvegsun_canopy
    shvegsha = mlcanopy_inst.shvegsha_canopy
    lhveg = mlcanopy_inst.lhveg_canopy
    lhvegsun = mlcanopy_inst.lhvegsun_canopy
    lhvegsha = mlcanopy_inst.lhvegsha_canopy
    etveg = mlcanopy_inst.etveg_canopy
    etvegsun = mlcanopy_inst.etvegsun_canopy
    etvegsha = mlcanopy_inst.etvegsha_canopy
    trveg = mlcanopy_inst.trveg_canopy
    evveg = mlcanopy_inst.evveg_canopy
    gppveg = mlcanopy_inst.gppveg_canopy
    gppvegsun = mlcanopy_inst.gppvegsun_canopy
    gppvegsha = mlcanopy_inst.gppvegsha_canopy
    vcmax25veg = mlcanopy_inst.vcmax25veg_canopy
    vcmax25sun = mlcanopy_inst.vcmax25sun_canopy
    vcmax25sha = mlcanopy_inst.vcmax25sha_canopy
    gsveg = mlcanopy_inst.gsveg_canopy
    gsvegsun = mlcanopy_inst.gsvegsun_canopy
    gsvegsha = mlcanopy_inst.gsvegsha_canopy
    windveg = mlcanopy_inst.windveg_canopy
    windvegsun = mlcanopy_inst.windvegsun_canopy
    windvegsha = mlcanopy_inst.windvegsha_canopy
    tlveg = mlcanopy_inst.tlveg_canopy
    tlvegsun = mlcanopy_inst.tlvegsun_canopy
    tlvegsha = mlcanopy_inst.tlvegsha_canopy
    taveg = mlcanopy_inst.taveg_canopy
    tavegsun = mlcanopy_inst.tavegsun_canopy
    tavegsha = mlcanopy_inst.tavegsha_canopy
    laisun = mlcanopy_inst.laisun_canopy
    laisha = mlcanopy_inst.laisha_canopy
    fracminlwp = mlcanopy_inst.fracminlwp_canopy
    swsrc = mlcanopy_inst.swsrc_profile
    lwsrc = mlcanopy_inst.lwsrc_profile
    rnsrc = mlcanopy_inst.rnsrc_profile
    stsrc = mlcanopy_inst.stsrc_profile
    shsrc = mlcanopy_inst.shsrc_profile
    lhsrc = mlcanopy_inst.lhsrc_profile
    etsrc = mlcanopy_inst.etsrc_profile
    trsrc = mlcanopy_inst.trsrc_profile
    evsrc = mlcanopy_inst.evsrc_profile
    fco2src = mlcanopy_inst.fco2src_profile
    swleaf_mean = mlcanopy_inst.swleaf_mean_profile
    lwleaf_mean = mlcanopy_inst.lwleaf_mean_profile
    rnleaf_mean = mlcanopy_inst.rnleaf_mean_profile
    stleaf_mean = mlcanopy_inst.stleaf_mean_profile
    shleaf_mean = mlcanopy_inst.shleaf_mean_profile
    lhleaf_mean = mlcanopy_inst.lhleaf_mean_profile
    etleaf_mean = mlcanopy_inst.etleaf_mean_profile
    trleaf_mean = mlcanopy_inst.trleaf_mean_profile
    evleaf_mean = mlcanopy_inst.evleaf_mean_profile
    fco2_mean = mlcanopy_inst.fco2_mean_profile
    apar_mean = mlcanopy_inst.apar_mean_profile
    gs_mean = mlcanopy_inst.gs_mean_profile
    tleaf_mean = mlcanopy_inst.tleaf_mean_profile
    lwp_mean = mlcanopy_inst.lwp_mean_profile

    # ==================================================================
    # Main patch loop — Fortran: do fp = 1, num_filter
    # ==================================================================
    for fp in range(1, num_filter + 1):
        p = int(filter[fp - 1])
        lay = layers(mlcanopy_inst, p)  # layer counts as data (plan Phase 5b)
        _ncan, _ntop = lay.ncan, lay.ntop

        def _pad(x):
            """Per-level array: padding slots (above ncan) get layer ncan's values."""
            return x.at[1:].set(pad_with_top(lay, x[1:]))

        # ----------------------------------------------------------------
        # Per-layer inputs — kept as JAX arrays (no D→H sync until needed
        # by Python-level energy balance checks below).
        # ----------------------------------------------------------------
        _dpai = _pad(mlcanopy_inst.dpai_profile[p])
        _fracsun = _pad(mlcanopy_inst.fracsun_profile[p])
        _fwet = _pad(mlcanopy_inst.fwet_profile[p])
        _fdry = _pad(mlcanopy_inst.fdry_profile[p])
        _wind = _pad(mlcanopy_inst.wind_profile[p])
        _tair = _pad(mlcanopy_inst.tair_profile[p])
        _stair = _pad(mlcanopy_inst.stair_profile[p])
        _vcmax25_prof = _pad(mlcanopy_inst.vcmax25_profile[p])
        _lwleaf_sun = _pad(mlcanopy_inst.lwleaf_leaf[p, :, isun])
        _lwleaf_sha = _pad(mlcanopy_inst.lwleaf_leaf[p, :, isha])
        _swleaf_sun_vis = _pad(mlcanopy_inst.swleaf_leaf[p, :, isun, ivis])
        _swleaf_sun_nir = _pad(mlcanopy_inst.swleaf_leaf[p, :, isun, inir])
        _swleaf_sha_vis = _pad(mlcanopy_inst.swleaf_leaf[p, :, isha, ivis])
        _swleaf_sha_nir = _pad(mlcanopy_inst.swleaf_leaf[p, :, isha, inir])
        _rnleaf_sun = _pad(mlcanopy_inst.rnleaf_leaf[p, :, isun])
        _rnleaf_sha = _pad(mlcanopy_inst.rnleaf_leaf[p, :, isha])
        _stleaf_sun = _pad(mlcanopy_inst.stleaf_leaf[p, :, isun])
        _stleaf_sha = _pad(mlcanopy_inst.stleaf_leaf[p, :, isha])
        _shleaf_sun = _pad(mlcanopy_inst.shleaf_leaf[p, :, isun])
        _shleaf_sha = _pad(mlcanopy_inst.shleaf_leaf[p, :, isha])
        _lhleaf_sun = _pad(mlcanopy_inst.lhleaf_leaf[p, :, isun])
        _lhleaf_sha = _pad(mlcanopy_inst.lhleaf_leaf[p, :, isha])
        _evleaf_sun = _pad(mlcanopy_inst.evleaf_leaf[p, :, isun])
        _evleaf_sha = _pad(mlcanopy_inst.evleaf_leaf[p, :, isha])
        _trleaf_sun = _pad(mlcanopy_inst.trleaf_leaf[p, :, isun])
        _trleaf_sha = _pad(mlcanopy_inst.trleaf_leaf[p, :, isha])
        _anet_sun = _pad(mlcanopy_inst.anet_leaf[p, :, isun])
        _anet_sha = _pad(mlcanopy_inst.anet_leaf[p, :, isha])
        _agross_sun = _pad(mlcanopy_inst.agross_leaf[p, :, isun])
        _agross_sha = _pad(mlcanopy_inst.agross_leaf[p, :, isha])
        _apar_sun = _pad(mlcanopy_inst.apar_leaf[p, :, isun])
        _apar_sha = _pad(mlcanopy_inst.apar_leaf[p, :, isha])
        _gs_sun = _pad(mlcanopy_inst.gs_leaf[p, :, isun])
        _gs_sha = _pad(mlcanopy_inst.gs_leaf[p, :, isha])
        _tleaf_sun = _pad(mlcanopy_inst.tleaf_leaf[p, :, isun])
        _tleaf_sha = _pad(mlcanopy_inst.tleaf_leaf[p, :, isha])
        _lwp_sun = _pad(mlcanopy_inst.lwp_leaf[p, :, isun])
        _lwp_sha = _pad(mlcanopy_inst.lwp_leaf[p, :, isha])
        _vcmax25_sun = _pad(mlcanopy_inst.vcmax25_leaf[p, :, isun])
        _vcmax25_sha = _pad(mlcanopy_inst.vcmax25_leaf[p, :, isha])
        _swskyb = mlcanopy_inst.swskyb_forcing[p]
        _swskyd = mlcanopy_inst.swskyd_forcing[p]
        _swbeam_ntop = mlcanopy_inst.swbeam_profile[p, _ntop]
        _swdwn_ntop = mlcanopy_inst.swdwn_profile[p, _ntop]
        _swupw_ntop = mlcanopy_inst.swupw_profile[p, _ntop]
        _lwdwn_ntop = mlcanopy_inst.lwdwn_profile[p, _ntop]
        _lwupw_ntop = mlcanopy_inst.lwupw_profile[p, _ntop]
        _swsoi_vis = mlcanopy_inst.swsoi_soil[p, ivis]
        _swsoi_nir = mlcanopy_inst.swsoi_soil[p, inir]
        _lwsoi_p = mlcanopy_inst.lwsoi_soil[p]
        _lwsky_p = mlcanopy_inst.lwsky_forcing[p]
        _lwup_p = mlcanopy_inst.lwup_canopy[p]
        _gsoi_p = mlcanopy_inst.gsoi_soil[p]
        lai_p = mlcanopy_inst.lai_canopy[p]
        sai_p = mlcanopy_inst.sai_canopy[p]

        # ----------------------------------------------------------------
        # Vectorized computation over ic=1.._ncan  (pure JAX — no D→H)
        # ----------------------------------------------------------------
        ics = lay.index  # every layer slot; has_pai below excludes padding
        dpai_v = _dpai[ics]
        fs_v = _fracsun[ics]
        fsh_v = 1.0 - fs_v
        fwet_v = _fwet[ics]
        fdry_v = _fdry[ics]
        has_pai = (dpai_v > 0.0) & lay.in_use
        dpai_s = jnp.where(has_pai, dpai_v, 0.0)

        # fracgreen = fdry / (1 - fwet), guard against fwet == 1
        fwet_safe = jnp.where(fwet_v < 1.0, fwet_v, 0.0)
        fracgreen_v = jnp.where(has_pai, fdry_v / (1.0 - fwet_safe), 0.0)

        # --- 1. Layer-mean leaf fluxes (per leaf area) ---
        def _blend(a_sun, a_sha):
            return jnp.where(has_pai, a_sun[ics] * fs_v + a_sha[ics] * fsh_v, 0.0)

        lw_m = _blend(_lwleaf_sun, _lwleaf_sha)
        sw_m_vis = _blend(_swleaf_sun_vis, _swleaf_sha_vis)
        sw_m_nir = _blend(_swleaf_sun_nir, _swleaf_sha_nir)
        rn_m = _blend(_rnleaf_sun, _rnleaf_sha)
        st_m = _blend(_stleaf_sun, _stleaf_sha)
        sh_m = _blend(_shleaf_sun, _shleaf_sha)
        lh_m = _blend(_lhleaf_sun, _lhleaf_sha)
        ev_m = _blend(_evleaf_sun, _evleaf_sha)
        tr_m = _blend(_trleaf_sun, _trleaf_sha)
        et_m = ev_m + tr_m
        fco2_m = _blend(_anet_sun, _anet_sha)
        apar_m = _blend(_apar_sun, _apar_sha)
        gs_m = _blend(_gs_sun, _gs_sha)
        tl_m = _blend(_tleaf_sun, _tleaf_sha)
        lp_m = _blend(_lwp_sun, _lwp_sha)

        # --- 1. Source terms (per ground area) ---
        lw_src = lw_m * dpai_s
        sw_src_vis = sw_m_vis * dpai_s
        sw_src_nir = sw_m_nir * dpai_s
        rn_src = rn_m * dpai_s
        st_src = st_m * dpai_s
        sh_src = sh_m * dpai_s
        lh_src = lh_m * dpai_s
        et_src = et_m * dpai_s
        tr_src = tr_m * dpai_s
        ev_src = ev_m * dpai_s
        fco2_src = fco2_m * dpai_s * fracgreen_v

        # --- 2. Vegetation totals ---
        swveg_vis = jnp.sum(sw_src_vis)
        swveg_nir = jnp.sum(sw_src_nir)
        lwveg_p = jnp.sum(lw_src)
        stflx_veg_p = jnp.sum(st_src)
        shveg_p = jnp.sum(sh_src)
        lhveg_p = jnp.sum(lh_src)
        etveg_p = jnp.sum(et_src)
        trveg_p = jnp.sum(tr_src)
        evveg_p = jnp.sum(ev_src)
        vcmax25veg_p = jnp.sum(_vcmax25_prof[ics] * dpai_s)
        gppveg_p = jnp.sum(
            (_agross_sun[ics] * fs_v + _agross_sha[ics] * fsh_v) * dpai_s * fracgreen_v
        )
        gsveg_p = jnp.sum(gs_m * dpai_s)

        swveg = swveg.at[p, ivis].set(swveg_vis)
        swveg = swveg.at[p, inir].set(swveg_nir)
        lwveg = lwveg.at[p].set(lwveg_p)
        stflx_veg = stflx_veg.at[p].set(stflx_veg_p)
        shveg = shveg.at[p].set(shveg_p)
        lhveg = lhveg.at[p].set(lhveg_p)
        etveg = etveg.at[p].set(etveg_p)
        trveg = trveg.at[p].set(trveg_p)
        evveg = evveg.at[p].set(evveg_p)
        gppveg = gppveg.at[p].set(gppveg_p)
        vcmax25veg = vcmax25veg.at[p].set(vcmax25veg_p)
        gsveg = gsveg.at[p].set(gsveg_p)

        # --- 3. Vegetation energy balance check ---
        err = swveg_vis + swveg_nir + lwveg_p - shveg_p - lhveg_p - stflx_veg_p
        mlcanopy_inst = record_balance_error(mlcanopy_inst, p, CANOPY_ENERGY_1, err)

        # --- 4. Albedo ---
        _inc_vis = _swskyb[ivis] + _swskyd[ivis]
        _inc_nir = _swskyb[inir] + _swskyd[inir]
        _alb_vis = jnp.where(_inc_vis > 0.0, _swupw_ntop[ivis] / jnp.maximum(_inc_vis, 1e-30), 0.0)
        _alb_nir = jnp.where(_inc_nir > 0.0, _swupw_ntop[inir] / jnp.maximum(_inc_nir, 1e-30), 0.0)
        albcan = albcan.at[p, ivis].set(_alb_vis)
        albcan = albcan.at[p, inir].set(_alb_nir)

        # --- 5. Turbulent fluxes ---
        if flux_profile_type in (0, -1):
            shflx_val = shveg_p + mlcanopy_inst.shsoi_soil[p]
            etflx_val = etveg_p + mlcanopy_inst.etsoi_soil[p]
            lhflx_val = lhveg_p + mlcanopy_inst.lhsoi_soil[p]
        elif flux_profile_type == 1:
            shflx_val = mlcanopy_inst.shair_profile[p, _ncan]
            etflx_val = mlcanopy_inst.etair_profile[p, _ncan]
            lhflx_val = etflx_val * LatVap(mlcanopy_inst.tref_forcing[p])
        else:
            endrun(msg=" ERROR: CanopyFluxesDiagnostics: turbulence type not valid")
            shflx_val = etflx_val = lhflx_val = 0.0  # Unreachable
        shflx = shflx.at[p].set(shflx_val)
        etflx = etflx.at[p].set(etflx_val)
        lhflx = lhflx.at[p].set(lhflx_val)

        # --- 6. Air heat storage ---
        stflx_air_p = jnp.sum(jnp.where(lay.in_use, _stair[ics], 0.0))
        stflx_air = stflx_air.at[p].set(stflx_air_p)

        # --- 7. Energy balance checks ---
        rnet_p = swveg_vis + swveg_nir + _swsoi_vis + _swsoi_nir + lwveg_p + _lwsoi_p
        rnet = rnet.at[p].set(rnet_p)

        radin = _swskyb[ivis] + _swskyd[ivis] + _swskyb[inir] + _swskyd[inir] + _lwsky_p
        radout = (
            _alb_vis * (_swskyb[ivis] + _swskyd[ivis])
            + _alb_nir * (_swskyb[inir] + _swskyd[inir])
            + _lwup_p
        )

        err = rnet_p - (radin - radout)
        mlcanopy_inst = record_balance_error(mlcanopy_inst, p, CANOPY_ENERGY_2, err)

        avail = radin - radout - _gsoi_p
        flux_p = shflx_val + lhflx_val + stflx_air_p + stflx_veg_p
        err = avail - flux_p
        mlcanopy_inst = record_balance_error(mlcanopy_inst, p, CANOPY_ENERGY_3, err)

        radin_top = (
            _swbeam_ntop[ivis]
            + _swbeam_ntop[inir]
            + _swdwn_ntop[ivis]
            + _swdwn_ntop[inir]
            + _lwdwn_ntop
        )
        radout_top = _swupw_ntop[ivis] + _swupw_ntop[inir] + _lwupw_ntop
        err = (radin_top - radout_top) - rnet_p
        mlcanopy_inst = record_balance_error(mlcanopy_inst, p, CANOPY_ENERGY_4, err)

        # --- 8. Sun/shade canopy totals ---
        laisun_p = jnp.sum(fs_v * dpai_s)
        laisha_p = jnp.sum(fsh_v * dpai_s)
        swvegsun_vis = jnp.sum(_swleaf_sun_vis[ics] * fs_v * dpai_s)
        swvegsun_nir = jnp.sum(_swleaf_sun_nir[ics] * fs_v * dpai_s)
        swvegsha_vis = jnp.sum(_swleaf_sha_vis[ics] * fsh_v * dpai_s)
        swvegsha_nir = jnp.sum(_swleaf_sha_nir[ics] * fsh_v * dpai_s)
        lwvegsun_p = jnp.sum(_lwleaf_sun[ics] * fs_v * dpai_s)
        lwvegsha_p = jnp.sum(_lwleaf_sha[ics] * fsh_v * dpai_s)
        shvegsun_p = jnp.sum(_shleaf_sun[ics] * fs_v * dpai_s)
        shvegsha_p = jnp.sum(_shleaf_sha[ics] * fsh_v * dpai_s)
        lhvegsun_p = jnp.sum(_lhleaf_sun[ics] * fs_v * dpai_s)
        lhvegsha_p = jnp.sum(_lhleaf_sha[ics] * fsh_v * dpai_s)
        etvegsun_p = jnp.sum((_evleaf_sun[ics] + _trleaf_sun[ics]) * fs_v * dpai_s)
        etvegsha_p = jnp.sum((_evleaf_sha[ics] + _trleaf_sha[ics]) * fsh_v * dpai_s)
        gppvegsun_p = jnp.sum(_agross_sun[ics] * fs_v * dpai_s * fracgreen_v)
        gppvegsha_p = jnp.sum(_agross_sha[ics] * fsh_v * dpai_s * fracgreen_v)
        vcmax25sun_p = jnp.sum(_vcmax25_sun[ics] * fs_v * dpai_s)
        vcmax25sha_p = jnp.sum(_vcmax25_sha[ics] * fsh_v * dpai_s)
        gsvegsun_p = jnp.sum(_gs_sun[ics] * fs_v * dpai_s)
        gsvegsha_p = jnp.sum(_gs_sha[ics] * fsh_v * dpai_s)

        laisun = laisun.at[p].set(laisun_p)
        laisha = laisha.at[p].set(laisha_p)
        swvegsun = swvegsun.at[p, ivis].set(swvegsun_vis)
        swvegsun = swvegsun.at[p, inir].set(swvegsun_nir)
        swvegsha = swvegsha.at[p, ivis].set(swvegsha_vis)
        swvegsha = swvegsha.at[p, inir].set(swvegsha_nir)
        lwvegsun = lwvegsun.at[p].set(lwvegsun_p)
        lwvegsha = lwvegsha.at[p].set(lwvegsha_p)
        shvegsun = shvegsun.at[p].set(shvegsun_p)
        shvegsha = shvegsha.at[p].set(shvegsha_p)
        lhvegsun = lhvegsun.at[p].set(lhvegsun_p)
        lhvegsha = lhvegsha.at[p].set(lhvegsha_p)
        etvegsun = etvegsun.at[p].set(etvegsun_p)
        etvegsha = etvegsha.at[p].set(etvegsha_p)
        gppvegsun = gppvegsun.at[p].set(gppvegsun_p)
        gppvegsha = gppvegsha.at[p].set(gppvegsha_p)
        vcmax25sun = vcmax25sun.at[p].set(vcmax25sun_p)
        vcmax25sha = vcmax25sha.at[p].set(vcmax25sha_p)
        gsvegsun = gsvegsun.at[p].set(gsvegsun_p)
        gsvegsha = gsvegsha.at[p].set(gsvegsha_p)

        # --- 9. PAI-weighted temperatures and wind (pure JAX, no D→H) ---
        total_pai = laisun_p + laisha_p
        wind_v = _wind[ics]
        tair_v = _tair[ics]
        # Guard against zero PAI with jnp.where — avoids D→H for the if/else
        total_safe = jnp.maximum(total_pai, 1e-30)
        laisun_safe = jnp.maximum(laisun_p, 1e-30)
        laisha_safe = jnp.maximum(laisha_p, 1e-30)
        windveg_p = jnp.where(total_pai > 0.0, jnp.sum(wind_v * dpai_s) / total_safe, 0.0)
        windvegsun_p = jnp.where(laisun_p > 0.0, jnp.sum(wind_v * fs_v * dpai_s) / laisun_safe, 0.0)
        windvegsha_p = jnp.where(
            laisha_p > 0.0, jnp.sum(wind_v * fsh_v * dpai_s) / laisha_safe, 0.0
        )
        tlveg_p = jnp.where(total_pai > 0.0, jnp.sum(tl_m * dpai_s) / total_safe, 0.0)
        tlvegsun_p = jnp.where(
            laisun_p > 0.0, jnp.sum(_tleaf_sun[ics] * fs_v * dpai_s) / laisun_safe, 0.0
        )
        tlvegsha_p = jnp.where(
            laisha_p > 0.0, jnp.sum(_tleaf_sha[ics] * fsh_v * dpai_s) / laisha_safe, 0.0
        )
        taveg_p = jnp.where(total_pai > 0.0, jnp.sum(tair_v * dpai_s) / total_safe, 0.0)
        tavegsun_p = jnp.where(laisun_p > 0.0, jnp.sum(tair_v * fs_v * dpai_s) / laisun_safe, 0.0)
        tavegsha_p = jnp.where(laisha_p > 0.0, jnp.sum(tair_v * fsh_v * dpai_s) / laisha_safe, 0.0)

        windveg = windveg.at[p].set(windveg_p)
        windvegsun = windvegsun.at[p].set(windvegsun_p)
        windvegsha = windvegsha.at[p].set(windvegsha_p)
        tlveg = tlveg.at[p].set(tlveg_p)
        tlvegsun = tlvegsun.at[p].set(tlvegsun_p)
        tlvegsha = tlvegsha.at[p].set(tlvegsha_p)
        taveg = taveg.at[p].set(taveg_p)
        tavegsun = tavegsun.at[p].set(tavegsun_p)
        tavegsha = tavegsha.at[p].set(tavegsha_p)

        # --- 10. Water-stress fraction ---
        total_pai_lai_sai = lai_p + sai_p
        fracminlwp_p = jnp.sum(jnp.where(lp_m <= -2.0, dpai_s, 0.0))
        fracminlwp_p = jnp.where(
            total_pai_lai_sai > 0.0, fracminlwp_p / total_pai_lai_sai, fracminlwp_p
        )
        fracminlwp = fracminlwp.at[p].set(fracminlwp_p)

        # ----------------------------------------------------------------
        # Bulk JAX write-backs for per-layer profile fields
        # ----------------------------------------------------------------
        swleaf_mean = swleaf_mean.at[p, 1:, ivis].set(
            keep(lay.in_use, sw_m_vis, swleaf_mean[p, 1:, ivis])
        )
        swleaf_mean = swleaf_mean.at[p, 1:, inir].set(
            keep(lay.in_use, sw_m_nir, swleaf_mean[p, 1:, inir])
        )
        lwleaf_mean = lwleaf_mean.at[p, 1:].set(keep(lay.in_use, lw_m, lwleaf_mean[p, 1:]))
        rnleaf_mean = rnleaf_mean.at[p, 1:].set(keep(lay.in_use, rn_m, rnleaf_mean[p, 1:]))
        stleaf_mean = stleaf_mean.at[p, 1:].set(keep(lay.in_use, st_m, stleaf_mean[p, 1:]))
        shleaf_mean = shleaf_mean.at[p, 1:].set(keep(lay.in_use, sh_m, shleaf_mean[p, 1:]))
        lhleaf_mean = lhleaf_mean.at[p, 1:].set(keep(lay.in_use, lh_m, lhleaf_mean[p, 1:]))
        etleaf_mean = etleaf_mean.at[p, 1:].set(keep(lay.in_use, et_m, etleaf_mean[p, 1:]))
        trleaf_mean = trleaf_mean.at[p, 1:].set(keep(lay.in_use, tr_m, trleaf_mean[p, 1:]))
        evleaf_mean = evleaf_mean.at[p, 1:].set(keep(lay.in_use, ev_m, evleaf_mean[p, 1:]))
        fco2_mean = fco2_mean.at[p, 1:].set(keep(lay.in_use, fco2_m, fco2_mean[p, 1:]))
        apar_mean = apar_mean.at[p, 1:].set(keep(lay.in_use, apar_m, apar_mean[p, 1:]))
        gs_mean = gs_mean.at[p, 1:].set(keep(lay.in_use, gs_m, gs_mean[p, 1:]))
        tleaf_mean = tleaf_mean.at[p, 1:].set(keep(lay.in_use, tl_m, tleaf_mean[p, 1:]))
        lwp_mean = lwp_mean.at[p, 1:].set(keep(lay.in_use, lp_m, lwp_mean[p, 1:]))
        swsrc = swsrc.at[p, 1:, ivis].set(keep(lay.in_use, sw_src_vis, swsrc[p, 1:, ivis]))
        swsrc = swsrc.at[p, 1:, inir].set(keep(lay.in_use, sw_src_nir, swsrc[p, 1:, inir]))
        lwsrc = lwsrc.at[p, 1:].set(keep(lay.in_use, lw_src, lwsrc[p, 1:]))
        rnsrc = rnsrc.at[p, 1:].set(keep(lay.in_use, rn_src, rnsrc[p, 1:]))
        stsrc = stsrc.at[p, 1:].set(keep(lay.in_use, st_src, stsrc[p, 1:]))
        shsrc = shsrc.at[p, 1:].set(keep(lay.in_use, sh_src, shsrc[p, 1:]))
        lhsrc = lhsrc.at[p, 1:].set(keep(lay.in_use, lh_src, lhsrc[p, 1:]))
        etsrc = etsrc.at[p, 1:].set(keep(lay.in_use, et_src, etsrc[p, 1:]))
        trsrc = trsrc.at[p, 1:].set(keep(lay.in_use, tr_src, trsrc[p, 1:]))
        evsrc = evsrc.at[p, 1:].set(keep(lay.in_use, ev_src, evsrc[p, 1:]))
        fco2src = fco2src.at[p, 1:].set(keep(lay.in_use, fco2_src, fco2src[p, 1:]))

    # ==================================================================
    # Commit all updates — single ._replace() call
    # ==================================================================
    return mlcanopy_inst._replace(
        rnet_canopy=rnet,
        stflx_air_canopy=stflx_air,
        stflx_veg_canopy=stflx_veg,
        shflx_canopy=shflx,
        lhflx_canopy=lhflx,
        etflx_canopy=etflx,
        albcan_canopy=albcan,
        swveg_canopy=swveg,
        swvegsun_canopy=swvegsun,
        swvegsha_canopy=swvegsha,
        lwveg_canopy=lwveg,
        lwvegsun_canopy=lwvegsun,
        lwvegsha_canopy=lwvegsha,
        shveg_canopy=shveg,
        shvegsun_canopy=shvegsun,
        shvegsha_canopy=shvegsha,
        lhveg_canopy=lhveg,
        lhvegsun_canopy=lhvegsun,
        lhvegsha_canopy=lhvegsha,
        etveg_canopy=etveg,
        etvegsun_canopy=etvegsun,
        etvegsha_canopy=etvegsha,
        trveg_canopy=trveg,
        evveg_canopy=evveg,
        gppveg_canopy=gppveg,
        gppvegsun_canopy=gppvegsun,
        gppvegsha_canopy=gppvegsha,
        vcmax25veg_canopy=vcmax25veg,
        vcmax25sun_canopy=vcmax25sun,
        vcmax25sha_canopy=vcmax25sha,
        gsveg_canopy=gsveg,
        gsvegsun_canopy=gsvegsun,
        gsvegsha_canopy=gsvegsha,
        windveg_canopy=windveg,
        windvegsun_canopy=windvegsun,
        windvegsha_canopy=windvegsha,
        tlveg_canopy=tlveg,
        tlvegsun_canopy=tlvegsun,
        tlvegsha_canopy=tlvegsha,
        taveg_canopy=taveg,
        tavegsun_canopy=tavegsun,
        tavegsha_canopy=tavegsha,
        laisun_canopy=laisun,
        laisha_canopy=laisha,
        fracminlwp_canopy=fracminlwp,
        swsrc_profile=swsrc,
        lwsrc_profile=lwsrc,
        rnsrc_profile=rnsrc,
        stsrc_profile=stsrc,
        shsrc_profile=shsrc,
        lhsrc_profile=lhsrc,
        etsrc_profile=etsrc,
        trsrc_profile=trsrc,
        evsrc_profile=evsrc,
        fco2src_profile=fco2src,
        swleaf_mean_profile=swleaf_mean,
        lwleaf_mean_profile=lwleaf_mean,
        rnleaf_mean_profile=rnleaf_mean,
        stleaf_mean_profile=stleaf_mean,
        shleaf_mean_profile=shleaf_mean,
        lhleaf_mean_profile=lhleaf_mean,
        etleaf_mean_profile=etleaf_mean,
        trleaf_mean_profile=trleaf_mean,
        evleaf_mean_profile=evleaf_mean,
        fco2_mean_profile=fco2_mean,
        apar_mean_profile=apar_mean,
        gs_mean_profile=gs_mean,
        tleaf_mean_profile=tleaf_mean,
        lwp_mean_profile=lwp_mean,
    )


# ---------------------------------------------------------------------------
# Differentiable forward pass factory
# ---------------------------------------------------------------------------


def make_clm_ml_forward(
    mlcanopy_inst_template: mlcanopy_type,
    bounds: Any,
    num_exposedvegp: int,
    filter_exposedvegp: Sequence[int],
    atm2lnd_inst: Any,
    canopystate_inst: Any,
    soilstate_inst: Any,
    temperature_inst: Any,
    waterstatebulk_inst: Any,
    waterfluxbulk_inst: Any,
    energyflux_inst: Any,
    frictionvel_inst: Any,
    surfalb_inst: Any,
    solarabs_inst: Any,
    wateratm2lndbulk_inst: Any,
    waterdiagnosticbulk_inst: Any,
    step_time: StepTime,
    forcing_bef: AtmForcingRecord,
    forcing_next: AtmForcingRecord,
    *,
    opts: MLOptions,
):
    """Create a differentiable forward function with baked-in structural params.

    Usage::

        forward_fn = make_clm_ml_forward(mlcanopy_inst, bounds, ...)
        grads = jax.grad(forward_fn)(mlcanopy_inst)

    Structural parameters (ncan, ntop, nbot, p) are extracted from the
    template instance as concrete Python ints *before* ``jax.grad`` tracing.
    All other CLM instances are captured via closure and treated as constants
    (not differentiated).  Only ``mlcanopy_inst`` is the differentiation
    variable.

    Returns:
        A function ``forward(mlcanopy_inst) -> scalar`` suitable for
        ``jax.grad``.
    """
    _p = int(filter_exposedvegp[0])
    grid, default_params = patch_setup(_p, mlcanopy_inst_template)

    def forward(
        mlcanopy_inst: mlcanopy_type,
        params: CanopyParams = default_params,
        forcing_bef: AtmForcingRecord = forcing_bef,
        forcing_next: AtmForcingRecord = forcing_next,
    ) -> jnp.ndarray:
        """Differentiable CLM-ML forward pass. Returns scalar loss. Differentiate with
        respect to the state (``argnums=0``), the parameters (``argnums=1``), or the forcing
        at the previous or next step (``argnums=2, 3``; used by the time interpolation)."""
        inst = MLCanopyFluxes(
            bounds=bounds,
            num_exposedvegp=num_exposedvegp,
            filter_exposedvegp=filter_exposedvegp,
            atm2lnd_inst=atm2lnd_inst,
            canopystate_inst=canopystate_inst,
            soilstate_inst=soilstate_inst,
            temperature_inst=temperature_inst,
            waterstatebulk_inst=waterstatebulk_inst,
            waterfluxbulk_inst=waterfluxbulk_inst,
            energyflux_inst=energyflux_inst,
            frictionvel_inst=frictionvel_inst,
            surfalb_inst=surfalb_inst,
            solarabs_inst=solarabs_inst,
            mlcanopy_inst=mlcanopy_inst,
            wateratm2lndbulk_inst=wateratm2lndbulk_inst,
            waterdiagnosticbulk_inst=waterdiagnosticbulk_inst,
            step_time=step_time,
            forcing_bef=forcing_bef,
            forcing_next=forcing_next,
            initialize=False,
            grid=grid,
            params=params,
            opts=opts,
        )
        p = grid.p
        use = layers(inst, p).in_use
        # Scalar loss: sum of sensible + latent heat flux profiles (layers 1..ncan)
        loss = jnp.sum(jnp.where(use, inst.shair_profile[p, 1:], 0.0)) + jnp.sum(
            jnp.where(use, inst.etair_profile[p, 1:], 0.0)
        )
        return loss

    return forward
