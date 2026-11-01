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

The legacy adapters of processes 3-9 imported below (``GetAtmForcing``, ``SolarRadiation``,
...) are no longer called by the time step (Phase 7.2 process 10b, decision C20): they are
re-exported for the process replay (``tests/regression/process_io.py``), which looks the
recorded routines up by name in this module.

Original Fortran module: MLCanopyFluxesMod
Fortran lines 1-400
"""

from __future__ import annotations

from typing import Any, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from clm_ml.io.rsl_tables import default_rsl_tables
from clm_ml.legacy.atm_forcing import (
    AtmForcingRecord,
    AtmForcingWindow,
    GetAtmForcing,  # noqa: F401
)
from clm_ml.legacy.canopy import (
    CanopyFluxesDiagnostics,
    MergeSunShade,
    PlantAreaProfiles,
    SoilSurfaceHumidity,
    WriteSubstepMeans,
    container_with_substep,
    substep_from_container,
    substep_inputs,
    substep_options,
)
from clm_ml.legacy.canopy_water import (  # noqa: F401
    CanopyEvaporation,
    CanopyInterception,
    CanopyWettedFraction,
)
from clm_ml.legacy.flux_profile import FluxProfileSolution  # noqa: F401
from clm_ml.legacy.integrators import RungeKuttaUpdate  # noqa: F401
from clm_ml.legacy.leaf_boundary_layer import LeafBoundaryLayer  # noqa: F401
from clm_ml.legacy.longwave import LongwaveRadiation  # noqa: F401
from clm_ml.legacy.nitrogen_profile import CanopyNitrogenProfile  # noqa: F401
from clm_ml.legacy.photosynthesis import LeafPhotosynthesis  # noqa: F401
from clm_ml.legacy.plant_hydraulics import (  # noqa: F401
    LeafHeatCapacity,
    LeafWaterPotential,
    PlantResistance,
    SoilResistance,
)
from clm_ml.legacy.solar_radiation import SolarRadiation  # noqa: F401
from clm_ml.legacy.turbulence import CanopyTurbulence  # noqa: F401
from clm_ml.physics import canopy
from clm_ml.physics.atm_forcing import clm_forcing_to_ml
from clm_ml.physics.math import is_concrete
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
from multilayer_canopy.MLCanopyParamsMod import CanopyParams, patch_setup
from multilayer_canopy.MLclm_varcon import mmh2o, rgas  # noqa: F401
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun, nleaf  # noqa: F401
from multilayer_canopy.MLinitVerticalMod import (  # noqa: F401
    getPADparameters,
    initVerticalProfiles,
    initVerticalStructure,
)
from multilayer_canopy.MLLayersMod import layers

# ---------------------------------------------------------------------------
# JIT-compiled bef-state copy helper
# ---------------------------------------------------------------------------


def _float_if_concrete(x):
    """``float(x)`` in a normal run (Python-float arithmetic, as before); ``x`` itself when
    it is traced (a chunk of steps compiled with jax.jit)."""
    return float(x) if is_concrete(x) else x


#: One sub-step compiled as one program (normal runs; a compiled chunk traces
#: ``canopy.substep`` itself)
_substep = jax.jit(canopy.substep, static_argnames=("options", "c3"))


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
    p = grid.p
    mlcanopy_inst = PlantAreaProfiles(mlcanopy_inst, canopystate_inst, p, opts.layer_pai_floor)

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

    # Soil surface relative humidity — Fortran lines 259-262 (patch hierarchy as numpy, so
    # the column index is concrete when the state is traced)
    mlcanopy_inst = SoilSurfaceHumidity(
        mlcanopy_inst, soilstate_inst, temperature_inst, p, int(np.asarray(patch.column)[p])
    )

    # ==================================================================
    # Multilayer canopy time-stepping loop — Fortran lines 261-330
    # ==================================================================
    # One sub-step is one function, clm_ml.physics.canopy.substep, on 0-based arrays (the
    # container is read once before the loop and written once after it). It is driven by
    # lax.scan (with jax.checkpoint, so backward memory does not grow with the number of
    # sub-steps) when the step is traced (a compiled chunk, jax.grad), and by a Python loop
    # over the compiled sub-step in a normal run.
    # ==================================================================
    sub_options = substep_options(opts)
    c3 = bool(grid.c3)
    times = tuple(
        jnp.asarray(t, dtype=jnp.float64)
        for t in (calday_interp_bef, calday_interp_cur, calday_interp_next)
    )
    inputs = substep_inputs(mlcanopy_inst, p, forcing_window, times)
    # The first (initializing) step keeps the package's psihat tables as a constant until the
    # setup leaves the containers (Phase 7.4); the time step takes them as an input (C22)
    tables = default_rsl_tables()

    if masterproc and is_concrete(nstep) and nstep == 1:
        print(
            f"MLCanopyFluxes: starting ML sub-steps (num_ml_steps={num_ml_steps}, "
            f"stages={canopy.n_stages(sub_options.integrator)}, num_mlcan={num_mlcan})",
            flush=True,
        )

    # Keep heartbeat output bounded while still proving forward progress.
    progress_stride = max(1, num_ml_steps // 10)

    # Calendar day of each ML sub-step — Fortran lines 263-270
    _dtime_ml = opts.dtime_ml
    _offset = 0.0 if met_type in (0, 2) else 0.5  # met_type 3: mid-sub-step
    caldays_ml = [
        curr_calday_beg + (float(k) - _offset) * (_dtime_ml / 86400.0)
        for k in range(1, num_ml_steps + 1)
    ]

    start = substep_from_container(mlcanopy_inst, p)
    carry = (start, canopy.zero_sums(canopy.substep_fluxes(start, inputs.layers)))
    _traced = not is_concrete(
        *jax.tree_util.tree_leaves((mlcanopy_inst, forcing_bef, forcing_cur, forcing_next))
    )
    if _traced:
        import os as _os

        def _scan_body(carry, calday_ml):
            return (
                canopy.substep(
                    carry, calday_ml, inputs, params, tables, options=sub_options, c3=c3
                ),
                None,
            )

        if _os.environ.get("CLM_ML_NO_CHECKPOINT", "0") != "1":
            _scan_body = jax.checkpoint(_scan_body)
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
            carry = _substep(carry, calday_ml, inputs, params, tables, options=sub_options, c3=c3)
    end, sums = carry
    mlcanopy_inst = container_with_substep(mlcanopy_inst, p, end)

    # Averages over the sub-steps — Fortran MLTimeStepFluxIntegration
    means = canopy.average(sums, num_ml_steps)
    mlcanopy_inst = WriteSubstepMeans(mlcanopy_inst, p, means)

    # End multilayer time-stepping loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Canopy-level diagnostics — Fortran line 346
    # ------------------------------------------------------------------
    mlcanopy_inst = CanopyFluxesDiagnostics(mlcanopy_inst, p, means, opts=opts)

    # ------------------------------------------------------------------
    # Merge sun/shade leaf temperature and water potential — Fortran lines 355-375
    # State is prognostic; sun/shade fraction changes between CLM steps,
    # so merge to a layer-mean before the next timestep.
    # ------------------------------------------------------------------
    mlcanopy_inst = MergeSunShade(mlcanopy_inst, p)

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
