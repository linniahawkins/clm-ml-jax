"""One time step of one patch on the 0-based step state.

``step(state, inputs, constants, params, options) -> StepState``: the soil routines of
CLM-ml v2's ``clm_driver`` around the multilayer canopy's ``MLCanopyFluxes``, built from the
processes in ``clm_ml.processes`` (soil, plant area and hydraulics, the canopy sub-steps, the
averages, totals and the sunlit/shaded merge). Pure: everything the step reads is an argument
and everything it changes is returned.

- :class:`State` (``clm_ml.state``): the prognostic variables carried from step to step.
- :class:`StepInputs`: this step's forcing and land data (tower meteorology, the CLM history
  record, the forcing records of the previous and next steps for the time interpolation) and
  its time.
- :class:`PatchConstants`: the patch's values that every step reads and none changes: the
  layer grid, PFT and soil properties, location (set up once; slots the step does not
  compute hold ``spval``).
- :class:`StepOptions`: the static choices (code paths, step lengths).

Fortran reference: CLM-ml v2 ``clm_driver.F90::clm_drv`` and
``MLCanopyFluxesMod.F90::MLCanopyFluxes`` (with ``GetCLMVar``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.checks import (  # the conservation checks' indices
    BALANCE_CHECKS,
    CANOPY_ENERGY_1,
    PLANT_AREA,
    SOIL_TEMPERATURE,
    SOIL_WATER,
)
from clm_ml.processes import canopy, soil
from clm_ml.processes.atm_forcing import ForcingRecord, clm_forcing_to_ml
from clm_ml.processes.constants import denh2o, denice, mmh2o
from clm_ml.processes.hydrology import soil_hydrology
from clm_ml.processes.layers import keep
from clm_ml.processes.leaf_heat_capacity import leaf_heat_capacity
from clm_ml.processes.plant_hydraulics import (
    RootParameters,
    fine_root_biomass,
    plant_resistance,
    root_water_uptake,
    root_zone_by_conductance,
    soil_resistance,
)
from clm_ml.processes.soil_water import ice_impedance, soil_water_fractions
from clm_ml.processes.surface_runoff import (
    ground_vapor_fluxes,
    microtopography,
    pond_fraction,
    pond_threshold,
)
from clm_ml.state import CanopyState, Diagnostics, PatchLayers, SoilState, State, StepState


class StepOptions(NamedTuple):
    """The step's static choices (hashable: an argument of compiled functions)."""

    substep: canopy.SubstepOptions  # the canopy sub-step's choices
    step_seconds: int  # model time step (s)
    layer_pai_floor: float  # minimum plant area of a leafy layer (m2/m2; not in CLM-ml v2)
    flux_profile_type: int  # canopy totals: fluxes at the top layer (1)
    n_soil_adjust: int  # soil levels multiplied by the history's adjustment factor
    n_soil: int  # hydrologically active soil levels (nlevsoi)
    # Orbital parameters for the solar zenith angle (clm_varorb, fixed for the run)
    eccen: float
    mvelpp: float
    lambm0: float
    obliqr: float
    #: jax.checkpoint on each sub-step too (model.yaml numerics.checkpoint_substeps, which
    #: follows the device): gradients keep one sub-step's intermediate values instead of a
    #: step's (less memory; much slower on the CPU)
    checkpoint_substeps: bool = False
    #: soil water (model.yaml soil.water): "prescribed" by the forcing at every step, or
    #: "predicted" by CLM's soil hydrology (not in CLM-ml v2)
    soil_water: str = "prescribed"
    #: passes of the Richards solver's loop with predicted soil water: the largest number
    #: of trial sub-steps a model step can take (soil_water.max_richards_trials)
    richards_trials: int = 0
    #: with predicted soil water, whether the roots move water from wetter to drier soil
    #: levels (model.yaml soil.root_redistribution)
    root_redistribution: bool = True
    #: soil water freezes and thaws (model.yaml soil.freeze_thaw; predicted soil water)
    freeze_thaw: bool = False
    #: "stabilized" | "exact": the derivative of the freezing (soil.phase_change)
    phase_change_derivative: str = "stabilized"
    #: with predicted soil water, the functions of each soil level's soil-to-root
    #: conductance (model.yaml soil.root_conductance; plant_hydraulics.ROOT_CONDUCTANCE)
    root_conductance: str = "clm_ml"
    #: with predicted soil water, the fine root biomass: the patch's prescribed value or
    #: from the leaf area (model.yaml soil.root_biomass; plant_hydraulics.ROOT_BIOMASS)
    root_biomass: str = "prescribed"

    @property
    def n_substeps(self) -> int:
        return int(self.step_seconds / self.substep.dtime)


class StepInputs(NamedTuple):
    """One step's forcing, land data and time (0-based bands, no patch axis)."""

    calday: Array  # () calendar day at the end of the step
    calday_beg: Array  # () calendar day at the beginning of the step
    # Tower (or gridcell) meteorology of this step, as TowerMetMod.TowerMet
    uref: Array  # () wind speed, m/s
    tref: Array  # () air temperature, K
    qref: Array  # () specific humidity, kg/kg
    pref: Array  # () air pressure, Pa
    lwsky: Array  # () atmospheric longwave, W/m2
    rain: Array  # () rainfall, kg H2O/m2/s
    snow: Array  # () snowfall, kg H2O/m2/s
    solad: Array  # (band,) direct beam solar radiation, W/m2
    solai: Array  # (band,) diffuse solar radiation, W/m2
    pco2: Array  # () CO2 partial pressure, Pa
    po2: Array  # () O2 partial pressure, Pa
    # CLM history record of this step (clmDataMod.ClmHistoryRecord)
    elai: Array  # () leaf area index, m2/m2
    esai: Array  # () stem area index, m2/m2
    h2osoi: Array  # (n_soil,) volumetric soil water
    h2osoi_factor: Array  # () soil water adjustment factor
    # Forcing records of the previous and next steps (3-point time interpolation)
    forcing_bef: ForcingRecord
    forcing_next: ForcingRecord


class MetInputs(NamedTuple):
    """One step's time and meteorology, per gridcell (the :class:`StepInputs` fields that
    every patch of a gridcell shares). A chunk holds them stacked along a leading step axis,
    then a gridcell axis (``calday``/``calday_beg`` may have no gridcell axis)."""

    calday: Array
    calday_beg: Array
    uref: Array
    tref: Array
    qref: Array
    pref: Array
    lwsky: Array
    rain: Array
    snow: Array
    solad: Array  # (band,)
    solai: Array  # (band,)
    pco2: Array
    po2: Array
    forcing_bef: ForcingRecord
    forcing_next: ForcingRecord


class LandInputs(NamedTuple):
    """One step's prescribed land data, per patch (the :class:`StepInputs` fields of the CLM
    history record or the land-data files). A chunk holds them stacked along a leading step
    axis, then a patch axis."""

    elai: Array
    esai: Array
    h2osoi: Array  # (n_soil,)
    h2osoi_factor: Array  # 1 when the history has no adjustment factor


def step_inputs(met: MetInputs, land: LandInputs) -> StepInputs:
    """A patch's :class:`StepInputs` from its gridcell's meteorology and its land data."""
    return StepInputs(**met._asdict(), **land._asdict())


class PatchConstants(NamedTuple):
    """The patch's values every step reads and no step changes (0-based, no patch axis)."""

    layers: PatchLayers
    # Leaf and stem area fractions of the layers
    dlai_frac: Array  # (layer,) (dlai_frac_profile)
    dsai_frac: Array  # (layer,) (dsai_frac_profile)
    zref: Array  # () reference height, m (zref_forcing)
    ztop: Array  # () canopy height, m (ztop_canopy)
    root_biomass: Array  # () fine root biomass, g/m2 (root_biomass_canopy)
    rootfr: Array  # (n_soil,) root fraction of each soil level (soilstate.rootfr_patch)
    # The soil column (soil levels 0-based from the top)
    dz: Array  # (soil_level,) level thickness, m (site.dz_col)
    z: Array  # (soil_level,) level depth, m (site.z_col)
    zi: Array  # (soil_level,) depth of the level's bottom interface, m (site.zi_col)
    soil_dz: Array  # () depth of the first level's middle below the surface, m (soil_dz_soil)
    nbedrock: Array  # () 1-based index of the deepest level above bedrock (site.nbedrock_col)
    watsat: Array  # (soil_level,) porosity
    sucsat: Array  # (soil_level,) saturated suction, mm
    bsw: Array  # (soil_level,) Clapp and Hornberger b
    hksat: Array  # (soil_level,) saturated hydraulic conductivity, mm/s
    tkmg: Array  # (soil_level,) thermal conductivity of the soil minerals
    tkdry: Array  # (soil_level,) thermal conductivity of dry soil
    csol: Array  # (soil_level,) heat capacity of the soil solids
    albsat: Array  # (band,) saturated soil albedo of the soil colour
    albdry: Array  # (band,) dry soil albedo of the soil colour
    latdeg: Array  # () latitude, degrees
    londeg: Array  # () longitude, degrees
    # Predicted soil water only (CLM's surface data FMAX and SLOPE)
    wtfact: Array  # () largest saturated fraction of the surface
    topo_slope: Array  # () topographic slope, degrees


def interpolation_days(calday, calday_beg, options: StepOptions) -> tuple:
    """Calendar days of the previous, current and next forcing records (centred in their
    steps for the 3-point interpolation; 0 without interpolation).

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes.
    """
    days = options.step_seconds / 86400.0
    if options.substep.interpolation == "none":
        return 0.0, 0.0, 0.0
    # The calendar day at the end of a step is below the day at its start where the calendar
    # day starts again (a step ending at 0Z 1 January; after 0Z 31 December of a Gregorian
    # leap year): the step's end is then its start plus a step.
    # Differs from CLM-ml v2: its mean of the two calendar days puts such a step's middle
    # half a year (or half a day) away from its sub-steps.
    end = jnp.where(calday < calday_beg, calday_beg + days, calday)
    cur = 0.5 * (end + calday_beg)
    return cur - days, cur, cur + days


def substep_days(calday_beg, options: StepOptions) -> Array:
    """Calendar day of each sub-step (the sub-step's middle for the 3-point interpolation).

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes.
    """
    offset = 0.0 if options.substep.interpolation == "none" else 0.5
    dt = options.substep.dtime
    return jnp.stack(
        [
            jnp.asarray(calday_beg + (float(k) - offset) * (dt / 86400.0), dtype=jnp.float64)
            for k in range(1, options.n_substeps + 1)
        ]
    )


def solar_zenith(calday_beg, latdeg, londeg, options: StepOptions) -> Array:
    """Solar zenith angle at the beginning of the step (cosine at least 0.01).

    Fortran reference: MLCanopyFluxesMod.F90::GetCLMVar.
    """
    from clm_ml.processes.constants import rpi as pi
    from clm_ml.processes.orbit import shr_orb_cosz, shr_orb_decl

    declin, _ = shr_orb_decl(
        calday_beg, options.eccen, options.mvelpp, options.lambm0, options.obliqr
    )
    lat = latdeg * pi / 180.0
    lon = londeg * pi / 180.0
    coszen = shr_orb_cosz(calday_beg, lat, lon, declin)
    return jnp.arccos(jnp.maximum(jnp.asarray(0.01), jnp.asarray(coszen)))


def _fill(like: Array) -> Array:
    """``spval`` shaped as ``like``: the value of slots the step does not compute (above the
    top layer in use, soil levels below bedrock), as CLM-ml v2 fills them."""
    return jnp.full(jnp.shape(like), canopy.SPVAL, dtype=jnp.float64)


def _record(errors: Array, check: int, err) -> Array:
    """``errors`` with ``|err|`` (largest element) recorded for ``check``."""
    return errors.at[check].max(jax.lax.stop_gradient(jnp.max(jnp.abs(jnp.asarray(err)))))


def soil_level_ice(ice, n_soil: int):
    """The ice of each hydrologically active soil level, ``(n_soil,)``: level j's own
    (index j, 0-based), which stops root uptake from a frozen level.

    Fortran reference: MLPlantHydraulicsMod.F90::SoilResistance (``h2osoi_ice(c,j)``)."""
    return ice[:n_soil]


def step(
    state: State,
    inputs: StepInputs,
    constants: PatchConstants,
    params,
    tables,
    *,
    options: StepOptions,
    c3: bool,
) -> StepState:
    """One time step of one patch: the step's results and the state after it.

    In the order of ``clm_drv``: the CLM history record's LAI, SAI and soil water (with
    predicted soil water: the state's soil water); soil
    albedo, evaporative resistance, thermal and hydraulic properties; the canopy's forcing,
    solar zenith angle, plant area, soil and plant resistance, leaf heat capacity and soil
    surface humidity; the canopy sub-steps (``canopy.substep``; a ``lax.scan``, with
    ``jax.checkpoint`` on each sub-step if ``options.checkpoint_substeps``); the averages
    over the sub-steps, the canopy totals, the sunlit/shaded merge; the soil temperature;
    with predicted soil water, the soil hydrology (``hydrology.soil_hydrology``), driven by
    the step means of the water reaching the ground, the ground evaporation and the
    transpiration: the canopy of a step sees the soil water left by the step before, as in
    CLM. The conservation checks of the step are
    recorded in ``diagnostics.balance_error`` (checked on the host). ``params``: the patch's
    PFT parameters (``CanopyParams``); ``tables``: the psihat lookup tables
    (``clm_ml.processes.turbulence.RSLTables``, one copy for all patches and steps of a chunk);
    ``c3`` (static): the photosynthetic pathway.

    Fortran reference: clm_driver.F90::clm_drv; MLCanopyFluxesMod.F90::MLCanopyFluxes,
    GetCLMVar.
    """
    k = constants
    lay = k.layers
    use = lay.in_use
    t_soil = state.soil.temperature
    n_soil = options.n_soil
    zero = jnp.zeros((), t_soil.dtype)

    predicted = options.soil_water == "predicted"
    freezes = predicted and options.freeze_thaw
    below = jnp.zeros(k.dz.shape[0] - n_soil, t_soil.dtype)
    if freezes:
        # the state's liquid water and ice. The fraction of the ground under ponded water
        # is found first, as in CLM: a pond too small to keep joins the top level's water.
        # Not in CLM-ml v2.
        pond = pond_fraction(
            state.soil.surface_water, state.soil.liquid_water[0],
            microtopography(
                k.topo_slope, params.hydrology.slopebeta, params.hydrology.slopemax
            ),
            float(options.step_seconds),
        )  # fmt: skip
        liq = jnp.concatenate([pond.h2osoi_liq_top[None], state.soil.liquid_water[1:], below])
        ice_state = jnp.concatenate([state.soil.ice, below])
        water = soil.SoilWater(liq / (k.dz * denh2o) + ice_state / (k.dz * denice), liq, ice_state)
    elif predicted:
        # the state's soil water (all liquid; no water in the levels below the
        # hydrologically active ones)
        liq = jnp.concatenate([state.soil.liquid_water, below])
        water = soil.SoilWater(liq / (k.dz * denh2o), liq, jnp.zeros_like(liq))
    else:
        # CLM history record: LAI, SAI (patch) and soil water (column) — clmDataMod::clmData
        water = soil.soil_water_from_history(
            inputs.h2osoi, inputs.h2osoi_factor, k.dz, k.watsat, k.nbedrock,
            n_soil=n_soil, n_adjust=options.n_soil_adjust,
        )  # fmt: skip
    liq, ice = water.h2osoi_liq, water.h2osoi_ice

    # Soil albedo, evaporative resistance, thermal and hydraulic properties (no snow, no
    # surface water: clm_drv sets them to 0)
    albgrd, albgri = soil.soil_albedo(water.h2osoi_vol[0], k.albsat, k.albdry)
    _, soilresis = soil.soil_evaporative_resistance(
        liq[0], ice[0], t_soil[0], k.dz[0], k.watsat[0], k.sucsat[0], k.bsw[0]
    )
    # the snow lying on the ground adds its heat capacity to the top level's (freeze and
    # thaw; no snow otherwise)
    snow_before = state.soil.snow if freezes else zero
    thermal = soil.soil_thermal_properties(
        t_soil, liq, ice, zero, snow_before, k.dz, k.z, k.zi, k.watsat, k.tkmg, k.tkdry,
        k.csol, k.nbedrock,
    )  # fmt: skip
    hk_l, smp_l = soil.soil_hydraulic_properties(
        liq[:n_soil], k.dz[:n_soil], k.watsat[:n_soil], k.hksat[:n_soil], k.sucsat[:n_soil],
        k.bsw[:n_soil], k.nbedrock, _fill(k.dz[:n_soil]), _fill(k.dz[:n_soil]),
    )  # fmt: skip

    # The canopy's forcing of this step — GetCLMVar
    u, t, q, p, co2, solad, solai, lw = clm_forcing_to_ml(
        inputs.uref, zero, inputs.tref, inputs.qref, inputs.pref, inputs.pco2,
        inputs.solad, inputs.solai, inputs.lwsky,
    )  # fmt: skip
    forcing_cur = ForcingRecord(u, t, q, p, co2, solad, solai, lw)
    o2ref = inputs.po2 / inputs.pref * 1.0e3  # Pa -> mmol/mol
    solar_zen = solar_zenith(inputs.calday_beg, k.latdeg, k.londeg, options)

    # Plant area of the layers for the step's LAI and SAI
    area = canopy.plant_area(
        lay, k.dlai_frac, k.dsai_frac, inputs.elai, inputs.esai, options.layer_pai_floor
    )
    fill_layer = _fill(lay.dz)
    dlai = keep(use, area.dlai, fill_layer)
    dsai = keep(use, area.dsai, fill_layer)
    dpai = keep(use, area.dpai, fill_layer)
    errors = _record(jnp.zeros(len(BALANCE_CHECKS), t_soil.dtype), PLANT_AREA, area.error)

    # Plant hydraulics and leaf heat capacity (every layer slot)
    # The soil as the canopy sees it. Prescribed soil water: CLM-ml v2. Predicted: the
    # levels weighted by their conductance (not in CLM-ml v2), with CLM-ml's or CLM's root
    # functions and the prescribed fine root biomass or CLM's from the leaf area.
    if predicted:
        if options.root_biomass == "from_leaf_area":
            root_biomass = fine_root_biomass(area.lai, params.sla_top, params.root_to_leaf_ratio)
        else:
            root_biomass = k.root_biomass
        # Freezing soil with CLM's roots: as CLM, no level is left out; the ice lowers the
        # level's conductivity (CLM's ice impedance), and its liquid water's potential
        # (the freezing-point depression) closes the root. CLM-ml's roots keep CLM-ml v2's
        # rule: no uptake from a level with any ice.
        clm_frozen = freezes and options.root_conductance == "clm"
        hk_roots = hk_l
        if clm_frozen:
            icefrac, _ = soil_water_fractions(ice[:n_soil], k.watsat[:n_soil], k.dz[:n_soil])
            hk_roots = hk_l * ice_impedance(icefrac, params.hydrology.e_ice)
        roots = root_zone_by_conductance(
            options.root_conductance, hk_roots, k.hksat[:n_soil], smp_l, k.dz[:n_soil],
            k.z[:n_soil], k.rootfr, soil_level_ice(ice, n_soil), k.nbedrock, area.lai,
            inputs.esai, root_biomass,
            RootParameters(*(getattr(params, name) for name in RootParameters._fields)),
            exclude_frozen=not clm_frozen,
        )  # fmt: skip
    else:
        roots = soil_resistance(
            hk_l, smp_l, k.dz[:n_soil], k.rootfr, soil_level_ice(ice, n_soil), k.nbedrock,
            area.lai, k.root_biomass, params.root_radius, params.root_density,
            params.root_resistivity,
        )  # fmt: skip
    lsc = plant_resistance(dpai, roots.rsoil, params.plant_conductance)
    cpleaf = leaf_heat_capacity(dpai, params.sla_top)
    rhg = canopy.soil_surface_humidity(smp_l[0], t_soil[0])

    # The canopy sub-steps
    times = tuple(
        jnp.asarray(d, dtype=jnp.float64)
        for d in interpolation_days(inputs.calday, inputs.calday_beg, options)
    )
    sub_inputs = canopy.SubstepInputs(
        layers=lay, forcing_bef=inputs.forcing_bef, forcing_cur=forcing_cur,
        forcing_next=inputs.forcing_next, time_bef=times[0], time_cur=times[1],
        time_next=times[2], zref=k.zref, solar_zen=solar_zen, dlai=dlai, dsai=dsai, dpai=dpai,
        albsoib=albgrd, albsoid=albgri, lai=area.lai, sai=area.sai,
        tacclim=state.canopy.acclimation_temperature, ztop=k.ztop, o2ref=o2ref,
        qflx_rain=inputs.rain, qflx_snow=inputs.snow, rhg=rhg, soilres=soilresis,
        soil_t=t_soil[0], soil_dz=k.soil_dz, soil_tk=thermal.thk[0], cpleaf=cpleaf, lsc=lsc,
        psis=roots.psis,
    )  # fmt: skip
    c = state.canopy
    n_slots = lay.dz.shape[-1]
    start = canopy.initial_substep(
        n_slots, canopy.stage_slots(options.substep.integrator), len(BALANCE_CHECKS)
    )._replace(
        tair_profile=c.air_temperature,
        eair_profile=c.vapor_pressure,
        cair_profile=c.co2,
        tleaf_leaf=c.leaf_temperature,
        lwp_leaf=c.leaf_water_potential,
        h2ocan_profile=c.intercepted_water,
        tg_soil=c.ground_temperature,
        balance_error=errors,
    )
    sub_options = options.substep

    def body(carry, calday):
        return (
            canopy.substep(carry, calday, sub_inputs, params, tables, options=sub_options, c3=c3),
            None,
        )

    if options.checkpoint_substeps:
        body = jax.checkpoint(body)
    carry = (start, canopy.zero_sums(canopy.substep_fluxes(start, lay)))
    (end, sums), _ = jax.lax.scan(body, carry, substep_days(inputs.calday_beg, options))

    # Averages over the sub-steps (layers in use, levels up to the top layer in use; the
    # other slots keep the sub-step values) — MLTimeStepFluxIntegration
    means = canopy.average(sums, options.n_substeps)
    up_to_top = jnp.arange(lay.dz.shape[0] + 1) <= lay.n_layers

    # Canopy totals (the averages through a barrier, as Fortran stores them first)
    leaves = canopy.CanopyLeaves(
        dpai=dpai, fracsun=end.fracsun_profile, fwet=end.fwet_profile, fdry=end.fdry_profile,
        wind=end.wind_profile, tair=end.tair_profile, vcmax25=end.vcmax25_profile,
        tleaf=end.tleaf_leaf, lwp=end.lwp_leaf, vcmax25_leaf=end.vcmax25_leaf,
    )  # fmt: skip
    totals = canopy.canopy_totals(
        lay,
        jax.lax.optimization_barrier(means),
        leaves,
        area.lai,
        area.sai,
        flux_profile_type=options.flux_profile_type,
    )
    errors = end.balance_error
    for i in range(4):
        errors = _record(errors, CANOPY_ENERGY_1 + i, totals.errors[i])

    # Sunlit and shaded leaves merged to their layer mean (the state is prognostic)
    merged = canopy.merge_sun_shade(lay, dpai, end.fracsun_profile, end.tleaf_leaf, end.lwp_leaf)

    # Soil temperature — MLSoilTemperatureMod::SoilTemperature (thermal properties of
    # before the canopy fluxes: unchanged in between)
    gsoi = means.gsoi
    t_soil_new, soil_error = soil.soil_temperature(
        t_soil, thermal.tk, thermal.cv, k.z, gsoi, options.step_seconds
    )
    dtime = float(options.step_seconds)
    if freezes:
        # Freeze and thaw, after the temperature solution (CLM's Phasechange_beta). The
        # snow that reached the ground in the step joins the snow lying there first. The
        # ground heat flux is given to the soil, so the routine has no surface feedback
        # (dhsdT = 0) and takes the heat capacity of the temperature solution: what the
        # levels lose in heat is the latent heat. Not in CLM-ml v2.
        snow = snow_before + means.qflx_tflsnow * dtime
        change = soil.phase_change(
            t_soil_new[:n_soil], liq[:n_soil], ice[:n_soil], snow, zero,
            dtime / thermal.cv[:n_soil], zero, pond.frac_h2osfc, k.watsat[:n_soil],
            k.sucsat[:n_soil], k.bsw[:n_soil], k.dz[:n_soil], dtime,
            derivative=options.phase_change_derivative,
        )  # fmt: skip
        heat = jnp.sum(thermal.cv[:n_soil] * (change.t_soisno - t_soil_new[:n_soil])) / dtime
        soil_error = soil_error - (heat + change.xmf)
        t_soil_new = jnp.concatenate([change.t_soisno, t_soil_new[n_soil:]])
    errors = _record(errors, SOIL_TEMPERATURE, soil_error)

    # Soil hydrology (predicted soil water): the step means of the canopy's fluxes move the
    # soil water. The ground's water vapour flux is evaporation from the soil and from the
    # pond alike when positive, dew on both when negative; snow reaching the ground enters
    # as liquid water (with freeze and thaw: when the snow lying on the ground melts, and
    # the soil's part of the vapour flux is taken from and given to liquid water and ice
    # as CLM does). Each level gives the roots its share of the transpiration plus the
    # water the roots move between the levels (the start-of-step soil water potentials);
    # without that redistribution no level receives water from the roots: the levels that
    # would are left out and the others give the transpiration in the proportions they
    # had. Not in CLM-ml v2.
    unset = _fill(zero)
    frozen_names = (
        "h2osoi_ice_soil", "h2osno_soil", "qflx_snomelt_soil", "xmf_soil", "frost_table_soil",
        "zwt_perched_soil", "qflx_drain_perched_soil", "qflx_ice_runoff_soil",
        "qflx_sublimation_soil",
    )  # fmt: skip
    frozen_diagnostics = {name: unset for name in frozen_names}
    frozen_diagnostics["h2osoi_ice_soil"] = _fill(liq)
    if predicted:
        hp = params.hydrology
        micro_sigma = microtopography(k.topo_slope, hp.slopebeta, hp.slopemax)
        ground_evaporation = means.etsoi * mmh2o  # mol H2O m-2 s-1 -> mm s-1
        transpiration = totals.trveg * mmh2o
        root_uptake = root_water_uptake(
            roots.soil_et_loss, roots.redistribution * (1.0e-3 * mmh2o), transpiration,
            options.root_redistribution,
        )  # fmt: skip
        soil_constants = (
            k.watsat[:n_soil], k.hksat[:n_soil], k.sucsat[:n_soil], k.bsw[:n_soil],
            k.dz[:n_soil], k.z[:n_soil], jnp.concatenate([zero[None], k.zi[:n_soil]]),
            k.nbedrock, k.wtfact, micro_sigma, pond_threshold(micro_sigma, hp.pc),
            k.topo_slope, dtime, hp,
        )  # fmt: skip
        if freezes:
            ground_input = means.qflx_tflrain + change.qflx_snomelt
            # frost or dew by the ground temperature the step began with (CLM's t_grnd)
            vapor = ground_vapor_fluxes(
                ground_evaporation, change.h2osoi_liq[0], change.h2osoi_ice[0],
                c.ground_temperature, pond.frac_h2osfc, dtime,
            )  # fmt: skip
            hyd = soil_hydrology(
                change.h2osoi_liq, change.h2osoi_ice, pond.h2osfc,
                state.soil.water_table_depth, ground_input,
                vapor.qflx_liqevap_from_top_layer, ground_evaporation,
                vapor.qflx_liqdew_to_top_layer, root_uptake, *soil_constants,
                qflx_soliddew_to_top_layer=vapor.qflx_soliddew_to_top_layer,
                qflx_solidevap_from_top_layer=vapor.qflx_solidevap_from_top_layer,
                t_soisno=change.t_soisno, frost_table=state.soil.frost_table_depth,
                zwt_perched=state.soil.perched_water_table_depth, pond=pond,
                fixed_trials=options.richards_trials,
            )  # fmt: skip
            soil_state = SoilState(
                temperature=t_soil_new,
                liquid_water=hyd.h2osoi_liq,
                surface_water=hyd.h2osfc,
                water_table_depth=hyd.zwt,
                ice=hyd.h2osoi_ice,
                snow=change.h2osno,
                frost_table_depth=hyd.frost_table,
                perched_water_table_depth=hyd.zwt_perched,
            )
            frozen_diagnostics = dict(
                h2osoi_ice_soil=ice,
                h2osno_soil=change.h2osno,
                qflx_snomelt_soil=change.qflx_snomelt,
                xmf_soil=change.xmf,
                frost_table_soil=hyd.frost_table,
                zwt_perched_soil=hyd.zwt_perched,
                qflx_drain_perched_soil=hyd.qflx_drain_perched,
                qflx_ice_runoff_soil=hyd.qflx_ice_runoff_xs,
                qflx_sublimation_soil=(1.0 - pond.frac_h2osfc)
                * vapor.qflx_solidevap_from_top_layer,
            )
        else:
            ground_input = means.qflx_tflrain + means.qflx_tflsnow
            hyd = soil_hydrology(
                liq[:n_soil], ice[:n_soil], state.soil.surface_water,
                state.soil.water_table_depth, ground_input,
                jnp.maximum(ground_evaporation, 0.0), ground_evaporation,
                jnp.maximum(-ground_evaporation, 0.0), root_uptake, *soil_constants,
                fixed_trials=options.richards_trials,
            )  # fmt: skip
            soil_state = state.soil._replace(
                temperature=t_soil_new,
                liquid_water=hyd.h2osoi_liq,
                surface_water=hyd.h2osfc,
                water_table_depth=hyd.zwt,
            )
        errors = _record(errors, SOIL_WATER, hyd.balance_error)
        hydrology_diagnostics = dict(
            qflx_rain_grnd_soil=ground_input,
            qflx_evap_grnd_soil=ground_evaporation,
            qflx_tran_soil=transpiration,
            qflx_infl_soil=hyd.qflx_infl,
            qflx_surf_soil=hyd.qflx_surf,
            qflx_drain_soil=hyd.qflx_drain,
            fsat_soil=hyd.fsat,
            frac_h2osfc_soil=hyd.frac_h2osfc,
            h2osfc_soil=hyd.h2osfc,
            zwt_soil=hyd.zwt,
            num_substeps_soil=hyd.num_substeps.astype(t_soil.dtype),
            errh2o_soil=hyd.balance_error,
        )
    else:
        soil_state = state.soil._replace(temperature=t_soil_new, liquid_water=liq[:n_soil])
        hydrology_diagnostics = {
            name: unset
            for name in (
                "qflx_rain_grnd_soil", "qflx_evap_grnd_soil", "qflx_tran_soil", "qflx_infl_soil",
                "qflx_surf_soil", "qflx_drain_soil", "fsat_soil", "frac_h2osfc_soil",
                "h2osfc_soil", "zwt_soil", "num_substeps_soil", "errh2o_soil",
            )
        }  # fmt: skip

    new_state = State(
        canopy=CanopyState(
            air_temperature=end.tair_profile,
            vapor_pressure=end.eair_profile,
            co2=end.cair_profile,
            leaf_temperature=keep(use, merged.tleaf, end.tleaf_leaf),
            leaf_water_potential=keep(use, merged.lwp, end.lwp_leaf),
            intercepted_water=end.h2ocan_profile,
            ground_temperature=end.tg_soil,
            acclimation_temperature=c.acclimation_temperature,
            acclimation_steps=c.acclimation_steps,
        ),
        soil=soil_state,
    )
    diagnostics = Diagnostics(
        tref_forcing=end.tref_forcing,
        pref_forcing=end.pref_forcing,
        rhomol_forcing=end.rhomol_forcing,
        swskyb_forcing=means.swskyb,
        swskyd_forcing=means.swskyd,
        solar_zen_forcing=solar_zen,
        lai_canopy=area.lai,
        sai_canopy=area.sai,
        sai_floor_canopy=area.sai_floor,
        laisun_canopy=totals.laisun,
        laisha_canopy=totals.laisha,
        dpai_profile=dpai,
        fracsun_profile=end.fracsun_profile,
        albcan_canopy=totals.albcan,
        swveg_canopy=totals.swveg,
        swvegsun_canopy=totals.swvegsun,
        swvegsha_canopy=totals.swvegsha,
        lwup_canopy=means.lwup,
        rnet_canopy=totals.rnet,
        shflx_canopy=totals.shflx,
        lhflx_canopy=totals.lhflx,
        stflx_air_canopy=totals.stflx_air,
        stflx_veg_canopy=totals.stflx_veg,
        ustar_canopy=means.ustar,
        beta_canopy=means.beta,
        shveg_canopy=totals.shveg,
        shvegsun_canopy=totals.shvegsun,
        shvegsha_canopy=totals.shvegsha,
        lhveg_canopy=totals.lhveg,
        lhvegsun_canopy=totals.lhvegsun,
        lhvegsha_canopy=totals.lhvegsha,
        trveg_canopy=totals.trveg,
        evveg_canopy=totals.evveg,
        gppveg_canopy=totals.gppveg,
        gppvegsun_canopy=totals.gppvegsun,
        gppvegsha_canopy=totals.gppvegsha,
        vcmax25veg_canopy=totals.vcmax25veg,
        vcmax25sun_canopy=totals.vcmax25sun,
        vcmax25sha_canopy=totals.vcmax25sha,
        gsveg_canopy=totals.gsveg,
        gsvegsun_canopy=totals.gsvegsun,
        gsvegsha_canopy=totals.gsvegsha,
        windveg_canopy=totals.windveg,
        windvegsun_canopy=totals.windvegsun,
        windvegsha_canopy=totals.windvegsha,
        tlveg_canopy=totals.tlveg,
        tlvegsun_canopy=totals.tlvegsun,
        tlvegsha_canopy=totals.tlvegsha,
        taveg_canopy=totals.taveg,
        tavegsun_canopy=totals.tavegsun,
        tavegsha_canopy=totals.tavegsha,
        fracminlwp_canopy=totals.fracminlwp,
        rnsoi_soil=means.rnsoi,
        shsoi_soil=means.shsoi,
        lhsoi_soil=means.lhsoi,
        gsoi_soil=gsoi,
        btran_soil=end.btran_soil,
        psis_soil=roots.psis,
        h2osoi_liq_soil=liq,
        h2osoi_vol_soil=water.h2osoi_vol,
        **hydrology_diagnostics,
        **frozen_diagnostics,
        wind_profile=end.wind_profile,
        shair_profile=keep(use, means.shair, end.shair_profile),
        etair_profile=keep(use, means.etair, end.etair_profile),
        mflx_profile=keep(use, means.mflx, end.mflx_profile),
        gac_profile=keep(use, means.gac, end.gac_profile),
        lsc_profile=lsc,
        lwp_mean_profile=keep(use, totals.lwp_mean, fill_layer),
        td_profile=end.td_profile,
        lwdwn_profile=keep(up_to_top, means.lwdwn, end.lwdwn_profile),
        lwupw_profile=keep(up_to_top, means.lwupw, end.lwupw_profile),
        swbeam_profile=keep(up_to_top, means.swbeam, end.swbeam_profile),
        swdwn_profile=keep(up_to_top, means.swdwn, end.swdwn_profile),
        swupw_profile=keep(up_to_top, means.swupw, end.swupw_profile),
        apar_leaf=end.apar_leaf,
        rnleaf_leaf=means.rnleaf,
        shleaf_leaf=means.shleaf,
        lhleaf_leaf=means.lhleaf,
        anet_leaf=means.anet,
        gs_leaf=means.gs,
        vcmax25_leaf=end.vcmax25_leaf,
        tleaf_hist_leaf=keep(use, merged.tleaf_hist, _fill(end.tleaf_leaf)),
        lwp_hist_leaf=keep(use, merged.lwp_hist, _fill(end.tleaf_leaf)),
        balance_error=errors,
    )
    return StepState(state=new_state, diagnostics=diagnostics)
