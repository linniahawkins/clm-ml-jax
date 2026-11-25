"""Soil hydrology of a model step: the surface, the Richards equation, the water table and
drainage, in CLM's order.

One soil column without snow. The canopy and the ground's energy balance give the water
reaching the ground, the evaporation, the dew and the root uptake of each level; this
module moves the water and returns the new state, the runoff and the water balance error.

Not in CLM-ml v2: translated from CLM itself (CTSM ``ctsm5.4.004``).

Fortran reference: CTSM ``HydrologyNoDrainageMod.F90::HydrologyNoDrainage``,
``HydrologyDrainageMod.F90::HydrologyDrainage``, ``clm_driver.F90`` (the call of
``UpdateFracH2oSfc``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from clm_ml.processes.drainage import subsurface_drainage, water_table_depth
from clm_ml.processes.soil_water import soil_water_fractions, soil_water_movement
from clm_ml.processes.surface_runoff import pond_fraction, renew_condensation, surface_fluxes


class HydrologyParameters(NamedTuple):
    """CLM's soil hydrology parameters (the values of ``ctsm60_params.c251211.nc`` and of
    the CLM6 namelist are given for orientation)."""

    e_ice: Array  # ice impedance of the hydraulic conductivity (6)
    fff: Array  # decay of the saturated fraction with water table depth (0.5 m-1)
    pc: Array  # ponded fraction above which ponds connect (0.4)
    mu: Array  # exponent of the connected ponded fraction (0.13889)
    baseflow_scalar: Array  # baseflow rate (0.001 mm/s per m of saturated zone)
    n_baseflow: Array  # exponent of the saturated zone's thickness (1)
    aq_sp_yield_min: Array  # smallest specific yield (0.02)


class SoilHydrology(NamedTuple):
    h2osoi_liq: Array  # (n,) liquid water (kg/m2)
    h2osoi_ice: Array  # (n,) ice (kg/m2)
    h2osfc: Array  # ponded surface water (mm)
    zwt: Array  # water table depth (m)
    frac_h2osfc: Array  # fraction of the ground under ponded water during the step
    fsat: Array  # saturated fraction of the surface
    qflx_infl: Array  # infiltration (mm/s)
    qflx_surf: Array  # surface runoff (mm/s)
    qflx_drain: Array  # drainage (mm/s)
    qflx_ice_runoff_xs: Array  # ice above the top level's pore space, removed (mm/s)
    num_substeps: Array  # trial sub-steps of the Richards solver
    hk: Array  # (n,) conductivity at the interface below each level (mm/s)
    smp: Array  # (n,) matric potential (mm)
    balance_error: Array  # change of storage minus net input over the step (mm)


def soil_hydrology(
    h2osoi_liq: Array,
    h2osoi_ice: Array,
    h2osfc,
    zwt,
    qflx_rain_plus_snomelt,
    qflx_liqevap_from_top_layer,
    qflx_ev_h2osfc,
    qflx_liqdew_to_top_layer,
    qflx_rootsoi: Array,
    watsat: Array,
    hksat: Array,
    sucsat: Array,
    bsw: Array,
    dz: Array,
    z: Array,
    zi: Array,
    nbedrock,
    wtfact,
    micro_sigma,
    h2osfc_thresh,
    topo_slope,
    dtime,
    params: HydrologyParameters,
    *,
    qflx_soliddew_to_top_layer=0.0,
    qflx_solidevap_from_top_layer=0.0,
    fixed_trials: int | None = None,
) -> SoilHydrology:
    """One model step of the soil hydrology of a column.

    In CLM's order: the fraction of the ground under ponded water; the ice fraction of the
    pores; runoff from the saturated fraction, the pond's budget and the infiltration; the
    Richards equation; the water table; dew; drainage and the bounds on the soil water.

    Args:
        h2osoi_liq, h2osoi_ice: (n,) liquid water and ice of the soil levels (kg m⁻²).
        h2osfc: ponded surface water (mm); zwt: water table depth (m).
        qflx_rain_plus_snomelt: liquid water reaching the ground (mm s⁻¹).
        qflx_liqevap_from_top_layer: evaporation from the soil (mm s⁻¹), per unit area of
            ground outside the pond.
        qflx_ev_h2osfc: evaporation from the pond (mm s⁻¹), per unit area of pond.
        qflx_liqdew_to_top_layer: dew (mm s⁻¹), per unit area of ground outside the pond.
        qflx_rootsoi: (n,) root uptake of each level (mm s⁻¹).
        watsat, hksat, sucsat, bsw: (n,) porosity, conductivity at saturation (mm s⁻¹),
            suction at saturation (mm), Clapp-Hornberger *b*.
        dz, z: (n,) thickness and node depth (m); zi: (n + 1,) interface depths (m).
        nbedrock: number of levels above the bedrock.
        wtfact: largest saturated fraction; micro_sigma: microtopography (m);
            h2osfc_thresh: pond threshold (mm); topo_slope: slope (degrees).
        dtime: model step (s).
        params: the hydrology parameters.
        qflx_soliddew_to_top_layer, qflx_solidevap_from_top_layer: frost and sublimation
            (mm s⁻¹), per unit area of ground outside the pond.
        fixed_trials: see :func:`clm_ml.processes.soil_water.soil_water_movement`.

    The water balance error is the change of soil water, ice and pond over the step minus
    (water reaching the ground + dew and frost − evaporation and sublimation − root uptake
    above the bedrock − surface runoff − drainage − ice runoff) × step.
    """
    active = jnp.arange(h2osoi_liq.shape[0]) < nbedrock
    storage_before = jnp.sum(h2osoi_liq) + jnp.sum(h2osoi_ice) + h2osfc

    pond = pond_fraction(h2osfc, h2osoi_liq[0], micro_sigma, dtime)
    liq = jnp.concatenate([pond.h2osoi_liq_top[None], h2osoi_liq[1:]])
    frac_h2osfc = pond.frac_h2osfc

    icefrac, eff_porosity = soil_water_fractions(h2osoi_ice, watsat, dz)
    surface = surface_fluxes(
        qflx_rain_plus_snomelt,
        qflx_liqevap_from_top_layer,
        qflx_ev_h2osfc,
        pond.h2osfc,
        frac_h2osfc,
        zwt,
        icefrac,
        hksat,
        wtfact,
        h2osfc_thresh,
        topo_slope,
        dtime,
        params.fff,
        params.e_ice,
        params.pc,
        params.mu,
    )
    movement = soil_water_movement(
        liq,
        surface.qflx_infl,
        qflx_rootsoi,
        icefrac,
        watsat,
        hksat,
        sucsat,
        bsw,
        dz,
        z,
        nbedrock,
        dtime,
        params.e_ice,
        fixed_trials=fixed_trials,
    )
    liq = movement.h2osoi_liq
    zwt = water_table_depth(liq, h2osoi_ice, watsat, dz, z, zi, nbedrock)

    liq_top, ice_top = renew_condensation(
        liq[0],
        h2osoi_ice[0],
        frac_h2osfc,
        qflx_liqdew_to_top_layer,
        qflx_soliddew_to_top_layer,
        qflx_solidevap_from_top_layer,
        dtime,
    )
    drained = subsurface_drainage(
        jnp.concatenate([liq_top[None], liq[1:]]),
        jnp.concatenate([ice_top[None], h2osoi_ice[1:]]),
        zwt,
        surface.h2osfc,
        eff_porosity,
        watsat,
        sucsat,
        bsw,
        dz,
        zi,
        nbedrock,
        topo_slope,
        dtime,
        params.baseflow_scalar,
        params.n_baseflow,
        params.aq_sp_yield_min,
        params.e_ice,
    )

    ground = 1.0 - frac_h2osfc
    net_input = (
        qflx_rain_plus_snomelt
        + ground * (qflx_liqdew_to_top_layer + qflx_soliddew_to_top_layer)
        - ground * (qflx_liqevap_from_top_layer + qflx_solidevap_from_top_layer)
        - frac_h2osfc * qflx_ev_h2osfc
        - jnp.sum(jnp.where(active, qflx_rootsoi, 0.0))
        - surface.qflx_surf
        - drained.qflx_drain
        - drained.qflx_ice_runoff_xs
    )
    storage = jnp.sum(drained.h2osoi_liq) + jnp.sum(drained.h2osoi_ice) + drained.h2osfc
    return SoilHydrology(
        h2osoi_liq=drained.h2osoi_liq,
        h2osoi_ice=drained.h2osoi_ice,
        h2osfc=drained.h2osfc,
        zwt=drained.zwt,
        frac_h2osfc=frac_h2osfc,
        fsat=surface.fsat,
        qflx_infl=surface.qflx_infl,
        qflx_surf=surface.qflx_surf,
        qflx_drain=drained.qflx_drain,
        qflx_ice_runoff_xs=drained.qflx_ice_runoff_xs,
        num_substeps=movement.num_substeps,
        hk=movement.hk,
        smp=movement.smp,
        balance_error=storage - storage_before - net_input * dtime,
    )
