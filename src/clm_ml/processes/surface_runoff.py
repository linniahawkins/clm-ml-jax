"""Soil hydrology at the surface: what becomes of the water reaching the ground.

Runoff from the saturated fraction of the surface, the limit on infiltration, the ponded
surface water (its covered fraction, its runoff and its drainage into the soil) and the
infiltration that the Richards equation takes as its upper boundary flux; and the dew and
frost added to the top soil level.

Each function computes one soil column without snow. Soil levels are 0-based from the top
down. Water stores are in mm (kg m⁻²) and fluxes in mm s⁻¹. Fluxes are averages over the
whole column unless a docstring says otherwise.

Not in CLM-ml v2: these are translated from CLM itself (CTSM ``ctsm5.4.004``), with CLM's
default choices: the TOPMODEL saturated fraction, infiltration limited by the saturated
conductivity, surface water active.

Fortran reference: CTSM ``SaturatedExcessRunoffMod.F90``, ``InfiltrationExcessRunoffMod.F90``,
``SurfaceWaterMod.F90``, ``SoilHydrologyMod.F90``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erf

from clm_ml.processes.soil_water import ice_impedance

#: Relative size below which a state that fluxes should have brought to zero is set to zero.
REL_EPSILON = 1.0e-13
#: Smallest ponded water kept as a pond (mm); less is moved into the top soil level.
MIN_H2OSFC = 1.0e-8
#: Smallest pond runoff kept (mm/s). CLM writes the number in single precision.
MIN_POND_RUNOFF = float(np.float32(1.0e-8))


def truncate_small_values(data: Array, baseline: Array, rel_epsilon=REL_EPSILON) -> Array:
    """``data`` with values smaller than ``rel_epsilon`` times ``baseline`` (in magnitude)
    set to zero: a state after fluxes that should have emptied it, ``baseline`` being the
    state before.

    Fortran reference: NumericsMod.F90::truncate_small_values.
    """
    return jnp.where(jnp.abs(data) < rel_epsilon * jnp.abs(baseline), 0.0, data)


# ---------------------------------------------------------------------------
# Constants of a column's surface
# ---------------------------------------------------------------------------


def microtopography(topo_slope, slopebeta, slopemax) -> Array:
    """Standard deviation of the surface's microtopography (m), a smooth function of the
    topographic slope ``topo_slope`` (degrees): rougher on flat ground.

    Fortran reference: initVerticalMod.F90::initVertical.
    """
    slope0 = slopemax ** (1.0 / slopebeta)
    return (topo_slope + slope0) ** slopebeta


def pond_threshold(micro_sigma, pc) -> Array:
    """Ponded water (mm) above which the ponds connect and run off ("fill and spill"): the
    water held when the fraction ``pc`` of the ground is under water, from four Newton steps
    for the water level at that fraction. Zero where the microtopography ``micro_sigma``
    (m) is 1e-6 m or less.

    Fortran reference: SoilHydrologyInitTimeConstMod.F90::SoilHydrologyInitTimeConst.
    """
    rough = micro_sigma > 1.0e-6
    sigma = jnp.where(rough, micro_sigma, 1.0)
    d = jnp.zeros_like(sigma)
    for _ in range(4):
        fd = 0.5 * (1.0 + erf(d / (sigma * math.sqrt(2.0)))) - pc
        dfdd = jnp.exp(-(d**2) / (2.0 * sigma**2)) / (sigma * math.sqrt(2.0 * math.pi))
        d = d - fd / dfdd
    thresh = 0.5 * d * (1.0 + erf(d / (sigma * math.sqrt(2.0)))) + sigma / math.sqrt(
        2.0 * math.pi
    ) * jnp.exp(-(d**2) / (2.0 * sigma**2))
    return jnp.where(rough, 1.0e3 * thresh, 0.0)


# ---------------------------------------------------------------------------
# The pond's covered fraction
# ---------------------------------------------------------------------------


class PondFraction(NamedTuple):
    frac_h2osfc: Array  # fraction of the ground under ponded water (0 to 1)
    h2osfc: Array  # ponded water (mm): zero if it was too little to keep
    h2osoi_liq_top: Array  # liquid water of the top soil level (kg/m2), with that water
    qflx_too_small_h2osfc_to_soil: Array  # the pond moved into the soil (mm/s)


def pond_fraction(h2osfc, h2osoi_liq_top, micro_sigma, dtime) -> PondFraction:
    """Fraction of the ground under ponded water, from the amount of water ``h2osfc`` (mm)
    and the microtopography ``micro_sigma`` (m): the water fills the lowest ground first,
    the surface heights being normally distributed. The water level is found with ten
    Newton steps from the mean height.

    A pond of ``MIN_H2OSFC`` or less (also a negative one) covers nothing and is moved
    into the top soil level.

    Fortran reference: SurfaceWaterMod.F90::UpdateFracH2oSfc, BulkDiag_FracH2oSfc,
    UpdateState_TooSmallH2osfcToSoil (no snow).
    """
    ponded = h2osfc > MIN_H2OSFC
    water = jnp.where(ponded, h2osfc, 1.0)
    sigma = 1.0e3 * micro_sigma  # mm

    def newton(d, _):
        covered = 0.5 * (1.0 + erf(d / (sigma * math.sqrt(2.0))))
        fd = (
            0.5 * d * (1.0 + erf(d / (sigma * math.sqrt(2.0))))
            + sigma / math.sqrt(2.0 * math.pi) * jnp.exp(-(d**2) / (2.0 * sigma**2))
            - water
        )
        return d - fd / covered, None

    d, _ = jax.lax.scan(newton, jnp.zeros_like(sigma * water), None, length=10)
    frac = jnp.where(ponded, 0.5 * (1.0 + erf(d / (sigma * math.sqrt(2.0)))), 0.0)
    to_soil = jnp.where(ponded, 0.0, h2osfc / dtime)
    emptied = truncate_small_values(h2osfc - to_soil * dtime, h2osfc)
    return PondFraction(frac, emptied, h2osoi_liq_top + to_soil * dtime, to_soil)


# ---------------------------------------------------------------------------
# Runoff, infiltration and the pond's budget
# ---------------------------------------------------------------------------


def saturated_fraction(zwt, wtfact, fff) -> Array:
    """Fraction of the surface that is saturated (the water table at the surface), from the
    water table depth ``zwt`` (m): ``wtfact`` (the largest fraction, from the topography)
    decaying with depth at the rate ``fff`` (m⁻¹).

    The perched water table that CLM uses instead above frozen soil is not included.

    Fortran reference: SaturatedExcessRunoffMod.F90::ComputeFsatTopmodel.
    """
    return wtfact * jnp.exp(-0.5 * fff * zwt)


def infiltration_capacity(icefrac: Array, hksat: Array, e_ice) -> Array:
    """Largest infiltration rate of unsaturated ground (mm s⁻¹): the smallest saturated
    conductivity of the top three soil levels, reduced by their ice.

    ``icefrac``, ``hksat``: (n,) ice-filled fraction of the pores and conductivity at
    saturation (mm s⁻¹) of the soil levels.

    Fortran reference: InfiltrationExcessRunoffMod.F90::ComputeQinmaxHksat.
    """
    return jnp.min(ice_impedance(icefrac[:3], e_ice) * hksat[:3])


def pond_runoff(h2osfc, h2osfc_thresh, frac_h2osfc, topo_slope, dtime, pc, mu) -> Array:
    """Runoff from the ponded water (mm s⁻¹): the water above the threshold
    ``h2osfc_thresh`` (mm) drains as a linear reservoir whose rate grows with the slope
    ``topo_slope`` (degrees) and with the connected part of the ponded fraction,
    ``(frac_h2osfc - pc)**mu``; at most the water above the threshold in a step, and nothing
    below ``MIN_POND_RUNOFF``.

    Fortran reference: SurfaceWaterMod.F90::QflxH2osfcSurf.
    """
    connected = frac_h2osfc > pc
    frac_infclust = jnp.where(connected, jnp.where(connected, frac_h2osfc - pc, 1.0) ** mu, 0.0)
    k_wet = 1.0e-4 * jnp.sin((math.pi / 180.0) * topo_slope)
    above = h2osfc - h2osfc_thresh
    runoff = jnp.minimum(k_wet * frac_infclust * above, above / dtime)
    runoff = jnp.where(h2osfc > h2osfc_thresh, runoff, 0.0)
    return jnp.where(runoff < MIN_POND_RUNOFF, 0.0, runoff)


def pond_drainage(h2osfc, frac_h2osfc, qinmax, dtime) -> Array:
    """Drainage of the ponded water into the soil (mm s⁻¹): the infiltration capacity
    ``qinmax`` over the ponded fraction, at most the pond in a step. A negative pond is
    refilled from the soil (a negative drainage).

    Fortran reference: SurfaceWaterMod.F90::QflxH2osfcDrain.
    """
    return jnp.where(
        h2osfc < 0.0, h2osfc / dtime, jnp.minimum(frac_h2osfc * qinmax, h2osfc / dtime)
    )


class SurfaceFluxes(NamedTuple):
    fsat: Array  # saturated fraction of the surface
    qflx_sat_excess_surf: Array  # runoff from the saturated fraction (mm/s)
    qflx_in_soil: Array  # input to the soil outside the pond, less evaporation (mm/s)
    qflx_top_soil_to_h2osfc: Array  # input to the pond, less its evaporation (mm/s)
    qinmax: Array  # largest infiltration rate (mm/s)
    qflx_infl_excess: Array  # input above that rate (mm/s): it goes to the pond
    qflx_in_soil_limited: Array  # input to the soil within that rate (mm/s)
    qflx_in_h2osfc: Array  # total input to the pond (mm/s)
    qflx_h2osfc_surf: Array  # runoff from the pond (mm/s)
    qflx_h2osfc_drain: Array  # drainage of the pond into the soil (mm/s)
    h2osfc: Array  # ponded water at the end of the step (mm)
    qflx_infl: Array  # infiltration: the Richards equation's flux at the surface (mm/s)
    qflx_surf: Array  # total surface runoff (mm/s)


def surface_fluxes(
    qflx_rain_plus_snomelt,
    qflx_liqevap_from_top_layer,
    qflx_ev_h2osfc,
    h2osfc,
    frac_h2osfc,
    zwt,
    icefrac: Array,
    hksat: Array,
    wtfact,
    h2osfc_thresh,
    topo_slope,
    dtime,
    fff,
    e_ice,
    pc,
    mu,
) -> SurfaceFluxes:
    """Partition the water reaching the ground over a model step into surface runoff,
    ponded water and infiltration.

    The saturated fraction of the surface sheds its share of the water at once. The rest
    falls on the pond and on the soil outside it in proportion to their areas, and each
    loses its evaporation. Soil input above the infiltration capacity is added to the pond.
    The pond then runs off (above its threshold) and drains into the soil, in that order.
    The infiltration is the limited soil input plus the pond's drainage; it is negative
    when evaporation exceeds the input.

    Args:
        qflx_rain_plus_snomelt: liquid water reaching the ground (mm s⁻¹).
        qflx_liqevap_from_top_layer: evaporation from the soil (mm s⁻¹, positive upward),
            as a rate per unit area of the ground outside the pond.
        qflx_ev_h2osfc: evaporation from the pond (mm s⁻¹; negative: dew), per unit area of
            pond.
        h2osfc: ponded water (mm); frac_h2osfc: fraction of the ground it covers
            (:func:`pond_fraction`).
        zwt: water table depth (m).
        icefrac, hksat: (n,) ice-filled fraction of the pores and conductivity at
            saturation (mm s⁻¹) of the soil levels.
        wtfact: largest saturated fraction; h2osfc_thresh: pond threshold (mm);
            topo_slope: slope (degrees).
        dtime: model step (s).
        fff, e_ice, pc, mu: CLM parameters (decay of the saturated fraction, ice impedance,
            pond connectivity threshold and exponent).

    Fortran reference: SaturatedExcessRunoffMod.F90::SaturatedExcessRunoff,
    SoilHydrologyMod.F90::SetQflxInputs, RouteInfiltrationExcess, Infiltration,
    TotalSurfaceRunoff, InfiltrationExcessRunoffMod.F90::InfiltrationExcessRunoff,
    SurfaceWaterMod.F90::UpdateH2osfc.
    """
    fsat = saturated_fraction(zwt, wtfact, fff)
    qflx_sat_excess_surf = fsat * qflx_rain_plus_snomelt

    available = qflx_rain_plus_snomelt - qflx_sat_excess_surf
    qflx_in_soil = (1.0 - frac_h2osfc) * available
    qflx_top_soil_to_h2osfc = frac_h2osfc * available
    qflx_in_soil = qflx_in_soil - (1.0 - frac_h2osfc) * qflx_liqevap_from_top_layer
    qflx_top_soil_to_h2osfc = qflx_top_soil_to_h2osfc - frac_h2osfc * qflx_ev_h2osfc

    qinmax = (1.0 - fsat) * infiltration_capacity(icefrac, hksat, e_ice)
    qflx_infl_excess = jnp.maximum(0.0, qflx_in_soil - (1.0 - frac_h2osfc) * qinmax)
    qflx_in_soil_limited = qflx_in_soil - qflx_infl_excess
    qflx_in_h2osfc = qflx_top_soil_to_h2osfc + qflx_infl_excess

    qflx_h2osfc_surf = pond_runoff(h2osfc, h2osfc_thresh, frac_h2osfc, topo_slope, dtime, pc, mu)
    h2osfc_partial = truncate_small_values(
        h2osfc + (qflx_in_h2osfc - qflx_h2osfc_surf) * dtime, h2osfc
    )
    qflx_h2osfc_drain = pond_drainage(h2osfc_partial, frac_h2osfc, qinmax, dtime)
    h2osfc_new = truncate_small_values(h2osfc_partial - qflx_h2osfc_drain * dtime, h2osfc_partial)

    return SurfaceFluxes(
        fsat=fsat,
        qflx_sat_excess_surf=qflx_sat_excess_surf,
        qflx_in_soil=qflx_in_soil,
        qflx_top_soil_to_h2osfc=qflx_top_soil_to_h2osfc,
        qinmax=qinmax,
        qflx_infl_excess=qflx_infl_excess,
        qflx_in_soil_limited=qflx_in_soil_limited,
        qflx_in_h2osfc=qflx_in_h2osfc,
        qflx_h2osfc_surf=qflx_h2osfc_surf,
        qflx_h2osfc_drain=qflx_h2osfc_drain,
        h2osfc=h2osfc_new,
        qflx_infl=qflx_in_soil_limited + qflx_h2osfc_drain,
        qflx_surf=qflx_sat_excess_surf + qflx_h2osfc_surf,
    )


# ---------------------------------------------------------------------------
# Dew and frost
# ---------------------------------------------------------------------------


def renew_condensation(
    h2osoi_liq_top,
    h2osoi_ice_top,
    frac_h2osfc,
    qflx_liqdew_to_top_layer,
    qflx_soliddew_to_top_layer,
    qflx_solidevap_from_top_layer,
    dtime,
) -> tuple[Array, Array]:
    """Liquid water and ice of the top soil level (kg m⁻²) after dew and frost are added
    and sublimation is removed, over the ground outside the pond. (Evaporation of liquid
    water is taken from the infiltration instead.) The fluxes are in mm s⁻¹.

    An ice amount that sublimation brings to within 1e-12 of zero, relative to the ice
    before, is set to zero. CLM stops if the ice turns negative; here it is returned.

    Fortran reference: SoilHydrologyMod.F90::RenewCondensation.
    """
    ground = 1.0 - frac_h2osfc
    liq = h2osoi_liq_top + ground * qflx_liqdew_to_top_layer * dtime
    ice_before = h2osoi_ice_top + ground * qflx_soliddew_to_top_layer * dtime
    ice = ice_before - ground * qflx_solidevap_from_top_layer * dtime
    return liq, truncate_small_values(ice, ice_before, 1.0e-12)
