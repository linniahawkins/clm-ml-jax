"""Soil: water from the prescribed history, albedo, evaporative resistance, thermal and
hydraulic properties, and the soil temperature solution.

Each function computes one soil column. Soil levels are 0-based (level ``k`` is Fortran's
layer ``j = k + 1``), from the top down, with ``n_levels`` (``nlevgrnd``) levels; bands are
0-based (0 visible, 1 near infrared). No snow layers and no surface water: the multilayer
canopy runs these CLM routines with ``snl = 0``, ``h2osfc = 0`` (as CLM-ml v2's
``clm_driver``).

Fortran reference: CLM-ml v2 ``clmDataMod.F90::clmData``, ``SurfaceAlbedoMod.F90::SoilAlbedo``,
``SurfaceResistanceMod.F90::calc_soil_resistance_sl14``,
``MLSoilTemperatureMod.F90::SoilThermProp``/``SoilTemperature``,
``SoilWaterMovementMod.F90::soilwater_moisture_form``/``compute_hydraulic_properties``.
The freezing and thawing of soil water is CLM's own (CTSM ``ctsm5.4.004``,
``SoilTemperatureMod.F90::Phasechange_beta``): CLM-ml v2 has none.

Each function's reads and writes are its arguments and results; ``Reads``/``Writes`` in the
docstrings name the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from clm_ml.processes.constants import (
    cpice,
    cpliq,
    csol_bedrock,
    denh2o,
    denice,
    tfrz,
    thk_bedrock,
    tkice,
    tkwat,
)
from clm_ml.processes.math import tridiag

#: 1e-3 as CLM writes it in the surface water's conductivity: without a kind, so a
#: single-precision number.
MM_TO_M_SINGLE = float(np.float32(1.0e-3))


def _below_bedrock(n: int, nbedrock) -> Array:
    """True for soil levels below the bedrock (Fortran layer ``j > nbedrock``)."""
    return jnp.arange(1, n + 1) > nbedrock


# ---------------------------------------------------------------------------
# Soil water from the CLM history record
# ---------------------------------------------------------------------------


class SoilWater(NamedTuple):
    h2osoi_vol: Array  # (n_levels,) volumetric soil water (m3/m3)
    h2osoi_liq: Array  # (n_levels,) liquid water (kg H2O/m2)
    h2osoi_ice: Array  # (n_levels,) ice (kg H2O/m2)


def soil_water_from_history(
    h2osoi: Array,
    h2osoi_factor,
    dz: Array,
    watsat: Array,
    nbedrock,
    *,
    n_soil: int,
    n_adjust: int,
) -> SoilWater:
    """Soil water of each level from the prescribed CLM history values.

    Fortran reference: clmDataMod.F90::clmData (soil water part).

    CLM histories give the ``n_soil`` (``nlevsoi``) soil levels, the bedrock levels below
    are 0, and the values are capped at porosity down to
    the bedrock. The first ``n_adjust`` levels are multiplied by ``h2osoi_factor``. All water
    is liquid.

    Reads: the history record (``h2osoi``, ``h2osoi_factor``),
    ``site.dz_col``, ``site.nbedrock_col``, ``soilstate.watsat_col``.
    Writes: ``waterstatebulk.h2osoi_vol_col``, ``h2osoi_liq_col``, ``h2osoi_ice_col``.

    Args:
        h2osoi: ``(n_soil,)`` history values.
        h2osoi_factor: Adjustment factor (used if ``n_adjust > 0``).
        dz: ``(n_levels,)`` level thickness (m).
        watsat: ``(n_levels,)`` porosity.
        nbedrock: Index (1-based) of the deepest level above bedrock (data).
        n_soil, n_adjust: Static settings (``nlevsoi``, the adjusted levels).
    """
    n = dz.shape[0]
    vol = jnp.concatenate([jnp.asarray(h2osoi, dtype=dz.dtype), jnp.zeros(n - n_soil, dz.dtype)])
    if n_adjust > 0:
        vol = vol.at[:n_adjust].set(vol[:n_adjust] * h2osoi_factor)
    top = vol[:n_soil]
    capped = jnp.where(
        jnp.arange(1, n_soil + 1) <= nbedrock, jnp.minimum(top, watsat[:n_soil]), top
    )
    vol = jnp.concatenate([capped, vol[n_soil:]])
    return SoilWater(vol, vol * dz * denh2o, jnp.zeros_like(vol))


# ---------------------------------------------------------------------------
# Soil albedo
# ---------------------------------------------------------------------------


def soil_albedo(h2osoi_vol_top, albsat: Array, albdry: Array) -> tuple[Array, Array]:
    """Direct-beam and diffuse soil albedo per band (equal).

    Fortran reference: SurfaceAlbedoMod.F90::SoilAlbedo.

    ``min(albsat + max(0.11 - 0.40 * h2osoi_vol(1), 0), albdry)``.

    Reads: ``waterstatebulk.h2osoi_vol_col`` (top level), the colour class's ``albsat`` and
    ``albdry`` (``SurfaceAlbedoMod`` tables, ``site.isoicol_col``).
    Writes: ``surfalb.albgrd_col``, ``surfalb.albgri_col``.

    Args:
        h2osoi_vol_top: Volumetric water of the top level (m3/m3).
        albsat, albdry: ``(band,)`` saturated and dry soil albedo of the column's colour.

    Returns:
        ``(albgrd, albgri)``, each ``(band,)``.
    """
    inc = jnp.maximum(0.11 - 0.40 * h2osoi_vol_top, 0.0)
    alb = jnp.minimum(albsat + inc, albdry)
    return alb, alb


# ---------------------------------------------------------------------------
# Soil evaporative resistance
# ---------------------------------------------------------------------------


def soil_evaporative_resistance(
    h2osoi_liq_top, h2osoi_ice_top, t_soisno_top, dz_top, watsat_top, sucsat_top, bsw_top
) -> tuple[Array, Array]:
    """Dry surface layer thickness and soil evaporative resistance (Swenson and Lawrence 2014).

    Fortran reference: SurfaceResistanceMod.F90::calc_soil_resistance_sl14.

    Reads (top level): ``waterstatebulk.h2osoi_liq_col``, ``h2osoi_ice_col``,
    ``temperature.t_soisno_col``, ``site.dz_col``, ``soilstate.watsat_col``, ``sucsat_col``,
    ``bsw_col``.
    Writes: ``soilstate.dsl_col`` (mm), ``soilstate.soilresis_col`` (s/m).

    Returns:
        ``(dsl, soilresis)``.
    """
    # Liquid volumetric water content and effective porosity of the top level
    vwc_liq = jnp.maximum(h2osoi_liq_top, 1.0e-6) / (dz_top * denh2o)
    eff_por_top = jnp.maximum(
        0.01, watsat_top - jnp.minimum(watsat_top, h2osoi_ice_top / (dz_top * denice))
    )
    # Air-dry soil moisture; diffusivity of water vapor (m2/s)
    aird = watsat_top * (sucsat_top / 1.0e7) ** (1.0 / bsw_top)
    d0 = 2.12e-5 * (t_soisno_top / 273.15) ** 1.75
    # Air-filled pore space and tortuosity
    eps = watsat_top - aird
    tort = eps * eps * (eps / watsat_top) ** (3.0 / jnp.maximum(3.0, bsw_top))
    # Dry surface layer thickness (mm), in [0, 200]
    dsl = (
        15.0
        * jnp.maximum(0.001, 0.8 * eff_por_top - vwc_liq)
        / jnp.maximum(0.001, 0.8 * watsat_top - aird)
    )
    dsl = jnp.maximum(dsl, 0.0)
    dsl = jnp.minimum(dsl, 200.0)
    # Soil evaporative resistance (s/m), at most 1e6
    soilresis = dsl / (d0 * tort * 1.0e3) + 20.0
    soilresis = jnp.minimum(1.0e6, soilresis)
    return dsl, soilresis


# ---------------------------------------------------------------------------
# Soil thermal properties
# ---------------------------------------------------------------------------


class ThermalProperties(NamedTuple):
    thk: Array  # (n_levels,) thermal conductivity of each level (W/m/K)
    tk: Array  # (n_levels,) conductivity at the bottom interface of each level (W/m/K); 0 at
    #            the bottom (zero flux)
    cv: Array  # (n_levels,) heat capacity (J/m2/K)
    tk_h2osfc: Array  # () thermal conductivity of surface water (W/m/K)


def soil_thermal_properties(
    t_soisno: Array,
    h2osoi_liq: Array,
    h2osoi_ice: Array,
    h2osfc,
    h2osno,
    dz: Array,
    z: Array,
    zi: Array,
    watsat: Array,
    tkmg: Array,
    tkdry: Array,
    csol: Array,
    nbedrock,
) -> ThermalProperties:
    """Thermal conductivity and heat capacity of the soil levels (no snow layers).

    Fortran reference: MLSoilTemperatureMod.F90::SoilThermProp (from CLM5).

    Conductivity from Farouki (1981) (Kersten number from saturation, frozen or thawed),
    bedrock below ``nbedrock``; interface conductivities between level depths; heat capacity
    from de Vries (1963), with the snow water ``h2osno`` added to the top level.

    Reads: ``temperature.t_soisno_col``, ``waterstatebulk.h2osoi_liq_col``, ``h2osoi_ice_col``,
    ``h2osfc_col``, ``water.h2osno_col``, ``site.dz_col``, ``z_col``, ``zi_col``,
    ``nbedrock_col``, ``soilstate.watsat_col``, ``tkmg_col``, ``tkdry_col``, ``csol_col``.
    Writes: ``soilstate.thk_col``; ``tk``, ``cv``, ``tk_h2osfc`` (work values of
    ``clm_driver`` and ``SoilTemperature``).

    Args:
        zi: ``(n_levels,)`` depth of each level's bottom interface (m) (Fortran ``zi(1:n)``).
        nbedrock: Index (1-based) of the deepest level above bedrock (data).
        Other arrays: ``(n_levels,)``; ``h2osfc``, ``h2osno``: scalars.
    """
    n = dz.shape[0]
    bedrock = _below_bedrock(n, nbedrock)

    # Thermal conductivity of each level
    liq_vol = h2osoi_liq / denh2o
    ice_vol = h2osoi_ice / denice
    ws_safe = jnp.where(watsat > 0.0, watsat, 1.0)  # watsat may be 0 in bedrock
    satw = jnp.minimum(1.0, (liq_vol + ice_vol) / (dz * ws_safe))
    fl_denom = liq_vol / dz + ice_vol / dz
    fl = (liq_vol / dz) / jnp.where(fl_denom > 0.0, fl_denom, 1.0)
    dksat = tkmg * tkwat ** (fl * watsat) * tkice ** ((1.0 - fl) * watsat)
    # Kersten number: log10(satw) + 1 (thawed) or satw (frozen); no log10(0)
    satw_safe = jnp.where(satw > 0.0, satw, 1.0e-10)
    dke = jnp.where(t_soisno >= tfrz, jnp.maximum(0.0, jnp.log10(satw_safe) + 1.0), satw)
    thk = jnp.where(satw > 1.0e-7, dke * dksat + (1.0 - dke) * tkdry, tkdry)
    thk = jnp.where(bedrock, thk_bedrock, thk)

    # Conductivity at the interface below each level; zero flux at the bottom
    thk_j, thk_jp1, z_j, z_jp1, zi_j = thk[:-1], thk[1:], z[:-1], z[1:], zi[:-1]
    denom = thk_j * (z_jp1 - zi_j) + thk_jp1 * (zi_j - z_j)
    denom_safe = jnp.where(jnp.abs(denom) > 0.0, denom, 1.0e-10)
    tk = jnp.concatenate([thk_j * thk_jp1 * (z_jp1 - z_j) / denom_safe, jnp.zeros(1, dz.dtype)])

    # Surface water conductivity (the conversion from mm to m is a single-precision number
    # in CLM)
    zh2osfc = MM_TO_M_SINGLE * (0.5 * h2osfc)
    denom = tkwat * z[0] + thk[0] * zh2osfc
    tk_h2osfc = jnp.where(
        jnp.abs(denom) > 0.0,
        tkwat * thk[0] * (z[0] + zh2osfc) / jnp.where(jnp.abs(denom) > 0.0, denom, 1.0),
        jnp.asarray(tkwat),
    )

    # Heat capacity; snow water added to the top level (no snow layers)
    cv = csol * (1.0 - watsat) * dz + h2osoi_ice * cpice + h2osoi_liq * cpliq
    cv = jnp.where(bedrock, csol_bedrock * dz, cv)
    cv = cv.at[0].set(cv[0] + jnp.where(h2osno > 0.0, cpice * h2osno, jnp.zeros(())))
    return ThermalProperties(thk, tk, cv, tk_h2osfc)


# ---------------------------------------------------------------------------
# Soil hydraulic properties
# ---------------------------------------------------------------------------


def soil_hydraulic_properties(
    h2osoi_liq: Array,
    dz: Array,
    watsat: Array,
    hksat: Array,
    sucsat: Array,
    bsw: Array,
    nbedrock,
    hk_l: Array,
    smp_l: Array,
) -> tuple[Array, Array]:
    """Hydraulic conductivity and matric potential of the soil levels above bedrock
    (Clapp and Hornberger 1978).

    Fortran reference: SoilWaterMovementMod.F90::soilwater_moisture_form,
    compute_hydraulic_properties.

    ``s = min(max(vwc_liq / watsat, 0.01), 1)``, ``hk = hksat s^(2b+3)``,
    ``smp = max(-sucsat s^-b, -1e8)``. Levels below the bedrock keep ``hk_l``/``smp_l``
    (Fortran's loop stops at ``nbedrock``); their inputs are the top level's, so they stay
    finite (and so do their gradients).

    Reads: ``waterstatebulk.h2osoi_liq_col``, ``site.dz_col``, ``site.nbedrock_col``,
    ``soilstate.watsat_col``, ``hksat_col``, ``sucsat_col``, ``bsw_col``, ``hk_l_col``,
    ``smp_l_col`` (below bedrock).
    Writes: ``soilstate.hk_l_col`` (mm H2O/s), ``soilstate.smp_l_col`` (mm).

    Args:
        All arrays ``(n_soil,)``: the ``nlevsoi`` soil levels.
        nbedrock: Index (1-based) of the deepest level above bedrock (data).
    """
    n = dz.shape[0]
    active = jnp.logical_not(_below_bedrock(n, nbedrock))
    vwc_liq = jnp.where(active, jnp.maximum(h2osoi_liq, 1.0e-6) / (dz * denh2o), 0.0)

    def top_if_inactive(x):
        return jnp.where(active, x, x[0])

    vwc, ws, hks, sucs, b = (top_if_inactive(x) for x in (vwc_liq, watsat, hksat, sucsat, bsw))
    s = vwc / ws
    s = jnp.minimum(s, 1.0)
    s = jnp.maximum(s, 0.01)
    hk = hks * s ** (2.0 * b + 3.0)
    smp = -sucs * s ** (-b)
    smp = jnp.maximum(smp, -1.0e8)
    return jnp.where(active, hk, hk_l), jnp.where(active, smp, smp_l)


# ---------------------------------------------------------------------------
# Soil temperature
# ---------------------------------------------------------------------------


def soil_temperature(
    t_soisno: Array, tk: Array, cv: Array, z: Array, gsoi, dtime
) -> tuple[Array, Array]:
    """Soil temperature after one step: implicit heat diffusion with the ground heat flux at
    the top and zero flux at the bottom (specific to the multilayer canopy, not CLM's).

    Fortran reference: MLSoilTemperatureMod.F90::SoilTemperature.

    Reads: ``temperature.t_soisno_col``, ``site.z_col``, ``mlcanopy.gsoi_soil``; ``tk`` and
    ``cv`` from :func:`soil_thermal_properties`.
    Writes: ``temperature.t_soisno_col``; the energy conservation error (W/m2; Fortran stops
    if it exceeds 1e-6; recorded as a balance check).

    Args:
        t_soisno, tk, cv, z: ``(n_levels,)``.
        gsoi: Ground heat flux into the soil (W/m2).
        dtime: Time step (s).

    Returns:
        ``(t_soisno, gsoi - energy change)``.
    """
    n = t_soisno.shape[0]
    k = jnp.arange(n)
    fact = dtime / cv
    tk_up = jnp.concatenate([jnp.zeros(1, tk.dtype), tk[:-1]])  # tk(j-1); unused at the top
    dzm = jnp.concatenate([jnp.ones(1, z.dtype), z[1:] - z[:-1]])  # z(j) - z(j-1)
    dzp = jnp.concatenate([z[1:] - z[:-1], jnp.ones(1, z.dtype)])  # z(j+1) - z(j)
    top, bottom = k == 0, k == n - 1
    # Each row type keeps Fortran's own expression
    a = jnp.where(top, 0.0, -fact * tk_up / dzm)
    b = jnp.where(
        top,
        1.0 + fact * tk / dzp,
        jnp.where(bottom, 1.0 + fact * tk_up / dzm, 1.0 + fact * (tk_up / dzm + tk / dzp)),
    )
    c = jnp.where(bottom, 0.0, -fact * tk / dzp)
    r = jnp.where(top, t_soisno + fact * gsoi, t_soisno)

    t_new = tridiag(a, b, c, r, min_pivot=1.0e-10)

    # Energy conservation: the levels' heat change summed in Fortran's order
    def add(edif, x):
        return edif + x, None

    edif, _ = jax.lax.scan(add, jnp.zeros((), cv.dtype), cv * (t_new - t_soisno) / dtime)
    return t_new, gsoi - edif


# ---------------------------------------------------------------------------
# Freezing and thawing of soil water
# ---------------------------------------------------------------------------

#: Latent heat of fusion (J/kg) and gravitational acceleration (m/s2) as CLM has them
#: (``SHR_CONST_LATICE``, ``SHR_CONST_G``). CLM-ml v2's gravity is 9.80665.
HFUS = 3.337e5
GRAV_CLM = 9.80616


def supercooled_water(t_soisno: Array, watsat: Array, sucsat: Array, bsw: Array, dz: Array):
    """Liquid water (kg m⁻²) that stays unfrozen in a soil level below freezing: the water
    whose matric potential equals the freezing-point depression at the level's temperature
    (Niu and Yang 2006). Zero at and above freezing, where the routine does not use it.

    Not in CLM-ml v2.

    Fortran reference: CTSM SoilTemperatureMod.F90::Phasechange_beta.

    Args:
        t_soisno: (n,) soil temperature (K).
        watsat, sucsat, bsw: (n,) porosity, suction at saturation (mm), Clapp-Hornberger *b*.
        dz: (n,) thickness (m).
    """
    below = t_soisno < tfrz
    t = jnp.where(below, t_soisno, tfrz - 1.0)  # a finite power on the unused side
    smp = HFUS * (tfrz - t) / (GRAV_CLM * t) * 1000.0  # mm
    supercool = watsat * (smp / sucsat) ** (-1.0 / bsw)
    supercool = supercool * dz * 1000.0
    return jnp.where(below, supercool, 0.0)


class PhaseChange(NamedTuple):
    t_soisno: Array  # (n,) soil temperature (K)
    h2osoi_liq: Array  # (n,) liquid water (kg/m2)
    h2osoi_ice: Array  # (n,) ice (kg/m2)
    h2osno: Array  # () snow lying on the ground without snow layers (kg/m2)
    snow_depth: Array  # () its depth (m)
    qflx_snomelt: Array  # () melt of that snow (kg/m2/s): it drains to the ground
    xmf: Array  # () latent heat taken up by melting, minus that freed by freezing (W/m2)
    imelt: Array  # (n,) 1 where a level melted, 2 where it froze, 0 elsewhere


def phase_change(
    t_soisno: Array,
    h2osoi_liq: Array,
    h2osoi_ice: Array,
    h2osno,
    snow_depth,
    fact: Array,
    dhsdT,
    frac_h2osfc,
    watsat: Array,
    sucsat: Array,
    bsw: Array,
    dz: Array,
    dtime,
    *,
    derivative: str = "exact",
) -> PhaseChange:
    """Freezing and thawing of the water of the soil levels after the temperature solution.

    A level above freezing that holds ice melts; a level below freezing that holds more
    liquid water than can stay unfrozen there (:func:`supercooled_water`) freezes. Such a
    level is put at the freezing point; the heat that this frees (melting) or lacks
    (freezing), ``-(tfrz - T) / fact`` per unit time, changes the phase of that much water,
    at most the ice there is, or the liquid above the supercooled amount. Heat left over
    moves the temperature away from the freezing point again. Snow lying on the ground
    without snow layers (``h2osno``) is melted first, with the top level's heat.

    ``fact`` is the temperature change of a level per unit of heat flux into it (the model
    step over the heat capacity; K m² W⁻¹). ``dhsdT`` is the derivative of the heat flux
    into the top level with respect to its temperature (W m⁻² K⁻¹): CLM's top soil level is
    also the surface, so resetting its temperature changes the surface fluxes; with
    ``frac_h2osfc``, the ponded fraction of the ground, which has its own surface
    temperature in CLM. Where the heat flux into the soil is given, ``dhsdT`` is zero and
    the routine conserves energy exactly: ``heat capacity × temperature change / dtime +
    xmf = 0`` summed over the levels.

    ``derivative`` (static) changes derivatives only, never a result. A level that freezes
    down to its supercooled water takes that amount at the temperature before the change,
    and then warms by the heat of fusion: it ends warmer than the temperature its liquid
    water belongs to. With respect to the temperature before, the temperature after has
    the slope ``1 - gain``, ``gain`` being the heat of fusion of the supercooled water's
    change per degree over the heat capacity: 3 at -1 C and 18 at -0.2 C in a clay. Step
    after step that slope multiplies, and the exact derivative grows without bound while a
    level freezes (the forward values stay within the size of the overshoot).
    ``"stabilized"``: for such a level, the derivative of the state in which the liquid
    water is the supercooled water of the temperature after (found by bisection; slope
    ``1 / (1 + gain)``): an approximation, not the derivative of the forward values.
    ``"exact"``: the exact derivative.

    Not in CLM-ml v2. Not translated: snow layers, excess ice, urban columns.

    Fortran reference: CTSM SoilTemperatureMod.F90::Phasechange_beta.

    Args:
        t_soisno: (n,) soil temperature from the temperature solution (K).
        h2osoi_liq, h2osoi_ice: (n,) liquid water and ice (kg m⁻²).
        h2osno, snow_depth: snow without layers (kg m⁻²) and its depth (m).
        fact: (n,) see above; dhsdT, frac_h2osfc: see above.
        watsat, sucsat, bsw: (n,) porosity, suction at saturation (mm), Clapp-Hornberger *b*.
        dz: (n,) thickness (m); dtime: model step (s).
    """
    top = jnp.arange(t_soisno.shape[0]) == 0
    wice0, wliq0 = h2osoi_ice, h2osoi_liq
    wmass0 = h2osoi_ice + h2osoi_liq
    snow = h2osno > 0.0

    # Which levels change phase; they are put at the freezing point
    supercool = supercooled_water(t_soisno, watsat, sucsat, bsw, dz)
    melting = (t_soisno > tfrz) & ((h2osoi_ice > 0.0) | (top & snow))
    freezing = (t_soisno < tfrz) & (h2osoi_liq > supercool)
    flagged = melting | freezing
    tinc = jnp.where(flagged, tfrz - t_soisno, 0.0)
    t = jnp.where(flagged, tfrz, t_soisno)

    # Heat available for melting (positive) or to be freed by freezing (negative), W/m2
    hm_top = dhsdT * tinc - tinc / fact
    hm_top = jnp.where(frac_h2osfc != 0.0, hm_top - frac_h2osfc * (dhsdT * tinc), hm_top)
    hm = jnp.where(flagged, jnp.where(top, hm_top, -tinc / fact), 0.0)
    # heat of the wrong sign (possible with a surface feedback): no phase change, and the
    # level stays at the freezing point (CLM's rule)
    wrong = (melting & (hm < 0.0)) | (freezing & (hm > 0.0))
    hm = jnp.where(wrong, 0.0, hm)
    melting, freezing = melting & ~wrong, freezing & ~wrong
    changing = (melting | freezing) & (jnp.abs(hm) > 0.0)
    xm = jnp.where(changing, hm * dtime / HFUS, 0.0)  # kg/m2

    # Snow without layers is melted first, by the top level
    melts_snow = top[0] & changing[0] & snow & (xm[0] > 0.0)
    temp1 = jnp.where(snow, h2osno, 1.0)
    h2osno_left = jnp.maximum(0.0, temp1 - xm[0])
    heatr_snow = hm[0] - HFUS * (temp1 - h2osno_left) / dtime
    more = heatr_snow > 0.0
    xm = xm.at[0].set(jnp.where(melts_snow, jnp.where(more, heatr_snow * dtime / HFUS, 0.0), xm[0]))
    hm = hm.at[0].set(jnp.where(melts_snow, jnp.where(more, heatr_snow, 0.0), hm[0]))
    qflx_snomelt = jnp.where(melts_snow, jnp.maximum(0.0, temp1 - h2osno_left) / dtime, 0.0)
    snow_depth = jnp.where(melts_snow, h2osno_left / temp1 * snow_depth, snow_depth)
    h2osno = jnp.where(melts_snow, h2osno_left, h2osno)

    # Ice after the change, and the heat left over
    melt = changing & (xm > 0.0)
    freeze = changing & (xm < 0.0)
    ice_melted = jnp.maximum(0.0, wice0 - xm)
    ice_frozen = jnp.where(wmass0 < supercool, 0.0, jnp.minimum(wmass0 - supercool, wice0 - xm))
    ice = jnp.where(melt, ice_melted, jnp.where(freeze, ice_frozen, wice0))
    heatr = jnp.where(melt | freeze, hm - HFUS * (wice0 - ice) / dtime, 0.0)
    liq = jnp.where(changing, jnp.maximum(0.0, wmass0 - ice), wliq0)

    # The heat left over moves the temperature away from the freezing point
    feedback = jnp.where(top, 1.0 - (1.0 - frac_h2osfc) * fact * dhsdT, 1.0)
    warmed = t + fact * heatr / feedback
    t = jnp.where(changing & (jnp.abs(heatr) > 0.0), warmed, t)

    if derivative == "stabilized":
        # Levels frozen down to their supercooled water: the derivatives of the state whose
        # liquid water is the supercooled water of the temperature after the change
        limited = freeze & (wmass0 >= supercool) & (wmass0 - supercool < wice0 - xm)

        def after(temperature):
            """Temperature after the change if the level kept the supercooled water of
            ``temperature`` as liquid; that water."""
            keeps = supercooled_water(temperature, watsat, sucsat, bsw, dz)
            return tfrz + fact * (hm - HFUS * (wice0 - (wmass0 - keeps)) / dtime) / feedback, keeps

        def bisect(bounds, _):
            low, high = bounds
            middle = 0.5 * (low + high)
            above = middle - after(middle)[0] > 0.0
            return (jnp.where(above, low, middle), jnp.where(above, middle, high)), None

        start = jax.lax.stop_gradient(jnp.where(limited, t_soisno, tfrz - 1.0))
        (low, high), _ = jax.lax.scan(bisect, (start, jnp.full_like(start, tfrz)), None, length=50)
        root = jax.lax.stop_gradient(0.5 * (low + high))
        # one Newton step from the root carries the derivative of the root (the implicit
        # function theorem) and leaves its value
        residual = root - after(root)[0]
        # each level's residual depends on its own temperature only
        _, slope = jax.jvp(lambda x: x - after(x)[0], (root,), (jnp.ones_like(root),))
        slope = jax.lax.stop_gradient(slope)
        consistent_t = root - (residual - jax.lax.stop_gradient(residual)) / slope
        consistent_liq = after(consistent_t)[1]
        consistent_ice = wmass0 - consistent_liq

        def with_rule(value, consistent):
            # the bracket is exactly zero: the value is the routine's own
            ruled = jax.lax.stop_gradient(value) + (consistent - jax.lax.stop_gradient(consistent))
            return jnp.where(limited, ruled, value)

        t = with_rule(t, consistent_t)
        ice = with_rule(ice, consistent_ice)
        liq = with_rule(liq, consistent_liq)
    elif derivative != "exact":
        raise ValueError(f"phase change derivative {derivative!r}: expected exact or stabilized")

    # Latent heat of the column: the snow's, then the levels' in order
    def add(total, x):
        return total + x, None

    xmf, _ = jax.lax.scan(
        add, HFUS * qflx_snomelt, jnp.where(changing, HFUS * (wice0 - ice) / dtime, 0.0)
    )
    imelt = jnp.where(melting, 1, jnp.where(freezing, 2, 0))
    return PhaseChange(t, liq, ice, h2osno, snow_depth, qflx_snomelt, xmf, imelt)
