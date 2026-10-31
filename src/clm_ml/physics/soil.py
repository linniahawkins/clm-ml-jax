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

Each function's reads and writes are its arguments and results; ``Reads``/``Writes`` in the
docstrings name the legacy container fields they come from and go to (through the adapters in
``clm_ml.legacy.soil`` until the containers leave the step, Phase 7.2 process 10).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.physics.constants import (
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
from clm_ml.physics.math import tridiag


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
    h2osoi_clm45: Array,
    h2osoi_clm50: Array,
    h2osoi_factor,
    dz: Array,
    watsat: Array,
    nbedrock,
    *,
    clm_phys: str,
    n_soil: int,
    n_adjust: int,
) -> SoilWater:
    """Soil water of each level from the prescribed CLM history values.

    Fortran reference: clmDataMod.F90::clmData (soil water part).

    CLM4.5 histories give every level; CLM5.0 histories give the ``n_soil`` (``nlevsoi``)
    soil levels, the bedrock levels below are 0, and the values are capped at porosity down to
    the bedrock. The first ``n_adjust`` levels are multiplied by ``h2osoi_factor``. All water
    is liquid.

    Reads: the history record (``h2osoi_clm45``/``h2osoi_clm50``, ``h2osoi_factor``),
    ``site.dz_col``, ``site.nbedrock_col``, ``soilstate.watsat_col``.
    Writes: ``waterstatebulk.h2osoi_vol_col``, ``h2osoi_liq_col``, ``h2osoi_ice_col``.

    Args:
        h2osoi_clm45: ``(n_levels,)`` history values (used if ``clm_phys == "CLM4_5"``).
        h2osoi_clm50: ``(n_soil,)`` history values (used if ``clm_phys == "CLM5_0"``).
        h2osoi_factor: Adjustment factor (used if ``n_adjust > 0``).
        dz: ``(n_levels,)`` level thickness (m).
        watsat: ``(n_levels,)`` porosity.
        nbedrock: Index (1-based) of the deepest level above bedrock (data).
        clm_phys, n_soil, n_adjust: Static settings (``clmSoilOptionMod``, ``nlevsoi``).
    """
    n = dz.shape[0]
    if clm_phys == "CLM4_5":
        vol = jnp.asarray(h2osoi_clm45, dtype=dz.dtype)
    elif clm_phys == "CLM5_0":
        vol = jnp.concatenate(
            [jnp.asarray(h2osoi_clm50, dtype=dz.dtype), jnp.zeros(n - n_soil, dz.dtype)]
        )
    else:
        raise ValueError(f"clm_phys = {clm_phys!r}: CLM4_5 or CLM5_0")
    if n_adjust > 0:
        vol = vol.at[:n_adjust].set(vol[:n_adjust] * h2osoi_factor)
    if clm_phys == "CLM5_0":
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

    # Surface water conductivity
    zh2osfc = 1.0e-3 * (0.5 * h2osfc)
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
