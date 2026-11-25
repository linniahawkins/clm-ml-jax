"""Soil hydrology below the surface: the water table and the drainage of the saturated zone.

The water table is diagnosed from the soil water. Drainage (baseflow) leaves the saturated
zone above the bedrock at a rate set by the slope and by the thickness of that zone, and is
taken from the soil levels from the water table downward. After it, water above saturation
is passed upward (from the top level to the ponded surface water) and levels with less than
a minimum of water are refilled from their neighbours.

Each function computes one soil column. Soil levels are 0-based from the top down (level
``k`` is Fortran's layer ``j = k + 1``); ``zi`` holds the depths of the level interfaces
from the surface (``zi[0] = 0``, ``zi[k + 1]`` the bottom of level ``k``). ``nbedrock`` is
the number of levels above the bedrock; the routines also handle the levels below it, as
CLM does. Water stores are in mm (kg m⁻²), fluxes in mm s⁻¹, depths in m.

Not in CLM-ml v2: these are translated from CLM itself (CTSM ``ctsm5.4.004``), for CLM's
default of no flow through the bottom of the column and no aquifer, and for a column that
is not part of a hillslope.

Fortran reference: CTSM ``SoilHydrologyMod.F90``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from clm_ml.processes.constants import denh2o, denice

#: Relative saturation at or below which a level counts as unsaturated in the search for
#: the water table. CLM writes 0.9 in single precision.
SAT_LEV = float(np.float32(0.9))
#: Smallest liquid water kept in a soil level (mm).
WATMIN = 0.01
#: Deepest water table (m).
ZWT_MAX = 80.0


def _sequential_sum(x: Array) -> Array:
    """Sum of ``x`` added in order from the first element (Fortran's accumulation loop)."""
    total, _ = jax.lax.scan(lambda acc, v: (acc + v, None), jnp.zeros((), x.dtype), x)
    return total


def water_table_depth(
    h2osoi_liq: Array,
    h2osoi_ice: Array,
    watsat: Array,
    dz: Array,
    z: Array,
    zi: Array,
    nbedrock,
) -> Array:
    """Depth of the water table (m) from the soil water.

    The deepest level above the bedrock whose water (liquid and ice) fills 90 % of the
    pores or less lies above the water table; the depth is interpolated between that
    level's node and the next, by their relative saturations. If that level is the top one,
    or every level is wetter, the water table is at the bottom of the top level; if it is
    the lowest level above the bedrock, at the bedrock.

    The depth jumps where the deepest unsaturated level changes; this is CLM's.

    Args:
        h2osoi_liq, h2osoi_ice: (n,) liquid water and ice (kg m⁻²).
        watsat: (n,) porosity; dz, z: (n,) thickness and node depth (m).
        zi: (n + 1,) interface depths (m).
        nbedrock: number of levels above the bedrock.

    Fortran reference: SoilHydrologyMod.F90::ThetaBasedWaterTable.
    """
    n = h2osoi_liq.shape[0]
    level = jnp.arange(n)
    s = (h2osoi_liq / (dz * denh2o) + h2osoi_ice / (dz * denice)) / watsat
    unsaturated = (s <= SAT_LEV) & (level < nbedrock)
    k = jnp.max(jnp.where(unsaturated, level, 0))  # 0 also when no level is unsaturated

    below = jnp.minimum(k + 1, n - 1)
    interpolate = (k > 0) & (k < nbedrock - 1)
    s1, s2 = s[k], s[below]
    m = (z[below] - z[k]) / jnp.where(interpolate, s2 - s1, 1.0)
    b = z[below] - m * s2
    return jnp.where(
        k == 0,
        zi[1],
        jnp.where(interpolate, jnp.maximum(0.0, m * SAT_LEV + b), zi[nbedrock]),
    )


class SubsurfaceDrainage(NamedTuple):
    h2osoi_liq: Array  # (n,) liquid water (kg/m2)
    h2osoi_ice: Array  # (n,) ice (kg/m2): the top level's is limited to its pore space
    zwt: Array  # water table depth (m)
    h2osfc: Array  # ponded water (mm), with the water passed up from the top level
    qflx_latflow_out: Array  # baseflow asked of the saturated zone (mm/s)
    qflx_rsub_sat: Array  # minus the water created to keep the minimum (mm/s)
    qflx_drain: Array  # drainage leaving the column (mm/s)
    qflx_ice_runoff_xs: Array  # ice above the top level's pore space, removed (mm/s)


def subsurface_drainage(
    h2osoi_liq: Array,
    h2osoi_ice: Array,
    zwt,
    h2osfc,
    eff_porosity: Array,
    watsat: Array,
    sucsat: Array,
    bsw: Array,
    dz: Array,
    zi: Array,
    nbedrock,
    topo_slope,
    dtime,
    baseflow_scalar,
    n_baseflow,
    aq_sp_yield_min,
    e_ice,
) -> SubsurfaceDrainage:
    """Drain the saturated zone over a model step and restore the bounds of the soil water.

    1. *Baseflow.* With the water table above the bedrock, ``baseflow_scalar ×
       tan(slope) × (bedrock depth − water table depth)^n_baseflow``, reduced by the ice of
       the levels from the water table down.
    2. *Removal.* The water is taken level by level from the one holding the water table
       down to the bedrock; each gives what lies between the water table and its bottom,
       times its specific yield (from the retention curve, at least ``aq_sp_yield_min``),
       and the water table falls accordingly. What the levels cannot give is not drained.
    3. *Water above saturation* (above the effective porosity) is passed to the level
       above, from the bottom up; from the top level to the ponded surface water. Ice of
       the top level above its pore space is removed as runoff.
    4. *Minimum water.* A level with less than ``WATMIN`` takes the difference from the
       level below; the lowest level takes it from the levels above that can spare it, and
       water still missing is created and subtracted from the drainage.

    Steps 3 and 4 run over every level, below the bedrock too.

    Args:
        h2osoi_liq, h2osoi_ice: (n,) liquid water and ice (kg m⁻²).
        zwt: water table depth (m); h2osfc: ponded water (mm).
        eff_porosity: (n,) porosity less the ice volume (``soil_water.soil_water_fractions``).
        watsat, sucsat, bsw: (n,) porosity, suction at saturation (mm), Clapp-Hornberger *b*.
        dz: (n,) thickness (m); zi: (n + 1,) interface depths (m).
        nbedrock: number of levels above the bedrock.
        topo_slope: topographic slope (degrees); dtime: model step (s).
        baseflow_scalar, n_baseflow, aq_sp_yield_min, e_ice: CLM parameters.

    Not translated: CLM's branch for a rising water table. It needs water flowing in from a
    neighbouring column, which only hillslope columns have.

    Fortran reference: SoilHydrologyMod.F90::SubsurfaceLateralFlow.
    """
    n = h2osoi_liq.shape[0]
    level = jnp.arange(n)
    dzmm = dz * 1.0e3
    zi_bottom = zi[1:]  # bottom of each level
    zbedrock = zi[nbedrock]

    vol_ice = jnp.minimum(watsat, h2osoi_ice / (dz * denice))
    icefrac = jnp.minimum(1.0, vol_ice / watsat)
    # number of levels wholly above the water table (Fortran's jwt)
    jwt = jnp.sum(zi_bottom < zwt)

    # 1. Baseflow, reduced by the mean ice fraction from the water table down
    from_table = level >= jnp.maximum(jwt, 1) - 1
    dzsum = _sequential_sum(jnp.where(from_table, dzmm, 0.0))
    icefracsum = _sequential_sum(jnp.where(from_table, icefrac * dzmm, 0.0))
    ice_imped_col = 10.0 ** (-e_ice * (icefracsum / dzsum))
    saturated_zone = zwt <= zbedrock
    thickness = zbedrock - zwt
    positive = saturated_zone & (thickness > 0.0)
    qflx_latflow_out = jnp.where(
        positive,
        ice_imped_col
        * baseflow_scalar
        * jnp.tan(math.pi / 180.0 * topo_slope)
        * jnp.where(positive, thickness, 1.0) ** n_baseflow,
        0.0,
    )
    drainage = jnp.where(saturated_zone, qflx_latflow_out, 0.0)

    # 2. Removal, from the level holding the water table down to the bedrock
    def remove(carry, row):
        drainage_tot, zwt, done = carry
        k, watsat_k, sucsat_k, bsw_k, zi_k = row
        active = (k >= jwt) & (k < nbedrock) & ~done
        s_y = watsat_k * (1.0 - (1.0 + 1.0e3 * zwt / sucsat_k) ** (-1.0 / bsw_k))
        s_y = jnp.maximum(s_y, aq_sp_yield_min)
        layer = jnp.maximum(drainage_tot, -(s_y * (zi_k - zwt) * 1.0e3))
        layer = jnp.where(active, jnp.minimum(layer, 0.0), 0.0)
        drainage_tot = drainage_tot - layer
        finished = active & (drainage_tot >= 0.0)
        zwt = jnp.where(finished, zwt - layer / s_y / 1000.0, jnp.where(active, zi_k, zwt))
        return (drainage_tot, zwt, done | finished), layer

    (drainage_tot, zwt, _), removed = jax.lax.scan(
        remove,
        (-drainage * dtime, jnp.asarray(zwt, dtype=h2osoi_liq.dtype), jnp.zeros((), bool)),
        (level, watsat, sucsat, bsw, zi_bottom),
    )
    liq = h2osoi_liq + removed
    drainage = drainage + drainage_tot / dtime
    zwt = jnp.minimum(ZWT_MAX, jnp.maximum(0.0, zwt))

    # 3. Water above saturation is passed upward
    capacity = eff_porosity * dzmm

    def pass_up(from_below, row):
        liq_k, capacity_k = row
        liq_k = liq_k + from_below
        return jnp.maximum(liq_k - capacity_k, 0.0), jnp.minimum(capacity_k, liq_k)

    to_top, liq_lower = jax.lax.scan(
        pass_up, jnp.zeros((), liq.dtype), (liq[1:], capacity[1:]), reverse=True
    )
    liq_top = liq[0] + to_top
    pores_top = watsat[0] * dzmm[0]
    ice_top = h2osoi_ice[0]
    xs1 = jnp.maximum(
        jnp.maximum(liq_top - WATMIN, 0.0) - jnp.maximum(0.0, pores_top - ice_top - WATMIN), 0.0
    )
    liq_top = liq_top - xs1
    h2osfc = h2osfc + xs1
    room_for_ice = jnp.maximum(0.0, pores_top - liq_top)
    qflx_ice_runoff_xs = jnp.maximum(jnp.maximum(ice_top, 0.0) - room_for_ice, 0.0) / dtime
    ice_top = jnp.minimum(room_for_ice, ice_top)

    # 4. Minimum water: each level but the lowest takes what it lacks from the level below
    liq = jnp.concatenate([liq_top[None], liq_lower])

    def refill(carry, row):
        given_up, zwt = carry
        k, liq_k, eff_porosity_k = row
        liq_k = liq_k - given_up
        short = liq_k < WATMIN
        xs = jnp.where(short, WATMIN - liq_k, 0.0)
        # the water table deepens if the level just above it takes water from below
        zwt = jnp.where(short & (k + 1 == jwt), zwt + xs / eff_porosity_k / 1000.0, zwt)
        return (xs, zwt), liq_k + xs

    (given_up, zwt), liq_upper = jax.lax.scan(
        refill, (jnp.zeros((), liq.dtype), zwt), (level[:-1], liq[:-1], eff_porosity[:-1])
    )
    liq_bottom = liq[-1] - given_up

    # the lowest level takes what it lacks from the levels above, nearest first
    def search(carry, liq_k):
        xs, liq_bottom, done = carry
        available = jnp.maximum(liq_k - WATMIN - xs, 0.0)
        enough = available >= xs
        moved = jnp.where(done, 0.0, jnp.where(enough, xs, available))
        xs = jnp.where(done, xs, jnp.where(enough, 0.0, xs - available))
        return (xs, liq_bottom + moved, done | enough), liq_k - moved

    short = liq_bottom < WATMIN
    xs0 = jnp.where(short, WATMIN - liq_bottom, 0.0)
    (xs, liq_bottom, _), liq_upper = jax.lax.scan(
        search, (xs0, liq_bottom, ~short), liq_upper, reverse=True
    )
    liq_bottom = liq_bottom + xs  # water that no level could give is created
    qflx_rsub_sat = 0.0 - xs / dtime

    return SubsurfaceDrainage(
        h2osoi_liq=jnp.concatenate([liq_upper, liq_bottom[None]]),
        h2osoi_ice=jnp.concatenate([ice_top[None], h2osoi_ice[1:]]),
        zwt=zwt,
        h2osfc=h2osfc,
        qflx_latflow_out=qflx_latflow_out,
        qflx_rsub_sat=qflx_rsub_sat,
        qflx_drain=qflx_rsub_sat + drainage,
        qflx_ice_runoff_xs=qflx_ice_runoff_xs,
    )
