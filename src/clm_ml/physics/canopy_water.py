"""Canopy water: wetted fraction of the leaves, interception and throughfall of precipitation,
and evaporation and dew on the intercepted water.

One patch. Layers are 0-based slots on axis 0; leaves 0 sunlit, 1 shaded. Every slot is
computed; layers without plant area (``dpai == 0``, which includes the layers above the
canopy and the padding) get 0 or keep their water, as in Fortran.

Fortran reference: CLM-ml v2 ``MLCanopyWaterMod.F90``.

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the legacy container fields they come from and go to (through the adapters in
``clm_ml.legacy.canopy_water`` until the containers leave the step, Phase 7.2 process 10).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from clm_ml.physics.constants import (
    dewmx,
    fwet_exponent,
    interception_fraction,
    maximum_leaf_wetted_fraction,
    mmh2o,
)


def wetted_fraction(h2ocan: Array, dpai: Array, dlai: Array) -> tuple[Array, Array]:
    """Wetted fraction of each layer's plant area, and the dry fraction of its leaves
    (layer,), from the intercepted water ``h2ocan`` (kg H2O/m2):
    ``fwet = min((h2ocan / (dewmx dpai))^fwet_exponent, maximum_leaf_wetted_fraction)``,
    ``fdry = (1 - fwet) dlai / dpai``; both 0 where ``dpai == 0``.

    Fortran reference: MLCanopyWaterMod.F90::CanopyWettedFraction.
    Reads: ``h2ocan_profile``, ``dpai_profile``, ``dlai_profile``. Writes: ``fwet_profile``,
    ``fdry_profile``.
    """
    has_pai = dpai > 0.0
    dpai_safe = jnp.where(has_pai, dpai, 1.0)  # avoid /0
    h2ocanmx = dewmx * dpai_safe
    # The base of the fractional power has a tiny floor: 0^0.67 has an infinite derivative,
    # which the jnp.where below would turn into a NaN gradient
    fwet_base = jnp.maximum(h2ocan / h2ocanmx, 1.0e-30)
    fwet = jnp.where(
        has_pai, jnp.minimum(fwet_base**fwet_exponent, maximum_leaf_wetted_fraction), 0.0
    )
    fdry = jnp.where(has_pai, (1.0 - fwet) * dlai / dpai_safe, 0.0)
    return fwet, fdry


class Interception(NamedTuple):
    h2ocan: Array  # (layer,) intercepted water after interception and drip, kg H2O/m2
    qflx_intr: Array  # () intercepted precipitation, kg H2O/m2/s
    qflx_tflrain: Array  # () rain onto the ground (throughfall and drip), kg H2O/m2/s
    qflx_tflsnow: Array  # () snow onto the ground (throughfall and drip), kg H2O/m2/s


def interception(
    qflx_rain, qflx_snow, lai, sai, dpai: Array, h2ocan_bef: Array, dtime: float
) -> Interception:
    """Interception of rain and snow (CLM5 form, ``interception_fraction tanh(lai + sai)``),
    shared equally by the layers with plant area; water above a layer's capacity
    ``dewmx dpai`` drips to the ground with the throughfall.

    Fortran reference: MLCanopyWaterMod.F90::CanopyInterception.
    Reads: ``qflx_rain_forcing``, ``qflx_snow_forcing``, ``lai_canopy``, ``sai_canopy``,
    ``dpai_profile``, ``h2ocan_bef_profile``. Writes: ``h2ocan_profile``,
    ``qflx_intr_canopy``, ``qflx_tflrain_canopy``, ``qflx_tflsnow_canopy``.
    """
    # Rain and snow fractions
    total_precip = qflx_snow + qflx_rain
    # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
    total_safe = jnp.maximum(total_precip, 1.0e-30)
    fracrain = jnp.where(total_precip > 0.0, qflx_rain / total_safe, 0.0)
    fracsnow = jnp.where(total_precip > 0.0, qflx_snow / total_safe, 0.0)

    # Intercepted fraction (CLM5 form), direct throughfall, intercepted precipitation
    fpi = interception_fraction * jnp.tanh(lai + sai)
    qflx_through_rain = qflx_rain * (1.0 - fpi)
    qflx_through_snow = qflx_snow * (1.0 - fpi)
    qflx_intr = total_precip * fpi

    # Water balance of the layers with plant area
    has_pai = dpai > 0.0
    n_active = jnp.sum(has_pai.astype(jnp.float64))
    n_safe = jnp.maximum(n_active, 1.0)  # avoid /0
    # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
    h2ocanmx = dewmx * jnp.maximum(dpai, 1.0e-30)
    add_per_layer = jnp.where(n_active > 0.0, qflx_intr * dtime / n_safe, 0.0)
    h2ocan = jnp.where(has_pai, h2ocan_bef + add_per_layer, 0.0)
    xrun = jnp.where(has_pai, (h2ocan - h2ocanmx) / dtime, 0.0)
    drip = xrun > 0.0

    # Canopy drip; intercepted water capped at the layer's capacity
    qflx_candrip = jnp.sum(jnp.where(drip, xrun, 0.0))
    h2ocan = jnp.where(drip, h2ocanmx, h2ocan)

    return Interception(
        h2ocan=h2ocan,
        qflx_intr=qflx_intr,
        qflx_tflrain=qflx_through_rain + qflx_candrip * fracrain,
        qflx_tflsnow=qflx_through_snow + qflx_candrip * fracsnow,
    )


def evaporation(
    h2ocan: Array, dpai: Array, fracsun: Array, evleaf: Array, trleaf: Array, dtime: float
) -> Array:
    """Intercepted water (layer,) after dew and evaporation of the sunlit and shaded leaves
    (``evleaf``, ``trleaf``: (layer, leaf), mol H2O/m2 leaf/s). Dew (a negative
    evaporation-plus-transpiration flux) is added first, then evaporation removed; layers
    without plant area keep their water. There is no floor at 0 (commented out in Fortran).

    Fortran reference: MLCanopyWaterMod.F90::CanopyEvaporation.
    Reads: ``h2ocan_profile``, ``dpai_profile``, ``fracsun_profile``, ``evleaf_leaf``,
    ``trleaf_leaf``. Writes: ``h2ocan_profile``.
    """
    has_pai = dpai > 0.0
    factor = jnp.where(has_pai, dpai, 0.0) * mmh2o * dtime
    fracsha = 1.0 - fracsun

    # Dew (a negative flux adds water)
    dew_sun = (evleaf[:, 0] + trleaf[:, 0]) * fracsun * factor
    dew_sha = (evleaf[:, 1] + trleaf[:, 1]) * fracsha * factor
    h2o = h2ocan - jnp.where(dew_sun < 0.0, dew_sun, 0.0)
    h2o = h2o - jnp.where(dew_sha < 0.0, dew_sha, 0.0)

    # Evaporation (a positive flux removes water)
    h2o = h2o - jnp.where(evleaf[:, 0] > 0.0, evleaf[:, 0] * fracsun * factor, 0.0)
    h2o = h2o - jnp.where(evleaf[:, 1] > 0.0, evleaf[:, 1] * fracsha * factor, 0.0)

    return jnp.where(has_pai, h2o, h2ocan)
