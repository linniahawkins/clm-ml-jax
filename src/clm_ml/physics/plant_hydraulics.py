"""Plant hydraulics (SPA-style): soil and plant resistance to water flow, the soil water
potential seen by the roots, and the leaf water potential of the sunlit and shaded leaves.

One patch. Layers are 0-based slots on axis 0; leaves 0 sunlit, 1 shaded; soil levels 0-based
(level 0 is the top soil layer, Fortran ``j = 1``).

Fortran reference: CLM-ml v2 ``MLPlantHydraulicsMod.F90``. Bonan et al. (2014) Geosci. Model
Dev. 7, 2193-2222, appendix A (eqs. A19-A28).

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the CLM-ml v2 (``mlcanopy_type``) fields they correspond to.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from clm_ml.physics.constants import denh2o, grav, mmh2o, rpi

#: Pressure of a 1 m water column, MPa/m
HEAD = denh2o * grav * 1.0e-6
#: Minimum leaf water potential, MPa (Fortran local parameter ``minlwp_SPA``)
MINLWP_SPA = -2.0


def plant_resistance(dpai: Array, rsoil, plant_conductance) -> Array:
    """Leaf-specific conductance from the soil to the leaves, mmol H2O/m2 leaf/s/MPa
    (layer,): ``1 / (rsoil + 1 / plant_conductance)``; 0 where ``dpai == 0``.

    Fortran reference: MLPlantHydraulicsMod.F90::PlantResistance (Bonan et al. 2014,
    eqs. A21-A22).
    Reads: ``dpai_profile``, ``rsoil_soil``, the PFT's ``gplant_SPA``. Writes: ``lsc_profile``.
    """
    rplant = 1.0 / plant_conductance
    return jnp.where(dpai > 0.0, 1.0 / (rsoil + rplant), 0.0)


class SoilResistance(NamedTuple):
    psis: Array  # () weighted soil water potential, MPa
    rsoil: Array  # () soil hydraulic resistance, MPa s m2/mmol H2O
    soil_et_loss: Array  # (soil_level,) fraction of the transpiration from each level


def soil_resistance(
    hk: Array,
    smp: Array,
    dz: Array,
    rootfr: Array,
    h2osoi_ice: Array,
    n_bedrock,
    lai,
    root_biomass,
    root_radius,
    root_density,
    root_resistivity,
) -> SoilResistance:
    """Soil-to-root and root-to-stem resistance of the soil levels above bedrock (soil level
    ``k`` is used if ``k < n_bedrock``), their sum as a whole-canopy resistance, and the soil
    water potential and transpiration fractions weighted by the maximum uptake from each level.
    Inputs per soil level: hydraulic conductivity ``hk`` (mm/s), matric potential ``smp``
    (mm), thickness ``dz`` (m), root fraction ``rootfr``, ice ``h2osoi_ice`` (frozen levels
    give no water).

    Fortran reference: MLPlantHydraulicsMod.F90::SoilResistance (Bonan et al. 2014,
    eqs. A23-A28).
    Reads: ``hk_l_col``, ``smp_l_col``, ``dz_col``, ``rootfr_patch``, ``h2osoi_ice_col``,
    ``nbedrock_col``, ``lai_canopy``, ``root_biomass_canopy``, the PFT's root parameters.
    Writes: ``psis_soil``, ``rsoil_soil``, ``soil_et_loss_soil``.
    """
    active = jnp.arange(hk.shape[0]) < n_bedrock
    root_cross_sec_area = rpi * root_radius * root_radius

    # Hydraulic conductivity (mm/s -> mmol/m/s/MPa) and matric potential (mm -> MPa)
    hk_v = hk * (1.0e-3 / HEAD) * denh2o / mmh2o * 1000.0
    smp_mpa = smp * 1.0e-3 * HEAD

    # Root biomass density (g/m3), root length density (m/m3), mean distance between roots
    rbd = jnp.maximum(root_biomass * rootfr / dz, 1.0e-10)
    # Guard rld > 0 for safe 1/rld and sqrt(1/rld) gradients
    rld = jnp.maximum(rbd / (root_density * root_cross_sec_area), 1.0e-30)
    root_dist = jnp.sqrt(1.0 / (rld * rpi))

    # Soil-to-root (A23) and root-to-stem (A24) resistance. hk is 0 in frozen levels; the
    # floor prevents NaN gradients.
    hk_safe = jnp.maximum(hk_v, 1.0e-30)
    soilr1 = jnp.log(root_dist / root_radius) / (2.0 * rpi * rld * dz * hk_safe)
    soilr2 = root_resistivity / (rbd * dz)
    # Guard soilr > 0 for safe division gradients inside jnp.where
    soilr_safe = jnp.maximum(soilr1 + soilr2, 1.0e-30)

    # Maximum transpiration of each level (A26); none from frozen levels or below bedrock
    evap = jnp.maximum((smp_mpa - MINLWP_SPA) / soilr_safe, 0.0)
    evap = jnp.where(active & ~(h2osoi_ice > 0.0), evap, 0.0)

    # Belowground resistance (A25)
    rsoil_sum = jnp.sum(jnp.where(active, 1.0 / soilr_safe, 0.0))
    rsoil = lai / jnp.maximum(rsoil_sum, 1.0e-30)

    # Weighted soil water potential and transpiration fractions (A27-A28); with no uptake,
    # the minimum potential and equal fractions above bedrock
    totevap = jnp.sum(evap)
    # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
    totevap_safe = jnp.maximum(totevap, 1.0e-30)
    psis = jnp.where(totevap > 0.0, jnp.sum(smp_mpa * evap) / totevap_safe, MINLWP_SPA)
    n_layers_safe = jnp.maximum(jnp.asarray(n_bedrock).astype(jnp.float64), 1.0)
    uniform = jnp.where(active, 1.0 / n_layers_safe, 0.0)
    soil_et_loss = jnp.where(totevap > 0.0, evap / totevap_safe, uniform)
    return SoilResistance(psis=psis, rsoil=rsoil, soil_et_loss=soil_et_loss)


def leaf_water_potential(
    dpai: Array,
    zs: Array,
    lsc: Array,
    trleaf: Array,
    lwp_bef: Array,
    psis,
    plant_capacitance,
    dtime: float,
) -> Array:
    """Leaf water potential, MPa (layer, leaf), of the sunlit and shaded leaves after one
    time step ``dtime`` (s): the analytical solution of ``dy/dt = (a - y) / b`` from
    ``lwp_bef``, with ``a = psis - HEAD zs - 1000 trleaf / lsc`` (transpiration ``trleaf``,
    mol H2O/m2 leaf/s; layer height ``zs``, m) and ``b = plant_capacitance / lsc``; 0 where
    ``dpai == 0``.

    Fortran reference: MLPlantHydraulicsMod.F90::LeafWaterPotential (Bonan et al. 2014,
    eqs. A19-A20; Fortran calls it once per leaf).
    Reads: ``dpai_profile``, ``zs_profile``, ``lsc_profile``, ``trleaf_leaf``,
    ``lwp_bef_leaf``, ``psis_soil``, the PFT's ``capac_SPA``. Writes: ``lwp_leaf``.
    """
    has_pai = (dpai > 0.0)[:, None]
    # jnp.maximum avoids select op -> prevents XLA select_divide_fusion bug
    lsc_safe = jnp.maximum(lsc, 1.0e-30)[:, None]  # avoid /0 in layers without plants
    a = psis - HEAD * zs[:, None] - 1000.0 * trleaf / lsc_safe
    b = plant_capacitance / lsc_safe
    dy = (a - lwp_bef) * (1.0 - jnp.exp(-dtime / b))
    return jnp.where(has_pai, lwp_bef + dy, 0.0)
