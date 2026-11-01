"""Leaf heat capacity of each canopy layer (the heat-storage term of the leaf energy balance).

One patch. Layers are 0-based slots on axis 0.

Fortran reference: CLM-ml v2 ``MLLeafHeatCapacityMod.F90::LeafHeatCapacity``. Bonan et al.
(2018) Geosci. Model Dev. 11, 1467-1496, eq. (A29).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import cpbio, cpliq, fcarbon, fwater


def leaf_heat_capacity(dpai: Array, sla_top) -> Array:
    """Leaf heat capacity, J/K/m2 leaf (layer,), from the specific leaf area at the canopy
    top ``sla_top`` (m2/gC): leaf dry mass and leaf water per area, each with its specific
    heat; 0 where ``dpai == 0``.

    Fortran reference: MLLeafHeatCapacityMod.F90::LeafHeatCapacity.
    Reads: ``dpai_profile``, the PFT's ``slatop``. Writes: ``cpleaf_profile``.
    """
    lma = 1.0 / sla_top * 0.001  # leaf carbon mass per area, kg C/m2
    dry_weight = lma / fcarbon  # leaf dry mass per area, kg DM/m2
    fresh_weight = dry_weight / (1.0 - fwater)  # leaf fresh mass per area, kg FM/m2
    leaf_water = fwater * fresh_weight  # leaf water per area, kg H2O/m2
    cpleaf = cpbio * dry_weight + cpliq * leaf_water
    return jnp.where(dpai > 0.0, cpleaf, 0.0)
