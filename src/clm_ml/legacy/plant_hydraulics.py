"""The plant hydraulics routines of the legacy time step, and the leaf heat capacity computed
with them at the start of each step, as adapters to ``clm_ml.physics.plant_hydraulics`` and
``clm_ml.physics.leaf_heat_capacity`` (Phase 7.2, process 5; transition only: they go when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (index 0 the ground), leaves
``isun = 1``/``isha = 2`` (index 0 unused), soil level ``j`` at index ``j`` (but
``h2osoi_ice_col`` holds level ``j`` at ``j - 1``). Every layer slot is written, as before.
``DECLARED`` lists the container fields each adapter reads and writes (checked by
``tests/clm_ml/test_physics_canopy_water_hydraulics.py``).
"""

from __future__ import annotations

from typing import Sequence

from clm_ml.physics import leaf_heat_capacity as heat
from clm_ml.physics import plant_hydraulics as physics
from clm_src_main.PatchType import patch
from clm_src_main.SiteType import site_or_modules
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
from multilayer_canopy.MLclm_varpar import isha, isun

_ML = "mlcanopy."
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "PlantResistance": (
        (_ML + "dpai_profile", _ML + "rsoil_soil", _ML + "lsc_profile"),
        (_ML + "lsc_profile",),
    ),
    "SoilResistance": (
        ("site.dz_col", "site.nbedrock_col", "soilstate.smp_l_col", "soilstate.hk_l_col",
         "soilstate.rootfr_patch", "waterstatebulk.h2osoi_ice_col",
         *(_ML + f for f in ("lai_canopy", "root_biomass_canopy", "psis_soil", "rsoil_soil",
                             "soil_et_loss_soil"))),
        tuple(_ML + f for f in ("psis_soil", "rsoil_soil", "soil_et_loss_soil")),
    ),
    "LeafWaterPotential": (
        tuple(_ML + f for f in ("dpai_profile", "zs_profile", "lsc_profile", "trleaf_leaf",
                                "lwp_bef_leaf", "psis_soil", "lwp_leaf")),
        (_ML + "lwp_leaf",),
    ),
    "LeafHeatCapacity": (
        (_ML + "dpai_profile", _ML + "cpleaf_profile"),
        (_ML + "cpleaf_profile",),
    ),
}  # fmt: skip
_LEAVES = slice(isun, isha + 1)  # sunlit, shaded


def PlantResistance(
    num_filter: int, filter_patch: Sequence[int], mlcanopy_inst: mlcanopy_type, params
) -> mlcanopy_type:
    """Leaf-specific conductance of each layer (``physics.plant_resistance``).

    Fortran reference: MLPlantHydraulicsMod.F90::PlantResistance.
    """
    inst = mlcanopy_inst
    lsc = inst.lsc_profile
    for fp in range(num_filter):
        p = filter_patch[fp]
        new = physics.plant_resistance(
            inst.dpai_profile[p, 1:], inst.rsoil_soil[p], params.plant_conductance
        )
        lsc = lsc.at[p, 1:].set(new)
    return inst._replace(lsc_profile=lsc)


def SoilResistance(
    num_filter: int,
    filter_patch: Sequence[int],
    soilstate_inst,
    waterstatebulk_inst,
    mlcanopy_inst: mlcanopy_type,
    params,
    site=None,
) -> mlcanopy_type:
    """Soil resistance, soil water potential and transpiration fractions of each patch
    (``physics.soil_resistance``; soil geometry and bedrock from ``site``, default: the
    subgrid modules' values).

    Fortran reference: MLPlantHydraulicsMod.F90::SoilResistance.

    The ice of soil level ``j`` is read at ``h2osoi_ice_col[c, j]``, which holds level
    ``j + 1`` (the legacy translation's indexing, kept unchanged in this move: Phase 7
    decision C14, open issue 19 in ``global_input.md``).
    """
    from clm_src_main import clm_varpar

    site = site_or_modules(site)
    lev = slice(1, clm_varpar.nlevsoi + 1)
    inst = mlcanopy_inst
    psis, rsoil, et_loss = inst.psis_soil, inst.rsoil_soil, inst.soil_et_loss_soil
    for fp in range(num_filter):
        p = filter_patch[fp]
        c = patch.column[p]
        r = physics.soil_resistance(
            soilstate_inst.hk_l_col[c, lev],
            soilstate_inst.smp_l_col[c, lev],
            site.dz_col[c, lev],
            soilstate_inst.rootfr_patch[p, lev],
            waterstatebulk_inst.h2osoi_ice_col[c, lev],
            site.nbedrock_col[c],
            inst.lai_canopy[p],
            inst.root_biomass_canopy[p],
            params.root_radius,
            params.root_density,
            params.root_resistivity,
        )
        psis = psis.at[p].set(r.psis)
        rsoil = rsoil.at[p].set(r.rsoil)
        et_loss = et_loss.at[p, lev].set(r.soil_et_loss)
    return inst._replace(psis_soil=psis, rsoil_soil=rsoil, soil_et_loss_soil=et_loss)


def LeafWaterPotential(
    num_filter: int, filter_patch: Sequence[int], mlcanopy_inst: mlcanopy_type, params, *, opts
) -> mlcanopy_type:
    """Leaf water potential of the sunlit and shaded leaves of each layer, in one call
    (``physics.leaf_water_potential``; the legacy routine took one leaf per call).

    Fortran reference: MLPlantHydraulicsMod.F90::LeafWaterPotential.
    """
    inst = mlcanopy_inst
    lwp = inst.lwp_leaf
    for fp in range(num_filter):
        p = filter_patch[fp]
        new = physics.leaf_water_potential(
            inst.dpai_profile[p, 1:],
            inst.zs_profile[p, 1:],
            inst.lsc_profile[p, 1:],
            inst.trleaf_leaf[p, 1:, _LEAVES],
            inst.lwp_bef_leaf[p, 1:, _LEAVES],
            inst.psis_soil[p],
            params.plant_capacitance,
            opts.dtime_ml,
        )
        lwp = lwp.at[p, 1:, _LEAVES].set(new)
    return inst._replace(lwp_leaf=lwp)


def LeafHeatCapacity(
    num_filter: int, filter_patch: Sequence[int], mlcanopy_inst: mlcanopy_type, params
) -> mlcanopy_type:
    """Leaf heat capacity of each layer (``clm_ml.physics.leaf_heat_capacity``).

    Fortran reference: MLLeafHeatCapacityMod.F90::LeafHeatCapacity.
    """
    inst = mlcanopy_inst
    cpleaf = inst.cpleaf_profile
    for fp in range(num_filter):
        p = filter_patch[fp]
        cpleaf = cpleaf.at[p, 1:].set(
            heat.leaf_heat_capacity(inst.dpai_profile[p, 1:], params.sla_top)
        )
    return inst._replace(cpleaf_profile=cpleaf)
