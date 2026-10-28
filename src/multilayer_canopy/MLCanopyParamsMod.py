"""
Per-patch canopy parameters, passed explicitly to the physics.

Plan Phase 3d. Before, each routine read its parameters from the global PFT tables
(``MLpftconMod.MLpftcon`` and ``pftconMod.pftcon``) as ``table[pft]``. The routines now take a
:class:`CanopyParams`, the values for one patch's plant functional type. It is a pytree of JAX
scalars, so ``jax.grad`` can differentiate with respect to any parameter, and ``vmap`` can run
patches with different PFTs.

Field names are the ``params.yaml`` names (``clm_ml.params.PARAM_SPECS``). Two table entries are
not here: ``c3_photosynthesis`` selects a code path (C3 or C4), so it is static and lives in
:class:`MLclm_varctl.GridInfo`; ``root_profile_beta`` is used only at initialization.

Fortran reference: MLpftconMod.F90 and pftconMod.F90 (the parameter tables).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array


class CanopyParams(NamedTuple):
    """Parameters of one patch's PFT (JAX scalars; spectral fields are ``[unused, vis, nir]``,
    indexed by the 1-based waveband as in the tables)."""

    # Photosynthesis and stomatal conductance
    vcmax25: Array
    iota: Array
    gs_min: Array
    g0_ball_berry: Array
    g1_ball_berry: Array
    g0_medlyn: Array
    g1_medlyn: Array
    psi50_gs: Array
    shape_gs: Array
    c3_photosynthesis: Array  # 1 for C3, 0 for C4 (data: both pathways are computed)
    # Plant hydraulics and roots
    plant_conductance: Array
    plant_capacitance: Array
    root_radius: Array
    root_density: Array
    root_resistivity: Array
    # Leaf structure and radiation
    leaf_dimension: Array
    leaf_angle_index: Array
    sla_top: Array
    leaf_emissivity: Array
    clumping_index: Array
    leaf_reflectance: Array
    leaf_transmittance: Array
    stem_reflectance: Array
    stem_transmittance: Array


# CanopyParams field -> (legacy table, legacy field)
LEGACY_FIELDS: dict[str, tuple[str, str]] = {
    "vcmax25": ("MLpftcon", "vcmaxpft"),
    "iota": ("MLpftcon", "iota_SPA"),
    "gs_min": ("MLpftcon", "gsmin_SPA"),
    "g0_ball_berry": ("MLpftcon", "g0_BB"),
    "g1_ball_berry": ("MLpftcon", "g1_BB"),
    "g0_medlyn": ("MLpftcon", "g0_MED"),
    "g1_medlyn": ("MLpftcon", "g1_MED"),
    "c3_photosynthesis": ("pftcon", "c3psn"),
    "psi50_gs": ("MLpftcon", "psi50_gs"),
    "shape_gs": ("MLpftcon", "shape_gs"),
    "plant_conductance": ("MLpftcon", "gplant_SPA"),
    "plant_capacitance": ("MLpftcon", "capac_SPA"),
    "root_radius": ("MLpftcon", "root_radius_SPA"),
    "root_density": ("MLpftcon", "root_density_SPA"),
    "root_resistivity": ("MLpftcon", "root_resist_SPA"),
    "leaf_dimension": ("pftcon", "dleaf"),
    "leaf_angle_index": ("pftcon", "xl"),
    "sla_top": ("pftcon", "slatop"),
    "leaf_emissivity": ("MLpftcon", "emleaf"),
    "clumping_index": ("MLpftcon", "clump_fac"),
    "leaf_reflectance": ("pftcon", "rhol"),
    "leaf_transmittance": ("pftcon", "taul"),
    "stem_reflectance": ("pftcon", "rhos"),
    "stem_transmittance": ("pftcon", "taus"),
}


def canopy_params_from_tables(pft: int) -> CanopyParams:
    """:class:`CanopyParams` for PFT ``pft`` from the legacy global tables (filled by
    ``MLpftcon.Init`` and ``pftcon.Init``). Phase 3e builds them from ``clm_ml.Params``."""
    from clm_src_main.pftconMod import pftcon
    from multilayer_canopy.MLpftconMod import MLpftcon

    tables = {"MLpftcon": MLpftcon, "pftcon": pftcon}
    return CanopyParams(
        **{
            name: jnp.asarray(getattr(tables[table], field)[pft], dtype=jnp.float64)
            for name, (table, field) in LEGACY_FIELDS.items()
        }
    )


def patch_setup(p: int, mlcanopy_inst) -> tuple:
    """``(GridInfo, CanopyParams)`` for patch ``p``: its index (static) and its PFT's
    parameters, from the patch's PFT (``patch.itype``). Layer counts are data in the
    container (``MLLayersMod``), the C3/C4 flag in the parameters."""
    from clm_src_main.PatchType import patch
    from multilayer_canopy.MLclm_varctl import GridInfo

    # Structure (NumPy indexing, so this works while a step is being traced)
    pft = int(np.asarray(patch.itype)[p])
    grid = GridInfo(p=p)
    return grid, canopy_params_from_tables(pft)
