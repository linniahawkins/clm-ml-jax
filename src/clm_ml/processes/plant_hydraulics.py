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

from clm_ml.processes.constants import denh2o, grav, mmh2o, rpi

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
    #: (soil_level,) water each level gives to the roots without transpiration, mmol H2O/m2
    #: ground/s (negative: the roots release water into the level); sums to 0. Zero in
    #: CLM-ml v2 (:func:`soil_resistance`).
    redistribution: Array


class RootParameters(NamedTuple):
    """The PFT's root parameters (:class:`clm_ml.params.CanopyParams` fields)."""

    root_radius: Array  # fine root radius, m
    root_density: Array  # fine root density, g biomass/m3 root
    root_resistivity: Array  # CLM-ml: resistivity of root tissue, MPa s g/mmol H2O
    root_to_leaf_ratio: Array  # CLM: fine root carbon / leaf carbon (froot_leaf)
    root_conductance_max: Array  # CLM: root conductance per root area and path length (krmax)
    root_psi50: Array  # CLM: water potential at 50 % loss of root conductance, MPa
    root_ck: Array  # CLM: shape of the root's vulnerability curve


#: The functions that give each soil level's soil-to-root conductance (model.yaml
#: ``soil.root_conductance``)
ROOT_CONDUCTANCE = ("clm_ml", "clm")
#: Where the fine root biomass comes from (model.yaml ``soil.root_biomass``)
ROOT_BIOMASS = ("prescribed", "from_leaf_area")
#: Biomass per carbon, g/g (CLM ``c_to_b``)
C_TO_B = 2.0
#: Length of the lateral coarse roots, m (CLM ``croot_lateral_length``)
CROOT_LATERAL_LENGTH = 0.25
#: mm H2O/s per mm of water potential -> mmol H2O/m2/s/MPa
PER_SECOND_TO_MMOL = 1.0e6 / (mmh2o * HEAD)


def fine_root_biomass(lai, sla_top, root_to_leaf_ratio):
    """Fine root biomass (g/m2) from the leaf area, as CLM without its carbon cycle: leaf
    carbon ``lai / sla_top`` (``sla_top`` in m2/gC) times the root-to-leaf ratio, times
    :data:`C_TO_B`.

    Fortran reference: CNVegetationFacade.F90::get_froot_carbon_patch (CTSM 5.4).
    """
    carbon = jnp.where(sla_top > 0.0, lai / jnp.maximum(sla_top, 1.0e-30), 0.0)
    return C_TO_B * carbon * root_to_leaf_ratio


def soil_root_resistance(
    hk: Array, dz: Array, rootfr: Array, root_biomass, root_radius, root_density, root_resistivity
) -> Array:
    """CLM-ml's resistance of each soil level between the soil and the stem, MPa s m2
    ground/mmol H2O (soil_level,): through the soil to the root (Bonan et al. 2014, eq.
    A23) plus through the root (A24). ``hk`` (mm/s), ``dz`` (m), root fraction ``rootfr``,
    fine ``root_biomass`` (g/m2).

    Fortran reference: MLPlantHydraulicsMod.F90::SoilResistance.
    """
    root_cross_sec_area = rpi * root_radius * root_radius

    # Hydraulic conductivity (mm/s -> mmol/m/s/MPa)
    hk_v = hk * (1.0e-3 / HEAD) * denh2o / mmh2o * 1000.0

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
    return jnp.maximum(soilr1 + soilr2, 1.0e-30)


def vulnerability(psi, psi50, ck):
    """Fraction of the largest conductance left at water potential ``psi`` (CLM's Weibull
    curve, ``2^-(psi / psi50)^ck``; 0 below 0.005). ``psi`` and ``psi50`` in the same
    units, both negative.

    Fortran reference: PhotosynthesisMod.F90::plc (CTSM 5.4).
    """
    ratio = jnp.maximum(psi / psi50, 1.0e-30)  # a positive base for the power's gradient
    left = 2.0 ** (-(ratio**ck))
    return jnp.where(left < 0.005, 0.0, left)


def soil_root_conductance_clm(
    hk: Array,
    hksat: Array,
    smp: Array,
    dz: Array,
    z: Array,
    rootfr: Array,
    root_biomass,
    lai,
    sai,
    p: RootParameters,
) -> Array:
    """CLM's conductance of each soil level between the soil and the root collar, mmol
    H2O/m2 ground/s/MPa (soil_level,): the soil's (the conductivity over the mean distance
    between roots) in series with the root's (root area index of the level times
    ``root_conductance_max``, reduced by the root's vulnerability at the level's matric
    potential, over the path length: the lateral roots plus the level's depth ``z``). The
    top level conducts nothing. ``hk``, ``hksat`` (mm/s), ``smp`` (mm), ``dz``, ``z`` (m),
    fine ``root_biomass`` (g/m2), leaf and stem area index ``lai``, ``sai``.

    Fortran reference: PhotosynthesisMod.F90::PhotosynthesisHydraulicStress (the
    root-soil interface conductance ``k_soil_root``; CTSM 5.4).
    """
    # Root biomass density (g/m3; at least 1 gC/m3) and root length density (m/m3)
    rbd = jnp.maximum(root_biomass * rootfr / dz, C_TO_B * 1.0)
    rld = rbd / (p.root_density * rpi * p.root_radius * p.root_radius)
    rai = (sai + lai) * p.root_to_leaf_ratio * rootfr  # root area index of the level

    # Soil: mm/s per mm of potential, over the distance between roots (m -> mm)
    root_dist = jnp.sqrt(1.0 / (rpi * rld))
    soil = jnp.minimum(hksat, hk) / (1.0e3 * root_dist)
    # Root: conductance per root area and path length
    left = vulnerability(smp * (1.0e-3 * HEAD), p.root_psi50, p.root_ck)
    root = left * rai * p.root_conductance_max / (CROOT_LATERAL_LENGTH + z)

    soil = jnp.maximum(soil, 1.0e-16)
    root = jnp.maximum(root, 1.0e-16)
    conductance = 1.0 / (1.0 / soil + 1.0 / root)
    below_top = jnp.arange(hk.shape[0]) > 0
    return jnp.where((rai * rootfr > 0.0) & below_top, conductance, 0.0) * PER_SECOND_TO_MMOL


def root_zone(conductance: Array, psi: Array, usable: Array, n_bedrock, lai) -> SoilResistance:
    """The soil levels as parallel conductors, each from its own water potential, joined
    at the roots: the belowground resistance per leaf area ``lai / sum(k)``, the soil
    water potential ``sum(k psi) / sum(k)`` (the roots' water potential when nothing is
    transpired), each level's fraction ``k / sum(k)`` of the transpiration and its
    ``redistribution`` ``k (psi - psis)``, by which the roots move water from wetter to
    drier levels. Their sum for a transpiration ``E`` is CLM's uptake ``k (psi -
    psi_root)`` with ``psi_root = psis - E / sum(k)``. ``conductance`` (soil_level,) in
    mmol H2O/m2 ground/s/MPa, ``psi`` in MPa; only the ``usable`` levels count. With no
    conductance at all: the minimum leaf water potential and equal fractions above bedrock.

    Fortran reference: PhotosynthesisMod.F90::getvegwp (the root water potential) and
    SoilWaterPlantSinkMod.F90::Compute_EffecRootFrac_And_VertTranSink_HydStress (the uptake
    of each level), CTSM 5.4.
    """
    cond = jnp.where(usable, conductance, 0.0)
    total = jnp.sum(cond)
    total_safe = jnp.maximum(total, 1.0e-30)
    psi_used = jnp.where(usable, psi, 0.0)
    psis = jnp.where(total > 0.0, jnp.sum(psi_used * cond) / total_safe, MINLWP_SPA)
    active = jnp.arange(cond.shape[0]) < n_bedrock
    n_layers_safe = jnp.maximum(jnp.asarray(n_bedrock).astype(jnp.float64), 1.0)
    uniform = jnp.where(active, 1.0 / n_layers_safe, 0.0)
    return SoilResistance(
        psis=psis,
        rsoil=lai / total_safe,
        soil_et_loss=jnp.where(total > 0.0, cond / total_safe, uniform),
        redistribution=cond * (psi_used - psis),
    )


def root_zone_by_conductance(
    scheme: str,
    hk: Array,
    hksat: Array,
    smp: Array,
    dz: Array,
    z: Array,
    rootfr: Array,
    h2osoi_ice: Array,
    n_bedrock,
    lai,
    sai,
    root_biomass,
    p: RootParameters,
) -> SoilResistance:
    """:func:`root_zone` of the soil levels above bedrock that hold no ice, with the
    levels' conductances by ``scheme`` (one of :data:`ROOT_CONDUCTANCE`):

    - ``"clm_ml"``: the inverse of CLM-ml's resistances (:func:`soil_root_resistance`),
      from the matric potentials;
    - ``"clm"``: CLM's (:func:`soil_root_conductance_clm`), from the matric potential
      less the gravitational potential of the level's depth, as CLM.

    Not in CLM-ml v2, which weights the levels by what each can supply
    (:func:`soil_resistance`).
    """
    if scheme not in ROOT_CONDUCTANCE:
        raise ValueError(f"root conductance {scheme!r} not one of {ROOT_CONDUCTANCE}")
    usable = (jnp.arange(hk.shape[0]) < n_bedrock) & ~(h2osoi_ice > 0.0)
    if scheme == "clm":
        conductance = soil_root_conductance_clm(
            hk, hksat, smp, dz, z, rootfr, root_biomass, lai, sai, p
        )
        psi = (smp - 1000.0 * z) * (1.0e-3 * HEAD)
    else:
        conductance = 1.0 / soil_root_resistance(
            hk, dz, rootfr, root_biomass, p.root_radius, p.root_density, p.root_resistivity
        )
        psi = smp * 1.0e-3 * HEAD
    return root_zone(conductance, psi, usable, n_bedrock, lai)


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
    smp_mpa = smp * 1.0e-3 * HEAD  # mm -> MPa
    soilr_safe = soil_root_resistance(
        hk, dz, rootfr, root_biomass, root_radius, root_density, root_resistivity
    )

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
    return SoilResistance(
        psis=psis, rsoil=rsoil, soil_et_loss=soil_et_loss, redistribution=jnp.zeros_like(evap)
    )


def root_water_uptake(soil_et_loss: Array, redistribution: Array, transpiration, redistribute):
    """Water the roots take from each soil level (soil_level,), in the units of
    ``transpiration`` and ``redistribution``: the level's fraction of the transpiration
    plus the water the roots move between the levels (:func:`root_zone`), which is
    ``k (psi - psi_root)`` with the roots' water potential under transpiration. A negative
    value is water released into the level. With ``redistribute`` false no level receives
    water: the levels that would are left out, and the others give the transpiration in
    the proportions they had. The sum is the transpiration.

    Fortran reference: SoilWaterPlantSinkMod.F90::
    Compute_EffecRootFrac_And_VertTranSink_HydStress (CTSM 5.4). Not in CLM: the uptake
    without redistribution.
    """
    shares = soil_et_loss * transpiration
    uptake = shares + redistribution
    if redistribute:
        return uptake
    giving = jnp.maximum(uptake, 0.0)
    total = jnp.sum(giving)
    return jnp.where(total > 0.0, giving * (transpiration / jnp.maximum(total, 1.0e-30)), shares)


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
