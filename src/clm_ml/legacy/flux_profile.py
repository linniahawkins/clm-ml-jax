"""The flux-profile solution of the legacy time step as an adapter to
``clm_ml.physics.flux_profile`` (Phase 7.2, process 8; transition only: it goes when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (index 0 unused), leaves
``isun = 1``/``isha = 2`` (index 0 unused). Results are written in the layers in use, the
padding keeps its values (as before). ``DECLARED`` lists the container fields the adapter
reads and writes (checked by ``tests/clm_ml/test_physics_flux_profile.py``).

The well-mixed variant (``flux_profile_type`` 0 or -1) is registered but not supported
(Phase 7 decision D8): its legacy translation is kept here unchanged except that its leaf and
soil fluxes come from the physics (the same arithmetic as the legacy ``LeafFluxes`` and
``SoilFluxes``), and the run stops after it, as Fortran does.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import flux_profile as physics
from clm_src_main.abortutils import endrun
from multilayer_canopy.MLCanopyFluxesType import (
    LEAF_ENERGY,
    SOIL_ENERGY,
    mlcanopy_type,
    record_balance_error,
)
from multilayer_canopy.MLclm_varctl import MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun

_ML = "mlcanopy."
_LEAF_WRITES = ("tleaf_leaf", "stleaf_leaf", "shleaf_leaf", "lhleaf_leaf", "evleaf_leaf",
                "trleaf_leaf")  # fmt: skip
_SOIL_WRITES = ("shsoi_soil", "lhsoi_soil", "gsoi_soil", "etsoi_soil", "tg_soil", "eg_soil")
_WRITES = ("tair_profile", "eair_profile", "cair_profile", "shair_profile", "etair_profile",
           "stair_profile", *_LEAF_WRITES, *_SOIL_WRITES, "balance_error")  # fmt: skip
_REFERENCE = ("pref_forcing", "rhomol_forcing", "cpair_forcing", "tref_forcing",
              "thref_forcing", "eref_forcing")  # fmt: skip
_GROUND = ("tg_bef_soil", "rhg_soil", "rnsoi_soil", "soilres_soil", "soil_t_soil",
           "soil_dz_soil", "soil_tk_soil", "gac0_soil")  # fmt: skip
_CANOPY = ("dpai_profile", "fwet_profile", "fdry_profile", "fracsun_profile", "cpleaf_profile",
           "gac_profile", "tair_bef_profile", "eair_bef_profile")  # fmt: skip
_CANOPY_LEAF = ("gbh_leaf", "gbv_leaf", "gs_leaf", "rnleaf_leaf", "tleaf_bef_leaf")
_LAYERS = ("ncan_canopy", "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile",
           "zw_profile")  # fmt: skip
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "FluxProfileSolution": (
        tuple(_ML + f for f in ("co2ref_forcing", *_REFERENCE, *_GROUND, *_CANOPY,
                                *_CANOPY_LEAF, *_LAYERS, *_WRITES)),
        tuple(_ML + f for f in _WRITES),
    )
}  # fmt: skip
_LEAVES = slice(isun, isha + 1)  # sunlit, shaded


def _reference(inst, p) -> physics.Reference:
    return physics.Reference(*(getattr(inst, f)[p] for f in _REFERENCE))


def _ground(inst, p) -> physics.Ground:
    return physics.Ground(*(getattr(inst, f)[p] for f in _GROUND))


def FluxProfileSolution(
    num_filter: int, filter: Array, mlcanopy_inst: mlcanopy_type, grid, *, opts: MLOptions
) -> mlcanopy_type:
    """Leaf and soil fluxes and the air temperature and vapor pressure profiles
    (``physics.implicit``, Fortran ``flux_profile_type = 1``); no CO2 profile: ``cair`` is
    the reference value.

    Fortran reference: MLFluxProfileSolutionMod.F90::FluxProfileSolution. ``cair`` is set
    in levels 0..ncan-1, not Fortran's layers 1..ncan, as the legacy translation did (no
    effect while ``cair`` is constant; numerics doc §4.5).
    """
    if opts.flux_profile_type in (0, -1):
        mlcanopy_inst = WellMixed(num_filter, filter, mlcanopy_inst, opts=opts)
        endrun(msg=" ERROR: WellMixed option not valid")
    elif opts.flux_profile_type != 1:
        endrun(msg=" ERROR: FluxProfileSolution: flux_profile_type not valid")
    inst = mlcanopy_inst
    for _ in range(num_filter):
        p = grid.p
        inst = _implicit(p, opts.dtime_ml, inst)
        cair = inst.cair_profile
        below_ncan = jnp.arange(cair.shape[1]) < inst.ncan_canopy[p]
        cair = cair.at[p].set(jnp.where(below_ncan, inst.co2ref_forcing[p], cair[p]))
        inst = inst._replace(cair_profile=cair)
    return inst


@partial(jax.jit, static_argnums=(0, 1))
def _implicit(p: int, dtime: float, inst: mlcanopy_type) -> mlcanopy_type:
    """``physics.implicit`` on patch ``p``, written into the container.

    Fortran reference: MLFluxProfileSolutionMod.F90::ImplicitFluxProfileSolution.
    """
    lay = mlcanopy_layers(inst, p)
    use = lay.in_use
    canopy = physics.Canopy(
        *(getattr(inst, f)[p, 1:] for f in _CANOPY),
        *(getattr(inst, f)[p, 1:, _LEAVES] for f in _CANOPY_LEAF),
    )
    f = physics.implicit(canopy, _ground(inst, p), _reference(inst, p), lay, dtime)

    def _profile(arr, new):
        return arr.at[p, 1:].set(jnp.where(use, new, arr[p, 1:]))

    def _leaf(arr, new):
        # leaf index 0 (unused) set to 0 in the layers in use, as the legacy code wrote it
        new = jnp.concatenate([jnp.zeros_like(new[:, :1]), new], axis=1)
        return arr.at[p, 1:].set(jnp.where(use[:, None], new, arr[p, 1:]))

    inst = record_balance_error(inst, p, SOIL_ENERGY, f.soil.error)
    inst = _write_soil(inst, p, f.soil)
    inst = inst._replace(
        tair_profile=_profile(inst.tair_profile, f.tair),
        eair_profile=_profile(inst.eair_profile, f.eair),
        shair_profile=_profile(inst.shair_profile, f.shair),
        etair_profile=_profile(inst.etair_profile, f.etair),
        stair_profile=_profile(inst.stair_profile, f.stair),
        **{name: _leaf(getattr(inst, name), getattr(f.leaf, name.split("_")[0]))
           for name in _LEAF_WRITES},
    )  # fmt: skip
    return record_balance_error(inst, p, LEAF_ENERGY, f.leaf.error)


def _write_soil(inst, p, s: physics.SoilFluxes):
    return inst._replace(
        **{name: getattr(inst, name).at[p].set(getattr(s, name.split("_")[0]))
           for name in _SOIL_WRITES}
    )  # fmt: skip


def WellMixed(
    num_filter: int, filter: Array, mlcanopy_inst: mlcanopy_type, *, opts: MLOptions
) -> mlcanopy_type:
    """Canopy profiles at the reference-height values (well-mixed, ``flux_profile_type = 0``)
    or from a dataset (``-1``): registered but not supported (D8; ``physics.well_mixed``).
    The legacy translation, unchanged except that the leaf and soil fluxes come from
    ``physics.leaf_fluxes`` and ``physics.soil_fluxes``; never validated (Fortran stops after
    it, and so does :func:`FluxProfileSolution`).

    Fortran reference: MLFluxProfileSolutionMod.F90::WellMixed.
    """
    inst = mlcanopy_inst
    for fp in range(num_filter):
        p = int(filter[fp])
        n = int(inst.ncan_canopy[p])
        cair, wind, tair, eair = (inst.cair_profile, inst.wind_profile, inst.tair_profile,
                                  inst.eair_profile)  # fmt: skip
        shair, etair, stair, mflx = (inst.shair_profile, inst.etair_profile,
                                     inst.stair_profile, inst.mflx_profile)  # fmt: skip
        for ic in range(n):  # levels 0..ncan-1, as translated
            cair = cair.at[p, ic].set(inst.co2ref_forcing[p])
            if opts.flux_profile_type == 0:
                wind = wind.at[p, ic].set(inst.uref_forcing[p])
                tair = tair.at[p, ic].set(inst.tref_forcing[p])
                eair = eair.at[p, ic].set(inst.eref_forcing[p])
            elif opts.flux_profile_type == -1:
                wind = wind.at[p, ic].set(inst.wind_data_profile[p, ic])
                tair = tair.at[p, ic].set(inst.tair_data_profile[p, ic])
                eair = eair.at[p, ic].set(inst.eair_data_profile[p, ic])
            shair, etair = shair.at[p, ic].set(0.0), etair.at[p, ic].set(0.0)
            stair, mflx = stair.at[p, ic].set(0.0), mflx.at[p, ic].set(0.0)
        inst = inst._replace(
            wind_profile=wind, tair_profile=tair, eair_profile=eair, cair_profile=cair,
            shair_profile=shair, etair_profile=etair, stair_profile=stair, mflx_profile=mflx,
        )  # fmt: skip

        # Leaf fluxes of levels 0..ncan-1 (sunlit and shaded)
        lv = slice(0, n)
        lf = physics.leaf_fluxes(
            inst.tair_profile[p, lv, None], inst.eair_profile[p, lv, None],
            inst.cpleaf_profile[p, lv, None], inst.fwet_profile[p, lv, None],
            inst.fdry_profile[p, lv, None], inst.gbh_leaf[p, lv, _LEAVES],
            inst.gbv_leaf[p, lv, _LEAVES], inst.gs_leaf[p, lv, _LEAVES],
            inst.rnleaf_leaf[p, lv, _LEAVES], inst.tleaf_bef_leaf[p, lv, _LEAVES],
            inst.dpai_profile[p, lv, None], inst.pref_forcing[p], inst.cpair_forcing[p],
            inst.tref_forcing[p], opts.dtime_ml,
        )  # fmt: skip
        inst = record_balance_error(inst, p, LEAF_ENERGY, lf.error)
        inst = inst._replace(
            **{name: getattr(inst, name).at[p, lv, _LEAVES].set(getattr(lf, name.split("_")[0]))
               for name in _LEAF_WRITES}
        )  # fmt: skip

        # Soil conductance: large resistance -> small soil fluxes
        inst = inst._replace(gac0_soil=inst.gac0_soil.at[p].set((1.0 / 100.0) * 42.3))
        s = physics.soil_fluxes(
            inst.tair_profile[p, 1], inst.eair_profile[p, 1], _ground(inst, p),
            _reference(inst, p),
        )  # fmt: skip
        inst = record_balance_error(inst, p, SOIL_ENERGY, s.error)
        inst = _write_soil(inst, p, s)

        # Output-only fields
        inst = inst._replace(
            uaf_canopy=inst.uaf_canopy.at[p].set(inst.uref_forcing[p]),
            taf_canopy=inst.taf_canopy.at[p].set(inst.tref_forcing[p]),
            qaf_canopy=inst.qaf_canopy.at[p].set(inst.qref_forcing[p]),
            gac_profile=inst.gac_profile.at[p, :n].set((1.0 / 10.0) * 42.3),
        )
    return inst
