"""The leaf photosynthesis routine of the legacy time step as an adapter to
``clm_ml.physics.photosynthesis`` (Phase 7.2, process 7; transition only: it goes when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (index 0 the ground), leaves
``isun = 1``/``isha = 2`` (index 0 unused). The results are written in the layers in use,
the padding keeps its values (as before). ``DECLARED`` lists the container fields the adapter
reads and writes (checked by ``tests/clm_ml/test_physics_photosynthesis.py``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax

from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import photosynthesis as physics
from clm_ml.physics.layers import keep
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun

_ML = "mlcanopy."
#: ``physics.Photosynthesis`` field -> container field
_LEAF_FIELDS = {f: f + "_leaf" for f in physics.Photosynthesis._fields}
_READS = ("dpai_profile", "eair_profile", "cair_profile", "tleaf_leaf", "vcmax25_leaf",
          "jmax25_leaf", "rd25_leaf", "kp25_leaf", "apar_leaf", "gbc_leaf", "gbv_leaf",
          "lwp_leaf", "o2ref_forcing", "pref_forcing", "tacclim_forcing", "ncan_canopy",
          "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile", "zw_profile")  # fmt: skip
#: adapter -> (fields read, fields written), as ``container.field``; ``g0_canopy`` and
#: ``g1_canopy`` are written only by the Medlyn and Ball-Berry models
DECLARED = {
    "LeafPhotosynthesis": (
        tuple(_ML + f for f in (*_READS, *_LEAF_FIELDS.values())),
        tuple(_ML + f for f in (*_LEAF_FIELDS.values(), "btran_soil", "g0_canopy",
                                "g1_canopy")),
    )
}  # fmt: skip
#: Fortran codes -> names (``gs_type``, ``gs_solver``)
_STOMATAL = {0: "medlyn", 1: "ball_berry", 2: "wue"}
_WUE_SOLVER = {1: "brent", 2: "bisection"}

# One program per call, as the legacy kernels were (step-by-step runs)
_leaf_photosynthesis = jax.jit(
    physics.leaf_photosynthesis,
    static_argnames=("c3", "stomatal_conductance", "water_stress", "colimitation",
                     "acclimation", "wue_solver"),
)  # fmt: skip


def LeafPhotosynthesis(
    num_filter: int,
    filter_patch: Sequence[int],
    il: Optional[int],
    mlcanopy_inst: mlcanopy_type,
    params,
    grid: GridInfo,
    *,
    opts: MLOptions,
) -> mlcanopy_type:
    """Photosynthesis and stomatal conductance of patch ``grid.p``
    (``physics.leaf_photosynthesis``): of the sunlit and shaded leaves in one call with
    ``il = None``, or of one leaf (``isun``, ``isha``; Fortran calls it once per leaf).

    Fortran reference: MLLeafPhotosynthesisMod.F90::LeafPhotosynthesis.
    """
    for name, code, valid in (("gs_type", opts.gs_type, _STOMATAL),
                              ("gs_solver", opts.gs_solver, _WUE_SOLVER),
                              ("gspot_type", opts.gspot_type, (0, 1)),
                              ("colim_type", opts.colim_type, (0, 1)),
                              ("acclim_type", opts.acclim_type, (0, 1))):  # fmt: skip
        if code not in valid:
            raise ValueError(f"LeafPhotosynthesis: {name} {code} not one of {tuple(valid)}")
    inst = mlcanopy_inst
    p = grid.p
    leaves = slice(isun, isha + 1) if il is None else slice(il, il + 1)
    lay = mlcanopy_layers(inst, p)

    def _leaf(field):
        return getattr(inst, field)[p, 1:, leaves]

    model = _STOMATAL[opts.gs_type]
    stomatal = {}
    if model != "wue":
        stomatal = dict(zip(("g0", "g1"), physics.stomatal_parameters(params, model)))
    r = _leaf_photosynthesis(
        lay,
        inst.dpai_profile[p, 1:],
        inst.eair_profile[p, 1:],
        inst.cair_profile[p, 1:],
        *(_leaf(f) for f in ("tleaf_leaf", "vcmax25_leaf", "jmax25_leaf", "rd25_leaf",
                             "kp25_leaf", "apar_leaf", "gbc_leaf", "gbv_leaf", "lwp_leaf")),
        inst.o2ref_forcing[p],
        inst.pref_forcing[p],
        inst.tacclim_forcing[p],
        iota=params.iota,
        gs_min=params.gs_min,
        psi50_gs=params.psi50_gs,
        shape_gs=params.shape_gs,
        **stomatal,
        c3=bool(grid.c3),
        stomatal_conductance=model,
        water_stress=bool(opts.gspot_type == 1),
        colimitation=bool(opts.colim_type == 1),
        acclimation=bool(opts.acclim_type == 1),
        wue_solver=_WUE_SOLVER[opts.gs_solver],
    )  # fmt: skip

    def _write(field, new):
        arr = getattr(inst, field)
        return arr.at[p, 1:, leaves].set(keep(lay.in_use, new, arr[p, 1:, leaves]))

    updates = {_LEAF_FIELDS[f]: _write(_LEAF_FIELDS[f], v) for f, v in zip(r._fields, r)}
    updates["btran_soil"] = inst.btran_soil.at[p].set(1.0)
    if model != "wue":
        updates["g0_canopy"] = inst.g0_canopy.at[p].set(stomatal["g0"])
        updates["g1_canopy"] = inst.g1_canopy.at[p].set(stomatal["g1"])
    return inst._replace(**updates)
