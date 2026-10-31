"""The leaf boundary layer routine of the legacy time step as an adapter to
``clm_ml.physics.leaf_boundary_layer`` (Phase 7.2, process 7; transition only: it goes when
the containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (index 0 the ground), leaves
``isun = 1``/``isha = 2`` (index 0 unused). Every layer slot is written, as before.
``DECLARED`` lists the container fields the adapter reads and writes (checked by
``tests/clm_ml/test_physics_photosynthesis.py``).
"""

from __future__ import annotations

from typing import Sequence

import jax

from clm_ml.physics import leaf_boundary_layer as physics
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
from multilayer_canopy.MLclm_varctl import MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun

_ML = "mlcanopy."
_WRITES = ("gbh_leaf", "gbv_leaf", "gbc_leaf")
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "LeafBoundaryLayer": (
        tuple(_ML + f for f in ("dpai_profile", "wind_profile", "tair_profile", "tleaf_leaf",
                                "pref_forcing", "tref_forcing", "rhomol_forcing", *_WRITES)),
        tuple(_ML + f for f in _WRITES),
    )
}  # fmt: skip
#: Fortran ``gb_type`` -> ``physics.REGIMES``
_REGIME = {1: "laminar", 2: "turbulent", 3: "free_convection"}
_LEAVES = slice(isun, isha + 1)  # sunlit, shaded

# One program per call, as the legacy routine was (step-by-step runs)
_leaf_boundary_layer = jax.jit(physics.leaf_boundary_layer, static_argnames=("regime",))


def LeafBoundaryLayer(
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    params,
    *,
    opts: MLOptions,
) -> mlcanopy_type:
    """Leaf boundary layer conductances of the sunlit and shaded leaves of each patch, in one
    call (``physics.leaf_boundary_layer``; Fortran calls it once per leaf).

    Fortran reference: MLLeafBoundaryLayerMod.F90::LeafBoundaryLayer.
    """
    if opts.gb_type not in _REGIME:
        raise ValueError(f"LeafBoundaryLayer: gb_type {opts.gb_type} not one of 1, 2, 3")
    inst = mlcanopy_inst
    out = {f: getattr(inst, f) for f in _WRITES}
    for fp in range(num_filter):
        p = filter_patch[fp]
        g = _leaf_boundary_layer(
            inst.dpai_profile[p, 1:],
            inst.wind_profile[p, 1:],
            inst.tair_profile[p, 1:],
            inst.tleaf_leaf[p, 1:, _LEAVES],
            inst.pref_forcing[p],
            inst.tref_forcing[p],
            inst.rhomol_forcing[p],
            params.leaf_dimension,
            regime=_REGIME[opts.gb_type],
        )
        for f, v in zip(_WRITES, g):
            out[f] = out[f].at[p, 1:, _LEAVES].set(v)
    return inst._replace(**out)
