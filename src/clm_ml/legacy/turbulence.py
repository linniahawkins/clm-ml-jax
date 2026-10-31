"""The canopy turbulence routine of the legacy time step as an adapter to
``clm_ml.physics.turbulence`` (Phase 7.2, process 6; transition only: it goes when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (level 0 the ground). The profiles are
written in the layers in use, the padding keeps its values (as before). ``DECLARED`` lists the
container fields the adapter reads and writes (checked by
``tests/clm_ml/test_physics_turbulence.py``).

The psihat lookup tables are the package's (``clm_ml.io.rsl_tables.default_rsl_tables``, read
once per process): an argument of the physics, a constant of the compiled step until the
step takes them as an input (Phase 7 decision C15).
"""

from __future__ import annotations

from typing import Sequence

import jax

from clm_ml.io.rsl_tables import default_rsl_tables
from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import turbulence as physics
from clm_ml.physics.constants import eta_max
from clm_ml.physics.layers import keep
from clm_ml.physics.math import is_concrete
from clm_src_main.abortutils import endrun
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions

_ML = "mlcanopy."
_SCALARS = ("Lc_canopy", "taf_canopy", "qaf_canopy", "obu_canopy", "beta_canopy",
            "zdisp_canopy", "PrSc_canopy", "ustar_canopy", "gac_to_hc_canopy", "z0m_canopy",
            "uaf_canopy", "gac0_soil")  # fmt: skip
_PROFILES = ("wind_profile", "mflx_profile", "gac_profile", "kc_eddy_profile")
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "CanopyTurbulence": (
        tuple(_ML + f for f in (
            "tair_profile", "eair_profile", "pref_forcing", "ztop_canopy", "lai_canopy",
            "sai_canopy", "zref_forcing", "uref_forcing", "thref_forcing", "thvref_forcing",
            "qref_forcing", "rhomol_forcing", "ncan_canopy", "ntop_canopy", "nbot_canopy",
            "dz_profile", "zs_profile", "zw_profile", *_PROFILES)),
        tuple(_ML + f for f in (*_SCALARS, *_PROFILES)),
    )
}  # fmt: skip
#: Fortran ``HF_extension_type`` -> ``physics.GROUND_CONDUCTANCE``
_GROUND = {1: "extend_hf", 2: "log_profile"}


# Compiled as one program, as the legacy routine was (run operation by operation, it would
# dispatch thousands of tiny operations per call in step-by-step runs)
_harman_finnigan = jax.jit(
    physics.harman_finnigan, static_argnames=("sparse_canopy", "ground_conductance")
)


def CanopyTurbulence(
    nstep_ml: int,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: GridInfo,
    nstep: int = 0,
    *,
    opts: MLOptions,
) -> mlcanopy_type:
    """Canopy turbulence of patch ``grid.p`` (``physics.harman_finnigan``, Fortran
    ``turb_type = 1``, the only scheme); ``nstep``, ``nstep_ml`` only for the Fortran
    warning when ``eta >= eta_max`` (printed in step-by-step runs).

    Fortran reference: MLCanopyTurbulenceMod.F90::CanopyTurbulence.
    """
    if opts.turb_type != 1:
        endrun(msg=" ERROR: CanopyTurbulence: turb_type not valid")
    if opts.HF_extension_type not in _GROUND:
        # Fortran leaves gac0 unchanged; the configuration accepts only 1 and 2 (C15)
        raise ValueError(f"HF_extension_type {opts.HF_extension_type} not one of 1, 2")
    inst = mlcanopy_inst
    p = grid.p
    lay = mlcanopy_layers(inst, p)
    t = _harman_finnigan(
        lay,
        inst.tair_profile[p, 1:],
        inst.eair_profile[p, 1:],
        inst.pref_forcing[p],
        inst.ztop_canopy[p],
        inst.lai_canopy[p],
        inst.sai_canopy[p],
        inst.zref_forcing[p],
        inst.uref_forcing[p],
        inst.thref_forcing[p],
        inst.thvref_forcing[p],
        inst.qref_forcing[p],
        inst.rhomol_forcing[p],
        default_rsl_tables(),
        sparse_canopy=bool(opts.sparse_canopy_type == 1),
        ground_conductance=_GROUND[opts.HF_extension_type],
    )
    if is_concrete(t.eta) and t.eta >= eta_max:
        print(" Warning: HF2008: lm/beta error")
        print(f" nstep = {nstep}  nstep_ml = {nstep_ml}")
        print(f" eta = {t.eta}")

    def _profile(arr, new):
        return arr.at[p, 1:].set(keep(lay.in_use, new, arr[p, 1:]))

    scalars = dict(zip(_SCALARS, (
        t.Lc, t.taf, t.qaf, t.obu, t.beta, t.zdisp, t.PrSc, t.ustar, t.gac_to_hc, t.z0m,
        t.uaf, t.gac0)))  # fmt: skip
    profiles = dict(zip(_PROFILES, (t.wind, t.mflx, t.gac, t.kc_eddy)))
    return inst._replace(
        **{f: getattr(inst, f).at[p].set(v) for f, v in scalars.items()},
        **{f: _profile(getattr(inst, f), v) for f, v in profiles.items()},
    )
