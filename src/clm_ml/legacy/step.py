"""The legacy run's inputs of :func:`clm_ml.step.step` (Phase 7.2, process 10c; transition
only: the constants come from the setup itself in Phase 7.4; the step inputs come from the
chunk's 0-based arrays since Phase 7.3, ``clm_ml.chunk_inputs``).

- :func:`patch_constants`: the patch's :class:`clm_ml.step.PatchConstants` from its
  containers after the first step (the *template*; legacy layout: a patch axis, layer slots
  1..max_layers, leaves and bands from index 1, soil level ``j`` at index ``j``).
- :func:`step_options`: the step's static choices from the legacy options (``MLOptions``).
"""

from __future__ import annotations

import jax.numpy as jnp

from clm_ml.legacy.canopy import substep_options
from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.step import PatchConstants, StepOptions


def step_options(opts, step_seconds: int) -> StepOptions:
    """The step's choices from the legacy options ``opts`` (``MLOptions``) and the model step
    (s). The coupling of the canopy's fluxes to CLM (``mlcan_to_clm``) is not supported (it
    wrote CLM containers that nothing reads offline)."""
    from clm_src_main import clm_varpar
    from clm_src_utils import clm_varorb
    from offline_driver import clmSoilOptionMod

    if opts.mlcan_to_clm != 0:
        raise NotImplementedError("mlcan_to_clm = 1 (fluxes passed to CLM) is not supported")
    sub = substep_options(opts)
    n_soil, n_levels = clm_varpar.nlevsoi, clm_varpar.nlevgrnd
    if not n_soil < n_levels:  # the ice of soil level j is read at level j + 1 (issue 19)
        raise NotImplementedError(f"nlevsoi ({n_soil}) must be less than nlevgrnd ({n_levels})")
    return StepOptions(
        substep=sub,
        step_seconds=int(step_seconds),
        layer_pai_floor=float(opts.layer_pai_floor),
        flux_profile_type=int(opts.flux_profile_type),
        clm_phys=str(clmSoilOptionMod.clm_phys),
        n_soil_adjust=int(clmSoilOptionMod.nlev_soil_adjust),
        n_soil=int(n_soil),
        eccen=float(clm_varorb.eccen),
        mvelpp=float(clm_varorb.mvelpp),
        lambm0=float(clm_varorb.lambm0),
        obliqr=float(clm_varorb.obliqr),
    )


def patch_constants(template, p: int = 0) -> PatchConstants:
    """Patch ``p``'s constants from its containers after the first step (``ClmInstances``;
    one patch per soil column, the column index of patch ``p``)."""
    import numpy as np

    from clm_src_biogeophys import SurfaceAlbedoMod
    from clm_src_main import clm_varpar
    from clm_src_main.PatchType import patch

    c = int(np.asarray(patch.column)[p])  # structure: static
    g = int(np.asarray(patch.gridcell)[p])
    ml, s, site = template.mlcanopy, template.soilstate, template.site
    n, n_soil = clm_varpar.nlevgrnd, clm_varpar.nlevsoi
    lev, lev_soil = slice(1, n + 1), slice(1, n_soil + 1)
    colour = site.isoicol_col[c] - 1  # 1-based class into the tables (data)
    bands = slice(1, clm_varpar.numrad + 1)
    return PatchConstants(
        layers=mlcanopy_layers(ml, p),
        dlai_frac=ml.dlai_frac_profile[p, 1:],
        dsai_frac=ml.dsai_frac_profile[p, 1:],
        zref=ml.zref_forcing[p],
        ztop=ml.ztop_canopy[p],
        root_biomass=ml.root_biomass_canopy[p],
        rootfr=s.rootfr_patch[p, lev_soil],
        dz=site.dz_col[c, lev],
        z=site.z_col[c, lev],
        zi=site.zi_col[c, lev],
        soil_dz=site.z_col[c, 1] - site.zi_col[c, 0],  # no snow layers (snl = 0)
        nbedrock=site.nbedrock_col[c],
        watsat=s.watsat_col[c, lev],
        sucsat=s.sucsat_col[c, lev],
        bsw=s.bsw_col[c, lev],
        hksat=s.hksat_col[c, lev],
        tkmg=s.tkmg_col[c, lev],
        tkdry=s.tkdry_col[c, lev],
        csol=s.csol_col[c, lev],
        albsat=jnp.asarray(SurfaceAlbedoMod.albsat)[colour, bands],
        albdry=jnp.asarray(SurfaceAlbedoMod.albdry)[colour, bands],
        latdeg=site.latdeg_grc[g],
        londeg=site.londeg_grc[g],
    )
