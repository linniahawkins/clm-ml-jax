"""
Per-site data read by the time step (Phase 6, step 2).

The CLM subgrid modules hold the soil column geometry (``ColumnType.col``), the gridcell
location (``GridcellType.grc``) and the soil colour class (``SurfaceAlbedoMod.isoicol``) as
module state set up once per tower. A compiled time step that read them there would bake one
site's values into the program. :class:`site_type` carries them in ``ClmInstances`` instead, so
they are part of the template containers, an input of the compiled chunk: data that differ
between patches (and are batched with ``vmap``), not constants.

Only what varies between sites is here. The snow-layer count ``snl`` (always 0 in CLM-ml) and
the patch/column/gridcell index maps are structure and stay in the subgrid modules.

Fortran reference: ColumnType.F90 (``z``, ``dz``, ``zi``, ``nbedrock``), GridcellType.F90
(``latdeg``, ``londeg``), SurfaceAlbedoMod.F90 (``isoicol``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class site_type(NamedTuple):
    """Soil column geometry, soil colour and gridcell location (same shapes and indexing as the
    subgrid modules)."""

    z_col: Array  # (endc+1, nlevsno+nlevgrnd+1) layer depth (m)
    dz_col: Array  # (endc+1, nlevsno+nlevgrnd+1) layer thickness (m)
    zi_col: Array  # (endc+1, nlevsno+nlevgrnd+1) interface depth (m)
    nbedrock_col: Array  # (endc+1,) int: index of the deepest soil layer above bedrock
    isoicol_col: Array  # (endc-begc+1,) int: soil colour class (1-based into albsat/albdry)
    latdeg_grc: Array  # (endg-begg+1,) latitude (degrees)
    londeg_grc: Array  # (endg-begg+1,) longitude (degrees)


def site_from_modules() -> site_type:
    """The current site data from the subgrid modules (after setup)."""
    from clm_src_biogeophys import SurfaceAlbedoMod
    from clm_src_main import ColumnType, GridcellType

    col, grc = ColumnType.col, GridcellType.grc
    return site_type(
        z_col=jnp.asarray(col.z, dtype=jnp.float64),
        dz_col=jnp.asarray(col.dz, dtype=jnp.float64),
        zi_col=jnp.asarray(col.zi, dtype=jnp.float64),
        nbedrock_col=jnp.asarray(col.nbedrock, dtype=jnp.int32),
        isoicol_col=jnp.asarray(SurfaceAlbedoMod.isoicol, dtype=jnp.int32),
        latdeg_grc=jnp.asarray(grc.latdeg, dtype=jnp.float64),
        londeg_grc=jnp.asarray(grc.londeg, dtype=jnp.float64),
    )


def site_or_modules(site) -> site_type:
    """``site`` if given, else the subgrid modules' values (for direct calls of a routine, e.g.
    the Fortran golden tests)."""
    return site if site is not None else site_from_modules()
