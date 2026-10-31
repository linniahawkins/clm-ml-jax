"""
JAX translation of clmDataMod Fortran module.

Read CLM forcing data for tower sites: leaf and stem area indices,
volumetric soil moisture, and optional soil moisture adjustment factors
from CLM netCDF history files. The records a run needs are read once
before the time loop (:func:`read_clm_history_series`); ``clm_ml.legacy.soil.clmData``
installs one record (``clm_ml.physics.soil.soil_water_from_history``).

Original Fortran module: clmDataMod
Fortran lines 1-250
"""

from typing import Dict, Iterable, NamedTuple, Optional

import jax.numpy as jnp
import netCDF4 as nc  # noqa: F401
from jax import Array

from clm_src_biogeophys.CanopyStateType import canopystate_type  # noqa: F401
from clm_src_biogeophys.SoilStateType import soilstate_type  # noqa: F401
from clm_src_biogeophys.SurfaceAlbedoType import surfalb_type  # noqa: F401
from clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from clm_src_main.abortutils import handle_err  # noqa: F401
from clm_src_main.clm_varcon import denh2o, spval  # noqa: F401
from clm_src_main.clm_varpar import nlevgrnd, nlevsoi  # noqa: F401
from offline_driver.clmSoilOptionMod import clm_phys, nlev_soil_adjust  # noqa: F401


class ClmHistoryRecord(NamedTuple):
    """One time slice of the CLM history (and soil adjustment) files, as read."""

    elai: float  # Leaf area index (m2/m2)
    esai: float  # Stem area index (m2/m2)
    h2osoi_clm45: Array  # (nlevgrnd,) volumetric soil water; spval unless clm_phys == CLM4_5
    h2osoi_clm50: Array  # (nlevsoi,) volumetric soil water; spval unless clm_phys == CLM5_0
    h2osoi_factor: Optional[float]  # soil moisture adjustment factor (None if not used)


def read_clm_history_series(
    fin_clm: str, fin_soil_adjust: str, strts: Iterable[int]
) -> Dict[int, ClmHistoryRecord]:
    """
    Read the CLM history records for the given 1-based time slices.

    Same variables and conversions as the per-step readers (Fortran
    ``readCLMveg``, ``readCLMsoil``, ``readSoilWatFactor``), read once for the
    whole run so that no file is opened inside the time loop.
    """
    records = {}
    adjust = nc.Dataset(fin_soil_adjust, "r") if nlev_soil_adjust > 0 else None
    try:
        with nc.Dataset(fin_clm, "r") as ds:
            for strt in sorted(set(strts)):
                elai, esai = readCLMveg(ds, strt)
                h2osoi_clm45, h2osoi_clm50 = readCLMsoil(ds, strt)
                factor = readSoilWatFactor(adjust, strt) if adjust is not None else None
                records[strt] = ClmHistoryRecord(elai, esai, h2osoi_clm45, h2osoi_clm50, factor)
    finally:
        if adjust is not None:
            adjust.close()
    return records


# ---------------------------------------------------------------------------
# Private: read LAI and SAI from CLM netCDF history file
# ---------------------------------------------------------------------------


def readCLMveg(
    ds: nc.Dataset,
    strt: int,
) -> tuple[float, float]:
    """
    Read leaf and stem area index from a CLM netCDF history file.

    Mirrors Fortran subroutine ``readCLMveg`` (lines 117-165).

    Reads a single spatial gridcell at time slice ``strt`` from the
    ``ELAI`` and ``ESAI`` variables. In the Fortran the arrays are
    dimensioned ``(1,1,1)`` (lon=1, lat=1, time=1 for a tower site);
    here scalar floats are returned directly.

    The netCDF convention note from the Fortran source (lines 136-138)
    applies: Fortran column-major ``(lon, lat, time)`` maps to Python/
    C row-major ``(time, lat, lon)``; indexing is adjusted accordingly.

    Args:
        ds: Open CLM netCDF history file.
        strt: 1-based time slice index.

    Returns:
        Tuple of ``(elai_mod, esai_mod)`` as scalar floats.
    """
    # Fortran: start2=(1, strt), count2=(1,1) — lon=1 gridcell, 1 timestep
    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    # ELAI(nlndgrid, ntime) — Fortran lines 142-146
    elai_mod = float(ds.variables["ELAI"][t, 0])

    # ESAI(nlndgrid, ntime) — Fortran lines 148-152
    esai_mod = float(ds.variables["ESAI"][t, 0])

    return elai_mod, esai_mod


# ---------------------------------------------------------------------------
# Private: read volumetric soil water from CLM netCDF history file
# ---------------------------------------------------------------------------


def readCLMsoil(
    ds: nc.Dataset,
    strt: int,
) -> tuple[Array, Array]:
    """
    Read volumetric soil water from a CLM netCDF history file.

    Mirrors Fortran subroutine ``readCLMsoil`` (lines 167-220).

    Reads the ``H2OSOI`` variable for the current time slice.
    Depending on ``clm_phys``:

    - ``'CLM4_5'``: reads ``nlevgrnd`` layers into ``h2osoi_clm45``
      (shape ``(nlevgrnd,)``); ``h2osoi_clm50`` is set to ``spval``.
    - ``'CLM5_0'``: reads ``nlevsoi`` layers into ``h2osoi_clm50``
      (shape ``(nlevsoi,)``); ``h2osoi_clm45`` is set to ``spval``.

    Both arrays are initialised to ``spval`` before reading, matching
    the Fortran ``h2osoi_clm45(:,:,:) = spval`` initialisation
    (Fortran lines 181-182).

    Args:
        ds: Open CLM netCDF history file.
        strt: 1-based time slice index.

    Returns:
        Tuple of ``(h2osoi_clm45, h2osoi_clm50)`` as 1-D JAX arrays
        of shapes ``(nlevgrnd,)`` and ``(nlevsoi,)`` respectively.
    """
    # Initialize to spval — Fortran lines 181-182
    h2osoi_clm45 = jnp.full((nlevgrnd,), spval, dtype=jnp.float64)
    h2osoi_clm50 = jnp.full((nlevsoi,), spval, dtype=jnp.float64)

    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    if clm_phys == "CLM4_5":
        # H2OSOI(nlndgrid, nlevgrnd, ntime) — Fortran lines 195-201
        # Fortran start3=(1,1,strt), count3=(1,nlevgrnd,1)
        data = ds.variables["H2OSOI"][t, :nlevgrnd, 0]  # (nlevgrnd,)
        h2osoi_clm45 = jnp.array(data, dtype=jnp.float64)

    elif clm_phys == "CLM5_0":
        # H2OSOI(nlndgrid, nlevsoi, ntime) — Fortran lines 205-211
        # Fortran start3=(1,1,strt), count3=(1,nlevsoi,1)
        data = ds.variables["H2OSOI"][t, :nlevsoi, 0]  # (nlevsoi,)
        h2osoi_clm50 = jnp.array(data, dtype=jnp.float64)

    return h2osoi_clm45, h2osoi_clm50


# ---------------------------------------------------------------------------
# Private: read soil moisture adjustment factor from netCDF file
# ---------------------------------------------------------------------------


def readSoilWatFactor(
    ds: nc.Dataset,
    strt: int,
) -> float:
    """
    Read soil moisture adjustment factor from a netCDF file.

    Mirrors Fortran subroutine ``readSoilWatFactor`` (lines 222-248).

    Reads a single scalar ``FACTOR`` value at time slice ``strt``
    from a one-dimensional time series variable.

    Args:
        ds: Open soil moisture adjustment netCDF file.
        strt: 1-based time slice index.

    Returns:
        Scalar soil moisture adjustment factor.
    """
    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    # factor(ntime) — Fortran lines 240-244
    h2osoi_factor_loc = float(ds.variables["FACTOR"][t])

    return h2osoi_factor_loc
