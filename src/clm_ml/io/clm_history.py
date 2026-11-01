"""CLM history files of tower runs (Phase 7.4; from the legacy ``offline_driver.clmDataMod``):
leaf and stem area index, volumetric soil water, the optional soil-water adjustment factor and
the initial soil temperature.

The records a run needs are read before its time loop, chunk by chunk
(:func:`read_clm_history_series`); the soil layer counts and the history's soil-water layout
are arguments (``layer_structure``: model.yaml ``soil.layer_structure``).

Fortran reference: CLM-ml v2 ``clmDataMod.F90`` (``readCLMveg``, ``readCLMsoil``,
``readSoilWatFactor``) and ``CLMml_driver.F90::SoilInit``.
"""

from __future__ import annotations

from typing import Dict, Iterable, NamedTuple, Optional

import netCDF4 as nc
import numpy as np

from clm_ml.setup.soil import SOIL_LEVELS

SPVAL = 1.0e36  # clm_varcon.spval: soil water of the layout the history does not use


class ClmHistoryRecord(NamedTuple):
    """One time slice of the CLM history (and soil adjustment) files, as read."""

    elai: float  # leaf area index (m2/m2)
    esai: float  # stem area index (m2/m2)
    h2osoi_clm45: np.ndarray  # (nlevgrnd,) volumetric soil water; spval unless clm4_5
    h2osoi_clm50: np.ndarray  # (nlevsoi,) volumetric soil water; spval unless clm5_0
    h2osoi_factor: Optional[float]  # soil water adjustment factor (None if not used)


def read_clm_history_series(
    path: str,
    layer_structure: str,
    time_slices: Iterable[int],
    soil_adjust_path: Optional[str] = None,
) -> Dict[int, ClmHistoryRecord]:
    """The CLM history records of the given 1-based time slices: ``ELAI``, ``ESAI``,
    ``H2OSOI`` (the layers of ``layer_structure``) and, with ``soil_adjust_path``, the
    adjustment factor ``FACTOR``."""
    nlevsoi, nlevgrnd = SOIL_LEVELS[layer_structure]
    records = {}
    adjust = nc.Dataset(soil_adjust_path, "r") if soil_adjust_path else None
    try:
        with nc.Dataset(path, "r") as ds:
            n_time = ds.dimensions["time"].size
            levels = ds.variables["H2OSOI"].shape[1]
            want = nlevgrnd if layer_structure == "clm4_5" else nlevsoi
            if levels != want:
                raise ValueError(
                    f"{path}: H2OSOI has {levels} soil levels; soil.layer_structure "
                    f"{layer_structure} needs a history with {want}"
                )
            for strt in sorted(set(time_slices)):
                if not 1 <= strt <= n_time:
                    raise ValueError(
                        f"{path}: time slice {strt} outside 1..{n_time} (does the run start "
                        "before forcing.clm_history.start or end after the file?)"
                    )
                t = strt - 1
                elai = float(ds.variables["ELAI"][t, 0])
                esai = float(ds.variables["ESAI"][t, 0])
                clm45 = np.full((nlevgrnd,), SPVAL)
                clm50 = np.full((nlevsoi,), SPVAL)
                if layer_structure == "clm4_5":
                    clm45 = np.array(ds.variables["H2OSOI"][t, :nlevgrnd, 0], dtype=np.float64)
                else:
                    clm50 = np.array(ds.variables["H2OSOI"][t, :nlevsoi, 0], dtype=np.float64)
                factor = float(adjust.variables["FACTOR"][t]) if adjust is not None else None
                records[strt] = ClmHistoryRecord(elai, esai, clm45, clm50, factor)
    finally:
        if adjust is not None:
            adjust.close()
    return records


def history_interval(path: str) -> int:
    """The interval of a CLM history file's time axis (s)."""
    with nc.Dataset(path) as ds:
        t = ds.variables["time"]
        return int(round(float(t[1] - t[0]) * 86400.0))
