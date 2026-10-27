"""
Forcing: everything prescribed from outside the model, loaded before the run.

Meteorology is per gridcell, shape ``(time, gridcell)``; CLM history data (leaf/stem area
and soil water) is per patch/column, shape ``(time, patch)`` / ``(time, column, layer)``.
Wavebands are ``(vis, nir)``.

``Forcing`` holds arrays only: no files are opened during a run. It is built by
:func:`clm_ml.legacy.forcing.load_forcing`, which applies the same derivations as the
Fortran ``TowerMetMod`` (humidity from relative humidity, longwave from emissivity, solar
partitioning).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class TowerForcing:
    """Meteorology at the forcing height, per gridcell. Shape ``(time, gridcell)``."""

    wind_speed: np.ndarray  # m s-1
    air_temperature: np.ndarray  # K
    specific_humidity: np.ndarray  # kg kg-1
    air_pressure: np.ndarray  # Pa
    longwave_down: np.ndarray  # W m-2
    rainfall: np.ndarray  # kg H2O m-2 s-1 (= mm s-1)
    forcing_height: np.ndarray  # m
    co2: np.ndarray  # umol mol-1
    co2_partial_pressure: np.ndarray  # Pa
    o2_partial_pressure: np.ndarray  # Pa
    shortwave_direct: np.ndarray  # (time, gridcell, 2) W m-2
    shortwave_diffuse: np.ndarray  # (time, gridcell, 2) W m-2


@dataclasses.dataclass(frozen=True)
class ClmHistoryForcing:
    """Prescribed land state from the CLM history file, per patch / soil column."""

    history_index: np.ndarray  # (time,) int, 1-based time slice in the CLM history file
    leaf_area_index: np.ndarray  # (time, patch) m2 m-2
    stem_area_index: np.ndarray  # (time, patch) m2 m-2
    soil_water_clm45: np.ndarray  # (time, column, nlevgrnd) m3 m-3 (spval unless CLM4.5)
    soil_water_clm50: np.ndarray  # (time, column, nlevsoi) m3 m-3 (spval unless CLM5.0)
    soil_water_factor: Optional[np.ndarray]  # (time, column) adjustment factor, or None


@dataclasses.dataclass(frozen=True)
class Forcing:
    tower: TowerForcing
    clm_history: ClmHistoryForcing

    @property
    def n_steps(self) -> int:
        return self.tower.air_temperature.shape[0]

    def window_indices(self, t: int) -> tuple[int, int, int]:
        """0-based (previous, current, next) record indices around step ``t``.

        At the first step "previous" is the current record, and at the last step "next" is
        the current record, as in the Fortran driver (``itim_next = min(itim+1, ntim)``).
        """
        return max(t - 1, 0), t, min(t + 1, self.n_steps - 1)
