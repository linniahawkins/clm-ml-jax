"""Health checks, restart boundaries and restart files (offline_executable.main)."""

import datetime as dt

import numpy as np
import pytest

from clm_ml.state import CanopyState, SoilState, State
from offline_executable.main import (
    RESTART_VERSION,
    check_health,
    read_restart,
    restart_boundary,
    write_restart,
)


def _state(tair=300.0, tleaf=(0.0, 295.0)):
    """One patch's state (0-based, no patch axis): two layers in use and one padding layer."""
    canopy = CanopyState(
        air_temperature=np.array([tair, 299.0, 1e36]),  # the padding layer holds a fill value
        vapor_pressure=np.array([1500.0, 1400.0, 0.0]),
        co2=np.array([380.0, 380.0, 0.0]),
        leaf_temperature=np.array([list(tleaf), [0.0, 0.0], [0.0, 0.0]]),  # 0: no leaves
        leaf_water_potential=np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 0.0]]),
        intercepted_water=np.array([0.0, 0.0, 0.0]),
        ground_temperature=np.array(290.0),
        acclimation_temperature=np.array(288.0),
        acclimation_steps=np.array(5.0),
    )
    return State(canopy=canopy, soil=SoilState(temperature=np.array([285.0, 284.0])))


def test_health_check_passes_and_ignores_fill_values_and_zeros():
    check_health(_state(), _state(), "test")


def test_health_check_catches_nan_and_implausible_temperatures():
    with pytest.raises(RuntimeError, match="air_temperature: 1 non-finite"):
        check_health(_state(tair=np.nan), _state(), "test")
    with pytest.raises(RuntimeError, match=r"leaf_temperature: range 400.0-400.0 K"):
        check_health(_state(tleaf=(0.0, 400.0)), _state(), "test")


def test_restart_boundaries():
    # a restart at the first chunk end on or after a boundary (decision 33)
    t = dt.datetime
    assert restart_boundary(t(2009, 12, 25), t(2010, 1, 1), (1, "year"))
    assert restart_boundary(t(2009, 12, 25), t(2010, 1, 4, 0, 30), (1, "year"))
    assert not restart_boundary(t(2010, 1, 1), t(2010, 1, 11), (1, "year"))  # at its start
    assert not restart_boundary(t(2010, 5, 25), t(2010, 6, 4), (1, "year"))
    assert restart_boundary(t(2009, 12, 25), t(2010, 1, 4), (10, "year"))
    assert not restart_boundary(t(2009, 12, 25), t(2010, 1, 4), (4, "year"))
    assert restart_boundary(t(2010, 5, 25), t(2010, 6, 4), (1, "month"))
    assert restart_boundary(t(2010, 3, 25), t(2010, 4, 4), (3, "month"))  # Jan, Apr, Jul, Oct
    assert not restart_boundary(t(2010, 5, 25), t(2010, 6, 4), (3, "month"))
    assert not restart_boundary(t(2010, 6, 1, 0, 30), t(2010, 6, 11), (1, "month"))
    assert not restart_boundary(t(2009, 12, 25), t(2010, 1, 4), None)


def test_restart_file_keeps_previous_and_falls_back(tmp_path):
    path = tmp_path / "restart.pkl"
    write_restart(path, {"version": RESTART_VERSION, "n": 1})
    write_restart(path, {"version": RESTART_VERSION, "n": 2})
    assert read_restart(path)["n"] == 2
    assert read_restart(tmp_path / "restart.pkl.prev")["n"] == 1
    path.write_bytes(b"truncated")  # e.g. killed while writing
    assert read_restart(path)["n"] == 1  # falls back to .prev
