"""CO2 of tower runs (run.yaml forcing.co2): the tower files' constant, another constant, or a
CO2 time series file read as CLM's data atmosphere does."""

import datetime as dt
from pathlib import Path

import netCDF4 as nc
import numpy as np

from clm_ml import load_experiment
from clm_ml.setup.towers import gridcell_sources

ROOT = Path(__file__).resolve().parents[2]
CHATS = ROOT / "configs" / "chats7_1day"
CO2_FILE = (
    ROOT / "data/input_files/ctsm5_4/fco2_datm_globalSSP3-7.0_simyr_1750-2501_CMIP6_c201101.nc"
)


def _first_steps(overrides=()):
    exp = load_experiment(CHATS, list(overrides))
    tower = gridcell_sources(exp)[0].chunk(0, 4).tower
    return tower.co2[:, 0], tower.co2_partial_pressure[:, 0], tower.air_pressure[:, 0]


def test_default_is_the_tower_files_383_ppm():
    co2, pco2, pref = _first_steps()
    np.testing.assert_array_equal(co2, 383.0)
    np.testing.assert_array_equal(pco2, (383.0 / 1.0e6) * pref)


def test_constant_co2():
    co2, pco2, pref = _first_steps(["run.forcing.co2.ppm=400"])
    np.testing.assert_array_equal(co2, 400.0)
    np.testing.assert_array_equal(pco2, (400.0 / 1.0e6) * pref)


def test_co2_file_linear_in_time():
    """Global annual values, linear in time at the start of each step (CLM's co2tseries
    stream); 1 May 2007 lies between the values of mid-2006 and mid-2007."""
    co2, pco2, pref = _first_steps(
        ["run.forcing.co2.source=file", f"run.forcing.co2.file={CO2_FILE}"]
    )
    with nc.Dataset(CO2_FILE) as ds:
        t = nc.num2date(ds["time"][:], ds["time"].units, ds["time"].calendar,
                        only_use_cftime_datetimes=False)  # fmt: skip
        values = np.asarray(ds["CO2"][:], dtype=float).reshape(-1)
    k = [i for i, x in enumerate(t) if x.year == 2006][0]
    lo, hi = t[k], t[k + 1]
    when = dt.datetime(2007, 5, 1)
    frac = (when - dt.datetime(lo.year, lo.month, lo.day, lo.hour)).total_seconds() / (
        dt.datetime(hi.year, hi.month, hi.day, hi.hour)
        - dt.datetime(lo.year, lo.month, lo.day, lo.hour)
    ).total_seconds()
    expected = values[k] + (values[k + 1] - values[k]) * frac
    np.testing.assert_allclose(co2[1], expected, rtol=1e-12)
    assert co2[2] > co2[1]
    np.testing.assert_array_equal(pco2, (co2 / 1.0e6) * pref)
