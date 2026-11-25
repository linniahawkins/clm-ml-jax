"""
A run across 1 January, end to end: CHATS7 with its data dated from 31 December 2007, two
days, against the same run at its own dates (from 1 May).

The two runs get the same meteorology, leaf area and soil water; only the calendar days
differ, hence the sun's position (how the measured shortwave splits into direct and diffuse
and where it is absorbed). The calendar day starts again at 1 January: the forcing's time
interpolation within the step that ends there must stay between its records, so the air
at the canopy top stays within about 1 K of the other run's through the new year.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
TOWER_FILE = ROOT / "data" / "input_files" / "tower-forcing" / "CHATS7" / "2007-05.nc"
VARIABLES = ["canopy_top_air_temperature", "latent_heat_flux", "sensible_heat_flux"]


def _run(workdir: Path, new_year: bool):
    import sys

    import netCDF4 as nc

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from tests.experiment_tools import make_experiment

    workdir.mkdir()
    run = {"time": {"start": "2007-05-01T00:00", "length": "2 days"}}
    run["output"] = {
        "directory": str(workdir / "out"),
        "streams": [{"name": "steps", "frequency": "step", "statistic": "mean",
                     "variables": VARIABLES}],
    }  # fmt: skip
    if new_year:  # the same records, 244 days later
        shutil.copy(TOWER_FILE, workdir / "forcing.nc")
        with nc.Dataset(workdir / "forcing.nc", "a") as data:
            data["time"].units = "days since 2007-12-31 00:00:00"
        run["time"]["start"] = "2007-12-31T00:00"
        run["forcing"] = {
            "meteorology": {"files": [str(workdir / "forcing.nc")]},
            "clm_history": {"start": "2007-12-01T00:00"},
        }
    make_experiment(workdir, namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day", run=run)
    return load_experiment(workdir).execute(quiet=True)["steps"]


def test_run_across_1_january(tmp_path):
    may = _run(tmp_path / "may", new_year=False)
    new_year = _run(tmp_path / "new_year", new_year=True)
    assert may.time.shape == new_year.time.shape == (96,)
    assert str(new_year.time[47])[:16] == "2008-01-01T00:00"  # the step ending at 0Z 1 January
    assert str(new_year.time[-1])[:16] == "2008-01-02T00:00"
    air, air_may = (r.data["canopy_top_air_temperature"][:, 0] for r in (new_year, may))
    assert np.abs(air - air_may).max() < 2.0, np.abs(air - air_may).max()
    for name in ("latent_heat_flux", "sensible_heat_flux"):
        a, b = new_year.data[name][:, 0], may.data[name][:, 0]
        assert np.isfinite(a).all()
        assert np.abs(a - b).max() < 150.0, (name, np.abs(a - b).max())
        # through the new year as through the same hours of the other run
        assert np.abs(a[44:52] - b[44:52]).max() < 30.0, (name, a[44:52], b[44:52])
