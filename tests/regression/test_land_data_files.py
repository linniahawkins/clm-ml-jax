"""
Prescribed land data from user files (decision 14), end to end on CHATS7:

1. LAI, SAI and soil water written from the CLM history file itself, stamped at the start of
   each half-hour interval, reproduce the default run exactly (each step's midpoint is a
   record's midpoint, so the interpolation returns the record).
2. A monthly LAI file gives the leaf area index interpolated between monthly midpoints.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "src" / "input_files" / "clm5_0" / "CHATS7"
HISTORY = next(HISTORY.glob("*.clm2.h1.*.nc"))
VARIABLES = ["leaf_area_index", "stem_area_index", "latent_heat_flux", "sensible_heat_flux",
             "gpp", "soil_water_stress", "soil_water_potential"]  # fmt: skip


def _from_history(path: Path, names) -> None:
    import netCDF4 as nc

    with nc.Dataset(HISTORY) as src, nc.Dataset(path, "w") as dst:
        n = src.dimensions["time"].size
        dst.createDimension("time", n)
        dst.createDimension("levsoi", src.dimensions["levsoi"].size)
        t = dst.createVariable("time", "f8", ("time",))
        t.units = src["time"].units
        t[:] = src["time"][:] - 1.0 / 48  # history stamps end the interval; files start it
        for name, src_name in names.items():
            v = src[src_name][:]
            dims = ("time", "levsoi") if v.ndim == 3 else ("time",)
            dst.createVariable(name, "f8", dims)[:] = v.reshape((n,) + v.shape[1:-1])


def _monthly_lai(path: Path) -> None:
    import netCDF4 as nc

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 3)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 2007-04-01 00:00:00"
        t[:] = [0, 30, 61]  # April, May, June
        ds.createVariable("LAI", "f8", ("time",))[:] = [1.0, 2.0, 3.0]
        ds.createVariable("SAI", "f8", ("time",))[:] = [0.5, 0.5, 0.5]


def _run(workdir: Path, forcing: dict):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.run import load_run

    workdir.mkdir()
    convert_namelist(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day", workdir)
    run_yaml = workdir / "run.yaml"
    data = yaml.safe_load(run_yaml.read_text())
    data["forcing"].update(forcing)
    data["output"] = {
        "directory": str(workdir / "out"),
        "streams": [{"name": "step", "frequency": "step", "statistic": "mean",
                     "variables": VARIABLES}],
    }  # fmt: skip
    run_yaml.write_text(yaml.safe_dump(data))
    return load_run(run_yaml).execute(quiet=True)["step"]


def test_land_data_files(tmp_path):
    _from_history(tmp_path / "veg.nc", {"LAI": "ELAI", "SAI": "ESAI"})
    _from_history(tmp_path / "soil.nc", {"H2OSOI": "H2OSOI"})
    _monthly_lai(tmp_path / "monthly.nc")
    base = _run(tmp_path / "base", {})
    files = _run(
        tmp_path / "files",
        {"vegetation": {"source": "file", "file": str(tmp_path / "veg.nc")},
         "soil_water": {"source": "file", "file": str(tmp_path / "soil.nc")}},
    )  # fmt: skip
    monthly = _run(
        tmp_path / "monthly",
        {"vegetation": {"source": "file", "file": str(tmp_path / "monthly.nc")}},
    )

    # 1. the history data as user files reproduce the default run: the inputs to round-off
    # (times in days -> seconds), the fluxes within the regression tolerance (round-off
    # moves single steps by ~1e-8; numerics_and_regression.md §1)
    for name in VARIABLES:
        rtol = 1e-12 if name.endswith("area_index") else 1e-6
        np.testing.assert_allclose(
            files.data[name], base.data[name], rtol=rtol, atol=1e-12, err_msg=name
        )

    # 2. monthly LAI, linear between 16 Apr 00:00 (1.0) and 16 May 12:00 (2.0)
    apr, may = dt.datetime(2007, 4, 16), dt.datetime(2007, 5, 16, 12)
    times = [dt.datetime.fromisoformat(str(t)[:19]) for t in np.asarray(monthly.time, "M8[s]")]
    step = times[1] - times[0]
    expected = [1.0 + (t - step / 2 - apr) / (may - apr) for t in times]  # t: step end
    # (the output is the layers' sum; the beta-distribution fractions, from a continued
    # fraction with tolerance ~3e-7, sum to 1 only to ~1e-7)
    lai = monthly.data["leaf_area_index"][:, 0]
    np.testing.assert_allclose(lai, expected, rtol=1e-6)
    np.testing.assert_allclose(monthly.data["stem_area_index"][:, 0], 0.5, rtol=1e-6)
    assert not np.allclose(monthly.data["latent_heat_flux"], base.data["latent_heat_flux"])
