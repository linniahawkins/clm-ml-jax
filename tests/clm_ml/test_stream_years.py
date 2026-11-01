"""CDEPS stream time axis for the gridded forcing (``taxmode``, data years): a run in the first
data year interpolates from the last record of the last data year (cycle), as CLM does."""

import datetime as dt

import netCDF4 as nc
import numpy as np
import pytest

from clm_ml.grid_forcing import PointMap, StreamYears, YearlyStream, lower_index, stream_time


def test_data_year_cycles_as_cdeps():
    y = StreamYears(taxmode="cycle", year_first=1901, year_last=2025)
    assert [y.data_year(m) for m in (1900, 1901, 2000, 2025, 2026)] == [
        2025,
        1901,
        2000,
        2025,
        1901,
    ]
    shifted = StreamYears(taxmode="cycle", year_first=1901, year_last=1920, year_align=1850)
    assert [shifted.data_year(m) for m in (1850, 1869, 1870)] == [1901, 1920, 1901]
    assert StreamYears(taxmode="limit", year_first=1901, year_last=2025).data_year(1900) == 1900


def _files(tmp_path, years):
    """Yearly noleap files, 4 records a day at 0, 6, 12, 18 h for the first 3 days; value =
    year + day/10 + hour/1000."""
    for year in years:
        with nc.Dataset(tmp_path / f"s.{year}.nc", "w") as ds:
            ds.createDimension("time", None)
            ds.createDimension("lat", 1)
            ds.createDimension("lon", 1)
            t = ds.createVariable("time", "f8", ("time",))
            t.units, t.calendar = f"days since {year}-01-01 00:00:00", "noleap"
            days = np.array([d + h / 24 for d in range(3) for h in (0, 6, 12, 18)])
            t[:] = days
            v = ds.createVariable("X", "f8", ("time", "lat", "lon"))
            v[:] = (year + np.floor(days) / 10 + (days % 1) * 24 / 1000)[:, None, None]


def test_first_data_year_reads_the_last_years_last_record(tmp_path):
    _files(tmp_path, (2001, 2002, 2003))
    pm = PointMap(
        points=np.array([0]), index=np.array([[0, 0, 0, 0]]), weight=np.array([[1.0, 0, 0, 0]])
    )
    loop = StreamYears(taxmode="cycle", year_first=2001, year_last=2003)
    stream = YearlyStream(str(tmp_path / "s.{year}.nc"), ["X"], pm, 1, loop)
    t0 = dt.datetime(2001, 1, 1, 0, 0)
    stamps, where, cal = stream.window(t0, t0)
    i = lower_index(stamps, stream_time(t0, cal) - 1.0)  # just before the first record
    assert where[i] == (2003, 11)  # 2003's last record, moved back to 2000
    assert where[i + 1] == (2001, 0)
    limit = YearlyStream(str(tmp_path / "s.{year}.nc"), ["X"], pm, 1)
    s2, _, _ = limit.window(t0, t0)
    with pytest.raises(ValueError, match="do not cover"):
        lower_index(s2, stream_time(t0, cal) - 1.0)
