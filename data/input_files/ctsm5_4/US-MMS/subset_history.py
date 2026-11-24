"""The CLM history record of a tower run for CLM-ML, cut from a CTSM single-point history.

CLM-ML reads ``ELAI``, ``ESAI``, ``H2OSOI`` and ``TSOI`` from a CLM history file at the
forcing interval (``run.yaml`` ``forcing.clm_history``), slice ``s`` covering the interval that
ends ``s`` intervals after the file's start. This script copies those fields (and
``levgrnd``, ``time_bounds``) for the intervals from ``start`` to ``end`` out of a CTSM
history file whose records are hourly averages, with ``time`` restated in days since
``start`` (the end of each interval, as CLM writes it) and the values kept in the file's
precision.

Usage::

    python subset_history.py CASE.clm2.h2.1999-01-01-18000.nc 2012-05-01T00:00 \\
        2012-10-01T00:00 OUT.nc
"""

import argparse
import datetime as dt

import netCDF4 as nc
import numpy as np

FIELDS = ("ELAI", "ESAI", "H2OSOI", "TSOI")


def subset(path: str, start: dt.datetime, end: dt.datetime, out_path: str) -> None:
    src = nc.Dataset(path)
    bounds = nc.num2date(src.variables["time_bounds"][:], src.variables["time"].units,
                         only_use_cftime_datetimes=False)  # fmt: skip
    first = np.array([dt.datetime(b.year, b.month, b.day, b.hour, b.minute) for b in bounds[:, 0]])
    last = np.array([dt.datetime(b.year, b.month, b.day, b.hour, b.minute) for b in bounds[:, 1]])
    keep = np.nonzero((first >= start) & (last <= end))[0]
    if keep.size == 0 or first[keep[0]] != start or last[keep[-1]] != end:
        raise ValueError(f"{path}: no complete records from {start} to {end}")
    if np.any(np.diff(keep) != 1):
        raise ValueError(f"{path}: records from {start} to {end} are not contiguous")
    i0, i1 = int(keep[0]), int(keep[-1]) + 1

    with nc.Dataset(out_path, "w") as out:
        out.source = f"{path}, records {i0}-{i1 - 1}, by subset_history.py"
        out.case_id = getattr(src, "case_id", "")
        out.Surface_dataset = getattr(src, "Surface_dataset", "")
        out.createDimension("time", i1 - i0)
        out.createDimension("lndgrid", 1)
        out.createDimension("levsoi", src.dimensions["levsoi"].size)
        out.createDimension("levgrnd", src.dimensions["levgrnd"].size)
        out.createDimension("hist_interval", 2)
        units = f"days since {start:%Y-%m-%d %H:%M:%S}"
        seconds_end = np.array([(t - start).total_seconds() for t in last[i0:i1]])
        seconds_start = np.array([(t - start).total_seconds() for t in first[i0:i1]])
        t = out.createVariable("time", "f8", ("time",))
        t.units = units
        t.long_name = "time"
        t[:] = seconds_end / 86400.0
        tb = out.createVariable("time_bounds", "f8", ("time", "hist_interval"))
        tb.long_name = "history time interval endpoints"
        tb[:, 0] = seconds_start / 86400.0
        tb[:, 1] = seconds_end / 86400.0
        lev = out.createVariable("levgrnd", "f8", ("levgrnd",))
        lev.units = "m"
        lev.long_name = "coordinate soil levels"
        lev[:] = src.variables["levgrnd"][:]
        for name in FIELDS:
            var = src.variables[name]
            v = out.createVariable(name, var.dtype, var.dimensions)
            for attr in ("units", "long_name"):
                if hasattr(var, attr):
                    v.setncattr(attr, getattr(var, attr))
            v[:] = var[i0:i1]
    src.close()
    print(out_path, i1 - i0, "records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("history_file")
    parser.add_argument("start", type=dt.datetime.fromisoformat)
    parser.add_argument("end", type=dt.datetime.fromisoformat)
    parser.add_argument("out")
    args = parser.parse_args()
    subset(args.history_file, args.start, args.end, args.out)
