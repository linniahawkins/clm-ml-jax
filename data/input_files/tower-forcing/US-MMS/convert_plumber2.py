"""Monthly tower-forcing files for CLM-ML from a CTSM PLUMBER2 DATM file.

The CTSM single-point forcing (``CTSM_DATM_<site>_<first>-<last>.nc``, hourly, UTC, from the
CESM input data) is cut into calendar months in the layout of
``data/input_files/tower-forcing/CHATS7``: one file per month, ``time`` in days since the
first day of the month at 00:00 UTC, the variables in double precision.

The PLUMBER2 file stores time as single-precision seconds, which cannot hold every hour
exactly after a few years; the records are therefore dated by their index (one record per
interval from the file's origin), after checking that the stored times agree to within the
single-precision spacing.

Usage::

    python convert_plumber2.py CTSM_DATM_US-MMS_1999-2014.nc 2012-05 2012-09 OUTDIR
"""

import argparse
import datetime as dt
import os

import netCDF4 as nc
import numpy as np

VARIABLES = ("PRECTmms", "FSDS", "TBOT", "PSRF", "QBOT", "WIND", "FLDS", "ZBOT")


def month_range(first: str, last: str):
    year, month = map(int, first.split("-"))
    end = tuple(map(int, last.split("-")))
    while (year, month) <= end:
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def convert(path: str, first: str, last: str, outdir: str) -> None:
    src = nc.Dataset(path)
    time = src.variables["time"]
    units = time.units
    if not units.startswith("seconds since "):
        raise ValueError(f"{path}: time units {units!r}, expected 'seconds since ...'")
    origin = dt.datetime.fromisoformat(units[len("seconds since ") :].strip())
    stored = np.asarray(time[:], dtype=np.float64)
    interval = int(round(np.median(np.diff(stored))))
    index_time = np.arange(stored.size) * float(interval)
    spacing = np.spacing(np.float32(stored[-1])) * 2.0
    if np.max(np.abs(stored - index_time)) > spacing:
        raise ValueError(f"{path}: records are not evenly spaced at {interval} s")
    data = {name: np.asarray(src.variables[name][:, 0, 0], dtype=np.float64) for name in VARIABLES}
    lon = float(src.variables["LONGXY"][0, 0])
    lat = float(src.variables["LATIXY"][0, 0])
    site = getattr(src, "location", "")

    os.makedirs(outdir, exist_ok=True)
    for year, month in month_range(first, last):
        start = dt.datetime(year, month, 1)
        stop = dt.datetime(year + (month == 12), month % 12 + 1, 1)
        i0 = (start - origin).total_seconds() / interval
        i1 = (stop - origin).total_seconds() / interval
        if i0 != int(i0) or i0 < 0 or i1 > stored.size:
            raise ValueError(f"{start:%Y-%m}: outside the file or not on a record")
        i0, i1 = int(i0), int(i1)
        out_path = os.path.join(outdir, f"{year:04d}-{month:02d}.nc")
        with nc.Dataset(out_path, "w") as out:
            out.location = site
            out.source = f"{os.path.basename(path)}, records {i0}-{i1 - 1}, by convert_plumber2.py"
            out.createDimension("lat", 1)
            out.createDimension("lon", 1)
            out.createDimension("time", i1 - i0)
            for name, value in (("LONGXY", lon), ("LATIXY", lat)):
                v = out.createVariable(name, "f8", ("lat", "lon"))
                v.units = "degrees_east" if name == "LONGXY" else "degrees_north"
                v[:] = value
            t = out.createVariable("time", "f8", ("time",))
            t.units = f"days since {start:%Y-%m-%d} 00:00:00"
            t.long_name = "observation time"
            t.calendar = "gregorian"
            t[:] = np.arange(i1 - i0) * interval / 86400.0
            for name in VARIABLES:
                v = out.createVariable(name, "f8", ("time", "lat", "lon"))
                for attr in ("units", "long_name"):
                    if hasattr(src.variables[name], attr):
                        v.setncattr(attr, getattr(src.variables[name], attr))
                if name == "QBOT":
                    # The PLUMBER2 file labels QBOT "relative humidity, unitless"; it is the
                    # specific humidity (values up to 0.026), which DATM reads as Sa_shum.
                    v.units = "kg/kg"
                    v.long_name = "specific humidity at the lowest atm level"
                v[:, 0, 0] = data[name][i0:i1]
        print(out_path)
    src.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datm_file")
    parser.add_argument("first_month", help="YYYY-MM")
    parser.add_argument("last_month", help="YYYY-MM")
    parser.add_argument("outdir")
    args = parser.parse_args()
    convert(args.datm_file, args.first_month, args.last_month, args.outdir)
