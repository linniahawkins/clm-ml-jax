"""
Time interpolation of forcing records to the model step (plan §6.9, decision 13).

A forcing record stamped ``t`` describes the interval ``[t, t + Δf]`` (``Δf`` the data
interval), as the tower files do: its value is the interval's mean, located at the
interval's midpoint. A model step ``[s, s + Δt]`` (``Δt`` dividing ``Δf``, steps aligned
with records) gets:

- **linear** quantities (air temperature, humidity, wind, pressure, longwave, height):
  linear interpolation between record midpoints, evaluated at the step's midpoint
  (held at the first/last record beyond the data);
- **shortwave**: the record's value times ``cosz(step) / cosz(interval)``, where ``cosz`` is
  the mean of ``max(cos(solar zenith), 0)`` over the step and the mean of those over all the
  interval's steps. The diurnal cycle within the interval follows the sun and the interval
  mean is conserved exactly. If the sun is down for the whole interval, the record's value is
  held constant;
- **precipitation**: the record's value (held constant), which conserves totals.

When ``Δt == Δf`` every step is a record and nothing is interpolated (callers bypass this
module, so results are unchanged).
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

#: How each raw tower variable is interpolated
RULES = {
    "zbot": "linear",
    "tbot": "linear",
    "rhbot": "linear",
    "qbot": "linear",
    "ubot": "linear",
    "fldsbot": "linear",
    "pbot": "linear",
    "fsdsbot": "coszen",
    "prect": "constant",
}
MISSING = -999.0  # tower files: variable absent


def interpolate_records(
    values: Mapping[str, np.ndarray],
    record_starts: Sequence[dt.datetime],
    record_seconds: int,
    step_starts: Sequence[dt.datetime],
    step_seconds: int,
    mean_cosz: Callable[[dt.datetime, int], float],
) -> dict[str, np.ndarray]:
    """``values[name]`` at consecutive records -> values at the model steps.

    ``record_starts`` must cover every step's interval; ``mean_cosz(start, seconds)`` is the
    mean of ``max(cos(zenith), 0)`` over ``[start, start + seconds]``.
    """
    if record_seconds % step_seconds:
        raise ValueError(f"model step {step_seconds} s must divide the data interval")
    per_record = record_seconds // step_seconds
    t0 = record_starts[0]

    def seconds(t):
        return (t - t0).total_seconds()

    rec_mid = np.array([seconds(t) for t in record_starts]) + record_seconds / 2.0
    step_mid = np.array([seconds(t) for t in step_starts]) + step_seconds / 2.0
    containing = np.floor(np.array([seconds(t) for t in step_starts]) / record_seconds)
    containing = containing.astype(int)
    if containing.min() < 0 or containing.max() >= len(record_starts):
        raise ValueError("the records do not cover the model steps")

    cosz_cache: dict = {}

    def step_cosz(start):
        if start not in cosz_cache:
            cosz_cache[start] = mean_cosz(start, step_seconds)
        return cosz_cache[start]

    out = {}
    for name, rule in RULES.items():
        if name not in values:
            continue
        v = np.asarray(values[name], dtype=float)
        if rule == "linear":
            missing = v == MISSING
            if missing.all():
                out[name] = np.full(len(step_starts), MISSING)
                continue
            if missing.any():
                raise ValueError(f"{name}: some records are missing (-999), others are not")
            out[name] = np.interp(step_mid, rec_mid, v)
        elif rule == "constant":
            out[name] = v[containing]
        else:  # coszen
            result = np.empty(len(step_starts))
            for k, (start, j) in enumerate(zip(step_starts, containing)):
                rec_start = record_starts[j]
                interval = np.mean(
                    [
                        step_cosz(rec_start + dt.timedelta(seconds=i * step_seconds))
                        for i in range(per_record)
                    ]
                )
                result[k] = v[j] * step_cosz(start) / interval if interval > 0 else v[j]
            out[name] = result
    return out


def mean_cosz_function(lat: float, lon: float, year: int, samples: int = 10):
    """``mean_cosz(start, seconds)`` for a site (``lat``, ``lon`` in radians): the mean of
    ``max(cos(solar zenith), 0)`` over the interval, from ``samples`` evenly spaced points,
    with the model's orbital functions (CLM's ``shr_orb_decl``/``shr_orb_cosz``; calendar
    day in UTC, as the model's own solar zenith angle)."""
    from clm_share.shr_orb_mod import shr_orb_cosz, shr_orb_decl, shr_orb_params
    from clm_src_utils.clm_time_manager import calday_of

    _eccen, _obliq, _mvelp, obliqr, lambm0, mvelpp = shr_orb_params(year)

    def calday(t: dt.datetime) -> float:
        return calday_of(t.year, t.month, t.day, t.hour * 3600 + t.minute * 60 + t.second)

    def mean_cosz(start: dt.datetime, seconds: int) -> float:
        total = 0.0
        for i in range(samples):
            t = start + dt.timedelta(seconds=(i + 0.5) * seconds / samples)
            day = calday(t)
            declin, _ = shr_orb_decl(day, _eccen, mvelpp, lambm0, obliqr)
            total += max(shr_orb_cosz(day, lat, lon, declin), 0.0)
        return total / samples

    return mean_cosz


def linear_between_midpoints(
    record_starts: Sequence[dt.datetime],
    values: np.ndarray,
    step_starts: Sequence[dt.datetime],
    step_seconds: int,
    last_end: Optional[dt.datetime] = None,
) -> np.ndarray:
    """Records stamped at the start of possibly irregular intervals (daily, monthly) ->
    values at the model steps' midpoints, linear between interval midpoints (held beyond
    the first and last). The last record's interval ends at ``last_end`` (default: as long
    as the one before). ``values`` has the record axis first; any further axes (e.g. soil
    layers) are kept."""
    t0 = record_starts[0]
    starts = np.array([(t - t0).total_seconds() for t in record_starts])
    if len(starts) > 1 and np.any(np.diff(starts) <= 0):
        raise ValueError("record times must increase")
    if last_end is not None:
        end = (last_end - t0).total_seconds()
    else:
        end = starts[-1] + (starts[-1] - starts[-2] if len(starts) > 1 else 0)
    mids = starts + np.diff(starts, append=end) / 2.0
    at = np.array([(t - t0).total_seconds() + step_seconds / 2.0 for t in step_starts])
    v = np.asarray(values, dtype=float)
    flat = v.reshape(v.shape[0], -1)
    out = np.stack([np.interp(at, mids, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape((len(at),) + v.shape[1:])


def read_land_data(
    path: str, names: Sequence[str], first: dt.datetime, last: dt.datetime
) -> tuple[list, dict, Optional[dt.datetime]]:
    """``(record_starts, {name: values}, last_end)`` from a daily/monthly land-data file, for the
    records whose midpoints bracket ``[first, last]`` (one record beyond each end, so
    memory does not depend on the file's length). The file has a ``time`` variable in
    ``days since <date>`` (each record starts its interval) and the named variables with
    time first (extra axes of size 1, e.g. a single gridcell, are dropped). ``last_end``
    is the start of the record after the window (None at the end of the file)."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        time = ds.variables["time"]
        origin = dt.datetime.fromisoformat(time.units.split("since")[1].strip())
        starts = [origin + dt.timedelta(days=float(d)) for d in np.asarray(time[:])]
        lo = max(sum(1 for t in starts if t <= first) - 2, 0)
        hi = min(sum(1 for t in starts if t <= last) + 1, len(starts))
        out = {}
        for name in names:
            if name not in ds.variables:
                raise ValueError(f"{path}: variable {name} not found")
            v = np.asarray(ds.variables[name][lo:hi], dtype=float)
            out[name] = v.reshape((v.shape[0],) + tuple(n for n in v.shape[1:] if n != 1))
    return starts[lo:hi], out, (starts[hi] if hi < len(starts) else None)
