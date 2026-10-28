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
from typing import Callable, Mapping, Sequence

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
