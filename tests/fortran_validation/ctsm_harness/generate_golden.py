"""Make the golden data of CLM's soil water routines: write the cases, run CLM's Fortran on
them in the test harness (``build.py``), and write ``golden_IO/*.json``.

    python tests/fortran_validation/ctsm_harness/generate_golden.py [routine ...]
        [--work-dir DIR] [--ctsm DIR]

Without routine names, every routine of ``ROUTINES`` is regenerated. The cases are idealized
sweeps: random draws with a fixed seed over the ranges a routine can meet, plus hand-picked
edge values. Regenerating gives the same cases and, from the same CTSM tag and compiler, the
same values.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
from build import CTSM_TAG, DEFAULT_CTSM, DEFAULT_WORK, build

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden_IO"
SEED = 20261124
N_RANDOM = 300


def _log_uniform(rng, low: float, high: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), n))


def _saturation(rng, n: int) -> np.ndarray:
    """Relative saturation over CLM's bounds [0.01, 1]: half of the draws uniform, half
    uniform in the logarithm (the dry end, where the curve is steep)."""
    s = np.where(rng.random(n) < 0.5, rng.uniform(0.01, 1.0, n), _log_uniform(rng, 0.01, 1.0, n))
    return np.clip(s, 0.01, 1.0)


def soil_hk_cases() -> list[dict]:
    """Hydraulic conductivity: saturation from the dry bound to saturated; ice impedance 1
    (unfrozen) or ``10**(-e_ice*icefrac)`` with ``e_ice = 6``; saturated conductivity from
    clay to sand (mm/s); Clapp-Hornberger ``b`` from organic soil to heavy clay."""
    rng = np.random.default_rng(SEED)
    n = N_RANDOM
    s = _saturation(rng, n)
    imped = np.where(rng.random(n) < 0.5, 1.0, 10.0 ** (-6.0 * rng.uniform(0.0, 1.0, n)))
    hksat = _log_uniform(rng, 1.0e-5, 1.0e-1, n)
    bsw = rng.uniform(2.5, 25.0, n)
    cases = [
        {"s": float(a), "imped": float(b), "hksat": float(c), "bsw": float(d)}
        for a, b, c, d in zip(s, imped, hksat, bsw)
    ]
    for s_edge in (0.01, 1.0):
        for bsw_edge in (2.5, 6.9, 25.0):
            for imped_edge in (1.0, 1.0e-6):
                cases.append({"s": s_edge, "imped": imped_edge, "hksat": 0.0035, "bsw": bsw_edge})
    return cases


def soil_suction_cases() -> list[dict]:
    """Matric potential: saturation from the dry bound to saturated; saturated suction (mm)
    and Clapp-Hornberger ``b`` over CLM's soils."""
    rng = np.random.default_rng(SEED + 1)
    n = N_RANDOM
    s = _saturation(rng, n)
    sucsat = _log_uniform(rng, 10.0, 1000.0, n)
    bsw = rng.uniform(2.5, 25.0, n)
    cases = [
        {"s": float(a), "sucsat": float(b), "bsw": float(c)} for a, b, c in zip(s, sucsat, bsw)
    ]
    for s_edge in (0.01, 1.0):
        for bsw_edge in (2.5, 6.9, 25.0):
            for sucsat_edge in (10.0, 230.0, 1000.0):
                cases.append({"s": s_edge, "sucsat": sucsat_edge, "bsw": bsw_edge})
    return cases


NLEVSOI = 20
STEPS = (1800, 3600)  # model steps of the sweeps (s)


def _soil_levels() -> tuple[np.ndarray, np.ndarray]:
    """Thickness and node depth (m) of CLM's 20 soil levels (as HarnessColumnMod)."""
    dz = np.zeros(NLEVSOI)
    for j in range(1, 5):
        dz[j - 1] = j * 0.02
    for j in range(5, 14):
        dz[j - 1] = dz[3] + (j - 4) * 0.04
    for j in range(14, NLEVSOI + 1):
        dz[j - 1] = dz[12] + (j - 13) * 0.10
    zi = np.concatenate([[0.0], np.cumsum(dz)])
    return dz, 0.5 * (zi[:-1] + zi[1:])


DZ, Z = _soil_levels()


def _soil_properties(sand, clay, organic) -> dict[str, np.ndarray]:
    """Porosity, saturated conductivity (mm/s), Clapp-Hornberger ``b`` and saturated suction
    (mm) of each level from sand and clay (%) by CLM's texture formulas, mixed linearly (the
    conductivity in its logarithm) with those of organic soil by the organic fraction. The
    purpose is combinations that span CLM's soils, not CLM's exact mixing."""
    watsat = 0.489 - 0.00126 * sand
    bsw = 2.91 + 0.159 * clay
    sucsat = 10.0 * 10.0 ** (1.88 - 0.0131 * sand)
    hksat = 0.0070556 * 10.0 ** (-0.884 + 0.0153 * sand)
    return {
        "watsat": (1.0 - organic) * watsat + organic * 0.9,
        "hksat": np.exp((1.0 - organic) * np.log(hksat) + organic * np.log(0.1)),
        "bsw": (1.0 - organic) * bsw + organic * 4.0,
        "sucsat": (1.0 - organic) * sucsat + organic * 10.2,
    }


def _texture(rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sand, clay (%) and organic fraction of each level: one soil, two horizons, or a
    different soil at every level."""

    def draw(n):
        sand = rng.uniform(2.0, 95.0, n)
        clay = rng.uniform(1.0, 60.0, n) * (100.0 - sand) / 100.0
        organic = np.where(rng.random(n) < 0.25, rng.uniform(0.0, 1.0, n), 0.0)
        return sand, clay, organic

    kind = rng.integers(3)
    if kind == 0:
        return tuple(np.full(NLEVSOI, v[0]) for v in draw(1))
    if kind == 1:
        upper, lower = zip(*draw(2))
        below = np.arange(NLEVSOI) >= rng.integers(1, NLEVSOI)
        return tuple(np.where(below, lo, up) for up, lo in zip(upper, lower))
    return draw(NLEVSOI)


def _water_profile(rng) -> np.ndarray:
    """Relative saturation of each level: uniform, different at every level, wet over dry
    (a wetting front), or dry over saturated (a water table); in one profile out of six
    some levels are filled above saturation."""
    kind = rng.integers(4)
    if kind == 0:
        s = np.full(NLEVSOI, _saturation(rng, 1)[0])
    elif kind == 1:
        s = _saturation(rng, NLEVSOI)
    else:
        upper = np.arange(NLEVSOI) < rng.integers(1, NLEVSOI)
        wet, dry = rng.uniform(0.8, 1.0, NLEVSOI), _log_uniform(rng, 0.01, 0.4, NLEVSOI)
        s = np.where(upper, wet, dry) if kind == 2 else np.where(upper, dry, 1.0)
    if rng.random() < 1.0 / 6.0:
        s = np.where(rng.random(NLEVSOI) < 0.3, rng.uniform(1.0, 1.08, NLEVSOI), s)
    return s


def _root_fraction(rng) -> np.ndarray:
    """Share of the root uptake of each level: decreasing with depth, summing to one."""
    f = np.exp(-Z / rng.uniform(0.1, 2.0))
    return f / f.sum()


def _soil_water_case(dtime: int, nbedrock: int, qflx_infl: float, props, s, icefrac, sink):
    floats = {
        **props,
        "h2osoi_liq": np.asarray(s) * props["watsat"] * DZ * 1000.0,
        "icefrac": icefrac,
        "qflx_rootsoi": sink,
    }
    return {
        "dtime": dtime,
        "nbedrock": int(nbedrock),
        "qflx_infl": float(qflx_infl),
        **{k: [float(x) for x in np.broadcast_to(v, NLEVSOI)] for k, v in floats.items()},
    }


def soil_water_cases() -> list[dict]:
    """A whole call of the Richards solver, at both model steps.

    Random draws: soils from sand to clay and organic, uniform and layered; water from the
    dry bound to saturation and above, uniform, layered, wetting fronts and water tables;
    bedrock from the second level to below the column; infiltration none, light, negative
    (evaporation exceeding the input) and heavy enough to force halved sub-steps; root
    uptake none, ordinary and larger than the water of dry levels; one case in seven with
    ice in some levels. Then hand-picked edge cases."""
    rng = np.random.default_rng(SEED + 2)
    cases = []
    for dtime in STEPS:
        for _ in range(150):
            props = _soil_properties(*_texture(rng))
            nbedrock = NLEVSOI if rng.random() < 0.4 else rng.integers(2, NLEVSOI + 1)
            u = rng.random()
            if u < 0.25:
                infl = 0.0
            elif u < 0.55:
                infl = _log_uniform(rng, 1.0e-6, 1.0e-3, 1)[0]
            elif u < 0.70:
                infl = -_log_uniform(rng, 1.0e-6, 2.0e-4, 1)[0]
            else:
                infl = _log_uniform(rng, 1.0e-3, 3.0e-2, 1)[0]
            u = rng.random()
            if u < 0.3:
                sink = np.zeros(NLEVSOI)
            elif u < 0.8:
                sink = _log_uniform(rng, 1.0e-6, 3.0e-4, 1)[0] * _root_fraction(rng)
            else:
                # strong uptake from a few levels, wherever they are
                sink = np.where(rng.random(NLEVSOI) < 0.25, 1.0, 0.0)
                sink = _log_uniform(rng, 1.0e-4, 3.0e-3, 1)[0] * sink / max(sink.sum(), 1.0)
            if rng.random() < 1.0 / 7.0:
                frozen = np.arange(NLEVSOI) < rng.integers(1, NLEVSOI + 1)
                icefrac = np.where(frozen, rng.uniform(0.0, 1.0, NLEVSOI), 0.0)
            else:
                icefrac = np.zeros(NLEVSOI)
            cases.append(
                _soil_water_case(dtime, nbedrock, infl, props, _water_profile(rng), icefrac, sink)
            )

        # Edge cases
        loam = _soil_properties(40.0, 20.0, 0.0)
        sand = _soil_properties(92.0, 3.0, 0.0)
        clay = _soil_properties(10.0, 55.0, 0.0)
        none = np.zeros(NLEVSOI)
        roots = 2.0e-4 * np.exp(-Z / 0.5) / np.exp(-Z / 0.5).sum()
        top4 = np.arange(NLEVSOI) < 4
        edge = [
            # exactly saturated: at rest, with heavy rain, with roots
            (NLEVSOI, 0.0, loam, 1.0, none, none),
            (NLEVSOI, 2.0e-2, loam, 1.0, none, none),
            (NLEVSOI, 0.0, clay, 1.0, none, roots),
            # exactly at the dry bound: at rest, heavy rain, roots (water turns negative)
            (NLEVSOI, 0.0, loam, 0.01, none, none),
            (NLEVSOI, 2.0e-2, loam, 0.01, none, none),
            (NLEVSOI, 2.0e-2, sand, 0.01, none, none),
            (NLEVSOI, 2.0e-2, clay, 0.01, none, none),
            (NLEVSOI, 0.0, loam, 0.01, none, roots),
            # no water at all, and less than the solver's floor of 1e-6 kg/m2
            (NLEVSOI, 0.0, loam, 0.0, none, none),
            (NLEVSOI, 1.0e-4, loam, 0.0, none, roots),
            (NLEVSOI, 0.0, loam, 1.0e-7 / (loam["watsat"] * DZ * 1000.0), none, none),
            # the shallowest column the solver takes (two levels), dry and wet
            (2, 5.0e-3, loam, 0.3, none, roots),
            (2, 5.0e-3, loam, 0.99, none, none),
            (2, -1.0e-4, sand, 0.05, none, roots),
            # heavy rain onto wet levels over dry soil (many sub-steps)
            (NLEVSOI, 2.0e-2, loam, np.where(top4, 0.44 / 0.45, 0.06 / 0.45), none, roots),
            (12, 3.0e-2, sand, np.where(top4, 0.95, 0.03), none, none),
            (12, 3.0e-2, clay, np.where(top4, 0.95, 0.03), none, none),
            # hydrostatic equilibrium above a water table at 3 m: no flow
            (NLEVSOI, 0.0, loam, None, none, none),
            # frozen: every level full of ice, and ice in the top levels only, with rain
            (NLEVSOI, 1.0e-3, loam, 0.2, np.ones(NLEVSOI), none),
            (NLEVSOI, 5.0e-3, loam, 0.5, np.where(top4, 0.8, 0.0), roots),
            (8, 5.0e-3, clay, 0.9, np.where(top4, 0.0, 0.6), none),
        ]
        for nbedrock, infl, props, s, icefrac, sink in edge:
            if s is None:
                head = props["sucsat"] + np.maximum(3.0 - Z, 0.0) * 1000.0
                s = (head / props["sucsat"]) ** (-1.0 / props["bsw"])
            cases.append(_soil_water_case(dtime, nbedrock, infl, props, s, icefrac, sink))
    return cases


def soil_water_sequence_cases() -> list[dict]:
    """Sequences of calls of the Richards solver, the water carried from step to step, at
    both model steps (120 steps each): a rain pulse on dry soil; a long dry-down with strong
    root uptake and evaporation; steady rain filling a clay column over shallow bedrock to
    saturation and above; a water table in the column, then storms; sand over very shallow
    bedrock with storms; rain on a soil with ice in its top levels."""
    rng = np.random.default_rng(SEED + 3)
    nsteps = 120
    k = np.arange(nsteps)
    none = np.zeros(NLEVSOI)
    loam = _soil_properties(40.0, 20.0, 0.0)
    sand = _soil_properties(92.0, 3.0, 0.0)
    clay = _soil_properties(10.0, 55.0, 0.0)
    layered = _soil_properties(
        np.where(Z < 0.5, 60.0, 15.0), np.where(Z < 0.5, 10.0, 45.0), np.where(Z < 0.1, 0.6, 0.0)
    )
    cases = []
    for dtime in STEPS:
        day = 0.5 * (1.0 - np.cos(2.0 * np.pi * k * dtime / 86400.0))  # 0 at night, 1 at noon
        storms = np.where(rng.random(nsteps) < 0.12, _log_uniform(rng, 5.0e-4, 2.0e-2, nsteps), 0.0)
        scenarios = [
            # (bedrock level, soil, saturation, ice fraction, infiltration, uptake, roots)
            (NLEVSOI, loam, 0.15, none, np.where((k >= 10) & (k < 20), 5.0e-3, 0.0),
             2.0e-5 * day, 0.5),
            (NLEVSOI, layered, 0.7, none, -3.0e-5 * day, 2.5e-4 * day, 0.3),
            (10, clay, 0.5, none, np.full(nsteps, 2.0e-3), 1.0e-5 * day, 0.5),
            (NLEVSOI, loam, np.where(Z > 1.5, 1.0, 0.2), none,
             np.where(k >= 40, storms, 0.0), 5.0e-5 * day, 1.0),
            (4, sand, 0.1, none, storms, 1.0e-4 * day, 0.1),
            (NLEVSOI, loam, 0.4, np.where(Z < 0.3, 0.6, 0.0), storms, np.zeros(nsteps), 0.5),
        ]  # fmt: skip
        for nbedrock, props, s, icefrac, infl, uptake, root_depth in scenarios:
            case = _soil_water_case(dtime, nbedrock, 0.0, props, s, icefrac, none)
            del case["qflx_infl"], case["qflx_rootsoi"]
            fraction = np.exp(-Z / root_depth)
            cases.append(
                {
                    **case,
                    "nsteps": nsteps,
                    "root_fraction": [float(x) for x in fraction / fraction.sum()],
                    "qflx_infl": [float(x) for x in infl],
                    "uptake": [float(x) for x in uptake],
                }
            )
    return cases


ZI = np.concatenate([[0.0], np.cumsum(DZ)])  # depth of the level interfaces (m)
SLOPEBETA, SLOPEMAX = -3.0, 0.4  # microtopography parameters (ctsm60_params.c251211.nc)
DENICE = 917.0


def _micro_sigma(slope):
    """Microtopography (m) of a slope (degrees), as CLM's initVertical."""
    return (slope + SLOPEMAX ** (1.0 / SLOPEBETA)) ** SLOPEBETA


def _levels(x) -> list[float]:
    return [float(v) for v in np.broadcast_to(x, NLEVSOI)]


def _ice(rng, watsat) -> np.ndarray:
    """Ice (kg/m2) of each level: none in three profiles out of four, otherwise ice in the
    levels above a random depth, from a trace to more than the pores hold."""
    if rng.random() < 0.75:
        return np.zeros(NLEVSOI)
    frozen = np.arange(NLEVSOI) < rng.integers(1, NLEVSOI + 1)
    return np.where(frozen, rng.uniform(0.0, 1.2, NLEVSOI), 0.0) * watsat * DZ * DENICE


def _pick(rng, *choices):
    """One of ``(probability, draw)`` choices."""
    u = rng.random()
    for probability, draw in choices:
        if u < probability:
            return float(draw())
        u -= probability
    return float(choices[-1][1]())


def surface_fluxes_cases() -> list[dict]:
    """From the water reaching the ground to surface runoff, pond and infiltration.

    Random draws at both model steps: water table from the surface to below the column;
    maximum saturated fraction from 0 to 0.9; rain from none to cloudbursts; evaporation
    from the soil and from the pond (the latter also negative: dew); a pond from none
    (and slightly negative) to deep, covering any fraction of the ground, with a runoff
    threshold from 0 to above the pond; slopes from 0.2 to 30 degrees; soils from sand to
    clay; ice in one profile out of four. Then edge values: at the thresholds of the pond's
    runoff, of its connected fraction (0.4) and of the smallest runoff kept (1e-8)."""
    rng = np.random.default_rng(SEED + 4)
    cases = []
    for i in range(300):
        props = _soil_properties(*_texture(rng))
        h2osfc = _pick(
            rng,
            (0.35, lambda: 0.0),
            (0.05, lambda: -_log_uniform(rng, 1e-6, 1e-2, 1)[0]),
            (0.60, lambda: _log_uniform(rng, 1e-6, 100.0, 1)[0]),
        )
        slope = _log_uniform(rng, 0.2, 30.0, 1)[0]
        cases.append(
            {
                "dtime": STEPS[i % 2],
                "zwt": _pick(
                    rng,
                    (0.1, lambda: 0.0),
                    (0.75, lambda: rng.uniform(0.0, ZI[-1])),
                    (0.15, lambda: rng.uniform(ZI[-1], 80.0)),
                ),
                "wtfact": _pick(rng, (0.1, lambda: 0.0), (0.9, lambda: rng.uniform(0.0, 0.9))),
                "qflx_rain_plus_snomelt": _pick(
                    rng, (0.3, lambda: 0.0), (0.7, lambda: _log_uniform(rng, 1e-6, 3e-2, 1)[0])
                ),
                "qflx_liqevap_from_top_layer": _pick(
                    rng, (0.4, lambda: 0.0), (0.6, lambda: _log_uniform(rng, 1e-7, 2e-4, 1)[0])
                ),
                "qflx_ev_h2osfc": _pick(
                    rng,
                    (0.4, lambda: 0.0),
                    (0.45, lambda: _log_uniform(rng, 1e-7, 2e-4, 1)[0]),
                    (0.15, lambda: -_log_uniform(rng, 1e-7, 5e-5, 1)[0]),
                ),
                "h2osfc": h2osfc,
                "frac_h2osfc": 0.0 if h2osfc <= 1e-8 else float(rng.uniform(0.0, 1.0)),
                "h2osfc_thresh": _pick(
                    rng,
                    (0.2, lambda: 0.0),
                    (0.4, lambda: rng.uniform(0.0, max(h2osfc, 1e-3))),
                    (0.4, lambda: _log_uniform(rng, 1e-2, 300.0, 1)[0]),
                ),
                "topo_slope": float(slope),
                "h2osoi_ice": _levels(_ice(rng, props["watsat"])),
                "watsat": _levels(props["watsat"]),
                "hksat": _levels(props["hksat"]),
            }
        )

    loam = _soil_properties(40.0, 20.0, 0.0)
    base = {
        "dtime": 1800,
        "zwt": 2.0,
        "wtfact": 0.4,
        "qflx_rain_plus_snomelt": 1.0e-3,
        "qflx_liqevap_from_top_layer": 2.0e-5,
        "qflx_ev_h2osfc": 2.0e-5,
        "h2osfc": 20.0,
        "frac_h2osfc": 0.6,
        "h2osfc_thresh": 5.0,
        "topo_slope": 2.0,
        "h2osoi_ice": _levels(0.0),
        "watsat": _levels(loam["watsat"]),
        "hksat": _levels(loam["hksat"]),
    }
    edges = [
        {},
        {"qflx_rain_plus_snomelt": 0.0, "h2osfc": 0.0, "frac_h2osfc": 0.0},
        {"zwt": 0.0, "wtfact": 0.9},  # the largest saturated fraction
        {"wtfact": 0.0},
        {"h2osfc": 5.0},  # exactly at the runoff threshold
        {"frac_h2osfc": 0.4},  # exactly at the connectivity threshold
        {"frac_h2osfc": 0.4000001},
        {"frac_h2osfc": 1.0, "h2osfc": 200.0},
        {"h2osfc": 5.0001, "frac_h2osfc": 0.41},  # a runoff below the smallest kept
        {"h2osfc": 5.01, "frac_h2osfc": 0.9, "topo_slope": 30.0},  # runoff limited by storage
        {"h2osfc": -1.0e-3, "frac_h2osfc": 0.0},  # a negative pond is refilled
        {"qflx_rain_plus_snomelt": 3.0e-2, "frac_h2osfc": 0.0, "h2osfc": 0.0},  # excess
        {"qflx_rain_plus_snomelt": 3.0e-2, "h2osoi_ice": _levels(0.9 * 0.44 * DZ * DENICE)},
        {"qflx_rain_plus_snomelt": 0.0, "qflx_liqevap_from_top_layer": 2.0e-4, "h2osfc": 0.0,
         "frac_h2osfc": 0.0},  # evaporation alone: a negative input to the soil
        {"h2osfc": 1.0e-3, "frac_h2osfc": 0.05, "qflx_ev_h2osfc": 2.0e-4,
         "qflx_rain_plus_snomelt": 0.0},  # evaporation empties the pond
    ]  # fmt: skip
    for dtime in STEPS:
        cases += [{**base, **edge, "dtime": dtime} for edge in edges]
    return cases


def frac_h2osfc_cases() -> list[dict]:
    """Fraction of the ground under ponded water: ponds from none, through the smallest
    kept (1e-8 mm), to deep; microtopography of slopes from 0.2 to 45 degrees."""
    rng = np.random.default_rng(SEED + 5)
    cases = []
    for i in range(200):
        cases.append(
            {
                "dtime": STEPS[i % 2],
                "h2osfc": _pick(
                    rng,
                    (0.1, lambda: 0.0),
                    (0.1, lambda: _log_uniform(rng, 1e-12, 1e-8, 1)[0]),
                    (0.8, lambda: _log_uniform(rng, 1e-8, 300.0, 1)[0]),
                ),
                "micro_sigma": float(_micro_sigma(_log_uniform(rng, 0.2, 45.0, 1)[0])),
                "h2osoi_liq_top": float(rng.uniform(0.0, 9.0)),
            }
        )
    for h2osfc in (1.0e-8, 1.0000001e-8, 1.0, 1000.0):
        for slope in (0.2, 2.0, 45.0):
            cases.append(
                {
                    "dtime": 1800,
                    "h2osfc": h2osfc,
                    "micro_sigma": float(_micro_sigma(slope)),
                    "h2osoi_liq_top": 3.0,
                }
            )
    return cases


def pond_constants_cases() -> list[dict]:
    """Microtopography and pond threshold from the slope (degrees): CLM's smallest slope
    (0.2) to cliffs; beyond 100 degrees the microtopography falls below the 1e-6 m under
    which CLM sets the threshold to zero."""
    rng = np.random.default_rng(SEED + 6)
    slopes = list(_log_uniform(rng, 0.2, 60.0, 60)) + [0.2, 1.0, 10.0, 90.0, 100.0, 120.0]
    return [{"topo_slope": float(x)} for x in slopes]


def _column_water(rng, watsat, nbedrock: int) -> np.ndarray:
    """Liquid water (kg/m2) of every level: a profile of ``_water_profile`` above the
    bedrock; below it CLM's minimum (0.01), nothing, or a profile as above."""
    liq = _water_profile(rng) * watsat * DZ * 1000.0
    below = np.arange(NLEVSOI) >= nbedrock
    u = rng.random()
    return np.where(below & (u < 0.9), 0.01 if u < 0.6 else 0.0, liq)


def water_table_cases() -> list[dict]:
    """Water table depth from the soil water: profiles of every kind (uniform, layered,
    wetting fronts, saturated below a random level, levels above saturation), ice in one
    profile out of four, bedrock from the second level to below the column; then levels
    exactly at the 90 % saturation that defines the water table."""
    rng = np.random.default_rng(SEED + 7)
    cases = []
    for _ in range(200):
        props = _soil_properties(*_texture(rng))
        nbedrock = NLEVSOI if rng.random() < 0.4 else int(rng.integers(2, NLEVSOI + 1))
        cases.append(
            {
                "nbedrock": nbedrock,
                "h2osoi_liq": _levels(_column_water(rng, props["watsat"], nbedrock)),
                "h2osoi_ice": _levels(_ice(rng, props["watsat"])),
                "watsat": _levels(props["watsat"]),
            }
        )
    # a water table inside the column: saturated below a level, any wetness above it,
    # in half of the cases with ice making up part of the saturation
    for _ in range(100):
        props = _soil_properties(*_texture(rng))
        nbedrock = NLEVSOI if rng.random() < 0.5 else int(rng.integers(3, NLEVSOI + 1))
        above = np.arange(NLEVSOI) < rng.integers(1, nbedrock)
        s = np.where(above, rng.uniform(0.05, 0.9, NLEVSOI), rng.uniform(0.9, 1.05, NLEVSOI))
        ice_share = rng.uniform(0.0, 0.6, NLEVSOI) * (rng.random() < 0.5)
        pores = props["watsat"] * DZ
        cases.append(
            {
                "nbedrock": nbedrock,
                "h2osoi_liq": _levels(s * (1.0 - ice_share) * pores * 1000.0),
                "h2osoi_ice": _levels(s * ice_share * pores * DENICE),
                "watsat": _levels(props["watsat"]),
            }
        )

    watsat = 0.45
    for nbedrock in (NLEVSOI, 9):
        for s in (
            np.full(NLEVSOI, 1.0),  # saturated throughout
            np.full(NLEVSOI, 0.3),  # dry throughout
            np.full(NLEVSOI, 0.9),  # exactly at the threshold
            np.where(np.arange(NLEVSOI) < 5, 0.5, 0.95),
            np.where(np.arange(NLEVSOI) < 5, 0.9, 0.9000001),
            np.where(np.arange(NLEVSOI) < 1, 0.2, 1.0),  # only the top level unsaturated
            np.where(np.arange(NLEVSOI) < nbedrock - 1, 1.0, 0.5),  # dry at the bedrock
            np.where(np.arange(NLEVSOI) < 5, 1.0, 0.5),  # wet over dry
        ):
            cases.append(
                {
                    "nbedrock": nbedrock,
                    "h2osoi_liq": _levels(s * watsat * DZ * 1000.0),
                    "h2osoi_ice": _levels(0.0),
                    "watsat": _levels(watsat),
                }
            )
    return cases


def renew_condensation_cases() -> list[dict]:
    """Dew and frost added to the top level, sublimation removed from it (never more than
    its ice: CLM stops there)."""
    rng = np.random.default_rng(SEED + 8)
    cases = []
    for i in range(60):
        dtime = STEPS[i % 2]
        ice = _pick(rng, (0.5, lambda: 0.0), (0.5, lambda: rng.uniform(0.0, 8.0)))
        frac = _pick(rng, (0.5, lambda: 0.0), (0.5, lambda: rng.uniform(0.0, 1.0)))
        frost = _pick(rng, (0.6, lambda: 0.0), (0.4, lambda: _log_uniform(rng, 1e-7, 5e-5, 1)[0]))
        available = (ice + (1.0 - frac) * frost * dtime) / dtime
        cases.append(
            {
                "dtime": dtime,
                "h2osoi_liq_top": float(rng.uniform(0.0, 9.0)),
                "h2osoi_ice_top": ice,
                "frac_h2osfc": frac,
                "qflx_liqdew_to_top_layer": _pick(
                    rng, (0.3, lambda: 0.0), (0.7, lambda: _log_uniform(rng, 1e-7, 5e-5, 1)[0])
                ),
                "qflx_soliddew_to_top_layer": frost,
                "qflx_solidevap_from_top_layer": _pick(
                    rng, (0.5, lambda: 0.0), (0.5, lambda: rng.uniform(0.0, 0.9) * available)
                ),
            }
        )
    return cases


def lateral_flow_cases() -> list[dict]:
    """Drainage out of the saturated zone and the corrections after it.

    Random draws at both model steps: water tables from the surface to below the bedrock
    (no drainage) and CLM's deepest (80 m); bedrock from the third level down; slopes from
    0.2 to 30 degrees; soils from sand to clay; water profiles of every kind, with levels
    above saturation (passed upward, to the pond from the top level) and levels below the
    smallest water CLM keeps (refilled from below, the lowest level from above); ice in one
    profile out of four, also more than the top level's pores hold; with and without a
    pond. Then edge cases."""
    rng = np.random.default_rng(SEED + 9)
    cases = []
    for i in range(240):
        props = _soil_properties(*_texture(rng))
        nbedrock = NLEVSOI if rng.random() < 0.4 else int(rng.integers(3, NLEVSOI + 1))
        liq = _column_water(rng, props["watsat"], nbedrock)
        if rng.random() < 0.3:  # some levels with (almost) no water, or less than none
            dry = rng.random(NLEVSOI) < 0.25
            liq = np.where(dry, rng.choice([0.0, 0.004, 0.01, -0.5, 0.02], NLEVSOI), liq)
        cases.append(
            {
                "dtime": STEPS[i % 2],
                "nbedrock": nbedrock,
                "zwt": _pick(
                    rng,
                    (0.05, lambda: 0.0),
                    (0.65, lambda: rng.uniform(0.0, ZI[nbedrock])),
                    (0.2, lambda: rng.uniform(ZI[nbedrock], 12.0)),
                    (0.1, lambda: 80.0),
                ),
                "h2osfc": _pick(rng, (0.6, lambda: 0.0), (0.4, lambda: rng.uniform(0.0, 30.0))),
                "topo_slope": float(_log_uniform(rng, 0.2, 30.0, 1)[0]),
                "h2osoi_liq": _levels(liq),
                "h2osoi_ice": _levels(_ice(rng, props["watsat"])),
                "watsat": _levels(props["watsat"]),
                "bsw": _levels(props["bsw"]),
                "sucsat": _levels(props["sucsat"]),
            }
        )

    # a shallow water table under a steep slope: the drainage of a step empties several
    # thin levels. Every fifth slope is a cliff (82 to 89 degrees) over shallow bedrock:
    # only there does the drainage of a step exceed what the levels above the bedrock give.
    for i in range(60):
        props = _soil_properties(*_texture(rng))
        nbedrock = NLEVSOI if i % 3 and i % 5 else int(rng.integers(3, 8))
        zwt = float(rng.uniform(0.0, min(0.5, ZI[nbedrock])))
        cases.append(
            {
                "dtime": STEPS[i % 2],
                "nbedrock": nbedrock,
                "zwt": zwt,
                "h2osfc": 0.0,
                "topo_slope": float(
                    rng.uniform(82.0, 89.0) if i % 5 == 0 else rng.uniform(15.0, 45.0)
                ),
                "h2osoi_liq": _levels(
                    np.where(Z > zwt, 1.0, rng.uniform(0.3, 0.9)) * props["watsat"] * DZ * 1000.0
                ),
                "h2osoi_ice": _levels(0.0),
                "watsat": _levels(props["watsat"]),
                "bsw": _levels(props["bsw"]),
                "sucsat": _levels(props["sucsat"]),
            }
        )

    loam = _soil_properties(40.0, 20.0, 0.0)
    full = loam["watsat"] * DZ * 1000.0
    level = np.arange(NLEVSOI)
    edges = [
        # (bedrock level, water table, slope, liquid water, ice)
        (NLEVSOI, 3.0, 2.0, np.where(Z > 3.0, 1.0, 0.4) * full, 0.0),
        (NLEVSOI, float(ZI[NLEVSOI]), 2.0, 0.4 * full, 0.0),  # exactly at the bedrock
        (12, float(ZI[12]) + 1e-9, 2.0, 0.4 * full, 0.0),  # just below it: no drainage
        (NLEVSOI, float(ZI[5]), 2.0, np.where(level >= 5, 1.0, 0.4) * full, 0.0),  # interface
        (NLEVSOI, 0.0, 30.0, 1.0 * full, 0.0),  # saturated to the surface, steep
        (NLEVSOI, 8.0, 30.0, np.where(level >= 19, 0.02, 0.3) * full, 0.0),  # runs dry
        (6, 0.3, 30.0, np.where(level >= 6, 0.0, 0.95) * full, 0.0),
        (4, 0.0, 89.0, 1.0 * full, 0.0),  # a cliff over shallow bedrock: the column
        (6, 0.03, 89.0, np.where(Z > 0.03, 1.0, 0.5) * full, 0.0),  # cannot give it all
        (NLEVSOI, 2.0, 2.0, 1.1 * full, 0.0),  # every level above saturation
        (NLEVSOI, 2.0, 2.0, np.where(level == 0, 0.0, 0.5) * full, 0.0),  # dry top level
        (NLEVSOI, 2.0, 2.0, np.where(level == 19, 0.0, 0.5) * full, 0.0),  # dry lowest level
        (NLEVSOI, 2.0, 2.0, np.where(level >= 17, 0.0, 0.011 / (DZ * 1000.0)), 0.0),
        (NLEVSOI, 2.0, 2.0, np.zeros(NLEVSOI), 0.0),  # no water anywhere
        (NLEVSOI, 2.0, 2.0, 0.6 * full, 0.6 * loam["watsat"] * DZ * DENICE),  # ice too much
        (NLEVSOI, 2.0, 2.0, 0.2 * full, np.where(level < 6, 0.5, 0.0) * full),
    ]
    for dtime in STEPS:
        for nbedrock, zwt, slope, liq, ice in edges:
            cases.append(
                {
                    "dtime": dtime,
                    "nbedrock": nbedrock,
                    "zwt": zwt,
                    "h2osfc": 0.0,
                    "topo_slope": slope,
                    "h2osoi_liq": _levels(liq),
                    "h2osoi_ice": _levels(ice),
                    "watsat": _levels(loam["watsat"]),
                    "bsw": _levels(loam["bsw"]),
                    "sucsat": _levels(loam["sucsat"]),
                }
            )
    return cases


class Routine(NamedTuple):
    fortran: str  # Module.F90::Subroutine
    cases: Callable[[], list[dict]]
    file: str | None = None  # the golden file's name, if not test_<Module>_<Subroutine>.json


# The harness's name of each routine (its case file, output file and driver).
ROUTINES = {
    "soil_hk": Routine("SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_hk", soil_hk_cases),
    "soil_suction": Routine(
        "SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_suction", soil_suction_cases
    ),
    "soil_water": Routine("SoilWaterMovementMod.F90::SoilWater", soil_water_cases),
    "soil_water_sequence": Routine(
        "SoilWaterMovementMod.F90::SoilWater",
        soil_water_sequence_cases,
        "test_SoilWaterMovementMod_SoilWater_sequence.json",
    ),
    "surface_fluxes": Routine(
        "HydrologyNoDrainageMod.F90::HydrologyNoDrainage (SetSoilWaterFractions to "
        "TotalSurfaceRunoff)",
        surface_fluxes_cases,
        "test_HydrologyNoDrainageMod_surface_fluxes.json",
    ),
    "frac_h2osfc": Routine("SurfaceWaterMod.F90::UpdateFracH2oSfc", frac_h2osfc_cases),
    "pond_constants": Routine(
        "initVerticalMod.F90::initVertical, "
        "SoilHydrologyInitTimeConstMod.F90::SoilHydrologyInitTimeConst (statements copied)",
        pond_constants_cases,
        "test_SoilHydrologyInitTimeConstMod_pond_constants.json",
    ),
    "water_table": Routine("SoilHydrologyMod.F90::ThetaBasedWaterTable", water_table_cases),
    "renew_condensation": Routine(
        "SoilHydrologyMod.F90::RenewCondensation", renew_condensation_cases
    ),
    "lateral_flow": Routine("SoilHydrologyMod.F90::SubsurfaceLateralFlow", lateral_flow_cases),
}


def golden_file(name: str) -> Path:
    if ROUTINES[name].file:
        return GOLDEN_DIR / ROUTINES[name].file
    module, routine = ROUTINES[name].fortran.split(".F90::")
    return GOLDEN_DIR / f"test_{module}_{routine}.json"


def _namelist_value(value) -> str:
    if isinstance(value, (list, tuple)):
        # four values per line: a namelist value may continue over lines
        items = [_namelist_value(v) for v in value]
        return ",\n   ".join(", ".join(items[i : i + 4]) for i in range(0, len(items), 4))
    # repr of a float has the 17 digits that identify the number; Fortran wants a "d" exponent
    return repr(value).replace("e", "d") if isinstance(value, float) else repr(value)


def write_cases(path: Path, cases: list[dict]) -> None:
    """One namelist group ``&case ... /`` per case."""
    with open(path, "w") as f:
        for case in cases:
            f.write("&case\n")
            for key, value in case.items():
                f.write(f" {key} = {_namelist_value(value)}\n")
            f.write("/\n")


def read_outputs(path: Path) -> list[dict]:
    """The outputs of every case: ``name=value`` scalars, and ``name(j)=value`` lines
    gathered into lists in the order of ``j`` (CLM's 1-based level index). In a sequence,
    the outputs after a line ``step=k`` belong to that model step: the case's outputs are
    then ``{"steps": [{...}, ...]}``."""
    cases: list[dict] = []
    target: dict = {}
    for line in path.read_text().splitlines():
        key, text = (part.strip() for part in line.split("="))
        if key == "case":
            assert int(text) == len(cases) + 1, f"{path}: case {text} out of order"
            cases.append({})
            target = cases[-1]
            continue
        if key == "step":
            steps = cases[-1].setdefault("steps", [])
            assert int(text) == len(steps) + 1, f"{path}: step {text} out of order"
            steps.append({})
            target = steps[-1]
            continue
        value = float(text) if any(ch in text for ch in ".EeNn") else int(text)
        if key.endswith(")"):
            name, index = key[:-1].split("(")
            array = target.setdefault(name, [])
            assert int(index) == len(array) + 1, f"{path}: {key} out of order"
            array.append(value)
        else:
            target[key] = value
    return cases


def write_golden(path: Path, data: dict) -> None:
    """The golden file. Cases of scalars are written one value per line; cases with arrays
    (a column's levels, a sequence's steps) one case per line, to keep the file small."""
    cases = data.pop("cases")
    with_arrays = any(isinstance(v, list) for v in cases[0]["inputs"].values())
    with open(path, "w") as f:
        if not with_arrays:
            json.dump({**data, "cases": cases}, f, indent=1)
        else:
            head = json.dumps(data, indent=1)
            f.write(head[: head.rindex("}")].rstrip() + ',\n "cases": [\n')
            f.write(",\n".join(json.dumps(case, separators=(",", ":")) for case in cases))
            f.write("\n ]\n}")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("routines", nargs="*", default=list(ROUTINES), help="default: all")
    parser.add_argument("--ctsm", type=Path, default=DEFAULT_CTSM, help="the CTSM checkout")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="scratch directory")
    args = parser.parse_args()
    work = args.work_dir.resolve()

    # Every routine's cases are written (a driver without a case file is skipped), so that a
    # run always exercises all the drivers; only the named routines' JSON files are rewritten.
    (work / "cases").mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    cases = {name: routine.cases() for name, routine in ROUTINES.items()}
    for name in ROUTINES:
        write_cases(work / "cases" / f"{name}.nml", cases[name])
        (work / "out" / f"{name}.out").unlink(missing_ok=True)

    build(args.ctsm.resolve(), work, only_harness=True)

    for name in args.routines:
        outputs = read_outputs(work / "out" / f"{name}.out")
        if len(outputs) != len(cases[name]):
            raise SystemExit(f"{name}: {len(cases[name])} cases, {len(outputs)} outputs")
        data = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "executable": f"ctsm_harness:{name}",
            "routine": ROUTINES[name].fortran,
            "ctsm_tag": CTSM_TAG,
            "cases": [{"inputs": i, "outputs": o} for i, o in zip(cases[name], outputs)],
        }
        write_golden(golden_file(name), data)
        print(f"{golden_file(name).name}: {len(outputs)} cases")


if __name__ == "__main__":
    main()
