"""The whole soil hydrology over many steps against CLM's Fortran (CTSM ``ctsm5.4.004``).

Golden data from the harness in ``ctsm_harness/``: for 14 sequences of 160 steps (seven
scenarios at each model step), CLM's routines called in CLM's order with the state carried
from step to step, driven by time series of the water reaching the ground, evaporation, dew
and root uptake (``test_Hydrology_sequence.json``).

Two comparisons of ``hydrology.soil_hydrology``:

- one step ahead (the acceptance test): every step starts from the Fortran's state of the
  step before, so that differences cannot accumulate and a mismatch names one step;
- free running: JAX carries its own state through the sequence.

In both, the number of trial sub-steps of the Richards solver must be CLM's in every step.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import hydrology  # noqa: E402
from clm_ml.setup.soil import soil_layers  # noqa: E402

pytestmark = pytest.mark.fortran_validation

GOLDEN = Path(__file__).resolve().parent / "golden_IO" / "test_Hydrology_sequence.json"
REL_TOL = 1e-9  # one step ahead
FREE_TOL = 1e-8  # free running: round-off may grow from step to step
ABS_TOL = 1e-15
NLEVSOI = 20
# The harness's parameters: e_ice, fff, pc, mu, baseflow_scalar, n_baseflow, aq_sp_yield_min
PARAMS = hydrology.HydrologyParameters(
    *(jnp.asarray(v) for v in (6.0, 0.5, 0.4, 0.13889, 0.001, 1.0, 0.02))
)
FORCING = (
    "qflx_rain_plus_snomelt",
    "qflx_liqevap_from_top_layer",
    "qflx_ev_h2osfc",
    "qflx_liqdew_to_top_layer",
    "uptake",
)
COMPARED = ("h2osoi_liq", "h2osfc", "zwt", "frac_h2osfc", "fsat", "qflx_infl", "qflx_surf", "qflx_drain")  # fmt: skip

_z, _dz, _zi = soil_layers()
Z, DZ, ZI = (jnp.asarray(x) for x in (_z[:NLEVSOI], _dz[:NLEVSOI], _zi[: NLEVSOI + 1]))

with open(GOLDEN) as _f:
    CASES = json.load(_f)["cases"]


def _step(inputs: dict):
    """One step of a sequence's column: (liquid water, top-level ice, pond, water table),
    forcing -> the new state and the step's results."""
    const = {k: jnp.asarray(inputs[k]) for k in ("watsat", "hksat", "sucsat", "bsw")}
    ice_below, roots = jnp.asarray(inputs["h2osoi_ice"])[1:], jnp.asarray(inputs["root_fraction"])

    def step(state, forcing):
        liq, ice_top, h2osfc, zwt = state
        rain, evap, pond_evap, dew, uptake = forcing
        out = hydrology.soil_hydrology(
            liq, jnp.concatenate([ice_top[None], ice_below]), h2osfc, zwt,
            rain, evap, pond_evap, dew, uptake * roots,
            const["watsat"], const["hksat"], const["sucsat"], const["bsw"], DZ, Z, ZI,
            inputs["nbedrock"], inputs["wtfact"], inputs["micro_sigma"],
            inputs["h2osfc_thresh"], inputs["topo_slope"], float(inputs["dtime"]), PARAMS,
        )  # fmt: skip
        return (out.h2osoi_liq, out.h2osoi_ice[0], out.h2osfc, out.zwt), out

    return step


def _golden(case: dict) -> dict:
    steps = case["outputs"]["steps"]
    return {k: np.array([s[k] for s in steps]) for k in steps[0]}


def _forcing(inputs: dict) -> tuple:
    return tuple(jnp.asarray(inputs[k][: inputs["nsteps"]]) for k in FORCING)


def _assert_close(out, golden: dict, tol: float, what: str) -> None:
    np.testing.assert_array_equal(
        np.asarray(out.num_substeps), golden["num_substeps"], err_msg=f"{what}: sub-steps"
    )
    got = {k: np.asarray(getattr(out, k)) for k in COMPARED}
    got["h2osoi_ice_top"] = np.asarray(out.h2osoi_ice)[:, 0]
    for key, value in got.items():
        expected = golden[key]
        scale = np.abs(expected)
        if key == "h2osoi_liq":  # a level's water against the column's largest
            scale = np.broadcast_to(scale.max(axis=-1, keepdims=True), scale.shape)
        excess = np.abs(value - expected) - (tol * scale + ABS_TOL)
        worst = np.unravel_index(np.argmax(excess), excess.shape)
        assert excess[worst] <= 0.0, (
            f"{what}: {key} at step {worst[0] + 1}: got {value[worst]!r}, Fortran "
            f"{expected[worst]!r} ({int((excess > 0).sum())} values differ)"
        )


@pytest.mark.parametrize("number", range(len(CASES)))
def test_one_step_ahead_equals_the_fortran(number):
    case = CASES[number]
    inputs, golden = case["inputs"], _golden(case)
    before = (
        np.vstack([inputs["h2osoi_liq"], golden["h2osoi_liq"][:-1]]),
        np.r_[inputs["h2osoi_ice"][0], golden["h2osoi_ice_top"][:-1]],
        np.r_[inputs["h2osfc"], golden["h2osfc"][:-1]],
        np.r_[case["outputs"]["zwt_initial"], golden["zwt"][:-1]],
    )
    step = _step(inputs)
    out = jax.jit(jax.vmap(lambda state, forcing: step(state, forcing)[1]))(
        tuple(jnp.asarray(x) for x in before), _forcing(inputs)
    )
    _assert_close(out, golden, REL_TOL, f"sequence {number}, one step ahead")


@pytest.mark.parametrize("number", range(len(CASES)))
def test_free_running_stays_with_the_fortran_and_conserves_water(number):
    case = CASES[number]
    inputs, golden = case["inputs"], _golden(case)
    start = (
        jnp.asarray(inputs["h2osoi_liq"]),
        jnp.asarray(inputs["h2osoi_ice"][0]),
        jnp.asarray(float(inputs["h2osfc"])),
        jnp.asarray(case["outputs"]["zwt_initial"]),
    )
    step = _step(inputs)
    out = jax.jit(lambda state, forcing: jax.lax.scan(step, state, forcing)[1])(
        start, _forcing(inputs)
    )
    _assert_close(out, golden, FREE_TOL, f"sequence {number}, free running")
    assert float(jnp.max(jnp.abs(out.balance_error))) < 1e-10


def test_the_sequences_reach_every_part_of_the_hydrology():
    """Ponds that fill and empty, surface runoff, drainage, a water table that moves through
    the levels, sub-steps from one to CLM's largest counts."""
    golden = [_golden(case) for case in CASES]
    assert sum(np.any(g["h2osfc"] > 0) and np.any(g["h2osfc"] == 0) for g in golden) >= 4
    assert sum(np.sum(g["qflx_surf"] > 0) > 10 for g in golden) >= 8
    assert sum(np.sum(g["qflx_drain"] > 0) > 20 for g in golden) >= 5
    assert sum(np.ptp(g["zwt"]) > 0.15 for g in golden) >= 5
    assert min(g["num_substeps"].min() for g in golden) == 1
    assert max(g["num_substeps"].max() for g in golden) > 60
