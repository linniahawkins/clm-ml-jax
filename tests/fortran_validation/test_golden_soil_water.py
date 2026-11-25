"""The Richards solver against CLM's Fortran (CTSM ``ctsm5.4.004``).

Golden data from the harness in ``ctsm_harness/``: whole calls of ``SoilWater`` on idealized
columns (``test_SoilWaterMovementMod_SoilWater.json``) and sequences of calls with the water
carried from step to step (``..._SoilWater_sequence.json``).

All the cases of a model step length run in one compiled, vectorized call. The number of
trial sub-steps must be CLM's exactly. The other outputs must agree to ``REL_TOL``, relative
to each quantity's scale (the largest water of the column; the value itself for the
conductivity and the matric potential; the terms of Darcy's law for a flux).

Some idealized columns are ill-conditioned: a nearly dry clay level beside a saturated one
has matric potentials that differ by twenty orders of magnitude, the matrix of a sub-step
has a condition number up to 1e15, and the solver's answer then depends on the last digit
of its input. For those the Fortran cannot be matched to ``REL_TOL`` by any arithmetic but
its own. The solver's sensitivity is measured with twins (the same case with the water
perturbed by 1e-15 of its value), and a difference from the Fortran is accepted up to
``TWIN_FACTOR`` times the twins' own difference. The share of cases that need this is
bounded (``MAX_SENSITIVE``).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import soil_water  # noqa: E402
from clm_ml.setup.soil import soil_layers  # noqa: E402

pytestmark = pytest.mark.fortran_validation

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_IO"
REL_TOL = 1e-9
TWIN_FACTOR = 20.0  # allowed difference, in units of the twins' difference
N_TWINS = 4
MAX_SENSITIVE = 0.06  # largest share of cases that may differ by more than REL_TOL
E_ICE = 6.0  # the harness's value (ctsm60_params.c251211.nc)
NLEVSOI = 20
PROPERTIES = ("watsat", "hksat", "sucsat", "bsw")

_z, _dz, _ = soil_layers()
Z, DZ = jnp.asarray(_z[:NLEVSOI]), jnp.asarray(_dz[:NLEVSOI])


def _cases(name: str) -> list[dict]:
    with open(GOLDEN_DIR / name) as f:
        return json.load(f)["cases"]


@functools.cache
def _solver(dtime: float, fixed_trials: int | None):
    """The solver for lanes of columns (with ``fixed_trials`` None the loop ends when every
    lane's step is covered)."""

    def column(liq, infl, sink, icefrac, watsat, hksat, sucsat, bsw, nbedrock):
        return soil_water.soil_water_movement(
            liq, infl, sink, icefrac, watsat, hksat, sucsat, bsw, DZ, Z, nbedrock, dtime, E_ICE,
            fixed_trials=fixed_trials,
        )  # fmt: skip

    return jax.jit(jax.vmap(column))


def _solve(inputs: dict, dtime: int, fixed_trials: int | None = None):
    args = [
        jnp.asarray(inputs[k], dtype=jnp.float64)
        for k in ("h2osoi_liq", "qflx_infl", "qflx_rootsoi", "icefrac", *PROPERTIES)
    ]
    return _solver(float(dtime), fixed_trials)(*args, jnp.asarray(inputs["nbedrock"]))


def _stack(rows: list[dict]) -> dict:
    return {k: np.asarray([row[k] for row in rows]) for k in rows[0]}


def _by_step_length(cases: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for case in cases:
        groups.setdefault(case["inputs"]["dtime"], []).append(case)
    return groups


def _fields(result) -> dict:
    """A result's outputs under the Fortran's names."""
    return {
        "h2osoi_liq": np.asarray(result.h2osoi_liq),
        "smp_l": np.asarray(result.smp),
        "hk_l": np.asarray(result.hk),
        "qout": np.asarray(result.qout),
    }


def _scales(golden: dict) -> dict:
    """The scale of each output, per level, that differences are relative to."""
    liq = np.abs(golden["h2osoi_liq"])
    scales = {"h2osoi_liq": np.broadcast_to(liq.max(axis=-1, keepdims=True), liq.shape)}
    if "qout" in golden:
        hk, smp = np.abs(golden["hk_l"]), np.abs(golden["smp_l"])
        below = np.concatenate([smp[:, 1:], smp[:, -1:]], axis=1)
        z = np.asarray(Z)
        dz_nodes = 1.0e3 * np.concatenate([z[1:] - z[:-1], [1.0]])
        scales |= {"smp_l": smp, "hk_l": hk, "qout": hk * (smp + below) / dz_nodes + hk}
    return scales


def _difference(a: dict, b: dict, scales: dict) -> dict:
    """The largest difference of each output in each lane, in units of its scale."""
    tiny = np.finfo(float).tiny
    return {
        k: np.max(np.abs(a[k] - b[k]) / np.maximum(v, tiny), axis=-1) for k, v in scales.items()
    }


def _compare(inputs: dict, expected: dict, dtime: int, what: str) -> None:
    result = _solve(inputs, dtime)
    np.testing.assert_array_equal(
        np.asarray(result.num_substeps), expected["num_substeps"], err_msg=f"{what}: sub-steps"
    )
    got, scales = _fields(result), _scales(expected)
    error = _difference(got, expected, scales)

    rng = np.random.default_rng(0)
    twin = {k: np.zeros_like(v) for k, v in error.items()}
    for _ in range(N_TWINS):
        liq = inputs["h2osoi_liq"]
        perturbed = {**inputs, "h2osoi_liq": liq * (1.0 + 1e-15 * rng.standard_normal(liq.shape))}
        spread = _difference(_fields(_solve(perturbed, dtime)), got, scales)
        twin = {k: np.maximum(twin[k], spread[k]) for k in twin}

    sensitive = np.zeros(len(expected["num_substeps"]), dtype=bool)
    for key in error:
        excess = error[key] - (REL_TOL + TWIN_FACTOR * twin[key])
        lane = int(np.argmax(excess))
        assert excess[lane] <= 0.0, (
            f"{what}: {key} of lane {lane} differs from the Fortran by {error[key][lane]:.2e} "
            f"of its scale; its twins differ by {twin[key][lane]:.2e} "
            f"({int((excess > 0).sum())} lanes fail)"
        )
        sensitive |= error[key] > REL_TOL
    assert (
        sensitive.mean() <= MAX_SENSITIVE
    ), f"{what}: {int(sensitive.sum())} of {sensitive.size} lanes differ by more than {REL_TOL}"


SWEEP = _by_step_length(_cases("test_SoilWaterMovementMod_SoilWater.json"))
SEQUENCES = _by_step_length(_cases("test_SoilWaterMovementMod_SoilWater_sequence.json"))


@pytest.mark.parametrize("dtime", sorted(SWEEP))
def test_a_whole_call_equals_the_fortran(dtime):
    cases = SWEEP[dtime]
    inputs = _stack([case["inputs"] for case in cases])
    expected = _stack([case["outputs"] for case in cases])
    _compare(inputs, expected, dtime, f"dtime {dtime}")
    # the sweep must reach the loop's corners: one trial, and many with rejections
    assert expected["num_substeps"].min() == 1
    assert expected["num_substeps"].max() > 30
    # no water through the bottom: the recharge CLM reports is zero
    assert not np.any(expected["qcharge"])


@pytest.mark.parametrize("dtime", sorted(SWEEP))
def test_the_fixed_count_loop_gives_the_same_values(dtime):
    """Bit for bit, and CLM's number of trials never exceeds the bound."""
    inputs = _stack([case["inputs"] for case in SWEEP[dtime]])
    bound = soil_water.max_richards_trials(dtime)
    ended = _solve(inputs, dtime)
    fixed = _solve(inputs, dtime, fixed_trials=bound)
    assert int(np.max(ended[1])) <= bound
    for a, b in zip(ended, fixed):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _sequence_steps(case: dict) -> tuple[dict, dict]:
    """The inputs and the Fortran's outputs of every step of a sequence, each step starting
    from the Fortran's water at the end of the step before."""
    inp, steps = case["inputs"], case["outputs"]["steps"]
    n = inp["nsteps"]
    liq = np.asarray([inp["h2osoi_liq"]] + [step["h2osoi_liq"] for step in steps[:-1]])
    uptake, fraction = np.asarray(inp["uptake"][:n]), np.asarray(inp["root_fraction"])
    inputs = {
        "h2osoi_liq": liq,
        "qflx_infl": np.asarray(inp["qflx_infl"][:n]),
        "qflx_rootsoi": uptake[:, None] * fraction[None, :],
        "nbedrock": np.full(n, inp["nbedrock"]),
        **{k: np.tile(inp[k], (n, 1)) for k in ("icefrac", *PROPERTIES)},
    }
    return inputs, _stack(steps)


@pytest.mark.parametrize("dtime", sorted(SEQUENCES))
def test_sequences_one_step_ahead_equal_the_fortran(dtime):
    """Every step of every sequence, from the Fortran's state of the step before."""
    pairs = [_sequence_steps(case) for case in SEQUENCES[dtime]]
    inputs = {k: np.concatenate([p[0][k] for p in pairs]) for k in pairs[0][0]}
    expected = {k: np.concatenate([p[1][k] for p in pairs]) for k in pairs[0][1]}
    _compare(inputs, expected, dtime, f"sequences, dtime {dtime}")


@pytest.mark.parametrize("dtime", sorted(SEQUENCES))
def test_sequences_free_running_stay_with_the_fortran(dtime):
    """JAX carries its own water through each sequence. Round-off can grow from step to
    step, and a sub-step decision at a threshold could differ, so the comparison is looser
    than one step ahead: 1e-6 of the column's largest water, the same sub-step counts."""
    for number, case in enumerate(SEQUENCES[dtime]):
        inputs, expected = _sequence_steps(case)
        static = {k: jnp.asarray(inputs[k][0]) for k in ("icefrac", *PROPERTIES)}
        nbedrock = case["inputs"]["nbedrock"]

        def step(liq, forcing, static=static, nbedrock=nbedrock):
            infl, sink = forcing
            out = soil_water.soil_water_movement(
                liq, infl, sink, static["icefrac"], *(static[k] for k in PROPERTIES),
                DZ, Z, nbedrock, float(dtime), E_ICE,
            )  # fmt: skip
            return out.h2osoi_liq, (out.h2osoi_liq, out.num_substeps)

        forcing = (jnp.asarray(inputs["qflx_infl"]), jnp.asarray(inputs["qflx_rootsoi"]))
        _, (liq, nsub) = jax.jit(lambda liq0, f: jax.lax.scan(step, liq0, f))(
            jnp.asarray(inputs["h2osoi_liq"][0]), forcing
        )
        what = f"sequence {number}, dtime {dtime}"
        np.testing.assert_array_equal(np.asarray(nsub), expected["num_substeps"], err_msg=what)
        golden = expected["h2osoi_liq"]
        scale = np.abs(golden).max(axis=-1, keepdims=True)
        assert np.max(np.abs(np.asarray(liq) - golden) / scale) < 1e-6, what
