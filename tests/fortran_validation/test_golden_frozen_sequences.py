"""A soil that freezes and thaws, over time, against CLM's Fortran (CTSM ``ctsm5.4.004``):
sequences of 240 model steps in which CLM's routines are called in CLM's order (new snow,
the ponded fraction, the heat capacity, the phase change, the division of the ground's
vapour flux, the hydrology with the frost table, the perched water table and its drainage),
the state carried from step to step. The temperature solution, which is the multilayer
canopy's own, is stood for by a relaxation of every level's temperature toward a given one.

Golden data from the harness in ``ctsm_harness/`` (``test_frozen_sequence.pf``; the
scenarios are described in ``generate_golden.py::frozen_sequence_cases``). Two comparisons
with the same steps of ``soil``, ``surface_runoff`` and ``hydrology``:

- *one step ahead*: every step starts from the Fortran's state of the step before, so that
  differences cannot accumulate; every output must equal the Fortran's to ``REL_TOL``;
- *free running*: the state is carried here too; the outputs must stay within
  ``FREE_TOL``, or within a tenth of what a start perturbed by 1e-12 changes them. CLM's
  phase change takes the supercooled water of a freezing level at the temperature before
  the change and so overshoots; while a fine-textured soil freezes this multiplies a
  difference of round-off size by up to 1e10 in a hundred steps (in CLM's Fortran as
  here), so that two correct runs can differ by 1e-6 there.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import hydrology, soil, surface_runoff  # noqa: E402
from clm_ml.processes.constants import tfrz  # noqa: E402
from clm_ml.setup.soil import soil_layers  # noqa: E402

pytestmark = pytest.mark.fortran_validation

GOLDEN = Path(__file__).resolve().parent / "golden_IO" / "test_FrozenSoil_sequence.json"
REL_TOL = 1e-9  # one step ahead
FREE_TOL = 1e-8  # free running: round-off may grow from step to step
ABS_TOL = 1e-13
NLEVSOI = 20
PARAMS = hydrology.HydrologyParameters(
    *(jnp.asarray(v) for v in (6.0, 0.5, 0.4, 0.13889, 0.001, 1.0, 0.02, -3.0, 0.4, 1.0e-5))
)
FORCING = ("qflx_liq_grnd", "qflx_snow_grnd", "qflx_ev", "uptake", "t_target")
STATE = ("t_soisno", "h2osoi_liq", "h2osoi_ice", "h2osno_no_layers", "h2osfc", "zwt",
         "frost_table", "zwt_perched")  # fmt: skip
FLUXES = ("frac_h2osfc", "fsat", "xmf", "qflx_snomelt", "qflx_infl", "qflx_surf",
          "qflx_drain", "qflx_drain_perched", "qflx_ice_runoff_xs",
          "qflx_solidevap_from_top_layer", "qflx_soliddew_to_top_layer")  # fmt: skip

_z, _dz, _zi = (jnp.asarray(x) for x in soil_layers())
Z, DZ, ZI = _z[:NLEVSOI], _dz[:NLEVSOI], _zi[: NLEVSOI + 1]
N_GROUND = _z.shape[0]

CASES = json.loads(GOLDEN.read_text())["cases"] if GOLDEN.exists() else []


def _ground(x, below):
    """A soil-level array extended to the ground levels below them."""
    return jnp.concatenate([x, jnp.full(N_GROUND - NLEVSOI, below, dtype=x.dtype)])


def make_step(inputs: dict, params=PARAMS, fixed_trials=None, derivative="exact"):
    """One step of a sequence's column: (state, forcing) -> (state, results), in the
    harness driver's order (CLM's). ``fixed_trials``: the Richards solver's loop with a
    fixed count (needed for reverse-mode derivatives)."""
    const = {
        k: jnp.asarray(inputs[k], dtype=jnp.float64)
        for k in ("watsat", "hksat", "sucsat", "bsw", "tkmg", "tkdry", "csol", "root_fraction")
    }
    dtime = float(inputs["dtime"])
    heat_profile = inputs["heat_rate"] * jnp.exp(-Z / inputs["heat_depth"])
    # the ground levels below the soil levels: the harness column's placeholders
    watsat_g, tkmg_g = _ground(const["watsat"], 0.45), _ground(const["tkmg"], 3.0)
    tkdry_g, csol_g = _ground(const["tkdry"], 0.25), _ground(const["csol"], 2.2e6)

    def step(state, forcing):
        t, liq, ice, h2osno, h2osfc, zwt, frost_table, zwt_perched = state
        rain, snow, vapour, uptake, t_target = forcing
        h2osno = h2osno + (snow * dtime)
        pond = surface_runoff.pond_fraction(h2osfc, liq[0], inputs["micro_sigma"], dtime)
        liq = jnp.concatenate([pond.h2osoi_liq_top[None], liq[1:]])
        t_grnd = t[0]

        thermal = soil.soil_thermal_properties(
            _ground(t, 285.0), _ground(liq, 0.0), _ground(ice, 0.0), pond.h2osfc, h2osno,
            _dz, _z, _zi[1:], watsat_g, tkmg_g, tkdry_g, csol_g, inputs["nbedrock"],
        )  # fmt: skip
        t_solved = t + heat_profile * (t_target - t)
        fact = dtime / thermal.cv[:NLEVSOI]
        change = soil.phase_change(
            t_solved, liq, ice, h2osno, 0.0, fact, 0.0, pond.frac_h2osfc, const["watsat"],
            const["sucsat"], const["bsw"], DZ, dtime, derivative=derivative,
        )  # fmt: skip
        vapor = surface_runoff.ground_vapor_fluxes(
            vapour, change.h2osoi_liq[0], change.h2osoi_ice[0], t_grnd, pond.frac_h2osfc, dtime
        )
        out = hydrology.soil_hydrology(
            change.h2osoi_liq, change.h2osoi_ice, pond.h2osfc, zwt,
            rain + change.qflx_snomelt, vapor.qflx_liqevap_from_top_layer, vapour,
            vapor.qflx_liqdew_to_top_layer, uptake * const["root_fraction"],
            const["watsat"], const["hksat"], const["sucsat"], const["bsw"], DZ, Z, ZI,
            inputs["nbedrock"], inputs["wtfact"], inputs["micro_sigma"],
            inputs["h2osfc_thresh"], inputs["topo_slope"], dtime, params,
            qflx_soliddew_to_top_layer=vapor.qflx_soliddew_to_top_layer,
            qflx_solidevap_from_top_layer=vapor.qflx_solidevap_from_top_layer,
            t_soisno=change.t_soisno, frost_table=frost_table, zwt_perched=zwt_perched,
            pond=pond, fixed_trials=fixed_trials,
        )  # fmt: skip
        results = {
            **{k: getattr(out, k) for k in out._fields},
            "t_soisno": change.t_soisno,
            "h2osno_no_layers": change.h2osno,
            "xmf": change.xmf,
            "qflx_snomelt": change.qflx_snomelt,
            "qflx_solidevap_from_top_layer": vapor.qflx_solidevap_from_top_layer,
            "qflx_soliddew_to_top_layer": vapor.qflx_soliddew_to_top_layer,
            # heat lost by the levels in the phase change plus the latent heat (W/m2)
            "energy_error": jnp.sum((change.t_soisno - t_solved) / fact) + change.xmf,
        }
        return tuple(results[k] for k in STATE), results

    return step


def initial_state(inputs: dict, zwt_initial) -> tuple:
    unfrozen = ZI[NLEVSOI - 1]  # CLM's frost table and perched water table of unfrozen soil
    return (
        jnp.asarray(inputs["t_soisno"]),
        jnp.asarray(inputs["h2osoi_liq"]),
        jnp.asarray(inputs["h2osoi_ice"]),
        jnp.asarray(float(inputs["h2osno_no_layers"])),
        jnp.asarray(float(inputs["h2osfc"])),
        jnp.asarray(zwt_initial),
        unfrozen,
        unfrozen,
    )


def forcing(inputs: dict) -> tuple:
    return tuple(jnp.asarray(inputs[k][: inputs["nsteps"]]) for k in FORCING)


def _golden(case: dict) -> dict:
    steps = case["outputs"]["steps"]
    return {k: np.array([s[k] for s in steps]) for k in steps[0]}


def _assert_close(out: dict, golden: dict, tol: float, what: str, spread=None) -> int:
    """Every state variable and flux against the Fortran. ``spread``: what a perturbation
    of round-off size changes each of them (free running), a tenth of which is allowed.
    Returns the number of steps in which the Richards solver took a different number of
    sub-steps (its accept-or-halve decision at a threshold; ``tol`` then does not apply to
    that step's water)."""
    flipped = np.asarray(out["num_substeps"]) != golden["num_substeps"]
    for key in STATE + FLUXES:
        value, expected = np.asarray(out[key]), golden[key]
        scale = np.abs(expected)
        if key in ("h2osoi_liq", "h2osoi_ice"):  # a level's water against the column's largest
            scale = np.broadcast_to(scale.max(axis=-1, keepdims=True), scale.shape)
        if key == "xmf":  # a sum of terms of both signs: against the largest heat flux
            scale = np.full_like(scale, max(scale.max(), 1.0))
        allowed = tol * scale + ABS_TOL
        if spread is not None:
            allowed = allowed + 0.1 * spread[key]
        excess = np.abs(value - expected) - allowed
        excess[flipped] = -1.0
        worst = np.unravel_index(np.argmax(excess), excess.shape)
        assert excess[worst] <= 0.0, (
            f"{what}: {key} at step {worst[0] + 1}: got {value[worst]!r}, Fortran "
            f"{expected[worst]!r} ({int((excess > 0).sum())} values differ)"
        )
    return int(flipped.sum())


@pytest.mark.parametrize("number", range(len(CASES)))
def test_one_step_ahead_equals_the_fortran(number):
    case = CASES[number]
    inputs, golden = case["inputs"], _golden(case)
    first = initial_state(inputs, case["outputs"]["zwt_initial"])
    before = tuple(
        jnp.asarray(np.concatenate([np.asarray(first[i])[None], golden[k][:-1]]))
        for i, k in enumerate(STATE)
    )
    step = make_step(inputs)
    out = jax.jit(jax.vmap(lambda state, f: step(state, f)[1]))(before, forcing(inputs))
    flipped = _assert_close(out, golden, REL_TOL, f"sequence {number}, one step ahead")
    assert flipped == 0
    assert float(jnp.max(jnp.abs(out["balance_error"]))) < 1e-10
    assert float(jnp.max(jnp.abs(out["energy_error"]))) < 1e-7


@pytest.mark.parametrize("number", range(len(CASES)))
def test_free_running_stays_with_the_fortran_and_conserves_water_and_energy(number):
    case = CASES[number]
    inputs, golden = case["inputs"], _golden(case)
    step = make_step(inputs)
    start = initial_state(inputs, case["outputs"]["zwt_initial"])
    # the run, and three twins whose temperature and water start 1e-12 away
    rng = np.random.default_rng(number)
    starts = tuple(
        (
            jnp.stack(
                [x] + [x * (1.0 + 1.0e-12 * rng.choice([-1.0, 1.0], np.shape(x))) for _ in range(3)]
            )
            if i < 3
            else jnp.broadcast_to(x, (4,))
        )
        for i, x in enumerate(start)
    )
    runs = jax.jit(jax.vmap(lambda state: jax.lax.scan(step, state, forcing(inputs))[1]))(starts)
    out = {k: v[0] for k, v in runs.items()}
    spread = {
        k: np.max(np.abs(np.asarray(v[1:]) - np.asarray(v[0])), axis=0) for k, v in runs.items()
    }
    flipped = _assert_close(out, golden, FREE_TOL, f"sequence {number}, free running", spread)
    assert flipped == 0
    assert float(jnp.max(jnp.abs(out["balance_error"]))) < 1e-10
    assert float(jnp.max(jnp.abs(out["energy_error"]))) < 1e-7


def test_the_sequences_reach_every_part_of_frozen_soil():
    """Levels that freeze and thaw, in part and wholly; a frost table that moves; perched
    water tables that form, drain and change the saturated fraction; ice runoff; snow that
    lies and melts; frost and sublimation."""
    golden = [_golden(case) for case in CASES]
    assert len(golden) == 16
    ice = [g["h2osoi_ice"] for g in golden]
    assert sum(np.any(np.diff(i, axis=0) > 1e-6) for i in ice) >= 12  # freezing
    assert sum(np.any(np.diff(i, axis=0) < -1e-6) for i in ice) >= 10  # thawing
    assert sum(np.any((g["t_soisno"] == tfrz) & (g["h2osoi_ice"] > 0)) for g in golden) >= 12
    assert sum(len(np.unique(g["frost_table"])) > 3 for g in golden) >= 8
    perched = [g["zwt_perched"] < g["frost_table"] for g in golden]
    assert sum(np.sum(p) > 5 for p in perched) >= 6
    assert sum(np.sum(g["qflx_drain_perched"] > 0) > 5 for g in golden) >= 6
    used = [(g["frost_table"] > g["zwt_perched"]) & (g["frost_table"] <= g["zwt"]) for g in golden]
    assert sum(np.sum(u) > 5 for u in used) >= 4  # the saturated fraction from the perched table
    assert sum(np.any(g["qflx_ice_runoff_xs"] > 0) for g in golden) >= 2
    assert sum(np.any(g["qflx_snomelt"] > 0) for g in golden) >= 4
    assert sum(np.any(g["h2osno_no_layers"] > 0) for g in golden) >= 4
    assert sum(np.any(g["qflx_solidevap_from_top_layer"] > 0) for g in golden) >= 4
    assert sum(np.any(g["qflx_soliddew_to_top_layer"] > 0) for g in golden) >= 4
    assert sum(np.any(g["h2osfc"] > 0) for g in golden) >= 2
    assert sum(np.any(g["qflx_surf"] > 0) for g in golden) >= 4


# Scenarios and steps for the derivatives: a soil freezing; a thaw with a perched water
# table; cycles over frozen ground with rain; frost and sublimation; snow that melts
DERIVATIVE_CASES = {0: (0, 40), 1: (130, 170), 2: (40, 80), 5: (0, 40), 6: (20, 60)}


@pytest.mark.parametrize("number", sorted(DERIVATIVE_CASES))
def test_derivatives_through_a_sequence_match_finite_differences(number):
    """Forty steps of freezing, thawing and hydrology, differentiated forward and in
    reverse with respect to the hydrology parameters ``e_ice`` and
    ``perched_baseflow_scalar``, the soil's ``bsw``, ``sucsat``, ``watsat``, ``hksat`` and
    heat capacity, the state at the start (temperature, liquid water, ice), the
    temperatures the soil is driven toward, the rain and the vapour flux. Compared with
    central finite differences wherever two step sizes agree: the model is piecewise
    smooth (a level starts to freeze, the frost table moves a level, the solver halves a
    sub-step), and a finite difference across such a switch is not a derivative."""
    from clm_ml.processes.soil_water import max_richards_trials

    case = CASES[number]
    inputs, golden = case["inputs"], _golden(case)
    first, last = DERIVATIVE_CASES[number]
    if first == 0:
        start = initial_state(inputs, case["outputs"]["zwt_initial"])
    else:
        start = tuple(jnp.asarray(golden[k][first - 1]) for k in STATE)
    series = tuple(f[first:last] for f in forcing(inputs))
    trials = max_richards_trials(float(inputs["dtime"]))

    def outputs(q):
        scaled = dict(inputs)
        for i, name in enumerate(("bsw", "sucsat", "watsat", "hksat", "csol")):
            scaled[name] = jnp.asarray(inputs[name]) * q[2 + i]
        params = PARAMS._replace(
            e_ice=PARAMS.e_ice * q[0], perched_baseflow_scalar=PARAMS.perched_baseflow_scalar * q[1]
        )
        step = make_step(scaled, params, fixed_trials=trials)
        t, liq, ice, *rest = start
        state = (t + (q[7] - 1.0), liq * q[8], ice * q[9], *rest)
        rain, snow, vapour, uptake, target = series
        _, out = jax.lax.scan(
            step, state, (rain * q[10], snow, vapour * q[11], uptake, target + (q[12] - 1.0))
        )
        return jnp.concatenate(
            [
                out["t_soisno"][-1],
                out["h2osoi_liq"][-1],
                out["h2osoi_ice"][-1],
                jnp.stack(
                    [
                        out["h2osfc"][-1],
                        out["zwt"][-1],
                        out["h2osno_no_layers"][-1],
                        1800.0 * jnp.sum(out["qflx_surf"]),
                        1800.0 * jnp.sum(out["qflx_drain"]),
                        1.0e6 * jnp.sum(out["qflx_drain_perched"]),
                        jnp.sum(out["xmf"]),
                    ]
                ),
            ]
        )

    q0 = jnp.ones(13)
    value = jax.jit(outputs)
    forward = np.asarray(jax.jit(jax.jacfwd(outputs))(q0))
    reverse = np.asarray(jax.jit(jax.jacrev(outputs))(q0))
    assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse))
    np.testing.assert_allclose(forward, reverse, rtol=1e-6, atol=1e-9)
    scale = 1.0 + np.abs(np.asarray(value(q0)))[:, None]

    def difference(h):
        columns = [(value(q0.at[i].add(h)) - value(q0.at[i].add(-h))) / (2 * h) for i in range(13)]
        return np.stack([np.asarray(c) for c in columns], axis=1)

    fine, coarse = difference(1.0e-6), difference(2.0e-6)
    comparable = np.abs(fine - coarse) <= 1e-3 * (np.abs(fine) + 1e-6 * scale)
    assert comparable.mean() > 0.9, comparable.mean()
    error = np.abs(forward - fine) - (5e-3 * np.abs(fine) + 1e-5 * scale)
    assert np.mean(error[comparable] <= 0.0) > 0.995, (
        f"{int(np.sum(error[comparable] > 0))} of {int(comparable.sum())} derivatives differ "
        f"from the finite differences (largest excess {np.max(error[comparable]):.3g})"
    )
    # the derivatives are not trivially zero: the parameters act
    assert np.any(forward[:, 0] != 0.0) and np.any(forward[:, 2:7] != 0.0)


def _sequence_means(number: int, rule: str):
    """Means over a sequence (total ice, temperature of the upper ten levels) as a function
    of a shift of the temperatures the soil is driven toward (K) and of relative changes
    of ``bsw`` and of the heat capacity of the solids."""
    from clm_ml.processes.soil_water import max_richards_trials

    case = CASES[number]
    inputs = case["inputs"]
    start = initial_state(inputs, case["outputs"]["zwt_initial"])
    rain, snow, vapour, uptake, target = forcing(inputs)
    trials = max_richards_trials(float(inputs["dtime"]))

    def means(q):
        scaled = dict(inputs)
        scaled["bsw"] = jnp.asarray(inputs["bsw"]) * (1.0 + q[1])
        scaled["csol"] = jnp.asarray(inputs["csol"]) * (1.0 + q[2])
        step = make_step(scaled, fixed_trials=trials, derivative=rule)
        _, out = jax.lax.scan(step, start, (rain, snow, vapour, uptake, target + q[0]))
        return jnp.stack(
            [jnp.mean(jnp.sum(out["h2osoi_ice"], axis=1)), jnp.mean(out["t_soisno"][:, :10])]
        )

    return means


@pytest.mark.parametrize("number", [1, 4, 9, 12])
def test_the_stabilized_derivative_of_a_freezing_soil_is_bounded_and_follows_finite_changes(
    number,
):
    """Frozen clay under rain and a saturated soil freezing, at both model steps: sequences
    in which the exact derivative of the phase change grows without bound (see the module
    docstring). The stabilized rule (``soil.phase_change``) changes no value; its
    derivatives of the mean ice and temperature with respect to the driving temperature
    and ``bsw`` stay near finite changes of 0.01 to 0.05, which average over the
    overshoots; the exact derivative is orders of magnitude off."""
    exact, stabilized = _sequence_means(number, "exact"), _sequence_means(number, "stabilized")
    q0 = jnp.zeros(3)
    value = jax.jit(exact)
    np.testing.assert_array_equal(np.asarray(value(q0)), np.asarray(jax.jit(stabilized)(q0)))
    ruled = np.asarray(jax.jit(jax.jacfwd(stabilized))(q0))
    reverse = np.asarray(jax.jit(jax.jacrev(stabilized))(q0))
    np.testing.assert_allclose(ruled, reverse, rtol=1e-6, atol=1e-9)
    unruled = np.asarray(jax.jit(jax.jacfwd(exact))(q0))

    def change(i, h):
        e = jnp.zeros(3).at[i].set(h)
        return (np.asarray(value(e)) - np.asarray(value(-e))) / (2 * h)

    for i in (0, 1):  # the driving temperature, bsw
        finite = 0.5 * (change(i, 0.01) + change(i, 0.05))
        # mean ice and mean temperature: within 25 % (and 0.05 K per unit for the weak
        # response of the temperature to bsw)
        assert abs(ruled[0, i] - finite[0]) < 0.25 * abs(finite[0]), (i, ruled[0, i], finite[0])
        assert abs(ruled[1, i] - finite[1]) < 0.25 * abs(finite[1]) + 0.05, (
            i,
            ruled[1, i],
            finite[1],
        )
    assert np.max(np.abs(unruled)) > 10.0 * np.max(np.abs(ruled))
