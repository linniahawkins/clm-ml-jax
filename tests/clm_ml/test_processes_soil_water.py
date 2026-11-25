"""clm_ml.processes.soil_water."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import soil_water  # noqa: E402

# relative saturation from CLM's dry bound to saturated; a loam
S = jnp.array([0.01, 0.05, 0.3, 0.75, 1.0])
HKSAT, SUCSAT, BSW = 0.0035, 230.0, 6.9


def test_conductivity_and_suction_at_saturation_are_the_saturated_values():
    hk, _ = soil_water.soil_hk(1.0, 1.0, HKSAT, BSW)
    smp, _ = soil_water.soil_suction(1.0, SUCSAT, BSW)
    assert float(hk) == HKSAT
    assert float(smp) == -SUCSAT


def test_hand_coded_derivatives_are_the_derivatives():
    """CLM's hand-coded slopes with respect to saturation equal automatic differentiation."""
    _, dhkds = soil_water.soil_hk(S, 0.5, HKSAT, BSW)
    _, dsmpds = soil_water.soil_suction(S, SUCSAT, BSW)
    auto_hk = jax.vmap(jax.grad(lambda s: soil_water.soil_hk(s, 0.5, HKSAT, BSW)[0]))(S)
    auto_smp = jax.vmap(jax.grad(lambda s: soil_water.soil_suction(s, SUCSAT, BSW)[0]))(S)
    np.testing.assert_allclose(np.asarray(dhkds), np.asarray(auto_hk), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(dsmpds), np.asarray(auto_smp), rtol=1e-12)


def test_gradients_with_respect_to_soil_parameters_match_finite_differences():
    """Including the exponent ``bsw``, at the dry bound and at saturation."""

    def outputs(p):
        hk, dhkds = soil_water.soil_hk(S, 0.5, p[0], p[2])
        smp, dsmpds = soil_water.soil_suction(S, p[1], p[2])
        # logarithms: the values span many orders of magnitude over the saturations
        return jnp.log(jnp.concatenate([hk, dhkds, -smp, dsmpds]))

    p0 = jnp.array([HKSAT, SUCSAT, BSW])
    jac = np.asarray(jax.jacfwd(outputs)(p0))
    assert np.all(np.isfinite(jac))
    for k in range(3):
        h = 1e-6 * float(p0[k])
        fd = (outputs(p0.at[k].add(h)) - outputs(p0.at[k].add(-h))) / (2 * h)
        np.testing.assert_allclose(jac[:, k], np.asarray(fd), rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# The Richards solver: properties that need no reference
# ---------------------------------------------------------------------------

from clm_ml.setup.soil import soil_layers  # noqa: E402

NLEVSOI = 20
_z, _dz, _ = soil_layers()
Z, DZ = jnp.asarray(_z[:NLEVSOI]), jnp.asarray(_dz[:NLEVSOI])
LOAM = {"watsat": 0.4386, "hksat": 0.0037, "sucsat": 227.0, "bsw": 6.09}
E_ICE = 6.0
ROOTS = 1.0e-4 * jnp.exp(-Z / 0.5) / jnp.sum(jnp.exp(-Z / 0.5))  # 0.36 mm/h in all


def _levels(x):
    return jnp.broadcast_to(jnp.asarray(x, dtype=jnp.float64), (NLEVSOI,))


def _liquid(s, watsat=LOAM["watsat"]):
    """Liquid water (kg/m2) of each level at relative saturation ``s``."""
    return _levels(s) * _levels(watsat) * DZ * 1000.0


def _solve(liq, infl=0.0, sink=0.0, ice=0.0, nbedrock=NLEVSOI, dtime=1800.0, soil=LOAM, **kw):
    return soil_water.soil_water_movement(
        liq, infl, _levels(sink), _levels(ice),
        *(_levels(soil[k]) for k in ("watsat", "hksat", "sucsat", "bsw")),
        DZ, Z, nbedrock, dtime, E_ICE, **kw,
    )  # fmt: skip


def _longest_trial_sequence(dtime: float, dtmin: float = 60.0) -> int:
    """The most trials the solver's sub-step rules allow, by trying every outcome of every
    trial (rejected; accepted and the next doubled; accepted and the next kept)."""
    import functools

    @functools.cache
    def longest(dtsub, dtdone):
        best = 1 + longest(max(dtsub / 2.0, dtmin), dtdone) if dtsub > dtmin else 0
        done = dtdone + dtsub
        if abs(dtime - done) < 1e-8:
            return max(best, 1)
        for following in (dtsub * 2.0, dtsub):
            best = max(best, 1 + longest(min(following, dtime - done), done))
        return best

    return longest(float(dtime), 0.0)


def test_trial_bound_covers_every_sequence_of_sub_steps():
    for dtime in (59, 60, 61, 300, 900, 1000, 1800, 3600):
        longest = _longest_trial_sequence(dtime)
        bound = soil_water.max_richards_trials(dtime)
        assert longest <= bound <= longest + 1, (dtime, longest, bound)
    assert soil_water.max_richards_trials(1800) == 64
    assert soil_water.max_richards_trials(3600) == 125


def test_water_is_conserved_and_levels_below_bedrock_keep_theirs():
    """Change of water above the bedrock = (infiltration - root uptake there) x step."""
    cases = [
        (_liquid(0.3), 8.0e-3, ROOTS, 0.0, NLEVSOI),  # heavy rain, halved sub-steps
        (_liquid(jnp.where(Z < 0.1, 0.9, 0.05)), -5.0e-5, ROOTS, 0.0, NLEVSOI),  # evaporation
        (_liquid(0.6), 2.0e-3, ROOTS, jnp.where(Z < 0.2, 0.5, 0.0), 6),  # bedrock, ice
        (_liquid(1.0), 1.0e-3, 0.0, 0.0, 12),  # saturated: water above saturation
    ]
    for liq, infl, sink, ice, nbedrock in cases:
        out = _solve(liq, infl, sink, ice, nbedrock)
        above = np.arange(NLEVSOI) < nbedrock
        change = float(jnp.sum((out.h2osoi_liq - liq)[above]))
        expected = (infl - float(jnp.sum(_levels(sink)[above]))) * 1800.0
        np.testing.assert_allclose(change, expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(np.asarray(out.h2osoi_liq)[~above], np.asarray(liq)[~above])


def test_hydrostatic_equilibrium_stays_at_rest():
    """Matric potential plus height the same at every level (a water table at 10 m, below
    the column): no flux, one sub-step. (With saturated levels in the column CLM's solver
    is not at rest: saturated soil has no potential gradient here, so gravity moves water
    down until other routines pass the excess back up.)"""
    s = ((LOAM["sucsat"] + (10.0 - Z) * 1000.0) / LOAM["sucsat"]) ** (-1.0 / LOAM["bsw"])
    out = _solve(_liquid(s))
    assert int(out.num_substeps) == 1
    np.testing.assert_allclose(np.asarray(out.h2osoi_liq), np.asarray(_liquid(s)), rtol=1e-12)
    assert float(jnp.max(jnp.abs(out.qout))) < 1e-12 * LOAM["hksat"]


def test_a_closed_column_redistributes_and_keeps_its_water():
    liq0 = _liquid(jnp.where(Z < 0.3, 0.95, 0.3))
    step = jax.jit(lambda liq: _solve(liq).h2osoi_liq)
    liq = liq0
    for _ in range(100):
        liq = step(liq)
    np.testing.assert_allclose(float(jnp.sum(liq)), float(jnp.sum(liq0)), rtol=1e-12)
    s = np.asarray(liq / _liquid(1.0))
    wet = np.asarray(Z) < 0.3
    assert np.all(s[wet] < 0.8)  # the wet levels have drained
    assert s[~wet][0] > 0.4 and np.all(s[~wet] >= 0.3)  # into the levels below


def test_steady_infiltration_approaches_unit_gradient_flow():
    """Behind the wetting front the flux equals the infiltration and the conductivity."""
    s_wet = 0.9
    infl = LOAM["hksat"] * s_wet ** (2.0 * LOAM["bsw"] + 3.0)
    step = jax.jit(lambda liq: _solve(liq, infl))
    liq = _liquid(0.5)
    for _ in range(300):
        out = step(liq)
        liq = out.h2osoi_liq
    np.testing.assert_allclose(np.asarray(out.qout[:8]), infl, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(liq / _liquid(1.0))[:8], s_wet, rtol=1e-5)


def test_sub_steps_converge_to_an_independent_integrator():
    """The same fluxes integrated by SciPy's Radau method with tight tolerances: the
    solver's difference from it falls in proportion to the sub-step (first order)."""
    integrate = pytest.importorskip("scipy.integrate")
    infl, liq0 = 2.0e-3, _liquid(0.3)

    @jax.jit
    def tendency(liq):
        vwc = jnp.maximum(liq, 1.0e-6) / (DZ * 1000.0)
        soil = [_levels(LOAM[k]) for k in ("watsat", "hksat", "sucsat", "bsw")]
        properties = soil_water.hydraulic_properties(vwc, _levels(0.0), *soil, E_ICE, NLEVSOI)
        fluxes = soil_water.moisture_fluxes(properties, infl, soil[0], Z, NLEVSOI)
        return fluxes.qin - fluxes.qout

    jacobian = jax.jit(jax.jacfwd(tendency))
    reference = integrate.solve_ivp(
        lambda t, y: np.asarray(tendency(y)), (0.0, 1800.0), np.asarray(liq0), method="Radau",
        jac=lambda t, y: np.asarray(jacobian(y)), rtol=1e-10, atol=1e-10,
    ).y[:, -1]  # fmt: skip

    def difference(dtmin):
        # every sub-step rejected down to dtmin and never doubled: sub-steps of dtmin
        out = _solve(liq0, infl, xtol_upper=0.0, xtol_lower=0.0, dtmin=dtmin)
        return float(np.max(np.abs(np.asarray(out.h2osoi_liq) - reference)))

    coarse, medium, fine = difference(16.0), difference(4.0), difference(1.0)
    assert 3.0 < coarse / medium < 5.0 and 3.0 < medium / fine < 5.0
    assert fine < 5e-4  # kg/m2, of 3.6 infiltrated


def test_levels_below_bedrock_do_not_change_the_result():
    nbedrock = 8
    below = np.arange(NLEVSOI) >= nbedrock
    liq = _liquid(jnp.where(Z < 0.1, 0.9, 0.3))
    ice = jnp.where(Z < 0.05, 0.3, 0.0)
    base = _solve(liq, 5.0e-3, ROOTS, ice, nbedrock)
    other_soil = {
        "watsat": jnp.where(below, 0.9, LOAM["watsat"]),
        "hksat": jnp.where(below, 0.1, LOAM["hksat"]),
        "sucsat": jnp.where(below, 10.0, LOAM["sucsat"]),
        "bsw": jnp.where(below, 12.0, LOAM["bsw"]),
    }
    other = _solve(
        jnp.where(below, 0.0, liq), 5.0e-3, jnp.where(below, 1.0, ROOTS),
        jnp.where(below, 1.0, ice), nbedrock, soil=other_soil,
    )  # fmt: skip
    assert int(base.num_substeps) == int(other.num_substeps) > 1
    for a, b in zip(base[2:], other[2:]):  # conductivity, potential, fluxes
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert not np.any(np.asarray(a)[below])
    np.testing.assert_array_equal(
        np.asarray(base.h2osoi_liq)[~below], np.asarray(other.h2osoi_liq)[~below]
    )


def test_solver_gradients_match_finite_differences():
    """Forward and reverse gradients of the water after a step with respect to the soil
    properties, the initial water, the infiltration, the root uptake and the ice parameter,
    through the fixed-count loop: finite everywhere (levels below bedrock and passes after
    the step is covered included) and equal to central differences. The cases stay clear of
    the points where the solver is not differentiable (water exactly at saturation or at
    the dry bound, an error exactly at a tolerance)."""
    trials = soil_water.max_richards_trials(1800.0)

    @jax.jit
    def water(q, s0, infl, ice, nbedrock):
        # q: multipliers of porosity, conductivity, suction, b, water, infiltration,
        # root uptake, ice parameter
        soil = {k: LOAM[k] * q[i] for i, k in enumerate(("watsat", "hksat", "sucsat", "bsw"))}
        out = soil_water.soil_water_movement(
            _liquid(s0) * q[4], infl * q[5], ROOTS * q[6], ice,
            *(_levels(soil[k]) for k in ("watsat", "hksat", "sucsat", "bsw")),
            DZ, Z, nbedrock, 1800.0, E_ICE * q[7], fixed_trials=trials,
        )  # fmt: skip
        return out.h2osoi_liq, out.num_substeps

    jac_forward = jax.jit(jax.jacfwd(lambda *x: water(*x)[0]))
    jac_reverse = jax.jit(jax.jacrev(lambda *x: water(*x)[0]))

    cases = {
        "one sub-step": (0.5, 1.0e-4, 0.0, NLEVSOI),
        "rain pulse, halved sub-steps": (jnp.where(Z < 0.1, 0.8, 0.3), 8.0e-3, 0.0, NLEVSOI),
        "shallow bedrock with ice": (0.6, 2.0e-3, jnp.where(Z < 0.2, 0.4, 0.0), 6),
        "dry top level": (jnp.where(Z < 0.05, 0.02, 0.4), 1.0e-3, 0.0, NLEVSOI),
        "above saturation": (jnp.where(Z > 1.0, 1.03, 0.97), 1.0e-3, 0.0, 12),
    }
    q0 = jnp.ones(8)
    several = 0
    for name, (s0, infl, ice, nbedrock) in cases.items():
        case = (_levels(s0), infl, _levels(ice), nbedrock)
        substeps = int(water(q0, *case)[1])
        several += substeps > 1
        forward = np.asarray(jac_forward(q0, *case))
        reverse = np.asarray(jac_reverse(q0, *case))
        assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse)), name
        np.testing.assert_allclose(forward, reverse, rtol=1e-9, atol=1e-9, err_msg=name)
        for k in range(8):
            h = 1e-6
            plus, minus = water(q0.at[k].add(h), *case), water(q0.at[k].add(-h), *case)
            assert int(plus[1]) == int(minus[1]) == substeps, name
            fd = np.asarray((plus[0] - minus[0]) / (2 * h))
            np.testing.assert_allclose(
                forward[:, k], fd, rtol=1e-4, atol=1e-5 * np.abs(fd).max() + 1e-9, err_msg=name
            )
    assert several >= 3  # cases with several sub-steps are among them
