"""clm_ml.processes.surface_runoff, drainage and hydrology: properties that need no
reference (the comparison with CLM's Fortran is in
``tests/fortran_validation/test_golden_soil_hydrology.py``)."""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import drainage, hydrology, soil_water, surface_runoff  # noqa: E402
from clm_ml.setup.soil import soil_layers  # noqa: E402

NLEVSOI = 20
_z, _dz, _zi = soil_layers()
Z, DZ, ZI = (jnp.asarray(x) for x in (_z[:NLEVSOI], _dz[:NLEVSOI], _zi[: NLEVSOI + 1]))
LEVEL = np.arange(NLEVSOI)
# ctsm60_params.c251211.nc and the CLM6 namelist: e_ice, fff, pc, mu, baseflow_scalar,
# n_baseflow, aq_sp_yield_min, slopebeta, slopemax
PARAMS = jnp.asarray([6.0, 0.5, 0.4, 0.13889, 0.001, 1.0, 0.02, -3.0, 0.4])
SLOPEBETA, SLOPEMAX = -3.0, 0.4
LOAM = jnp.asarray([0.4386, 0.0037, 227.0, 6.09])  # watsat, hksat, sucsat, bsw
ROOTS = jnp.exp(-Z / 0.5) / jnp.sum(jnp.exp(-Z / 0.5))


# ---------------------------------------------------------------------------
# The pond
# ---------------------------------------------------------------------------


def test_pond_fraction_grows_with_the_water_and_small_ponds_go_to_the_soil():
    sigma = surface_runoff.microtopography(2.0, SLOPEBETA, SLOPEMAX)
    water = jnp.asarray([0.0, 5e-9, 1e-3, 1.0, 10.0, 100.0, 1000.0])
    pond = jax.vmap(lambda h: surface_runoff.pond_fraction(h, 3.0, sigma, 1800.0))(water)
    frac = np.asarray(pond.frac_h2osfc)
    assert np.all(frac[:2] == 0.0) and np.all(np.diff(frac[1:]) > 0.0) and frac[-1] > 0.999
    # a pond too small to keep is moved to the top soil level, the rest is untouched
    np.testing.assert_array_equal(np.asarray(pond.h2osfc)[:2], 0.0)
    np.testing.assert_array_equal(np.asarray(pond.h2osfc)[2:], np.asarray(water)[2:])
    np.testing.assert_allclose(np.asarray(pond.h2osoi_liq_top) - 3.0, [0, 5e-9, 0, 0, 0, 0, 0])


def test_pond_threshold_is_the_water_held_at_the_connectivity_fraction():
    """A pond holding the threshold covers the fraction ``pc`` of the ground (to the
    accuracy of the four Newton steps CLM takes for the threshold)."""
    for slope in (0.2, 2.0, 20.0):
        sigma = surface_runoff.microtopography(slope, SLOPEBETA, SLOPEMAX)
        thresh = surface_runoff.pond_threshold(sigma, 0.4)
        frac = surface_runoff.pond_fraction(thresh, 0.0, sigma, 1800.0).frac_h2osfc
        np.testing.assert_allclose(float(frac), 0.4, rtol=1e-6)
    # flat ground is rougher: it holds more water before the ponds connect
    rough, smooth = (
        surface_runoff.pond_threshold(surface_runoff.microtopography(s, SLOPEBETA, SLOPEMAX), 0.4)
        for s in (0.2, 20.0)
    )
    assert float(rough) > float(smooth) > 0.0


def test_surface_fluxes_account_for_all_the_water():
    """Water reaching the ground = runoff + infiltration + evaporation + change of the pond."""
    rng = np.random.default_rng(0)
    n = 500
    rain = np.where(rng.random(n) < 0.7, np.exp(rng.uniform(np.log(1e-6), np.log(3e-2), n)), 0)
    evap, pond_evap = rng.uniform(0, 2e-4, n), rng.uniform(-5e-5, 2e-4, n)
    h2osfc = np.where(rng.random(n) < 0.6, np.exp(rng.uniform(np.log(1e-6), np.log(100), n)), 0)
    frac = np.where(h2osfc > 1e-8, rng.uniform(0, 1, n), 0.0)
    hksat = np.exp(rng.uniform(np.log(1e-5), np.log(1e-1), (n, NLEVSOI)))
    icefrac = np.where(rng.random((n, 1)) < 0.3, rng.uniform(0, 1, (n, NLEVSOI)), 0.0)
    out = jax.vmap(
        lambda *x: surface_runoff.surface_fluxes(*x, 1800.0, *PARAMS[jnp.asarray([1, 0, 2, 3])])
    )(
        *(jnp.asarray(x) for x in (rain, evap, pond_evap, h2osfc, frac, rng.uniform(0, 10, n),
                                   icefrac, hksat, rng.uniform(0, 0.9, n),
                                   rng.uniform(0, 30, n), rng.uniform(0.2, 30, n)))
    )  # fmt: skip
    leaving = out.qflx_surf + out.qflx_infl + (1 - frac) * evap + frac * pond_evap
    residual = (rain - leaving) * 1800.0 - (out.h2osfc - h2osfc)
    assert float(jnp.max(jnp.abs(residual))) < 1e-11
    assert np.all(np.asarray(out.h2osfc) >= 0.0)
    assert np.all(np.asarray(out.qflx_surf) >= 0.0)


# ---------------------------------------------------------------------------
# Water table and drainage
# ---------------------------------------------------------------------------


def _liquid(s):
    return jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float64), (NLEVSOI,)) * LOAM[0] * DZ * 1000.0


def _soil():
    return [jnp.full(NLEVSOI, LOAM[i]) for i in range(4)]


def _water_table(s, nbedrock=NLEVSOI):
    return float(
        drainage.water_table_depth(_liquid(s), jnp.zeros(NLEVSOI), _soil()[0], DZ, Z, ZI, nbedrock)
    )


def test_water_table_lies_between_the_unsaturated_and_the_saturated_level():
    for k in (3, 9, 15):  # levels k and below saturated
        zwt = _water_table(np.where(LEVEL >= k, 1.0, 0.5))
        assert float(Z[k - 1]) < zwt < float(Z[k])
    assert _water_table(0.5) == float(ZI[NLEVSOI])  # no saturated level: at the bottom
    assert _water_table(0.5, nbedrock=8) == float(ZI[8])  # or at the bedrock
    assert _water_table(1.0) == float(ZI[1])  # saturated throughout: bottom of the top level
    # water below the bedrock is not looked at
    assert _water_table(np.where(LEVEL >= 12, 1.0, 0.5), nbedrock=8) == float(ZI[8])


def _drain(s, zwt, nbedrock=NLEVSOI, slope=5.0, h2osfc=0.0):
    watsat, _, sucsat, bsw = _soil()
    return drainage.subsurface_drainage(
        _liquid(s), jnp.zeros(NLEVSOI), zwt, h2osfc, watsat, watsat, sucsat, bsw, DZ, ZI,
        nbedrock, slope, 1800.0, *PARAMS[jnp.asarray([4, 5, 6, 0])],
    )  # fmt: skip


def test_drainage_comes_from_the_saturated_zone_and_lowers_the_water_table():
    s = np.where(Z > 2.0, 1.0, 0.5)
    out = _drain(s, 2.0)
    removed = np.asarray(_liquid(s) - out.h2osoi_liq)
    assert float(out.qflx_drain) > 0.0 and float(out.zwt) > 2.0
    np.testing.assert_allclose(removed.sum(), float(out.qflx_drain) * 1800.0, rtol=1e-12)
    assert np.all(removed[np.asarray(ZI[1:]) <= 2.0] == 0.0)  # nothing from above the table
    # steeper slopes and thicker saturated zones drain faster
    assert float(_drain(s, 2.0, slope=20.0).qflx_drain) > float(out.qflx_drain)
    assert float(_drain(s, 1.0).qflx_drain) > float(out.qflx_drain)
    # a water table below the bedrock: no drainage, nothing changes
    dry = _drain(0.5, 9.0)
    assert float(dry.qflx_drain) == 0.0 and float(dry.zwt) == 9.0
    np.testing.assert_array_equal(np.asarray(dry.h2osoi_liq), np.asarray(_liquid(0.5)))


def test_water_above_saturation_is_passed_up_to_the_pond():
    out = _drain(np.where(LEVEL >= 10, 1.2, 1.0), 9.0)  # the lower half overfull
    excess = float(jnp.sum(_liquid(np.where(LEVEL >= 10, 0.2, 0.0))))
    full = np.asarray(_liquid(1.0))
    np.testing.assert_allclose(np.asarray(out.h2osoi_liq)[1:], full[1:], rtol=1e-12)
    np.testing.assert_allclose(float(out.h2osfc), excess, rtol=1e-9)


def test_levels_keep_a_minimum_of_water_taken_from_their_neighbours():
    s = np.where((LEVEL == 4) | (LEVEL == NLEVSOI - 1), 0.0, 0.5)
    out = _drain(s, 9.0)
    liq = np.asarray(out.h2osoi_liq)
    np.testing.assert_allclose(liq[[4, NLEVSOI - 1]], drainage.WATMIN)
    np.testing.assert_allclose(liq.sum(), float(jnp.sum(_liquid(s))), rtol=1e-13)
    assert float(out.qflx_rsub_sat) == 0.0
    # no water anywhere: the minimum of the lowest level is created, and counted
    empty = _drain(0.0, 9.0)
    np.testing.assert_allclose(
        float(jnp.sum(empty.h2osoi_liq)), -float(empty.qflx_drain) * 1800.0, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# The assembled hydrology
# ---------------------------------------------------------------------------


def _columns(rng, n):
    """Random columns: soils from sand to clay, bedrock at any level from the fourth, any
    wetness, in some a saturated zone, ice in the top levels or a pond."""
    sand = rng.uniform(5, 90, (n, 1))
    clay = rng.uniform(2, 50, (n, 1)) * (100 - sand) / 100
    ones = np.ones((n, NLEVSOI))
    watsat = (0.489 - 0.00126 * sand) * ones
    soil = {
        "watsat": watsat,
        "hksat": 0.0070556 * 10 ** (-0.884 + 0.0153 * sand) * ones,
        "sucsat": 10 * 10 ** (1.88 - 0.0131 * sand) * ones,
        "bsw": (2.91 + 0.159 * clay) * ones,
    }
    nbedrock = np.where(rng.random(n) < 0.5, NLEVSOI, rng.integers(4, NLEVSOI + 1, n))
    s = rng.uniform(0.15, 1.0, (n, 1)) * ones
    saturated_below = (LEVEL[None, :] >= rng.integers(1, NLEVSOI, (n, 1))) & (
        rng.random((n, 1)) < 0.4
    )
    liq = np.where(saturated_below, 1.0, s) * watsat * np.asarray(DZ) * 1000.0
    liq = np.where(LEVEL[None, :] >= nbedrock[:, None], drainage.WATMIN, liq)
    frozen = (rng.random((n, 1)) < 0.2) & (LEVEL[None, :] < 4)
    slope = np.exp(rng.uniform(np.log(0.2), np.log(30.0), n))
    return {
        **soil,
        "nbedrock": nbedrock,
        "h2osoi_liq": liq,
        "h2osoi_ice": np.where(frozen, 0.3 * watsat * np.asarray(DZ) * 917.0, 0.0),
        "h2osfc": np.where(rng.random(n) < 0.3, rng.uniform(0, 40, n), 0.0),
        "topo_slope": slope,
        "wtfact": rng.uniform(0, 0.8, n),
    }


def _run(column, forcing, dtime, params=PARAMS, fixed_trials=None):
    """Steps of the hydrology of one column, the state carried; ``forcing``: per step the
    water reaching the ground, evaporation, dew and total root uptake (mm/s)."""
    c = column
    p = hydrology.HydrologyParameters(*params)
    micro_sigma = surface_runoff.microtopography(c["topo_slope"], p.slopebeta, p.slopemax)
    thresh = surface_runoff.pond_threshold(micro_sigma, p.pc)
    zwt = drainage.water_table_depth(
        c["h2osoi_liq"], c["h2osoi_ice"], c["watsat"], DZ, Z, ZI, c["nbedrock"]
    )

    def step(state, f):
        rain, evap, dew, uptake = f
        out = hydrology.soil_hydrology(
            *state, rain, evap, evap, dew, uptake * ROOTS, c["watsat"], c["hksat"], c["sucsat"],
            c["bsw"], DZ, Z, ZI, c["nbedrock"], c["wtfact"], micro_sigma, thresh,
            c["topo_slope"], dtime, p, fixed_trials=fixed_trials,
        )  # fmt: skip
        return (out.h2osoi_liq, out.h2osoi_ice, out.h2osfc, out.zwt), out

    return jax.lax.scan(step, (c["h2osoi_liq"], c["h2osoi_ice"], c["h2osfc"], zwt), forcing)[1]


def test_the_hydrology_conserves_water_over_two_days():
    """Storms, evaporation, dew and root uptake on 200 random columns, at both step
    lengths: the water balance closes every step. Round-off, except in the rare step where
    the Richards solver's matrix is ill-conditioned (a level dried to the minimum beside a
    wet one), where the error stays far below the 1e-5 mm at which CLM stops."""
    rng = np.random.default_rng(3)
    n, steps = 200, 48
    columns = {k: jnp.asarray(v) for k, v in _columns(rng, n).items()}
    storms = np.exp(rng.uniform(np.log(1e-5), np.log(2e-2), (steps, n)))
    forcing = (
        np.where(rng.random((steps, n)) < 0.25, storms, 0.0),
        rng.uniform(0, 1e-4, (steps, n)),
        np.where(rng.random((steps, n)) < 0.2, 1e-5, 0.0),
        rng.uniform(0, 2e-4, (steps, n)),
    )
    forcing = tuple(jnp.asarray(f) for f in forcing)
    for dtime in (1800.0, 3600.0):
        out = jax.jit(jax.vmap(lambda c, f: _run(c, f, dtime), in_axes=(0, 1)))(columns, forcing)
        error = np.abs(np.asarray(out.balance_error))
        assert np.all(np.isfinite(np.asarray(out.h2osoi_liq)))
        assert error.max() < 1e-7 and np.mean(error < 1e-10) > 0.999, (dtime, error.max())
        # the runs reach every part: runoff, drainage, ponds, many sub-steps
        assert np.mean(np.asarray(out.qflx_surf) > 0) > 0.2
        assert np.mean(np.asarray(out.qflx_drain) > 0) > 0.2
        assert np.mean(np.asarray(out.h2osfc) > 0) > 0.2
        assert int(np.max(out.num_substeps)) > 20
        assert float(np.min(out.h2osoi_liq)) > 0.99 * drainage.WATMIN


def test_hydrology_gradients_are_finite_and_match_finite_differences():
    """Three steps of the hydrology, differentiated (forward and reverse, through the
    fixed-count solver loop) with respect to the nine hydrology parameters, the four soil
    properties, the initial water and pond, the rain, the slope and the largest saturated
    fraction. Cases: dry soil without rain (no pond, no runoff, no drainage: every masked
    branch), a storm that fills a pond, a deep pond running off, a water table with
    drainage, shallow bedrock, ice."""
    trials = soil_water.max_richards_trials(1800.0)

    def outputs(q, s0, h2osfc, rain, evap, nbedrock, slope, ice_share):
        soil = [jnp.full(NLEVSOI, LOAM[i] * q[9 + i]) for i in range(4)]
        column = {
            **dict(zip(("watsat", "hksat", "sucsat", "bsw"), soil)),
            "nbedrock": nbedrock,
            "h2osoi_liq": s0 * LOAM[0] * DZ * 1000.0 * q[13],
            "h2osoi_ice": ice_share * LOAM[0] * DZ * 917.0,
            "h2osfc": h2osfc * q[17],
            "topo_slope": slope * q[15],
            "wtfact": 0.4 * q[16],
        }
        forcing = tuple(jnp.full(3, x) for x in (rain * q[14], evap, 0.0, 5.0e-5))
        out = _run(column, forcing, 1800.0, PARAMS * q[:9], trials)
        state = jnp.concatenate([out.h2osoi_liq[-1], jnp.stack([out.h2osfc[-1], out.zwt[-1]])])
        return jnp.concatenate([state, jnp.stack([out.qflx_surf.sum(), out.qflx_drain.sum()])])

    zeros = jnp.zeros(NLEVSOI)
    cases = {
        "dry, no rain": (0.3, 0.0, 0.0, 2e-5, NLEVSOI, 2.0, zeros),
        "storm on wet soil": (0.85, 0.0, 2e-2, 0.0, NLEVSOI, 2.0, zeros),
        "deep pond running off": (0.7, 60.0, 1e-3, 1e-5, NLEVSOI, 5.0, zeros),
        "water table, drainage": (jnp.where(Z > 1.0, 0.98, 0.5), 0.0, 1e-4, 1e-5, NLEVSOI, 10.0, zeros),
        "shallow bedrock": (jnp.where(Z < 0.4, 0.95, 0.0), 5.0, 5e-3, 0.0, 6, 10.0, zeros),
        "ice in the top levels": (0.4, 0.0, 5e-3, 0.0, NLEVSOI, 2.0, jnp.where(Z < 0.2, 0.4, 0.0)),
    }  # fmt: skip
    value = jax.jit(outputs)
    jac_forward, jac_reverse = jax.jit(jax.jacfwd(outputs)), jax.jit(jax.jacrev(outputs))
    q0 = jnp.ones(18)
    for name, (s0, *rest) in cases.items():
        case = (jnp.broadcast_to(jnp.asarray(s0, dtype=jnp.float64), (NLEVSOI,)), *rest)
        forward = np.asarray(jac_forward(q0, *case))
        reverse = np.asarray(jac_reverse(q0, *case))
        assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse)), name
        np.testing.assert_allclose(forward, reverse, rtol=1e-6, atol=1e-9, err_msg=name)
        scale = 1.0 + float(jnp.max(jnp.abs(value(q0, *case))))
        for k in range(18):
            h = 1e-6
            fd = (value(q0.at[k].add(h), *case) - value(q0.at[k].add(-h), *case)) / (2 * h)
            np.testing.assert_allclose(
                forward[:, k], np.asarray(fd), rtol=2e-3, atol=1e-7 * scale,
                err_msg=f"{name}, input {k}",
            )  # fmt: skip
