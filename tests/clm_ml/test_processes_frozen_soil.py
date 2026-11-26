"""Frozen soil: the freezing and thawing of soil water, the frost table and the perched
water table, the perched zone's drainage and the division of the ground's vapour flux.
Properties that need no reference, and derivatives against finite differences (the
comparison with CLM's Fortran is in ``tests/fortran_validation/test_golden_frozen_soil.py``).
"""

import math

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import drainage, soil, surface_runoff  # noqa: E402
from clm_ml.processes.constants import cpice, cpliq, denice, tfrz  # noqa: E402
from clm_ml.setup.soil import soil_layers  # noqa: E402

NLEVSOI = 20
_z, _dz, _zi = soil_layers()
Z, DZ, ZI = (jnp.asarray(x) for x in (_z[:NLEVSOI], _dz[:NLEVSOI], _zi[: NLEVSOI + 1]))
LEVEL = np.arange(NLEVSOI)
LOAM = dict(watsat=0.4386, hksat=0.0037, sucsat=227.0, bsw=6.09)
CSOL = 2.2e6  # heat capacity of the soil solids (J m-3 K-1)


def _levels(x):
    return jnp.broadcast_to(jnp.asarray(x, dtype=jnp.float64), (NLEVSOI,))


def _fact(liq, ice, dtime, watsat=LOAM["watsat"]):
    """The step over the heat capacity of each level (K m2 W-1)."""
    return dtime / (CSOL * (1.0 - watsat) * DZ + ice * cpice + liq * cpliq)


def _change(t, liq, ice, dtime=1800.0, snow=0.0, soil_=LOAM, dhsdt=0.0, frac_h2osfc=0.0):
    t, liq, ice = _levels(t), _levels(liq), _levels(ice)
    return soil.phase_change(
        t, liq, ice, snow, snow / 250.0, _fact(liq, ice, dtime, soil_["watsat"]), dhsdt,
        frac_h2osfc, _levels(soil_["watsat"]), _levels(soil_["sucsat"]), _levels(soil_["bsw"]),
        DZ, dtime,
    )  # fmt: skip


PORES = LOAM["watsat"] * DZ  # m


# ---------------------------------------------------------------------------
# Freezing and thawing
# ---------------------------------------------------------------------------


def test_supercooled_water_falls_with_temperature_and_is_zero_above_freezing():
    t = jnp.asarray([tfrz + 5.0, tfrz, tfrz - 1e-6, tfrz - 0.1, tfrz - 1.0, tfrz - 10.0, 243.15])
    w = np.asarray(
        jax.vmap(
            lambda x: soil.supercooled_water(
                _levels(x), _levels(0.4386), _levels(227.0), _levels(6.09), DZ
            )[3]
        )(t)
    )
    assert w[0] == 0.0 and w[1] == 0.0
    assert np.all(np.diff(w[2:]) < 0.0) and w[-1] > 0.0
    # at -1 C a loam keeps about a third of its pore space as liquid water
    assert 0.25 < w[4] / float(PORES[3] * 1000.0) < 0.45


def test_phase_change_conserves_water_and_energy():
    """With the heat flux into the soil given (no surface feedback), the heat the levels
    lose or gain equals the latent heat, and every level keeps its water."""
    rng = np.random.default_rng(1)
    n = 400
    t = rng.uniform(tfrz - 15.0, tfrz + 15.0, (n, NLEVSOI))
    s = rng.uniform(0.02, 1.0, (n, NLEVSOI))
    share = np.where(rng.random((n, 1)) < 0.3, 0.0, rng.uniform(0.0, 1.0, (n, NLEVSOI)))
    liq = s * (1.0 - share) * np.asarray(PORES) * 1000.0
    ice = s * share * np.asarray(PORES) * denice
    snow = np.where(rng.random(n) < 0.3, rng.uniform(0.0, 5.0, n), 0.0)
    for dtime in (1800.0, 3600.0):
        out = jax.jit(jax.vmap(lambda a, b, c, d: _change(a, b, c, dtime=dtime, snow=d)))(
            *(jnp.asarray(x) for x in (t, liq, ice, snow))
        )
        water = np.asarray(out.h2osoi_liq + out.h2osoi_ice)
        np.testing.assert_allclose(water, liq + ice, rtol=1e-13, atol=1e-13)
        fact = np.asarray(
            jax.vmap(lambda a, b: _fact(a, b, dtime))(jnp.asarray(liq), jnp.asarray(ice))
        )
        heat = np.sum((np.asarray(out.t_soisno) - t) / fact, axis=1) + np.asarray(out.xmf)
        assert np.max(np.abs(heat)) < 1e-8
        # the snow's water is melted or kept
        np.testing.assert_allclose(
            np.asarray(out.h2osno) + np.asarray(out.qflx_snomelt) * dtime, snow, atol=1e-12
        )
        # afterwards: ice only at or below freezing; below freezing no more liquid than the
        # supercooled water; a level between the two states is at the freezing point
        t1, liq1, ice1 = (np.asarray(x) for x in (out.t_soisno, out.h2osoi_liq, out.h2osoi_ice))
        assert not np.any((t1 > tfrz) & (ice1 > 0.0))
        supercool = np.asarray(
            jax.vmap(
                lambda x: soil.supercooled_water(
                    x, _levels(0.4386), _levels(227.0), _levels(6.09), DZ
                )
            )(jnp.asarray(t1))
        )
        # a level that froze only to the supercooled water of its temperature before the
        # leftover heat cooled it further may keep a little more liquid than its new
        # temperature allows: it freezes in the next step
        froze_out = (np.asarray(out.imelt) == 2) & (t1 < tfrz)
        assert not np.any((t1 < tfrz) & (liq1 > supercool * (1 + 1e-9) + 1e-12) & ~froze_out)
        partial = (np.asarray(out.imelt) > 0) & (ice1 > 0) & (liq1 > supercool + 1e-9)
        assert np.all(t1[partial & ~froze_out] == tfrz)


def test_an_unfrozen_or_a_wholly_frozen_dry_column_is_left_alone():
    warm = _change(285.0, 0.5 * PORES * 1000.0, 0.0)
    np.testing.assert_array_equal(np.asarray(warm.t_soisno), 285.0)
    np.testing.assert_array_equal(np.asarray(warm.h2osoi_liq), np.asarray(0.5 * PORES * 1000.0))
    assert float(warm.xmf) == 0.0 and not np.any(np.asarray(warm.imelt))
    # so little water that all of it stays liquid at -3 C
    cold = _change(tfrz - 3.0, 0.01 * PORES * 1000.0, 0.0)
    np.testing.assert_array_equal(np.asarray(cold.t_soisno), tfrz - 3.0)
    assert float(jnp.sum(cold.h2osoi_ice)) == 0.0


def test_snow_without_layers_is_melted_by_the_top_level_before_its_ice():
    liq, ice = 0.5 * PORES * 1000.0, 0.2 * PORES * denice
    # a little warmth: part of the snow melts, the soil's ice is untouched, the top level
    # is at freezing
    out = _change(jnp.where(LEVEL == 0, tfrz + 0.3, tfrz - 1.0), liq, ice, snow=5.0)
    assert 0.0 < float(out.h2osno) < 5.0
    np.testing.assert_allclose(float(out.h2osoi_ice[0]), float(ice[0]), rtol=1e-12)
    assert float(out.t_soisno[0]) == tfrz
    np.testing.assert_allclose(
        float(out.qflx_snomelt) * 1800.0, 5.0 - float(out.h2osno), rtol=1e-12
    )
    # more warmth: all the snow, then some ice
    out = _change(jnp.where(LEVEL == 0, tfrz + 10.0, tfrz - 1.0), liq, ice, snow=0.05)
    assert float(out.h2osno) == 0.0 and float(out.h2osoi_ice[0]) < float(ice[0])
    # snow on cold ground stays
    out = _change(tfrz - 1.0, liq, ice, snow=5.0)
    assert float(out.h2osno) == 5.0 and float(out.qflx_snomelt) == 0.0


def test_the_freezing_front_follows_the_analytical_solution():
    """Stefan's problem: a wet column at the freezing point whose surface is held at -5 C.
    The frozen depth grows as ``2 lambda sqrt(alpha t)``, with ``lambda exp(lambda^2)
    erf(lambda) = Ste / sqrt(pi)``. The model's heat diffusion (CLM-ml's) with the phase
    change (CLM's) and the thermal properties of frozen soil reproduce it to the accuracy
    of the level spacing. The soil is a coarse sand, which keeps little water unfrozen."""
    watsat, sucsat, bsw, tkmg = 0.40, 10.0, 2.0, 2.0
    surface, dtime, days = tfrz - 5.0, 100.0, 40
    n = 25
    z, dz, zi = (jnp.asarray(x) for x in (_z, _dz, _zi))
    wat = jnp.where(jnp.arange(n) < NLEVSOI, watsat, 0.0)
    liq0 = wat * dz * 1000.0
    args = (dz, z, zi[1:], jnp.full(n, watsat), jnp.full(n, tkmg), jnp.full(n, 0.25),
            jnp.full(n, CSOL), NLEVSOI)  # fmt: skip

    def step(state, _):
        t, liq, ice = state
        thermal = soil.soil_thermal_properties(t, liq, ice, 0.0, 0.0, *args)
        # the heat flux through the half level between the surface and the top node
        gsoi = thermal.thk[0] * (surface - t[0]) / z[0]
        t_solved, _ = soil.soil_temperature(t, thermal.tk, thermal.cv, z, gsoi, dtime)
        out = soil.phase_change(
            t_solved[:NLEVSOI], liq[:NLEVSOI], ice[:NLEVSOI], 0.0, 0.0,
            dtime / thermal.cv[:NLEVSOI], 0.0, 0.0, jnp.full(NLEVSOI, watsat),
            jnp.full(NLEVSOI, sucsat), jnp.full(NLEVSOI, bsw), DZ, dtime,
        )  # fmt: skip
        pad = jnp.zeros(n - NLEVSOI)
        state = (
            jnp.concatenate([out.t_soisno, t_solved[NLEVSOI:]]),
            jnp.concatenate([out.h2osoi_liq, pad]),
            jnp.concatenate([out.h2osoi_ice, pad]),
        )
        energy = gsoi - (jnp.sum(thermal.cv * (state[0] - t)) / dtime + out.xmf)
        return state, (jnp.sum(out.h2osoi_ice / (watsat * DZ * 1000.0) * DZ), energy)

    steps = int(days * 86400 / dtime)
    start = (jnp.full(n, tfrz), liq0, jnp.zeros(n))
    _, (frozen_depth, energy) = jax.jit(lambda s: jax.lax.scan(step, s, None, length=steps))(start)
    assert float(jnp.max(jnp.abs(energy))) < 1e-6  # W m-2, every step

    # the analytical depth, with the frozen soil's conductivity and heat capacity
    k_frozen = tkmg * 2.29**watsat
    c_frozen = CSOL * (1.0 - watsat) + watsat * 1000.0 * cpice
    latent = soil.HFUS * watsat * 1000.0
    stefan = c_frozen * (tfrz - surface) / latent
    lam = 0.3
    for _ in range(60):  # Newton's method for lambda
        f = lam * math.exp(lam**2) * math.erf(lam) - stefan / math.sqrt(math.pi)
        df = (1 + 2 * lam**2) * math.exp(lam**2) * math.erf(lam) + 2 * lam / math.sqrt(math.pi)
        lam -= f / df
    time = dtime * np.arange(1, steps + 1)
    exact = 2.0 * lam * np.sqrt(k_frozen / c_frozen * time)
    got = np.asarray(frozen_depth)
    late = time > 5 * 86400.0
    assert 0.5 < exact[-1] < 1.2
    np.testing.assert_allclose(got[late], exact[late], rtol=0.08)
    # and the depth grows as the square root of time
    np.testing.assert_allclose(got[-1] / got[steps // 4 - 1], 2.0, rtol=0.05)


def _matches_finite_differences(function, x, what, share=0.9, rtol=2e-3):
    """Forward and reverse derivatives of ``function`` at ``x`` are finite and equal, and
    equal central finite differences wherever two step sizes agree (a finite difference
    across a switch of the routine is not a derivative); at least ``share`` of the
    entries must be so comparable."""
    forward = np.asarray(jax.jacfwd(function)(x))
    reverse = np.asarray(jax.jacrev(function)(x))
    assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse)), what
    np.testing.assert_allclose(forward, reverse, rtol=1e-8, atol=1e-11, err_msg=what)
    value = jax.jit(function)
    scale = 1.0 + np.abs(np.asarray(value(x)))[:, None]

    def difference(h):
        columns = []
        for k in range(x.shape[0]):
            step = h * max(1.0, abs(float(x[k])))
            columns.append((value(x.at[k].add(step)) - value(x.at[k].add(-step))) / (2 * step))
        return np.stack([np.asarray(c) for c in columns], axis=1)

    fine, coarse = difference(1e-6), difference(2e-6)
    comparable = np.abs(fine - coarse) <= 1e-4 * (np.abs(fine) + 1e-6 * scale)
    assert comparable.mean() >= share, (what, comparable.mean())
    error = np.abs(forward - fine) - (rtol * np.abs(fine) + 1e-6 * scale)
    assert np.all(error[comparable] <= 0.0), (what, np.max(error[comparable]))


def test_phase_change_derivatives_match_finite_differences():
    """Derivatives of the temperatures, liquid water, ice, snow and latent heat with
    respect to the temperature before, the water, the soil's parameters and the heat
    capacity factor, in every regime: melting in part and wholly, freezing in part and to
    the supercooled water, snow, a surface feedback, levels left alone and levels exactly
    at the freezing point (where the supercooled water's power has a base of zero)."""
    liq, ice = 0.5 * PORES * 1000.0, 0.3 * PORES * denice

    def outputs(q, t, liq, ice, snow, dhsdt, frac):
        out = soil.phase_change(
            t + q[0] - 1.0, liq * q[1], ice * q[2], snow * q[3], snow / 250.0,
            _fact(liq, ice, 1800.0) * q[4], dhsdt * q[5], frac,
            _levels(LOAM["watsat"]) * q[6], _levels(LOAM["sucsat"]) * q[7],
            _levels(LOAM["bsw"]) * q[8], DZ, 1800.0,
        )  # fmt: skip
        return jnp.concatenate(
            [out.t_soisno, out.h2osoi_liq, out.h2osoi_ice,
             jnp.stack([out.h2osno, out.qflx_snomelt, out.xmf])]
        )  # fmt: skip

    profile = jnp.asarray(np.linspace(-4.0, 4.0, NLEVSOI))
    cases = {
        "a thawing front": (tfrz + profile, liq, ice, 0.0, 0.0, 0.0),
        "a freezing front": (tfrz - profile, liq, 0.0 * ice, 0.0, 0.0, 0.0),
        "cold: to the supercooled water": (_levels(tfrz - 20.0), liq, ice, 0.0, 0.0, 0.0),
        "warm: all ice melts": (_levels(tfrz + 15.0), liq, 0.05 * ice, 0.0, 0.0, 0.0),
        "snow melting in part": (tfrz + 0.2 + 0 * profile, liq, ice, 5.0, 0.0, 0.0),
        "snow melting wholly": (tfrz + 8.0 + 0 * profile, liq, ice, 0.05, 0.0, 0.0),
        "a surface feedback under a pond": (tfrz + profile, liq, ice, 0.5, -15.0, 0.4),
        "unfrozen": (_levels(285.0), liq, 0.0 * ice, 0.0, 0.0, 0.0),
        "dry and cold": (_levels(tfrz - 3.0), 0.01 * liq, 0.0 * ice, 0.0, 0.0, 0.0),
    }
    for name, case in cases.items():
        _matches_finite_differences(
            lambda q: outputs(q, *(jnp.asarray(c, dtype=jnp.float64) for c in case)),
            jnp.ones(9), name,
        )  # fmt: skip

    # exactly at the freezing point nothing changes phase and every derivative is finite
    at_freezing = (jnp.full(NLEVSOI, tfrz), liq, ice, 2.0, 0.0, 0.0)
    for jac in (jax.jacfwd, jax.jacrev):
        d = np.asarray(jac(lambda q: outputs(q, *at_freezing))(jnp.ones(9)))
        assert np.all(np.isfinite(d))


def test_the_stabilized_derivative_changes_no_value_and_only_the_supercooled_limit():
    """A level frozen down to its supercooled water: the exact slope of its temperature
    with respect to the temperature before is ``1 - gain`` (the routine takes the
    supercooled water at the temperature before and then warms the level); the stabilized
    rule gives ``1 / (1 + gain)``, the slope of the state whose liquid water is the
    supercooled water of the temperature after. Everywhere else the two are the same."""
    clay = dict(watsat=0.48, sucsat=400.0, bsw=11.7)
    pores = clay["watsat"] * DZ

    def after(t, liq, ice, rule):
        out = soil.phase_change(
            t, liq, ice, 0.0, 0.0, _fact(liq, ice, 1800.0, clay["watsat"]), 0.0, 0.0,
            _levels(clay["watsat"]), _levels(clay["sucsat"]), _levels(clay["bsw"]), DZ,
            1800.0, derivative=rule,
        )  # fmt: skip
        return jnp.concatenate([out.t_soisno, out.h2osoi_liq, out.h2osoi_ice])

    # a frozen clay holding 0.3 % more liquid water than can stay unfrozen at its temperature
    t0 = tfrz - jnp.asarray(np.linspace(0.3, 6.0, NLEVSOI))
    liq = 1.003 * soil.supercooled_water(
        t0, _levels(clay["watsat"]), _levels(clay["sucsat"]), _levels(clay["bsw"]), DZ
    )
    ice = 0.95 * pores * 1000.0 - liq
    exact, ruled = (after(t0, liq, ice, rule) for rule in ("exact", "stabilized"))
    np.testing.assert_array_equal(np.asarray(exact), np.asarray(ruled))
    t1 = np.asarray(exact)[:NLEVSOI]
    assert np.all((t1 > np.asarray(t0)) & (t1 < tfrz))  # frozen to the supercooled water

    def slopes(rule, t):
        return np.diagonal(np.asarray(jax.jacfwd(lambda x: after(x, liq, ice, rule))(t))[:NLEVSOI])

    gain = 1.0 - slopes("exact", t0)
    assert gain[0] > 5.0 and gain[-1] < 1.0  # steep near freezing, mild when cold
    d_ruled = slopes("stabilized", t0)
    assert np.all((d_ruled > 0.0) & (d_ruled < 1.0))
    # 1 / (1 + gain), the gain being that of the temperature after the change (a little
    # larger than before it: the level has warmed toward the freezing point)
    assert np.all(d_ruled < 1.0 / (1.0 + gain)) and np.all(d_ruled > 0.8 / (1.0 + gain))
    # reverse mode agrees, and liquid water + ice keeps a zero derivative
    j = np.asarray(jax.jacrev(lambda x: after(x, liq, ice, "stabilized"))(t0))
    np.testing.assert_allclose(
        j, np.asarray(jax.jacfwd(lambda x: after(x, liq, ice, "stabilized"))(t0)), atol=1e-9
    )
    np.testing.assert_allclose(j[NLEVSOI : 2 * NLEVSOI] + j[2 * NLEVSOI :], 0.0, atol=1e-9)

    # other regimes (melting, freezing limited by the heat, nothing): the same derivatives
    water = 0.8 * pores * 1000.0
    for t, ice_share in ((tfrz + 2.0, 0.3), (tfrz - 2.0, 0.0), (285.0, 0.0)):
        args = (_levels(t), water * (1.0 - ice_share), water * ice_share)
        a, b = (
            np.asarray(jax.jacfwd(lambda x: after(x, *args[1:], rule))(args[0]))
            for rule in ("exact", "stabilized")
        )
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# The frost table, the perched water table and its drainage
# ---------------------------------------------------------------------------


def _perched(t, s, zwt=6.0, ice_share=0.0):
    t, s = _levels(t), _levels(s)
    return tuple(
        float(x)
        for x in drainage.perched_water_table(
            t, s * (1 - ice_share) * PORES * 1000.0, s * ice_share * PORES * denice, zwt,
            _levels(LOAM["watsat"]), DZ, Z, ZI,
        )  # fmt: skip
    )


def test_frost_table_is_the_top_of_the_first_frozen_level_under_thawed_soil():
    warm, cold = tfrz + 5.0, tfrz - 5.0
    # unfrozen soil: CLM puts both tables at the top of the lowest level
    assert _perched(warm, 0.5) == (float(ZI[NLEVSOI - 1]),) * 2
    # frozen from the surface: at the surface, nothing perched
    assert _perched(cold, 0.95) == (0.0, 0.0)
    assert _perched(np.where(LEVEL < 3, cold, warm), 0.95) == (0.0, 0.0)
    # thawed over frozen: at the top of the first frozen level; a level exactly at freezing
    # counts as frozen
    for k in (1, 8, 19):
        assert _perched(np.where(LEVEL < k, warm, cold), 0.5)[0] == float(ZI[k])
        assert _perched(np.where(LEVEL < k, warm, tfrz), 0.5)[0] == float(ZI[k])
    # a frozen band: its top
    assert _perched(np.where((LEVEL >= 5) & (LEVEL < 9), cold, warm), 0.5)[0] == float(ZI[5])


def test_perched_water_table_lies_in_the_wet_soil_above_the_frost_table():
    warm, cold = tfrz + 5.0, tfrz - 5.0
    t = np.where(LEVEL < 10, warm, cold)
    frost = float(ZI[10])
    # dry soil above the frost table, or a dry frozen level: nothing perched
    assert _perched(t, 0.5) == (frost, frost)
    assert _perched(t, np.where(LEVEL < 10, 0.97, 0.5)) == (frost, frost)
    # levels 6 and below wet: between the nodes of levels 5 and 6
    frost_table, perched = _perched(t, np.where(LEVEL < 6, 0.5, 0.97))
    assert frost_table == frost and float(Z[5]) < perched < float(Z[6])
    # ice counts as water: frozen levels wet with ice give the same
    iced = _perched(t, np.where(LEVEL < 6, 0.5, 0.97), ice_share=np.where(LEVEL < 10, 0.0, 0.8))
    np.testing.assert_allclose(iced[1], perched, rtol=1e-12)
    # wet to the surface: within the top level
    assert _perched(t, 0.95 + 0.002 * LEVEL)[1] < float(Z[1])
    # the water table above the frost table: the perched one is the frost table
    assert _perched(t, np.where(LEVEL < 6, 0.5, 0.97), zwt=0.3) == (frost, frost)
    # two equally saturated levels (CLM divides by zero): at the top of the upper one
    assert _perched(t, 0.95) == (frost, 0.0)


def _drain_perched(frost_table, zwt_perched, slope=10.0, scalar=1.0e-5, s=0.95, nbedrock=NLEVSOI):
    liq = _levels(s) * PORES * 1000.0
    out, qflx = drainage.perched_drainage(
        liq, frost_table, zwt_perched, _levels(LOAM["watsat"]), _levels(LOAM["hksat"]),
        _levels(LOAM["sucsat"]), _levels(LOAM["bsw"]), DZ, ZI, nbedrock, slope, 1800.0,
        scalar, 0.02,
    )  # fmt: skip
    return np.asarray(liq - out), float(qflx)


def test_perched_drainage_takes_the_water_from_the_perched_zone():
    frost, perched = float(ZI[10]), float(Z[5])
    removed, qflx = _drain_perched(frost, perched)
    expected = 1.0e-5 * math.sin(math.radians(10.0)) * LOAM["hksat"] * (frost - perched)
    np.testing.assert_allclose(qflx, expected, rtol=1e-12)
    np.testing.assert_allclose(removed.sum(), qflx * 1800.0, rtol=1e-8)
    assert removed[5] > 0 and np.all(removed[:5] == 0) and np.all(removed[10:] == 0)
    # it grows with the slope and with the parameter, and there is none without a zone
    assert _drain_perched(frost, perched, slope=20.0)[1] > qflx
    np.testing.assert_allclose(_drain_perched(frost, perched, scalar=2.0e-5)[1], 2 * qflx)
    assert _drain_perched(frost, frost)[1] == 0.0
    assert _drain_perched(float(ZI[NLEVSOI - 1]), float(ZI[NLEVSOI - 1]))[1] == 0.0
    # the two tables in one level: no level lies between them, nothing is drained
    assert _drain_perched(float(ZI[10]) + 0.02, float(ZI[10]) + 0.01)[1] == 0.0


def test_saturated_fraction_uses_the_perched_water_table_over_a_frost_table():
    plain = float(surface_runoff.saturated_fraction(3.0, 0.4, 0.5))
    perched = float(surface_runoff.saturated_fraction(0.4, 0.4, 0.5))
    f = surface_runoff.saturated_fraction
    assert float(f(3.0, 0.4, 0.5, 1.0, 0.4)) == perched  # frost table between the two
    assert float(f(3.0, 0.4, 0.5, 1.0, 1.0)) == plain  # nothing perched
    assert float(f(0.5, 0.4, 0.5, 1.0, 0.4)) != perched  # water table above the frost table
    assert perched > plain


def test_ground_vapor_fluxes_add_up_and_sublimation_stays_within_the_ice():
    rng = np.random.default_rng(5)
    n = 500
    flux = np.where(rng.random(n) < 0.6, 1, -1) * np.exp(rng.uniform(np.log(1e-8), np.log(3e-4), n))
    liq = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0, 9, n))
    ice = np.where(rng.random(n) < 0.4, 0.0, np.exp(rng.uniform(np.log(1e-6), np.log(8), n)))
    t = rng.uniform(tfrz - 10, tfrz + 10, n)
    frac = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 1, n))
    out = surface_runoff.ground_vapor_fluxes(
        *(jnp.asarray(x) for x in (flux, liq, ice, t, frac)), 1800.0
    )
    liqevap, solidevap, liqdew, soliddew = (np.asarray(x) for x in out)
    np.testing.assert_allclose(liqevap + solidevap - liqdew - soliddew, flux, rtol=1e-12)
    assert np.all(solidevap * 1800.0 * (1 - frac) <= ice * (1 + 1e-12))
    assert np.all(soliddew[t >= tfrz] == 0) and np.all(liqdew[t < tfrz] == 0)
    assert np.all(solidevap[ice == 0] == 0)


def test_frost_table_perched_table_and_drainage_derivatives_match_finite_differences():
    """Derivatives with respect to the water, the temperatures, the soil's parameters, the
    slope and the hydrology parameters. The frost table is a level's top, so its derivative
    is zero; the perched water table's is the interpolation's between two nodes."""
    warm, cold = tfrz + 5.0, tfrz - 5.0
    cases = {
        "a perched zone": (np.where(LEVEL < 10, warm, cold), np.where(LEVEL < 6, 0.5, 0.97), 6.0),
        "wet to the surface": (np.where(LEVEL < 10, warm, cold), 0.93 + 0.003 * LEVEL, 6.0),
        "nothing perched": (np.where(LEVEL < 10, warm, cold), _levels(0.5), 6.0),
        "unfrozen": (_levels(warm), np.where(LEVEL < 6, 0.5, 0.97), 6.0),
        "frozen from the surface": (_levels(cold), _levels(0.95), 6.0),
        "equally saturated levels": (np.where(LEVEL < 10, warm, cold), _levels(0.95), 6.0),
    }

    def outputs(q, t, s, zwt):
        watsat = _levels(LOAM["watsat"]) * q[2]
        liq = s * 0.7 * PORES * 1000.0 * q[0]
        ice = s * 0.3 * PORES * denice * q[1]
        frost_table, zwt_perched = drainage.perched_water_table(
            t + q[3] - 1.0, liq, ice, zwt * q[4], watsat, DZ, Z, ZI
        )
        liq_after, qflx = drainage.perched_drainage(
            liq, frost_table, zwt_perched, watsat, _levels(LOAM["hksat"]) * q[5],
            _levels(LOAM["sucsat"]) * q[6], _levels(LOAM["bsw"]) * q[7], DZ, ZI, NLEVSOI,
            10.0 * q[8], 1800.0, 1.0e-5 * q[9], 0.02 * q[10],
        )  # fmt: skip
        fsat = surface_runoff.saturated_fraction(
            zwt * q[4], 0.4, 0.5 * q[11], frost_table, zwt_perched
        )
        return jnp.concatenate([jnp.stack([frost_table, zwt_perched, 1e9 * qflx, fsat]), liq_after])

    for name, (t, s, zwt) in cases.items():
        _matches_finite_differences(
            lambda q: outputs(q, jnp.asarray(t), jnp.asarray(s), zwt), jnp.ones(12), name
        )


def test_ground_vapor_flux_derivatives_match_finite_differences():
    def outputs(q, flux, liq, ice, t, frac):
        return (
            jnp.stack(
                surface_runoff.ground_vapor_fluxes(
                    flux * q[0], liq * q[1], ice * q[2], t, frac * q[3], 1800.0
                )
            )
            * 1e6
        )

    cases = {
        "evaporation from liquid and ice": (2e-5, 4.0, 2.0, tfrz - 1.0, 0.2),
        "sublimation limited by the ice": (2e-4, 4.0, 0.01, tfrz - 1.0, 0.0),
        "no water in the level": (2e-5, 0.0, 0.0, tfrz + 1.0, 0.0),
        "frost": (-2e-5, 4.0, 2.0, tfrz - 1.0, 0.3),
        "dew": (-2e-5, 4.0, 0.0, tfrz + 1.0, 0.3),
        "wholly ponded": (2e-5, 4.0, 2.0, tfrz - 1.0, 1.0),
    }
    for name, case in cases.items():
        _matches_finite_differences(lambda q: outputs(q, *case), jnp.ones(4), name)
