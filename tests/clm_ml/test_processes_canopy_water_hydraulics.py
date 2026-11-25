"""clm_ml.processes.canopy_water and plant_hydraulics, checked against
what they must conserve."""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import canopy_water, plant_hydraulics  # noqa: E402
from clm_ml.processes.constants import (  # noqa: E402
    dewmx,
    fwet_exponent,
    maximum_leaf_wetted_fraction,
    mmh2o,
)

N_SLOTS = 6  # layer slots
N_SOI = 4  # soil levels
DTIME = 300.0
IN_USE = jnp.ones(N_SLOTS, dtype=bool)
SPVAL = 1.0e36  # what the slots above the layers in use hold
Params = namedtuple(
    "Params",
    "plant_conductance plant_capacitance root_radius root_density root_resistivity sla_top",
)
PARAMS = Params(4.0, 2500.0, 0.29e-3, 0.31e6, 14.0, 0.03)


def _dpai():
    v = np.zeros(N_SLOTS)
    v[1:3] = [1.2, 0.9]
    return jnp.asarray(v)


def test_interception_conserves_water():
    dpai = _dpai()
    bef = jnp.where(dpai > 0, 0.1, 0.0)
    r = canopy_water.interception(5e-3, 1e-3, 1.4, 0.7, dpai, bef, DTIME, IN_USE)
    drip = r.qflx_tflrain + r.qflx_tflsnow - (6e-3 - r.qflx_intr)  # beyond direct throughfall
    stored = float(jnp.sum(r.h2ocan - bef))
    np.testing.assert_allclose(stored + float(drip) * DTIME, float(r.qflx_intr) * DTIME, rtol=1e-12)
    assert np.all(np.asarray(r.h2ocan) <= np.asarray(dewmx * dpai) + 1e-15)  # capacity
    assert np.all(np.asarray(r.h2ocan)[np.asarray(dpai) == 0] == 0.0)


def _with_unused_slots(values, n_unused, fill=SPVAL):
    return jnp.concatenate([jnp.asarray(values), jnp.full(n_unused, fill)])


def test_interception_does_not_depend_on_unused_slots():
    """Rain on the same canopy with 0, 4 and 54 slots above the layers in use: the same
    water on every layer, the same throughfall, and the unused slots untouched."""
    dpai = _dpai()
    bef = jnp.where(dpai > 0, 0.02, 0.0)
    for rain in (2e-4, 5e-3):  # below and above the layers' capacity
        ref = canopy_water.interception(rain, 1e-4, 1.4, 0.7, dpai, bef, DTIME, IN_USE)
        assert float(ref.qflx_tflrain) > 0.0
        for n_unused in (4, 54):
            use = jnp.arange(N_SLOTS + n_unused) < N_SLOTS
            r = canopy_water.interception(
                rain, 1e-4, 1.4, 0.7, _with_unused_slots(dpai, n_unused),
                _with_unused_slots(bef, n_unused), DTIME, use,
            )  # fmt: skip
            np.testing.assert_array_equal(r.h2ocan[:N_SLOTS], ref.h2ocan)
            np.testing.assert_array_equal(r.h2ocan[N_SLOTS:], SPVAL)
            assert float(r.qflx_tflrain) == float(ref.qflx_tflrain)
            assert float(r.qflx_tflsnow) == float(ref.qflx_tflsnow)
            assert float(r.qflx_intr) == float(ref.qflx_intr)


def test_evaporation_does_not_depend_on_unused_slots():
    dpai, fracsun = _dpai(), jnp.full(N_SLOTS, 0.3)
    h2ocan = jnp.where(dpai > 0, 0.1, 0.0)
    ev = jnp.full((N_SLOTS, 2), 1e-5)
    tr = jnp.full((N_SLOTS, 2), -2e-6)
    ref = canopy_water.evaporation(h2ocan, dpai, fracsun, ev, tr, DTIME, IN_USE)
    n_unused = 4
    use = jnp.arange(N_SLOTS + n_unused) < N_SLOTS
    pad2 = lambda v: jnp.concatenate([v, jnp.full((n_unused, 2), SPVAL)])  # noqa: E731
    new = canopy_water.evaporation(
        _with_unused_slots(h2ocan, n_unused), _with_unused_slots(dpai, n_unused),
        _with_unused_slots(fracsun, n_unused), pad2(ev), pad2(tr), DTIME, use,
    )  # fmt: skip
    np.testing.assert_array_equal(new[:N_SLOTS], ref)
    np.testing.assert_array_equal(new[N_SLOTS:], SPVAL)


def test_evaporation_removes_what_the_leaves_evaporate():
    dpai, fracsun = _dpai(), jnp.full(N_SLOTS, 0.3)
    h2ocan = jnp.where(dpai > 0, 0.1, 0.05)
    ev = jnp.full((N_SLOTS, 2), 1e-5)
    new = canopy_water.evaporation(
        h2ocan, dpai, fracsun, ev, jnp.zeros((N_SLOTS, 2)), DTIME, IN_USE
    )
    expected = h2ocan - jnp.where(dpai > 0, 1e-5 * dpai * mmh2o * DTIME, 0.0)
    np.testing.assert_allclose(new, expected, rtol=1e-12)


def test_wetted_fraction_limits():
    dpai = _dpai()
    fwet, fdry = canopy_water.wetted_fraction(dewmx * dpai * 10.0, dpai, 0.5 * dpai)
    assert np.all(np.asarray(fwet)[np.asarray(dpai) > 0] == maximum_leaf_wetted_fraction)
    assert np.all(np.asarray(fdry)[np.asarray(dpai) == 0] == 0.0)


def _implicit_cases():
    """Water (as a fraction of the capacity) and demand (as a multiple of it) over many
    orders of magnitude, on a layer with plant area."""
    rel, dem = np.meshgrid(10.0 ** np.arange(-12.0, 0.5, 1.0), [0.0, 1e-3, 1.0, 30.0, 1e3])
    dpai = jnp.full(rel.size, 0.8)
    return dpai, jnp.asarray(rel.ravel() * dewmx * 0.8), jnp.asarray(dem.ravel() * dewmx * 0.8)


def test_implicit_wetted_fraction_solves_the_water_balance():
    """The water left (by the explicit relation) plus the water evaporated is
    the water there was, so a layer never evaporates more than it holds; at the cap where
    the water allows it; the explicit value without demand; 0 without plant area."""
    dpai, water, demand = _implicit_cases()
    on = jnp.ones(dpai.shape, bool)
    fwet, fdry = canopy_water.wetted_fraction_implicit(water, dpai, 0.5 * dpai, demand, on)
    fwet, water, demand = np.asarray(fwet), np.asarray(water), np.asarray(demand)
    left = dewmx * 0.8 * fwet ** (1.0 / fwet_exponent)
    capped = fwet == maximum_leaf_wetted_fraction
    assert capped.any() and (~capped).any() and np.all(fwet <= maximum_leaf_wetted_fraction)
    np.testing.assert_allclose((left + demand * fwet)[~capped], water[~capped], rtol=1e-10)
    assert np.all((left + demand * fwet)[capped] <= water[capped])
    assert np.all(demand * fwet <= water * (1 + 1e-10))
    np.testing.assert_allclose(fdry, (1.0 - fwet) * 0.5, rtol=1e-14)
    explicit, _ = canopy_water.wetted_fraction(water, dpai, 0.5 * dpai)
    np.testing.assert_allclose(fwet[demand == 0], np.asarray(explicit)[demand == 0], rtol=1e-12)
    assert np.all(fwet <= np.asarray(explicit) * (1 + 1e-12))
    # negative water counts as none; no plant area, or a slot not in use: 0
    dpai0 = _dpai()
    f, d = canopy_water.wetted_fraction_implicit(
        jnp.full(N_SLOTS, -1e-3), dpai0, dpai0, jnp.full(N_SLOTS, 0.1), jnp.ones(N_SLOTS, bool)
    )
    assert np.all(np.asarray(f) == 0.0) and np.all(np.asarray(d)[np.asarray(dpai0) == 0] == 0.0)
    f, _ = canopy_water.wetted_fraction_implicit(
        jnp.full(N_SLOTS, 0.01), dpai0, dpai0, jnp.full(N_SLOTS, 0.1), jnp.zeros(N_SLOTS, bool)
    )
    assert np.all(np.asarray(f) == 0.0)


def test_implicit_wetted_fraction_gradient():
    """The implicit-function gradient against central finite differences, in the water, the
    demand and the plant area; 0 at the cap; finite for a dry layer and without plant area;
    and the forward value is the solver's own (the gradient step adds exactly 0)."""
    dpai, water, demand = _implicit_cases()
    on = jnp.ones(dpai.shape, bool)

    def f(scale):
        w, dm, pai = water * scale[0], demand * scale[1], dpai * scale[2]
        return canopy_water.wetted_fraction_implicit(w, pai, 0.5 * pai, dm, on)[0]

    one = jnp.ones(3)
    jac = np.asarray(jax.jacfwd(f)(one))
    rev = np.asarray(jax.jacrev(f)(one))
    np.testing.assert_allclose(rev, jac, rtol=1e-10, atol=1e-300)
    fwet = np.asarray(f(one))
    capped = fwet == maximum_leaf_wetted_fraction
    h = 1e-6
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        fd = (np.asarray(f(one + e)) - np.asarray(f(one - e))) / (2 * h)
        np.testing.assert_allclose(jac[~capped, k], fd[~capped], rtol=1e-5, atol=1e-12)
    assert np.all(jac[capped] == 0.0) and np.all(np.asarray(f(one))[capped] == 0.05)
    assert np.all(jac[:, 0] >= 0.0) and np.all(jac[:, 1] <= 0.0)  # more water, less demand
    dpai0 = _dpai()
    g = jax.jacrev(
        lambda w: canopy_water.wetted_fraction_implicit(
            w, dpai0, dpai0, jnp.full(N_SLOTS, 0.1), jnp.ones(N_SLOTS, bool)
        )[0]
    )(jnp.zeros(N_SLOTS))
    assert np.all(np.isfinite(np.asarray(g)))


def test_wet_leaf_demand():
    """The water a fully wet layer evaporates in a sub-step: the evaporation term of the
    leaf fluxes; 0 with dew, without plant area and in slots not in use (whose inputs may
    be fill values)."""
    from clm_ml.processes.water_vapor import sat_vap

    dpai, fracsun = _dpai(), jnp.full(N_SLOTS, 0.3)
    tleaf = jnp.stack([jnp.full(N_SLOTS, 300.0), jnp.full(N_SLOTS, 295.0)], axis=1)
    gbv = jnp.full((N_SLOTS, 2), 1.5)
    on = jnp.arange(N_SLOTS) < 2  # slot 2 has plant area but is not in use
    spval = jnp.where(on, 0.0, 1e36)
    d = np.asarray(
        canopy_water.wet_leaf_demand(
            tleaf + spval[:, None], 1500.0 + spval, gbv, fracsun, dpai, 1e5, DTIME, on
        )
    )
    esat = np.asarray(sat_vap(jnp.asarray([300.0, 295.0]))[0])
    one = 1.2 * mmh2o * DTIME * 1.5 * (0.3 * (esat[0] - 1500.0) + 0.7 * (esat[1] - 1500.0)) / 1e5
    np.testing.assert_allclose(d, [0.0, one, 0.0, 0.0, 0.0, 0.0], rtol=1e-12)
    dew = canopy_water.wet_leaf_demand(
        tleaf, jnp.full(N_SLOTS, 1e4), gbv, fracsun, dpai, 1e5, DTIME, on
    )
    assert np.all(np.asarray(dew) == 0.0)


def test_soil_uptake_fractions_sum_to_one_above_bedrock():
    rng = np.random.default_rng(2)
    r = plant_hydraulics.soil_resistance(
        jnp.asarray(rng.uniform(1e-6, 1e-4, N_SOI)), jnp.asarray(rng.uniform(-3e4, -1e3, N_SOI)),
        jnp.full(N_SOI, 0.2), jnp.full(N_SOI, 0.25), jnp.zeros(N_SOI), 3, 1.4, 500.0,
        *PARAMS[2:5],
    )  # fmt: skip
    np.testing.assert_allclose(float(jnp.sum(r.soil_et_loss)), 1.0, rtol=1e-12)
    assert float(r.soil_et_loss[3]) == 0.0  # below bedrock


ROOTS = plant_hydraulics.RootParameters(
    root_radius=PARAMS[2], root_density=PARAMS[3], root_resistivity=PARAMS[4],
    root_to_leaf_ratio=1.39, root_conductance_max=1.037788e-10, root_psi50=-2.6476,
    root_ck=3.95,
)  # fmt: skip
Z_SOI = jnp.asarray([0.1, 0.3, 0.5, 0.7])  # level depths, m


def _soil_columns():
    rng = np.random.default_rng(5)
    hk = jnp.asarray(rng.uniform(1e-7, 1e-4, N_SOI))
    smp = jnp.asarray(rng.uniform(-2.5e5, -1e3, N_SOI))  # mm: to -2.5 MPa
    args = (
        jnp.full(N_SOI, 0.2),
        jnp.asarray([0.4, 0.3, 0.2, 0.1]),
        jnp.zeros(N_SOI),
        3,
        1.4,
        500.0,
    )
    return hk, smp, args


def _by_conductance(scheme, hk, smp, args, ice=None, biomass=500.0):
    dz, rootfr, no_ice, n_bedrock, lai, _ = args
    return plant_hydraulics.root_zone_by_conductance(
        scheme, hk, jnp.full(N_SOI, 5e-5), smp, dz, Z_SOI, rootfr,
        no_ice if ice is None else ice, n_bedrock, lai, 0.5, biomass, ROOTS,
    )  # fmt: skip


def test_soil_weighted_by_conductance_is_the_root_water_potential():
    """The levels act as parallel conductors, each from its own potential: the soil water
    potential is the potential of their common end (the roots) without flow, and each
    level's uptake is its conductance times its potential difference to the roots under
    transpiration."""
    hk, smp, args = _soil_columns()
    r = _by_conductance("clm_ml", hk, smp, args)
    v2 = plant_hydraulics.soil_resistance(hk, smp, *args, *PARAMS[2:5])
    np.testing.assert_allclose(float(r.rsoil), float(v2.rsoil), rtol=1e-14)  # A25 either way
    assert np.all(np.asarray(v2.redistribution) == 0.0)
    psi = np.asarray(smp)[:3] * 1e-3 * plant_hydraulics.HEAD
    resistance = plant_hydraulics.soil_root_resistance(hk, *args[:2], 500.0, *PARAMS[2:5])
    cond = 1.0 / np.asarray(resistance)[:3]
    np.testing.assert_allclose(float(r.rsoil), 1.4 / cond.sum(), rtol=1e-12)
    np.testing.assert_allclose(float(r.psis), (cond * psi).sum() / cond.sum(), rtol=1e-12)
    assert psi.min() < float(r.psis) < psi.max()
    np.testing.assert_allclose(np.asarray(r.soil_et_loss)[:3], cond / cond.sum(), rtol=1e-12)
    assert float(r.soil_et_loss[3]) == 0.0 and float(r.redistribution[3]) == 0.0  # bedrock
    redistribution = np.asarray(r.redistribution)[:3]
    np.testing.assert_allclose(redistribution, cond * (psi - float(r.psis)), rtol=1e-12)
    assert abs(redistribution.sum()) < 1e-12 * np.abs(redistribution).max()
    assert np.all((redistribution > 0) == (psi > float(r.psis)))  # wetter levels give
    # the dry levels count: drier than CLM-ml v2's weighting by possible supply
    assert float(r.psis) < float(v2.psis)

    transpiration = 2.0  # mmol m-2 s-1
    uptake = np.asarray(
        plant_hydraulics.root_water_uptake(r.soil_et_loss, r.redistribution, transpiration, True)
    )
    psi_root = float(r.psis) - transpiration / cond.sum()
    np.testing.assert_allclose(uptake[:3], cond * (psi - psi_root), rtol=1e-12)
    np.testing.assert_allclose(uptake.sum(), transpiration, rtol=1e-12)


def test_soil_weightings_agree_in_a_uniform_soil_and_skip_frozen_levels():
    hk, _, args = _soil_columns()
    smp = jnp.full(N_SOI, -5.0e4)
    r = _by_conductance("clm_ml", hk, smp, args)
    v2 = plant_hydraulics.soil_resistance(hk, smp, *args, *PARAMS[2:5])
    np.testing.assert_allclose(float(r.psis), float(v2.psis), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(r.soil_et_loss), np.asarray(v2.soil_et_loss), rtol=1e-12)
    assert np.abs(np.asarray(r.redistribution)).max() < 1e-12

    hk, smp, args = _soil_columns()
    for scheme in plant_hydraulics.ROOT_CONDUCTANCE:
        r = _by_conductance(scheme, hk, smp, args, ice=jnp.asarray([0.0, 5.0, 0.0, 0.0]))
        assert float(r.soil_et_loss[1]) == 0.0 and float(r.redistribution[1]) == 0.0
        r = _by_conductance(scheme, hk, smp, args, ice=jnp.ones(N_SOI))
        assert float(r.psis) == plant_hydraulics.MINLWP_SPA
        np.testing.assert_allclose(np.asarray(r.soil_et_loss), [1 / 3, 1 / 3, 1 / 3, 0.0])
    with pytest.raises(ValueError, match="root conductance"):
        _by_conductance("roots", hk, smp, args)


def test_clm_soil_root_conductance_follows_clm_formulas():
    """CLM's root-soil interface conductance written out from PhotosynthesisMod.F90 in
    CLM's units (mm/s per mm), level by level, against the function."""
    hk, smp, args = _soil_columns()
    dz, rootfr = np.asarray(args[0]), np.asarray(args[1])
    hksat, lai, sai, froot_carbon = np.full(N_SOI, 5e-5), 1.4, 0.5, 250.0
    psi50_mm = ROOTS.root_psi50 / (1e-3 * plant_hydraulics.HEAD)
    expected = np.zeros(N_SOI)
    for j in range(N_SOI):
        density = max(2.0 * 1.0, 2.0 * froot_carbon * rootfr[j] / dz[j])
        length_density = density / (0.31e6 * np.pi * 0.29e-3**2)
        rai = (sai + lai) * 1.39 * rootfr[j]
        r_soil = np.sqrt(1.0 / (np.pi * length_density))
        soil = max(min(hksat[j], float(hk[j])) / (1e3 * r_soil), 1e-16)
        plc = 2.0 ** (-((float(smp[j]) / psi50_mm) ** 3.95))
        plc = 0.0 if plc < 0.005 else plc
        root = max(plc * rai * 1.037788e-10 / (0.25 + float(Z_SOI[j])), 1e-16)
        expected[j] = 1.0 / (1.0 / soil + 1.0 / root) if j > 0 else 0.0
    k = plant_hydraulics.soil_root_conductance_clm(
        hk, jnp.asarray(hksat), smp, args[0], Z_SOI, args[1], 2.0 * froot_carbon, lai, sai, ROOTS
    )
    per_mm = 1e6 / (0.01802 * plant_hydraulics.HEAD)  # 1/s -> mmol m-2 s-1 MPa-1
    np.testing.assert_allclose(np.asarray(k) / per_mm, expected, rtol=1e-8)
    assert expected[1:].min() > 0.0

    # the root water potential includes the gravitational potential of the levels' depths
    r = _by_conductance("clm", hk, smp, args)
    cond = np.asarray(k)[:3]
    psi = (np.asarray(smp) - 1000.0 * np.asarray(Z_SOI))[:3] * 1e-3 * plant_hydraulics.HEAD
    np.testing.assert_allclose(float(r.psis), (cond * psi).sum() / cond.sum(), rtol=1e-12)
    assert float(r.soil_et_loss[0]) == 0.0  # CLM's top level gives no water
    # dry roots lose conductance; deeper levels conduct less for the same roots
    assert float(plant_hydraulics.vulnerability(-2.6476, -2.6476, 3.95)) == 0.5
    assert float(plant_hydraulics.vulnerability(-8.0, -2.6476, 3.95)) == 0.0
    even = plant_hydraulics.soil_root_conductance_clm(
        jnp.full(N_SOI, 1e-4), jnp.asarray(hksat), jnp.full(N_SOI, -1e3), args[0], Z_SOI,
        jnp.full(N_SOI, 0.25), 500.0, lai, sai, ROOTS,
    )  # fmt: skip
    assert np.all(np.diff(np.asarray(even)[1:]) < 0.0)


def test_fine_root_biomass_from_leaf_area():
    """CLM without its carbon cycle: leaf carbon (LAI / SLA) times the root-to-leaf ratio,
    2 g biomass per g carbon. A temperate deciduous forest at LAI 5: about 420 g m-2."""
    b = plant_hydraulics.fine_root_biomass(5.0, 0.03281, 1.39)
    np.testing.assert_allclose(float(b), 2.0 * 5.0 / 0.03281 * 1.39, rtol=1e-14)
    assert 400.0 < float(b) < 450.0
    assert float(plant_hydraulics.fine_root_biomass(5.0, 0.0, 1.39)) == 0.0
    assert float(plant_hydraulics.fine_root_biomass(0.0, 0.03, 1.39)) == 0.0


def test_root_uptake_without_redistribution_gives_no_level_water():
    shares = jnp.asarray([0.5, 0.3, 0.2, 0.0])
    exchange = jnp.asarray([-3.0, 1.0, 2.0, 0.0])  # the roots would wet the top level
    on = np.asarray(plant_hydraulics.root_water_uptake(shares, exchange, 2.0, True))
    np.testing.assert_allclose(on, [-2.0, 1.6, 2.4, 0.0], rtol=1e-12)
    off = np.asarray(plant_hydraulics.root_water_uptake(shares, exchange, 2.0, False))
    np.testing.assert_allclose(off, [0.0, 0.8, 1.2, 0.0], rtol=1e-12)  # 1.6 : 2.4, sum 2
    # without transpiration nothing moves
    assert np.all(np.asarray(plant_hydraulics.root_water_uptake(shares, exchange, 0.0, False)) == 0)
    # no exchange (CLM-ml v2's weighting): the fractions of the transpiration either way
    for redistribute in (True, False):
        u = plant_hydraulics.root_water_uptake(shares, jnp.zeros(4), 2.0, redistribute)
        np.testing.assert_allclose(np.asarray(u), [1.0, 0.6, 0.4, 0.0], rtol=1e-12)


def test_root_zone_by_conductance_has_finite_derivatives():
    hk, smp, args = _soil_columns()
    for scheme in plant_hydraulics.ROOT_CONDUCTANCE:

        def f(hk, smp, biomass, ice=None, scheme=scheme):
            r = _by_conductance(scheme, hk, smp, args, ice=ice, biomass=biomass)
            u = plant_hydraulics.root_water_uptake(r.soil_et_loss, r.redistribution, 2.0, False)
            return r.psis + r.rsoil + jnp.sum(u * jnp.arange(N_SOI))

        g = jax.grad(f, argnums=(0, 1, 2))(hk, smp, 500.0)
        assert all(np.isfinite(np.asarray(x)).all() for x in g), scheme
        assert np.abs(np.asarray(g[1])[1:3]).min() > 0.0
        g = jax.grad(f, argnums=(0, 1, 2))(hk, smp, 500.0, jnp.ones(N_SOI))  # no conductance
        assert all(np.isfinite(np.asarray(x)).all() for x in g), scheme


def test_leaf_water_potential_relaxes_to_steady_state():
    dpai = _dpai()
    lsc = plant_hydraulics.plant_resistance(dpai, 0.5, 4.0)
    tr = jnp.full((N_SLOTS, 2), 1e-3)
    zs = jnp.arange(N_SLOTS, dtype=float)
    lwp = plant_hydraulics.leaf_water_potential(
        dpai, zs, lsc, tr, jnp.full((N_SLOTS, 2), -0.5), -0.3, 2500.0, 1e9
    )
    head = plant_hydraulics.HEAD
    steady = -0.3 - head * zs[:, None] - 1000.0 * tr / jnp.maximum(lsc, 1e-30)[:, None]
    np.testing.assert_allclose(np.asarray(lwp)[1:3], np.asarray(steady)[1:3], rtol=1e-12)
    assert np.all(np.asarray(lwp)[np.asarray(dpai) == 0] == 0.0)
