"""clm_ml.physics.nitrogen_profile, clm_ml.physics.integrators and their legacy adapters
(Phase 7.2, process 9).

Each adapter runs on fake containers holding only the fields it declares (``DECLARED`` in
``clm_ml.legacy.nitrogen_profile`` and ``clm_ml.legacy.integrators``): reading an undeclared
field fails, and exactly the declared writes must change. The physics is checked against the
analytical canopy integral and the definition of each Runge-Kutta method.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import integrators as rk_adapter  # noqa: E402
from clm_ml.legacy import nitrogen_profile as nitrogen_adapter  # noqa: E402
from clm_ml.physics import integrators, nitrogen_profile  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

N_SLOTS = 6  # layer slots; legacy arrays have N_SLOTS + 1 entries (index 0 unused)
N_CAN = 4  # layers in use (legacy slots 1..4)
LEAFY = slice(1, 4)  # legacy slots with plant area
N_STAGES = 5  # stage axis of the legacy increments (longer than rk4's 4 stages)
N_CHECKS = 13
Opts = namedtuple("Opts", "acclim_type leaf_optics_type kn_val runge_kutta_type")
OPTS = Opts(1, 0, -999.0, 41)
Params = namedtuple("Params", "vcmax25 clumping_index")
PARAMS = Params(125.0, 0.85)
Grid = namedtuple("Grid", "p c3")


def _value(field, rng):
    n = N_SLOTS + 1
    name = field.split(".")[1]
    if name in ("dpai_profile",):
        v = np.zeros(n)
        v[LEAFY] = [0.3, 0.5, 0.4]
        v[N_CAN + 1 :] = 1.0e36  # padding holds fill values
        return v[None]
    special = {
        "ncan_canopy": np.array([N_CAN]),
        "ntop_canopy": np.array([3]),
        "nbot_canopy": np.array([1]),
        "lai_canopy": np.array([0.9]),
        "sai_canopy": np.array([0.3]),
        "tacclim_forcing": np.array([290.0]),
        "kb_profile": np.full((1, n), 0.6),
        "tbi_profile": np.linspace(0.3, 1.0, n)[None],
        "fracsun_profile": np.linspace(0.2, 0.7, n)[None],
        "balance_error": np.zeros((1, N_CHECKS)),
        "tg_soil": np.array([288.0]),
        "tg_bef_soil": np.array([287.5]),
        "dtg_soil": rng.uniform(-0.2, 0.2, (1, N_STAGES)),
    }
    if name in special:
        return special[name]
    if name.endswith("_leaf"):
        shape = (1, n, 3, N_STAGES) if name.startswith("d") and "_bef" not in name else (1, n, 3)
        if name in ("dtleaf_leaf", "dlwp_leaf"):
            return rng.uniform(-0.2, 0.2, shape)
        return rng.uniform(1.0, 2.0, (1, n, 3))
    if name in ("dtair_profile", "deair_profile", "dh2ocan_profile"):
        return rng.uniform(-0.2, 0.2, (1, n, N_STAGES))
    return rng.uniform(1.0, 2.0, (1, n))


def _fakes(declared, rng):
    fields = {}
    for key in (*declared[0], *declared[1]):
        fields[key.split(".")[1]] = jnp.asarray(_value(key, rng))
    names = sorted(fields)
    return namedtuple("mlcanopy", names)(**{f: fields[f] for f in names})


CASES = {
    "CanopyNitrogenProfile": (
        nitrogen_adapter.DECLARED["CanopyNitrogenProfile"],
        lambda ml: nitrogen_adapter.CanopyNitrogenProfile(
            1, [0], ml, PARAMS, Grid(0, True), opts=OPTS
        ),
    ),
    "RungeKuttaUpdate": (
        rk_adapter.DECLARED["RungeKuttaUpdate"],
        lambda ml: rk_adapter.RungeKuttaUpdate(
            2, rk_adapter.integrator_tableau(OPTS), ml, Grid(0, True)
        ),
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_adapter_reads_and_writes_what_it_declares(name):
    declared, call = CASES[name]
    before = _fakes(declared, np.random.default_rng(5))
    after = call(before)
    changed = set()
    for field in after._fields:
        old, new = np.asarray(getattr(before, field)), np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        if not np.array_equal(old, new):
            changed.add("mlcanopy." + field)
    assert changed == set(declared[1])


def _layers(n_layers=N_CAN):
    return PatchLayers(
        n_layers=jnp.asarray(n_layers),
        n_within=jnp.asarray(3),
        first_leafy=jnp.asarray(0),
        dz=jnp.ones(N_SLOTS),
        zs=jnp.arange(N_SLOTS, dtype=float),
        zw=jnp.arange(N_SLOTS + 1, dtype=float),
    )


@pytest.mark.parametrize("vary_with_height", [False, True])
@pytest.mark.parametrize("c3", [True, False])
def test_nitrogen_profile_integrates_to_the_analytical_value(vary_with_height, c3):
    dpai = jnp.array([0.3, 0.5, 0.4, 0.0, 1.0e36, 1.0e36])  # top layer leafless, padding
    r = nitrogen_profile.canopy_nitrogen_profile(
        _layers(), dpai, jnp.full(N_SLOTS, 0.6), jnp.linspace(0.3, 1.0, N_SLOTS),
        jnp.linspace(0.2, 0.7, N_SLOTS), 0.9, 0.3, 290.0, 125.0, 0.85,
        c3=c3, acclimation=True, kn_val=-999.0, vary_with_height=vary_with_height,
    )  # fmt: skip
    assert abs(float(r.integration_error)) < 1e-12
    fracsun = np.linspace(0.2, 0.7, N_SLOTS)
    mean = np.asarray(r.vcmax25_leaf) @ np.array([1.0, 0.0]) * fracsun + np.asarray(
        r.vcmax25_leaf
    ) @ np.array([0.0, 1.0]) * (1.0 - fracsun)
    np.testing.assert_allclose(mean[:3], np.asarray(r.vcmax25_profile)[:3], rtol=1e-12)
    assert np.all(np.asarray(r.vcmax25_leaf)[3:] == 0.0)  # no plant area and padding
    assert np.all(np.asarray(r.vcmax25_profile)[:3] > 0.0)
    assert np.all(np.diff(np.asarray(r.vcmax25_profile)[:3]) > 0.0)  # declines downward
    assert np.all(np.asarray(r.kp25_leaf if c3 else r.jmax25_leaf) == 0.0)


@pytest.mark.parametrize("name", [n for n in integrators.TABLEAUS if n != "euler"])
def test_tableaus_are_consistent(name):
    a, b, c = integrators.TABLEAUS[name]
    np.testing.assert_allclose(b.sum(), 1.0, rtol=1e-15)
    lower = np.where(np.tril(np.ones_like(a), -1) > 0, a, 0.0)
    np.testing.assert_allclose(lower.sum(axis=1), c, rtol=1e-15)


def _rk4_solve(f, y0, h):
    """``y' = f(y)`` over one step with the update: each stage's physics gives
    ``bef + h f(x)`` (a step from the start of the sub-step with the tendency at the stage's
    state ``x``, as the canopy solves from the ``*_bef`` values). Returns the state after
    the last stage and after the physics-only pass that follows it."""
    tab = integrators.TABLEAUS["rk4"]
    n = len(tab.b)
    lay = _layers()

    def state(v):
        return integrators.Integrated(
            v, v, v, jnp.stack([v, v], axis=-1), jnp.stack([v, v], axis=-1), v[0]
        )

    bef = state(y0)
    x = bef
    dx = integrators.Integrated(*(jnp.zeros(v.shape + (n,)) for v in bef))
    for stage in range(1, n + 2):
        cur = integrators.Integrated(*(b + h * f(v) for b, v in zip(bef, x)))  # physics
        x, inc, k = integrators.runge_kutta_update(stage, tab, lay, bef, cur, dx)
        dx = integrators.Integrated(*(d.at[..., k].set(i) for d, i in zip(dx, inc)))
        if stage == n:
            final = x
    return final, x, cur


def test_rk4_update_is_fourth_order_runge_kutta():
    y0 = jnp.linspace(1.0, 2.0, N_SLOTS)
    h = 0.1
    final, after_last, physics = _rk4_solve(lambda y: -2.0 * y, y0, h)
    z = -2.0 * h
    rk4 = 1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24  # classical RK4 on y' = -2y
    in_use = np.arange(N_SLOTS) < N_CAN
    for v in final:
        got = np.asarray(v)
        if got.ndim == 0:  # ground temperature
            np.testing.assert_allclose(got, float(y0[0]) * rk4, rtol=1e-14)
            continue
        expect = np.asarray(y0) * rk4
        if got.ndim == 2:
            expect = np.stack([expect, expect], axis=-1)
        np.testing.assert_allclose(got[in_use], expect[in_use], rtol=1e-14)
    # the physics-only pass after the last stage leaves the state as the physics left it
    for v, w in zip(after_last, physics):
        np.testing.assert_array_equal(np.asarray(v), np.asarray(w))


def test_rk_update_keeps_padding():
    tab = integrators.TABLEAUS["rk4"]
    lay = _layers()
    v = jnp.linspace(1.0, 2.0, N_SLOTS)
    bef = integrators.Integrated(v, v, v, jnp.stack([v, v], -1), jnp.stack([v, v], -1), v[0])
    cur = integrators.Integrated(*(x + 1.0 for x in bef))
    dx = integrators.Integrated(*(jnp.full(x.shape + (4,), 7.0) for x in bef))
    new, inc, k = integrators.runge_kutta_update(1, tab, lay, bef, cur, dx)
    assert int(k) == 0
    pad = np.arange(N_SLOTS) >= N_CAN
    np.testing.assert_array_equal(np.asarray(new.tair)[pad], np.asarray(cur.tair)[pad])
    np.testing.assert_array_equal(np.asarray(inc.tair)[pad], 7.0)
    # stage 1 of rk4: the next stage starts from bef + k1 / 2
    np.testing.assert_allclose(np.asarray(new.tair)[~pad], np.asarray(v + 0.5)[~pad], rtol=1e-15)
