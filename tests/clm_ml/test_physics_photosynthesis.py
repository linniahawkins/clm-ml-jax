"""clm_ml.physics.leaf_boundary_layer, clm_ml.physics.photosynthesis and their legacy adapters
(Phase 7.2, process 7).

The adapters run on fake multilayer canopy containers holding only the fields they declare
(``DECLARED``): reading an undeclared field fails, and exactly the declared writes must
change. The physics is checked against what its definitions imply (Fortran's error checks of
the stomatal models and the diffusion equation, the WUE optimum is the root of the marginal
efficiency check, the water-stress rule), sunlit and shaded leaves in one call against one
call per leaf, and the stomatal conductance's implicit-function gradient against finite
differences.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import leaf_boundary_layer as lbl_adapter  # noqa: E402
from clm_ml.legacy import photosynthesis as adapter  # noqa: E402
from clm_ml.physics import leaf_boundary_layer as lbl  # noqa: E402
from clm_ml.physics import photosynthesis as ps  # noqa: E402
from clm_ml.physics.constants import dh2o_to_dco2, gbh_min  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

# 6 slots: layer 0 without plant area, 1-3 leafy, 4 above the canopy, 5 padding
N_SLOTS, NCAN, NTOP, NBOT = 6, 5, 4, 2  # legacy 1-based counts
DPAI = np.array([0.0, 0.6, 0.9, 0.5, 0.0, 0.0])
Params = namedtuple(
    "Params",
    "g0_medlyn g1_medlyn g0_ball_berry g1_ball_berry iota gs_min psi50_gs shape_gs "
    "leaf_dimension",
)
PARAMS = Params(*(jnp.asarray(v) for v in (1e-4, 2.35, 0.01, 9.0, 375.0, 0.002, -1.6, 40.0,
                                           0.04)))  # fmt: skip
Grid = namedtuple("Grid", "p c3")
Opts = namedtuple("Opts", "gs_type gspot_type gs_solver colim_type acclim_type gb_type")
OPTS = Opts(2, 1, 2, 1, 1, 3)


def _layers():
    zw = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0])
    dz = np.diff(zw)
    return PatchLayers(
        n_layers=jnp.asarray(NCAN), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.asarray(dz),
        zs=jnp.asarray(zw[:-1] + 0.5 * dz), zw=jnp.asarray(zw),
    )  # fmt: skip


def _inputs(apar_sun=1200.0, lwp=-0.5):
    """Per-layer and per-leaf inputs of ``leaf_photosynthesis`` (sunlit, shaded)."""
    n = N_SLOTS
    leaf = lambda sun, sha: jnp.tile(jnp.array([sun, sha]), (n, 1))  # noqa: E731
    return dict(
        dpai=jnp.asarray(DPAI), eair=jnp.full(n, 1500.0), cair=jnp.full(n, 383.0),
        tleaf=leaf(300.0, 297.0), vcmax25=leaf(125.0, 90.0), jmax25=leaf(209.0, 150.0),
        rd25=leaf(1.9, 1.35), kp25=leaf(2.5, 1.8), apar=leaf(apar_sun, 150.0),
        gbc=leaf(1.4, 1.3), gbv=leaf(1.9, 1.8), lwp=leaf(lwp, lwp - 0.1),
    )  # fmt: skip


def _run(inputs=None, **kw):
    f = inputs or _inputs()
    kw.setdefault("c3", True)
    model = kw.get("stomatal_conductance", "wue")
    if model != "wue":
        kw.update(zip(("g0", "g1"), ps.stomatal_parameters(PARAMS, model)))
    return ps.leaf_photosynthesis(
        _layers(), f["dpai"], f["eair"], f["cair"], f["tleaf"], f["vcmax25"], f["jmax25"],
        f["rd25"], f["kp25"], f["apar"], f["gbc"], f["gbv"], f["lwp"], 209.0, 101325.0,
        293.0, iota=PARAMS.iota, gs_min=PARAMS.gs_min, psi50_gs=PARAMS.psi50_gs,
        shape_gs=PARAMS.shape_gs, **kw,
    )  # fmt: skip


LEAFY = np.flatnonzero(DPAI > 0)


# ---------------------------------------------------------------------------
# Adapters: declared reads and writes
# ---------------------------------------------------------------------------


def _fake_mlcanopy(declared, rng):
    reads, writes = declared
    fields = sorted({k.split(".")[1] for k in (*reads, *writes)})
    n = N_SLOTS + 1
    f = _inputs()

    def shape(name):
        if name.endswith("_leaf"):
            return (1, n, 3)
        return (1, n) if name.endswith("_profile") else (1,)

    values = {k: jnp.asarray(rng.uniform(-1.0, -0.5, shape(k))) for k in fields}
    for k in ("eair", "cair", "dpai"):
        values[k + "_profile"] = jnp.concatenate([jnp.zeros(1), f[k]])[None]
    for k in ("tleaf", "vcmax25", "jmax25", "rd25", "kp25", "apar", "gbc", "gbv", "lwp"):
        values[k + "_leaf"] = jnp.zeros((1, n, 3)).at[0, 1:, 1:].set(f[k])
    lay = _layers()
    scalars = dict(o2ref_forcing=209.0, pref_forcing=101325.0, tacclim_forcing=293.0,
                   tref_forcing=295.0, rhomol_forcing=41.0, ncan_canopy=NCAN,
                   ntop_canopy=NTOP, nbot_canopy=NBOT)  # fmt: skip
    values.update({k: jnp.array([v]) for k, v in scalars.items() if k in values})
    values.update(
        wind_profile=jnp.linspace(0.5, 3.0, n)[None],
        tair_profile=jnp.full((1, n), 296.0),
        dz_profile=jnp.concatenate([jnp.zeros(1), lay.dz])[None],
        zs_profile=jnp.concatenate([jnp.zeros(1), lay.zs])[None],
        zw_profile=lay.zw[None],
    )
    return namedtuple("mlcanopy", fields)(**{k: values[k] for k in fields})


def _changed(before, after):
    out = set()
    for field in after._fields:
        old, new = np.asarray(getattr(before, field)), np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        assert np.all(np.isfinite(new)), field
        if not np.array_equal(old, new):
            out.add(f"mlcanopy.{field}")
    return out


@pytest.mark.parametrize("gs_type", [0, 1, 2])
def test_photosynthesis_adapter_reads_and_writes_what_it_declares(gs_type):
    declared = adapter.DECLARED["LeafPhotosynthesis"]
    before = _fake_mlcanopy(declared, np.random.default_rng(3))
    after = adapter.LeafPhotosynthesis(
        1, [0], None, before, PARAMS, Grid(0, True), opts=OPTS._replace(gs_type=gs_type)
    )
    expected = set(declared[1])
    if gs_type == 2:  # g0, g1 are the Medlyn and Ball-Berry parameters
        expected -= {"mlcanopy.g0_canopy", "mlcanopy.g1_canopy"}
    assert _changed(before, after) == expected
    for field in adapter._LEAF_FIELDS.values():  # ground slot, padding, leaf index 0 kept
        a, b = np.asarray(getattr(after, field)), np.asarray(getattr(before, field))
        np.testing.assert_array_equal(a[0, NCAN + 1 :], b[0, NCAN + 1 :])
        np.testing.assert_array_equal(a[0, 0], b[0, 0])
        np.testing.assert_array_equal(a[0, :, 0], b[0, :, 0])


def test_photosynthesis_adapter_one_leaf_matches_both():
    declared = adapter.DECLARED["LeafPhotosynthesis"]
    before = _fake_mlcanopy(declared, np.random.default_rng(3))
    both = adapter.LeafPhotosynthesis(1, [0], None, before, PARAMS, Grid(0, True), opts=OPTS)
    one = before
    for il in (1, 2):
        one = adapter.LeafPhotosynthesis(1, [0], il, one, PARAMS, Grid(0, True), opts=OPTS)
    for field in adapter._LEAF_FIELDS.values():
        np.testing.assert_array_equal(getattr(one, field), getattr(both, field), err_msg=field)


def test_photosynthesis_adapter_refuses_invalid_codes():
    before = _fake_mlcanopy(adapter.DECLARED["LeafPhotosynthesis"], np.random.default_rng(3))
    for bad in (dict(gs_type=3), dict(gs_solver=0), dict(colim_type=2)):
        with pytest.raises(ValueError):
            adapter.LeafPhotosynthesis(1, [0], None, before, PARAMS, Grid(0, True),
                                       opts=OPTS._replace(**bad))  # fmt: skip


def test_boundary_layer_adapter_reads_and_writes_what_it_declares():
    declared = lbl_adapter.DECLARED["LeafBoundaryLayer"]
    before = _fake_mlcanopy(declared, np.random.default_rng(4))
    after = lbl_adapter.LeafBoundaryLayer(1, [0], before, PARAMS, opts=OPTS)
    assert _changed(before, after) == set(declared[1])
    for f in ("gbh_leaf", "gbv_leaf", "gbc_leaf"):
        a, b = np.asarray(getattr(after, f)), np.asarray(getattr(before, f))
        np.testing.assert_array_equal(a[0, :, 0], b[0, :, 0])
        np.testing.assert_array_equal(a[0, 0], b[0, 0])


# ---------------------------------------------------------------------------
# Leaf boundary layer
# ---------------------------------------------------------------------------


def _gb(regime, tleaf=300.0, wind=None):
    n = N_SLOTS
    wind = jnp.linspace(0.2, 3.0, n) if wind is None else wind
    tl = jnp.stack([jnp.full(n, tleaf), jnp.full(n, 294.0)], axis=1)
    return lbl.leaf_boundary_layer(jnp.asarray(DPAI), wind, jnp.full(n, 296.0), tl,
                                   101325.0, 295.0, 41.0, 0.04, regime=regime)  # fmt: skip


def test_boundary_layer_regimes():
    lam, turb, free = (_gb(r) for r in lbl.REGIMES)
    for g in (lam, turb, free):
        for x in g:
            np.testing.assert_array_equal(np.asarray(x)[DPAI == 0], 0.0)
        assert np.all(np.asarray(g.gbh)[LEAFY] >= gbh_min)
        # diffusivity ratios: CO2 < heat < water vapor
        assert np.all((g.gbc < g.gbh)[LEAFY]) and np.all((g.gbh < g.gbv)[LEAFY])
    assert np.all((lam.gbh <= turb.gbh)[LEAFY]) and np.all((turb.gbh <= free.gbh)[LEAFY])
    # free convection only adds where the leaf is warmer than the air (sunlit at 300 K); the
    # floor on the Grashof number (finite gradients) leaves ~4e-10 relative otherwise
    np.testing.assert_allclose(free.gbh[:, 1], turb.gbh[:, 1], rtol=1e-9)
    assert np.all((free.gbh[:, 0] > turb.gbh[:, 0])[LEAFY])
    with pytest.raises(ValueError):
        _gb("other")


# ---------------------------------------------------------------------------
# Photosynthesis
# ---------------------------------------------------------------------------


def _gleaf(gbc, gs):
    return 1.0 / (1.0 / gbc + dh2o_to_dco2 / gs)


@pytest.mark.parametrize("model", ps.STOMATAL_CONDUCTANCE)
@pytest.mark.parametrize("c3", [True, False])
def test_diffusion_and_layers_without_plant_area(model, c3):
    f = _inputs()
    r = _run(f, c3=c3, stomatal_conductance=model)
    for name, x in zip(r._fields, r):
        x = np.asarray(x)
        assert np.all(np.isfinite(x)), name
        if name != "gspot":
            np.testing.assert_array_equal(x[DPAI == 0], 0.0, err_msg=name)
    # An = (ca - ci) gleaf (Fortran's diffusion error check), with the stressed gs
    gl = _gleaf(f["gbc"], r.gs)
    an = (f["cair"][:, None] - r.ci) * gl
    np.testing.assert_allclose(np.asarray(an)[LEAFY], np.asarray(r.anet)[LEAFY], rtol=1e-10)
    assert np.all(np.asarray(r.anet)[LEAFY, 0] > 0)  # sunlit leaves photosynthesize


def test_ball_berry_and_medlyn_satisfy_their_models():
    """Fortran's error checks of LeafPhotosynthesis, on the unstressed conductance. Fortran
    checks before the water-stress pass, which recomputes photosynthesis for gs (CiFuncGs,
    even without stress); with co-limitation that solves ac and aj separately and moves anet,
    so the check here is without co-limitation, where the recomputation is the solver's."""
    f = _inputs()
    for model in ("ball_berry", "medlyn"):
        r = _run(f, stomatal_conductance=model, water_stress=False, colimitation=False)
        gs, an, cs, esat = (np.asarray(x)[LEAFY] for x in (r.gspot, r.anet, r.cs, r.leaf_esat))
        ceair = np.asarray(r.ceair)[LEAFY]
        gbv = np.asarray(f["gbv"])[LEAFY]
        hs = (gbv * ceair + gs * esat) / ((gbv + gs) * esat)
        if model == "ball_berry":
            expected = 0.01 + 9.0 * np.maximum(an, 0.0) * hs / cs
        else:
            vpd = (esat - hs * esat) * 0.001
            expected = 1e-4 + dh2o_to_dco2 * (1.0 + 2.35 / np.sqrt(vpd)) * np.maximum(an, 0) / cs
        np.testing.assert_allclose(gs, expected, atol=1e-6)


def test_wue_optimum_is_the_root_of_the_efficiency_check():
    f = _inputs()
    r = _run(f, water_stress=False)
    # Rebuild the marginal efficiency check of one leafy layer, sunlit leaf
    i = LEAFY[1]
    temp = ps.temperature_parameters(True, 293.0)
    rates = ps.leaf_rates(f["tleaf"][i, 0], f["vcmax25"][i, 0], f["jmax25"][i, 0],
                          f["rd25"][i, 0], f["kp25"][i, 0], temp, True)  # fmt: skip
    bio = ps.Biochemistry(rates, r.je[i, 0], f["apar"][i, 0], 209.0)
    leaf = ps._Leaf(f["gbc"][i, 0], f["gbv"][i, 0], f["cair"][i], f["eair"][i], r.ceair[i, 0],
                    r.leaf_esat[i, 0])  # fmt: skip
    check = ps.stomata_efficiency(r.gspot[i, 0], leaf, bio, PARAMS.iota, 101325.0, True, True)
    # 20 halvings of [gs_min, 2] and a Newton step: the residual is near round-off of the
    # finite difference that defines it
    assert abs(float(check)) < 1e-9
    assert float(r.gspot[i, 0]) > float(PARAMS.gs_min)


def test_low_light_gives_minimum_conductance():
    r = _run(_inputs(apar_sun=0.0), water_stress=False)
    np.testing.assert_allclose(np.asarray(r.gspot)[LEAFY, 0], float(PARAMS.gs_min))


def test_water_stress():
    f = _inputs(lwp=-1.4)
    r = _run(f)
    fpsi = 1.0 / (1.0 + (np.asarray(f["lwp"]) / -1.6) ** 40.0)
    expected = np.maximum(np.asarray(r.gspot) * fpsi, 0.002)
    np.testing.assert_array_equal(np.asarray(r.gs)[LEAFY], expected[LEAFY])
    assert np.all(np.asarray(r.gs)[LEAFY] < np.asarray(r.gspot)[LEAFY])
    unstressed = _run(f, water_stress=False)
    np.testing.assert_array_equal(unstressed.gs, unstressed.gspot)


def test_sun_and_shade_in_one_call_match_one_call_per_leaf():
    f = _inputs()
    both = jax.jit(_run)(f)
    for leaf in (0, 1):
        one = {k: (v[:, leaf : leaf + 1] if v.ndim == 2 else v) for k, v in f.items()}
        r = jax.jit(_run)(one)
        for name, a, b in zip(r._fields, r, both):
            # XLA compiles the two shapes differently: last-bit differences (Phase 7 C16), which
            # the WUE check amplifies (a difference of net photosynthesis 0.001 apart in gs) to
            # ~3e-12 in gs, the noise-twin level; 1e-10 is the policy's defect threshold
            np.testing.assert_allclose(a[:, 0], b[:, leaf], rtol=1e-10, atol=0, err_msg=name)


def test_options_without_acclimation_and_colimitation():
    for kw in (dict(acclimation=False), dict(colimitation=False), dict(c3=False)):
        r = _run(**kw)
        assert np.all(np.isfinite(np.asarray(r.anet)))
        assert np.all(np.asarray(r.anet)[LEAFY, 0] > 0)


def test_brent_is_refused():
    with pytest.raises(NotImplementedError):
        _run(wue_solver="brent")
    _run(stomatal_conductance="medlyn", wue_solver="brent")  # not used by Medlyn
    with pytest.raises(ValueError):
        _run(stomatal_conductance="other")


def test_realized_rate():
    ac, aj, ap = jnp.array(10.0), jnp.array(8.0), jnp.array(6.0)
    assert float(ps.realized_rate(ac, aj, ap, c3=True, colimitation=False)) == 8.0
    assert float(ps.realized_rate(ac, aj, ap, c3=False, colimitation=False)) == 6.0
    co = float(ps.realized_rate(ac, aj, ap, c3=True, colimitation=True))
    assert co < 8.0 and 0.98 * co**2 - 18.0 * co + 80.0 == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("model", ["wue", "medlyn"])
def test_conductance_gradient_is_the_implicit_one(model):
    """d gspot / d vcmax25 by jax.grad (implicit-function correction of the bisection or
    secant solver) vs central differences of the forward value."""
    f = _inputs()
    i = LEAFY[1]

    def gspot(scale):
        g = dict(f, vcmax25=f["vcmax25"] * scale)
        return _run(g, stomatal_conductance=model, water_stress=False).gspot[i, 0]

    grad = float(jax.grad(gspot)(1.0))
    h = 1e-4
    fd = (float(gspot(1.0 + h)) - float(gspot(1.0 - h))) / (2 * h)
    assert np.isfinite(grad) and grad != 0.0
    assert grad == pytest.approx(fd, rel=2e-3)
