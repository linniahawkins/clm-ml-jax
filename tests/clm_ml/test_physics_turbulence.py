"""clm_ml.physics.turbulence and its legacy adapter (Phase 7.2, process 6).

The adapter runs on a fake multilayer canopy container holding only the fields it declares
(``clm_ml.legacy.turbulence.DECLARED``): reading an undeclared field fails, and exactly the
declared writes must change. The physics is checked against what its definitions imply
(the wind profile recovers the forcing wind and is continuous at the canopy top, the solvers
find their roots) and the Obukhov length's implicit-function gradient against finite
differences.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.io.rsl_tables import default_rsl_tables  # noqa: E402
from clm_ml.legacy import turbulence as adapter  # noqa: E402
from clm_ml.physics import turbulence as turb  # noqa: E402
from clm_ml.physics.constants import vkc  # noqa: E402
from clm_ml.state import PatchLayers  # noqa: E402

# A 10 m canopy under a 23 m reference height: 4 layers within, 3 above, 1 padding slot
ZTOP, ZREF = 10.0, 23.0
ZW = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 14.0, 18.5, 23.0, 23.0])  # ground + layer tops
N_SLOTS, NCAN, NTOP, NBOT = 8, 7, 4, 2  # legacy 1-based counts
Grid = namedtuple("Grid", "p")
Opts = namedtuple("Opts", "turb_type sparse_canopy_type HF_extension_type")
OPTS = Opts(1, 1, 2)


def _layers():
    dz = np.diff(ZW)
    zs = ZW[:-1] + 0.5 * dz
    return PatchLayers(
        n_layers=jnp.asarray(NCAN), n_within=jnp.asarray(NTOP),
        first_leafy=jnp.asarray(NBOT - 1), dz=jnp.asarray(dz), zs=jnp.asarray(zs),
        zw=jnp.asarray(ZW),
    )  # fmt: skip


def _forcing(thref=296.0, taf=294.0, uref=3.0):
    qref = 0.010
    return dict(
        pref=101325.0, ztop=ZTOP, lai=2.0, sai=0.7, zref=ZREF, uref=uref, thref=thref,
        thvref=thref * (1.0 + 0.61 * qref), qref=qref, rhomol=41.0,
        tair=jnp.full(N_SLOTS, taf), eair=jnp.full(N_SLOTS, 1619.0),
    )  # fmt: skip


def _run(ground="log_profile", **forcing):
    f = _forcing(**forcing)
    return turb.harman_finnigan(
        _layers(), f["tair"], f["eair"], f["pref"], f["ztop"], f["lai"], f["sai"], f["zref"],
        f["uref"], f["thref"], f["thvref"], f["qref"], f["rhomol"], default_rsl_tables(),
        sparse_canopy=True, ground_conductance=ground,
    )  # fmt: skip


def _fake_mlcanopy(rng):
    reads, writes = adapter.DECLARED["CanopyTurbulence"]
    fields = sorted({k.split(".")[1] for k in (*reads, *writes)})
    n = N_SLOTS + 1
    values = {
        f: jnp.asarray(rng.uniform(-1.0, -0.5, (1, n) if f.endswith("_profile") else (1,)))
        for f in fields
    }
    lay, f = _layers(), _forcing()
    values.update(
        {k + "_forcing": jnp.array([f[k]]) for k in
         ("pref", "zref", "uref", "thref", "thvref", "qref", "rhomol")},
        ztop_canopy=jnp.array([ZTOP]), lai_canopy=jnp.array([2.0]), sai_canopy=jnp.array([0.7]),
        tair_profile=jnp.full((1, n), 294.0), eair_profile=jnp.full((1, n), 1619.0),
        ncan_canopy=jnp.array([NCAN]), ntop_canopy=jnp.array([NTOP]),
        nbot_canopy=jnp.array([NBOT]),
        dz_profile=jnp.concatenate([jnp.zeros(1), lay.dz])[None],
        zs_profile=jnp.concatenate([jnp.zeros(1), lay.zs])[None],
        zw_profile=lay.zw[None],
    )  # fmt: skip
    return namedtuple("mlcanopy", fields)(**values)


def test_adapter_reads_and_writes_what_it_declares():
    before = _fake_mlcanopy(np.random.default_rng(5))
    after = adapter.CanopyTurbulence(0, 1, [0], before, Grid(0), opts=OPTS)
    changed = set()
    for field in after._fields:
        old, new = np.asarray(getattr(before, field)), np.asarray(getattr(after, field))
        assert old.shape == new.shape, field
        assert np.all(np.isfinite(new)), field
        if not np.array_equal(old, new):
            changed.add(f"mlcanopy.{field}")
    assert changed == set(adapter.DECLARED["CanopyTurbulence"][1])
    for f in ("wind_profile", "mflx_profile", "gac_profile", "kc_eddy_profile"):
        # ground slot and padding kept
        np.testing.assert_array_equal(
            getattr(after, f)[0, NCAN + 1 :], getattr(before, f)[0, NCAN + 1 :]
        )
        assert getattr(after, f)[0, 0] == getattr(before, f)[0, 0]


def test_adapter_refuses_unknown_ground_conductance():
    with pytest.raises(ValueError):
        adapter.CanopyTurbulence(0, 1, [0], _fake_mlcanopy(np.random.default_rng(5)), Grid(0),
                                 opts=Opts(1, 1, 3))  # fmt: skip


# canopy-top air colder (stable) .. warmer (unstable) than the reference air at 296 K, the
# same humidity; 294 K and 299 K give roots beyond the theory's range -2 <= Lc/L <= 1
@pytest.mark.parametrize("taf", [294.0, 295.5, 295.8, 296.2, 296.5, 297.0, 299.0])
def test_obukhov_length_is_the_root_or_the_limit(taf):
    f = _forcing(taf=taf)
    t = _run(taf=taf)
    assert (float(t.obu) > 0) == (taf < 296.0)
    LcL = float(t.Lc / t.obu)
    if taf in (294.0, 299.0):
        assert LcL == pytest.approx(1.0 if taf < 296.0 else -2.0, rel=1e-12)
        return
    assert -2.0 < LcL < 1.0
    inp = turb.ObuInputs(t.Lc, ZTOP, f["lai"], f["sai"], ZREF, f["uref"], f["thref"],
                         f["thvref"], f["qref"], f["rhomol"], t.taf, t.qaf)  # fmt: skip
    obu_new = turb.obukhov_terms(t.obu, inp, default_rsl_tables(), True).obu_new
    assert abs(float(obu_new - t.obu)) < 1e-9 * abs(float(t.obu))


def test_wind_profile_recovers_forcing_wind_and_is_continuous():
    t = _run()
    tables = default_rsl_tables()
    psim_ref = turb.psi_rsl(ZREF, ZTOP, t.zdisp, t.obu, t.beta, t.PrSc, tables)[0]
    u_ref = t.ustar / vkc * (jnp.log((ZREF - t.zdisp) / (ZTOP - t.zdisp)) + psim_ref)
    np.testing.assert_allclose(u_ref, 3.0, rtol=1e-12)
    psim_top = turb.psi_rsl(ZTOP, ZTOP, t.zdisp, t.obu, t.beta, t.PrSc, tables)[0]
    np.testing.assert_allclose(t.ustar / vkc * psim_top, t.uaf, rtol=1e-12)  # above = below
    wind = np.asarray(t.wind)[:NCAN]
    assert np.all(np.diff(wind) > 0)  # increases with height
    assert np.all(np.asarray(t.mflx)[:NCAN] <= 0) and float(t.mflx[NTOP]) == -float(t.ustar**2)


def test_roughness_length_is_the_root():
    t = _run()
    _, _, psim_hc, psim_hat_hc = turb.psi_rsl(
        ZREF, ZTOP, t.zdisp, t.obu, t.beta, t.PrSc, default_rsl_tables()
    )
    r = turb._z0m_residual(t.z0m, t.obu, ZTOP - t.zdisp, jnp.exp(-vkc / t.beta),
                           jnp.exp(psim_hat_hc), psim_hc)  # fmt: skip
    assert 0.0 < float(t.z0m) < ZTOP and abs(float(r)) < 0.002


def test_conductances_positive_and_limited():
    for ground in turb.GROUND_CONDUCTANCE:
        t = _run(ground=ground)
        gac = np.asarray(t.gac)[:NCAN]
        assert np.all(gac >= 41.0 / 500.0 * (1 - 1e-12)) and float(t.gac0) > 0
        zs = np.asarray(_layers().zs)[:NCAN]
        to_next = np.append(np.diff(zs), ZREF - zs[-1])  # mid-layer to mid-layer, top to zref
        np.testing.assert_allclose(np.asarray(t.kc_eddy)[:NCAN], gac / 41.0 * to_next)
    with pytest.raises(ValueError):
        _run(ground="other")


def test_lookup_psihat_on_nodes_and_outside():
    m = default_rsl_tables().momentum
    i, j = 100, 20
    assert float(turb.lookup_psihat(m.zdt[i], m.dtl[j], m)) == pytest.approx(m.psi[i, j])
    # beyond the grid both neighbours are the edge value
    assert float(turb.lookup_psihat(m.zdt[0] + 1.0, m.dtl[-1] + 1.0, m)) == pytest.approx(
        m.psi[0, -1]
    )


def test_obukhov_gradient_is_the_implicit_one():
    """d obu / d thref by jax.grad (implicit-function correction) vs central differences of
    the solver's forward value."""

    def obu(thref):
        return _run(thref=thref, taf=294.0).obu

    g = float(jax.grad(obu)(296.0))
    h = 1e-3
    fd = (float(obu(296.0 + h)) - float(obu(296.0 - h))) / (2 * h)
    assert np.isfinite(g) and g == pytest.approx(fd, rel=1e-3)


def test_registry():
    assert turb.ROUGHNESS_SUBLAYER == {"harman_finnigan": turb.harman_finnigan}
