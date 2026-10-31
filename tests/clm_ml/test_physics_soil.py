"""clm_ml.physics.soil and its legacy adapters (Phase 7.2, process 2).

Each adapter runs on fake containers that hold only the fields it declares
(``clm_ml.legacy.soil.DECLARED``): reading an undeclared field fails, and exactly the
declared writes must change. Values are random but physical (one column, 5 soil levels,
4 above the bedrock layer count 3). The arithmetic is checked bit for bit end to end.
"""

from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.legacy import soil as adapters  # noqa: E402
from clm_ml.physics import soil  # noqa: E402

N_GRND, N_SOI, N_COLOURS = 5, 4, 3


@pytest.fixture(autouse=True)
def _module_tables(monkeypatch):
    from clm_src_biogeophys import SurfaceAlbedoMod
    from clm_src_main import clm_varpar

    rng = np.random.default_rng(7)
    monkeypatch.setattr(clm_varpar, "nlevgrnd", N_GRND)
    monkeypatch.setattr(clm_varpar, "nlevsoi", N_SOI)
    sat = rng.uniform(0.05, 0.15, (N_COLOURS, 3))
    monkeypatch.setattr(SurfaceAlbedoMod, "albsat", jnp.asarray(sat))
    monkeypatch.setattr(SurfaceAlbedoMod, "albdry", jnp.asarray(sat + 0.1))


def _values(rng):
    """Physical random values of every field an adapter may read, legacy shapes."""
    lev = N_GRND + 1  # legacy soil-level axis (index j = level j; index 0 unused)
    z = np.concatenate([[0.0], np.cumsum(rng.uniform(0.05, 0.3, N_GRND))])
    zi = np.concatenate([[0.0], z[1:] + rng.uniform(0.01, 0.02, N_GRND)])
    dz = np.concatenate([[0.0], np.diff(zi)])
    vol = rng.uniform(0.15, 0.3, lev)
    row = lambda lo, hi: rng.uniform(lo, hi, (1, lev))  # noqa: E731
    return {
        "site.z_col": z[None],
        "site.zi_col": zi[None],
        "site.dz_col": dz[None],
        "site.nbedrock_col": np.array([3]),
        "site.isoicol_col": np.array([2]),
        "soilstate.watsat_col": row(0.35, 0.5),
        "soilstate.sucsat_col": row(100.0, 300.0),
        "soilstate.bsw_col": row(4.0, 8.0),
        "soilstate.hksat_col": row(0.002, 0.005),
        "soilstate.tkmg_col": row(2.0, 3.0),
        "soilstate.tkdry_col": row(0.2, 0.3),
        "soilstate.csol_col": row(2.0e6, 2.5e6),
        "soilstate.thk_col": row(0.0, 1.0),
        "soilstate.hk_l_col": row(0.0, 1.0),
        "soilstate.smp_l_col": row(-1.0, 0.0),
        "soilstate.dsl_col": rng.uniform(0, 1, 1),
        "soilstate.soilresis_col": rng.uniform(0, 1, 1),
        "temperature.t_soisno_col": row(270.0, 290.0),
        "waterstatebulk.h2osoi_vol_col": vol[None, :N_GRND],  # legacy: no bottom level
        "waterstatebulk.h2osoi_liq_col": (vol[1:] * dz[1:] * 1000.0)[None],
        "waterstatebulk.h2osoi_ice_col": row(0.0, 1.0)[:, :N_GRND],
        "waterstatebulk.h2osfc_col": np.zeros(1),
        "water.h2osno_col": np.zeros(1),
        "canopystate.elai_patch": rng.uniform(0, 1, 1),
        "canopystate.esai_patch": rng.uniform(0, 1, 1),
        "surfalb.albgrd_col": rng.uniform(0, 1, (1, 3)),
        "surfalb.albgri_col": rng.uniform(0, 1, (1, 3)),
        "mlcanopy.gsoi_soil": np.array([35.0]),
    }


def _fake_containers(name):
    """Containers holding only the fields ``name`` declares."""
    reads, writes = adapters.DECLARED[name]
    values = _values(np.random.default_rng(0))
    fields = {}
    for key in sorted({*reads, *writes}):
        inst, field = key.split(".")
        fields.setdefault(inst, {})[field] = jnp.asarray(values[key])
    names = ("site", "soilstate", "waterstatebulk", "canopystate", "surfalb", "temperature",
             "water", "waterdiagnosticbulk", "mlcanopy")  # fmt: skip
    return {n: namedtuple(n, tuple(fields.get(n, {})))(**fields.get(n, {})) for n in names}


Bounds = namedtuple("Bounds", "begc endc begp endp")
B, FILTER = Bounds(0, 0, 0, 0), np.array([0])
Record = namedtuple("Record", "elai esai h2osoi_clm45 h2osoi_clm50 h2osoi_factor")


def _run(name, k):
    """Call adapter ``name`` on the containers ``k``; the containers it returns, by name."""
    w, s, t = k["waterstatebulk"], k["soilstate"], k["temperature"]
    if name == "clmData":
        rec = Record(1.5, 0.4, jnp.full(N_GRND, 0.3), jnp.full(N_SOI, 0.6), None)
        out = adapters.clmData(rec, 0, 0, 0, 0, s, w, k["canopystate"], k["surfalb"], k["site"])
        return dict(zip(("waterstatebulk", "canopystate", "surfalb"), out))
    if name == "SoilAlbedo":
        return {"surfalb": adapters.SoilAlbedo(B, 1, FILTER, w, k["surfalb"], site=k["site"])}
    if name == "calc_soilevap_resis":
        return {"soilstate": adapters.calc_soilevap_resis(B, 1, FILTER, s, w, t, site=k["site"])}
    if name == "SoilThermProp":
        z = jnp.zeros((1, N_GRND + 1))
        s, t, *_ = adapters.SoilThermProp(
            B, 1, FILTER, z, z, jnp.zeros(1), t, k["waterdiagnosticbulk"], w, k["water"], s,
            site=k["site"],
        )  # fmt: skip
        return {"soilstate": s, "temperature": t}
    if name == "SoilWater":
        return {"soilstate": adapters.SoilWater(B, 1, FILTER, s, w, site=k["site"])}
    if name == "SoilTemperature":
        t, s, err = adapters.SoilTemperature(
            B, 1, FILTER, s, t, k["waterdiagnosticbulk"], w, k["water"], k["mlcanopy"], 1800.0,
            site=k["site"],
        )  # fmt: skip
        assert abs(float(err[0])) < 1e-8  # the implicit solve conserves energy
        return {"temperature": t, "soilstate": s}
    raise KeyError(name)


@pytest.mark.parametrize("name", sorted(adapters.DECLARED))
def test_adapter_reads_and_writes_what_it_declares(name):
    before = _fake_containers(name)
    after = _run(name, before)
    changed = set()
    for inst, new in after.items():
        for field in new._fields:
            old, value = np.asarray(getattr(before[inst], field)), np.asarray(getattr(new, field))
            assert old.shape == value.shape, (inst, field)
            assert np.all(np.isfinite(value)), (inst, field)
            if not np.array_equal(old, value):
                changed.add(f"{inst}.{field}")
    assert changed == set(adapters.DECLARED[name][1])


def test_soil_temperature_keeps_a_uniform_profile_without_heat_flux():
    n = 6
    z = np.cumsum(np.full(n, 0.2))
    t, err = soil.soil_temperature(
        jnp.full(n, 283.0), jnp.full(n, 1.5).at[-1].set(0.0), jnp.full(n, 2.0e6), z, 0.0, 1800.0
    )
    np.testing.assert_allclose(np.asarray(t), 283.0, rtol=0, atol=1e-10)
    assert abs(float(err)) < 1e-9
