"""
Fortran validation tests for CLM-ML-JAX.

For each Fortran subroutine that has a golden JSON file in
tests/fortran_validation/golden_IO/, this module loads the reference inputs/outputs
from the Fortran build and verifies that the JAX implementation produces
the same values within a tight numerical tolerance.

Tolerance policy:
  REL_TOL = 1e-9   relative tolerance
  ABS_TOL = 1e-15  absolute guard near zero

This is slightly looser than the Fortran-only golden regression (1e-14)
to account for minor floating-point ordering differences between JAX and
nvfortran while still catching any genuine numerical deviations.

Each adapter function accepts a dict of golden inputs and returns a dict
of computed outputs using the same key names as the golden JSON.

Functions with module-level configuration globals (colim_type, gb_type,
sparse_canopy_type, leaf_optics_type) temporarily override the module
binding and restore it after the call.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_GOLDEN_DIR = _HERE / "golden_IO"

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

REL_TOL = 1e-9
ABS_TOL = 1e-15

# ---------------------------------------------------------------------------
# Lazy imports — done at session start to ensure src/ is on sys.path
# (conftest.py adds it before these run)
# ---------------------------------------------------------------------------


def _import_jax_modules():
    """Return a namespace of all needed JAX functions."""
    import importlib

    mods = {}
    for name in [
        "clm_ml.processes.water_vapor",
        "clm_ml.processes.math",
        "clm_ml.processes.photosynthesis",
        "clm_ml.processes.turbulence",
        "clm_ml.processes.leaf_heat_capacity",
        "clm_ml.processes.canopy_water",
        "clm_ml.processes.leaf_boundary_layer",
        "clm_ml.processes.integrators",
        "clm_ml.processes.nitrogen_profile",
        "clm_ml.state",
        "clm_ml.processes.constants",
        "clm_ml.processes.orbit",
    ]:
        mods[name] = importlib.import_module(name)
    return mods


_M: dict = {}  # populated on first use by _get


def _get(mod_dotted: str):
    global _M
    if not _M:
        _M = _import_jax_modules()
    return _M[mod_dotted]


# ---------------------------------------------------------------------------
# Adapters — one per executable
# ---------------------------------------------------------------------------


def _adapt_satvap(inp: dict) -> dict:
    SatVap = _get("clm_ml.processes.water_vapor").sat_vap
    es, desdt = SatVap(inp["t"])
    return {"es": float(es), "desdt": float(desdt)}


def _adapt_latvap(inp: dict) -> dict:
    LatVap = _get("clm_ml.processes.water_vapor").lat_vap
    lam = LatVap(inp["t"])
    return {"lambda": float(lam)}


def _adapt_quadratic(inp: dict) -> dict:
    quadratic = _get("clm_ml.processes.math").quadratic
    r1, r2 = quadratic(inp["a"], inp["b"], inp["c"])
    return {"r1": float(r1), "r2": float(r2)}


def _adapt_log_gamma(inp: dict) -> dict:
    log_gamma_function = _get("clm_ml.processes.math").log_gamma_function
    return {"gammaln": float(log_gamma_function(inp["x"]))}


def _adapt_beta_function(inp: dict) -> dict:
    beta_function = _get("clm_ml.processes.math").beta_function
    return {"beta": float(beta_function(inp["a"], inp["b"]))}


def _adapt_beta_pdf(inp: dict) -> dict:
    beta_distribution_pdf = _get("clm_ml.processes.math").beta_distribution_pdf
    return {"beta_pdf": float(beta_distribution_pdf(inp["a"], inp["b"], inp["x"]))}


def _adapt_beta_cdf(inp: dict) -> dict:
    beta_distribution_cdf = _get("clm_ml.processes.math").beta_distribution_cdf
    return {"beta_cdf": float(beta_distribution_cdf(inp["a"], inp["b"], inp["x"]))}


def _adapt_tridiag(inp: dict) -> dict:
    """tridiag takes 0-based rows; the golden outputs are named u_1..u_n (Fortran's rows).
    Fortran's unguarded pivots (the guarded variant: tests/clm_ml/test_processes_math.py)."""
    tridiag = _get("clm_ml.processes.math").tridiag
    n = int(inp["n"])

    def _arr(key):
        return jnp.array(inp.get(key, [0.0] * n)[:n], dtype=jnp.float64)

    u = tridiag(_arr("a"), _arr("b"), _arr("c"), _arr("r"))
    return {f"u_{i + 1}": float(u[i]) for i in range(n)}


def _adapt_tridiag_2eq(inp: dict) -> dict:
    """tridiag_2eq uses 0-based indexing; missing off-diagonal keys default to 0."""
    tridiag_2eq = _get("clm_ml.processes.math").tridiag_2eq
    n = int(inp["n"])

    def _arr(key):
        raw = inp.get(key, [0.0] * n)
        return jnp.array(raw[:n], dtype=jnp.float64)

    a1 = _arr("a1")
    b11 = _arr("b11")
    b12 = _arr("b12")
    c1 = _arr("c1")
    d1 = _arr("d1")
    a2 = _arr("a2")
    b21 = _arr("b21")
    b22 = _arr("b22")
    c2 = _arr("c2")
    d2 = _arr("d2")

    t, q = tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
    out = {}
    for i in range(n):
        out[f"t_{i+1}"] = float(t[i])
        out[f"q_{i+1}"] = float(q[i])
    return out


def _adapt_ft(inp: dict) -> dict:
    _ft = _get("clm_ml.processes.photosynthesis").ft
    return {"ans": float(_ft(inp["tl"], inp["ha"]))}


def _adapt_fth(inp: dict) -> dict:
    _fth = _get("clm_ml.processes.photosynthesis").fth
    return {"ans": float(_fth(inp["tl"], inp["hd"], inp["se"], inp["c"]))}


def _adapt_fth25(inp: dict) -> dict:
    _fth25 = _get("clm_ml.processes.photosynthesis").fth25
    return {"ans": float(_fth25(inp["hd"], inp["se"]))}


def _adapt_realized_rate(inp: dict) -> dict:
    """``realized_rate`` takes the pathway and co-limitation as switches."""
    photo = _get("clm_ml.processes.photosynthesis")
    agross = photo.realized_rate(
        inp["ac"],
        inp["aj"],
        inp["ap"],
        c3=round(inp["c3psn"]) == 1,
        colimitation=int(inp["colim_type_in"]) == 1,
    )
    return {"agross": float(agross)}


def _adapt_phim_monin_obukhov(inp: dict) -> dict:
    turb = _get("clm_ml.processes.turbulence")
    zeta = inp["zeta"]
    return {
        "phim": float(turb.phim_monin_obukhov(zeta)),
        "phic": float(turb.phic_monin_obukhov(zeta)),
    }


def _adapt_psim_monin_obukhov(inp: dict) -> dict:
    turb = _get("clm_ml.processes.turbulence")
    zeta = inp["zeta"]
    return {
        "psim": float(turb.psim_monin_obukhov(zeta)),
        "psic": float(turb.psic_monin_obukhov(zeta)),
    }


def _adapt_get_beta(inp: dict) -> dict:
    turb = _get("clm_ml.processes.turbulence")
    beta = turb.beta_stability(inp["beta_neutral"], inp["LcL"])
    return {"beta": float(beta)}


def _adapt_get_prsc(inp: dict) -> dict:
    """``prandtl_schmidt`` takes the sparse-canopy switch as an argument."""
    turb = _get("clm_ml.processes.turbulence")
    sparse_in = int(inp["sparse_canopy_type_in"]) == 1
    PrSc = turb.prandtl_schmidt(inp["beta_neutral"], inp["LcL"], sparse_in)
    return {"PrSc": float(PrSc)}


def _adapt_get_psi_rsl(inp: dict) -> dict:
    """``psi_rsl`` with the package's RSL lookup tables (read in conftest)."""
    from tests.fortran_validation.conftest import RSL_AVAILABLE

    if not RSL_AVAILABLE:
        pytest.skip("RSL psihat lookup tables unavailable (netCDF4 load failed)")
    turb = _get("clm_ml.processes.turbulence")
    from clm_ml.io.rsl_tables import default_rsl_tables

    psim, psic, psim2, psim_hat2 = turb.psi_rsl(
        inp["za"],
        inp["hc"],
        inp["disp"],
        inp["obu"],
        inp["beta"],
        inp["PrSc"],
        default_rsl_tables(),
    )
    return {
        "psim": float(psim),
        "psic": float(psic),
        "psim2": float(psim2),
        "psim_hat2": float(psim_hat2),
    }


def _adapt_wetted_fraction(inp: dict) -> dict:
    """CanopyWettedFraction for a single layer (``clm_ml.processes.canopy_water``)."""
    wetted_fraction = _get("clm_ml.processes.canopy_water").wetted_fraction
    fwet, fdry = wetted_fraction(
        jnp.array([inp["h2ocan"]]), jnp.array([inp["dpai"]]), jnp.array([inp["dlai"]])
    )
    return {"fwet": float(fwet[0]), "fdry": float(fdry[0])}


def _adapt_leaf_boundary_layer(inp: dict) -> dict:
    """``leaf_boundary_layer`` for one layer with plant area and one leaf; the regime from
    the golden input's ``gb_type``."""
    lbl = _get("clm_ml.processes.leaf_boundary_layer")
    regime = {1: "laminar", 2: "turbulent", 3: "free_convection"}[int(inp["gb_type_in"])]
    g = lbl.leaf_boundary_layer(
        jnp.array([1.0]),  # dpai > 0 so the layer is active
        jnp.array([float(inp["u"])]),
        jnp.array([float(inp["tair"])]),
        jnp.array([[float(inp["tleaf"])]]),
        float(inp["pref"]),
        float(inp["tref"]),
        float(inp["rhomol"]),
        float(inp["d"]),
        regime=regime,
    )
    return {"gbh": float(g.gbh[0, 0]), "gbv": float(g.gbv[0, 0]), "gbc": float(g.gbc[0, 0])}


def _adapt_leaf_heat_capacity(inp: dict) -> dict:
    """LeafHeatCapacity for a single layer with plant area (``dpai = 1``)."""
    leaf_heat_capacity = _get("clm_ml.processes.leaf_heat_capacity").leaf_heat_capacity
    cpleaf = leaf_heat_capacity(jnp.array([1.0]), jnp.asarray(float(inp["sla"])))
    return {"cpleaf": float(cpleaf[0])}


def _adapt_runge_kutta_ini(inp: dict) -> dict:
    """RungeKuttaIni: the Butcher tableau (a, b, c) of the Fortran default
    ``runge_kutta_type = 41`` (``rk4``), as when the golden data were recorded.

    The arrays are 0-based; golden keys use 1-based indices. Unused upper-triangular entries
    in ``a`` hold spval (1e36) matching Fortran.
    """
    a, b, c = _get("clm_ml.processes.integrators").TABLEAUS["rk4"]
    nrk = len(b)
    out = {}
    for i in range(1, nrk + 1):
        out[f"b_{i}"] = float(b[i - 1])
        out[f"c_{i}"] = float(c[i - 1])
    for i in range(1, nrk + 1):
        for j in range(1, nrk + 1):
            out[f"a_{i}_{j}"] = float(a[i - 1, j - 1])
    return out


def _adapt_nitrogen_scale(inp: dict) -> dict:
    """NitrogenScale: the sunlit and shaded nitrogen scaling factors of one layer
    (``physics.nitrogen_profile.nitrogen_scale``); 0 without plant area, as Fortran."""
    physics = _get("clm_ml.processes.nitrogen_profile")
    if float(inp["dpai"]) <= 0.0:
        return {"nscale_sun": 0.0, "nscale_sha": 0.0}
    vary = int(inp["leaf_optics_type_in"]) == 1
    v = {k: jnp.asarray(float(inp[k])) for k in ("kn", "pai_above", "dpai", "kb", "tbi",
                                                  "fracsun", "clump_fac")}  # fmt: skip
    fn, fn_sun = physics.nitrogen_fractions(
        v["kn"], v["pai_above"], v["dpai"], v["kb"], v["tbi"], v["clump_fac"],
        vary_with_height=vary,
    )  # fmt: skip
    sun, sha = physics.nitrogen_scale(fn, fn_sun, v["dpai"], v["fracsun"], vary_with_height=vary)
    return {"nscale_sun": float(sun), "nscale_sha": float(sha)}


def _adapt_canopy_nitrogen_profile(inp: dict) -> dict:
    """CanopyNitrogenProfile of a C3 canopy without acclimation, the decay coefficient
    given (``kn_val``); the scaling factors are the leaf Vcmax25 over ``vcmax25top``."""
    physics = _get("clm_ml.processes.nitrogen_profile")
    n = len(inp["dpai"])
    ncan = int(inp["ncan"])
    layers = _get("clm_ml.state").PatchLayers(
        n_layers=jnp.asarray(ncan), n_within=jnp.asarray(ncan), first_leafy=jnp.asarray(0),
        dz=jnp.zeros(n), zs=jnp.zeros(n), zw=jnp.zeros(n + 1),
    )  # fmt: skip
    dpai = jnp.asarray(inp["dpai"], dtype=jnp.float64)
    top = float(inp["vcmax25top"])
    r = physics.canopy_nitrogen_profile(
        layers, dpai, *(jnp.asarray(inp[k], dtype=jnp.float64) for k in ("kb", "tbi", "fracsun")),
        float(inp["lai"]), float(inp["sai"]), 0.0, top, float(inp["clump_fac"]),
        c3=True, acclimation=False, kn_val=float(inp["kn"]),
        vary_with_height=int(inp["leaf_optics_type_in"]) == 1,
    )  # fmt: skip
    numerical = float(jnp.sum(r.vcmax25_profile * dpai))
    out = {"numerical": numerical, "analytical": numerical - float(r.integration_error)}
    for ic in range(ncan):
        out[f"nscale_sun_{ic + 1:02d}"] = float(r.vcmax25_leaf[ic, 0]) / top
        out[f"nscale_sha_{ic + 1:02d}"] = float(r.vcmax25_leaf[ic, 1]) / top
        out[f"vcmax25_profile_{ic + 1:02d}"] = float(r.vcmax25_profile[ic])
    return out


def _adapt_shr_orb_params(inp: dict) -> dict:
    shr_orb_mod = _get("clm_ml.processes.orbit")
    # shr_orb_params returns (eccen, obliq, mvelp, obliqr, lambm0, mvelpp)
    eccen, obliq, mvelp, obliqr, lambm0, mvelpp = shr_orb_mod.shr_orb_params(int(inp["iyear_AD"]))
    return {
        "eccen": float(eccen),
        "obliq": float(obliq),
        "mvelp": float(mvelp),
        "obliqr": float(obliqr),
        "lambm0": float(lambm0),
        "mvelpp": float(mvelpp),
    }


def _adapt_shr_orb_decl(inp: dict) -> dict:
    """
    Chains shr_orb_params (to get orbital constants) then shr_orb_decl.

    The Fortran test driver calls both for a given iyear_AD + calday.
    The golden outputs include eccen and obliqr (from params) as well as
    delta and eccf (from decl).
    shr_orb_params returns (eccen, obliq, mvelp, obliqr, lambm0, mvelpp).
    """
    shr_orb_mod = _get("clm_ml.processes.orbit")
    eccen, obliq, mvelp, obliqr, lambm0, mvelpp = shr_orb_mod.shr_orb_params(int(inp["iyear_AD"]))
    delta, eccf = shr_orb_mod.shr_orb_decl(float(inp["calday"]), eccen, mvelpp, lambm0, obliqr)
    return {
        "eccen": float(eccen),
        "obliqr": float(obliqr),
        "delta": float(delta),
        "eccf": float(eccf),
    }


# ---------------------------------------------------------------------------
# Dispatcher: maps executable name → adapter function
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, Callable[[dict], dict]] = {
    "test_MLWaterVaporMod_SatVap.exe": _adapt_satvap,
    "test_MLWaterVaporMod_LatVap.exe": _adapt_latvap,
    "test_MLMathToolsMod_quadratic.exe": _adapt_quadratic,
    "test_MLMathToolsMod_log_gamma_function.exe": _adapt_log_gamma,
    "test_MLMathToolsMod_beta_function.exe": _adapt_beta_function,
    "test_MLMathToolsMod_beta_distribution_pdf.exe": _adapt_beta_pdf,
    "test_MLMathToolsMod_beta_distribution_cdf.exe": _adapt_beta_cdf,
    "test_MLMathToolsMod_tridiag.exe": _adapt_tridiag,
    "test_MLMathToolsMod_tridiag_2eq.exe": _adapt_tridiag_2eq,
    "test_MLLeafPhotosynthesisMod_ft.exe": _adapt_ft,
    "test_MLLeafPhotosynthesisMod_fth.exe": _adapt_fth,
    "test_MLLeafPhotosynthesisMod_fth25.exe": _adapt_fth25,
    "test_MLLeafPhotosynthesisMod_RealizedRate.exe": _adapt_realized_rate,
    "test_MLCanopyTurbulenceMod_phim_monin_obukhov.exe": _adapt_phim_monin_obukhov,
    "test_MLCanopyTurbulenceMod_psim_monin_obukhov.exe": _adapt_psim_monin_obukhov,
    "test_MLCanopyTurbulenceMod_GetBeta.exe": _adapt_get_beta,
    "test_MLCanopyTurbulenceMod_GetPrSc.exe": _adapt_get_prsc,
    "test_MLCanopyTurbulenceMod_GetPsiRSL.exe": _adapt_get_psi_rsl,
    "test_MLCanopyWaterMod_WettedFraction.exe": _adapt_wetted_fraction,
    "test_MLLeafBoundaryLayerMod_LeafBoundaryLayer.exe": _adapt_leaf_boundary_layer,
    "test_MLLeafHeatCapacityMod_LeafHeatCapacity.exe": _adapt_leaf_heat_capacity,
    "test_MLRungeKuttaMod_RungeKuttaIni.exe": _adapt_runge_kutta_ini,
    "test_MLCanopyNitrogenProfileMod_NitrogenScale.exe": _adapt_nitrogen_scale,
    "test_MLCanopyNitrogenProfileMod_CanopyNitrogenProfile.exe": _adapt_canopy_nitrogen_profile,
    "test_shr_orb_mod_shr_orb_params.exe": _adapt_shr_orb_params,
    "test_shr_orb_mod_shr_orb_decl.exe": _adapt_shr_orb_decl,
}


# ---------------------------------------------------------------------------
# Golden data loading
# ---------------------------------------------------------------------------


def _load_all_cases() -> list:
    if not _GOLDEN_DIR.is_dir():
        return []

    params = []
    for fname in sorted(os.listdir(_GOLDEN_DIR)):
        if not fname.endswith(".json"):
            continue
        path = _GOLDEN_DIR / fname
        with open(path) as f:
            data = json.load(f)
        exe = data["executable"]
        if exe not in ADAPTERS:
            continue
        for idx, case in enumerate(data["cases"]):
            params.append(
                pytest.param(
                    exe,
                    case["inputs"],
                    case["outputs"],
                    id=f"{fname[:-5]}[{idx}]",
                    marks=pytest.mark.fortran_validation,
                )
            )
    return params


_ALL_CASES = _load_all_cases()

if not _ALL_CASES:
    pytest.skip(
        "No golden data found in tests/fortran_validation/golden_IO/. "
        "Ensure the golden_IO/ directory contains JSON files.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Parametrized golden test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exe, inputs, expected", _ALL_CASES)
def test_golden_jax(exe: str, inputs: dict, expected: dict):
    """
    Run one golden case through the JAX implementation and compare against
    the Fortran reference value for every output key.

    Tolerance: REL_TOL=1e-9 relative, ABS_TOL=1e-15 absolute guard.
    """
    adapter = ADAPTERS[exe]
    actual = adapter(inputs)

    for key, golden_val in expected.items():
        assert key in actual, (
            f"{exe}: output key '{key}' missing from adapter result.\n"
            f"  inputs  = {inputs}\n"
            f"  got keys = {sorted(actual.keys())}"
        )
        got = float(actual[key])
        diff = abs(got - golden_val)
        tol = REL_TOL * abs(golden_val) + ABS_TOL
        assert diff <= tol, (
            f"{exe}: output '{key}' differs from Fortran golden value.\n"
            f"  inputs  = {inputs}\n"
            f"  got     = {got!r}\n"
            f"  golden  = {golden_val!r}\n"
            f"  |diff|  = {diff:.3e}  (tol = {tol:.3e})"
        )
