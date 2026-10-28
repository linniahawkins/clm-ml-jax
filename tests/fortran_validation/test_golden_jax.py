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

import contextlib
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
        "multilayer_canopy.MLWaterVaporMod",
        "multilayer_canopy.MLMathToolsMod",
        "multilayer_canopy.MLLeafPhotosynthesisMod",
        "multilayer_canopy.MLCanopyTurbulenceMod",
        "multilayer_canopy.MLLeafHeatCapacityMod",
        "multilayer_canopy.MLLeafBoundaryLayerMod",
        "multilayer_canopy.MLRungeKuttaMod",
        "multilayer_canopy.MLclm_varcon",
        "multilayer_canopy.MLclm_varctl",
        "clm_src_main.clm_varcon",
        "clm_share.shr_orb_mod",
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
# Helper: temporary module-global override
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _override(module_path: str, attr: str, value):
    """Temporarily set a module-level name and restore it afterward."""
    mod = _get(module_path)
    old = getattr(mod, attr)
    setattr(mod, attr, value)
    try:
        yield
    finally:
        setattr(mod, attr, old)


# ---------------------------------------------------------------------------
# Adapters — one per executable
# ---------------------------------------------------------------------------


def _adapt_satvap(inp: dict) -> dict:
    SatVap = _get("multilayer_canopy.MLWaterVaporMod").SatVap
    es, desdt = SatVap(inp["t"])
    return {"es": float(es), "desdt": float(desdt)}


def _adapt_latvap(inp: dict) -> dict:
    LatVap = _get("multilayer_canopy.MLWaterVaporMod").LatVap
    lam = LatVap(inp["t"])
    return {"lambda": float(lam)}


def _adapt_quadratic(inp: dict) -> dict:
    quadratic = _get("multilayer_canopy.MLMathToolsMod").quadratic
    r1, r2 = quadratic(inp["a"], inp["b"], inp["c"])
    return {"r1": float(r1), "r2": float(r2)}


def _adapt_log_gamma(inp: dict) -> dict:
    log_gamma_function = _get("multilayer_canopy.MLMathToolsMod").log_gamma_function
    return {"gammaln": float(log_gamma_function(inp["x"]))}


def _adapt_beta_function(inp: dict) -> dict:
    beta_function = _get("multilayer_canopy.MLMathToolsMod").beta_function
    return {"beta": float(beta_function(inp["a"], inp["b"]))}


def _adapt_beta_pdf(inp: dict) -> dict:
    beta_distribution_pdf = _get("multilayer_canopy.MLMathToolsMod").beta_distribution_pdf
    return {"beta_pdf": float(beta_distribution_pdf(inp["a"], inp["b"], inp["x"]))}


def _adapt_beta_cdf(inp: dict) -> dict:
    beta_distribution_cdf = _get("multilayer_canopy.MLMathToolsMod").beta_distribution_cdf
    return {"beta_cdf": float(beta_distribution_cdf(inp["a"], inp["b"], inp["x"]))}


def _adapt_tridiag(inp: dict) -> dict:
    """tridiag uses 1-based indexing: prepend a sentinel 0 to each array."""
    tridiag = _get("multilayer_canopy.MLMathToolsMod").tridiag
    n = int(inp["n"])

    # Golden arrays are 0-indexed length-n; JAX tridiag wants 1-based length-(n+1)
    def _pad(key):
        arr_raw = inp.get(key, [0.0] * n)
        return jnp.array([0.0] + list(arr_raw), dtype=jnp.float64)

    a = _pad("a")
    b = _pad("b")
    c = _pad("c")
    r = _pad("r")
    u = tridiag(a, b, c, r, n)
    return {f"u_{i}": float(u[i]) for i in range(1, n + 1)}


def _adapt_tridiag_2eq(inp: dict) -> dict:
    """tridiag_2eq uses 0-based indexing; missing off-diagonal keys default to 0."""
    tridiag_2eq = _get("multilayer_canopy.MLMathToolsMod").tridiag_2eq
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
    _ft = _get("multilayer_canopy.MLLeafPhotosynthesisMod")._ft
    return {"ans": float(_ft(inp["tl"], inp["ha"]))}


def _adapt_fth(inp: dict) -> dict:
    _fth = _get("multilayer_canopy.MLLeafPhotosynthesisMod")._fth
    return {"ans": float(_fth(inp["tl"], inp["hd"], inp["se"], inp["c"]))}


def _adapt_fth25(inp: dict) -> dict:
    _fth25 = _get("multilayer_canopy.MLLeafPhotosynthesisMod")._fth25
    return {"ans": float(_fth25(inp["hd"], inp["se"]))}


def _adapt_realized_rate(inp: dict) -> dict:
    photo_mod = _get("multilayer_canopy.MLLeafPhotosynthesisMod")
    colim_in = int(inp["colim_type_in"])
    with _override("multilayer_canopy.MLLeafPhotosynthesisMod", "colim_type", colim_in):
        agross = photo_mod._RealizedRate(inp["c3psn"], inp["ac"], inp["aj"], inp["ap"])
    return {"agross": float(agross)}


def _adapt_phim_monin_obukhov(inp: dict) -> dict:
    turb = _get("multilayer_canopy.MLCanopyTurbulenceMod")
    zeta = inp["zeta"]
    return {
        "phim": float(turb._phim_monin_obukhov(zeta)),
        "phic": float(turb._phic_monin_obukhov(zeta)),
    }


def _adapt_psim_monin_obukhov(inp: dict) -> dict:
    turb = _get("multilayer_canopy.MLCanopyTurbulenceMod")
    zeta = inp["zeta"]
    return {
        "psim": float(turb._psim_monin_obukhov(zeta)),
        "psic": float(turb._psic_monin_obukhov(zeta)),
    }


def _adapt_get_beta(inp: dict) -> dict:
    turb = _get("multilayer_canopy.MLCanopyTurbulenceMod")
    beta = turb._GetBeta(inp["beta_neutral"], inp["LcL"])
    return {"beta": float(beta)}


def _adapt_get_prsc(inp: dict) -> dict:
    """_GetPrSc reads sparse_canopy_type from its module namespace."""
    turb = _get("multilayer_canopy.MLCanopyTurbulenceMod")
    varcon = _get("multilayer_canopy.MLclm_varcon")
    sparse_in = int(inp["sparse_canopy_type_in"])
    with _override("multilayer_canopy.MLCanopyTurbulenceMod", "sparse_canopy_type", sparse_in):
        PrSc = turb._GetPrSc(inp["beta_neutral"], varcon.beta_neutral_max, inp["LcL"])
    return {"PrSc": float(PrSc)}


def _adapt_get_psi_rsl(inp: dict) -> dict:
    """_GetPsiRSL uses RSL lookup tables (initialized in conftest)."""
    from tests.fortran_validation.conftest import RSL_AVAILABLE

    if not RSL_AVAILABLE:
        pytest.skip("RSL psihat lookup tables unavailable (netCDF4 load failed)")
    turb = _get("multilayer_canopy.MLCanopyTurbulenceMod")
    psim, psic, psim2, psim_hat2 = turb._GetPsiRSL(
        inp["za"], inp["hc"], inp["disp"], inp["obu"], inp["beta"], inp["PrSc"]
    )
    return {
        "psim": float(psim),
        "psic": float(psic),
        "psim2": float(psim2),
        "psim_hat2": float(psim_hat2),
    }


def _adapt_wetted_fraction(inp: dict) -> dict:
    """
    CanopyWettedFraction inner math (single layer).

    The public JAX function takes mlcanopy_type state; here we apply the
    per-layer formula directly using constants from MLclm_varcon.
    """
    varcon = _get("multilayer_canopy.MLclm_varcon")
    h2ocan = float(inp["h2ocan"])
    dpai = float(inp["dpai"])
    dlai = float(inp["dlai"])

    if dpai <= 0.0:
        return {"fwet": 0.0, "fdry": 0.0}

    h2ocanmx = varcon.dewmx * dpai
    fwet_base = max(h2ocan / h2ocanmx, 1.0e-30)
    fwet = min(fwet_base**varcon.fwet_exponent, varcon.maximum_leaf_wetted_fraction)
    fdry = (1.0 - fwet) * dlai / dpai
    return {"fwet": fwet, "fdry": fdry}


def _adapt_leaf_boundary_layer(inp: dict) -> dict:
    """
    LeafBoundaryLayer inner kernel (_gb_layer) for a single layer.

    Computes molecular diffusivity correction from tref and pref, then
    calls the per-layer kernel with gb_type set from the golden input.
    gb_type is a module-local binding in MLLeafBoundaryLayerMod.
    """
    lbl_mod = _get("multilayer_canopy.MLLeafBoundaryLayerMod")
    varcon = _get("multilayer_canopy.MLclm_varcon")
    clm_var = _get("clm_src_main.clm_varcon")

    d = float(inp["d"])
    u = float(inp["u"])
    tleaf = float(inp["tleaf"])
    tair = float(inp["tair"])
    tref = float(inp["tref"])
    pref = float(inp["pref"])
    rhomol = float(inp["rhomol"])
    gb_in = int(inp["gb_type_in"])

    tfrz = clm_var.tfrz
    fac = 101325.0 / pref * (tref / tfrz) ** 1.81
    visc = varcon.visc0 * fac
    dh = varcon.dh0 * fac
    dv = varcon.dv0 * fac
    dc = varcon.dc0 * fac
    dv_dh = dv / dh
    dc_dh = dc / dh

    with _override("multilayer_canopy.MLLeafBoundaryLayerMod", "gb_type", gb_in):
        gbh, gbv, gbc = lbl_mod._gb_layer(
            jnp.asarray(1.0),  # dpai > 0 so the layer is active
            jnp.asarray(u),
            jnp.asarray(tair),
            jnp.asarray(tleaf),
            visc,
            dh,
            dv_dh,
            dc_dh,
            d,
            rhomol,
        )

    return {"gbh": float(gbh), "gbv": float(gbv), "gbc": float(gbc)}


def _adapt_leaf_heat_capacity(inp: dict) -> dict:
    """
    LeafHeatCapacity inner kernel (_cpleaf_layer) for a single layer.

    dpai is set to 1.0 so the layer is active (the mask dpai > 0 is True).
    """
    lbl_mod = _get("multilayer_canopy.MLLeafHeatCapacityMod")
    sla = float(inp["sla"])
    cpleaf = lbl_mod._cpleaf_layer(jnp.asarray(1.0), jnp.asarray(sla))
    return {"cpleaf": float(cpleaf)}


def _adapt_runge_kutta_ini(inp: dict) -> dict:
    """RungeKuttaIni returns (a, b, c) coefficient arrays for the selected RK method.

    The JAX arrays are 0-indexed; golden keys use 1-based indices.
    Unused upper-triangular entries in ``a`` hold spval (1e36) matching Fortran.
    """
    rk_mod = _get("multilayer_canopy.MLRungeKuttaMod")
    a, b, c = rk_mod.RungeKuttaIni()  # returns (a, b, c) — note order
    nrk = len(b)  # number of stages from array length
    out = {}
    for i in range(1, nrk + 1):
        out[f"b_{i}"] = float(b[i - 1])  # 0-based access
        out[f"c_{i}"] = float(c[i - 1])
    for i in range(1, nrk + 1):
        for j in range(1, nrk + 1):
            out[f"a_{i}_{j}"] = float(a[i - 1, j - 1])
    return out


def _adapt_nitrogen_scale(inp: dict) -> dict:
    """
    Per-layer nitrogen scaling (NitrogenScale).

    This logic is inline inside CanopyNitrogenProfile in JAX.  We
    implement the documented formulas directly so the golden values can
    be verified without requiring full mlcanopy_type state initialization.

    Formulas (Bonan et al. 2021, supplemental eqs. 6-12):

      fn = exp(-kn*pai_above) * (1 - exp(-kn*dpai)) / kn

      leaf_optics_type == 0:
        fn_sun = clump / (kn + kb*clump)
                 * exp(-kn*pai_above) * tbi * (1 - exp(-(kn+kb*clump)*dpai))
        fn_sha = fn - fn_sun
        nscale_sun = fn_sun / (fracsun * dpai)
        nscale_sha = fn_sha / ((1-fracsun) * dpai)

      leaf_optics_type == 1:
        nscale_sun = nscale_sha = fn / dpai
    """
    kn = float(inp["kn"])
    pai_above = float(inp["pai_above"])
    dpai = float(inp["dpai"])
    kb = float(inp["kb"])
    clump_fac = float(inp["clump_fac"])
    fracsun = float(inp["fracsun"])
    tbi = float(inp["tbi"])
    leaf_optics_type = int(inp["leaf_optics_type_in"])

    if dpai <= 0.0:
        return {"nscale_sun": 0.0, "nscale_sha": 0.0}

    # L'Hôpital limit as kn→0: exp(-kn*pai_above)*(1-exp(-kn*dpai))/kn → dpai
    if kn == 0.0:
        fn = dpai
    else:
        fn = math.exp(-kn * pai_above) * (1.0 - math.exp(-kn * dpai)) / kn

    if leaf_optics_type == 0:
        fn_sun = (
            clump_fac
            / (kn + kb * clump_fac)
            * math.exp(-kn * pai_above)
            * tbi
            * (1.0 - math.exp(-(kn + kb * clump_fac) * dpai))
        )
        fn_sha = fn - fn_sun
        nscale_sun = fn_sun / (fracsun * dpai) if fracsun > 0.0 else 0.0
        nscale_sha = fn_sha / ((1.0 - fracsun) * dpai) if fracsun < 1.0 else 0.0
    else:
        nscale_sun = nscale_sha = fn / dpai

    return {"nscale_sun": nscale_sun, "nscale_sha": nscale_sha}


def _adapt_shr_orb_params(inp: dict) -> dict:
    shr_orb_mod = _get("clm_share.shr_orb_mod")
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
    shr_orb_mod = _get("clm_share.shr_orb_mod")
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
    "test_shr_orb_mod_shr_orb_params.exe": _adapt_shr_orb_params,
    "test_shr_orb_mod_shr_orb_decl.exe": _adapt_shr_orb_decl,
    # test_MLCanopyNitrogenProfileMod_CanopyNitrogenProfile.exe:
    #   Requires full mlcanopy_type state initialization — covered separately.
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
