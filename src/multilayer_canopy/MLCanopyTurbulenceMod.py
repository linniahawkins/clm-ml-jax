"""
JAX translation of MLCanopyTurbulenceMod Fortran module.

Canopy turbulence parameterisation: roughness sublayer (RSL) theory
of Harman and Finnigan (2008), wind speed profile, and aerodynamic
conductances.

Public routines
---------------
- :func:`CanopyTurbulence`: main driver.
- :func:`LookupPsihatINI`: initialise RSL psihat look-up tables from
  a NetCDF file.

Private helpers
---------------
- :func:`_HF2008`: HF2008 RSL canopy turbulence.
- :func:`_GetObu`: solve for Obukhov length.
- :func:`_GetBeta`: β = u*/u(h) for a given stability.
- :func:`_GetPrSc`: Prandtl/Schmidt number at canopy top.
- :func:`_GetPsiRSL`: RSL-modified stability functions.
- :func:`_phim_monin_obukhov`: MO φ for momentum.
- :func:`_phic_monin_obukhov`: MO φ for scalars.
- :func:`_psim_monin_obukhov`: MO ψ for momentum.
- :func:`_psic_monin_obukhov`: MO ψ for scalars.
- :func:`_LookupPsihat`: bilinear interpolation in psihat table.
- :func:`_RoughnessLength`: roughness length for momentum.
- :func:`_WindProfile`: wind speed above and within canopy.
- :func:`_AerodynamicConductance`: aerodynamic conductances.

Original Fortran module: MLCanopyTurbulenceMod
Fortran lines 1-560
"""

from __future__ import annotations

import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_ml.physics.math import (  # noqa: F401
    bisection_traceable,
    is_concrete,
)
from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_varcon import grav, vkc  # noqa: F401
from clm_src_main.clm_varcon import pi as rpi
from clm_src_main.clm_varctl import iulog, rslfile  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from multilayer_canopy.MLclm_varcon import (  # noqa: F401
    LcL_max,
    LcL_min,
    Pr0,
    Pr1,
    Pr2,
    aH12,
    beta_neutral_max,
    c2,
    cd,
    cr,
    dtLgridH,
    dtLgridM,
    eta_max,
    mmdry,
    mmh2o,
    nL,
    nZ,
    psigridH,
    psigridM,
    ra_max,
    z0mg,
    zdtgridH,
    zdtgridM,
)
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions
from multilayer_canopy.MLLayersMod import Layers, keep, layers, pad_with_top

# Python-float aliases for aH12 elements — used inside _ObuFuncPure which is called
# on every solver iteration; accessing aH12[i] (a JAX array) would force an XLA sync
# each time, adding ~100–200 syncs per _GetObu call.
_aH12_0: float = 0.89
_aH12_1: float = -0.07
_aH12_2: float = 2.19

# ---------------------------------------------------------------------------
# Cached 1D views of psihat look-up table grids — populated by LookupPsihatINI.
# _LookupPsihat creates a 276-element negated array (-zdtg) on every call for
# np.searchsorted; with ~220 calls/patch/step that's ~110,000 allocations/run.
# Using module-level pre-computed arrays eliminates those allocations and the
# 2D→1D slice operations (dtLgrid[0], zdtgrid[:, 0]) done on every call.
# ---------------------------------------------------------------------------
_zdtgM_1d: np.ndarray = np.empty(0)  # zdtgridM[:, 0]  (populated after init)
_dtLgM_1d: np.ndarray = np.empty(0)  # dtLgridM[0]
_neg_zdtgM_1d: np.ndarray = np.empty(0)  # -zdtgridM[:, 0]
_nZ_M: int = nZ
_nL_M: int = nL
_zdtgH_1d: np.ndarray = np.empty(0)  # zdtgridH[:, 0]
_dtLgH_1d: np.ndarray = np.empty(0)  # dtLgridH[0]
_neg_zdtgH_1d: np.ndarray = np.empty(0)  # -zdtgridH[:, 0]
_nZ_H: int = nZ
_nL_H: int = nL

# Pre-converted JAX array versions of lookup tables — populated by LookupPsihatINI.
# _LookupPsihatM/H called jnp.asarray(...) on every invocation (~10K times per
# forward pass), triggering repeated CPU→GPU copies of the psihat grid.
# These module-level JAX arrays eliminate that overhead.
_dtLgM_jax: jax.Array | None = None
_zdtgM_jax: jax.Array | None = None
_neg_zdtgM_jax: jax.Array | None = None
_psigridM_jax: jax.Array | None = None
_dtLgH_jax: jax.Array | None = None
_zdtgH_jax: jax.Array | None = None
_neg_zdtgH_jax: jax.Array | None = None
_psigridH_jax: jax.Array | None = None

# ===========================================================================
# Private: Monin-Obukhov stability functions
# ===========================================================================


def _phim_monin_obukhov(zeta):
    """
    Monin-Obukhov φ stability function for momentum.

    Mirrors Fortran function ``phim_monin_obukhov`` (lines 330-343).

    Reference: Bonan et al. (2018), eq. (A10).

    .. code-block:: none

        zeta < 0 (unstable): phi = 1 / (1 - 16*zeta)^(1/4)
        zeta ≥ 0 (stable):   phi = 1 + 5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        φ for momentum.
    """
    zeta = jnp.asarray(zeta)
    # jnp.abs avoids sqrt(negative) NaN for stable (zeta>0) branch gradient
    unstable = 1.0 / jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    stable = 1.0 + 5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def _phic_monin_obukhov(zeta):
    """
    Monin-Obukhov φ stability function for scalars.

    Mirrors Fortran function ``phic_monin_obukhov`` (lines 345-358).

    Reference: Bonan et al. (2018), eq. (A11).

    .. code-block:: none

        zeta < 0 (unstable): phi = 1 / (1 - 16*zeta)^(1/2)
        zeta ≥ 0 (stable):   phi = 1 + 5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        φ for scalars.
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids sqrt(negative) NaN for stable (zeta>0) branch gradient
    unstable = 1.0 / jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10))
    stable = 1.0 + 5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def _psim_monin_obukhov(zeta):
    """
    Monin-Obukhov ψ stability function for momentum.

    Mirrors Fortran function ``psim_monin_obukhov`` (lines 360-378).

    Reference: Bonan et al. (2018), eq. (A12).

    .. code-block:: none

        zeta < 0 (unstable):
            x   = (1 - 16*zeta)^(1/4)
            psi = 2*log((1+x)/2) + log((1+x^2)/2) - 2*atan(x) + pi/2
        zeta ≥ 0 (stable):
            psi = -5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        ψ for momentum.
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids jnp.abs(0)^(-3/4) * sign(0) = inf * 0 = NaN in backward.
    # When zeta >= 0 the stable branch is selected; this branch expression is
    # discarded but its VJP is still evaluated. Using jnp.maximum clips the
    # derivative at the boundary (d(max(u,1e-10))/du = 0 when u <= 1e-10)
    # so the unselected-branch gradient is 0 * finite = 0, not 0 * inf = NaN.
    # Forward values are unchanged for any zeta actually selected (zeta < 0).
    x = jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x * x) / 2.0)
        - 2.0 * jnp.arctan(x)
        + rpi * 0.5
    )
    stable = -5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def _psic_monin_obukhov(zeta):
    """
    Monin-Obukhov ψ stability function for scalars.

    Mirrors Fortran function ``psic_monin_obukhov`` (lines 380-395).

    Reference: Bonan et al. (2018), eq. (A13).

    .. code-block:: none

        zeta < 0 (unstable):
            x   = (1 - 16*zeta)^(1/4)
            psi = 2*log((1+x^2)/2)
        zeta ≥ 0 (stable):
            psi = -5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        ψ for scalars.
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids jnp.abs(0)^(-3/4) * sign(0) = inf * 0 = NaN in backward.
    # Same reasoning as _psim_monin_obukhov above.
    x = jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    unstable = 2.0 * jnp.log((1.0 + x * x) / 2.0)
    stable = -5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


# ===========================================================================
# Private: RSL psihat look-up table interpolation
# ===========================================================================


def _LookupPsihat(
    zdt: float,
    dtL: float,
    zdtgrid,  # shape (nZ, 1)
    dtLgrid,  # shape (1, nL)
    psigrid,  # shape (nZ, nL)
) -> float:
    """
    Bilinear interpolation of the RSL psihat function from a look-up table.

    Mirrors Fortran subroutine ``LookupPsihat`` (lines 398-448).

    Reference: Bonan et al. (2018), eq. (A21).

    The returned value is the unscaled amplitude ``A[(z-h)/(h-d),(h-d)/L]``;
    it must be multiplied by ``c1`` before it fully represents psihat.

    Look-up table grid conventions (1-based in Fortran, 0-based here):

    - ``zdtgrid`` decreases with increasing row index (height above canopy
      decreasing toward zero as index increases); boundary clamping applies
      when ``zdt`` exceeds the table range.
    - ``dtLgrid`` increases with increasing column index; boundary clamping
      applies when ``dtL`` is outside the table range.

    Args:
        zdt: Normalised height above canopy ``(z-h)/(h-d)``.
        dtL: Stability ratio ``(h-d)/L``.
        zdtgrid: Grid of ``zdt`` values; shape ``(nZ, 1)``.
        dtLgrid: Grid of ``dtL`` values; shape ``(1, nL)``.
        psigrid: Psihat values on the grid; shape ``(nZ, nL)``.

    Returns:
        Interpolated (unscaled) psihat value.
    """
    dtLg = jnp.asarray(dtLgrid[0])  # 1-D, length nL, increasing
    zdtg = jnp.asarray(zdtgrid[:, 0])  # 1-D, length nZ, DECREASING
    nZ_tbl = len(zdtg)
    nL_tbl = len(dtLg)

    dtL = jnp.asarray(dtL)
    zdt = jnp.asarray(zdt)

    # --- dtL bracketing — Fortran lines 412-428 ---
    at_left_L = dtL <= dtLg[0]
    at_right_L = dtL > dtLg[-1]
    L2_raw = jnp.clip(jnp.searchsorted(dtLg, dtL, side="right"), 0, nL_tbl - 1)
    L1_raw = jnp.maximum(L2_raw - 1, 0)
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_L = jnp.maximum(dtLg[L2_raw] - dtLg[L1_raw], 1.0e-30)
    wL1_raw = (dtLg[L2_raw] - dtL) / denom_L
    wL2_raw = 1.0 - wL1_raw
    at_bnd_L = at_left_L | at_right_L
    wL1 = jnp.where(at_bnd_L, 0.5, wL1_raw)
    wL2 = jnp.where(at_bnd_L, 0.5, wL2_raw)
    L1 = jnp.where(at_left_L, 0, jnp.where(at_right_L, nL_tbl - 1, L1_raw))
    L2 = jnp.where(at_left_L, 0, jnp.where(at_right_L, nL_tbl - 1, L2_raw))

    # --- zdt bracketing — Fortran lines 430-448 (zdtg DECREASES) ---
    neg_zdtg = -zdtg
    at_top_Z = zdt > zdtg[0]
    at_bottom_Z = zdt < zdtg[-1]
    Z1_raw = jnp.clip(jnp.searchsorted(neg_zdtg, -zdt, side="right") - 1, 0, nZ_tbl - 2)
    Z2_raw = Z1_raw + 1
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_Z = jnp.maximum(zdtg[Z1_raw] - zdtg[Z2_raw], 1.0e-30)
    wZ1_raw = (zdt - zdtg[Z2_raw]) / denom_Z
    wZ2_raw = 1.0 - wZ1_raw
    at_bnd_Z = at_top_Z | at_bottom_Z
    wZ1 = jnp.where(at_bnd_Z, 0.5, wZ1_raw)
    wZ2 = jnp.where(at_bnd_Z, 0.5, wZ2_raw)
    Z1 = jnp.where(at_top_Z, 0, jnp.where(at_bottom_Z, nZ_tbl - 1, Z1_raw))
    Z2 = jnp.where(at_top_Z, 0, jnp.where(at_bottom_Z, nZ_tbl - 1, Z2_raw))

    psigrid = jnp.asarray(psigrid)
    # Bilinear interpolation — Fortran lines 450-452
    return (
        wZ1 * wL1 * psigrid[Z1, L1]
        + wZ2 * wL1 * psigrid[Z2, L1]
        + wZ1 * wL2 * psigrid[Z1, L2]
        + wZ2 * wL2 * psigrid[Z2, L2]
    )


# ---------------------------------------------------------------------------
# Fast specialized lookups using module-level pre-computed 1D arrays.
# These eliminate the per-call slice and negation overhead of _LookupPsihat.
# Called by _GetPsiRSL after LookupPsihatINI has populated the caches.
# ---------------------------------------------------------------------------


def _LookupPsihatM(zdt, dtL):
    """Momentum psihat lookup using module-level cached JAX arrays.

    Uses pre-converted _*_jax arrays (populated by LookupPsihatINI) to avoid
    jnp.asarray() on every call — eliminates ~10K GPU copies per forward pass.
    """
    dtL = jnp.asarray(dtL)
    zdt = jnp.asarray(zdt)
    # Use pre-converted JAX arrays; fall back to on-the-fly conversion only
    # before LookupPsihatINI has been called (should not happen in practice).
    dtLg = _dtLgM_jax if _dtLgM_jax is not None else jnp.asarray(_dtLgM_1d)
    zdtg = _zdtgM_jax if _zdtgM_jax is not None else jnp.asarray(_zdtgM_1d)
    neg_zdtg = _neg_zdtgM_jax if _neg_zdtgM_jax is not None else jnp.asarray(_neg_zdtgM_1d)
    nL = _nL_M
    nZ = _nZ_M

    at_bnd_L = (dtL <= dtLg[0]) | (dtL > dtLg[-1])
    L2_raw = jnp.clip(jnp.searchsorted(dtLg, dtL, side="right"), 0, nL - 1)
    L1_raw = jnp.maximum(L2_raw - 1, 0)
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_L = jnp.maximum(dtLg[L2_raw] - dtLg[L1_raw], 1.0e-30)
    wL1 = jnp.where(at_bnd_L, 0.5, (dtLg[L2_raw] - dtL) / denom_L)
    wL2 = 1.0 - wL1
    L1 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L1_raw))
    L2 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L2_raw))

    at_bnd_Z = (zdt > zdtg[0]) | (zdt < zdtg[-1])
    Z1_raw = jnp.clip(jnp.searchsorted(neg_zdtg, -zdt, side="right") - 1, 0, nZ - 2)
    Z2_raw = Z1_raw + 1
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_Z = jnp.maximum(zdtg[Z1_raw] - zdtg[Z2_raw], 1.0e-30)
    wZ1 = jnp.where(at_bnd_Z, 0.5, (zdt - zdtg[Z2_raw]) / denom_Z)
    wZ2 = 1.0 - wZ1
    Z1 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z1_raw))
    Z2 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z2_raw))

    pg = _psigridM_jax if _psigridM_jax is not None else jnp.asarray(psigridM)
    return (
        wZ1 * wL1 * pg[Z1, L1]
        + wZ2 * wL1 * pg[Z2, L1]
        + wZ1 * wL2 * pg[Z1, L2]
        + wZ2 * wL2 * pg[Z2, L2]
    )


def _LookupPsihatH(zdt, dtL):
    """Heat/scalar psihat lookup using module-level cached JAX arrays.

    Uses pre-converted _*_jax arrays (populated by LookupPsihatINI) to avoid
    jnp.asarray() on every call — eliminates ~10K GPU copies per forward pass.
    """
    dtL = jnp.asarray(dtL)
    zdt = jnp.asarray(zdt)
    dtLg = _dtLgH_jax if _dtLgH_jax is not None else jnp.asarray(_dtLgH_1d)
    zdtg = _zdtgH_jax if _zdtgH_jax is not None else jnp.asarray(_zdtgH_1d)
    neg_zdtg = _neg_zdtgH_jax if _neg_zdtgH_jax is not None else jnp.asarray(_neg_zdtgH_1d)
    nL = _nL_H
    nZ = _nZ_H

    at_bnd_L = (dtL <= dtLg[0]) | (dtL > dtLg[-1])
    L2_raw = jnp.clip(jnp.searchsorted(dtLg, dtL, side="right"), 0, nL - 1)
    L1_raw = jnp.maximum(L2_raw - 1, 0)
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_L = jnp.maximum(dtLg[L2_raw] - dtLg[L1_raw], 1.0e-30)
    wL1 = jnp.where(at_bnd_L, 0.5, (dtLg[L2_raw] - dtL) / denom_L)
    wL2 = 1.0 - wL1
    L1 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L1_raw))
    L2 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L2_raw))

    at_bnd_Z = (zdt > zdtg[0]) | (zdt < zdtg[-1])
    Z1_raw = jnp.clip(jnp.searchsorted(neg_zdtg, -zdt, side="right") - 1, 0, nZ - 2)
    Z2_raw = Z1_raw + 1
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_Z = jnp.maximum(zdtg[Z1_raw] - zdtg[Z2_raw], 1.0e-30)
    wZ1 = jnp.where(at_bnd_Z, 0.5, (zdt - zdtg[Z2_raw]) / denom_Z)
    wZ2 = 1.0 - wZ1
    Z1 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z1_raw))
    Z2 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z2_raw))

    pg = _psigridH_jax if _psigridH_jax is not None else jnp.asarray(psigridH)
    return (
        wZ1 * wL1 * pg[Z1, L1]
        + wZ2 * wL1 * pg[Z2, L1]
        + wZ1 * wL2 * pg[Z1, L2]
        + wZ2 * wL2 * pg[Z2, L2]
    )


# ===========================================================================
# Private: Prandtl / Schmidt number
# ===========================================================================


def _GetPrSc(
    beta_neutral: float,
    beta_neutral_max: float,
    LcL: float,
    sparse_canopy_type: int,
) -> float:
    """
    Calculate the Prandtl (= Schmidt) number at the canopy top.

    Mirrors Fortran subroutine ``GetPrSc`` (lines 292-310).

    Reference: Bonan et al. (2018), eqs. (A25), (A34).

    .. code-block:: none

        PrSc = Pr0 + Pr1 * tanh(Pr2 * LcL)

    For sparse canopies (``sparse_canopy_type == 1``):

    .. code-block:: none

        PrSc = (1 - beta_n/beta_n_max) * 1.0
               + (beta_n/beta_n_max) * PrSc

    Args:
        beta_neutral: Neutral β.
        beta_neutral_max: Maximum allowed neutral β.
        LcL: Canopy density scale / Obukhov length.

    Returns:
        PrSc value.
    """
    PrSc = Pr0 + Pr1 * jnp.tanh(Pr2 * LcL)  # Fortran line 304

    if sparse_canopy_type == 1:  # Fortran lines 306-308
        f = beta_neutral / beta_neutral_max
        PrSc = (1.0 - f) * 1.0 + f * PrSc

    return PrSc


# ===========================================================================
# Private: RSL-modified psi functions
# ===========================================================================


def _GetPsiRSL(
    za: float,
    hc: float,
    disp: float,
    obu: float,
    beta: float,
    PrSc: float,
) -> Tuple[float, float, float, float]:
    """
    RSL-modified ψ stability functions for momentum and scalars.

    Mirrors Fortran subroutine ``GetPsiRSL`` (lines 312-398).

    Reference: Bonan et al. (2018), appendix A2.

    Computes the integrated stability corrections between ``hc`` and
    ``za`` including the roughness sublayer psihat modifications.

    **Momentum** (Bonan et al. 2018, eq. A16, A21, 19):

    .. code-block:: none

        phim = phim_monin_obukhov((hc-disp)/obu)
        c1   = (1 - vkc/(2*beta*phim)) * exp(0.5*c2)
        psim = -psim1 + psim2 + c1*psihat_m(za) - c1*psihat_m(hc) + vkc/beta

    **Scalars** (eqs. A19, A21, 20):

    .. code-block:: none

        phic = phic_monin_obukhov((hc-disp)/obu)
        c1   = (1 - PrSc*vkc/(2*beta*phic)) * exp(0.5*c2)
        psic = -psic1 + psic2 + c1*psihat_h(za) - c1*psihat_h(hc)

    Args:
        za: Atmospheric height (m).
        hc: Canopy top height (m).
        disp: Displacement height (m).
        obu: Obukhov length (m).
        beta: u*/u(h) at canopy top.
        PrSc: Prandtl/Schmidt number.

    Returns:
        Tuple ``(psim, psic, psim2, psim_hat2)`` where:
        - ``psim``: ψ for momentum including RSL (evaluated za→hc).
        - ``psic``: ψ for scalars including RSL.
        - ``psim2``: MO ψ for momentum evaluated at hc.
        - ``psim_hat2``: RSL psihat for momentum evaluated at hc.
    """
    dt = hc - disp  # Fortran line 358
    zdt_za = (za - hc) / dt  # normalised height at za
    zdt_hc = 0.0  # (hc - hc) / dt == 0 always
    dtL = dt / obu  # stability ratio

    # Momentum — Fortran lines 360-376
    phim = _phim_monin_obukhov((hc - disp) / obu)
    c1_m = (1.0 - vkc / (2.0 * beta * phim)) * jnp.exp(0.5 * c2)

    psim_hat1_raw = _LookupPsihatM(zdt_za, dtL)
    psim_hat2_raw = _LookupPsihatM(zdt_hc, dtL)
    psim_hat1 = psim_hat1_raw * c1_m
    psim_hat2 = psim_hat2_raw * c1_m

    psim1 = _psim_monin_obukhov((za - disp) / obu)
    psim2 = _psim_monin_obukhov((hc - disp) / obu)

    psim = -psim1 + psim2 + psim_hat1 - psim_hat2 + vkc / beta

    # Scalars — Fortran lines 378-393
    phic = _phic_monin_obukhov((hc - disp) / obu)
    c1_c = (1.0 - PrSc * vkc / (2.0 * beta * phic)) * jnp.exp(0.5 * c2)

    psic_hat1_raw = _LookupPsihatH(zdt_za, dtL)
    psic_hat2_raw = _LookupPsihatH(zdt_hc, dtL)
    psic_hat1 = psic_hat1_raw * c1_c
    psic_hat2 = psic_hat2_raw * c1_c

    psic1 = _psic_monin_obukhov((za - disp) / obu)
    psic2 = _psic_monin_obukhov((hc - disp) / obu)

    psic = -psic1 + psic2 + psic_hat1 - psic_hat2

    return psim, psic, psim2, psim_hat2


# ===========================================================================
# Private: beta = u*/u(h)
# Psi functions at many heights z (vector), other arguments shared: one traced call.
_psi_rsl_at = jax.vmap(
    lambda z, hc, disp, obu, beta, PrSc: _GetPsiRSL(z, hc, disp, obu, beta, PrSc),
    in_axes=(0, None, None, None, None, None),
)


# ===========================================================================


def _GetBeta(beta_neutral, LcL):
    """
    JAX-traceable β = u*/u(h) solver.

    Mirrors :func:`_GetBeta` but replaces Python ``if/else`` on ``LcL``
    with ``jnp.where`` so the function is ``jax.jit``- and
    ``jax.grad``-compatible.  The error-check ``endrun`` is omitted since
    it uses Python ``if`` on a traced value.

    Both branches are always evaluated; the stable-branch denominator
    ``aa_s`` is guarded to ``1.0`` when ``LcL <= 0`` to prevent division
    by zero in the branch that will be discarded by ``jnp.where``.

    Args:
        beta_neutral: Neutral β — JAX scalar.
        LcL: Canopy density scale / Obukhov length — JAX scalar.

    Returns:
        β — JAX scalar.
    """
    # --- Unstable branch (LcL <= 0): solve β² = (-b + sqrt(b²-4ac)) / 2a ---
    bb_u = 16.0 * LcL * beta_neutral**4
    cc_u = -(beta_neutral**4)
    disc_u = jnp.maximum(bb_u * bb_u - 4.0 * cc_u, 1.0e-30)  # a=1
    beta_u = jnp.sqrt(jnp.maximum((-bb_u + jnp.sqrt(disc_u)) / 2.0, 1.0e-30))

    # --- Stable branch (LcL > 0): Cardano formula for 5*LcL*β³ + β = β_n ---
    # Guard aa_s != 0 when LcL==0 so no NaN before jnp.where selects branch.
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug.
    # LcL ≥ 0 in the stable branch; when LcL < 0 the branch is discarded anyway.
    aa_s = jnp.maximum(5.0 * LcL, 1.0e-10)
    # bb_s == 0, cc_s == 1, dd_s == -beta_neutral
    dd_s = -beta_neutral
    qq_s = jnp.sqrt(
        jnp.maximum(
            (27.0 * aa_s**2 * dd_s) ** 2 + 108.0 * aa_s**3,
            1.0e-30,
        )
    )
    rr_s = jnp.maximum(
        0.5 * (qq_s + 27.0 * aa_s**2 * dd_s),
        1e-30,
    ) ** (1.0 / 3.0)
    beta_s = -rr_s / (3.0 * aa_s) + 1.0 / rr_s  # simplifies with bb=0,cc=1

    return jnp.where(LcL <= 0.0, beta_u, beta_s)


# ===========================================================================
# Private: ObuFunc residual
# ===========================================================================


def _ObuFuncPure_jax(
    sparse_canopy_type: int,
    obu_val,
    *,
    Lc_p: float,
    ztop_p: float,
    lai_p: float,
    sai_p: float,
    zref_p: float,
    uref_p: float,
    thref_p: float,
    thvref_p: float,
    qref_p: float,
    rhomol_p: float,
    taf_p: float,
    qaf_p: float,
):
    """
    JAX-traceable Obukhov-length residual.

    Mirrors :func:`_ObuFuncPure` but uses ``jnp.*`` throughout and
    replaces the Python ``if obu_val >= 0.0`` branch with ``jnp.where``.
    Calls :func:`_GetBeta` (JAX-traceable β), :func:`_GetPrSc`
    (already uses ``jnp.tanh``), and :func:`_GetPsiRSL` (already uses
    ``jnp.searchsorted`` / ``jnp.exp``).

    The ``zref_p - zdisp_val < 0`` error check is dropped: it uses
    Python ``if`` on a traced value and would break JIT.

    Args:
        obu_val: Trial Obukhov length (m) — JAX scalar.
        Remaining keyword args: patch-level constants (Python floats).

    Returns:
        Residual ``obu_new - obu_val`` (m) — JAX scalar.
    """
    obu_min_stable = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    obu_cur = jnp.where(
        obu_val >= 0.0,
        jnp.maximum(obu_val, obu_min_stable),
        jnp.minimum(obu_val, obu_max_unstable),
    )
    _eps_obu = jnp.asarray(1e-30)
    # Use sign * max(|obu|, eps) to avoid select-as-denominator pattern
    # which triggers XLA select_divide_fusion compilation failure.
    _obu_abs_safe = jnp.maximum(jnp.abs(obu_cur), _eps_obu)
    _obu_sign = jnp.where(obu_cur < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    LcL_val = Lc_p * _obu_sign / _obu_abs_safe

    z0mg_safe = jnp.maximum(z0mg, _eps_obu)
    c1_n = (vkc / jnp.log((ztop_p + z0mg_safe) / z0mg_safe)) ** 2
    beta_neutral = jnp.minimum(
        jnp.sqrt(c1_n + cr * (lai_p + sai_p)),
        jnp.asarray(beta_neutral_max),
    )

    beta_HF = _GetBeta(beta_neutral, LcL_val)
    beta_norsl = _GetBeta(jnp.asarray(vkc / 2.0), LcL_val)
    beta_val = jnp.where(
        LcL_val > _aH12_1,
        beta_HF,
        (
            beta_norsl
            + (beta_HF - beta_norsl) / (1.0 + _aH12_0 * jnp.abs(LcL_val - _aH12_1) ** _aH12_2)
        ),
    )

    hc_minus_d = beta_val**2 * Lc_p
    if sparse_canopy_type == 1:  # static Python branch — baked in at trace time
        hc_minus_d = hc_minus_d * (1.0 - jnp.exp(-0.25 * (lai_p + sai_p) / beta_val**2))
    hc_minus_d = jnp.minimum(hc_minus_d, jnp.asarray(ztop_p))
    zdisp_val = ztop_p - hc_minus_d

    PrSc_val = _GetPrSc(beta_neutral, beta_neutral_max, LcL_val, sparse_canopy_type)

    psim, psic, _dum1, _dum2 = _GetPsiRSL(zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val)

    _hc_d = jnp.maximum(ztop_p - zdisp_val, _eps_obu)
    zlog = jnp.log(jnp.maximum((zref_p - zdisp_val) / _hc_d, _eps_obu))
    # sign * max(|denom|, eps) avoids select-as-denominator → no select_divide_fusion
    _dm_abs = jnp.maximum(jnp.abs(zlog + psim), _eps_obu)
    _dm_sign = jnp.where((zlog + psim) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    _dc_abs = jnp.maximum(jnp.abs(zlog + psic), _eps_obu)
    _dc_sign = jnp.where((zlog + psic) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    ustar_val = uref_p * vkc * _dm_sign / _dm_abs
    tstar = (thref_p - taf_p) * vkc * _dc_sign / _dc_abs
    qstar = (qref_p - qaf_p) * vkc * _dc_sign / _dc_abs

    tvstar = tstar + 0.61 * thref_p * qstar
    # Use sign * max(|tvstar|, eps) to avoid select-as-denominator pattern
    # which triggers XLA select_divide_fusion compilation failure.
    _tvstar_abs_safe = jnp.maximum(jnp.abs(tvstar), _eps_obu)
    _tvstar_sign = jnp.where(tvstar < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    obu_new = ustar_val**2 * thvref_p * _tvstar_sign / (vkc * grav * _tvstar_abs_safe)
    return obu_new - obu_val


# ===========================================================================
# Private: fixed-iteration Obukhov solver
# ===========================================================================


def _obu_body_pure(sparse_canopy_type, i, carry):
    """Pure fori_loop body for the Obukhov secant solver.

    Defined at module level (not as a closure) so ``jax.lax.fori_loop``
    traces it once and reuses the compiled XLA program for every patch and
    every sub-step.  All patch-level constants are threaded through ``carry``
    as JAX arrays rather than closed-over Python scalars.

    Carry layout: (x0, x1, f0, f1,
                   Lc_p, ztop_p, lai_p, sai_p,
                   zref_p, uref_p, thref_p, thvref_p,
                   qref_p, rhomol_p, taf_p, qaf_p)
    """
    (
        x0,
        x1,
        f0,
        f1,
        Lc_p,
        ztop_p,
        lai_p,
        sai_p,
        zref_p,
        uref_p,
        thref_p,
        thvref_p,
        qref_p,
        rhomol_p,
        taf_p,
        qaf_p,
    ) = carry
    # sign * max(|df|, eps) avoids select-as-denominator → no select_divide_fusion
    _df = f1 - f0
    _df_abs_safe = jnp.maximum(jnp.abs(_df), jnp.asarray(1e-30))
    _df_sign = jnp.where(_df < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    # Identical residuals (f1 == f0, typically both round-off after convergence) leave
    # the secant step undefined: take no step. Dividing by the 1e-30 floor instead
    # throws a converged iterate far away (e.g. Obukhov length 73.6 m -> -128 m).
    dx = jnp.where(_df == 0.0, 0.0, -f1 * (x1 - x0) * _df_sign / _df_abs_safe)
    x_new = x1 + dx
    f_new = _ObuFuncPure_jax(
        sparse_canopy_type,
        x_new,
        Lc_p=Lc_p,
        ztop_p=ztop_p,
        lai_p=lai_p,
        sai_p=sai_p,
        zref_p=zref_p,
        uref_p=uref_p,
        thref_p=thref_p,
        thvref_p=thvref_p,
        qref_p=qref_p,
        rhomol_p=rhomol_p,
        taf_p=taf_p,
        qaf_p=qaf_p,
    )
    return (
        x1,
        x_new,
        f1,
        f_new,
        Lc_p,
        ztop_p,
        lai_p,
        sai_p,
        zref_p,
        uref_p,
        thref_p,
        thvref_p,
        qref_p,
        rhomol_p,
        taf_p,
        qaf_p,
    )


def _obu_fixed_iter(init_obu, kwargs, sparse_canopy_type: int, n_iter: int = 25):
    """
    Fixed-iteration Obukhov-length solver via ``jax.lax.fori_loop``.

    Replaces Fortran's ``hybrid`` (MLMathToolsMod.F90; a Python ``while``-loop)
    with a ``jax.lax.fori_loop`` of exactly ``n_iter`` secant steps so the
    solver is ``jax.jit``-compatible.

    The secant update is::

        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

    ``n_iter=25`` is conservative; Bonan et al. (2018) report typical
    convergence in < 10 iterations for moderate stability regimes.

    The body function is defined at module level (:func:`_obu_body_pure`) so
    ``jax.lax.fori_loop`` traces it **once** regardless of how many patches
    or sub-steps are run.  Patch constants are passed through the carry tuple
    as JAX scalars rather than being closed over as Python floats, which
    previously caused one XLA ``scan`` recompilation per patch per sub-step.

    Args:
        init_obu: Initial Obukhov length (m) — JAX scalar.
        kwargs:   Dict of patch-level constants passed to
                  :func:`_ObuFuncPure_jax`.  Values may be Python floats or
                  JAX scalars; they are converted to JAX arrays here.
        n_iter:   Number of fixed iterations (default 25).

    Returns:
        Converged Obukhov length estimate (m) — JAX scalar.
    """
    # Convert all kwargs to JAX scalars so they have stable abstract types
    # and are passed through the carry rather than baked into the trace.
    Lc_p = jnp.asarray(kwargs["Lc_p"])
    ztop_p = jnp.asarray(kwargs["ztop_p"])
    lai_p = jnp.asarray(kwargs["lai_p"])
    sai_p = jnp.asarray(kwargs["sai_p"])
    zref_p = jnp.asarray(kwargs["zref_p"])
    uref_p = jnp.asarray(kwargs["uref_p"])
    thref_p = jnp.asarray(kwargs["thref_p"])
    thvref_p = jnp.asarray(kwargs["thvref_p"])
    qref_p = jnp.asarray(kwargs["qref_p"])
    rhomol_p = jnp.asarray(kwargs["rhomol_p"])
    taf_p = jnp.asarray(kwargs["taf_p"])
    qaf_p = jnp.asarray(kwargs["qaf_p"])

    obu0 = init_obu
    obu1 = jnp.asarray(-100.0)

    f0 = _ObuFuncPure_jax(
        sparse_canopy_type,
        obu0,
        Lc_p=Lc_p,
        ztop_p=ztop_p,
        lai_p=lai_p,
        sai_p=sai_p,
        zref_p=zref_p,
        uref_p=uref_p,
        thref_p=thref_p,
        thvref_p=thvref_p,
        qref_p=qref_p,
        rhomol_p=rhomol_p,
        taf_p=taf_p,
        qaf_p=qaf_p,
    )
    f1 = _ObuFuncPure_jax(
        sparse_canopy_type,
        obu1,
        Lc_p=Lc_p,
        ztop_p=ztop_p,
        lai_p=lai_p,
        sai_p=sai_p,
        zref_p=zref_p,
        uref_p=uref_p,
        thref_p=thref_p,
        thvref_p=thvref_p,
        qref_p=qref_p,
        rhomol_p=rhomol_p,
        taf_p=taf_p,
        qaf_p=qaf_p,
    )

    init_carry = (
        obu0,
        obu1,
        f0,
        f1,
        Lc_p,
        ztop_p,
        lai_p,
        sai_p,
        zref_p,
        uref_p,
        thref_p,
        thvref_p,
        qref_p,
        rhomol_p,
        taf_p,
        qaf_p,
    )

    final_carry = jax.lax.fori_loop(
        0, n_iter, functools.partial(_obu_body_pure, sparse_canopy_type), init_carry
    )
    obu_final = final_carry[1]  # x1 from final iteration
    return obu_final


# ===========================================================================
# Private: GetObu driver
# ===========================================================================


def _GetObu(p: int, mlcanopy_inst: mlcanopy_type, sparse_canopy_type: int) -> mlcanopy_type:
    """
    Solve for the Obukhov length and update the dependent quantities.

    Mirrors Fortran subroutine ``GetObu`` (lines 160-178), but with 25 fixed secant
    steps instead of Fortran's hybrid solver (docs/design/numerics_and_regression.md §4.1).
    The forward value is the solver's root; the gradient is the implicit-function one (§2).

    Args:
        p: Patch index.
        mlcanopy_inst: Canopy container.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    _kwargs = dict(
        Lc_p=mlcanopy_inst.Lc_canopy[p],
        ztop_p=mlcanopy_inst.ztop_canopy[p],
        lai_p=mlcanopy_inst.lai_canopy[p],
        sai_p=mlcanopy_inst.sai_canopy[p],
        zref_p=mlcanopy_inst.zref_forcing[p],
        uref_p=mlcanopy_inst.uref_forcing[p],
        thref_p=mlcanopy_inst.thref_forcing[p],
        thvref_p=mlcanopy_inst.thvref_forcing[p],
        qref_p=mlcanopy_inst.qref_forcing[p],
        rhomol_p=mlcanopy_inst.rhomol_forcing[p],
        taf_p=mlcanopy_inst.taf_canopy[p],
        qaf_p=mlcanopy_inst.qaf_canopy[p],
    )

    # IFT-based gradient for the Obukhov-length fixed-point solver.
    #
    # Differentiating through the 25-iteration secant solver (_obu_fixed_iter)
    # gives exploding gradients: each secant step's Jacobian accumulates over
    # 25 iterations, producing |J|^25 amplification (observed: ~1e+144).
    #
    # Fix (same pattern as _bisect_gs_ift):
    #   obu_ift = obu0 - F(obu0; theta) / stop_grad(dF/dobu)
    # where F(obu) = _ObuFuncPure_jax(obu, theta) = obu_new(obu, theta) - obu.
    #
    # Forward: exactly obu0 (the correction applies to the gradient only, below)
    # Backward: d(obu_ift)/dtheta = -∂F/∂theta / stop_grad(dF/dobu) = IFT  ✓
    #
    # dF/dobu = dobu_new/dobu - 1 < 0 for a converging fixed-point.

    # Forward solve (stop gradient through iterations)
    obu_raw = _obu_fixed_iter(jnp.asarray(100.0), _kwargs, sparse_canopy_type, n_iter=25)
    obu0 = jax.lax.stop_gradient(obu_raw)

    # Residual at obu0 — gradient flows through theta (thref_p, thvref_p, ...)
    f0 = _ObuFuncPure_jax(sparse_canopy_type, obu0, **_kwargs)

    # dF/dobu via central FD — stop_gradient (denominator only)
    _delta_obu = jnp.asarray(1.0, dtype=obu0.dtype)  # 1 m step (obu is O(100-10000 m))
    df_dobu = jax.lax.stop_gradient(
        (
            _ObuFuncPure_jax(sparse_canopy_type, obu0 + _delta_obu, **_kwargs)
            - _ObuFuncPure_jax(sparse_canopy_type, obu0 - _delta_obu, **_kwargs)
        )
        / (2.0 * _delta_obu)
    )

    # Gate Newton step on |df_dobu| > eps (IFT well-conditioned)
    _eps_ift = jnp.asarray(1e-6)
    apply = jnp.abs(df_dobu) > _eps_ift
    safe_f0 = jnp.where(apply, f0, jnp.zeros_like(f0))
    safe_denom = jnp.where(apply, df_dobu, -jnp.ones_like(df_dobu))
    # Default safe_denom = -1.0: when apply=False, safe_f0/safe_denom = 0 → obu_ift = obu0 ✓
    # The Newton correction is applied to the gradient only: subtracting
    # (step - stop_gradient(step)) leaves the forward value exactly equal to the
    # solver's root (as in the non-differentiable path and Fortran) while the
    # derivative is the implicit-function one, -dF/dtheta / dF/dx.
    step = safe_f0 / safe_denom
    obu_ift = obu0 - (step - jax.lax.stop_gradient(step))

    # Recompute all derived quantities from the IFT-corrected obu (JAX-traceable)
    mlcanopy_inst = _obu_writeback_jax(p, obu_ift, _kwargs, sparse_canopy_type, mlcanopy_inst)
    return mlcanopy_inst


def _obu_writeback_jax(p, obu_val, kwargs, sparse_canopy_type, mlcanopy_inst):
    """JAX-traceable write-back of Obukhov-derived quantities.

    Recomputes beta, zdisp, PrSc, ustar, gac_to_hc from ``obu_val``
    using the same physics as ``_ObuFuncPure_jax``, then writes them
    back into ``mlcanopy_inst``.
    """
    Lc_p = kwargs["Lc_p"]
    ztop_p = kwargs["ztop_p"]
    lai_p = kwargs["lai_p"]
    sai_p = kwargs["sai_p"]
    zref_p = kwargs["zref_p"]
    uref_p = kwargs["uref_p"]
    rhomol_p = kwargs["rhomol_p"]

    obu_min_stable = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    obu_cur = jnp.where(
        obu_val >= 0.0,
        jnp.maximum(obu_val, obu_min_stable),
        jnp.minimum(obu_val, obu_max_unstable),
    )
    _eps_obu = jnp.asarray(1e-30)
    # Use sign * max(|obu|, eps) to avoid select-as-denominator pattern
    # which triggers XLA select_divide_fusion compilation failure.
    _obu_abs_safe = jnp.maximum(jnp.abs(obu_cur), _eps_obu)
    _obu_sign = jnp.where(obu_cur < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    LcL_val = Lc_p * _obu_sign / _obu_abs_safe

    z0mg_safe = jnp.maximum(z0mg, _eps_obu)
    c1_n = (vkc / jnp.log((ztop_p + z0mg_safe) / z0mg_safe)) ** 2
    beta_neutral = jnp.minimum(
        jnp.sqrt(c1_n + cr * (lai_p + sai_p)),
        jnp.asarray(beta_neutral_max),
    )

    beta_HF = _GetBeta(beta_neutral, LcL_val)
    beta_norsl = _GetBeta(jnp.asarray(vkc / 2.0), LcL_val)
    beta_val = jnp.where(
        LcL_val > _aH12_1,
        beta_HF,
        (
            beta_norsl
            + (beta_HF - beta_norsl) / (1.0 + _aH12_0 * jnp.abs(LcL_val - _aH12_1) ** _aH12_2)
        ),
    )

    hc_minus_d = beta_val**2 * Lc_p
    if sparse_canopy_type == 1:
        hc_minus_d = hc_minus_d * (1.0 - jnp.exp(-0.25 * (lai_p + sai_p) / beta_val**2))
    hc_minus_d = jnp.minimum(hc_minus_d, ztop_p)
    zdisp_val = ztop_p - hc_minus_d

    PrSc_val = _GetPrSc(beta_neutral, beta_neutral_max, LcL_val, sparse_canopy_type)

    psim, psic, _, _ = _GetPsiRSL(zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val)

    _hc_d2 = jnp.maximum(ztop_p - zdisp_val, _eps_obu)
    zlog = jnp.log(jnp.maximum((zref_p - zdisp_val) / _hc_d2, _eps_obu))
    # Use sign * max(|denom|, eps) — avoids jnp.where-as-denominator NaN grads
    # (jnp.where differentiates both branches; 0/0 = NaN even when masked).
    _dm2_abs = jnp.maximum(jnp.abs(zlog + psim), _eps_obu)
    _dm2_sign = jnp.where((zlog + psim) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    _dc2_abs = jnp.maximum(jnp.abs(zlog + psic), _eps_obu)
    _dc2_sign = jnp.where((zlog + psic) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    ustar_val = uref_p * vkc * _dm2_sign / _dm2_abs
    gac_to_hc_v = rhomol_p * vkc * ustar_val * _dc2_sign / _dc2_abs

    return mlcanopy_inst._replace(
        zdisp_canopy=mlcanopy_inst.zdisp_canopy.at[p].set(zdisp_val),
        beta_canopy=mlcanopy_inst.beta_canopy.at[p].set(beta_val),
        PrSc_canopy=mlcanopy_inst.PrSc_canopy.at[p].set(PrSc_val),
        ustar_canopy=mlcanopy_inst.ustar_canopy.at[p].set(ustar_val),
        gac_to_hc_canopy=mlcanopy_inst.gac_to_hc_canopy.at[p].set(gac_to_hc_v),
        obu_canopy=mlcanopy_inst.obu_canopy.at[p].set(obu_cur),
    )


# ===========================================================================
# Private: RoughnessLength
# ===========================================================================


def _z0m_residual(trial, obu_p, hc_minus_d, exp1, exp2, psim_hc):
    """Residual whose root is z0m (Fortran ``RoughnessLength``, function ``z0m_func``)."""
    psim_t = _psim_monin_obukhov(trial / obu_p)
    return hc_minus_d * exp1 * jnp.exp(-psim_hc + psim_t) * exp2 - trial


def _RoughnessLength(p: int, mlcanopy_inst: mlcanopy_type) -> mlcanopy_type:
    """Roughness length for momentum, by bisection with the Fortran stopping rule.

    Mirrors Fortran subroutine ``RoughnessLength`` (via
    :func:`clm_ml.physics.math.bisection_traceable`). JAX-traceable.
    """
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    zdisp_p = mlcanopy_inst.zdisp_canopy[p]
    obu_p = mlcanopy_inst.obu_canopy[p]
    beta_p = mlcanopy_inst.beta_canopy[p]
    PrSc_p = mlcanopy_inst.PrSc_canopy[p]
    zref_p = mlcanopy_inst.zref_forcing[p]

    # psi and psihat at canopy top (JAX version)
    _, _, psim_hc, psim_hat_hc = _GetPsiRSL(zref_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)

    hc_minus_d = ztop_p - zdisp_p
    exp1 = jnp.exp(-vkc / beta_p)
    exp2 = jnp.exp(psim_hat_hc)

    aval = ztop_p
    bval = jnp.asarray(0.0)

    # Same bisection as _RoughnessLength and Fortran RoughnessLength (lines 478-500):
    # stop once the bracket is narrower than err = 0.001 m (at most nmax = 20
    # halvings) and return the last midpoint (0 if none).
    z0m_val = bisection_traceable(
        _z0m_residual,
        aval,
        bval,
        tol=0.001,
        itmax=20,
        c_init=0.0,
        args=(obu_p, hc_minus_d, exp1, exp2, psim_hc),
    )

    return mlcanopy_inst._replace(z0m_canopy=mlcanopy_inst.z0m_canopy.at[p].set(z0m_val))


# ===========================================================================
# Private: WindProfile
# ===========================================================================


def _WindProfile(
    lay: Layers,
    p: int,
    lm_over_beta,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """Wind speed profile above and within the canopy (Fortran ``WindProfile``).

    Vectorized over every layer slot: layers ntop+1..ncan get the above-canopy profile,
    1..ntop the within-canopy profile (``lay``: data, plan Phase 5b); padding is kept.
    """
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    zdisp_p = mlcanopy_inst.zdisp_canopy[p]
    obu_p = mlcanopy_inst.obu_canopy[p]
    beta_p = mlcanopy_inst.beta_canopy[p]
    PrSc_p = mlcanopy_inst.PrSc_canopy[p]
    ustar_p = mlcanopy_inst.ustar_canopy[p]

    wind = mlcanopy_inst.wind_profile
    uaf = mlcanopy_inst.uaf_canopy
    zs = pad_with_top(lay, mlcanopy_inst.zs_profile[p, 1:])  # slots 1..max_layers

    # Above-canopy wind (layers ntop+1..ncan)
    zs_above = jnp.where(lay.above_canopy, zs, jnp.take(zs, lay.ncan - 1))
    psim, _, _, _ = _psi_rsl_at(zs_above, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    _hcd_w = jnp.maximum(ztop_p - zdisp_p, 1e-30)
    zlog_m = jnp.log(jnp.maximum((zs_above - zdisp_p) / _hcd_w, 1e-30))
    wind_above = ustar_p / vkc * (zlog_m + psim)

    # Wind at canopy top
    uaf_val = ustar_p / jnp.maximum(beta_p, 1e-30)
    uaf = uaf.at[p].set(uaf_val)

    # Within-canopy wind (layers 1..ntop)
    zs_within = jnp.where(lay.in_canopy, zs, ztop_p)
    wind_within = uaf_val * jnp.exp((zs_within - ztop_p) / lm_over_beta)

    new = jnp.where(lay.above_canopy, wind_above, wind_within)
    wind = wind.at[p, 1:].set(keep(lay.in_use, new, wind[p, 1:]))
    return mlcanopy_inst._replace(
        uaf_canopy=uaf,
        wind_profile=wind,
    )


# ===========================================================================
# Private: AerodynamicConductance
# ===========================================================================


def _AerodynamicConductance(
    lay: Layers,
    p: int,
    lm_over_beta,
    mlcanopy_inst: mlcanopy_type,
    HF_extension_type: int,
) -> mlcanopy_type:
    """Aerodynamic conductances and eddy diffusivity (Fortran ``AerodynamicConductance``).

    Vectorized over every layer slot, the formula chosen per layer (``lay``: data, plan
    Phase 5b): above the canopy (ntop+1..ncan), within it (1..ntop-1), its top layer
    (ntop); padding is kept. Error checks are omitted for differentiability.
    """
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    zdisp_p = mlcanopy_inst.zdisp_canopy[p]
    obu_p = mlcanopy_inst.obu_canopy[p]
    beta_p = mlcanopy_inst.beta_canopy[p]
    PrSc_p = mlcanopy_inst.PrSc_canopy[p]
    ustar_p = mlcanopy_inst.ustar_canopy[p]
    rhomol_p = mlcanopy_inst.rhomol_forcing[p]
    zref_p = mlcanopy_inst.zref_forcing[p]

    gac = mlcanopy_inst.gac_profile
    gac0 = mlcanopy_inst.gac0_soil
    kc_eddy = mlcanopy_inst.kc_eddy_profile
    ic = lay.index  # layer numbers 1..max_layers
    zs = pad_with_top(lay, mlcanopy_inst.zs_profile[p, 1:])
    is_top_layer = ic == lay.ncan
    # Height of the next level up: zs(ic+1), zref above the top layer ncan
    zs_next = jnp.where(is_top_layer, zref_p, jnp.concatenate([zs[1:], zs[-1:]]))

    # --- Above-canopy conductances ---
    # Layer ic (ntop+1..ncan) spans zs(ic) to zs(ic+1), the top layer to zref. psi_c at each
    # height: every slot's zs, and zref and ztop.
    _eps_ac = jnp.asarray(1e-30)
    z_psi = jnp.concatenate([zs, jnp.stack([zref_p, ztop_p])])
    _, psic_z, _, _ = _psi_rsl_at(z_psi, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psic_zs, psic_zref, psic_ztop = psic_z[:-2], psic_z[-2], psic_z[-1]
    psic_next = jnp.where(is_top_layer, psic_zref, jnp.concatenate([psic_zs[1:], psic_zs[-1:]]))
    psic_d = psic_next - psic_zs
    zlog_c = jnp.log(jnp.maximum((zs_next - zdisp_p) / jnp.maximum(zs - zdisp_p, _eps_ac), _eps_ac))
    # sign * max(|x|, eps) avoids jnp.where-as-denominator NaN grads
    _dc_abs = jnp.maximum(jnp.abs(zlog_c + psic_d), _eps_ac)
    _dc_sign = jnp.where((zlog_c + psic_d) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    gac_above = rhomol_p * vkc * ustar_p * _dc_sign / _dc_abs

    # ztop to zs(ntop+1)
    zs_ntop1 = jnp.take(zs, lay.ntop)  # zs(ntop+1)
    psic_d = jnp.take(psic_zs, lay.ntop) - psic_ztop
    _hcd_ac = jnp.maximum(ztop_p - zdisp_p, _eps_ac)
    zlog_c = jnp.log(jnp.maximum((zs_ntop1 - zdisp_p) / _hcd_ac, _eps_ac))
    _dc_abs = jnp.maximum(jnp.abs(zlog_c + psic_d), _eps_ac)
    _dc_sign = jnp.where((zlog_c + psic_d) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    gac_above_foliage = rhomol_p * vkc * ustar_p * _dc_sign / _dc_abs

    # --- Within-canopy conductances (layers 1..ntop-1: zs(ic) to zs(ic+1)) ---
    _bu_safe = jnp.maximum(beta_p * ustar_p, _eps_ac)
    zl = jnp.where(lay.in_canopy, zs, ztop_p) - ztop_p
    zu = jnp.where(lay.in_canopy, jnp.concatenate([zs[1:], zs[-1:]]), ztop_p) - ztop_p
    res = PrSc_p / _bu_safe * (jnp.exp(-zl / lm_over_beta) - jnp.exp(-zu / lm_over_beta))
    res_safe = jnp.maximum(jnp.abs(res), _eps_ac)
    gac_within = rhomol_p / res_safe

    # Top foliage layer: combine below and above
    zl = jnp.take(zs, lay.ntop - 1) - ztop_p
    res = PrSc_p / _bu_safe * (jnp.exp(-zl / lm_over_beta) - 1.0)  # exp(0)=1
    res_safe = jnp.maximum(jnp.abs(res), _eps_ac)
    gac_below_foliage = rhomol_p / res_safe
    gac_top = 1.0 / (1.0 / gac_below_foliage + 1.0 / gac_above_foliage)

    gac_new = jnp.where(lay.above_canopy, gac_above, jnp.where(ic == lay.ntop, gac_top, gac_within))

    # --- Soil surface aerodynamic conductance ---
    z0cg = 0.1 * z0mg
    zs1 = mlcanopy_inst.zs_profile[p, 1]
    z0mg_safe = jnp.maximum(z0mg, _eps_ac)

    if HF_extension_type == 1:  # static branch
        zl = z0cg - ztop_p
        zu = zs1 - ztop_p
        res = PrSc_p / _bu_safe * (jnp.exp(-zl / lm_over_beta) - jnp.exp(-zu / lm_over_beta))
        res_safe = jnp.maximum(jnp.abs(res), _eps_ac)
        _gac0_val = rhomol_p / res_safe
    elif HF_extension_type == 2:  # static branch
        zlog_m = jnp.log(jnp.maximum(zs1 / z0mg_safe, _eps_ac))
        zlog_m_safe = jnp.maximum(jnp.abs(zlog_m), _eps_ac)
        ustar_g = jnp.maximum(mlcanopy_inst.wind_profile[p, 1], 0.1) * vkc / zlog_m_safe
        _gac0_val = rhomol_p * vkc * ustar_g / zlog_m_safe
    else:
        _gac0_val = gac0[p]

    # Resistance cap
    _gac0_val = rhomol_p / jnp.minimum(rhomol_p / _gac0_val, ra_max)
    gac0 = gac0.at[p].set(_gac0_val)

    gac_new = rhomol_p / jnp.minimum(rhomol_p / gac_new, ra_max)
    gac = gac.at[p, 1:].set(keep(lay.in_use, gac_new, gac[p, 1:]))

    # Eddy diffusivity: dz to the next level (zref above the top layer)
    kc_new = gac_new / rhomol_p * (zs_next - zs)
    kc_eddy = kc_eddy.at[p, 1:].set(keep(lay.in_use, kc_new, kc_eddy[p, 1:]))

    return mlcanopy_inst._replace(
        gac0_soil=gac0,
        gac_profile=gac,
        kc_eddy_profile=kc_eddy,
    )


# ===========================================================================
# Private: HF2008 main solver
# ===========================================================================


@functools.partial(jax.jit, static_argnames=("grid", "opts"))
def _HF2008(grid: GridInfo, mlcanopy_inst: mlcanopy_type, opts: MLOptions) -> mlcanopy_type:
    """Canopy turbulence from the roughness sublayer theory of Harman and Finnigan (2008).

    Mirrors Fortran subroutine ``HF2008``. Compiled (run operation by operation, it would
    dispatch thousands of tiny JAX operations per call); layer counts are data.
    """
    p = grid.p
    lay = layers(mlcanopy_inst, p)  # layer counts as data (plan Phase 5b)

    Lc = mlcanopy_inst.Lc_canopy
    taf = mlcanopy_inst.taf_canopy
    qaf = mlcanopy_inst.qaf_canopy
    mflx = mlcanopy_inst.mflx_profile

    pref_p = mlcanopy_inst.pref_forcing[p]
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    lai_p = mlcanopy_inst.lai_canopy[p]
    sai_p = mlcanopy_inst.sai_canopy[p]

    # Canopy density length scale
    Lc_val = ztop_p / (cd * (lai_p + sai_p))
    Lc = Lc.at[p].set(Lc_val)
    mlcanopy_inst = mlcanopy_inst._replace(Lc_canopy=Lc)

    # Temperature and humidity at canopy top
    tair_ntop = mlcanopy_inst.tair_profile[p, lay.ntop]
    eair_ntop = mlcanopy_inst.eair_profile[p, lay.ntop]
    taf_val = tair_ntop
    qaf_val = mmh2o / mmdry * eair_ntop / (pref_p - (1.0 - mmh2o / mmdry) * eair_ntop)
    taf = taf.at[p].set(taf_val)
    qaf = qaf.at[p].set(qaf_val)
    mlcanopy_inst = mlcanopy_inst._replace(taf_canopy=taf, qaf_canopy=qaf)

    # Obukhov length (differentiable)
    mlcanopy_inst = _GetObu(p, mlcanopy_inst, opts.sparse_canopy_type)

    # Roughness length (differentiable)
    mlcanopy_inst = _RoughnessLength(p, mlcanopy_inst)

    # lm / beta
    beta_p = mlcanopy_inst.beta_canopy[p]
    lm = 2.0 * beta_p**3 * Lc_val
    eta = beta_p / lm * ztop_p  # Fortran warns if eta >= eta_max: see CanopyTurbulence
    lm_over_beta = ztop_p / eta

    # Wind profile (differentiable)
    mlcanopy_inst = _WindProfile(lay, p, lm_over_beta, mlcanopy_inst)

    # Momentum flux profile — use jnp.where instead of if
    ustar_p = mlcanopy_inst.ustar_canopy[p]
    zw_p = mlcanopy_inst.zw_profile[p]
    zw_v = pad_with_top(lay, zw_p[1:])
    mflx_v = jnp.where(
        zw_v > ztop_p,
        -(ustar_p**2),
        -(ustar_p**2) * jnp.exp(2.0 * (jnp.minimum(zw_v, ztop_p) - ztop_p) / lm_over_beta),
    )
    mflx = mflx.at[p, 1:].set(keep(lay.in_use, mflx_v, mflx[p, 1:]))
    mlcanopy_inst = mlcanopy_inst._replace(mflx_profile=mflx)

    # Aerodynamic conductances (differentiable)
    mlcanopy_inst = _AerodynamicConductance(
        lay, p, lm_over_beta, mlcanopy_inst, opts.HF_extension_type
    )

    return mlcanopy_inst


# ===========================================================================
# Public: CanopyTurbulence driver
# ===========================================================================


def CanopyTurbulence(
    nstep_ml: int,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: GridInfo,
    nstep: int = 0,
    *,
    opts: MLOptions,
) -> mlcanopy_type:
    """
    Main driver for the canopy turbulence parameterisation.

    Mirrors Fortran subroutine ``CanopyTurbulence`` (lines 43-67).

    Args:
        nstep_ml: Current multilayer timestep counter.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; all turbulence output fields
            updated by :func:`_HF2008`.
        grid: Patch index and canopy layer indices (:class:`GridInfo`).
        nstep: Current CLM time step (for warning messages only).

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    if opts.turb_type == 1:
        mlcanopy_inst = _HF2008(grid, mlcanopy_inst, opts)
        # Fortran HF2008 warning, from the outputs (it cannot print from compiled code)
        p = grid.p
        beta_p = mlcanopy_inst.beta_canopy[p]
        eta = mlcanopy_inst.ztop_canopy[p] / (2.0 * beta_p**2 * mlcanopy_inst.Lc_canopy[p])
        if is_concrete(eta) and eta >= eta_max:
            print(" Warning: HF2008: lm/beta error")
            print(f" nstep = {nstep}  nstep_ml = {nstep_ml}")
            print(f" eta = {eta}")
        return mlcanopy_inst
    else:
        endrun(msg=" ERROR: CanopyTurbulence: turb_type not valid")
        return mlcanopy_inst  # Unreachable


# ===========================================================================
# Public: LookupPsihatINI
# ===========================================================================


def LookupPsihatINI() -> None:
    """
    Initialise the RSL psihat look-up tables from a NetCDF file.

    Mirrors Fortran subroutine ``LookupPsihatINI`` (lines 455-530).

    Reads variables ``dtLgridM``, ``zdtgridM``, ``psigridM``,
    ``dtLgridH``, ``zdtgridH``, and ``psigridH`` from the file whose
    path is given by the module-level variable ``rslfile``.

    Fortran stores the grids as:

    .. code-block:: none

        zdtgridM(nZ, 1), dtLgridM(1, nL), psigridM(nZ, nL)

    while the NetCDF file has dimensions in the opposite order:

    .. code-block:: none

        psigridM_nc(nL, nZ)  →  psigridM[ii, jj] = psigridM_nc[jj, ii]

    The copy loop transposes the NetCDF arrays into the Fortran (and
    Python) convention.  After this routine the module-level arrays
    ``zdtgridM``, ``dtLgridM``, ``psigridM``, ``zdtgridH``,
    ``dtLgridH``, and ``psigridH`` are populated and ready for use
    by :func:`_LookupPsihat`.

    Raises:
        SystemExit: (via :func:`endrun`) if the NetCDF dimensions do
            not match the expected ``nZ`` and ``nL`` or if any variable
            cannot be read.
    """
    global dtLgridM, dtLgridH, zdtgridM, zdtgridH, psigridM, psigridH
    global _zdtgM_1d, _dtLgM_1d, _neg_zdtgM_1d
    global _zdtgH_1d, _dtLgH_1d, _neg_zdtgH_1d
    global _dtLgM_jax, _zdtgM_jax, _neg_zdtgM_jax, _psigridM_jax
    global _dtLgH_jax, _zdtgH_jax, _neg_zdtgH_jax, _psigridH_jax

    import netCDF4 as nc  # noqa: F401 — defer import

    print("Attempting to read RSL look-up table .....")

    with nc.Dataset(rslfile, "r") as ncid:

        # Check dimensions — Fortran lines 487-498
        nZ_nc = len(ncid.dimensions["nZ"])
        nL_nc = len(ncid.dimensions["nL"])
        if nZ_nc != nZ:
            endrun(msg=" ERROR: LookupPsihatINI: nZ does not equal expected value")
        if nL_nc != nL:
            endrun(msg=" ERROR: LookupPsihatINI: nL does not equal expected value")

        # Read 1-D coordinate arrays and 2-D psihat grids — Fortran lines 500-520
        def _read(name: str):
            if name not in ncid.variables:
                endrun(msg=f" ERROR: LookupPsihatINI: error reading {name}")
            return ncid.variables[name][:]

        dtLgridM_nc = np.asarray(_read("dtLgridM"), dtype=float)  # shape (nL,)
        zdtgridM_nc = np.asarray(_read("zdtgridM"), dtype=float)  # shape (nZ,)
        psigridM_nc = np.asarray(
            _read("psigridM"), dtype=float
        )  # Actually shape (nZ, nL) not (nL, nZ)!

        dtLgridH_nc = np.asarray(_read("dtLgridH"), dtype=float)
        zdtgridH_nc = np.asarray(_read("zdtgridH"), dtype=float)
        psigridH_nc = np.asarray(_read("psigridH"), dtype=float)

    print("Successfully read RSL look-up table")

    # Copy into module-level arrays with Fortran index conventions —
    # Fortran lines 522-530:
    #   dtLgridM(1, jj) = dtLgridM_nc(jj)  →  dtLgridM[0, jj]
    #   zdtgridM(ii, 1) = zdtgridM_nc(ii)  →  zdtgridM[ii, 0]
    #   psigridM(ii,jj) = psigridM_nc(jj,ii) →  psigridM[ii, jj]
    #
    # NOTE: psigridM_nc is already in shape (nZ, nL), so no transpose needed

    # Populate numpy arrays directly (no JAX immutable operations needed)
    dtLgridM[0, :] = dtLgridM_nc
    dtLgridH[0, :] = dtLgridH_nc
    zdtgridM[:, 0] = zdtgridM_nc
    zdtgridH[:, 0] = zdtgridH_nc
    psigridM[:, :] = psigridM_nc
    psigridH[:, :] = psigridH_nc

    # Populate cached 1D views — eliminates per-call slicing and negation in _LookupPsihatM/H
    _dtLgM_1d = dtLgridM[0].copy()
    _zdtgM_1d = zdtgridM[:, 0].copy()
    _neg_zdtgM_1d = -_zdtgM_1d
    _dtLgH_1d = dtLgridH[0].copy()
    _zdtgH_1d = zdtgridH[:, 0].copy()
    _neg_zdtgH_1d = -_zdtgH_1d

    # Pre-convert to JAX arrays once — avoids jnp.asarray() on every
    # _LookupPsihatM/H call (~10K times per diff-mode forward pass).
    _dtLgM_jax = jnp.asarray(_dtLgM_1d)
    _zdtgM_jax = jnp.asarray(_zdtgM_1d)
    _neg_zdtgM_jax = jnp.asarray(_neg_zdtgM_1d)
    _psigridM_jax = jnp.asarray(psigridM)
    _dtLgH_jax = jnp.asarray(_dtLgH_1d)
    _zdtgH_jax = jnp.asarray(_zdtgH_1d)
    _neg_zdtgH_jax = jnp.asarray(_neg_zdtgH_1d)
    _psigridH_jax = jnp.asarray(psigridH)
