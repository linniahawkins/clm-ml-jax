"""
JAX translation of MLLeafPhotosynthesisMod Fortran module.

Leaf photosynthesis and stomatal conductance for the multilayer
canopy model.  Provides one public routine (:func:`LeafPhotosynthesis`)
and the following private helpers:

- :func:`_ft`: Arrhenius activation temperature response.
- :func:`_fth`: High-temperature deactivation factor.
- :func:`_fth25`: Deactivation scaling factor at 25 °C.
- :func:`_CiFunc`: Photosynthesis + gs for a specified Ci (hybrid callback).
- :func:`_CiFuncGs`: Photosynthesis for a specified gs.
- :func:`_StomataOptimization`: WUE-optimal stomatal conductance.
- :func:`_StomataEfficiency`: Marginal WUE check (zbrent/bisection callback).
- :func:`_RealizedRate`: Minimum or co-limited gross photosynthesis.

Original Fortran module: MLLeafPhotosynthesisMod
Fortran lines 1-500
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_varcon import tfrz  # noqa: F401
from clm_src_main.clm_varctl import iulog  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from multilayer_canopy.MLCanopyParamsMod import CanopyParams
from multilayer_canopy.MLclm_varcon import (  # noqa: F401
    colim_c3a,
    colim_c4a,
    colim_c4b,
    cp25,
    cpha,
    dh2o_to_dco2,
    jmaxha_acclim,
    jmaxha_noacclim,
    jmaxhd_acclim,
    jmaxhd_noacclim,
    jmaxse_acclim,
    jmaxse_noacclim,
    kc25,
    kcha,
    ko25,
    koha,
    phi_psII,
    qe_c4,
    rdha,
    rdhd,
    rdse,
    rgas,
    rh_min_BB,
    theta_j,
    vcmaxha_acclim,
    vcmaxha_noacclim,
    vcmaxhd_acclim,
    vcmaxhd_noacclim,
    vcmaxse_acclim,
    vcmaxse_noacclim,
    vpd_min_MED,
)
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions
from multilayer_canopy.MLMathToolsMod import (
    bisection,  # noqa: F401
    quadratic,
)
from multilayer_canopy.MLWaterVaporMod import SatVap, SatVap_py  # noqa: F401

# ---------------------------------------------------------------------------
# Private: temperature response functions
# ---------------------------------------------------------------------------


def _ft(tl: float, ha: float) -> float:
    """
    Arrhenius activation temperature response for photosynthesis.

    Mirrors Fortran function ``ft`` (private, lines 35-48).

    .. code-block:: none

        ans = exp( ha / (rgas*(tfrz+25)) * (1 - (tfrz+25)/tl) )

    Args:
        tl: Leaf temperature (K).
        ha: Activation energy (J/mol).

    Returns:
        Dimensionless temperature response factor.
    """
    return jnp.exp(ha / (rgas * (tfrz + 25.0)) * (1.0 - (tfrz + 25.0) / tl))


def _fth(tl: float, hd: float, se: float, c: float) -> float:
    """
    High-temperature deactivation factor for photosynthesis.

    Mirrors Fortran function ``fth`` (private, lines 50-67).

    .. code-block:: none

        ans = c / (1 + exp((-hd + se*tl) / (rgas*tl)))

    Args:
        tl: Leaf temperature (K).
        hd: Deactivation energy (J/mol).
        se: Entropy term (J/mol/K).
        c:  Scaling factor (= 1.0 at 25 °C).

    Returns:
        Dimensionless deactivation factor.
    """
    return c / (1.0 + jnp.exp((-hd + se * tl) / (rgas * tl)))


def _fth25(hd: float, se: float) -> float:
    """
    Scaling factor for deactivation such that :func:`_fth` = 1 at 25 °C.

    Mirrors Fortran function ``fth25`` (private, lines 69-85).

    .. code-block:: none

        ans = 1 + exp((-hd + se*(tfrz+25)) / (rgas*(tfrz+25)))

    Args:
        hd: Deactivation energy (J/mol).
        se: Entropy term (J/mol/K).

    Returns:
        Dimensionless scaling factor ``c`` for :func:`_fth`.
    """
    t25 = tfrz + 25.0
    return 1.0 + jnp.exp((-hd + se * t25) / (rgas * t25))


# ---------------------------------------------------------------------------
# Private: pure-Python scalar versions of temperature response functions
# ---------------------------------------------------------------------------


def _fth25_py(hd: float, se: float) -> float:
    """Pure-Python (math.exp) version of :func:`_fth25` for per-layer loops."""
    t25 = tfrz + 25.0
    return 1.0 + math.exp((-hd + se * t25) / (rgas * t25))


# ---------------------------------------------------------------------------
# Private: minimum or co-limited photosynthesis
# ---------------------------------------------------------------------------


def _RealizedRate(
    colim_type: int,
    c3psn_val: float,
    ac: float,
    aj: float,
    ap: float,
) -> float:
    """
    Return the realized (gross) photosynthesis rate as the minimum or
    co-limited combination of the three limiting rates.

    Mirrors Fortran subroutine ``RealizedRate`` (private, lines 450-500).

    ``colim_type == 0``: simple minimum of rates.
    ``colim_type == 1``: quadratic co-limitation of Ac and Aj first,
    then a second co-limitation with Ap for C4.

    Args:
        c3psn_val: Photosynthetic pathway flag (1 = C3, 0 = C4).
        ac: Rubisco-limited gross photosynthesis (umol CO2/m2/s).
        aj: RuBP-regeneration-limited gross photosynthesis (umol CO2/m2/s).
        ap: Product-limited (C3) or CO2-limited (C4) gross photosynthesis.

    Returns:
        ``agross``: Gross photosynthesis (umol CO2/m2/s).
    """
    is_c3 = jnp.round(jnp.asarray(c3psn_val)) == 1  # Fortran: nint(c3psn) == 1

    if colim_type == 0:  # Fortran lines 465-474 (static Python branch)
        c3_val = jnp.minimum(ac, aj)
        c4_val = jnp.minimum(jnp.minimum(ac, aj), ap)
        return jnp.where(is_c3, c3_val, c4_val)

    elif colim_type == 1:  # Fortran lines 476-496 (static Python branch)
        aquad = jnp.where(is_c3, colim_c3a, colim_c4a)
        bquad = -(ac + aj)
        cquad = ac * aj
        r1, r2 = quadratic(aquad, bquad, cquad)
        ai = jnp.minimum(r1, r2)
        # C4 second co-limitation
        aquad2 = colim_c4b
        bquad2 = -(ai + ap)
        cquad2 = ai * ap
        r1b, r2b = quadratic(aquad2, bquad2, cquad2)
        ai_c4 = jnp.minimum(r1b, r2b)
        return jnp.where(is_c3, ai, ai_c4)

    else:
        endrun(msg=" ERROR: RealizedRate: colim_type not valid")
        return jnp.zeros(())  # Unreachable


# ---------------------------------------------------------------------------
# Task D helpers: JAX-traceable Ci residual, scan-based solver, layer kernel
# ---------------------------------------------------------------------------


def _CiFuncPure_jax(
    ci_val,
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    vcmax_ic,
    je_ic,
    kp_ic,
    rd_ic,
    kc_ic,
    ko_ic,
    cp_ic,
    o2ref_p: float,
    cair_ic,
    apar_ic,
    gbc_ic,
    gbv_ic,
    g0_p: float,
    g1_p: float,
    ceair_ic,
    lesat_ic,
    c3psn_pft_val: float,
    dpai_ic,
):
    """
    JAX-traceable version of :func:`_CiFuncPure`.

    Replaces ``math.*`` with ``jnp.*`` and Python ``if`` on traced values
    with ``jnp.where``, enabling use inside ``jax.lax.scan`` (Task D) and
    ``jax.lax.fori_loop``.

    Static branches on ``is_c3``, ``gs_type``, and ``colim_type`` remain as
    Python ``if/else`` because they are compile-time constants (PFT-level or
    module-level globals) — JAX bakes them in at trace time.

    Returns ``ci_dif = ci_new - ci_val`` (zero at convergence) as a JAX
    scalar.  Returns ``jnp.zeros(())`` when ``dpai_ic <= 0``.

    Mirrors :func:`_CiFuncPure` (lines 581-667).
    """
    if dpai_ic <= 0.0:  # static Python check — dpai is pre-extracted float
        return jnp.zeros(())

    if is_c3:  # static Python branch — baked in at trace time
        ac_val = (
            vcmax_ic * jnp.maximum(ci_val - cp_ic, 0.0) / (ci_val + kc_ic * (1.0 + o2ref_p / ko_ic))
        )
        aj_val = je_ic * jnp.maximum(ci_val - cp_ic, 0.0) / (4.0 * ci_val + 8.0 * cp_ic)
        ap_val = jnp.zeros(())
    else:
        ac_val = jnp.asarray(vcmax_ic)
        aj_val = jnp.asarray(qe_c4 * apar_ic)
        ap_val = kp_ic * jnp.maximum(ci_val, 0.0)

    agross_val = _RealizedRate(colim_type, c3psn_pft_val, ac_val, aj_val, ap_val)
    ac_val = jnp.maximum(ac_val, 0.0)
    aj_val = jnp.maximum(aj_val, 0.0)
    ap_val = jnp.maximum(ap_val, 0.0)
    agross_val = jnp.maximum(agross_val, 0.0)
    anet_val = agross_val - rd_ic
    # Guard denominators against zero for NaN-safe backward pass
    _eps = jnp.asarray(1e-30)
    gbc_safe = jnp.maximum(gbc_ic, _eps)
    cs_val = jnp.maximum(cair_ic - anet_val / gbc_safe, 1.0)

    if gs_type == 1:  # Ball-Berry — static Python branch
        term = anet_val / cs_val
        bq_gs = gbv_ic - g0_p - g1_p * term
        lesat_safe = jnp.maximum(lesat_ic, _eps)
        cq_gs = -gbv_ic * (g0_p + g1_p * term * ceair_ic / lesat_safe)
        r1, r2 = quadratic(1.0, bq_gs, cq_gs)
        gs_pos = jnp.maximum(r1, r2)
        gs_val = jnp.where(anet_val > 0.0, gs_pos, jnp.asarray(g0_p))

    elif gs_type == 0:  # Medlyn — static Python branch
        vpd_term = jnp.maximum(lesat_ic - ceair_ic, vpd_min_MED) * 0.001
        term = dh2o_to_dco2 * anet_val / cs_val
        gbv_safe = jnp.maximum(gbv_ic, _eps)
        bq_gs = -(2.0 * (g0_p + term) + (g1_p * term) ** 2 / (gbv_safe * vpd_term))
        cq_gs = g0_p * g0_p + (2.0 * g0_p + term * (1.0 - g1_p * g1_p / vpd_term)) * term
        r1, r2 = quadratic(1.0, bq_gs, cq_gs)
        gs_pos = jnp.maximum(r1, r2)
        gs_val = jnp.where(anet_val > 0.0, gs_pos, jnp.asarray(g0_p))

    else:  # WUE fallback (gs_type == 2)
        gs_val = jnp.asarray(g0_p)

    gs_safe = jnp.maximum(gs_val, _eps)
    gleaf = 1.0 / (1.0 / gbc_safe + dh2o_to_dco2 / gs_safe)
    cinew = cair_ic - anet_val / gleaf
    ci_dif = cinew - ci_val
    # When anet_val < 0: solver sets ci_dif = 0 (Fortran line 330)
    return jnp.where(anet_val < 0.0, jnp.zeros(()), ci_dif)


def _ci_solver_scan(ci0, ci1, func_kwargs, n_iter=40):
    """
    Fixed-iteration secant solver for intercellular CO2 concentration (Ci).

    Replaces :func:`hybrid_scalar` (Python ``while``-loop, max ``itmax=40``
    per Fortran) with ``jax.lax.scan`` of exactly ``n_iter`` steps.  The
    secant update mirrors the iteration structure of the Fortran ``hybrid()``
    root-finder, enabling JIT compilation and vmap batching.

    **n_iter = 40**: equals Fortran's ``itmax`` parameter in ``hybrid()``.
    Empirically, convergence to ``|dci| < 0.1 µmol mol⁻¹`` occurs in
    < 15 iterations for typical CHATS7 tower-site forcing (C3 plants
    starting from ``ci0 = 0.7 * cair``); 40 is conservative and safe.

    **Physical approximation**: Unlike ``hybrid_scalar`` (which can switch
    to Brent's method when a bracket is found), this solver runs exactly
    ``n_iter`` secant steps.  For well-behaved Ci functions the results
    are identical to ``rtol < 1e-10``; for edge cases (e.g. very low light)
    they agree to ``rtol ~ 1e-6``.

    **Parity test**::

        ci_hybrid = hybrid_scalar('test', _CiFuncPure_closure, ci0, ci1, tol)
        ci_scan   = _ci_solver_scan(
            jnp.float64(ci0), jnp.float64(ci1), func_kwargs, n_iter=40
        )
        assert jnp.allclose(ci_scan, ci_hybrid, rtol=1e-10)

    Args:
        ci0: First initial estimate (JAX scalar, float64).
        ci1: Second initial estimate (JAX scalar, float64).
        func_kwargs: Dict of keyword arguments forwarded to
            :func:`_CiFuncPure_jax`; contains JAX scalars and Python
            constants (compile-time constants under lax.scan).
        n_iter: Number of secant iterations (default 40 = Fortran itmax).

    Returns:
        Converged Ci estimate as a JAX scalar.
    """
    f0 = _CiFuncPure_jax(ci0, **func_kwargs)
    f1 = _CiFuncPure_jax(ci1, **func_kwargs)

    def body(carry, _):
        x0, x1, f0, f1 = carry
        # sign * max(|df|, eps) avoids select-as-denominator → no select_divide_fusion
        _df = f1 - f0
        _df_abs_safe = jnp.maximum(jnp.abs(_df), jnp.asarray(1.0e-30))
        _df_sign = jnp.where(_df < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        # Identical residuals (f1 == f0, typically both round-off after convergence) leave
        # the secant step undefined: take no step. Dividing by the 1e-30 floor instead
        # throws a converged iterate far away (e.g. Obukhov length 73.6 m -> -128 m).
        dx = jnp.where(_df == 0.0, 0.0, -f1 * (x1 - x0) * _df_sign / _df_abs_safe)
        x_new = x1 + dx
        f_new = _CiFuncPure_jax(x_new, **func_kwargs)
        return (x1, x_new, f1, f_new), None

    (_, x_final, _, _), _ = jax.lax.scan(
        body,
        (ci0, ci1, f0, f1),
        None,
        length=n_iter,
    )
    return x_final


def _ci_solver_scan_ift(ci0, ci1, func_kwargs, n_iter=40):
    """Ci secant solver with IFT-corrected gradient (Implicit Function Theorem).

    Mirrors the pattern used in ``_bisect_gs_ift`` for WUE:
    - Forward: run secant scan to find ci_root (stop its gradient)
    - Backward: one Newton refinement  ci_ift = ci0 - F(ci0;θ)/stop_grad(∂F/∂ci)
      where θ = {g0, g1, ...} in func_kwargs. This gives d(ci_ift)/dθ = -(∂F/∂θ)/(∂F/∂ci),
      the exact IFT gradient, without differentiating through 40 scan iterations.

    Used by Medlyn / Ball-Berry kernels where _ci_solver_scan gradient is NaN.
    """
    ci_scan = _ci_solver_scan(ci0, ci1, func_kwargs, n_iter=n_iter)
    ci0_ift = jax.lax.stop_gradient(ci_scan)

    # Residual at ci0_ift: gradient flows through func_kwargs (g0, g1, etc.)
    f_ci = _CiFuncPure_jax(ci0_ift, **func_kwargs)

    # ∂F/∂ci: central FD at ci0_ift; fully stop_gradient'd (denominator only)
    _delta = jnp.asarray(1e-4, dtype=ci0_ift.dtype)
    df_dci = jax.lax.stop_gradient(
        (
            _CiFuncPure_jax(ci0_ift + _delta, **func_kwargs)
            - _CiFuncPure_jax(ci0_ift - _delta, **func_kwargs)
        )
        / (2.0 * _delta)
    )

    # Apply Newton step only when ∂F/∂ci is well-defined
    _eps = jnp.asarray(1e-8, dtype=ci0_ift.dtype)
    apply = jnp.abs(df_dci) > _eps
    safe_f = jnp.where(apply, f_ci, jnp.zeros_like(f_ci))
    safe_denom = jnp.where(apply, df_dci, jnp.ones_like(df_dci))
    # Forward:  ci0_ift - ~0/df = ci0_ift = ci_root  ✓
    # Backward: -(∂F/∂θ) / safe_denom = IFT  ✓
    return ci0_ift - safe_f / safe_denom


@functools.lru_cache(maxsize=None)
def _make_leaf_photo_kernel(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val: float,
    vcmaxha: float,
    vcmaxhd: float,
    vcmaxse: float,
    vcmaxc: float,
    jmaxha: float,
    jmaxhd: float,
    jmaxse: float,
    jmaxc: float,
    rdc: float,
):
    """
    Factory returning a per-layer leaf-photosynthesis kernel for
    ``jax.vmap`` over the canopy layer dimension (Task D).

    Patch-level constants (PFT pathway, temperature-response coefficients,
    stomatal parameters) are closed over and treated as compile-time
    constants by JAX.  Only per-layer inputs vary across the vmap batch.

    Applies to ``gs_type in {0, 1}`` (Medlyn / Ball-Berry).  ``gs_type == 2``
    (WUE / :func:`_StomataOptimization`) retains the existing Python loop.

    **Parity test** (run before accepting this change)::

        kernel = _make_leaf_photo_kernel(...)
        vmapped = jax.jit(jax.vmap(kernel, in_axes=0))
        out_vmap = vmapped(dpai_arr, tleaf_arr, ...)
        # Compare against existing per-layer Python loop results:
        assert jnp.allclose(out_vmap[10], gs_ref_arr, rtol=1e-10)  # gs field

    Expected speedup: O(ncan)-× on GPU via parallel layer execution;
    on CPU: ~2–4× from eliminating Python dispatch overhead per layer.

    Returns:
        A function ``kernel(dpai_ic, tleaf_ic, vcmax25_ic, jmax25_ic,
        rd25_ic, kp25_ic, eair_ic, apar_ic, gbc_ic, gbv_ic, cair_ic)``
        returning an 18-tuple of per-layer JAX scalars.
    """

    def kernel(
        dpai_ic,
        tleaf_ic,
        vcmax25_ic,
        jmax25_ic,
        rd25_ic,
        kp25_ic,
        eair_ic,
        apar_ic,
        gbc_ic,
        gbv_ic,
        cair_ic,
        g0_rt,  # broadcast scalar: stomatal min conductance (JAX runtime arg — differentiable)
        g1_rt,  # broadcast scalar: stomatal slope parameter (JAX runtime arg — differentiable)
        o2ref_p,  # broadcast scalar: O2 (mmol/mol), forcing (runtime arg)
    ):
        """Per-layer photosynthesis + stomatal conductance kernel."""
        active = dpai_ic > 0.0

        # Guard against division-by-zero for inactive layers (NaN-safe backward pass)
        _safe = jnp.asarray(1.0)
        gbc_ic = jnp.where(active, gbc_ic, _safe)
        gbv_ic = jnp.where(active, gbv_ic, _safe)
        cair_ic = jnp.where(active, cair_ic, _safe)
        eair_ic = jnp.where(active, eair_ic, _safe)

        # --- Temperature responses — Fortran lines 161-175 ---
        # Uses _ft/_fth (jnp.exp) instead of _ft_py/_fth_py (math.exp)
        # so the kernel is fully JAX-traceable for scan / vmap / jit.
        kc_val = kc25 * _ft(tleaf_ic, kcha)
        ko_val = ko25 * _ft(tleaf_ic, koha)
        cp_val = cp25 * _ft(tleaf_ic, cpha)
        vcmax_val = vcmax25_ic * _ft(tleaf_ic, vcmaxha) * _fth(tleaf_ic, vcmaxhd, vcmaxse, vcmaxc)
        jmax_val = jmax25_ic * _ft(tleaf_ic, jmaxha) * _fth(tleaf_ic, jmaxhd, jmaxse, jmaxc)
        rd_val = rd25_ic * _ft(tleaf_ic, rdha) * _fth(tleaf_ic, rdhd, rdse, rdc)
        kp_val = jnp.zeros(())

        if not is_c3:  # static Python branch — C4 Q10 override
            t1 = jnp.exp(jnp.log(jnp.asarray(2.0)) * ((tleaf_ic - (tfrz + 25.0)) / 10.0))
            t2 = 1.0 + jnp.exp(jnp.asarray(0.2) * ((tfrz + 15.0) - tleaf_ic))
            t3 = 1.0 + jnp.exp(jnp.asarray(0.3) * (tleaf_ic - (tfrz + 40.0)))
            t4 = 1.0 + jnp.exp(jnp.asarray(1.3) * (tleaf_ic - (tfrz + 55.0)))
            vcmax_val = vcmax25_ic * t1 / (t2 * t3)
            rd_val = rd25_ic * t1 / t4
            kp_val = kp25_ic * t1

        # --- Saturation vapour pressure — Fortran lines 190-198 ---
        # SatVap uses jnp.where throughout — already JAX-traceable.
        lesat_val, _ = SatVap(tleaf_ic)
        ceair_val = jnp.minimum(eair_ic, lesat_val)
        if gs_type == 1:  # static Python branch
            ceair_val = jnp.maximum(ceair_val, rh_min_BB * lesat_val)

        # --- Electron transport rate — Fortran lines 200-205 ---
        qabs = 0.5 * phi_psII * apar_ic
        bq_j = -(qabs + jmax_val)
        cq_j = qabs * jmax_val
        r1j, r2j = quadratic(theta_j, bq_j, cq_j)
        je_val = jnp.minimum(r1j, r2j)

        # --- Ci root solver — Fortran lines 207-221 ---
        # Replaces hybrid_scalar (Python while-loop, blocks JIT/vmap)
        # with _ci_solver_scan (jax.lax.scan, 40 fixed iterations).
        # n_iter=40 matches Fortran's itmax; typical convergence < 15 iters.
        ci0_v = jnp.where(is_c3, 0.7 * cair_ic, 0.4 * cair_ic)
        ci1_v = ci0_v * 0.99

        _fkw = dict(
            is_c3=is_c3,
            gs_type=gs_type,
            colim_type=colim_type,
            vcmax_ic=vcmax_val,
            je_ic=je_val,
            kp_ic=kp_val,
            rd_ic=rd_val,
            kc_ic=kc_val,
            ko_ic=ko_val,
            cp_ic=cp_val,
            o2ref_p=o2ref_p,
            cair_ic=cair_ic,
            apar_ic=apar_ic,
            gbc_ic=gbc_ic,
            gbv_ic=gbv_ic,
            g0_p=g0_rt,
            g1_p=g1_rt,
            ceair_ic=ceair_val,
            lesat_ic=lesat_val,
            c3psn_pft_val=c3psn_pft_val,
            dpai_ic=1.0,  # always 1.0: inactive layers masked by jnp.where(active,...) below
        )
        ci_root = _ci_solver_scan_ift(ci0_v, ci1_v, _fkw, n_iter=40)

        # --- Final photosynthesis at ci_root — Fortran lines 221-240 ---
        if is_c3:
            _ac = (
                vcmax_val
                * jnp.maximum(ci_root - cp_val, 0.0)
                / (ci_root + kc_val * (1.0 + o2ref_p / ko_val))
            )
            _aj = je_val * jnp.maximum(ci_root - cp_val, 0.0) / (4.0 * ci_root + 8.0 * cp_val)
            _ap = jnp.zeros(())
        else:
            _ac = vcmax_val
            _aj = qe_c4 * apar_ic
            _ap = kp_val * jnp.maximum(ci_root, 0.0)

        _agross = _RealizedRate(colim_type, c3psn_pft_val, _ac, _aj, _ap)
        _ac = jnp.maximum(_ac, 0.0)
        _aj = jnp.maximum(_aj, 0.0)
        _ap = jnp.maximum(_ap, 0.0)
        _agross = jnp.maximum(_agross, 0.0)
        _anet = _agross - rd_val
        _eps_k = jnp.asarray(1e-30)
        _cs = jnp.maximum(cair_ic - _anet / gbc_ic, 1.0)

        if gs_type == 1:  # Ball-Berry — static Python branch
            _term = _anet / _cs
            _bq2 = gbv_ic - g0_rt - g1_rt * _term
            _lesat_safe = jnp.maximum(lesat_val, _eps_k)
            _cq2 = -gbv_ic * (g0_rt + g1_rt * _term * ceair_val / _lesat_safe)
            _r1, _r2 = quadratic(1.0, _bq2, _cq2)
            _gs_pos = jnp.maximum(_r1, _r2)
            _gs = jnp.where(_anet > 0.0, _gs_pos, g0_rt)
        else:  # Medlyn (gs_type == 0) — static Python branch
            _vpdt = jnp.maximum(lesat_val - ceair_val, vpd_min_MED) * 0.001
            _term = dh2o_to_dco2 * _anet / _cs
            _gbv_vpdt = jnp.maximum(gbv_ic * _vpdt, _eps_k)
            _bq2 = -(2.0 * (g0_rt + _term) + (g1_rt * _term) ** 2 / _gbv_vpdt)
            _cq2 = g0_rt * g0_rt + (2.0 * g0_rt + _term * (1.0 - g1_rt * g1_rt / _vpdt)) * _term
            _r1, _r2 = quadratic(1.0, _bq2, _cq2)
            _gs_pos = jnp.maximum(_r1, _r2)
            _gs = jnp.where(_anet > 0.0, _gs_pos, g0_rt)

        # --- Mask outputs for empty layers — Fortran lines 247-257 ---
        zero = jnp.zeros(())
        return (
            jnp.where(active, kc_val, zero),  # [0]  kc
            jnp.where(active, ko_val, zero),  # [1]  ko
            jnp.where(active, cp_val, zero),  # [2]  cp
            jnp.where(active, vcmax_val, zero),  # [3]  vcmax
            jnp.where(active, jmax_val, zero),  # [4]  jmax
            jnp.where(active, rd_val, zero),  # [5]  rd
            jnp.where(active, kp_val, zero),  # [6]  kp
            jnp.where(active, lesat_val, zero),  # [7]  leaf_esat
            jnp.where(active, ceair_val, zero),  # [8]  ceair
            jnp.where(active, je_val, zero),  # [9]  je
            jnp.where(active, _gs, zero),  # [10] gs
            jnp.where(active, ci_root, zero),  # [11] ci
            jnp.where(active, _ac, zero),  # [12] ac
            jnp.where(active, _aj, zero),  # [13] aj
            jnp.where(active, _ap, zero),  # [14] ap
            jnp.where(active, _agross, zero),  # [15] agross
            jnp.where(active, _anet, zero),  # [16] anet
            jnp.where(active, _cs, zero),  # [17] cs
        )

    return kernel


@functools.lru_cache(maxsize=None)
def _get_vmapped_photo_kernel(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val: float,
    vcmaxha: float,
    vcmaxhd: float,
    vcmaxse: float,
    vcmaxc: float,
    jmaxha: float,
    jmaxhd: float,
    jmaxse: float,
    jmaxc: float,
    rdc: float,
):
    """Return a JIT-compiled vmapped leaf-photosynthesis kernel (gs_type 0/1).

    Results are cached by parameter values so the same XLA compilation is
    reused across all sub-steps for a fixed site.  g0_val and g1_val are
    NOT in the cache key — they are broadcast runtime args (in_axes=None)
    so gradients flow through them for d(GPP)/d(g0) and d(GPP)/d(g1).
    """
    kernel = _make_leaf_photo_kernel(
        is_c3=is_c3,
        gs_type=gs_type,
        colim_type=colim_type,
        c3psn_pft_val=c3psn_pft_val,
        vcmaxha=vcmaxha,
        vcmaxhd=vcmaxhd,
        vcmaxse=vcmaxse,
        vcmaxc=vcmaxc,
        jmaxha=jmaxha,
        jmaxhd=jmaxhd,
        jmaxse=jmaxse,
        jmaxc=jmaxc,
        rdc=rdc,
    )
    # First 11 args are per-layer (axis 0); g0_rt, g1_rt and o2ref_p are broadcast scalars.
    return jax.jit(jax.vmap(kernel, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None)))


# ---------------------------------------------------------------------------
# Private: vmapped leaf-photosynthesis kernel for acclim_type == 1
# (vcmaxse, vcmaxc, jmaxse, jmaxc are JAX runtime scalars, not closed-over
#  Python floats, so this kernel is compatible with jax.checkpoint tracing)
# ---------------------------------------------------------------------------


def _make_leaf_photo_kernel_acclim(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val: float,
    vcmaxha: float,
    vcmaxhd: float,
    jmaxha: float,
    jmaxhd: float,
    rdc: float,
):
    """
    Factory returning a per-layer leaf-photosynthesis kernel for acclim_type==1.

    Like :func:`_make_leaf_photo_kernel` but accepts ``vcmaxse_rt``,
    ``vcmaxc_rt``, ``jmaxse_rt``, ``jmaxc_rt`` as runtime scalar arguments
    (broadcast via ``in_axes=None`` in vmap) so this kernel is compatible
    with ``jax.checkpoint`` where these values are JAX traced arrays, not
    Python floats.

    ``g0_rt`` and ``g1_rt`` are also runtime args so gradients flow through
    stomatal parameters (d(GPP)/d(g0), d(GPP)/d(g1)).

    The non-acclim params (vcmaxha, vcmaxhd, jmaxha, jmaxhd, rdc, etc.)
    remain closed-over Python floats for XLA constant folding.
    """

    def kernel(
        dpai_ic,
        tleaf_ic,
        vcmax25_ic,
        jmax25_ic,
        rd25_ic,
        kp25_ic,
        eair_ic,
        apar_ic,
        gbc_ic,
        gbv_ic,
        cair_ic,
        g0_rt,  # broadcast scalar: stomatal min conductance (JAX runtime arg)
        g1_rt,  # broadcast scalar: stomatal slope parameter (JAX runtime arg)
        vcmaxse_rt,  # broadcast scalar: JAX runtime arg
        vcmaxc_rt,  # broadcast scalar: JAX runtime arg
        jmaxse_rt,  # broadcast scalar: JAX runtime arg
        jmaxc_rt,  # broadcast scalar: JAX runtime arg
        o2ref_p,  # broadcast scalar: O2 (mmol/mol), forcing (runtime arg)
    ):
        """Per-layer photosynthesis + stomatal conductance kernel (acclim)."""
        active = dpai_ic > 0.0

        # Guard against division-by-zero for inactive layers (NaN-safe backward pass)
        _safe = jnp.asarray(1.0)
        gbc_ic = jnp.where(active, gbc_ic, _safe)
        gbv_ic = jnp.where(active, gbv_ic, _safe)
        cair_ic = jnp.where(active, cair_ic, _safe)
        eair_ic = jnp.where(active, eair_ic, _safe)

        # --- Temperature responses (uses runtime vcmaxse_rt / jmaxse_rt) ---
        kc_val = kc25 * _ft(tleaf_ic, kcha)
        ko_val = ko25 * _ft(tleaf_ic, koha)
        cp_val = cp25 * _ft(tleaf_ic, cpha)
        vcmax_val = (
            vcmax25_ic * _ft(tleaf_ic, vcmaxha) * _fth(tleaf_ic, vcmaxhd, vcmaxse_rt, vcmaxc_rt)
        )
        jmax_val = jmax25_ic * _ft(tleaf_ic, jmaxha) * _fth(tleaf_ic, jmaxhd, jmaxse_rt, jmaxc_rt)
        rd_val = rd25_ic * _ft(tleaf_ic, rdha) * _fth(tleaf_ic, rdhd, rdse, rdc)
        kp_val = jnp.zeros(())

        if not is_c3:
            t1 = jnp.exp(jnp.log(jnp.asarray(2.0)) * ((tleaf_ic - (tfrz + 25.0)) / 10.0))
            t2 = 1.0 + jnp.exp(jnp.asarray(0.2) * ((tfrz + 15.0) - tleaf_ic))
            t3 = 1.0 + jnp.exp(jnp.asarray(0.3) * (tleaf_ic - (tfrz + 40.0)))
            t4 = 1.0 + jnp.exp(jnp.asarray(1.3) * (tleaf_ic - (tfrz + 55.0)))
            vcmax_val = vcmax25_ic * t1 / (t2 * t3)
            rd_val = rd25_ic * t1 / t4
            kp_val = kp25_ic * t1

        lesat_val, _ = SatVap(tleaf_ic)
        ceair_val = jnp.minimum(eair_ic, lesat_val)
        if gs_type == 1:
            ceair_val = jnp.maximum(ceair_val, rh_min_BB * lesat_val)

        qabs = 0.5 * phi_psII * apar_ic
        bq_j = -(qabs + jmax_val)
        cq_j = qabs * jmax_val
        r1j, r2j = quadratic(theta_j, bq_j, cq_j)
        je_val = jnp.minimum(r1j, r2j)

        ci0_v = jnp.where(is_c3, 0.7 * cair_ic, 0.4 * cair_ic)
        ci1_v = ci0_v * 0.99

        _fkw = dict(
            is_c3=is_c3,
            gs_type=gs_type,
            colim_type=colim_type,
            vcmax_ic=vcmax_val,
            je_ic=je_val,
            kp_ic=kp_val,
            rd_ic=rd_val,
            kc_ic=kc_val,
            ko_ic=ko_val,
            cp_ic=cp_val,
            o2ref_p=o2ref_p,
            cair_ic=cair_ic,
            apar_ic=apar_ic,
            gbc_ic=gbc_ic,
            gbv_ic=gbv_ic,
            g0_p=g0_rt,
            g1_p=g1_rt,
            ceair_ic=ceair_val,
            lesat_ic=lesat_val,
            c3psn_pft_val=c3psn_pft_val,
            dpai_ic=1.0,
        )
        ci_root = _ci_solver_scan_ift(ci0_v, ci1_v, _fkw, n_iter=40)

        if is_c3:
            _ac = (
                vcmax_val
                * jnp.maximum(ci_root - cp_val, 0.0)
                / (ci_root + kc_val * (1.0 + o2ref_p / ko_val))
            )
            _aj = je_val * jnp.maximum(ci_root - cp_val, 0.0) / (4.0 * ci_root + 8.0 * cp_val)
            _ap = jnp.zeros(())
        else:
            _ac = vcmax_val
            _aj = qe_c4 * apar_ic
            _ap = kp_val * jnp.maximum(ci_root, 0.0)

        _agross = _RealizedRate(colim_type, c3psn_pft_val, _ac, _aj, _ap)
        _ac = jnp.maximum(_ac, 0.0)
        _aj = jnp.maximum(_aj, 0.0)
        _ap = jnp.maximum(_ap, 0.0)
        _agross = jnp.maximum(_agross, 0.0)
        _anet = _agross - rd_val
        _eps_k = jnp.asarray(1e-30)
        _cs = jnp.maximum(cair_ic - _anet / gbc_ic, 1.0)

        if gs_type == 1:
            _term = _anet / _cs
            _bq2 = gbv_ic - g0_rt - g1_rt * _term
            _lesat_safe = jnp.maximum(lesat_val, _eps_k)
            _cq2 = -gbv_ic * (g0_rt + g1_rt * _term * ceair_val / _lesat_safe)
            _r1, _r2 = quadratic(1.0, _bq2, _cq2)
            _gs_pos = jnp.maximum(_r1, _r2)
            _gs = jnp.where(_anet > 0.0, _gs_pos, g0_rt)
        else:
            _vpdt = jnp.maximum(lesat_val - ceair_val, vpd_min_MED) * 0.001
            _term = dh2o_to_dco2 * _anet / _cs
            _gbv_vpdt = jnp.maximum(gbv_ic * _vpdt, _eps_k)
            _bq2 = -(2.0 * (g0_rt + _term) + (g1_rt * _term) ** 2 / _gbv_vpdt)
            _cq2 = g0_rt * g0_rt + (2.0 * g0_rt + _term * (1.0 - g1_rt * g1_rt / _vpdt)) * _term
            _r1, _r2 = quadratic(1.0, _bq2, _cq2)
            _gs_pos = jnp.maximum(_r1, _r2)
            _gs = jnp.where(_anet > 0.0, _gs_pos, g0_rt)

        zero = jnp.zeros(())
        return (
            jnp.where(active, kc_val, zero),
            jnp.where(active, ko_val, zero),
            jnp.where(active, cp_val, zero),
            jnp.where(active, vcmax_val, zero),
            jnp.where(active, jmax_val, zero),
            jnp.where(active, rd_val, zero),
            jnp.where(active, kp_val, zero),
            jnp.where(active, lesat_val, zero),
            jnp.where(active, ceair_val, zero),
            jnp.where(active, je_val, zero),
            jnp.where(active, _gs, zero),
            jnp.where(active, ci_root, zero),
            jnp.where(active, _ac, zero),
            jnp.where(active, _aj, zero),
            jnp.where(active, _ap, zero),
            jnp.where(active, _agross, zero),
            jnp.where(active, _anet, zero),
            jnp.where(active, _cs, zero),
        )

    return kernel


@functools.lru_cache(maxsize=None)
def _get_vmapped_photo_kernel_acclim(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val: float,
    vcmaxha: float,
    vcmaxhd: float,
    jmaxha: float,
    jmaxhd: float,
    rdc: float,
):
    """Return a JIT-compiled vmapped leaf-photosynthesis kernel for acclim_type==1.

    ``vcmaxse``, ``vcmaxc``, ``jmaxse``, ``jmaxc`` are NOT passed here; they
    are broadcast runtime arguments (in_axes=None) to the vmapped kernel so
    this function can be cached independently of their values, and the kernel
    remains compatible with ``jax.checkpoint`` tracing (no ``float()`` required).

    ``g0_rt`` and ``g1_rt`` are also runtime args (not in cache key) so
    gradients flow through d(GPP)/d(g0) and d(GPP)/d(g1).
    """
    kernel = _make_leaf_photo_kernel_acclim(
        is_c3=is_c3,
        gs_type=gs_type,
        colim_type=colim_type,
        c3psn_pft_val=c3psn_pft_val,
        vcmaxha=vcmaxha,
        vcmaxhd=vcmaxhd,
        jmaxha=jmaxha,
        jmaxhd=jmaxhd,
        rdc=rdc,
    )
    # First 11 args are per-layer (in_axes=0); g0_rt, g1_rt, 4 acclim scalars and o2ref_p
    # are broadcast.
    return jax.jit(
        jax.vmap(
            kernel,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None),
        )
    )


# ---------------------------------------------------------------------------
# Private: JAX-traceable scalar versions for differentiable gs_type==2 path
# ---------------------------------------------------------------------------


def _CiFuncGsJax(
    gs_val,
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    dpai_ic,
    gbc_ic,
    cair_ic,
    vcmax_ic,
    je_ic,
    kp_ic,
    rd_ic,
    kc_ic,
    ko_ic,
    cp_ic,
    o2ref_p,
    apar_ic,
    c3psn_pft_val,
):
    """JAX-traceable version of _CiFuncGsPure. Uses jnp ops throughout."""
    _eps = jnp.asarray(1e-30)
    gbc_safe = jnp.maximum(gbc_ic, _eps)
    gs_safe = jnp.maximum(gs_val, _eps)
    gleaf = 1.0 / (1.0 / gbc_safe + dh2o_to_dco2 / gs_safe)

    if is_c3:
        a0 = vcmax_ic
        ko_safe = jnp.maximum(ko_ic, _eps)
        b0 = kc_ic * (1.0 + o2ref_p / ko_safe)
        aq = 1.0 / gleaf
        bq = -(cair_ic + b0) - (a0 - rd_ic) / gleaf
        cq = a0 * (cair_ic - cp_ic) - rd_ic * (cair_ic + b0)
        r1, r2 = quadratic(aq, bq, cq)
        ac_val = jnp.minimum(r1, r2) + rd_ic

        a0j = je_ic / 4.0
        b0j = 2.0 * cp_ic
        aqj = 1.0 / gleaf
        bqj = -(cair_ic + b0j) - (a0j - rd_ic) / gleaf
        cqj = a0j * (cair_ic - cp_ic) - rd_ic * (cair_ic + b0j)
        r1j, r2j = quadratic(aqj, bqj, cqj)
        aj_val = jnp.minimum(r1j, r2j) + rd_ic

        ap_val = jnp.zeros(())
    else:
        ac_val = vcmax_ic
        aj_val = qe_c4 * apar_ic
        ap_val = kp_ic * (cair_ic * gleaf + rd_ic) / (gleaf + kp_ic)

    agross_val = _RealizedRate(colim_type, c3psn_pft_val, ac_val, aj_val, ap_val)
    anet_val = agross_val - rd_ic
    cs_val = jnp.maximum(cair_ic - anet_val / gbc_safe, 1.0)
    ci_val = cair_ic - anet_val / gleaf

    return ci_val, ac_val, aj_val, ap_val, agross_val, anet_val, cs_val


def _StomataEfficiencyJax(
    gs_val,
    *,
    iota,
    pref_p,
    eair_ic,
    gbv_ic,
    lesat_ic,
    **ci_kwargs,
):
    """JAX-traceable version of _StomataEfficiencyPure."""
    _eps = jnp.asarray(1e-30)
    delta = 0.001
    _, _, _, _, _, an_low, _ = _CiFuncGsJax(gs_val - delta, **ci_kwargs)
    _, _, _, _, _, an_high, _ = _CiFuncGsJax(gs_val, **ci_kwargs)

    lesat_safe = jnp.maximum(lesat_ic, _eps)
    denom = jnp.maximum((gbv_ic + gs_val) * lesat_safe, _eps)
    hs = (gbv_ic * eair_ic + gs_val * lesat_safe) / denom
    vpd = jnp.maximum(lesat_safe - hs * lesat_safe, vpd_min_MED)

    pref_safe = jnp.maximum(pref_p, _eps)
    return (an_high - an_low) - iota * delta * (vpd / pref_safe)


def _bisect_gs_jax(gsmin, gs_upper, se_kwargs, n_iter=20):
    """JAX-traceable bisection for stomatal optimization via fori_loop.

    Returns (gs_opt, bracket_ok) where bracket_ok=True means a sign change
    was found and gs_opt is near the root; bracket_ok=False means no root
    exists and gs_opt=gsmin.
    """
    fa = _StomataEfficiencyJax(gsmin, **se_kwargs)
    fb = _StomataEfficiencyJax(gs_upper, **se_kwargs)

    # If no sign change, return gsmin
    bracket_ok = fa * fb < 0.0

    def body(_, carry):
        a, b, f_a = carry
        mid = 0.5 * (a + b)
        f_mid = _StomataEfficiencyJax(mid, **se_kwargs)
        # If f_a and f_mid have same sign, root is in [mid, b]; else [a, mid]
        same_sign = f_a * f_mid > 0.0
        new_a = jnp.where(same_sign, mid, a)
        new_b = jnp.where(same_sign, b, mid)
        new_fa = jnp.where(same_sign, f_mid, f_a)
        return (new_a, new_b, new_fa)

    a0, b0, fa0 = jax.lax.fori_loop(0, n_iter, body, (gsmin, gs_upper, fa))
    gs_opt = jnp.where(bracket_ok, 0.5 * (a0 + b0), gsmin)
    return gs_opt, bracket_ok


def _bisect_gs_ift(gsmin, gs_upper, se_kwargs, n_iter=20):
    """Bisection root-finder with IFT-based gradient via Newton refinement.

    Uses the Implicit Function Theorem for the backward pass instead of
    differentiating through bisection iterations (which gives wrong gradients
    due to discrete jnp.where branch selections):

        d(gs*)/d(theta) = -(df/dtheta) / (df/dgs)  at gs*

    Implementation — Newton refinement identity:
        gs_ift = gs0 - f(gs0; theta) / stop_grad(df/dgs)

    where gs0 = stop_gradient(bisection result).

    - Forward: f(gs0) ≈ 0  ⟹  gs_ift ≈ gs0 = gs*  (correct root value)
    - Backward: d(gs_ift)/d(theta) = -∂f/∂theta / stop_grad(df/dgs) = IFT  ✓

    Avoids jax.lax.custom_root (which has version-dependent API for
    tangent_solve argument type).
    """
    # Forward: find root via bisection (gradient through bisection is wrong)
    gs_opt, bracket_ok = _bisect_gs_jax(gsmin, gs_upper, se_kwargs, n_iter)
    gs0 = jax.lax.stop_gradient(gs_opt)  # stop bisection gradient

    # Residual at gs0: gradient through se_kwargs flows here (correct IFT gradient)
    f0 = _StomataEfficiencyJax(gs0, **se_kwargs)

    # df/dgs at gs0: central finite difference, gradients stopped (denominator only)
    _delta = jnp.asarray(1e-4, dtype=gs0.dtype)
    df_dgs = jax.lax.stop_gradient(
        (
            _StomataEfficiencyJax(gs0 + _delta, **se_kwargs)
            - _StomataEfficiencyJax(gs0 - _delta, **se_kwargs)
        )
        / (2.0 * _delta)
    )

    # Apply Newton refinement only when:
    #   (a) bracket_ok=True: bisection found a root, so f(gs0) ≈ 0 and the
    #       Newton step is a small correction, not a catastrophic extrapolation.
    #       When bracket_ok=False (dark/no-root layers), f(gsmin) can be O(0.01),
    #       making gs_ift = gsmin - f/df blow up far from gsmin.
    #   (b) |df/dgs| > eps: the IFT is well-defined (non-degenerate root).
    #
    # Forward: safe_f0/safe_denom = 0/1 = 0  →  gs_ift = gs0  ✓
    # Backward: d(gs_ift)/dθ = 0  (zero gradient for no-root / degenerate layers) ✓
    _eps = jnp.asarray(1e-6, dtype=gs0.dtype)
    apply = bracket_ok & (jnp.abs(df_dgs) > _eps)
    safe_f0 = jnp.where(apply, f0, jnp.zeros_like(f0))
    safe_denom = jnp.where(apply, df_dgs, jnp.ones_like(df_dgs))

    # gs_ift = gs0 - f0 / safe_denom
    # Forward: gs0 - ~0 = gs0 = gs*  ✓
    # Backward: -∂f/∂theta / safe_denom = IFT  ✓
    return gs0 - safe_f0 / safe_denom


@functools.lru_cache(maxsize=None)
def _make_leaf_photo_kernel_wue(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val,
    vcmaxha,
    vcmaxhd,
    vcmaxse,
    vcmaxc,
    jmaxha,
    jmaxhd,
    jmaxse,
    jmaxc,
    rdc,
):
    """Factory returning a per-layer WUE photosynthesis kernel for jax.vmap.

    ``o2ref_p`` and ``pref_p`` are *not* closed over — they are passed as
    explicit broadcast (``in_axes=None``) arguments to ``jax.vmap`` so that
    per-sub-step changes in atmospheric pressure do not invalidate the
    JIT-compiled kernel cache.

    ``iota_rt`` and ``gsmin_rt`` are also runtime args (not in cache key) so
    gradients flow through d(GPP)/d(iota) and d(GPP)/d(gsmin) via the IFT.
    """

    def kernel(
        dpai_ic,
        tleaf_ic,
        vcmax25_ic,
        jmax25_ic,
        rd25_ic,
        kp25_ic,
        eair_ic,
        apar_ic,
        gbc_ic,
        gbv_ic,
        cair_ic,
        o2ref_p,
        pref_p,
        iota_rt,
        gsmin_rt,  # broadcast scalars: WUE params (JAX runtime args — differentiable)
    ):
        active = dpai_ic > 0.0

        # Guard against division-by-zero for inactive layers (NaN-safe backward pass)
        _safe = jnp.asarray(1.0)
        gbc_ic = jnp.where(active, gbc_ic, _safe)
        gbv_ic = jnp.where(active, gbv_ic, _safe)
        cair_ic = jnp.where(active, cair_ic, _safe)
        eair_ic = jnp.where(active, eair_ic, _safe)
        apar_ic = jnp.where(active, apar_ic, _safe)

        # Temperature responses (JAX)
        kc_val = kc25 * _ft(tleaf_ic, kcha)
        ko_val = ko25 * _ft(tleaf_ic, koha)
        cp_val = cp25 * _ft(tleaf_ic, cpha)
        vcmax_val = vcmax25_ic * _ft(tleaf_ic, vcmaxha) * _fth(tleaf_ic, vcmaxhd, vcmaxse, vcmaxc)
        jmax_val = jmax25_ic * _ft(tleaf_ic, jmaxha) * _fth(tleaf_ic, jmaxhd, jmaxse, jmaxc)
        rd_val = rd25_ic * _ft(tleaf_ic, rdha) * _fth(tleaf_ic, rdhd, rdse, rdc)
        kp_val = jnp.zeros(())

        if not is_c3:
            t1 = jnp.exp(jnp.log(jnp.asarray(2.0)) * ((tleaf_ic - (tfrz + 25.0)) / 10.0))
            t2 = 1.0 + jnp.exp(jnp.asarray(0.2) * ((tfrz + 15.0) - tleaf_ic))
            t3 = 1.0 + jnp.exp(jnp.asarray(0.3) * (tleaf_ic - (tfrz + 40.0)))
            t4 = 1.0 + jnp.exp(jnp.asarray(1.3) * (tleaf_ic - (tfrz + 55.0)))
            vcmax_val = vcmax25_ic * t1 / (t2 * t3)
            rd_val = rd25_ic * t1 / t4
            kp_val = kp25_ic * t1

        # Saturation vapour pressure
        lesat_val, _ = SatVap(tleaf_ic)
        ceair_val = jnp.minimum(eair_ic, lesat_val)

        # Electron transport rate
        qabs = 0.5 * phi_psII * apar_ic
        bq_j = -(qabs + jmax_val)
        cq_j = qabs * jmax_val
        r1j, r2j = quadratic(theta_j, bq_j, cq_j)
        je_val = jnp.minimum(r1j, r2j)

        # Stomatal optimization (bisection)
        ci_kwargs = dict(
            is_c3=is_c3,
            gs_type=gs_type,
            colim_type=colim_type,
            dpai_ic=dpai_ic,
            gbc_ic=gbc_ic,
            cair_ic=cair_ic,
            vcmax_ic=vcmax_val,
            je_ic=je_val,
            kp_ic=kp_val,
            rd_ic=rd_val,
            kc_ic=kc_val,
            ko_ic=ko_val,
            cp_ic=cp_val,
            o2ref_p=o2ref_p,
            apar_ic=apar_ic,
            c3psn_pft_val=c3psn_pft_val,
        )
        se_kwargs = dict(
            iota=iota_rt,
            pref_p=pref_p,
            eair_ic=eair_ic,
            gbv_ic=gbv_ic,
            lesat_ic=lesat_val,
            **ci_kwargs,
        )
        gs_opt = _bisect_gs_ift(gsmin_rt, 2.0, se_kwargs)

        # Final photosynthesis at gs_opt
        ci_f, ac_f, aj_f, ap_f, agross_f, anet_f, cs_f = _CiFuncGsJax(gs_opt, **ci_kwargs)

        # Mask inactive layers
        zero = jnp.zeros(())
        kc_val = jnp.where(active, kc_val, zero)
        ko_val = jnp.where(active, ko_val, zero)
        cp_val = jnp.where(active, cp_val, zero)
        vcmax_val = jnp.where(active, vcmax_val, zero)
        jmax_val = jnp.where(active, jmax_val, zero)
        rd_val = jnp.where(active, rd_val, zero)
        kp_val = jnp.where(active, kp_val, zero)
        lesat_val = jnp.where(active, lesat_val, zero)
        ceair_val = jnp.where(active, ceair_val, zero)
        je_val = jnp.where(active, je_val, zero)
        gs_opt = jnp.where(active, gs_opt, zero)
        ci_f = jnp.where(active, ci_f, zero)
        ac_f = jnp.where(active, ac_f, zero)
        aj_f = jnp.where(active, aj_f, zero)
        ap_f = jnp.where(active, ap_f, zero)
        agross_f = jnp.where(active, agross_f, zero)
        anet_f = jnp.where(active, anet_f, zero)
        cs_f = jnp.where(active, cs_f, zero)

        return (
            kc_val,
            ko_val,
            cp_val,
            vcmax_val,
            jmax_val,
            rd_val,
            kp_val,
            lesat_val,
            ceair_val,
            je_val,
            gs_opt,
            ci_f,
            ac_f,
            aj_f,
            ap_f,
            agross_f,
            anet_f,
            cs_f,
        )

    return kernel


@functools.lru_cache(maxsize=None)
def _get_vmapped_photo_kernel_wue(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val,
    vcmaxha,
    vcmaxhd,
    vcmaxse,
    vcmaxc,
    jmaxha,
    jmaxhd,
    jmaxse,
    jmaxc,
    rdc,
):
    """Return a JIT-compiled vmapped WUE leaf-photosynthesis kernel.

    ``o2ref_p`` and ``pref_p`` are passed via ``in_axes=None`` (broadcast)
    at vmap call time, so per-sub-step pressure variations do not trigger
    recompilation.

    ``iota_rt`` and ``gsmin_rt`` are NOT in the cache key — they are
    broadcast runtime args so gradients flow through d(GPP)/d(iota) and
    d(GPP)/d(gsmin) via the IFT path.
    """
    kernel = _make_leaf_photo_kernel_wue(
        is_c3=is_c3,
        gs_type=gs_type,
        colim_type=colim_type,
        c3psn_pft_val=c3psn_pft_val,
        vcmaxha=vcmaxha,
        vcmaxhd=vcmaxhd,
        vcmaxse=vcmaxse,
        vcmaxc=vcmaxc,
        jmaxha=jmaxha,
        jmaxhd=jmaxhd,
        jmaxse=jmaxse,
        jmaxc=jmaxc,
        rdc=rdc,
    )
    # in_axes: 11 per-layer (axis 0); o2ref_p, pref_p, iota_rt, gsmin_rt broadcast.
    return jax.jit(
        jax.vmap(
            kernel,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
        )
    )


@functools.lru_cache(maxsize=None)
def _make_leaf_photo_kernel_wue_acclim(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val,
    vcmaxha,
    vcmaxhd,
    jmaxha,
    jmaxhd,
    rdc,
):
    """Like _make_leaf_photo_kernel_wue but accepts vcmaxse/vcmaxc/jmaxse/jmaxc
    as runtime scalar arguments for jax.checkpoint compatibility (acclim_type==1).
    iota_rt and gsmin_rt are also runtime args so gradients flow through them.
    """

    def kernel(
        dpai_ic,
        tleaf_ic,
        vcmax25_ic,
        jmax25_ic,
        rd25_ic,
        kp25_ic,
        eair_ic,
        apar_ic,
        gbc_ic,
        gbv_ic,
        cair_ic,
        o2ref_p,
        pref_p,
        iota_rt,
        gsmin_rt,  # broadcast scalars: WUE params (JAX runtime args)
        vcmaxse_rt,
        vcmaxc_rt,
        jmaxse_rt,
        jmaxc_rt,
    ):
        active = dpai_ic > 0.0

        _safe = jnp.asarray(1.0)
        gbc_ic = jnp.where(active, gbc_ic, _safe)
        gbv_ic = jnp.where(active, gbv_ic, _safe)
        cair_ic = jnp.where(active, cair_ic, _safe)
        eair_ic = jnp.where(active, eair_ic, _safe)
        apar_ic = jnp.where(active, apar_ic, _safe)

        kc_val = kc25 * _ft(tleaf_ic, kcha)
        ko_val = ko25 * _ft(tleaf_ic, koha)
        cp_val = cp25 * _ft(tleaf_ic, cpha)
        vcmax_val = (
            vcmax25_ic * _ft(tleaf_ic, vcmaxha) * _fth(tleaf_ic, vcmaxhd, vcmaxse_rt, vcmaxc_rt)
        )
        jmax_val = jmax25_ic * _ft(tleaf_ic, jmaxha) * _fth(tleaf_ic, jmaxhd, jmaxse_rt, jmaxc_rt)
        rd_val = rd25_ic * _ft(tleaf_ic, rdha) * _fth(tleaf_ic, rdhd, rdse, rdc)
        kp_val = jnp.zeros(())

        if not is_c3:
            t1 = jnp.exp(jnp.log(jnp.asarray(2.0)) * ((tleaf_ic - (tfrz + 25.0)) / 10.0))
            t2 = 1.0 + jnp.exp(jnp.asarray(0.2) * ((tfrz + 15.0) - tleaf_ic))
            t3 = 1.0 + jnp.exp(jnp.asarray(0.3) * (tleaf_ic - (tfrz + 40.0)))
            t4 = 1.0 + jnp.exp(jnp.asarray(1.3) * (tleaf_ic - (tfrz + 55.0)))
            vcmax_val = vcmax25_ic * t1 / (t2 * t3)
            rd_val = rd25_ic * t1 / t4
            kp_val = kp25_ic * t1

        lesat_val, _ = SatVap(tleaf_ic)
        ceair_val = jnp.minimum(eair_ic, lesat_val)

        qabs = 0.5 * phi_psII * apar_ic
        bq_j = -(qabs + jmax_val)
        cq_j = qabs * jmax_val
        r1j, r2j = quadratic(theta_j, bq_j, cq_j)
        je_val = jnp.minimum(r1j, r2j)

        ci_kwargs = dict(
            is_c3=is_c3,
            gs_type=gs_type,
            colim_type=colim_type,
            dpai_ic=dpai_ic,
            gbc_ic=gbc_ic,
            cair_ic=cair_ic,
            vcmax_ic=vcmax_val,
            je_ic=je_val,
            kp_ic=kp_val,
            rd_ic=rd_val,
            kc_ic=kc_val,
            ko_ic=ko_val,
            cp_ic=cp_val,
            o2ref_p=o2ref_p,
            apar_ic=apar_ic,
            c3psn_pft_val=c3psn_pft_val,
        )
        se_kwargs = dict(
            iota=iota_rt,
            pref_p=pref_p,
            eair_ic=eair_ic,
            gbv_ic=gbv_ic,
            lesat_ic=lesat_val,
            **ci_kwargs,
        )
        gs_opt = _bisect_gs_ift(gsmin_rt, 2.0, se_kwargs)

        ci_f, ac_f, aj_f, ap_f, agross_f, anet_f, cs_f = _CiFuncGsJax(gs_opt, **ci_kwargs)

        zero = jnp.zeros(())
        kc_val = jnp.where(active, kc_val, zero)
        ko_val = jnp.where(active, ko_val, zero)
        cp_val = jnp.where(active, cp_val, zero)
        vcmax_val = jnp.where(active, vcmax_val, zero)
        jmax_val = jnp.where(active, jmax_val, zero)
        rd_val = jnp.where(active, rd_val, zero)
        kp_val = jnp.where(active, kp_val, zero)
        lesat_val = jnp.where(active, lesat_val, zero)
        ceair_val = jnp.where(active, ceair_val, zero)
        je_val = jnp.where(active, je_val, zero)
        gs_opt = jnp.where(active, gs_opt, zero)
        ci_f = jnp.where(active, ci_f, zero)
        ac_f = jnp.where(active, ac_f, zero)
        aj_f = jnp.where(active, aj_f, zero)
        ap_f = jnp.where(active, ap_f, zero)
        agross_f = jnp.where(active, agross_f, zero)
        anet_f = jnp.where(active, anet_f, zero)
        cs_f = jnp.where(active, cs_f, zero)

        return (
            kc_val,
            ko_val,
            cp_val,
            vcmax_val,
            jmax_val,
            rd_val,
            kp_val,
            lesat_val,
            ceair_val,
            je_val,
            gs_opt,
            ci_f,
            ac_f,
            aj_f,
            ap_f,
            agross_f,
            anet_f,
            cs_f,
        )

    return kernel


@functools.lru_cache(maxsize=None)
def _get_vmapped_photo_kernel_wue_acclim(
    *,
    is_c3: bool,
    gs_type: int,
    colim_type: int,
    c3psn_pft_val,
    vcmaxha,
    vcmaxhd,
    jmaxha,
    jmaxhd,
    rdc,
):
    """Like _get_vmapped_photo_kernel_wue but for acclim_type==1.

    vcmaxse, vcmaxc, jmaxse, jmaxc are NOT closed-over; they are passed as
    runtime broadcast scalars (in_axes=None) so the kernel is compatible
    with jax.checkpoint tracing (no float() required on JAX traced values).

    iota_rt and gsmin_rt are also runtime args (not in cache key) so
    gradients flow through d(GPP)/d(iota) and d(GPP)/d(gsmin).
    """
    kernel = _make_leaf_photo_kernel_wue_acclim(
        is_c3=is_c3,
        gs_type=gs_type,
        colim_type=colim_type,
        c3psn_pft_val=c3psn_pft_val,
        vcmaxha=vcmaxha,
        vcmaxhd=vcmaxhd,
        jmaxha=jmaxha,
        jmaxhd=jmaxhd,
        rdc=rdc,
    )
    # in_axes: 11 per-layer + o2ref_p + pref_p + iota_rt + gsmin_rt + 4 acclim scalars
    return jax.jit(
        jax.vmap(
            kernel,
            in_axes=(
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Public: leaf photosynthesis and stomatal conductance
# ---------------------------------------------------------------------------


def LeafPhotosynthesis(
    num_filter: int,
    filter_patch,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    params: CanopyParams,
    grid: GridInfo,
    *,
    opts: MLOptions,
) -> mlcanopy_type:
    """
    Calculate leaf photosynthesis and stomatal conductance for sunlit
    (``il = isun``) or shaded (``il = isha``) leaves in each canopy
    layer.

    Mirrors Fortran subroutine ``LeafPhotosynthesis`` (public,
    lines 87-220).

    References
    ----------
    Bonan et al. (2014) *Geosci. Model Dev.*, 7, 2193-2222.
    Bonan (2019) *Climate Change and Terrestrial Ecosystem Modeling*,
    Chapters 11-12.
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eq. (14).

    **First loop** — for each leaf layer with ``dpai > 0``:

    1. Temperature acclimation (``acclim_type``): selects ha, hd, se
       constants; for ``acclim_type = 1`` the entropy terms are
       computed from the mean air temperature ``tacclim``.
    2. Deactivation scaling at 25 °C via :func:`_fth25`.
    3. Temperature responses of Kc, Ko, Γ*, Vcmax, Jmax, Rd via
       :func:`_ft` × :func:`_fth`.
    4. C4 override of Vcmax and Rd using Q10 / sigmoid forms.
    5. ``btran = 1`` (soil moisture effect not yet implemented).
    6. Stomatal model parameters selected by ``gs_type``.
    7. Saturation vapour pressure at leaf temperature via
       :func:`MLWaterVaporMod.SatVap`; ceair constrained.
    8. Electron transport rate ``je`` via quadratic.
    9. Photosynthesis + gs solved by :func:`hybrid` (gs_type 0/1)
       or :func:`_StomataOptimization` (gs_type 2).
    10. Error checks: gs > 0, Ball-Berry/Medlyn consistency,
        diffusion consistency.

    **Second loop** — soil moisture adjustment (Fortran lines 192-212):

    - Save ``gspot = gs`` (potential unstressed conductance).
    - Apply water-stress factor ``fpsi`` from leaf water potential
      ``lwp`` via a sigmoidal function (``gspot_type = 1``) or
      ``fpsi = 1`` (``gspot_type = 0``).
    - ``gs = max(gspot * fpsi, gsmin_SPA)``.
    - Recompute photosynthesis at stressed gs via :func:`_CiFuncGs`.
    - Compute ``hs`` and ``vpd`` at the leaf surface.

    Args:
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; all photosynthesis and
            stomatal conductance fields are updated.
        params: Parameters of the patch's PFT (:class:`CanopyParams`).
        grid: Patch index, canopy layer indices and C3/C4 flag (:class:`GridInfo`).

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # ------------------------------------------------------------------
    # First loop: temperature responses + photosynthesis
    # ------------------------------------------------------------------
    p = grid.p  # Fortran loops over the filter; the patch is the unit of computation
    _ncan_p = grid.ncan
    is_c3 = grid.c3  # static: selects the C3 or C4 kernel
    gs_type, gspot_type, colim_type = opts.gs_type, opts.gspot_type, opts.colim_type
    acclim_type = opts.acclim_type
    _c3psn_val = 1.0 if is_c3 else 0.0  # pftcon c3psn flag value

    # --- Temperature acclimation — Fortran lines 135-153 ---
    # (Same for all ic within this patch, compute once.)
    if acclim_type == 0:
        vcmaxha = vcmaxha_noacclim
        jmaxha = jmaxha_noacclim
        vcmaxhd = vcmaxhd_noacclim
        jmaxhd = jmaxhd_noacclim
        vcmaxse = vcmaxse_noacclim
        jmaxse = jmaxse_noacclim
    elif acclim_type == 1:
        vcmaxha = vcmaxha_acclim
        jmaxha = jmaxha_acclim
        vcmaxhd = vcmaxhd_acclim
        jmaxhd = jmaxhd_acclim
        ta_c = jnp.clip(mlcanopy_inst.tacclim_forcing[p] - tfrz, 11.0, 35.0)
        vcmaxse = 668.39 - 1.07 * ta_c
        jmaxse = 659.70 - 0.75 * ta_c
    else:
        endrun(msg=" ERROR: LeafPhotosynthesis: acclim_type not valid")
        vcmaxha = vcmaxhd = vcmaxse = jmaxha = jmaxhd = jmaxse = 0.0

    # High-temperature scaling factors — Fortran lines 155-157
    # rdc: rdse/rdhd are module-level Python floats, so use _fth25_py
    # (math.exp) to get a concrete Python float usable as a cache key even
    # inside jax.checkpoint where jnp.exp would produce an abstract tracer.
    rdc = _fth25_py(rdhd, rdse)  # Python float always (rdhd/rdse are constants)
    # For acclim_type == 0: vcmaxse/jmaxse are Python floats (module-level
    # constants) — use _fth25_py to get Python floats for lru_cache keys.
    # For acclim_type == 1: vcmaxse/jmaxse are JAX traced arrays derived
    # from tacclim_forcing; float() would fail under jax.checkpoint.
    # Keep them as JAX arrays and use _get_vmapped_photo_kernel_acclim
    # which accepts them as runtime broadcast args (in_axes=None).
    if acclim_type == 0:
        vcmaxc = _fth25_py(vcmaxhd, vcmaxse)  # Python float (vcmaxse is float)
        jmaxc = _fth25_py(jmaxhd, jmaxse)  # Python float (jmaxse is float)
    else:
        # acclim_type == 1: vcmaxse/jmaxse are JAX traced — keep as JAX
        vcmaxc = _fth25(vcmaxhd, vcmaxse)  # JAX scalar, passed as runtime arg
        jmaxc = _fth25(jmaxhd, jmaxse)  # JAX scalar, passed as runtime arg

    # --- Stomatal model parameters — same for all ic ---
    if gs_type == 0:
        g0_val = params.g0_medlyn
        g1_val = params.g1_medlyn
    elif gs_type == 1:
        g0_val = params.g0_ball_berry
        g1_val = params.g1_ball_berry
    else:
        g0_val = jnp.asarray(-999.0)
        g1_val = jnp.asarray(-999.0)

    # Per-pft constants for gs_type==2
    if gs_type == 2:
        _iota_pft = params.iota  # differentiable via the implicit function theorem
        _gsmin_pft = params.gs_min

    # --- Pre-extract constant input slices (JAX arrays, no D→H syncs) ---
    _dpai_p = mlcanopy_inst.dpai_profile[p]
    _tleaf_p = mlcanopy_inst.tleaf_leaf[p, :, il]
    _vcmax25_p = mlcanopy_inst.vcmax25_leaf[p, :, il]
    _jmax25_p = mlcanopy_inst.jmax25_leaf[p, :, il]
    _rd25_p = mlcanopy_inst.rd25_leaf[p, :, il]
    _kp25_p = mlcanopy_inst.kp25_leaf[p, :, il]
    _eair_p = mlcanopy_inst.eair_profile[p]
    _apar_p = mlcanopy_inst.apar_leaf[p, :, il]
    _gbc_p = mlcanopy_inst.gbc_leaf[p, :, il]
    _gbv_p = mlcanopy_inst.gbv_leaf[p, :, il]
    _cair_p = mlcanopy_inst.cair_profile[p]
    _o2ref_p = mlcanopy_inst.o2ref_forcing[p]  # JAX scalar (runtime arg of every kernel)
    _pref_p = mlcanopy_inst.pref_forcing[p]

    if gs_type in (0, 1):
        # ---- Differentiable vmap path: no numpy accumulators ---
        # For acclim_type == 0: vcmaxse/jmaxse/vcmaxc/jmaxc are Python
        # floats, so use the standard lru_cache'd kernel factory.
        # For acclim_type == 1: vcmaxse/jmaxse/vcmaxc/jmaxc are JAX traced
        # arrays (computed from tacclim_forcing); use the acclim variant
        # that accepts them as runtime broadcast scalars (in_axes=None)
        # so the kernel is compatible with jax.checkpoint.
        _sl = slice(1, _ncan_p + 1)
        if acclim_type == 0:
            vmapped = _get_vmapped_photo_kernel(
                is_c3=is_c3,
                gs_type=gs_type,
                colim_type=colim_type,
                c3psn_pft_val=_c3psn_val,
                vcmaxha=vcmaxha,
                vcmaxhd=vcmaxhd,
                vcmaxse=vcmaxse,
                vcmaxc=vcmaxc,
                jmaxha=jmaxha,
                jmaxhd=jmaxhd,
                jmaxse=jmaxse,
                jmaxc=jmaxc,
                rdc=rdc,
            )
            layer_out = vmapped(
                _dpai_p[_sl],
                _tleaf_p[_sl],
                _vcmax25_p[_sl],
                _jmax25_p[_sl],
                _rd25_p[_sl],
                _kp25_p[_sl],
                _eair_p[_sl],
                _apar_p[_sl],
                _gbc_p[_sl],
                _gbv_p[_sl],
                _cair_p[_sl],
                g0_val,
                g1_val,  # JAX scalar broadcast — gradients flow here
                _o2ref_p,
            )
        else:
            # acclim_type == 1: vcmaxse/vcmaxc/jmaxse/jmaxc are JAX scalars
            vmapped = _get_vmapped_photo_kernel_acclim(
                is_c3=is_c3,
                gs_type=gs_type,
                colim_type=colim_type,
                c3psn_pft_val=_c3psn_val,
                vcmaxha=vcmaxha,
                vcmaxhd=vcmaxhd,
                jmaxha=jmaxha,
                jmaxhd=jmaxhd,
                rdc=rdc,
            )
            layer_out = vmapped(
                _dpai_p[_sl],
                _tleaf_p[_sl],
                _vcmax25_p[_sl],
                _jmax25_p[_sl],
                _rd25_p[_sl],
                _kp25_p[_sl],
                _eair_p[_sl],
                _apar_p[_sl],
                _gbc_p[_sl],
                _gbv_p[_sl],
                _cair_p[_sl],
                g0_val,
                g1_val,  # JAX scalar broadcast — gradients flow here
                vcmaxse,
                vcmaxc,
                jmaxse,
                jmaxc,
                _o2ref_p,
            )
        # Write vmap JAX output directly (18-tuple) — skip numpy round-trip
        _out_names = [
            "kc_leaf",
            "ko_leaf",
            "cp_leaf",
            "vcmax_leaf",
            "jmax_leaf",
            "rd_leaf",
            "kp_leaf",
            "leaf_esat_leaf",
            "ceair_leaf",
            "je_leaf",
            "gs_leaf",
            "ci_leaf",
            "ac_leaf",
            "aj_leaf",
            "ap_leaf",
            "agross_leaf",
            "anet_leaf",
            "cs_leaf",
        ]
        updates = {}
        for idx, name in enumerate(_out_names):
            arr = getattr(mlcanopy_inst, name)
            updates[name] = arr.at[p, _sl, il].set(layer_out[idx])
        updates["btran_soil"] = mlcanopy_inst.btran_soil.at[p].set(1.0)
        updates["g0_canopy"] = mlcanopy_inst.g0_canopy.at[p].set(g0_val)
        updates["g1_canopy"] = mlcanopy_inst.g1_canopy.at[p].set(g1_val)
        mlcanopy_inst = mlcanopy_inst._replace(**updates)

    elif gs_type == 2:
        # ---- vmap path for WUE stomatal optimization (unified diff/non-diff) ---
        # For acclim_type == 0: use standard kernel (vcmaxse/jmaxse are floats).
        # For acclim_type == 1: use acclim variant that accepts vcmaxse/vcmaxc/
        # jmaxse/jmaxc as runtime broadcast scalars (in_axes=None) to avoid
        # float() on JAX traced values inside jax.checkpoint.
        _sl = slice(1, _ncan_p + 1)
        if acclim_type == 0:
            vmapped = _get_vmapped_photo_kernel_wue(
                is_c3=is_c3,
                gs_type=gs_type,
                colim_type=colim_type,
                c3psn_pft_val=_c3psn_val,
                vcmaxha=vcmaxha,
                vcmaxhd=vcmaxhd,
                vcmaxse=vcmaxse,
                vcmaxc=vcmaxc,
                jmaxha=jmaxha,
                jmaxhd=jmaxhd,
                jmaxse=jmaxse,
                jmaxc=jmaxc,
                rdc=rdc,
            )
            layer_out = vmapped(
                _dpai_p[_sl],
                _tleaf_p[_sl],
                _vcmax25_p[_sl],
                _jmax25_p[_sl],
                _rd25_p[_sl],
                _kp25_p[_sl],
                _eair_p[_sl],
                _apar_p[_sl],
                _gbc_p[_sl],
                _gbv_p[_sl],
                _cair_p[_sl],
                _o2ref_p,
                _pref_p,
                _iota_pft,
                _gsmin_pft,  # JAX scalar broadcast — gradients flow via IFT
            )
        else:
            # acclim_type == 1: vcmaxse/vcmaxc/jmaxse/jmaxc are JAX scalars
            vmapped = _get_vmapped_photo_kernel_wue_acclim(
                is_c3=is_c3,
                gs_type=gs_type,
                colim_type=colim_type,
                c3psn_pft_val=_c3psn_val,
                vcmaxha=vcmaxha,
                vcmaxhd=vcmaxhd,
                jmaxha=jmaxha,
                jmaxhd=jmaxhd,
                rdc=rdc,
            )
            layer_out = vmapped(
                _dpai_p[_sl],
                _tleaf_p[_sl],
                _vcmax25_p[_sl],
                _jmax25_p[_sl],
                _rd25_p[_sl],
                _kp25_p[_sl],
                _eair_p[_sl],
                _apar_p[_sl],
                _gbc_p[_sl],
                _gbv_p[_sl],
                _cair_p[_sl],
                _o2ref_p,
                _pref_p,
                _iota_pft,
                _gsmin_pft,  # JAX scalar broadcast — gradients flow via IFT
                vcmaxse,
                vcmaxc,
                jmaxse,
                jmaxc,
            )
        _out_names = [
            "kc_leaf",
            "ko_leaf",
            "cp_leaf",
            "vcmax_leaf",
            "jmax_leaf",
            "rd_leaf",
            "kp_leaf",
            "leaf_esat_leaf",
            "ceair_leaf",
            "je_leaf",
            "gs_leaf",
            "ci_leaf",
            "ac_leaf",
            "aj_leaf",
            "ap_leaf",
            "agross_leaf",
            "anet_leaf",
            "cs_leaf",
        ]
        updates = {}
        for idx, name in enumerate(_out_names):
            arr = getattr(mlcanopy_inst, name)
            updates[name] = arr.at[p, _sl, il].set(layer_out[idx])
        updates["btran_soil"] = mlcanopy_inst.btran_soil.at[p].set(1.0)
        mlcanopy_inst = mlcanopy_inst._replace(**updates)

    else:
        endrun(msg=" ERROR: LeafPhotosynthesis: gs_type not valid")

    # ------------------------------------------------------------------
    # Second loop: soil moisture adjustment — Fortran lines 262-212
    # ------------------------------------------------------------------
    _gsmin_pft2 = params.gs_min

    # Pre-extract slices for second loop (JAX arrays, no D→H syncs)
    _gs_p2 = mlcanopy_inst.gs_leaf[p, :, il]
    _dpai_p2 = mlcanopy_inst.dpai_profile[p]
    _gbv_p2 = mlcanopy_inst.gbv_leaf[p, :, il]
    _eair_p2 = mlcanopy_inst.eair_profile[p]
    _lesat_p2 = mlcanopy_inst.leaf_esat_leaf[p, :, il]
    _gbc_p2 = mlcanopy_inst.gbc_leaf[p, :, il]
    _vcmax_p2 = mlcanopy_inst.vcmax_leaf[p, :, il]
    _je_p2 = mlcanopy_inst.je_leaf[p, :, il]
    _kp_p2 = mlcanopy_inst.kp_leaf[p, :, il]
    _rd_p2 = mlcanopy_inst.rd_leaf[p, :, il]
    _kc_p2 = mlcanopy_inst.kc_leaf[p, :, il]
    _ko_p2 = mlcanopy_inst.ko_leaf[p, :, il]
    _cp_p2 = mlcanopy_inst.cp_leaf[p, :, il]
    _o2ref_p2 = mlcanopy_inst.o2ref_forcing[p]
    _cair_p2 = mlcanopy_inst.cair_profile[p]
    _apar_p2 = mlcanopy_inst.apar_leaf[p, :, il]
    _ceair_p2 = mlcanopy_inst.ceair_leaf[p, :, il]

    if gspot_type == 1:
        _lwp_p2 = mlcanopy_inst.lwp_leaf[p, :, il]
        _psi50_pft2 = params.psi50_gs
        _shape_pft2 = params.shape_gs

    # --- Second loop: vectorized over layers (replaces per-ic Python loop) ---
    # All operations are element-wise; gspot_type and is_c3 are static Python
    # ints/bools so their if/else branches are resolved at trace time.
    # Replaces 46 × ~10 scalar XLA ops with ~10 slice XLA ops — reduces
    # trace depth and improves kernel fusion inside lax.scan.
    _sl2 = slice(1, _ncan_p + 1)
    _gs_sl = _gs_p2[_sl2]
    _dpai_sl = _dpai_p2[_sl2]
    _gbc_sl = _gbc_p2[_sl2]
    _gbv_sl = _gbv_p2[_sl2]
    _eair_sl = _eair_p2[_sl2]
    _lesat_sl = _lesat_p2[_sl2]
    _vcmax_sl = _vcmax_p2[_sl2]
    _je_sl = _je_p2[_sl2]
    _kp_sl = _kp_p2[_sl2]
    _rd_sl = _rd_p2[_sl2]
    _kc_sl = _kc_p2[_sl2]
    _ko_sl = _ko_p2[_sl2]
    _cp_sl = _cp_p2[_sl2]
    _cair_sl = _cair_p2[_sl2]
    _apar_sl = _apar_p2[_sl2]
    _ceair_sl = _ceair_p2[_sl2]
    active_sl = _dpai_sl > 0.0

    # Water-stress factor (gspot_type is a static Python int)
    if gspot_type == 0:
        fpsi = jnp.ones_like(_gs_sl)
    elif gspot_type == 1:
        _lwp_sl = _lwp_p2[_sl2]
        fpsi = 1.0 / (1.0 + (_lwp_sl / _psi50_pft2) ** _shape_pft2)
    else:
        fpsi = jnp.ones_like(_gs_sl)

    gs_new_sl = jnp.maximum(_gs_sl * fpsi, _gsmin_pft2)

    # Safe gleaf: prevent divide-by-zero for inactive layers.
    # For inactive layers _gbc_sl == 0 → 1/_gbc_sl = inf → gleaf_sl = 0.
    # But ci_sl = cair - anet_sl / gleaf_sl would then be anet_sl/0.
    # d(ci_sl)/d(anet_sl) = 1/gleaf_sl = 1/0 = inf, and 0*inf = NaN in
    # backward for alpha_pbot.  Guard _gbc_sl too so gleaf_sl > 0.
    _gs_safe = jnp.maximum(gs_new_sl, 1.0e-30)
    _gbc_safe = jnp.maximum(_gbc_sl, 1.0e-30)
    gleaf_sl = 1.0 / (1.0 / _gbc_safe + dh2o_to_dco2 / _gs_safe)

    # Photosynthesis recompute (is_c3 is a static Python bool)
    if is_c3:
        # Guard _ko_sl against zero for inactive layers (dpai==0) where
        # kc/ko were zeroed out in the first loop.  For active layers
        # _ko_sl > 0 always, so jnp.maximum does not change the value.
        # Without the guard: _o2ref_p2 / 0 → inf, and _kc_sl * inf = 0*inf
        # = NaN in forward; d/d(_o2ref_p2) = _kc_sl/_ko_sl = 0/0 = NaN in
        # backward — causing NaN JAX gradient for alpha_pbot.
        _ko_safe2 = jnp.maximum(_ko_sl, jnp.asarray(1.0e-30))
        b0_sl = _kc_sl * (1.0 + _o2ref_p2 / _ko_safe2)
        bq_sl = -(_cair_sl + b0_sl) - (_vcmax_sl - _rd_sl) / gleaf_sl
        cq_sl = _vcmax_sl * (_cair_sl - _cp_sl) - _rd_sl * (_cair_sl + b0_sl)
        r1_sl, r2_sl = quadratic(1.0 / gleaf_sl, bq_sl, cq_sl)
        ac_sl = jnp.minimum(r1_sl, r2_sl) + _rd_sl
        b0j_sl = 2.0 * _cp_sl
        bqj_sl = -(_cair_sl + b0j_sl) - (_je_sl / 4.0 - _rd_sl) / gleaf_sl
        cqj_sl = (_je_sl / 4.0) * (_cair_sl - _cp_sl) - _rd_sl * (_cair_sl + b0j_sl)
        r1j_sl, r2j_sl = quadratic(1.0 / gleaf_sl, bqj_sl, cqj_sl)
        aj_sl = jnp.minimum(r1j_sl, r2j_sl) + _rd_sl
        ap_sl = jnp.zeros_like(ac_sl)
    else:
        ac_sl = _vcmax_sl
        aj_sl = qe_c4 * _apar_sl
        ap_sl = _kp_sl * (_cair_sl * gleaf_sl + _rd_sl) / (gleaf_sl + _kp_sl)

    agross_sl = _RealizedRate(colim_type, _c3psn_val, ac_sl, aj_sl, ap_sl)
    anet_sl = agross_sl - _rd_sl
    # Use _gbc_safe (clamped from 0) so d(cs_sl)/d(anet_sl) = 1/_gbc_safe
    # is finite for inactive layers; otherwise -1/0 → -inf and
    # d(jnp.maximum)/d(anet_sl) = 0 * (-inf) = NaN for alpha_pbot.
    cs_sl = jnp.maximum(_cair_sl - anet_sl / _gbc_safe, 1.0)
    ci_sl = _cair_sl - anet_sl / gleaf_sl

    # Guard hs_sl denominator against zero for inactive layers (dpai==0)
    # where _lesat_sl == 0 and _gbv_sl == 0: denominator = gs_new_sl * 0 = 0,
    # giving 0/0 = NaN.  d(hs_sl)/d(_eair_sl) = _gbv_sl / denom = 0/0 = NaN
    # in backward pass.  Since _eair_sl is traced by alpha_pbot, this NaN
    # multiplied by the jnp.where mask (0 for inactive) gives 0*NaN = NaN
    # gradient for alpha_pbot.  Fix: clamp denominator to eps for inactive layers.
    _hs_denom = jnp.maximum((_gbv_sl + gs_new_sl) * _lesat_sl, jnp.asarray(1.0e-30))
    hs_sl = (_gbv_sl * _eair_sl + gs_new_sl * _lesat_sl) / _hs_denom
    vpd_sl = jnp.maximum(_lesat_sl - hs_sl * _lesat_sl, 0.1)

    # Mask inactive layers and write all outputs back in one _replace()
    _z = jnp.zeros_like(active_sl, dtype=jnp.float64)
    gspot_arr = mlcanopy_inst.gspot_leaf.at[p, _sl2, il].set(_gs_sl)
    gs_arr = mlcanopy_inst.gs_leaf.at[p, _sl2, il].set(jnp.where(active_sl, gs_new_sl, _gs_sl))
    ci_arr = mlcanopy_inst.ci_leaf.at[p, _sl2, il].set(
        jnp.where(active_sl, ci_sl, mlcanopy_inst.ci_leaf[p, _sl2, il])
    )
    ac_arr = mlcanopy_inst.ac_leaf.at[p, _sl2, il].set(
        jnp.where(active_sl, ac_sl, mlcanopy_inst.ac_leaf[p, _sl2, il])
    )
    aj_arr = mlcanopy_inst.aj_leaf.at[p, _sl2, il].set(
        jnp.where(active_sl, aj_sl, mlcanopy_inst.aj_leaf[p, _sl2, il])
    )
    ap_arr = mlcanopy_inst.ap_leaf.at[p, _sl2, il].set(
        jnp.where(active_sl, ap_sl, mlcanopy_inst.ap_leaf[p, _sl2, il])
    )
    agross_arr = mlcanopy_inst.agross_leaf.at[p, _sl2, il].set(
        jnp.where(active_sl, agross_sl, mlcanopy_inst.agross_leaf[p, _sl2, il])
    )
    anet_arr = mlcanopy_inst.anet_leaf.at[p, _sl2, il].set(
        jnp.where(active_sl, anet_sl, mlcanopy_inst.anet_leaf[p, _sl2, il])
    )
    cs_arr = mlcanopy_inst.cs_leaf.at[p, _sl2, il].set(
        jnp.where(active_sl, cs_sl, mlcanopy_inst.cs_leaf[p, _sl2, il])
    )
    hs_arr = mlcanopy_inst.hs_leaf.at[p, _sl2, il].set(jnp.where(active_sl, hs_sl, _z))
    vpd_arr = mlcanopy_inst.vpd_leaf.at[p, _sl2, il].set(jnp.where(active_sl, vpd_sl, _z))

    mlcanopy_inst = mlcanopy_inst._replace(
        gspot_leaf=gspot_arr,
        gs_leaf=gs_arr,
        ci_leaf=ci_arr,
        ac_leaf=ac_arr,
        aj_leaf=aj_arr,
        ap_leaf=ap_arr,
        agross_leaf=agross_arr,
        anet_leaf=anet_arr,
        cs_leaf=cs_arr,
        hs_leaf=hs_arr,
        vpd_leaf=vpd_arr,
    )

    return mlcanopy_inst
