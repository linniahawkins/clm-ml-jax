"""
JAX translation of MLSolarRadiationMod Fortran module.

Solar radiation transfer through the multilayer canopy.
Provides one public driver (:func:`SolarRadiation`) and two private
solvers (:func:`_Norman`, :func:`_TwoStream`).

Original Fortran module: MLSolarRadiationMod
Fortran lines 1-490

Differentiability notes
-----------------------
* All ``import numpy as np`` and ``import math`` removed — JAX used throughout.
* All ``np.asarray()`` calls removed — JAX arrays used directly.
* All ``float()`` wrappers on JAX scalars removed.
* All ``math.*`` calls (cos, exp, sqrt, log) replaced by ``jnp.*``.
* Numpy intermediate working arrays (``_avmu``, ``_betad``, etc.) replaced
  by ``jnp.zeros(...)``; element assignments use ``.at[].set()``.
* Tridiagonal coefficient lists hold ``jnp.zeros(())`` JAX scalars so the
  Thomas algorithm is fully differentiable at trace time.
* ``int()`` wrappers on ``ncan[p]``, ``ntop[p]``, ``nbot[p]`` are kept —
  these must be concrete Python ints for loop bounds and static slices.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_varcon import rpi as pi  # noqa: F401
from clm_src_main.clm_varctl import iulog  # noqa: F401
from clm_src_main.clm_varpar import ivis, numrad  # noqa: F401
from clm_src_main.decompMod import bounds_type  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from multilayer_canopy.MLCanopyParamsMod import CanopyParams
from multilayer_canopy.MLclm_varcon import J_to_umol, chil_max, chil_min, kb_max  # noqa: F401
from multilayer_canopy.MLclm_varctl import (  # noqa: F401
    GridInfo,  # noqa: F401
    leaf_optics_type,
    light_type,
)
from multilayer_canopy.MLclm_varpar import isha, isun, nlevmlcan  # noqa: F401
from multilayer_canopy.MLMathToolsMod import is_concrete, tridiag  # noqa: F401

# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def SolarRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    params: CanopyParams,
    grid: GridInfo,
) -> mlcanopy_type:
    """
    Solar radiation transfer through the multilayer canopy.

    Mirrors Fortran subroutine ``SolarRadiation`` (lines 28-195).

    References
    ----------
    Bonan (2019) *Climate Change and Terrestrial Ecosystem Modeling*,
    Chapter 14.
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eqs. (1)-(5).

    Execution sequence
    ------------------
    1. For each patch, zero all layer optical variables.
    2. Loop over leaf layers (``ntop`` down to ``nbot``): compute
       leaf/stem reflectance (``rho``), transmittance (``tau``),
       scattering (``omega``), leaf angle distribution (``chil``,
       ``phi1``, ``phi2``, ``gdir``), direct beam extinction
       coefficient ``kb``, clumping factor, single-layer direct
       transmittance ``tb``, diffuse transmittance ``td`` (9-angle
       numerical integration), cumulative direct beam transmittance
       ``tbi``, sunlit fraction ``fracsun``, and two-stream parameters
       ``avmu``, ``betad``, ``betab``.
    3. Compute ``tbi(p,0)`` (transmittance onto the ground).
    4. Dispatch to :func:`_Norman` (``light_type == 1``) or
       :func:`_TwoStream` (``light_type == 2``).
    5. Convert PAR absorbed by leaves from W/m2 to umol/m2/s via
       ``J_to_umol``.

    Optical properties follow ``leaf_optics_type``:

    - **0** (constant with height): weighted average of leaf and stem
      properties by ``wl = dlai/dpai``, ``ws = dsai/dpai``.
    - **1** (vary with height): fatal — vertical profiles not
      implemented.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Multilayer canopy container; optical diagnostics,
            flux profiles, and absorbed radiation fields are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # ------------------------------------------------------------------
    # Unpack inputs (Fortran associate block, lines 65-95)
    # ------------------------------------------------------------------

    solar_zen = mlcanopy_inst.solar_zen_forcing
    dlai = mlcanopy_inst.dlai_profile
    dsai = mlcanopy_inst.dsai_profile
    dpai = mlcanopy_inst.dpai_profile

    fracsun = mlcanopy_inst.fracsun_profile
    kb = mlcanopy_inst.kb_profile
    tb = mlcanopy_inst.tb_profile
    td = mlcanopy_inst.td_profile
    tbi = mlcanopy_inst.tbi_profile
    apar = mlcanopy_inst.apar_leaf

    # ------------------------------------------------------------------
    # Working JAX arrays for optical properties
    # ------------------------------------------------------------------
    n_idx = bounds.endp + 1
    n_lev = nlevmlcan + 1
    n_rad = numrad + 1
    _avmu = jnp.zeros((n_idx, n_lev))
    _betad = jnp.zeros((n_idx, n_lev, n_rad))
    _betab = jnp.zeros((n_idx, n_lev, n_rad))
    _cf_ic = jnp.zeros((n_idx, n_lev))
    _rho_arr = jnp.zeros((n_idx, n_lev, n_rad))
    _tau_arr = jnp.zeros((n_idx, n_lev, n_rad))
    _om_arr = jnp.zeros((n_idx, n_lev, n_rad))

    # ------------------------------------------------------------------
    # Calculate canopy layer optical properties — Fortran lines 100-182
    # ------------------------------------------------------------------
    # Vectorized over the foliage layers nbot..ntop (top-down recurrence for tbi only).
    p = grid.p
    _ncan, _ntop, _nbot = grid.ncan, grid.ntop, grid.nbot
    ls = slice(_nbot, _ntop + 1)
    rb = slice(1, numrad + 1)
    zen = solar_zen[p]
    cos_zen = jnp.cos(zen)

    if leaf_optics_type == 0:
        _xl_pft = params.leaf_angle_index
        _cf_pft = params.clumping_index
        _rhol, _taul = params.leaf_reflectance[rb], params.leaf_transmittance[rb]
        _rhos, _taus = params.stem_reflectance[rb], params.stem_transmittance[rb]
    else:
        endrun(msg=" ERROR: SolarRadiation: need to specify vertical profile for rho & tau")
        _xl_pft = 0.01
        _cf_pft = 1.0
        _rhol = _taul = _rhos = _taus = jnp.full(numrad, 1.0e-6)

    dpai_v = dpai[p, ls]
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    dpai_safe = jnp.maximum(dpai_v, 1.0e-30)
    wl = (dlai[p, ls] / dpai_safe)[:, None]
    ws = (dsai[p, ls] / dpai_safe)[:, None]

    # Reflectance, transmittance, scattering — Fortran lines 121-132 (layer, band)
    r_v = jnp.maximum(_rhol * wl + _rhos * ws, 1.0e-6)
    t_v = jnp.maximum(_taul * wl + _taus * ws, 1.0e-6)
    om_v = r_v + t_v
    _rho_arr = _rho_arr.at[p, ls, rb].set(r_v)
    _tau_arr = _tau_arr.at[p, ls, rb].set(t_v)
    _om_arr = _om_arr.at[p, ls, rb].set(om_v)

    # Leaf angle distribution — Fortran lines 134-141 (one PFT: same in every layer)
    chil = jnp.clip(_xl_pft, chil_min, chil_max)
    chil = jnp.where(jnp.abs(chil) <= 0.01, 0.01, chil)

    # Ross-Goudriaan phi1, phi2, gdir — Fortran lines 143-148
    p1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    p2 = 0.877 * (1.0 - 2.0 * p1)
    gd = p1 + p2 * cos_zen

    # Direct beam extinction coefficient — Fortran lines 150-152
    # jnp.maximum avoids NaN grad: cos_zen=0 at horizon gives gd/0=inf
    # in the True branch of jnp.minimum, producing 0*inf=NaN in backward.
    cos_zen_safe = jnp.maximum(cos_zen, 1.0e-10)
    kb_s = jnp.minimum(gd / cos_zen_safe, kb_max)
    kb_v = jnp.broadcast_to(kb_s, dpai_v.shape)

    # Clumping factor — Fortran lines 154-159
    cf = _cf_pft
    _cf_ic = _cf_ic.at[p, ls].set(cf)

    # Direct beam single-layer transmittance — Fortran line 161
    tb_v = jnp.exp(-kb_v * dpai_v * cf)

    # Diffuse transmittance (9-angle integration) — Fortran lines 163-170
    td_v = jnp.zeros_like(dpai_v)
    for j in range(1, 10):
        angle = (5.0 + (j - 1) * 10.0) * pi / 180.0
        gdirj = p1 + p2 * jnp.cos(jnp.asarray(angle))
        td_v = td_v + (
            jnp.exp(-gdirj / jnp.cos(jnp.asarray(angle)) * dpai_v * cf)
            * jnp.sin(jnp.asarray(angle))
            * jnp.cos(jnp.asarray(angle))
        )
    td_v = td_v * 2.0 * (10.0 * pi / 180.0)

    # Cumulative direct beam transmittance tbi — Fortran lines 172-177 (top down;
    # tbi(ic) = tbi(ic+1) * exp(-kb dpai cf) of the layer above, which is tb_v)
    nf = _ntop - _nbot + 1
    _tbi_list = [jnp.ones(())]
    for k in range(nf - 2, -1, -1):  # layer index k <-> ic = nbot + k
        _tbi_list.append(_tbi_list[-1] * tb_v[k + 1])
    tbi_v = jnp.stack(_tbi_list[::-1])

    # Sunlit fraction — Fortran lines 179-188
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    # When kb*dpai=0, the exp term also =0 so fracsun=0 regardless of denom.
    kb_dp_safe = jnp.maximum(kb_v * dpai_v, 1.0e-30)
    fracsun_v = tbi_v / kb_dp_safe * (1.0 - jnp.exp(-kb_v * cf * dpai_v))
    if is_concrete(fracsun_v) and jnp.any(fracsun_v <= 0.0):
        endrun(msg=" ERROR: SolarRadiation: fracsun is too small")
    if is_concrete(fracsun_v) and jnp.any((1.0 - fracsun_v) <= 0.0):
        endrun(msg=" ERROR: SolarRadiation: fracsha is too small")

    # Two-stream avmu — Fortran lines 190-194
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    p1_safe = jnp.maximum(p1, 1.0e-10)
    p2_safe = jnp.maximum(p2, 1.0e-10)
    avmu_s = (1.0 - p1_safe / p2_safe * jnp.log((p1_safe + p2_safe) / p1_safe)) / p2_safe
    _avmu = _avmu.at[p, ls].set(avmu_s)

    # betad, betab — Fortran lines 196-209 (layer, band)
    _betad = _betad.at[p, ls, rb].set(
        0.5 / om_v * (r_v + t_v + (r_v - t_v) * ((1.0 + chil) / 2.0) ** 2)
    )
    tmp0 = gd + p2 * cos_zen
    tmp1 = p1 * cos_zen
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    # tmp0 = gd + p2*cos_zen, tmp1 = p1*cos_zen — both non-negative physically
    tmp0_safe = jnp.maximum(tmp0, 1.0e-10)
    tmp1_safe = jnp.maximum(tmp1, 1.0e-10)
    tmp2 = 1.0 - tmp1_safe / tmp0_safe * jnp.log((tmp1_safe + tmp0_safe) / tmp1_safe)
    asu = 0.5 * om_v * gd / tmp0_safe * tmp2
    _betab = _betab.at[p, ls, rb].set((1.0 + avmu_s * kb_s) / (om_v * avmu_s * kb_s) * asu)

    # Write-back: foliage layers; other canopy layers 0; tbi(0) is onto the ground
    # (Fortran lines 211-213)
    def _profile(v):
        return jnp.zeros(_ncan + 2).at[ls].set(v)

    _tbi_p = _profile(tbi_v).at[0].set(tbi_v[0] * tb_v[0])
    _sl = slice(1, _ncan + 1)
    _sl0 = slice(0, _ncan + 1)
    kb = kb.at[p, _sl].set(_profile(kb_v)[_sl])
    fracsun = fracsun.at[p, _sl].set(_profile(fracsun_v)[_sl])
    tb = tb.at[p, _sl].set(_profile(tb_v)[_sl])
    td = td.at[p, _sl].set(_profile(td_v)[_sl])
    tbi = tbi.at[p, _sl0].set(_tbi_p[_sl0])

    # ------------------------------------------------------------------
    # Commit optical properties before radiative transfer
    # ------------------------------------------------------------------
    mlcanopy_inst = mlcanopy_inst._replace(
        fracsun_profile=fracsun,
        kb_profile=kb,
        tb_profile=tb,
        td_profile=td,
        tbi_profile=tbi,
    )

    # ------------------------------------------------------------------
    # Radiative transfer solver — Fortran lines 216-222
    # ------------------------------------------------------------------
    if light_type == 1:
        mlcanopy_inst = _Norman(
            bounds,
            num_filter,
            filter_patch,
            _rho_arr,
            _tau_arr,
            _om_arr,
            mlcanopy_inst,
            grid=grid,
        )
    elif light_type == 2:
        mlcanopy_inst = _TwoStream(
            bounds,
            num_filter,
            filter_patch,
            _om_arr,
            _avmu,
            _betad,
            _betab,
            _cf_ic,
            mlcanopy_inst,
            grid=grid,
        )
    else:
        endrun(msg=" ERROR: SolarRadiation: light_type not valid")

    # ------------------------------------------------------------------
    # APAR: W/m2 → umol/m2/s — Fortran lines 224-229 (vectorized)
    # ------------------------------------------------------------------
    swleaf = mlcanopy_inst.swleaf_leaf
    apar = mlcanopy_inst.apar_leaf
    for fp in range(1, num_filter + 1):
        p = grid.p
        _ncan = grid.ncan
        _sl = slice(1, _ncan + 1)
        apar = apar.at[p, _sl, :].set(swleaf[p, _sl, :, ivis] * J_to_umol)

    mlcanopy_inst = mlcanopy_inst._replace(apar_leaf=apar)
    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Private: Norman (1979) radiative transfer
# ---------------------------------------------------------------------------


def _Norman(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    rho: Array,
    tau: Array,
    omega: Array,
    mlcanopy_inst: mlcanopy_type,
    grid: GridInfo,
) -> mlcanopy_type:
    """
    Norman (1979) tridiagonal radiative transfer through the canopy.

    Mirrors Fortran subroutine ``Norman`` (private, lines 232-370).

    For each waveband ``ib`` and each patch ``p``, sets up a tridiagonal
    system of ``2*(nbot-to-ntop layers + soil)`` equations for the
    upward and downward diffuse fluxes at each layer interface and
    solves it via :func:`MLMathToolsMod.tridiag`. The ordering of
    equations follows the Fortran exactly: soil upward (m=1), soil
    downward (m=2), then for ``ic = nbot, ..., ntop-1`` upward (odd)
    and downward (even), finally top-layer upward and downward.

    After solving, absorbed direct-beam and diffuse fluxes per layer
    are partitioned into sunlit (``swleaf[..,isun,..]``) and shaded
    (``swleaf[..,isha,..]``) components per unit leaf area. Energy
    conservation is checked to within 1e-3 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch filter.
        rho: Leaf/stem reflectance ``(p, ic, ib)``.
        tau: Leaf/stem transmittance ``(p, ic, ib)``.
        omega: Scattering coefficient ``(p, ic, ib)``.
        mlcanopy_inst: Canopy container; radiation flux fields updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    neq = (nlevmlcan + 1) * 2  # Fortran: parameter neq = (nlevmlcan+1)*2

    swskyb = mlcanopy_inst.swskyb_forcing
    swskyd = mlcanopy_inst.swskyd_forcing
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    dpai = mlcanopy_inst.dpai_profile
    fracsun = mlcanopy_inst.fracsun_profile
    tb = mlcanopy_inst.tb_profile
    td = mlcanopy_inst.td_profile
    tbi = mlcanopy_inst.tbi_profile

    swveg = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    albcan = mlcanopy_inst.albcan_canopy
    swsoi = mlcanopy_inst.swsoi_soil
    swleaf = mlcanopy_inst.swleaf_leaf
    swupw = mlcanopy_inst.swupw_profile
    swdwn = mlcanopy_inst.swdwn_profile
    swbeam = mlcanopy_inst.swbeam_profile

    for ib in range(1, numrad + 1):  # Fortran: do ib = 1, numrad
        for fp in range(1, num_filter + 1):
            p = grid.p
            _ncan = grid.ncan
            _ntop = grid.ntop
            _nbot = grid.nbot

            # Scalar inputs — JAX scalars directly
            _swskyb_ib = swskyb[p, ib]
            _swskyd_ib = swskyd[p, ib]
            _albsoib_ib = albsoib[p, ib]
            _albsoid_ib = albsoid[p, ib]

            # Zero swleaf for all layers (bulk — 1 JAX op)
            swleaf = swleaf.at[p, 1 : _ncan + 1, :, ib].set(0.0)

            # ----------------------------------------------------------------
            # Build tridiagonal system — Fortran lines 284-340
            # Lists hold JAX traced scalars; Thomas algorithm is differentiable
            # ----------------------------------------------------------------
            atri = [jnp.zeros(())] * (neq + 1)
            btri = [jnp.zeros(())] * (neq + 1)
            ctri = [jnp.zeros(())] * (neq + 1)
            dtri = [jnp.zeros(())] * (neq + 1)

            m = 0

            # Soil: upward flux — Fortran lines 287-290
            m += 1
            atri[m] = jnp.zeros(())
            btri[m] = jnp.ones(())
            ctri[m] = -_albsoid_ib
            dtri[m] = _swskyb_ib * tbi[p, 0] * _albsoib_ib

            # Soil: downward flux — Fortran lines 292-301
            td_nb = td[p, _nbot]
            rho_nb = rho[p, _nbot, ib]
            tau_nb = tau[p, _nbot, ib]
            tb_nb = tb[p, _nbot]
            tbi_nb = tbi[p, _nbot]
            refld = (1.0 - td_nb) * rho_nb
            trand = (1.0 - td_nb) * tau_nb + td_nb
            aic = refld - trand * trand / refld
            bic = trand / refld
            m += 1
            atri[m] = -aic
            btri[m] = jnp.ones(())
            ctri[m] = -bic
            dtri[m] = _swskyb_ib * tbi_nb * (1.0 - tb_nb) * (tau_nb - rho_nb * bic)

            # Leaf layers except top — Fortran lines 303-326
            for ic in range(_nbot, _ntop):  # Fortran: do ic = nbot, ntop-1

                # Upward flux
                td_ic = td[p, ic]
                rho_ic = rho[p, ic, ib]
                tau_ic = tau[p, ic, ib]
                refld = (1.0 - td_ic) * rho_ic
                trand = (1.0 - td_ic) * tau_ic + td_ic
                fic = refld - trand * trand / refld
                eic = trand / refld
                m += 1
                atri[m] = -eic
                btri[m] = jnp.ones(())
                ctri[m] = -fic
                dtri[m] = _swskyb_ib * tbi[p, ic] * (1.0 - tb[p, ic]) * (rho_ic - tau_ic * eic)

                # Downward flux
                ic1 = ic + 1
                td_ic1 = td[p, ic1]
                rho_ic1 = rho[p, ic1, ib]
                tau_ic1 = tau[p, ic1, ib]
                refld = (1.0 - td_ic1) * rho_ic1
                trand = (1.0 - td_ic1) * tau_ic1 + td_ic1
                aic = refld - trand * trand / refld
                bic = trand / refld
                m += 1
                atri[m] = -aic
                btri[m] = jnp.ones(())
                ctri[m] = -bic
                dtri[m] = _swskyb_ib * tbi[p, ic1] * (1.0 - tb[p, ic1]) * (tau_ic1 - rho_ic1 * bic)

            # Top layer: upward flux — Fortran lines 328-337
            ic = _ntop
            td_ic = td[p, ic]
            rho_ic = rho[p, ic, ib]
            tau_ic = tau[p, ic, ib]
            refld = (1.0 - td_ic) * rho_ic
            trand = (1.0 - td_ic) * tau_ic + td_ic
            fic = refld - trand * trand / refld
            eic = trand / refld
            m += 1
            atri[m] = -eic
            btri[m] = jnp.ones(())
            ctri[m] = -fic
            dtri[m] = _swskyb_ib * tbi[p, ic] * (1.0 - tb[p, ic]) * (rho_ic - tau_ic * eic)

            # Top layer: downward flux — Fortran lines 339-343
            m += 1
            atri[m] = jnp.zeros(())
            btri[m] = jnp.ones(())
            ctri[m] = jnp.zeros(())
            dtri[m] = _swskyd_ib

            # Solve — Fortran line 345
            utri = tridiag(atri, btri, ctri, dtri, m)

            # ----------------------------------------------------------------
            # Unpack solution into JAX arrays
            # ----------------------------------------------------------------
            swupw_new = jnp.zeros(_ncan + 2)
            swdwn_new = jnp.zeros(_ncan + 2)
            m_sol = 0
            m_sol += 1
            swupw_new = swupw_new.at[0].set(utri[m_sol])
            m_sol += 1
            swdwn_new = swdwn_new.at[0].set(utri[m_sol])
            for ic in range(_nbot, _ntop + 1):
                m_sol += 1
                swupw_new = swupw_new.at[ic].set(utri[m_sol])
                m_sol += 1
                swdwn_new = swdwn_new.at[ic].set(utri[m_sol])

            # ----------------------------------------------------------------
            # Compute fluxes — Fortran lines 365-430
            # ----------------------------------------------------------------

            # Ground absorption — Fortran lines 368-372
            _swbeam_0 = tbi[p, 0] * _swskyb_ib
            _swsoi_ib = _swbeam_0 * (1.0 - _albsoib_ib) + swdwn_new[0] * (1.0 - _albsoid_ib)

            # Per-layer accumulators (JAX arrays)
            swbeam_new = jnp.zeros(_ncan + 2)
            swleaf_sun_new = jnp.zeros(_ncan + 2)
            swleaf_sha_new = jnp.zeros(_ncan + 2)
            swbeam_new = swbeam_new.at[0].set(_swbeam_0)
            _swveg_acc = jnp.zeros(())
            _swvegsun_acc = jnp.zeros(())
            _swvegsha_acc = jnp.zeros(())

            for ic in range(_nbot, _ntop + 1):  # Fortran: do ic = nbot, ntop
                _swbeam_ic = tbi[p, ic] * _swskyb_ib
                _om_ic = omega[p, ic, ib]
                _swabsb_ic = _swbeam_ic * (1.0 - tb[p, ic]) * (1.0 - _om_ic)

                icm1 = 0 if ic == _nbot else ic - 1
                _swabsd_ic = (swdwn_new[ic] + swupw_new[icm1]) * (1.0 - td[p, ic]) * (1.0 - _om_ic)

                _fs = fracsun[p, ic]
                _swsha = _swabsd_ic * (1.0 - _fs)
                _swsun = _swabsd_ic * _fs + _swabsb_ic

                _dpai_ic = dpai[p, ic]
                # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
                fs_safe = jnp.maximum(_fs, 1.0e-30)
                fsha_safe = jnp.maximum(1.0 - _fs, 1.0e-30)
                dpai_safe = jnp.maximum(_dpai_ic, 1.0e-30)
                swleaf_sun_new = swleaf_sun_new.at[ic].set(_swsun / (fs_safe * dpai_safe))
                swleaf_sha_new = swleaf_sha_new.at[ic].set(_swsha / (fsha_safe * dpai_safe))
                swbeam_new = swbeam_new.at[ic].set(_swbeam_ic)
                _swveg_acc = _swveg_acc + _swabsb_ic + _swabsd_ic
                _swvegsun_acc = _swvegsun_acc + _swsun
                _swvegsha_acc = _swvegsha_acc + _swsha

            # Albedo — Fortran lines 410-414
            _suminc = _swskyb_ib + _swskyd_ib
            # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
            _suminc_safe = jnp.maximum(_suminc, 1.0e-30)
            _albcan_ib = jnp.where(_suminc > 0.0, swupw_new[_ntop] / _suminc_safe, 0.0)

            # Conservation checks — Fortran lines 416-426
            _sumref = _albcan_ib * _suminc
            _sumabs = _suminc - _sumref
            _err = _sumabs - (_swveg_acc + _swsoi_ib)
            if is_concrete(_err) and jnp.abs(_err) > 1.0e-3:
                endrun(msg="ERROR: Norman: total solar conservation error")
            _err2 = (_swvegsun_acc + _swvegsha_acc) - _swveg_acc
            if is_concrete(_err2) and jnp.abs(_err2) > 1.0e-3:
                endrun(msg="ERROR: Norman: sunlit/shade solar conservation error")

            # Bulk JAX write-back
            _sl = slice(0, _ncan + 1)
            _sl_layers = slice(_nbot, _ntop + 1)
            swupw = swupw.at[p, _sl, ib].set(swupw_new[: _ncan + 1])
            swdwn = swdwn.at[p, _sl, ib].set(swdwn_new[: _ncan + 1])
            swbeam = swbeam.at[p, _sl, ib].set(swbeam_new[: _ncan + 1])
            swleaf = swleaf.at[p, _sl_layers, isun, ib].set(swleaf_sun_new[_nbot : _ntop + 1])
            swleaf = swleaf.at[p, _sl_layers, isha, ib].set(swleaf_sha_new[_nbot : _ntop + 1])
            swsoi = swsoi.at[p, ib].set(_swsoi_ib)
            swveg = swveg.at[p, ib].set(_swveg_acc)
            swvegsun = swvegsun.at[p, ib].set(_swvegsun_acc)
            swvegsha = swvegsha.at[p, ib].set(_swvegsha_acc)
            albcan = albcan.at[p, ib].set(_albcan_ib)

        # end patch loop
    # end waveband loop

    return mlcanopy_inst._replace(
        swveg_canopy=swveg,
        swvegsun_canopy=swvegsun,
        swvegsha_canopy=swvegsha,
        albcan_canopy=albcan,
        swsoi_soil=swsoi,
        swleaf_leaf=swleaf,
        swupw_profile=swupw,
        swdwn_profile=swdwn,
        swbeam_profile=swbeam,
    )


# ---------------------------------------------------------------------------
# Private: two-stream radiative transfer
# ---------------------------------------------------------------------------


def _TwoStream(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    omega: Array,
    avmu: Array,
    betad: Array,
    betab: Array,
    clump_fac_ic: Array,
    mlcanopy_inst: mlcanopy_type,
    grid: GridInfo,
) -> mlcanopy_type:
    """
    Two-stream radiative transfer integrated over each canopy layer.

    Mirrors Fortran subroutine ``TwoStream`` (private, lines 372-490).

    References
    ----------
    Bonan et al. (2021) *Agric. For. Met.* 306, 108435,
    supplemental eqs. (1)-(5).

    The algorithm proceeds in two sweeps:

    **Bottom-to-top sweep**: For each layer compute analytic two-stream
    solutions for unit direct beam (``unitb = 1``) and unit diffuse
    (``unitd = 1``) forcing. Boundary condition at the bottom is the
    soil albedo; at each subsequent layer the upward flux from the
    layer below is used. Store per-layer scattered fluxes
    ``iupwb0``, ``iupwb``, ``idwnb`` (direct) and ``iupwd0``,
    ``iupwd``, ``idwnd`` (diffuse), plus absorbed terms partitioned
    into sunlit/shaded components.

    **Top-to-bottom sweep**: Propagate the actual incident fluxes
    downward to compute ``swbeam``, ``swdwn``, ``swupw``, and per-leaf
    absorptions ``swleaf``. Accumulate canopy totals and check energy
    conservation to within 1e-6 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch filter.
        omega: Scattering coefficient ``(p, ic, ib)``.
        avmu: Average inverse diffuse optical depth ``(p, ic)``.
        betad: Diffuse upscatter parameter ``(p, ic, ib)``.
        betab: Direct beam upscatter parameter ``(p, ic, ib)``.
        clump_fac_ic: Foliage clumping index ``(p, ic)``.
        mlcanopy_inst: Canopy container; radiation fields updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    swskyb = mlcanopy_inst.swskyb_forcing
    swskyd = mlcanopy_inst.swskyd_forcing
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    dpai = mlcanopy_inst.dpai_profile
    fracsun = mlcanopy_inst.fracsun_profile
    kb = mlcanopy_inst.kb_profile
    tbi = mlcanopy_inst.tbi_profile

    swveg = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    albcan = mlcanopy_inst.albcan_canopy
    swsoi = mlcanopy_inst.swsoi_soil
    swleaf = mlcanopy_inst.swleaf_leaf
    swupw = mlcanopy_inst.swupw_profile
    swdwn = mlcanopy_inst.swdwn_profile
    swbeam = mlcanopy_inst.swbeam_profile

    # Both wavebands at once; the two sweeps over the foliage layers are lax.scan loops
    # (one traced layer, outputs stacked) with the same arithmetic per layer as Fortran.
    p = grid.p
    _ncan, _ntop, _nbot = grid.ncan, grid.ntop, grid.nbot
    ls = slice(_nbot, _ntop + 1)
    rb = slice(1, numrad + 1)
    albsoib_p, albsoid_p = albsoib[p, rb], albsoid[p, rb]
    swskyb_p, swskyd_p = swskyb[p, rb], swskyd[p, rb]

    # ----------------------------------------------------------
    # Bottom-to-top sweep — Fortran lines 444-540
    # ----------------------------------------------------------
    def _upward(carry, x):
        albb, albd = carry
        om_ic, av_ic, kb_ic, cf_ic, dp_ic, bd_ic, bb_ic, tbi_ic = x

        b = (1.0 - (1.0 - bd_ic) * om_ic) / av_ic
        c = bd_ic * om_ic / av_ic
        # jnp.maximum guards sqrt(0) and /0 in backward pass (NaN-safe grads)
        h = jnp.sqrt(jnp.maximum(b * b - c * c, 1.0e-30))
        u = (h - b - c) / (2.0 * h)
        v = (h + b + c) / (2.0 * h)
        # Guard h²−kb² denominator for NaN-safe grads when h ≈ kb
        _h2_kb2 = h * h - kb_ic * kb_ic
        _h2_kb2_sign = jnp.where(_h2_kb2 < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        _h2_kb2_safe = _h2_kb2_sign * jnp.maximum(jnp.abs(_h2_kb2), 1.0e-30)
        d = om_ic * kb_ic / _h2_kb2_safe  # unitb=1
        g1 = (bb_ic * kb_ic - b * bb_ic - c * (1.0 - bb_ic)) * d
        g2 = ((1.0 - bb_ic) * kb_ic + c * bb_ic + b * (1.0 - bb_ic)) * d
        s1 = jnp.exp(-h * cf_ic * dp_ic)
        s2 = jnp.exp(-kb_ic * cf_ic * dp_ic)

        # Direct beam terms (unitb = 1)
        num1 = v * (g1 + g2 * albd + albb) * s2
        num2 = g2 * (u + v * albd) * s1
        den1 = v * (v + u * albd) / s1
        den2 = u * (u + v * albd) * s1
        # Guard Wronskian-like denominator for NaN-safe backward pass
        _d1d2 = den1 - den2
        _d1d2_sgn = jnp.where(_d1d2 < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        _d1d2_saf = _d1d2_sgn * jnp.maximum(jnp.abs(_d1d2), 1.0e-30)
        n2b = (num1 - num2) / _d1d2_saf
        n1b = (g2 - n2b * u) / v

        # Guard kb±h denominators — degenerate when kb≈h (already h≠0 from sqrt guard)
        _kbph = kb_ic + h  # always > 0
        _kbmh = kb_ic - h
        _kbmh_sgn = jnp.where(_kbmh < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        _kbmh_saf = _kbmh_sgn * jnp.maximum(jnp.abs(_kbmh), 1.0e-30)
        _2kb_saf = jnp.maximum(2.0 * kb_ic, 1.0e-30)
        a1b = (
            -g1 * (1.0 - s2 * s2) / _2kb_saf
            + n1b * u * (1.0 - s2 * s1) / _kbph
            + n2b * v * (1.0 - s2 / s1) / _kbmh_saf
        ) * tbi_ic
        a2b = (
            g2 * (1.0 - s2 * s2) / _2kb_saf
            - n1b * v * (1.0 - s2 * s1) / _kbph
            - n2b * u * (1.0 - s2 / s1) / _kbmh_saf
        ) * tbi_ic

        iupwb0_ic = -g1 + n1b * u + n2b * v
        iupwb_ic = -g1 * s2 + n1b * u * s1 + n2b * v / s1
        idwnb_ic = g2 * s2 - n1b * v * s1 - n2b * u / s1
        abs_b = (1.0 - s2) - iupwb0_ic + iupwb_ic - idwnb_ic
        abs_b_sun = (1.0 - om_ic) * ((1.0 - s2) + cf_ic / av_ic * (a1b + a2b))

        # Diffuse terms (unitd = 1) — reuse guards from direct-beam block
        num1 = (u + v * albd) * s1
        n2d = num1 / _d1d2_saf
        n1d = -(1.0 + n2d * u) / v

        a1d = (n1d * u * (1.0 - s2 * s1) / _kbph + n2d * v * (1.0 - s2 / s1) / _kbmh_saf) * tbi_ic
        a2d = (-n1d * v * (1.0 - s2 * s1) / _kbph - n2d * u * (1.0 - s2 / s1) / _kbmh_saf) * tbi_ic

        iupwd0_ic = n1d * u + n2d * v
        iupwd_ic = n1d * u * s1 + n2d * v / s1
        idwnd_ic = -n1d * v * s1 - n2d * u / s1
        abs_d = 1.0 - iupwd0_ic + iupwd_ic - idwnd_ic
        abs_d_sun = (1.0 - om_ic) * cf_ic / av_ic * (a1d + a2d)

        # Albedos seen by the next layer up
        out = (
            iupwb0_ic,
            idwnb_ic,
            abs_b_sun,
            abs_b - abs_b_sun,
            iupwd0_ic,
            idwnd_ic,
            abs_d_sun,
            abs_d - abs_d_sun,
        )
        return (iupwb0_ic, iupwd0_ic), out

    col = lambda a: a[:, None]  # noqa: E731  (layer value shared by both bands)
    xs_up = (
        omega[p, ls, rb],
        col(avmu[p, ls]),
        col(kb[p, ls]),
        col(clump_fac_ic[p, ls]),
        col(dpai[p, ls]),
        betad[p, ls, rb],
        betab[p, ls, rb],
        col(tbi[p, ls]),
    )
    _, (iupwb0, idwnb, iabsb_sun, iabsb_sha, iupwd0, idwnd, iabsd_sun, iabsd_sha) = jax.lax.scan(
        _upward, (albsoib_p, albsoid_p), xs_up
    )

    # ----------------------------------------------------------
    # Top-to-bottom sweep — Fortran lines 536-570
    # ----------------------------------------------------------
    def _downward(carry, x):
        dir_, dif = carry
        iu_d0, iu_b0, ab_sun, ab_sha, ad_sun, ad_sha, id_b, id_d, fs, dp_ic, kb_ic, cf_ic = x
        swupw_ic = iu_d0 * dif + iu_b0 * dir_
        # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
        fs_safe = jnp.maximum(fs, 1.0e-30)
        fsha_safe = jnp.maximum(1.0 - fs, 1.0e-30)
        dpai_safe = jnp.maximum(dp_ic, 1.0e-30)
        sun = (ab_sun * dir_ + ad_sun * dif) / (fs_safe * dpai_safe)
        sha = (ab_sha * dir_ + ad_sha * dif) / (fsha_safe * dpai_safe)
        dif_new = dir_ * id_b + dif * id_d
        dir_new = dir_ * jnp.exp(-kb_ic * cf_ic * dp_ic)
        return (dir_new, dif_new), (dir_, dif, swupw_ic, sun, sha)

    xs_down = (
        iupwd0,
        iupwb0,
        iabsb_sun,
        iabsb_sha,
        iabsd_sun,
        iabsd_sha,
        idwnb,
        idwnd,
        col(fracsun[p, ls]),
        col(dpai[p, ls]),
        col(kb[p, ls]),
        col(clump_fac_ic[p, ls]),
    )
    (dir_, dif), (beam_v, dwn_v, upw_v, sun_v, sha_v) = jax.lax.scan(
        _downward, (swskyb_p, swskyd_p), xs_down, reverse=True
    )

    # Ground fluxes
    swsoi_p = dir_ * (1.0 - albsoib_p) + dif * (1.0 - albsoid_p)

    # Canopy albedo
    suminc = swskyb_p + swskyd_p
    sumref = iupwb0[-1] * swskyb_p + iupwd0[-1] * swskyd_p
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    suminc_safe = jnp.maximum(suminc, 1.0e-30)
    albcan_p = jnp.where(suminc > 0.0, sumref / suminc_safe, 0.0)

    # Sum vegetation absorption, bottom to top as in Fortran
    fs_v = fracsun[p, ls][:, None]
    dp_v = dpai[p, ls][:, None]
    sun_abs = sun_v * fs_v * dp_v
    sha_abs = sha_v * (1.0 - fs_v) * dp_v
    swveg_p = svs_p = svsh_p = jnp.zeros(numrad)
    for k in range(_ntop - _nbot + 1):
        swveg_p = swveg_p + sun_abs[k] + sha_abs[k]
        svs_p = svs_p + sun_abs[k]
        svsh_p = svsh_p + sha_abs[k]

    # Conservation check
    sumabs = swveg_p + swsoi_p
    if is_concrete(suminc) and jnp.any(jnp.abs(suminc - (sumabs + albcan_p * suminc)) >= 1.0e-6):
        endrun(msg="ERROR: TwoStream: total solar radiation conservation error")

    # Write-back: levels 0 (ground) and the foliage layers; other levels 0
    _sl_lev = slice(0, _ncan + 1)
    _sl_canopy = slice(1, _ncan + 1)

    def _levels(v, ground):
        return jnp.zeros((_ncan + 2, numrad)).at[ls].set(v).at[0].set(ground)

    swbeam = swbeam.at[p, _sl_lev, rb].set(_levels(beam_v, dir_)[_sl_lev])
    swdwn = swdwn.at[p, _sl_lev, rb].set(_levels(dwn_v, dif)[_sl_lev])
    swupw = swupw.at[p, _sl_lev, rb].set(
        _levels(upw_v, albsoid_p * dif + albsoib_p * dir_)[_sl_lev]
    )
    swsoi = swsoi.at[p, rb].set(swsoi_p)
    albcan = albcan.at[p, rb].set(albcan_p)
    swveg = swveg.at[p, rb].set(swveg_p)
    swvegsun = swvegsun.at[p, rb].set(svs_p)
    swvegsha = swvegsha.at[p, rb].set(svsh_p)
    _zl = jnp.zeros((_ncan + 2, numrad))
    swleaf = swleaf.at[p, _sl_canopy, isun, rb].set(_zl.at[ls].set(sun_v)[_sl_canopy])
    swleaf = swleaf.at[p, _sl_canopy, isha, rb].set(_zl.at[ls].set(sha_v)[_sl_canopy])

    return mlcanopy_inst._replace(
        swveg_canopy=swveg,
        swvegsun_canopy=swvegsun,
        swvegsha_canopy=swvegsha,
        albcan_canopy=albcan,
        swsoi_soil=swsoi,
        swleaf_leaf=swleaf,
        swupw_profile=swupw,
        swdwn_profile=swdwn,
        swbeam_profile=swbeam,
    )
