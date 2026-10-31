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

from clm_ml.physics.math import is_concrete, tridiag  # noqa: F401
from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_varcon import rpi as pi  # noqa: F401
from clm_src_main.clm_varctl import iulog  # noqa: F401
from clm_src_main.clm_varpar import ivis, numrad  # noqa: F401
from clm_src_main.decompMod import bounds_type  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import (
    NORMAN_SOLAR,
    SOLAR,
    SUNLIT_FRACTION,
    mlcanopy_type,  # noqa: F401
    record_balance_error,
)
from multilayer_canopy.MLCanopyParamsMod import CanopyParams
from multilayer_canopy.MLclm_varcon import J_to_umol, chil_max, chil_min, kb_max  # noqa: F401
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401
from multilayer_canopy.MLLayersMod import keep, layers, pad_with_top

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
    *,
    opts: MLOptions,
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
    nlevmlcan = mlcanopy_inst.dpai_profile.shape[1] - 1  # layer slots (model.yaml max_layers)
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
    # Vectorized over every layer slot (plan Phase 5b); values are kept in the leafy layers
    # nbot..ntop (data), other layers up to ncan get 0, padding is kept.
    p = grid.p
    lay = layers(mlcanopy_inst, p)
    leafy = lay.leafy
    ls = slice(1, None)
    rb = slice(1, numrad + 1)
    zen = solar_zen[p]
    cos_zen = jnp.cos(zen)

    if opts.leaf_optics_type == 0:
        _xl_pft = params.leaf_angle_index
        _cf_pft = params.clumping_index
        _rhol, _taul = params.leaf_reflectance[rb], params.leaf_transmittance[rb]
        _rhos, _taus = params.stem_reflectance[rb], params.stem_transmittance[rb]
    else:
        endrun(msg=" ERROR: SolarRadiation: need to specify vertical profile for rho & tau")
        _xl_pft = 0.01
        _cf_pft = 1.0
        _rhol = _taul = _rhos = _taus = jnp.full(numrad, 1.0e-6)

    dpai_v = pad_with_top(lay, dpai[p, ls])  # padding: the top layer's (0)
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    dpai_safe = jnp.maximum(dpai_v, 1.0e-30)
    wl = (pad_with_top(lay, dlai[p, ls]) / dpai_safe)[:, None]
    ws = (pad_with_top(lay, dsai[p, ls]) / dpai_safe)[:, None]

    # Reflectance, transmittance, scattering — Fortran lines 121-132 (layer, band)
    r_v = jnp.maximum(_rhol * wl + _rhos * ws, 1.0e-6)
    t_v = jnp.maximum(_taul * wl + _taus * ws, 1.0e-6)
    om_v = r_v + t_v
    _rho_arr = _rho_arr.at[p, ls, rb].set(keep(leafy, r_v, 0.0))
    _tau_arr = _tau_arr.at[p, ls, rb].set(keep(leafy, t_v, 0.0))
    _om_arr = _om_arr.at[p, ls, rb].set(keep(leafy, om_v, 0.0))

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
    _cf_ic = _cf_ic.at[p, ls].set(jnp.where(leafy, cf, 0.0))

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
    # Over all slots from the top: layers above ntop multiply by 1 (exact), so tbi(ntop) = 1.
    tb_eff = jnp.where(leafy, tb_v, 1.0)

    def _tbi_step(carry, tb_above):
        return carry * tb_above, carry

    # tbi(ic) = tbi(ic+1) * tb(ic+1): the product over the slots above, multiplied in the
    # same order as Fortran (from the top)
    _, tbi_v = jax.lax.scan(_tbi_step, jnp.ones(()), tb_eff, reverse=True)

    # Sunlit fraction — Fortran lines 179-188
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    # When kb*dpai=0, the exp term also =0 so fracsun=0 regardless of denom.
    kb_dp_safe = jnp.maximum(kb_v * dpai_v, 1.0e-30)
    fracsun_v = tbi_v / kb_dp_safe * (1.0 - jnp.exp(-kb_v * cf * dpai_v))
    # Fortran stops if fracsun or fracsha is not positive; recorded (1 = violated), checked
    # on the host
    bad = jnp.any(leafy & ((fracsun_v <= 0.0) | ((1.0 - fracsun_v) <= 0.0)))
    mlcanopy_inst = record_balance_error(
        mlcanopy_inst, p, SUNLIT_FRACTION, jnp.where(bad, 1.0, 0.0)
    )

    # Two-stream avmu — Fortran lines 190-194
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    p1_safe = jnp.maximum(p1, 1.0e-10)
    p2_safe = jnp.maximum(p2, 1.0e-10)
    avmu_s = (1.0 - p1_safe / p2_safe * jnp.log((p1_safe + p2_safe) / p1_safe)) / p2_safe
    _avmu = _avmu.at[p, ls].set(jnp.where(leafy, avmu_s, 0.0))

    # betad, betab — Fortran lines 196-209 (layer, band)
    _betad = _betad.at[p, ls, rb].set(
        keep(leafy, 0.5 / om_v * (r_v + t_v + (r_v - t_v) * ((1.0 + chil) / 2.0) ** 2), 0.0)
    )
    tmp0 = gd + p2 * cos_zen
    tmp1 = p1 * cos_zen
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    # tmp0 = gd + p2*cos_zen, tmp1 = p1*cos_zen — both non-negative physically
    tmp0_safe = jnp.maximum(tmp0, 1.0e-10)
    tmp1_safe = jnp.maximum(tmp1, 1.0e-10)
    tmp2 = 1.0 - tmp1_safe / tmp0_safe * jnp.log((tmp1_safe + tmp0_safe) / tmp1_safe)
    asu = 0.5 * om_v * gd / tmp0_safe * tmp2
    _betab = _betab.at[p, ls, rb].set(
        keep(leafy, (1.0 + avmu_s * kb_s) / (om_v * avmu_s * kb_s) * asu, 0.0)
    )

    # Write-back: foliage layers; other canopy layers 0; tbi(0) is onto the ground
    # (Fortran lines 211-213); padding kept
    def _set(arr, v):
        return arr.at[p, ls].set(keep(lay.in_use, jnp.where(leafy, v, 0.0), arr[p, ls]))

    kb = _set(kb, kb_v)
    fracsun = _set(fracsun, fracsun_v)
    tb = _set(tb, tb_v)
    td = _set(td, td_v)
    tbi = _set(tbi, tbi_v)
    tbi = tbi.at[p, 0].set(tbi_v[lay.nbot - 1] * tb_v[lay.nbot - 1])

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
    if opts.light_type == 1:
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
    elif opts.light_type == 2:
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
        new = swleaf[p, 1:, :, ivis] * J_to_umol
        apar = apar.at[p, 1:, :].set(keep(lay.in_use, new, apar[p, 1:, :]))

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
    solves it via :func:`clm_ml.physics.math.tridiag`. The ordering of
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

    # Any layer counts (plan Phase 5b): the leafy layers nbot..ntop are data; leafy slot
    # j = 0..max_layers-1 is layer nbot + j. The tridiagonal system always has
    # 2 * max_layers + 2 rows: the real rows in Fortran order, the sky row (top layer
    # downward flux, no upper coupling) right after them, then identity rows.
    p = grid.p
    lay = layers(mlcanopy_inst, p)
    _ncan, _ntop, _nbot = lay.ncan, lay.ntop, lay.nbot
    n_slots = lay.index.shape[0]
    jv = jnp.arange(n_slots)
    valid = jv < _ntop - _nbot + 1
    ics = jnp.minimum(_nbot + jv, _ntop)  # unused slots repeat layer ntop (finite)
    m_max = 2 * n_slots + 2
    sky_row = 2 * (_ntop - _nbot + 1) + 2
    upto = jnp.arange(n_slots + 1) <= _ncan  # levels 0..ncan written, padding kept
    leafy0 = jnp.concatenate([jnp.zeros(1, bool), lay.leafy])  # levels 0..n_slots

    for ib in range(1, numrad + 1):  # Fortran: do ib = 1, numrad
        for fp in range(1, num_filter + 1):
            _swskyb_ib = swskyb[p, ib]
            _swskyd_ib = swskyd[p, ib]
            _albsoib_ib = albsoib[p, ib]
            _albsoid_ib = albsoid[p, ib]

            # Zero swleaf for layers 1..ncan
            swleaf = swleaf.at[p, 1:, :, ib].set(
                keep(lay.in_use, jnp.zeros_like(swleaf[p, 1:, :, ib]), swleaf[p, 1:, :, ib])
            )

            td_v, tb_v, tbi_v = td[p, ics], tb[p, ics], tbi[p, ics]
            rho_v, tau_v = rho[p, ics, ib], tau[p, ics, ib]
            refld = (1.0 - td_v) * rho_v
            trand = (1.0 - td_v) * tau_v + td_v
            fic = refld - trand * trand / refld  # "upward" row of layer ic
            eic = trand / refld
            d_up = _swskyb_ib * tbi_v * (1.0 - tb_v) * (rho_v - tau_v * eic)
            # "downward" rows: layer nbot's comes after the soil row, layer ic+1's after
            # layer ic's upward row; both use the layer's own properties
            aic = refld - trand * trand / refld
            bic = trand / refld
            d_dn = _swskyb_ib * tbi_v * (1.0 - tb_v) * (tau_v - rho_v * bic)

            zero, one = jnp.zeros(()), jnp.ones(())
            atri = [zero] * (m_max + 1)
            btri = [one] * (m_max + 1)
            ctri = [zero] * (m_max + 1)
            dtri = [zero] * (m_max + 1)
            # Soil: upward flux — Fortran lines 287-290
            ctri[1] = -_albsoid_ib
            dtri[1] = _swskyb_ib * tbi[p, 0] * _albsoib_ib
            # Layer j: downward row (m = 2 + 2j) and upward row (m = 3 + 2j)
            for jj in range(n_slots):
                v = valid[jj]
                ma, mf = 2 + 2 * jj, 3 + 2 * jj
                atri[ma] = jnp.where(v, -aic[jj], 0.0)
                ctri[ma] = jnp.where(v, -bic[jj], 0.0)
                dtri[ma] = jnp.where(v, d_dn[jj], 0.0)
                atri[mf] = jnp.where(v, -eic[jj], 0.0)
                ctri[mf] = jnp.where(v, -fic[jj], 0.0)
                dtri[mf] = jnp.where(v, d_up[jj], 0.0)
            # Top layer: downward flux (sky) — Fortran lines 339-343
            for m in range(4, m_max + 1, 2):
                is_sky = m == sky_row
                atri[m] = jnp.where(is_sky, 0.0, atri[m])
                ctri[m] = jnp.where(is_sky, 0.0, ctri[m])
                dtri[m] = jnp.where(is_sky, _swskyd_ib, dtri[m])

            # Solve — Fortran line 345
            rows = [jnp.stack(x[1:]) for x in (atri, btri, ctri, dtri)]  # rows 1..m_max
            u = tridiag(*rows, min_pivot=1.0e-10)

            ics_out = jnp.where(valid, ics, n_slots + 1)  # unused slots: dropped entry
            swupw_new = jnp.zeros(n_slots + 2).at[0].set(u[0]).at[ics_out].set(u[2::2][:n_slots])
            swdwn_new = jnp.zeros(n_slots + 2).at[0].set(u[1]).at[ics_out].set(u[3::2][:n_slots])

            # Ground absorption — Fortran lines 368-372
            _swbeam_0 = tbi[p, 0] * _swskyb_ib
            _swsoi_ib = _swbeam_0 * (1.0 - _albsoib_ib) + swdwn_new[0] * (1.0 - _albsoid_ib)

            # Per-layer fluxes — Fortran lines 374-408
            _swbeam_v = tbi_v * _swskyb_ib
            _om_v = omega[p, ics, ib]
            _swabsb_v = _swbeam_v * (1.0 - tb_v) * (1.0 - _om_v)
            icm1 = jnp.where(ics == _nbot, 0, ics - 1)
            _swabsd_v = (swdwn_new[ics] + swupw_new[icm1]) * (1.0 - td_v) * (1.0 - _om_v)
            _fs = fracsun[p, ics]
            _swsha_v = _swabsd_v * (1.0 - _fs)
            _swsun_v = _swabsd_v * _fs + _swabsb_v
            fs_safe = jnp.maximum(_fs, 1.0e-30)
            fsha_safe = jnp.maximum(1.0 - _fs, 1.0e-30)
            dpai_safe = jnp.maximum(dpai[p, ics], 1.0e-30)
            sun_v = _swsun_v / (fs_safe * dpai_safe)
            sha_v = _swsha_v / (fsha_safe * dpai_safe)
            _swveg_acc = _swvegsun_acc = _swvegsha_acc = jnp.zeros(())
            for jj in range(n_slots):  # bottom to top, as Fortran; unused slots add 0
                v = valid[jj]
                _swveg_acc = _swveg_acc + jnp.where(v, _swabsb_v[jj] + _swabsd_v[jj], 0.0)
                _swvegsun_acc = _swvegsun_acc + jnp.where(v, _swsun_v[jj], 0.0)
                _swvegsha_acc = _swvegsha_acc + jnp.where(v, _swsha_v[jj], 0.0)
            swbeam_new = jnp.zeros(n_slots + 2).at[0].set(_swbeam_0).at[ics_out].set(_swbeam_v)
            swleaf_sun_new = jnp.zeros(n_slots + 2).at[ics_out].set(sun_v)
            swleaf_sha_new = jnp.zeros(n_slots + 2).at[ics_out].set(sha_v)

            # Albedo — Fortran lines 410-414
            _suminc = _swskyb_ib + _swskyd_ib
            _suminc_safe = jnp.maximum(_suminc, 1.0e-30)
            _albcan_ib = jnp.where(_suminc > 0.0, swupw_new[_ntop] / _suminc_safe, 0.0)

            # Conservation checks — Fortran lines 416-426; recorded, checked on the host
            _sumref = _albcan_ib * _suminc
            _sumabs = _suminc - _sumref
            _err = _sumabs - (_swveg_acc + _swsoi_ib)
            _err2 = (_swvegsun_acc + _swvegsha_acc) - _swveg_acc
            mlcanopy_inst = record_balance_error(mlcanopy_inst, p, NORMAN_SOLAR, _err)
            mlcanopy_inst = record_balance_error(mlcanopy_inst, p, NORMAN_SOLAR, _err2)

            # Write-back: levels 0..ncan; leaf fluxes in the leafy layers
            n1 = n_slots + 1
            swupw = swupw.at[p, :, ib].set(jnp.where(upto, swupw_new[:n1], swupw[p, :, ib]))
            swdwn = swdwn.at[p, :, ib].set(jnp.where(upto, swdwn_new[:n1], swdwn[p, :, ib]))
            swbeam = swbeam.at[p, :, ib].set(jnp.where(upto, swbeam_new[:n1], swbeam[p, :, ib]))
            swleaf = swleaf.at[p, :, isun, ib].set(
                jnp.where(leafy0, swleaf_sun_new[:n1], swleaf[p, :, isun, ib])
            )
            swleaf = swleaf.at[p, :, isha, ib].set(
                jnp.where(leafy0, swleaf_sha_new[:n1], swleaf[p, :, isha, ib])
            )
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
    # Sweeps run over every layer slot (plan Phase 5b): leafy layers nbot..ntop (data) are
    # computed as in Fortran; other slots pass the sweep's values through unchanged and get
    # a leafy layer's inputs (finite arithmetic; their outputs are discarded).
    p = grid.p
    lay = layers(mlcanopy_inst, p)
    leafy = lay.leafy
    ls = slice(1, None)
    rb = slice(1, numrad + 1)

    def _in(x):
        """Per-slot input, with non-leafy slots given layer ntop's value."""
        return keep(leafy, x, jnp.take(x, lay.ntop - 1, axis=0))

    albsoib_p, albsoid_p = albsoib[p, rb], albsoid[p, rb]
    swskyb_p, swskyd_p = swskyb[p, rb], swskyd[p, rb]

    # ----------------------------------------------------------
    # Bottom-to-top sweep — Fortran lines 444-540
    # ----------------------------------------------------------
    def _upward(carry, x):
        albb, albd = carry
        is_leafy, om_ic, av_ic, kb_ic, cf_ic, dp_ic, bd_ic, bb_ic, tbi_ic = x

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
        carry = (jnp.where(is_leafy, iupwb0_ic, albb), jnp.where(is_leafy, iupwd0_ic, albd))
        return carry, out

    col = lambda a: a[:, None]  # noqa: E731  (layer value shared by both bands)
    xs_up = (
        leafy,
        _in(omega[p, ls, rb]),
        col(_in(avmu[p, ls])),
        col(_in(kb[p, ls])),
        col(_in(clump_fac_ic[p, ls])),
        col(_in(dpai[p, ls])),
        _in(betad[p, ls, rb]),
        _in(betab[p, ls, rb]),
        col(_in(tbi[p, ls])),
    )
    _, (iupwb0, idwnb, iabsb_sun, iabsb_sha, iupwd0, idwnd, iabsd_sun, iabsd_sha) = jax.lax.scan(
        _upward, (albsoib_p, albsoid_p), xs_up
    )

    # ----------------------------------------------------------
    # Top-to-bottom sweep — Fortran lines 536-570
    # ----------------------------------------------------------
    def _downward(carry, x):
        dir_, dif = carry
        (is_leafy, iu_d0, iu_b0, ab_sun, ab_sha, ad_sun, ad_sha, id_b, id_d, fs, dp_ic, kb_ic,
         cf_ic) = x  # fmt: skip
        swupw_ic = iu_d0 * dif + iu_b0 * dir_
        # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
        fs_safe = jnp.maximum(fs, 1.0e-30)
        fsha_safe = jnp.maximum(1.0 - fs, 1.0e-30)
        dpai_safe = jnp.maximum(dp_ic, 1.0e-30)
        sun = (ab_sun * dir_ + ad_sun * dif) / (fs_safe * dpai_safe)
        sha = (ab_sha * dir_ + ad_sha * dif) / (fsha_safe * dpai_safe)
        dif_new = dir_ * id_b + dif * id_d
        dir_new = dir_ * jnp.exp(-kb_ic * cf_ic * dp_ic)
        dir_new = jnp.where(is_leafy, dir_new, dir_)
        dif_new = jnp.where(is_leafy, dif_new, dif)
        return (dir_new, dif_new), (dir_, dif, swupw_ic, sun, sha)

    xs_down = (
        leafy,
        iupwd0,
        iupwb0,
        iabsb_sun,
        iabsb_sha,
        iabsd_sun,
        iabsd_sha,
        idwnb,
        idwnd,
        col(_in(fracsun[p, ls])),
        col(_in(dpai[p, ls])),
        col(_in(kb[p, ls])),
        col(_in(clump_fac_ic[p, ls])),
    )
    (dir_, dif), (beam_v, dwn_v, upw_v, sun_v, sha_v) = jax.lax.scan(
        _downward, (swskyb_p, swskyd_p), xs_down, reverse=True
    )

    # Ground fluxes
    swsoi_p = dir_ * (1.0 - albsoib_p) + dif * (1.0 - albsoid_p)

    # Canopy albedo
    suminc = swskyb_p + swskyd_p
    sumref = iupwb0[lay.ntop - 1] * swskyb_p + iupwd0[lay.ntop - 1] * swskyd_p
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    suminc_safe = jnp.maximum(suminc, 1.0e-30)
    albcan_p = jnp.where(suminc > 0.0, sumref / suminc_safe, 0.0)

    # Sum vegetation absorption, bottom to top as in Fortran (other slots add exactly 0)
    fs_v = _in(fracsun[p, ls])[:, None]
    dp_v = _in(dpai[p, ls])[:, None]
    sun_abs = jnp.where(leafy[:, None], sun_v * fs_v * dp_v, 0.0)
    sha_abs = jnp.where(leafy[:, None], sha_v * (1.0 - fs_v) * dp_v, 0.0)
    swveg_p = svs_p = svsh_p = jnp.zeros(numrad)
    for k in range(sun_abs.shape[0]):
        swveg_p = swveg_p + sun_abs[k] + sha_abs[k]
        svs_p = svs_p + sun_abs[k]
        svsh_p = svsh_p + sha_abs[k]

    # Conservation check — Fortran stops here; recorded and checked on the host
    sumabs = swveg_p + swsoi_p
    err = suminc - (sumabs + albcan_p * suminc)
    mlcanopy_inst = record_balance_error(mlcanopy_inst, p, SOLAR, err)

    # Write-back: levels 0 (ground) and the foliage layers; other levels up to ncan 0;
    # padding slots kept
    upto = jnp.arange(dpai.shape[1]) <= lay.ncan

    def _levels(v, ground):
        return jnp.concatenate([ground[None], jnp.where(leafy[:, None], v, 0.0)])

    def _set(arr, v):
        return arr.at[p, :, rb].set(keep(upto, v, arr[p, :, rb]))

    swbeam = _set(swbeam, _levels(beam_v, dir_))
    swdwn = _set(swdwn, _levels(dwn_v, dif))
    swupw = _set(swupw, _levels(upw_v, albsoid_p * dif + albsoib_p * dir_))
    swsoi = swsoi.at[p, rb].set(swsoi_p)
    albcan = albcan.at[p, rb].set(albcan_p)
    swveg = swveg.at[p, rb].set(swveg_p)
    swvegsun = swvegsun.at[p, rb].set(svs_p)
    swvegsha = swvegsha.at[p, rb].set(svsh_p)
    swleaf = swleaf.at[p, ls, isun, rb].set(
        keep(lay.in_use, jnp.where(leafy[:, None], sun_v, 0.0), swleaf[p, ls, isun, rb])
    )
    swleaf = swleaf.at[p, ls, isha, rb].set(
        keep(lay.in_use, jnp.where(leafy[:, None], sha_v, 0.0), swleaf[p, ls, isha, rb])
    )

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
