"""The solar radiation routine of the legacy time step as an adapter to
``clm_ml.physics.solar_radiation`` (Phase 7.2, process 4; transition only: it goes when the
containers leave the step, process 10).

Legacy layout: a patch axis, layer slots 1..max_layers (level 0 the ground), leaves
``isun = 1``/``isha = 2``, bands ``ivis = 1``/``inir = 2`` (index 0 unused). ``DECLARED``
lists the container fields the adapter reads and writes (checked by
``tests/clm_ml/test_physics_solar_radiation.py``).

The Norman scheme (``light_type = 1``) is registered but not supported (Phase 7 decision D8:
``clm_ml.run`` refuses it). Its legacy translation is kept here unchanged (``_Norman``),
never validated, until it is fixed and moved after Phase 8.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from jax import Array

from clm_ml.legacy.state import mlcanopy_layers
from clm_ml.physics import solar_radiation as physics
from clm_ml.physics.layers import keep as keep0
from clm_ml.physics.math import tridiag
from clm_src_main.abortutils import endrun
from clm_src_main.clm_varpar import numrad
from clm_src_main.decompMod import bounds_type
from multilayer_canopy.MLCanopyFluxesType import (
    NORMAN_SOLAR,
    SOLAR,
    SUNLIT_FRACTION,
    mlcanopy_type,
    record_balance_error,
)
from multilayer_canopy.MLclm_varctl import GridInfo
from multilayer_canopy.MLclm_varpar import isha, isun
from multilayer_canopy.MLLayersMod import keep, layers

_ML = "mlcanopy."
#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "SolarRadiation": (
        tuple(_ML + f for f in (
            "solar_zen_forcing", "dlai_profile", "dsai_profile", "dpai_profile",
            "ncan_canopy", "ntop_canopy", "nbot_canopy", "dz_profile", "zs_profile",
            "zw_profile", "swskyb_forcing", "swskyd_forcing", "albsoib_soil", "albsoid_soil",
            "fracsun_profile", "kb_profile", "tb_profile", "td_profile", "tbi_profile",
            "swbeam_profile", "swdwn_profile", "swupw_profile", "swleaf_leaf", "swsoi_soil",
            "swveg_canopy", "swvegsun_canopy", "swvegsha_canopy", "albcan_canopy",
            "apar_leaf", "balance_error")),
        tuple(_ML + f for f in (
            "fracsun_profile", "kb_profile", "tb_profile", "td_profile", "tbi_profile",
            "swbeam_profile", "swdwn_profile", "swupw_profile", "swleaf_leaf", "swsoi_soil",
            "swveg_canopy", "swvegsun_canopy", "swvegsha_canopy", "albcan_canopy",
            "apar_leaf", "balance_error")),
    )
}  # fmt: skip


def SolarRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    params,
    grid: GridInfo,
    *,
    opts,
) -> mlcanopy_type:
    """Solar radiation of the canopy layers and the ground (``physics.canopy_optics``, then
    the scheme ``opts.light_type`` chooses, then ``physics.absorbed_par``).

    Fortran reference: MLSolarRadiationMod.F90::SolarRadiation. Values are written in the
    leafy layers, 0 in the other layers in use; padding slots are kept.
    """
    if opts.leaf_optics_type != 0:
        endrun(msg=" ERROR: SolarRadiation: need to specify vertical profile for rho & tau")
    inst = mlcanopy_inst
    p = grid.p
    lay = mlcanopy_layers(inst, p)
    ls, rb = slice(1, None), slice(1, numrad + 1)

    optics = physics.canopy_optics(
        inst.solar_zen_forcing[p],
        inst.dlai_profile[p, ls],
        inst.dsai_profile[p, ls],
        inst.dpai_profile[p, ls],
        lay,
        params.leaf_reflectance[rb],
        params.leaf_transmittance[rb],
        params.stem_reflectance[rb],
        params.stem_transmittance[rb],
        params.leaf_angle_index,
        params.clumping_index,
    )
    inst = record_balance_error(inst, p, SUNLIT_FRACTION, optics.sunlit_fraction_error)

    def _layers(arr, v):
        return arr.at[p, ls].set(keep0(lay.in_use, v, arr[p, ls]))

    inst = inst._replace(
        kb_profile=_layers(inst.kb_profile, optics.kb),
        fracsun_profile=_layers(inst.fracsun_profile, optics.fracsun),
        tb_profile=_layers(inst.tb_profile, optics.tb),
        td_profile=_layers(inst.td_profile, optics.td),
        tbi_profile=_layers(inst.tbi_profile, optics.tbi).at[p, 0].set(optics.tbi_ground),
    )

    if opts.light_type == 2:
        f = physics.two_stream(
            optics,
            inst.dpai_profile[p, ls],
            inst.swskyb_forcing[p, rb],
            inst.swskyd_forcing[p, rb],
            inst.albsoib_soil[p, rb],
            inst.albsoid_soil[p, rb],
            lay,
        )
        inst = record_balance_error(inst, p, SOLAR, f.error)
        upto = jnp.arange(inst.dpai_profile.shape[1]) <= lay.n_layers  # levels 0..ncan

        def _levels(arr, v):
            return arr.at[p, :, rb].set(keep0(upto, v, arr[p, :, rb]))

        swleaf = inst.swleaf_leaf
        for leaf, il in ((0, isun), (1, isha)):
            old = swleaf[p, ls, il, rb]
            swleaf = swleaf.at[p, ls, il, rb].set(keep0(lay.in_use, f.swleaf[:, leaf], old))
        inst = inst._replace(
            swbeam_profile=_levels(inst.swbeam_profile, f.swbeam),
            swdwn_profile=_levels(inst.swdwn_profile, f.swdwn),
            swupw_profile=_levels(inst.swupw_profile, f.swupw),
            swleaf_leaf=swleaf,
            swsoi_soil=inst.swsoi_soil.at[p, rb].set(f.swsoi),
            swveg_canopy=inst.swveg_canopy.at[p, rb].set(f.swveg),
            swvegsun_canopy=inst.swvegsun_canopy.at[p, rb].set(f.swvegsun),
            swvegsha_canopy=inst.swvegsha_canopy.at[p, rb].set(f.swvegsha),
            albcan_canopy=inst.albcan_canopy.at[p, rb].set(f.albcan),
        )
    elif opts.light_type == 1:
        shape = inst.swupw_profile.shape  # (patch, level, 1 + band)

        def _legacy(v):
            return jnp.zeros(shape).at[p, ls, rb].set(v)

        inst = _Norman(
            bounds,
            num_filter,
            filter_patch,
            _legacy(optics.rho),
            _legacy(optics.tau),
            _legacy(optics.omega),
            inst,
            grid=grid,
        )
    else:
        endrun(msg=" ERROR: SolarRadiation: light_type not valid")

    # Absorbed PAR (umol/m2/s) of every leaf index, from the visible band
    apar = inst.apar_leaf
    new = physics.absorbed_par(inst.swleaf_leaf[p, ls, :, rb])
    return inst._replace(apar_leaf=apar.at[p, ls, :].set(keep0(lay.in_use, new, apar[p, ls, :])))


# ---------------------------------------------------------------------------
# Norman (1979) radiative transfer: registered but not supported (decision D8); the legacy
# translation, unchanged
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
