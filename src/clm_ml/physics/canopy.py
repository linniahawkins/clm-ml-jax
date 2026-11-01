"""The step-level canopy pieces of the multilayer canopy time step: the plant area of each
layer for the step's LAI and SAI, the soil surface humidity, the averages of the fluxes over
the sub-steps, the canopy totals and conservation checks, and the merge of sunlit and shaded
leaves at the end of the step.

One patch. Layers are 0-based slots on axis 0 (``clm_ml.state.PatchLayers``); levels (the
ground and the layers above it) on axis 0 of the ``level`` arrays; leaves 0 sunlit, 1 shaded
on axis 1; bands 0 visible, 1 near-infrared on the last axis.

Fortran reference: CLM-ml v2 ``MLCanopyFluxesMod.F90`` (``MLCanopyFluxes``,
``MLTimeStepFluxIntegration``, ``CanopyFluxesDiagnostics``).

Reads and writes are the arguments and results; the docstrings name the legacy container
fields they come from and go to (through the adapters in ``clm_ml.legacy.canopy`` until the
containers leave the step, Phase 7.2 process 10).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.physics.constants import grav, mmh2o, rgas
from clm_ml.physics.layers import pad_with_top
from clm_ml.physics.water_vapor import lat_vap
from clm_ml.state import PatchLayers

# ---------------------------------------------------------------------------
# Plant area of each layer — Fortran MLCanopyFluxes, lines 220-243
# ---------------------------------------------------------------------------


class PlantArea(NamedTuple):
    """Leaf and stem area of the canopy and its layers for the step's LAI and SAI."""

    lai: Array  # () leaf area index, m2/m2 (lai_canopy)
    sai: Array  # () stem area index with sai_floor, m2/m2 (sai_canopy)
    sai_floor: Array  # () stem area added by the per-layer minimum, m2/m2 (sai_floor_canopy)
    dlai: Array  # (layer,) leaf area index of the layer (dlai_profile)
    dsai: Array  # (layer,) stem area index of the layer (dsai_profile)
    dpai: Array  # (layer,) plant area index of the layer (dpai_profile)
    error: Array  # () sum of dpai minus (lai + sai): the check PLANT_AREA


def plant_area(
    layers: PatchLayers,
    dlai_frac: Array,
    dsai_frac: Array,
    lai: Array,
    sai: Array,
    layer_pai_floor: float,
) -> PlantArea:
    """The layers' leaf and stem area: the canopy's LAI and SAI (``elai``, ``esai`` of the
    CLM history) spread with the layers' fractions (``dlai_frac_profile``,
    ``dsai_frac_profile``, set at initialization).

    Not Fortran (Phase 6b, issue 14): every leafy layer keeps at least ``layer_pai_floor`` of
    plant area, the missing part added as stem area, and the canopy's stem area includes it.
    CLM-ml's 9-angle diffuse transmittance exceeds 1 below ~0.005 m2/m2, which makes a layer's
    net longwave grow with its temperature and the leaf temperature run away. Layers at or
    above the floor are unchanged (``x + 0.0``), so CLM-ml v2 results are kept where no layer
    is that thin (CHATS7). Value: CLM-ml v1's ``dpai_min`` (m2/m2).

    Every slot is computed (padding with the top layer's fractions); the caller keeps the
    layers in use. Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 220-243).
    """
    dlai = pad_with_top(layers, dlai_frac) * lai
    dsai = pad_with_top(layers, dsai_frac) * sai
    add = jnp.where(layers.leafy, jnp.maximum(layer_pai_floor - (dlai + dsai), 0.0), 0.0)
    dsai = dsai + add
    sai_floor = jnp.sum(add)
    sai = sai + sai_floor
    dpai = dlai + dsai
    error = jnp.sum(jnp.where(layers.in_use, dpai, 0.0)) - (lai + sai)
    return PlantArea(lai, sai, sai_floor, dlai, dsai, dpai, error)


def soil_surface_humidity(smp: Array, t_soil: Array) -> Array:
    """Relative humidity of the air at the soil surface (``rhg_soil``) from the matric
    potential (mm) and temperature (K) of the first soil layer.

    Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes (lines 259-262).
    """
    return jnp.exp(grav * mmh2o * smp * 1.0e-3 / (rgas * t_soil))


# ---------------------------------------------------------------------------
# Averages over the sub-steps — Fortran MLTimeStepFluxIntegration
# ---------------------------------------------------------------------------


class SubstepFluxes(NamedTuple):
    """The values averaged over the sub-steps of a step (the legacy container's field of the
    same name, without its suffix). Every slot of the layer, level and leaf axes."""

    ustar: Array  # () friction velocity, m/s (_canopy)
    beta: Array  # () u*/U(h) (_canopy)
    obu: Array  # () Obukhov length, m (_canopy)
    z0m: Array  # () roughness length for momentum, m (_canopy)
    zdisp: Array  # () displacement height, m (_canopy)
    lwup: Array  # () upward longwave above the canopy, W/m2 (_canopy)
    swsoi: Array  # (band,) shortwave absorbed by the ground, W/m2 (_soil)
    lwsoi: Array  # () longwave absorbed by the ground, W/m2 (_soil)
    rnsoi: Array  # () net radiation of the ground, W/m2 (_soil)
    shsoi: Array  # () sensible heat flux of the ground, W/m2 (_soil)
    lhsoi: Array  # () latent heat flux of the ground, W/m2 (_soil)
    etsoi: Array  # () water vapor flux of the ground, mol H2O/m2/s (_soil)
    gsoi: Array  # () soil heat flux, W/m2 (_soil)
    gac0: Array  # () aerodynamic conductance of the ground, mol/m2/s (_soil)
    qflx_intr: Array  # () intercepted precipitation, kg H2O/m2/s (_canopy)
    qflx_tflrain: Array  # () rain throughfall, kg H2O/m2/s (_canopy)
    qflx_tflsnow: Array  # () snow throughfall, kg H2O/m2/s (_canopy)
    swskyb: Array  # (band,) direct beam shortwave, W/m2 (_forcing)
    swskyd: Array  # (band,) diffuse shortwave, W/m2 (_forcing)
    lwsky: Array  # () atmospheric longwave, W/m2 (_forcing)
    # Not Fortran (issue 12): the canopy-top latent heat flux of the sub-step, with the
    # sub-step's own latent heat (Fortran multiplies the averaged water vapor flux by the last
    # sub-step's LatVap(tref), which differs when tref crosses freezing within the step)
    lhtop: Array  # () W/m2 (_canopy)
    shair: Array  # (layer,) air sensible heat flux, W/m2 (_profile)
    etair: Array  # (layer,) air water vapor flux, mol H2O/m2/s (_profile)
    stair: Array  # (layer,) air storage heat flux, W/m2 (_profile)
    mflx: Array  # (layer,) momentum flux, m2/s2 (_profile)
    kc_eddy: Array  # (layer,) eddy diffusivity, m2/s (_profile)
    gac: Array  # (layer,) aerodynamic conductance for scalars, mol/m2/s (_profile)
    swupw: Array  # (level, band) upward diffuse shortwave, W/m2 (_profile)
    swdwn: Array  # (level, band) downward diffuse shortwave, W/m2 (_profile)
    swbeam: Array  # (level, band) direct beam shortwave, W/m2 (_profile)
    lwupw: Array  # (level,) upward longwave, W/m2 (_profile)
    lwdwn: Array  # (level,) downward longwave, W/m2 (_profile)
    swleaf: Array  # (layer, leaf, band) leaf absorbed shortwave, W/m2 leaf (_leaf)
    lwleaf: Array  # (layer, leaf) leaf absorbed longwave, W/m2 leaf (_leaf)
    rnleaf: Array  # (layer, leaf) leaf net radiation, W/m2 leaf (_leaf)
    shleaf: Array  # (layer, leaf) leaf sensible heat flux, W/m2 leaf (_leaf)
    lhleaf: Array  # (layer, leaf) leaf latent heat flux, W/m2 leaf (_leaf)
    trleaf: Array  # (layer, leaf) leaf transpiration, mol H2O/m2 leaf/s (_leaf)
    evleaf: Array  # (layer, leaf) leaf evaporation, mol H2O/m2 leaf/s (_leaf)
    stleaf: Array  # (layer, leaf) leaf storage heat flux, W/m2 leaf (_leaf)
    anet: Array  # (layer, leaf) net photosynthesis, umol CO2/m2 leaf/s (_leaf)
    agross: Array  # (layer, leaf) gross photosynthesis, umol CO2/m2 leaf/s (_leaf)
    gs: Array  # (layer, leaf) stomatal conductance, mol H2O/m2 leaf/s (_leaf)


def canopy_top_latent_heat(layers: PatchLayers, etair: Array, tref: Array) -> Array:
    """The sub-step's latent heat flux at the canopy top (top layer in use), W/m2: the
    ``lhtop`` entry of :class:`SubstepFluxes` (not Fortran, issue 12)."""
    return jnp.take(etair, layers.n_layers - 1, axis=0) * lat_vap(tref)


def zero_sums(fluxes: SubstepFluxes) -> SubstepFluxes:
    """Sums of no sub-steps, shaped as ``fluxes``."""
    return jax.tree.map(jnp.zeros_like, fluxes)


def accumulate(sums: SubstepFluxes, fluxes: SubstepFluxes) -> SubstepFluxes:
    """``sums`` with one sub-step's ``fluxes`` added.

    Fortran reference: MLCanopyFluxesMod.F90::MLTimeStepFluxIntegration (the sums).
    """
    return jax.tree.map(jnp.add, sums, fluxes)


def average(sums: SubstepFluxes, n_substeps: int) -> SubstepFluxes:
    """The averages of ``sums`` over ``n_substeps`` sub-steps (multiplied by
    ``1 / n_substeps``, as Fortran).

    Fortran reference: MLCanopyFluxesMod.F90::MLTimeStepFluxIntegration (the averages).
    """
    scale = 1.0 / float(n_substeps)
    return jax.tree.map(lambda s: s * scale, sums)


# ---------------------------------------------------------------------------
# Canopy totals and checks — Fortran CanopyFluxesDiagnostics
# ---------------------------------------------------------------------------


class CanopyLeaves(NamedTuple):
    """End-of-step values of the layers and leaves read by the canopy totals (not
    averaged)."""

    dpai: Array  # (layer,) plant area index (dpai_profile)
    fracsun: Array  # (layer,) sunlit fraction (fracsun_profile)
    fwet: Array  # (layer,) wet fraction of the plant area (fwet_profile)
    fdry: Array  # (layer,) green and dry fraction of the plant area (fdry_profile)
    wind: Array  # (layer,) wind speed, m/s (wind_profile)
    tair: Array  # (layer,) air temperature, K (tair_profile)
    vcmax25: Array  # (layer,) Vcmax at 25 C of the layer (vcmax25_profile)
    tleaf: Array  # (layer, leaf) leaf temperature, K (tleaf_leaf)
    lwp: Array  # (layer, leaf) leaf water potential, MPa (lwp_leaf)
    vcmax25_leaf: Array  # (layer, leaf) Vcmax at 25 C (vcmax25_leaf)


class CanopyTotals(NamedTuple):
    """Canopy totals of a step (``*_canopy`` fields; per m2 ground) and the layers' mean leaf
    water potential, with the errors of Fortran's four energy checks. Only what outputs and
    checks read (D6; the other totals and profiles of Fortran's ``CanopyFluxesDiagnostics``
    are listed in ``docs/design/phase7_removed_diagnostics.md``)."""

    swveg: Array  # (band,) absorbed shortwave: vegetation, W/m2
    swvegsun: Array  # (band,) absorbed shortwave: sunlit canopy, W/m2
    swvegsha: Array  # (band,) absorbed shortwave: shaded canopy, W/m2
    lwveg: Array  # () absorbed longwave: vegetation, W/m2
    stflx_veg: Array  # () vegetation heat storage flux, W/m2
    shveg: Array  # () sensible heat: vegetation, W/m2
    shvegsun: Array  # () sensible heat: sunlit canopy, W/m2
    shvegsha: Array  # () sensible heat: shaded canopy, W/m2
    lhveg: Array  # () latent heat: vegetation, W/m2
    lhvegsun: Array  # () latent heat: sunlit canopy, W/m2
    lhvegsha: Array  # () latent heat: shaded canopy, W/m2
    trveg: Array  # () transpiration, mol H2O/m2/s
    evveg: Array  # () evaporation of intercepted water, mol H2O/m2/s
    gppveg: Array  # () GPP: vegetation, umol CO2/m2/s
    gppvegsun: Array  # () GPP: sunlit canopy, umol CO2/m2/s
    gppvegsha: Array  # () GPP: shaded canopy, umol CO2/m2/s
    vcmax25veg: Array  # () Vcmax25: canopy, umol/m2/s
    vcmax25sun: Array  # () Vcmax25: sunlit canopy, umol/m2/s
    vcmax25sha: Array  # () Vcmax25: shaded canopy, umol/m2/s
    gsveg: Array  # () stomatal conductance: canopy, mol H2O/m2/s
    gsvegsun: Array  # () stomatal conductance: sunlit canopy, mol H2O/m2/s
    gsvegsha: Array  # () stomatal conductance: shaded canopy, mol H2O/m2/s
    albcan: Array  # (band,) canopy albedo
    shflx: Array  # () total sensible heat flux, W/m2
    etflx: Array  # () total water vapor flux, mol H2O/m2/s (CLM coupling, mlcan_to_clm)
    lhflx: Array  # () total latent heat flux, W/m2
    stflx_air: Array  # () canopy air storage heat flux, W/m2
    rnet: Array  # () net radiation, W/m2
    laisun: Array  # () sunlit plant area index, m2/m2
    laisha: Array  # () shaded plant area index, m2/m2
    windveg: Array  # () wind speed: vegetation (plant-area weighted), m/s
    windvegsun: Array  # () wind speed: sunlit canopy, m/s
    windvegsha: Array  # () wind speed: shaded canopy, m/s
    tlveg: Array  # () leaf temperature: vegetation, K
    tlvegsun: Array  # () leaf temperature: sunlit canopy, K
    tlvegsha: Array  # () leaf temperature: shaded canopy, K
    taveg: Array  # () air temperature: vegetation, K
    tavegsun: Array  # () air temperature: sunlit canopy, K
    tavegsha: Array  # () air temperature: shaded canopy, K
    fracminlwp: Array  # () fraction of the canopy at leaf water potential <= -2 MPa
    lwp_mean: Array  # (layer,) leaf water potential, sunlit/shaded mean, MPa (lwp_mean_profile)
    errors: Array  # (4,) the checks CANOPY_ENERGY_1 .. CANOPY_ENERGY_4 (signed)


def canopy_totals(
    layers: PatchLayers,
    means: SubstepFluxes,
    leaves: CanopyLeaves,
    lai: Array,
    sai: Array,
    *,
    flux_profile_type: int,
) -> CanopyTotals:
    """Canopy totals from the step's averaged fluxes (``means``) and the end-of-step layers
    and leaves; ``lai``, ``sai`` the canopy's leaf and stem area (``sai`` with the per-layer
    minimum). Sunlit and shaded leaves are weighted by the sunlit fraction, layers by their
    plant area; layers without plant area and the padding contribute nothing.

    ``flux_profile_type`` (static): 1 the canopy's turbulent fluxes are those at the top layer
    (``lhflx`` the sub-step average of the canopy-top latent heat flux, issue 12; Fortran
    ``etair(ncan) * LatVap(tref)`` with the averaged ``etair`` and the last sub-step's
    ``tref``); 0 or -1 the vegetation's plus the ground's.

    Fortran reference: MLCanopyFluxesMod.F90::CanopyFluxesDiagnostics.
    """
    if flux_profile_type not in (-1, 0, 1):
        raise ValueError(f"canopy_totals: flux_profile_type {flux_profile_type} not valid")

    def pad(x):
        return pad_with_top(layers, x)

    dpai = pad(leaves.dpai)
    fs = pad(leaves.fracsun)
    fsh = 1.0 - fs
    fwet = pad(leaves.fwet)
    fdry = pad(leaves.fdry)
    has_pai = (dpai > 0.0) & layers.in_use
    dpai_s = jnp.where(has_pai, dpai, 0.0)

    # fracgreen = fdry / (1 - fwet), guarded against fwet == 1
    fwet_safe = jnp.where(fwet < 1.0, fwet, 0.0)
    fracgreen = jnp.where(has_pai, fdry / (1.0 - fwet_safe), 0.0)

    swleaf = pad(means.swleaf)
    sw_sun_vis, sw_sun_nir = swleaf[:, 0, 0], swleaf[:, 0, 1]
    sw_sha_vis, sw_sha_nir = swleaf[:, 1, 0], swleaf[:, 1, 1]

    def sun_sha(x):
        x = pad(x)
        return x[:, 0], x[:, 1]

    lw_sun, lw_sha = sun_sha(means.lwleaf)
    st_sun, st_sha = sun_sha(means.stleaf)
    sh_sun, sh_sha = sun_sha(means.shleaf)
    lh_sun, lh_sha = sun_sha(means.lhleaf)
    ev_sun, ev_sha = sun_sha(means.evleaf)
    tr_sun, tr_sha = sun_sha(means.trleaf)
    ag_sun, ag_sha = sun_sha(means.agross)
    gs_sun, gs_sha = sun_sha(means.gs)
    tl_sun, tl_sha = sun_sha(leaves.tleaf)
    lp_sun, lp_sha = sun_sha(leaves.lwp)
    vc_sun, vc_sha = sun_sha(leaves.vcmax25_leaf)

    # Layer means per leaf area, then sources per ground area
    def blend(a_sun, a_sha):
        return jnp.where(has_pai, a_sun * fs + a_sha * fsh, 0.0)

    lw_m = blend(lw_sun, lw_sha)
    sw_m_vis = blend(sw_sun_vis, sw_sha_vis)
    sw_m_nir = blend(sw_sun_nir, sw_sha_nir)
    st_m = blend(st_sun, st_sha)
    sh_m = blend(sh_sun, sh_sha)
    lh_m = blend(lh_sun, lh_sha)
    ev_m = blend(ev_sun, ev_sha)
    tr_m = blend(tr_sun, tr_sha)
    et_m = ev_m + tr_m
    gs_m = blend(gs_sun, gs_sha)
    tl_m = blend(tl_sun, tl_sha)
    lp_m = blend(lp_sun, lp_sha)

    # Vegetation totals
    swveg_vis = jnp.sum(sw_m_vis * dpai_s)
    swveg_nir = jnp.sum(sw_m_nir * dpai_s)
    lwveg = jnp.sum(lw_m * dpai_s)
    stflx_veg = jnp.sum(st_m * dpai_s)
    shveg = jnp.sum(sh_m * dpai_s)
    lhveg = jnp.sum(lh_m * dpai_s)
    etveg = jnp.sum(et_m * dpai_s)
    trveg = jnp.sum(tr_m * dpai_s)
    evveg = jnp.sum(ev_m * dpai_s)
    vcmax25veg = jnp.sum(pad(leaves.vcmax25) * dpai_s)
    gppveg = jnp.sum((ag_sun * fs + ag_sha * fsh) * dpai_s * fracgreen)
    gsveg = jnp.sum(gs_m * dpai_s)

    # Vegetation energy balance
    err1 = swveg_vis + swveg_nir + lwveg - shveg - lhveg - stflx_veg

    # Albedo (0 without incoming shortwave)
    inc_vis = means.swskyb[0] + means.swskyd[0]
    inc_nir = means.swskyb[1] + means.swskyd[1]
    swupw_top = jnp.take(means.swupw, layers.n_within, axis=0)
    alb_vis = jnp.where(inc_vis > 0.0, swupw_top[0] / jnp.maximum(inc_vis, 1e-30), 0.0)
    alb_nir = jnp.where(inc_nir > 0.0, swupw_top[1] / jnp.maximum(inc_nir, 1e-30), 0.0)

    # Turbulent fluxes
    if flux_profile_type in (0, -1):
        shflx = shveg + means.shsoi
        etflx = etveg + means.etsoi
        lhflx = lhveg + means.lhsoi
    else:
        shflx = jnp.take(means.shair, layers.n_layers - 1, axis=0)
        etflx = jnp.take(means.etair, layers.n_layers - 1, axis=0)
        lhflx = means.lhtop

    # Air heat storage
    stflx_air = jnp.sum(jnp.where(layers.in_use, pad(means.stair), 0.0))

    # Energy balance checks
    rnet = swveg_vis + swveg_nir + means.swsoi[0] + means.swsoi[1] + lwveg + means.lwsoi
    radin = means.swskyb[0] + means.swskyd[0] + means.swskyb[1] + means.swskyd[1] + means.lwsky
    radout = (
        alb_vis * (means.swskyb[0] + means.swskyd[0])
        + alb_nir * (means.swskyb[1] + means.swskyd[1])
        + means.lwup
    )
    err2 = rnet - (radin - radout)
    avail = radin - radout - means.gsoi
    flux = shflx + lhflx + stflx_air + stflx_veg
    err3 = avail - flux
    swbeam_top = jnp.take(means.swbeam, layers.n_within, axis=0)
    swdwn_top = jnp.take(means.swdwn, layers.n_within, axis=0)
    lwdwn_top = jnp.take(means.lwdwn, layers.n_within, axis=0)
    lwupw_top = jnp.take(means.lwupw, layers.n_within, axis=0)
    radin_top = swbeam_top[0] + swbeam_top[1] + swdwn_top[0] + swdwn_top[1] + lwdwn_top
    radout_top = swupw_top[0] + swupw_top[1] + lwupw_top
    err4 = (radin_top - radout_top) - rnet

    # Sunlit and shaded canopy
    laisun = jnp.sum(fs * dpai_s)
    laisha = jnp.sum(fsh * dpai_s)
    swvegsun_vis = jnp.sum(sw_sun_vis * fs * dpai_s)
    swvegsun_nir = jnp.sum(sw_sun_nir * fs * dpai_s)
    swvegsha_vis = jnp.sum(sw_sha_vis * fsh * dpai_s)
    swvegsha_nir = jnp.sum(sw_sha_nir * fsh * dpai_s)
    shvegsun = jnp.sum(sh_sun * fs * dpai_s)
    shvegsha = jnp.sum(sh_sha * fsh * dpai_s)
    lhvegsun = jnp.sum(lh_sun * fs * dpai_s)
    lhvegsha = jnp.sum(lh_sha * fsh * dpai_s)
    gppvegsun = jnp.sum(ag_sun * fs * dpai_s * fracgreen)
    gppvegsha = jnp.sum(ag_sha * fsh * dpai_s * fracgreen)
    vcmax25sun = jnp.sum(vc_sun * fs * dpai_s)
    vcmax25sha = jnp.sum(vc_sha * fsh * dpai_s)
    gsvegsun = jnp.sum(gs_sun * fs * dpai_s)
    gsvegsha = jnp.sum(gs_sha * fsh * dpai_s)

    # Plant-area weighted temperatures and wind (0 without plant area)
    total = laisun + laisha
    wind = pad(leaves.wind)
    tair = pad(leaves.tair)
    total_safe = jnp.maximum(total, 1e-30)
    laisun_safe = jnp.maximum(laisun, 1e-30)
    laisha_safe = jnp.maximum(laisha, 1e-30)

    def mean_all(x):
        return jnp.where(total > 0.0, jnp.sum(x * dpai_s) / total_safe, 0.0)

    def mean_sun(x):
        return jnp.where(laisun > 0.0, jnp.sum(x * fs * dpai_s) / laisun_safe, 0.0)

    def mean_sha(x):
        return jnp.where(laisha > 0.0, jnp.sum(x * fsh * dpai_s) / laisha_safe, 0.0)

    # Fraction of the canopy at the minimum leaf water potential
    fracminlwp = jnp.sum(jnp.where(lp_m <= -2.0, dpai_s, 0.0))
    fracminlwp = jnp.where(lai + sai > 0.0, fracminlwp / (lai + sai), fracminlwp)

    return CanopyTotals(
        swveg=jnp.stack([swveg_vis, swveg_nir]),
        swvegsun=jnp.stack([swvegsun_vis, swvegsun_nir]),
        swvegsha=jnp.stack([swvegsha_vis, swvegsha_nir]),
        lwveg=lwveg,
        stflx_veg=stflx_veg,
        shveg=shveg,
        shvegsun=shvegsun,
        shvegsha=shvegsha,
        lhveg=lhveg,
        lhvegsun=lhvegsun,
        lhvegsha=lhvegsha,
        trveg=trveg,
        evveg=evveg,
        gppveg=gppveg,
        gppvegsun=gppvegsun,
        gppvegsha=gppvegsha,
        vcmax25veg=vcmax25veg,
        vcmax25sun=vcmax25sun,
        vcmax25sha=vcmax25sha,
        gsveg=gsveg,
        gsvegsun=gsvegsun,
        gsvegsha=gsvegsha,
        albcan=jnp.stack([alb_vis, alb_nir]),
        shflx=shflx,
        etflx=etflx,
        lhflx=lhflx,
        stflx_air=stflx_air,
        rnet=rnet,
        laisun=laisun,
        laisha=laisha,
        windveg=mean_all(wind),
        windvegsun=mean_sun(wind),
        windvegsha=mean_sha(wind),
        tlveg=mean_all(tl_m),
        tlvegsun=mean_sun(tl_sun),
        tlvegsha=mean_sha(tl_sha),
        taveg=mean_all(tair),
        tavegsun=mean_sun(tair),
        tavegsha=mean_sha(tair),
        fracminlwp=fracminlwp,
        lwp_mean=lp_m,
        errors=jnp.stack([err1, err2, err3, err4]),
    )


# ---------------------------------------------------------------------------
# Sunlit and shaded leaves merged — Fortran MLCanopyFluxes, lines 355-375
# ---------------------------------------------------------------------------


class MergedLeaves(NamedTuple):
    """Leaf state at the end of a step."""

    tleaf: Array  # (layer, leaf) leaf temperature, merged, K (tleaf_leaf)
    lwp: Array  # (layer, leaf) leaf water potential, merged, MPa (lwp_leaf)
    tleaf_hist: Array  # (layer, leaf) leaf temperature before the merge, K (tleaf_hist_leaf)
    lwp_hist: Array  # (layer, leaf) leaf water potential before the merge (lwp_hist_leaf)


def merge_sun_shade(
    layers: PatchLayers, dpai: Array, fracsun: Array, tleaf: Array, lwp: Array
) -> MergedLeaves:
    """Leaf temperature and water potential of the sunlit and shaded leaves replaced by their
    layer mean (weighted by the sunlit fraction) in layers with plant area: the state is
    prognostic and the sunlit fraction changes between steps. The values before the merge
    are kept for output (``*_hist``).

    Every slot is computed (layers without plant area and the padding unchanged); the caller
    keeps the layers in use. Fortran reference: MLCanopyFluxesMod.F90::MLCanopyFluxes
    (lines 355-375).
    """
    has_pai = (dpai > 0.0) & layers.in_use

    def merge(x):
        mean = x[:, 0] * fracsun + x[:, 1] * (1.0 - fracsun)
        return jnp.stack(
            [jnp.where(has_pai, mean, x[:, 0]), jnp.where(has_pai, mean, x[:, 1])], axis=1
        )

    return MergedLeaves(tleaf=merge(tleaf), lwp=merge(lwp), tleaf_hist=tleaf, lwp_hist=lwp)
