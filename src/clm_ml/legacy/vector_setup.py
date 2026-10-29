"""
Setup of many patches at once (plan Phase 6, step 6): the per-patch legacy setup replaced by
array computations over all patches.

Setting a patch up with the legacy code (``setup_run``, then the first-step initialization)
takes about a second; a global domain has ~10^5 patches. Most of a patch's containers do not
depend on the patch at all, and the parts that do are a handful of formulas. So:

- **Prototypes.** One legacy setup per distinct *forcing key* (the gridcell's forcing files
  and measurement height, which set the forcing records and everything read from them: initial
  soil temperature and water, acclimation, first-step air profiles) and PFT. A prototype's
  containers, after the first-step initialization, give every field that does not depend on
  the patch's own inputs.
- **Per-patch fields** (found by varying one towers.yaml input at a time, plan Phase 6 step 6):
  the site data (latitude, longitude, soil colour, bedrock index), the soil hydraulic and
  thermal properties (sand, clay, organic matter), the root profile (bedrock depth and PFT),
  the canopy height, root biomass and leaf/stem beta profiles, the canopy's vertical structure
  (measurement height, canopy height, profiles, layer counts) and the canopy-air and leaf
  states set from the first step's forcing on that structure. These are computed here for all
  patches with the legacy formulas, in the same order of operations (NumPy for arithmetic,
  ``jax.numpy`` where the legacy code used it for ``exp``/``log``), so the result equals the
  legacy setup bit for bit (``tests/regression/test_vector_setup.py``).

Fortran reference: SoilStateInitTimeConstMod.F90 (SoilStateInitTimeConst), initVerticalMod.F90
(initVertical: nbedrock), CLMml_driver.F90 (TowerVeg), MLinitVerticalMod.F90
(initVerticalStructure, initVerticalProfiles, getPADparameters), MLMathToolsMod.F90 (beta
distribution functions).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------
# Beta distribution (MLMathToolsMod), vectorized
# ---------------------------------------------------------------------------------------


def _log_gamma(x):
    """``MLMathToolsMod.log_gamma_function`` on arrays (same operations)."""
    import jax.numpy as jnp

    coef = (
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    )
    x = np.asarray(x, dtype=float)
    y = x.copy()
    tmp = x + 5.5
    tmp = np.asarray((x + 0.5) * jnp.log(tmp) - tmp)
    ser = np.full_like(x, 1.000000000190015)
    for c in coef:
        y = y + 1.0
        ser = ser + c / y
    return np.asarray(tmp + jnp.log(2.5066282746310005 * ser / x))


def _beta_cf(a, b, x):
    """``MLMathToolsMod._beta_function_incomplete_cf`` on arrays: each element stops at its
    own iteration (modified Lentz, maxit 100, eps 3e-7, fpmin 1e-30)."""
    a, b, x = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(a, b, x))
    fpmin, eps = 1.0e-30, 3.0e-7
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < fpmin, fpmin, d)
    d = 1.0 / d
    h = d.copy()
    done = np.zeros(x.shape, dtype=bool)
    for m in range(1, 101):
        m2 = 2 * m
        aa = float(m) * (b - float(m)) * x / ((qam + float(m2)) * (a + float(m2)))
        d1 = 1.0 + aa * d
        d1 = np.where(np.abs(d1) < fpmin, fpmin, d1)
        c1 = 1.0 + aa / c
        c1 = np.where(np.abs(c1) < fpmin, fpmin, c1)
        d1 = 1.0 / d1
        h1 = h * d1 * c1
        aa = -(a + float(m)) * (qab + float(m)) * x / ((qap + float(m2)) * (a + float(m2)))
        d2 = 1.0 + aa * d1
        d2 = np.where(np.abs(d2) < fpmin, fpmin, d2)
        c2 = 1.0 + aa / c1
        c2 = np.where(np.abs(c2) < fpmin, fpmin, c2)
        d2 = 1.0 / d2
        delta = d2 * c2
        h2 = h1 * delta
        h, c, d = np.where(done, h, h2), np.where(done, c, c2), np.where(done, d, d2)
        done = done | (np.abs(delta - 1.0) < eps)
        if done.all():
            return h
    raise RuntimeError("beta_function_incomplete_cf error (no convergence)")


def beta_cdf(a, b, x):
    """``MLMathToolsMod.beta_distribution_cdf`` on arrays (same operations and branches)."""
    import jax.numpy as jnp

    a, b, x = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(a, b, x))
    edge = (x == 0.0) | (x == 1.0)
    xs = np.where(edge, 0.5, x)  # placeholder where bt = 0 (log of 0 never taken)
    bt = np.asarray(
        jnp.exp(
            _log_gamma(a + b)
            - _log_gamma(a)
            - _log_gamma(b)
            + a * np.asarray(jnp.log(xs))
            + b * np.asarray(jnp.log(1.0 - xs))
        )
    )
    bt = np.where(edge, 0.0, bt)
    direct = x < (a + 1.0) / (a + b + 2.0)
    out = np.empty_like(x)
    d, r = direct, ~direct  # each element's own branch only (the other need not converge)
    if d.any():
        out[d] = np.asarray(bt[d] * _beta_cf(a[d], b[d], x[d]) / a[d])
    if r.any():
        out[r] = np.asarray(1.0 - bt[r] * _beta_cf(b[r], a[r], 1.0 - x[r]) / b[r])
    return out


# ---------------------------------------------------------------------------------------
# Soil column (initVertical, SoilStateInitTimeConst)
# ---------------------------------------------------------------------------------------


def bedrock_index(zbed, zi, nlevsoi: int, zmin_bedrock: float):
    """``initVertical``: the bedrock layer index of each column (``zbed`` (C,), -999: none;
    ``zi`` (C, levels) interface depths)."""
    out = np.empty(len(zbed), dtype=int)
    for c in range(len(zbed)):
        jmin = 3
        for j in range(3, nlevsoi + 1):
            if zi[c, j - 1] < zmin_bedrock <= zi[c, j]:
                jmin = j
        nb = nlevsoi
        for j in range(jmin, nlevsoi + 1):
            if zi[c, j - 1] < zbed[c] <= zi[c, j]:
                nb = j
        out[c] = nb
    return out


def root_fractions(base, beta, zi, nbedrock, nlevsoi: int, nlevgrnd: int):
    """``SoilStateInitTimeConst``: the Jackson (1996) root profile per patch (``base``
    (P, levels): the patch's ``rootfr`` row before; ``beta`` (P,) its PFT's ``rootprof_beta``;
    ``zi`` (P, levels) its column's interfaces), roots below bedrock moved up."""
    m_to_cm = 1.0e2
    out = np.array(base, dtype=float)
    for p in range(len(beta)):
        b = float(beta[p])
        for j in range(1, nlevsoi + 1):
            out[p, j] = b ** (float(zi[p, j - 1]) * m_to_cm) - b ** (float(zi[p, j]) * m_to_cm)
        for j in range(nlevsoi + 1, nlevgrnd + 1):
            out[p, j] = 0.0
        nb = int(nbedrock[p])
        surplus = sum(float(out[p, j]) for j in range(nb + 1, nlevsoi + 1))
        for j in range(1, nb + 1):
            out[p, j] = float(out[p, j]) + surplus / float(nb)
        for j in range(nb + 1, nlevsoi + 1):
            out[p, j] = 0.0
    return out


def soil_properties(base: dict, sand, clay, organic, texture, z, nlevsoi: int, nlevgrnd: int):
    """``SoilStateInitTimeConst``: the soil hydraulic and thermal properties of each column
    (``base``: ``{field: (C, levels)}`` before, layers 1..nlevgrnd are set) from its sand and
    clay (percent; negative: from the texture class ``texture``) and organic matter (kg m-3);
    ``z`` (C, levels) node depths."""
    from clm_src_main.clm_varcon import csol_bedrock
    from offline_driver.SoilTexMod import (
        bsw_tex,
        clay_tex,
        hksat_tex,
        ntex,
        sand_tex,
        smpsat_tex,
        soil_tex,
        watsat_tex,
    )

    organic_max, zsapric, pcalpha, pcbeta = 130.0, 0.5, 0.5, 0.139
    out = {k: np.array(v, dtype=float) for k, v in base.items()}
    for c in range(len(sand)):
        om_frac = float(organic[c]) / organic_max
        if float(clay[c]) >= 0.0 and float(sand[c]) >= 0.0:
            tex, cl, s = 0, float(clay[c]), float(sand[c])
        else:
            tex = next((m + 1 for m in range(ntex) if texture[c] == soil_tex[m]), 0)
            if tex == 0:
                raise ValueError(f"soil type {texture[c]!r} not found (column {c})")
            cl, s = float(clay_tex[tex]) * 100.0, float(sand_tex[tex]) * 100.0
        for j in range(1, nlevgrnd + 1):
            if float(z[c, j]) > 0.5:
                om_frac = 0.0
            if j <= nlevsoi:
                out["cellsand_col"][c, j] = s
                out["cellclay_col"][c, j] = cl
                out["cellorg_col"][c, j] = om_frac * organic_max
            if tex == 0:
                watsat = 0.489 - 0.00126 * s
                sucsat = 10.0 * (10.0 ** (1.88 - 0.0131 * s))
                hksat = 0.0070556 * (10.0 ** (-0.884 + 0.0153 * s))
                bsw = 2.91 + 0.159 * cl
            else:
                watsat = float(watsat_tex[tex])
                sucsat = -float(smpsat_tex[tex])
                hksat = float(hksat_tex[tex]) / 60.0
                bsw = float(bsw_tex[tex])
            z_ratio = float(z[c, j]) / zsapric
            om_watsat = max(0.93 - 0.1 * z_ratio, 0.83)
            om_sucsat = min(10.3 - 0.2 * z_ratio, 10.1)
            om_hksat = max(0.28 - 0.2799 * z_ratio, hksat)
            om_b = min(2.7 + 9.3 * z_ratio, 12.0)
            watsat = (1.0 - om_frac) * watsat + om_watsat * om_frac
            sucsat = (1.0 - om_frac) * sucsat + om_sucsat * om_frac
            bsw = (1.0 - om_frac) * bsw + om_frac * om_b
            if om_frac > pcalpha:
                perc_norm = (1.0 - pcalpha) ** (-pcbeta)
                perc_frac = perc_norm * (om_frac - pcalpha) ** pcbeta
            else:
                perc_frac = 0.0
            uncon_frac = (1.0 - om_frac) + (1.0 - perc_frac) * om_frac
            if om_frac < 1.0:
                uncon_hksat = uncon_frac / (
                    (1.0 - om_frac) / hksat + ((1.0 - perc_frac) * om_frac) / om_hksat
                )
            else:
                uncon_hksat = 0.0
            hksat = uncon_frac * uncon_hksat + (perc_frac * om_frac) * om_hksat
            om_frac_therm = om_frac
            if j <= nlevsoi:
                bulk = 2700.0 * (1.0 - watsat)
                tkdry_min = (0.135 * bulk + 64.7) / (2700.0 - 0.947 * bulk)
                tkdry = (1.0 - om_frac_therm) * tkdry_min + om_frac_therm * 0.05
                tksol_min = (8.80 * s + 2.92 * cl) / (s + cl)
                tkm = (1.0 - om_frac_therm) * tksol_min + om_frac_therm * 0.25
                tkmg = tkm ** (1.0 - watsat)
            else:
                bulk = 2700.0
                tkdry = (0.135 * bulk + 64.7) / (2700.0 - 0.947 * bulk)
                tkmg = 3.0
            cvsol = ((2.128 * s + 2.385 * cl) / (s + cl)) * 1.0e6 if tex == 0 else 1.926e6
            if j <= nlevsoi:
                csol = (1.0 - om_frac_therm) * cvsol + om_frac_therm * 2.5e6
            else:
                csol = csol_bedrock
            for k, v in (("watsat_col", watsat), ("sucsat_col", sucsat), ("hksat_col", hksat),
                         ("bsw_col", bsw), ("tkdry_col", tkdry), ("tkmg_col", tkmg),
                         ("csol_col", csol)):  # fmt: skip
                out[k][c, j] = v
    return out


# ---------------------------------------------------------------------------------------
# Canopy (TowerVeg, getPADparameters, initVerticalStructure, initVerticalProfiles)
# ---------------------------------------------------------------------------------------

_PBETA_PFT = {1: (11.5, 3.5), **{i: (3.5, 2.0) for i in range(2, 12)}}
_PBETA_PFT.update({i: (2.5, 2.5) for i in range(12, 17)})


def pad_parameters(tower_lai, tower_sai, pft):
    """``TowerVeg`` + ``getPADparameters``: the leaf and stem beta parameters per patch:
    the tower's when all four are positive, otherwise the PFT default (``(P, 2)`` each)."""
    lai, sai = np.array(tower_lai, dtype=float), np.array(tower_sai, dtype=float)
    for p in range(len(pft)):
        if not (lai[p] > 0.0).all() or not (sai[p] > 0.0).all():
            d = _PBETA_PFT.get(int(pft[p]), (-999.0, -999.0))
            lai[p], sai[p] = d, d
    return lai, sai


def vertical_structure(zref, htop, pbeta_lai, pbeta_sai, nlayer_within, nlayer_above, opts,
                       base: dict):  # fmt: skip
    """``initVerticalStructure`` for all patches (Python floats per patch, as the legacy code;
    the beta CDF in one vectorized call). ``base``: the canopy fields before
    (``{field: (P, slots)}``); slots up to ``ncan`` are set. Returns the fields."""
    out = {k: np.array(v) for k, v in base.items()}
    n_p, nlev = len(zref), out["zw_profile"].shape[1] - 1
    geo = []
    for p in range(n_p):
        zr, zt = float(zref[p]), float(htop[p])
        out["zref_forcing"][p], out["ztop_canopy"][p] = zr, zt
        ztop_to_zref = zr - zt
        if nlayer_within[p] > 0 and nlayer_above[p] > 0:
            ntop = int(nlayer_within[p])
            dz_within = zt / float(ntop)
            nabove = int(nlayer_above[p])
            ncan = ntop + nabove
            dz_above = ztop_to_zref / float(nabove)
        else:
            dz_within = opts.dz_tall if zt > opts.dz_param else opts.dz_short
            ntop = round(zt / dz_within)
            dz_within = zt / float(ntop)
            dz_above = dz_within
            nabove = round(ztop_to_zref / dz_above)
            ncan = ntop + nabove
            dz_above = ztop_to_zref / float(nabove)
        if ncan > nlev:
            raise ValueError(f"patch {p} needs {ncan} canopy layers; max_layers is {nlev}")
        zw = [0.0] * (ncan + 1)
        zw[ntop] = zt
        for ic in range(ntop - 1, -1, -1):
            zw[ic] = zw[ic + 1] - dz_within
        if abs(zw[0]) > 1.0e-10:
            raise ValueError(f"patch {p}: zw(0) improperly defined")
        zw[0] = min(max(zw[0], 0.0), 0.0)
        zw[ncan] = zr
        for ic in range(ncan - 1, ntop, -1):
            zw[ic] = zw[ic + 1] - dz_above
        for ic in range(0, ncan + 1):
            out["zw_profile"][p, ic] = zw[ic]
        for ic in range(1, ncan + 1):
            out["dz_profile"][p, ic] = zw[ic] - zw[ic - 1]
            out["zs_profile"][p, ic] = 0.5 * (zw[ic] + zw[ic - 1])
        out["ntop_canopy"][p], out["ncan_canopy"][p] = ntop, ncan
        geo.append((ntop, ncan, zt, zw))
    # beta CDFs at every layer bound of every patch, in one call
    pts = [(p, ic) for p, (ntop, _, _, _) in enumerate(geo) for ic in range(1, ntop + 1)]
    zb = np.array([min(geo[p][3][ic - 1] / geo[p][2], 1.0) for p, ic in pts])
    zt_ = np.array([min(geo[p][3][ic] / geo[p][2], 1.0) for p, ic in pts])
    a_l = np.array([pbeta_lai[p][0] for p, _ in pts])
    b_l = np.array([pbeta_lai[p][1] for p, _ in pts])
    a_s = np.array([pbeta_sai[p][0] for p, _ in pts])
    b_s = np.array([pbeta_sai[p][1] for p, _ in pts])
    dlai_all = beta_cdf(a_l, b_l, zt_) - beta_cdf(a_l, b_l, zb)
    dsai_all = beta_cdf(a_s, b_s, zt_) - beta_cdf(a_s, b_s, zb)
    k = 0
    for p, (ntop, ncan, zt, zw) in enumerate(geo):
        dlai = [0.0] + [float(v) * 1.0 for v in dlai_all[k : k + ntop]]
        dsai = [0.0] + [float(v) * 1.0 for v in dsai_all[k : k + ntop]]
        k += ntop
        dpai_min = opts.dpai_min
        if abs(sum(dlai[ic] + dsai[ic] for ic in range(1, ntop + 1)) - 2.0) > 1.0e-6:
            raise ValueError(f"patch {p}: plant area profile does not sum to canopy total")
        lai_miss = sai_miss = 0.0
        for ic in range(1, ntop + 1):
            if dlai[ic] + dsai[ic] < dpai_min:
                lai_miss += dlai[ic]
                sai_miss += dsai[ic]
                dlai[ic] = dsai[ic] = 0.0
        if lai_miss > 0.0:
            lai_sum = sum(dlai[ic] for ic in range(1, ntop + 1))
            for ic in range(1, ntop + 1):
                dlai[ic] = dlai[ic] + lai_miss * (dlai[ic] / lai_sum)
        if sai_miss > 0.0:
            sai_sum = sum(dsai[ic] for ic in range(1, ntop + 1))
            for ic in range(1, ntop + 1):
                dsai[ic] = dsai[ic] + sai_miss * (dsai[ic] / sai_sum)
        nbot = 0
        for ic in range(ntop, 0, -1):
            if dlai[ic] + dsai[ic] > 0.0:
                nbot = ic
        if nbot == 0:
            raise ValueError(f"patch {p}: nbot not defined")
        out["nbot_canopy"][p] = nbot
        out["zbot_canopy"][p] = zw[nbot - 1]
        dlai += [0.0] * (ncan - ntop)
        dsai += [0.0] * (ncan - ntop)
        for ic in range(1, ncan + 1):
            out["dlai_frac_profile"][p, ic] = dlai[ic] / 1.0
            out["dsai_frac_profile"][p, ic] = dsai[ic] / 1.0
    return out


def vertical_profiles(base: dict, ncan, forcing: dict):
    """``initVerticalProfiles`` for all patches: the canopy-air and leaf states of layers
    1..ncan from the first step's forcing (``forcing``: per patch ``u, v, t, q, pbot, pco2``);
    ``base``: the fields before (``{field: (P, slots[, leaf])}``)."""
    import jax.numpy as jnp

    from multilayer_canopy.MLclm_varcon import mmdry, mmh2o
    from multilayer_canopy.MLclm_varpar import isha, isun

    out = {k: np.array(v) for k, v in base.items()}
    for p in range(len(ncan)):
        u, v = float(forcing["u"][p]), float(forcing["v"][p])
        t, q = float(forcing["t"][p]), float(forcing["q"][p])
        pb, pco2 = float(forcing["pbot"][p]), float(forcing["pco2"][p])
        wind = float(jnp.sqrt(u**2 + v**2))
        e = q * pb / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * q)
        for ic in range(1, int(ncan[p]) + 1):
            out["wind_profile"][p, ic] = wind
            out["tair_profile"][p, ic] = t
            out["eair_profile"][p, ic] = e
            out["cair_profile"][p, ic] = pco2 / pb * 1.0e6
            out["tleaf_leaf"][p, ic, isun] = t
            out["tleaf_leaf"][p, ic, isha] = t
            out["lwp_leaf"][p, ic, isun] = -0.1
            out["lwp_leaf"][p, ic, isha] = -0.1
            out["h2ocan_profile"][p, ic] = 0.0
        out["tg_soil"][p] = t
    return out


# ---------------------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------------------

#: Patch-dependent fields (found by varying one towers.yaml input at a time; every other
#: field of the first-step-initialized containers depends only on the forcing key and PFT)
SOIL_FIELDS = (
    "cellsand_col", "cellclay_col", "cellorg_col", "watsat_col", "sucsat_col", "hksat_col",
    "bsw_col", "tkmg_col", "tkdry_col", "csol_col",
)  # fmt: skip
STRUCTURE_FIELDS = (
    "zref_forcing", "ztop_canopy", "zbot_canopy", "ncan_canopy", "ntop_canopy", "nbot_canopy",
    "dlai_frac_profile", "dsai_frac_profile", "zs_profile", "zw_profile", "dz_profile",
)  # fmt: skip
PROFILE_FIELDS = (
    "tg_soil", "wind_profile", "tair_profile", "eair_profile", "cair_profile", "h2ocan_profile",
    "lwp_leaf", "tleaf_leaf",
)  # fmt: skip


def forcing_key(spec: dict, patch: dict) -> tuple:
    """What the forcing records of a patch's gridcell depend on: the files, the measurement
    height and data interval, and the location when shortwave is interpolated by the solar
    zenith angle (model step shorter than the data interval)."""
    nl, tw = patch["namelist"], patch["tower"]
    files = tuple(
        str(nl[k]) if not isinstance(nl[k], list) else tuple(nl[k])
        for k in ("fin_tower_files", "fin_clm", "fin_soil_adjust", "fin_vegetation",
                  "fin_soil_water")
    )  # fmt: skip
    interpolating = int(nl["dtstep_seconds"]) != int(tw["tower_time"]) * 60
    where = (tw["tower_lat"], tw["tower_lon"]) if interpolating else ()
    return files + (tw["tower_ht"], tw["tower_time"]) + where


#: towers.yaml inputs of each patch, in TowerDataMod form (-999: default)
TOWER_INPUTS = (
    "tower_lat", "tower_lon", "tower_tex", "tower_sand", "tower_clay", "tower_organic",
    "tower_isoicol", "tower_zbed", "tower_pft", "tower_canht", "tower_root", "tower_pbeta_lai",
    "tower_pbeta_sai",
)  # fmt: skip


def setup_batch(spec: dict):
    """Every patch of ``spec`` as an ``offline_executable.main.PatchBatch`` with its
    first-step-initialized containers computed for all patches at once (module docstring):
    one legacy setup per (forcing key, PFT) prototype, the patch-dependent fields by the
    vectorized legacy formulas (:func:`setup_arrays`)."""
    patches = spec["patches"]
    keys = [forcing_key(spec, ps) + (int(ps["tower"]["tower_pft"]),) for ps in patches]
    proto_specs: dict = {}
    for ps, key in zip(patches, keys):
        proto_specs.setdefault(key, {**spec, **ps})
    tower = {k: [ps["tower"][k] for ps in patches] for k in TOWER_INPUTS}
    within = [ps["varctl"].get("nlayer_within", 0) for ps in patches]
    above = [ps["varctl"].get("nlayer_above", 0) for ps in patches]
    batch = spec["batch"]
    return setup_arrays(
        proto_specs,
        keys,
        tower,
        within,
        above,
        batch["gridcell_index"],
        batch["weight"],
        int(batch["n_gridcells"]),
        [ps["tower_name"] for ps in patches],
    )


def setup_arrays(proto_specs: dict, keys, tower: dict, within, above, gridcell_index, weight,
                 n_gridcells: int, names, host: bool = False):  # fmt: skip
    """The batch from per-patch input arrays: ``proto_specs`` (prototype key -> a legacy
    patch spec set up as that prototype), ``keys`` the prototype of each patch, ``tower``
    ``{TOWER_INPUTS name: (P, ...)}`` the patches' own inputs, ``within``/``above`` explicit
    layer counts (0: model.yaml spacing). ``host``: keep the containers as NumPy arrays
    (large domains: placed on the devices block by block later)."""
    import jax
    import jax.numpy as jnp

    import offline_executable.main as M
    from clm_ml.legacy import worker
    from clm_src_main import pftconMod
    from clm_src_main.clm_varcon import zmin_bedrock
    from multilayer_canopy.MLclm_varctl import MLOptions

    protos: dict = {}
    for key in keys:
        if key not in protos:
            ps = proto_specs[key]
            worker.prepare_process(ps)
            run = worker.setup_legacy_run(ps)
            grid = M.patch_grid(run)
            insts, filt, inputs = M.PatchBatch.initialize_patch(run, grid)
            protos[key] = (run, grid, run.insts, insts, filt, inputs)
    # the soil layer counts are set by the first setup (read them only now: importing them
    # before copies -1)
    from clm_src_main import clm_varpar

    nlevgrnd, nlevsoi = clm_varpar.nlevgrnd, clm_varpar.nlevsoi
    order = list(protos)
    proto = np.array([order.index(k) for k in keys])
    plist = [protos[k] for k in order]

    def stacked(trees):  # (U, ...) numpy, then per patch (P, ...)
        u = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *trees)
        return jax.tree.map(lambda a: a[proto], u)

    pre, init = stacked([q[2] for q in plist]), stacked([q[3] for q in plist])
    xs = stacked([(i.now, i.met, i.clm_record, i.forcing_bef, i.forcing_next) for *_, i in plist])

    tw = tower
    f = lambda k: np.array(tw[k], dtype=float)  # noqa: E731
    pft = np.array(tw["tower_pft"], dtype=int)
    within = np.array(within, dtype=int)
    above = np.array(above, dtype=int)
    opts = MLOptions.from_codes(proto_specs[keys[0]]["varctl"])

    # site data
    site = init.site
    zi = np.asarray(site.zi_col)[:, 0, :]
    z = np.asarray(site.z_col)[:, 0, :]
    nbed = bedrock_index(f("tower_zbed"), zi, nlevsoi, zmin_bedrock)
    site = site._replace(
        latdeg_grc=f("tower_lat")[:, None],
        londeg_grc=f("tower_lon")[:, None],
        isoicol_col=np.array(tw["tower_isoicol"], dtype=np.asarray(site.isoicol_col).dtype)[
            :, None
        ],
        nbedrock_col=nbed.astype(np.asarray(site.nbedrock_col).dtype)[:, None],
    )

    # soil properties and roots
    soil = init.soilstate
    props = soil_properties(
        {k: np.asarray(getattr(soil, k))[:, 0, :] for k in SOIL_FIELDS},
        f("tower_sand"),
        f("tower_clay"),
        f("tower_organic"),
        list(tw["tower_tex"]),
        z,
        nlevsoi,
        nlevgrnd,
    )
    beta = np.asarray(pftconMod.pftcon.rootprof_beta)[pft]
    rootfr = root_fractions(
        np.asarray(soil.rootfr_patch)[:, 0, :], beta, zi, nbed, nlevsoi, nlevgrnd
    )
    soil = soil._replace(
        rootfr_patch=rootfr[:, None, :], **{k: v[:, None, :] for k, v in props.items()}
    )

    # canopy height, roots, profiles, vertical structure, initial canopy states
    from offline_driver import CLMml_driver

    htop_pft = CLMml_driver.HTOP_PFT
    canht = f("tower_canht")
    htop = np.where(canht > 0.0, canht, np.array([htop_pft[k] for k in pft]))
    root = f("tower_root")
    if (root <= 0.0).any():
        raise ValueError("TowerVeg ERROR: invalid root biomass (a patch has none)")
    lai_b, sai_b = pad_parameters(tw["tower_pbeta_lai"], tw["tower_pbeta_sai"], pft)
    ml, ml0 = init.mlcanopy, pre.mlcanopy
    pbeta_lai = np.array(np.asarray(ml0.pbeta_lai_canopy), dtype=float)
    pbeta_sai = np.array(np.asarray(ml0.pbeta_sai_canopy), dtype=float)
    pbeta_lai[:, 0, 1:3], pbeta_sai[:, 0, 1:3] = lai_b, sai_b
    zref = np.asarray(init.frictionvel.forc_hgt_u_patch)[:, 0]
    struct = vertical_structure(
        zref, htop, lai_b, sai_b, within, above, opts,
        {k: np.asarray(getattr(ml0, k))[:, 0] for k in STRUCTURE_FIELDS},
    )  # fmt: skip
    a2l, w2l = init.atm2lnd, init.wateratm2lndbulk
    forcing = {
        "u": np.asarray(a2l.forc_u_grc)[:, 0],
        "v": np.asarray(a2l.forc_v_grc)[:, 0],
        "t": np.asarray(a2l.forc_t_downscaled_col)[:, 0],
        "q": np.asarray(w2l.forc_q_downscaled_col)[:, 0],
        "pbot": np.asarray(a2l.forc_pbot_downscaled_col)[:, 0],
        "pco2": np.asarray(a2l.forc_pco2_grc)[:, 0],
    }
    prof = vertical_profiles(
        {k: np.asarray(getattr(ml0, k))[:, 0] for k in PROFILE_FIELDS},
        struct["ncan_canopy"],
        forcing,
    )
    fields = {**struct, **prof}
    ml = ml._replace(
        root_biomass_canopy=root[:, None],
        pbeta_lai_canopy=pbeta_lai,
        pbeta_sai_canopy=pbeta_sai,
        **{k: v.astype(np.asarray(getattr(ml, k)).dtype)[:, None] for k, v in fields.items()},
    )
    canopy = init.canopystate._replace(htop_patch=htop[:, None])
    insts = init._replace(site=site, soilstate=soil, mlcanopy=ml, canopystate=canopy)
    if not host:
        insts = jax.tree.map(jnp.asarray, insts)
        xs = jax.tree.map(jnp.asarray, xs)
    return M.PatchBatch(
        [plist[k][0] for k in proto],
        np.asarray(gridcell_index, dtype=int),
        np.asarray(weight, dtype=float),
        int(n_gridcells),
        list(names),
        [plist[k][1] for k in proto],
        init_stack=(insts, xs),
        init_filt=plist[0][4],
    )
