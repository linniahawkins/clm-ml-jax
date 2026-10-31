"""The soil routines of the legacy time step as adapters to ``clm_ml.physics.soil`` (Phase 7.2,
process 2; transition only: they go when the containers leave the step, process 10).

Each adapter keeps the legacy routine's name and arguments, takes the soil column's arrays
from the containers (legacy layout: soil level ``j`` at index ``j``, except
``h2osoi_liq_col``/``h2osoi_ice_col`` at ``j - 1``), calls the process on 0-based levels and
writes the results back into the same entries. ``DECLARED`` lists the container fields each
adapter reads and writes (checked by ``tests/clm_ml/test_physics_soil.py``).
"""

from __future__ import annotations

import numpy as np

from clm_ml.physics import soil
from clm_src_main.SiteType import site_or_modules

#: adapter -> (fields read, fields written), as ``container.field``
DECLARED = {
    "clmData": (
        ("site.dz_col", "site.nbedrock_col", "soilstate.watsat_col", "canopystate.elai_patch",
         "canopystate.esai_patch", "waterstatebulk.h2osoi_vol_col",
         "waterstatebulk.h2osoi_liq_col", "waterstatebulk.h2osoi_ice_col"),
        ("canopystate.elai_patch", "canopystate.esai_patch", "waterstatebulk.h2osoi_vol_col",
         "waterstatebulk.h2osoi_liq_col", "waterstatebulk.h2osoi_ice_col"),
    ),
    "SoilAlbedo": (
        ("site.isoicol_col", "waterstatebulk.h2osoi_vol_col", "surfalb.albgrd_col",
         "surfalb.albgri_col"),
        ("surfalb.albgrd_col", "surfalb.albgri_col"),
    ),
    "calc_soilevap_resis": (
        ("site.dz_col", "soilstate.watsat_col", "soilstate.sucsat_col", "soilstate.bsw_col",
         "soilstate.dsl_col", "soilstate.soilresis_col", "temperature.t_soisno_col",
         "waterstatebulk.h2osoi_liq_col", "waterstatebulk.h2osoi_ice_col"),
        ("soilstate.dsl_col", "soilstate.soilresis_col"),
    ),
    "SoilThermProp": (
        ("site.dz_col", "site.z_col", "site.zi_col", "site.nbedrock_col",
         "temperature.t_soisno_col", "waterstatebulk.h2osfc_col", "water.h2osno_col",
         "waterstatebulk.h2osoi_liq_col", "waterstatebulk.h2osoi_ice_col",
         "soilstate.tkmg_col", "soilstate.tkdry_col", "soilstate.csol_col",
         "soilstate.watsat_col", "soilstate.thk_col"),
        ("soilstate.thk_col",),
    ),
    "SoilWater": (
        ("site.dz_col", "site.nbedrock_col", "waterstatebulk.h2osoi_liq_col",
         "soilstate.watsat_col", "soilstate.hksat_col", "soilstate.sucsat_col",
         "soilstate.bsw_col", "soilstate.hk_l_col", "soilstate.smp_l_col"),
        ("soilstate.hk_l_col", "soilstate.smp_l_col"),
    ),
}  # fmt: skip
# SoilTemperature computes the thermal properties again (SoilThermProp), as Fortran
DECLARED["SoilTemperature"] = (
    tuple(sorted({*DECLARED["SoilThermProp"][0], "mlcanopy.gsoi_soil"})),
    ("soilstate.thk_col", "temperature.t_soisno_col"),
)


def _levels(n=None):
    """Legacy entries of soil levels 1..n (``nlevgrnd`` by default)."""
    from clm_src_main import clm_varpar

    return slice(1, (clm_varpar.nlevgrnd if n is None else n) + 1)


def clmData(
    record,
    begp: int,
    endp: int,
    begc: int,
    endc: int,
    soilstate_inst,
    waterstatebulk_inst,
    canopystate_inst,
    surfalb_inst,
    site=None,
):
    """Install a CLM history record: LAI and SAI of every patch, soil water of every column.

    Fortran reference: clmDataMod.F90::clmData (``soil.soil_water_from_history``).
    Returns ``(waterstatebulk_inst, canopystate_inst, surfalb_inst)`` (the last unchanged).
    """
    from clm_src_main import clm_varpar
    from offline_driver.clmSoilOptionMod import clm_phys, nlev_soil_adjust

    site = site_or_modules(site)
    lev, n = _levels(), clm_varpar.nlevgrnd
    # One gridcell value for every patch
    elai = canopystate_inst.elai_patch.at[begp : endp + 1].set(record.elai)
    esai = canopystate_inst.esai_patch.at[begp : endp + 1].set(record.esai)
    vol = waterstatebulk_inst.h2osoi_vol_col
    liq = waterstatebulk_inst.h2osoi_liq_col
    ice = waterstatebulk_inst.h2osoi_ice_col
    for c in range(begc, endc + 1):
        ci = c - begc
        w = soil.soil_water_from_history(
            record.h2osoi_clm45,
            record.h2osoi_clm50,
            record.h2osoi_factor,
            site.dz_col[c, lev],
            soilstate_inst.watsat_col[c, lev],
            site.nbedrock_col[c],
            clm_phys=clm_phys,
            n_soil=clm_varpar.nlevsoi,
            n_adjust=nlev_soil_adjust,
        )
        # h2osoi_vol_col has nlevgrnd entries for levels 1..nlevgrnd (index 0 unused): the
        # bottom level has no entry (the legacy element writes past the end were dropped)
        vol = vol.at[ci, 1:].set(w.h2osoi_vol[: vol.shape[1] - 1])
        liq = liq.at[ci, 0:n].set(w.h2osoi_liq)
        ice = ice.at[ci, 0:n].set(w.h2osoi_ice)
    return (
        waterstatebulk_inst._replace(h2osoi_vol_col=vol, h2osoi_liq_col=liq, h2osoi_ice_col=ice),
        canopystate_inst._replace(elai_patch=elai, esai_patch=esai),
        surfalb_inst,
    )


def SoilAlbedo(bounds, num_nourbanc, filter_nourbanc, waterstatebulk_inst, surfalb_inst, site=None):
    """Soil albedo of the non-urban columns (``soil.soil_albedo``).

    Fortran reference: SurfaceAlbedoMod.F90::SoilAlbedo.
    """
    from clm_src_biogeophys import SurfaceAlbedoMod
    from clm_src_main.clm_varpar import numrad

    soil_color = site.isoicol_col if site is not None else SurfaceAlbedoMod.isoicol
    bands = slice(1, numrad + 1)
    albgrd, albgri = surfalb_inst.albgrd_col, surfalb_inst.albgri_col
    for f in range(num_nourbanc):
        c = int(np.asarray(filter_nourbanc)[f])  # structure: static
        ci = c - bounds.begc
        colour = soil_color[ci] - 1  # 1-based class (data) into the tables
        direct, diffuse = soil.soil_albedo(
            waterstatebulk_inst.h2osoi_vol_col[ci, 1],
            SurfaceAlbedoMod.albsat[colour, bands],
            SurfaceAlbedoMod.albdry[colour, bands],
        )
        albgrd = albgrd.at[c, bands].set(direct)
        albgri = albgri.at[c, bands].set(diffuse)
    return surfalb_inst._replace(albgrd_col=albgrd, albgri_col=albgri)


def calc_soilevap_resis(
    bounds,
    num_nolakec,
    filter_nolakec,
    soilstate_inst,
    waterstatebulk_inst,
    temperature_inst,
    site=None,
):
    """Soil evaporative resistance of the non-lake columns (``soil.soil_evaporative_resistance``).

    Fortran reference: SurfaceResistanceMod.F90::calc_soilevap_resis, calc_soil_resistance_sl14.
    """
    dz = site_or_modules(site).dz_col
    s, w = soilstate_inst, waterstatebulk_inst
    dsl, soilresis = s.dsl_col, s.soilresis_col
    for f in range(num_nolakec):
        c = int(filter_nolakec[f])
        ci = c - bounds.begc
        dsl_c, soilresis_c = soil.soil_evaporative_resistance(
            w.h2osoi_liq_col[ci, 0],
            w.h2osoi_ice_col[ci, 0],
            temperature_inst.t_soisno_col[ci, 1],
            dz[c, 1],
            s.watsat_col[c, 1],
            s.sucsat_col[c, 1],
            s.bsw_col[c, 1],
        )
        dsl = dsl.at[c].set(dsl_c)
        soilresis = soilresis.at[c].set(soilresis_c)
    return s._replace(dsl_col=dsl, soilresis_col=soilresis)


def SoilThermProp(
    bounds,
    num_nolakec,
    filter_nolakec,
    tk,
    cv,
    tk_h2osfc,
    temperature_inst,
    waterdiagnosticbulk_inst,
    waterstatebulk_inst,
    water_inst,
    soilstate_inst,
    site=None,
):
    """Soil thermal properties of the non-lake columns (``soil.soil_thermal_properties``).

    Fortran reference: MLSoilTemperatureMod.F90::SoilThermProp. No snow layers: ``clm_drv``
    sets ``snl = 0`` before (Fortran ``clm_drv``). ``tk``, ``cv`` (column, legacy level) and
    ``tk_h2osfc`` (column) are filled and returned:
    ``(soilstate_inst, temperature_inst, tk, cv, tk_h2osfc)``.
    """
    site = site_or_modules(site)
    lev = _levels()
    s, w = soilstate_inst, waterstatebulk_inst
    thk = s.thk_col
    for f in range(num_nolakec):
        c = int(np.asarray(filter_nolakec)[f])  # structure: static
        props = soil.soil_thermal_properties(
            temperature_inst.t_soisno_col[c, lev],
            w.h2osoi_liq_col[c, 0 : lev.stop - 1],
            w.h2osoi_ice_col[c, 0 : lev.stop - 1],
            w.h2osfc_col[c],
            water_inst.h2osno_col[c],
            site.dz_col[c, lev],
            site.z_col[c, lev],
            site.zi_col[c, lev],
            s.watsat_col[c, lev],
            s.tkmg_col[c, lev],
            s.tkdry_col[c, lev],
            s.csol_col[c, lev],
            site.nbedrock_col[c],
        )
        thk = thk.at[c, lev].set(props.thk)
        tk = tk.at[c, lev].set(props.tk)
        cv = cv.at[c, lev].set(props.cv)
        tk_h2osfc = tk_h2osfc.at[c].set(props.tk_h2osfc)
    return s._replace(thk_col=thk), temperature_inst, tk, cv, tk_h2osfc


def SoilWater(
    bounds, num_hydrologyc, filter_hydrologyc, soilstate_inst, waterstatebulk_inst, site=None
):
    """Hydraulic conductivity and matric potential of the hydrology columns
    (``soil.soil_hydraulic_properties``, on the ``nlevsoi`` soil levels).

    Fortran reference: SoilWaterMovementMod.F90::SoilWater, soilwater_moisture_form.
    """
    from clm_src_main import clm_varpar

    site = site_or_modules(site)
    n_soil = clm_varpar.nlevsoi
    lev = _levels(n_soil)
    s = soilstate_inst
    hk_l, smp_l = s.hk_l_col, s.smp_l_col
    for fc in range(num_hydrologyc):
        c = int(np.asarray(filter_hydrologyc)[fc])  # structure: static
        ci = c - bounds.begc
        hk, smp = soil.soil_hydraulic_properties(
            waterstatebulk_inst.h2osoi_liq_col[ci, 0:n_soil],
            site.dz_col[c, lev],
            s.watsat_col[c, lev],
            s.hksat_col[c, lev],
            s.sucsat_col[c, lev],
            s.bsw_col[c, lev],
            site.nbedrock_col[c],
            hk_l[c, lev],
            smp_l[c, lev],
        )
        hk_l = hk_l.at[c, lev].set(hk)
        smp_l = smp_l.at[c, lev].set(smp)
    return s._replace(hk_l_col=hk_l, smp_l_col=smp_l)


def SoilTemperature(
    bounds,
    num_nolakec,
    filter_nolakec,
    soilstate_inst,
    temperature_inst,
    waterdiagnosticbulk_inst,
    waterstatebulk_inst,
    water_inst,
    mlcanopy_inst,
    dtime,
    site=None,
):
    """Soil temperature of the non-lake columns after one step (``soil.soil_temperature``),
    with the thermal properties computed again (``SoilThermProp``, as Fortran).

    Fortran reference: MLSoilTemperatureMod.F90::SoilTemperature. Standalone only (p == c).
    Returns ``(temperature_inst, soilstate_inst, {patch: energy conservation error})``.
    """
    import jax.numpy as jnp

    from clm_src_main import clm_varpar

    site = site_or_modules(site)
    lev = _levels()
    nc = bounds.endc + 1
    shape = (nc, clm_varpar.nlevgrnd + 1)
    soilstate_inst, temperature_inst, tk, cv, _ = SoilThermProp(
        bounds,
        num_nolakec,
        filter_nolakec,
        jnp.zeros(shape, dtype=jnp.float64),
        jnp.zeros(shape, dtype=jnp.float64),
        jnp.zeros(nc, dtype=jnp.float64),
        temperature_inst,
        waterdiagnosticbulk_inst,
        waterstatebulk_inst,
        water_inst,
        soilstate_inst,
        site=site,
    )
    t_soisno = temperature_inst.t_soisno_col
    energy_error = {}
    for f in range(num_nolakec):
        c = int(np.asarray(filter_nolakec)[f])  # structure: static
        p = c  # standalone: one patch per column
        t_new, energy_error[p] = soil.soil_temperature(
            t_soisno[c, lev], tk[c, lev], cv[c, lev], site.z_col[c, lev],
            mlcanopy_inst.gsoi_soil[p], dtime,
        )  # fmt: skip
        t_soisno = t_soisno.at[c, lev].set(t_new)
    return temperature_inst._replace(t_soisno_col=t_soisno), soilstate_inst, energy_error
