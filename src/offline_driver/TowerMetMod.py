"""
JAX translation of TowerMetMod Fortran module.

Read tower meteorology forcing for offline CLM simulations.

The whole tower record needed by a run is read once before the time loop
(:func:`read_tower_met_series`); :func:`derive_tower_met` derives missing
quantities (specific humidity, longwave radiation) and partitions solar
radiation into direct and diffuse components per waveband;
:func:`TowerMetCurr` and :func:`TowerMetNext` install one derived record into
the CLM and CLMml forcing containers.

Original Fortran module: TowerMetMod
Fortran lines 1-290
"""

from typing import List, NamedTuple, Tuple

import jax.numpy as jnp
import netCDF4 as nc  # noqa: F401
from jax import Array

from clm_src_biogeophys.FrictionVelocityMod import frictionvel_type  # noqa: F401
from clm_src_main import GridcellType  # noqa: F401
from clm_src_main.abortutils import endrun, handle_err  # noqa: F401
from clm_src_main.atm2lndType import atm2lnd_type  # noqa: F401
from clm_src_main.clm_varcon import sb  # noqa: F401
from clm_src_main.clm_varpar import inir, ivis  # noqa: F401
from clm_src_main.PatchType import patch  # noqa: F401
from clm_src_main.wateratm2lndBulkType import wateratm2lndbulk_type  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from multilayer_canopy.MLclm_varcon import mmdry, mmh2o  # noqa: F401
from multilayer_canopy.MLWaterVaporMod import SatVap  # noqa: F401
from offline_driver.TowerDataMod import tower_ht, tower_lat, tower_lon  # noqa: F401

# ---------------------------------------------------------------------------
# Private: atmospheric CO2 concentration
# ---------------------------------------------------------------------------


def TowerMetCO2() -> float:
    """
    Return the atmospheric CO2 concentration (ppm).

    Mirrors Fortran function ``TowerMetCO2`` (lines 35-48).
    The commented-out alternative value of 367 ppm is preserved.

    Returns:
        Atmospheric CO2 concentration in ppm (umol/mol).
    """
    #   return 367.0    # Fortran line 44 (commented out)
    return 383.0  # Fortran line 45


# ---------------------------------------------------------------------------
# Private: solar radiation partitioning
# ---------------------------------------------------------------------------


def TowerMetSolarRad(
    fsds: float,
) -> Tuple[float, float, float, float]:
    """
    Partition total solar radiation into direct and diffuse components
    for visible and near-infrared wavebands.

    Mirrors Fortran subroutine ``TowerMetSolarRad`` (lines 50-100).

    Total solar radiation is split 50/50 between the visible and
    near-infrared wavebands (Fortran lines 82-83). Within each waveband
    the direct beam fraction is computed from a third-order polynomial
    in the waveband irradiance and clamped to ``[0.01, 0.99]``:

    .. code-block:: none

        fsds_vis = 0.5 * fsds
        rvis = a0 + fsds_vis*(a1 + fsds_vis*(a2 + fsds_vis*a3))
        rvis = clamp(rvis, 0.01, 0.99)

        fsds_nir = 0.5 * fsds
        rnir = b0 + fsds_nir*(b1 + fsds_nir*(b2 + fsds_nir*b3))
        rnir = clamp(rnir, 0.01, 0.99)

    Args:
        fsds: Total incident solar radiation (W/m2).

    Returns:
        Tuple of ``(forc_solad_vis, forc_solai_vis, forc_solad_nir,
        forc_solai_nir)`` in W/m2.
    """
    # Polynomial coefficients — Fortran lines 68-76
    a0 = 0.17639
    a1 = 0.00380
    a2 = -9.0039e-6
    a3 = 8.1351e-9
    b0 = 0.29548
    b1 = 0.00504
    b2 = -1.4957e-5
    b3 = 1.4881e-8

    # Visible waveband — Fortran lines 82-86
    fsds_vis = 0.5 * fsds
    rvis = a0 + fsds_vis * (a1 + fsds_vis * (a2 + fsds_vis * a3))
    rvis = float(jnp.maximum(0.01, jnp.minimum(0.99, rvis)))

    # Near-infrared waveband — Fortran lines 88-90
    fsds_nir = 0.5 * fsds
    rnir = b0 + fsds_nir * (b1 + fsds_nir * (b2 + fsds_nir * b3))
    rnir = float(jnp.maximum(0.01, jnp.minimum(0.99, rnir)))

    # Direct beam and diffuse per waveband — Fortran lines 92-95
    forc_solad_vis = fsds_vis * rvis
    forc_solai_vis = fsds_vis * (1.0 - rvis)
    forc_solad_nir = fsds_nir * rnir
    forc_solai_nir = fsds_nir * (1.0 - rnir)

    return forc_solad_vis, forc_solai_vis, forc_solad_nir, forc_solai_nir


# ---------------------------------------------------------------------------
# Private: atmospheric emissivity
# ---------------------------------------------------------------------------


def TowerMetEmiss(eair: float, tair: float) -> float:
    """
    Compute atmospheric emissivity from vapor pressure and temperature.

    Used when longwave radiation is missing from the tower forcing file.
    Mirrors Fortran function ``TowerMetEmiss`` (lines 102-120).

    .. code-block:: none

        emiss = 0.7 + 5.95e-5 * 0.01 * eair * exp(1500 / tair)

    Args:
        eair: Atmospheric vapor pressure (Pa).
        tair: Atmospheric temperature (K).

    Returns:
        Dimensionless atmospheric emissivity.
    """
    return 0.7 + 5.95e-5 * 0.01 * eair * float(jnp.exp(1500.0 / tair))


# ---------------------------------------------------------------------------
# Public: read forcing for the current CLM time step
# ---------------------------------------------------------------------------


class TowerMetRaw(NamedTuple):
    """One record of the tower netCDF file, as read (``-999.0`` = variable absent)."""

    zbot: float  # Observational height (m)
    tbot: float  # Air temperature (K)
    rhbot: float  # Relative humidity (%)
    qbot: float  # Specific humidity (kg/kg)
    ubot: float  # Wind speed (m/s)
    fsdsbot: float  # Solar radiation (W/m2)
    fldsbot: float  # Longwave radiation (W/m2)
    pbot: float  # Atmospheric pressure (Pa)
    prect: float  # Precipitation (mm/s)


class TowerMet(NamedTuple):
    """Forcing derived from one tower record: what TowerMetCurr/TowerMetNext install.

    Wavebands are (vis, nir).
    """

    uref: float  # Wind speed (m/s)
    solad: Tuple[float, float]  # Direct beam solar radiation (W/m2)
    solai: Tuple[float, float]  # Diffuse solar radiation (W/m2)
    tref: float  # Air temperature (K)
    qref: float  # Specific humidity (kg/kg)
    pref: float  # Air pressure (Pa)
    lwsky: float  # Atmospheric longwave radiation (W/m2)
    rain: float  # Rainfall (kg H2O/m2/s)
    hgt: float  # Forcing (reference) height (m)
    co2_ppm: float  # Atmospheric CO2 (umol/mol)
    pco2: float  # CO2 partial pressure (Pa)
    po2: float  # O2 partial pressure (Pa)


def read_tower_met_series(ncfilename: str, nsteps: int) -> List[TowerMetRaw]:
    """
    Read the first ``nsteps`` records of a tower netCDF file.

    Same variables, defaults and value conversion as the per-step
    ``readTowerMet`` (Fortran lines 218-280), read once for the whole run so
    that no file is opened inside the time loop.
    """
    with nc.Dataset(ncfilename, "r") as ds:
        for name in ("FSDS", "PRECTmms", "TBOT", "WIND"):
            if name not in ds.variables:
                handle_err(-1, name)

        def series(name: str) -> list:
            if name not in ds.variables:
                return [-999.0] * nsteps
            values = ds.variables[name][:nsteps, 0, 0]
            if len(values) < nsteps:
                endrun(msg=f" TowerMet error: {ncfilename} has fewer than {nsteps} records")
            return [float(values[t]) for t in range(nsteps)]

        columns = {
            "zbot": series("ZBOT"),
            "tbot": series("TBOT"),
            "rhbot": series("RH"),
            "qbot": series("QBOT"),
            "ubot": series("WIND"),
            "fsdsbot": series("FSDS"),
            "fldsbot": series("FLDS"),
            "pbot": series("PSRF"),
            "prect": series("PRECTmms"),
        }
    return [TowerMetRaw(**{k: v[t] for k, v in columns.items()}) for t in range(nsteps)]


def derive_tower_met(raw: TowerMetRaw, tower_height: float) -> TowerMet:
    """
    Derive the forcing for one tower record.

    The calculations shared by Fortran ``TowerMetCurr`` (lines 122-215) and
    ``TowerMetNext`` (lines 216-280), in the same order:

    - shortwave partitioned into direct/diffuse vis/nir (``TowerMetSolarRad``)
    - forcing height from the tower table (30 m if missing)
    - pressure 101325 Pa if missing
    - specific humidity from relative humidity when available
    - longwave from atmospheric emissivity when missing
    - CO2 from ``TowerMetCO2``, O2 = 0.209 * pressure
    """
    fsds = float(jnp.maximum(raw.fsdsbot, 0.0))
    solad_vis, solai_vis, solad_nir, solai_nir = TowerMetSolarRad(fsds)

    hgt = float(tower_height)
    if round(hgt) == -999:
        hgt = 30.0
    pref = raw.pbot
    if round(pref) == -999:
        pref = 101325.0

    tref = raw.tbot
    qref = raw.qbot
    if round(raw.rhbot) != -999:
        esat, _ = SatVap(tref)
        eair = (raw.rhbot / 100.0) * esat
        qref = mmh2o / mmdry * eair / (pref - (1.0 - mmh2o / mmdry) * eair)
    elif round(qref) != -999:
        eair = qref * pref / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * qref)
    else:
        endrun(msg=" TowerMet error: rhref and qref not valid")
        eair = 0.0  # Unreachable; satisfies type checker

    lwsky = raw.fldsbot
    if round(lwsky) == -999:
        emiss = TowerMetEmiss(eair, tref)
        lwsky = emiss * sb * tref**4

    co2 = TowerMetCO2()
    return TowerMet(
        uref=raw.ubot,
        solad=(solad_vis, solad_nir),
        solai=(solai_vis, solai_nir),
        tref=tref,
        qref=float(qref),
        pref=pref,
        lwsky=float(lwsky),
        rain=raw.prect,
        hgt=hgt,
        co2_ppm=co2,
        pco2=(co2 / 1.0e6) * pref,
        po2=0.209 * pref,
    )


def TowerMetCurr(
    met: TowerMet,
    it: int,
    begp: int,
    endp: int,
    atm2lnd_inst: atm2lnd_type,
    wateratm2lndbulk_inst: wateratm2lndbulk_type,
    frictionvel_inst: frictionvel_type,
) -> Tuple[atm2lnd_type, wateratm2lndbulk_type, frictionvel_type]:
    """
    Install the forcing for the current CLM time step.

    Mirrors Fortran subroutine ``TowerMetCurr`` (lines 122-215). The file read
    and the derivations are done beforehand (:func:`read_tower_met_series`,
    :func:`derive_tower_met`); this routine copies ``met`` into the CLM forcing
    containers:

    - Grid-cell level: ``forc_u_grc``, ``forc_v_grc`` (= 0),
      ``forc_solai_grc``, ``forc_pco2_grc``, ``forc_po2_grc``.
    - Column level: ``forc_t``, ``forc_q``, ``forc_pbot``, ``forc_lwrad``,
      ``forc_rain``, ``forc_snow`` (= 0), ``forc_solad``.
    - Patch level: ``forc_hgt_u_patch``.

    Tower latitude and longitude are written into ``grc.latdeg`` and ``grc.londeg``.

    Args:
        met: Derived forcing for this time step.
        it: Tower site index into ``TowerDataMod`` arrays.
        begp: First patch index.
        endp: Last patch index.
        atm2lnd_inst, wateratm2lndbulk_inst, frictionvel_inst: Forcing containers.

    Returns:
        Tuple of updated ``(atm2lnd_inst, wateratm2lndbulk_inst, frictionvel_inst)``.
    """
    forc_u: Array = atm2lnd_inst.forc_u_grc
    forc_v: Array = atm2lnd_inst.forc_v_grc
    forc_solad: Array = atm2lnd_inst.forc_solad_downscaled_col
    forc_solai: Array = atm2lnd_inst.forc_solai_grc
    forc_pco2: Array = atm2lnd_inst.forc_pco2_grc
    forc_po2: Array = atm2lnd_inst.forc_po2_grc
    forc_t: Array = atm2lnd_inst.forc_t_downscaled_col
    forc_pbot: Array = atm2lnd_inst.forc_pbot_downscaled_col
    forc_lwrad: Array = atm2lnd_inst.forc_lwrad_downscaled_col
    forc_q: Array = wateratm2lndbulk_inst.forc_q_downscaled_col
    forc_rain: Array = wateratm2lndbulk_inst.forc_rain_downscaled_col
    forc_snow: Array = wateratm2lndbulk_inst.forc_snow_downscaled_col
    forc_hgt_u: Array = frictionvel_inst.forc_hgt_u_patch

    for p in range(begp, endp + 1):  # Fortran: do p = begp, endp
        c = int(patch.column[p])
        g = int(patch.gridcell[p])

        forc_u = forc_u.at[g].set(met.uref)
        forc_v = forc_v.at[g].set(0.0)
        forc_solad = forc_solad.at[c, ivis].set(met.solad[0])
        forc_solad = forc_solad.at[c, inir].set(met.solad[1])
        forc_solai = forc_solai.at[g, ivis].set(met.solai[0])
        forc_solai = forc_solai.at[g, inir].set(met.solai[1])
        forc_t = forc_t.at[c].set(met.tref)
        forc_q = forc_q.at[c].set(met.qref)
        forc_pbot = forc_pbot.at[c].set(met.pref)
        forc_lwrad = forc_lwrad.at[c].set(met.lwsky)
        forc_rain = forc_rain.at[c].set(met.rain)
        forc_snow = forc_snow.at[c].set(0.0)
        forc_hgt_u = forc_hgt_u.at[p].set(met.hgt)
        forc_pco2 = forc_pco2.at[g].set(met.pco2)
        forc_po2 = forc_po2.at[g].set(met.po2)

        # Tower latitude and longitude — Fortran lines 211-212
        assert GridcellType.grc.latdeg is not None
        assert GridcellType.grc.londeg is not None
        latdeg_jax = jnp.asarray(GridcellType.grc.latdeg)
        londeg_jax = jnp.asarray(GridcellType.grc.londeg)
        GridcellType.grc = GridcellType.grc._replace(
            latdeg=latdeg_jax.at[g].set(float(tower_lat[it])),
            londeg=londeg_jax.at[g].set(float(tower_lon[it])),
        )

    return (
        atm2lnd_inst._replace(
            forc_u_grc=forc_u,
            forc_v_grc=forc_v,
            forc_solad_downscaled_col=forc_solad,
            forc_solai_grc=forc_solai,
            forc_pco2_grc=forc_pco2,
            forc_po2_grc=forc_po2,
            forc_t_downscaled_col=forc_t,
            forc_pbot_downscaled_col=forc_pbot,
            forc_lwrad_downscaled_col=forc_lwrad,
        ),
        wateratm2lndbulk_inst._replace(
            forc_q_downscaled_col=forc_q,
            forc_rain_downscaled_col=forc_rain,
            forc_snow_downscaled_col=forc_snow,
        ),
        frictionvel_inst._replace(
            forc_hgt_u_patch=forc_hgt_u,
        ),
    )


def TowerMetNext(
    met_next: TowerMet,
    begp: int,
    endp: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Install the forcing for the next CLM time step.

    Mirrors Fortran subroutine ``TowerMetNext`` (lines 216-280). Required by the
    3-point time interpolation of atmospheric forcing (``met_type == 3``).
    Populates ``*_next_forcing`` fields in ``mlcanopy_type``; CO2 is stored in
    ppm, matching the CLMml convention (Fortran line 280).

    Args:
        met_next: Derived forcing for the next time step.
        begp: First patch index.
        endp: Last patch index.
        mlcanopy_inst: Multilayer canopy container.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    tref_next: Array = mlcanopy_inst.tref_next_forcing
    qref_next: Array = mlcanopy_inst.qref_next_forcing
    uref_next: Array = mlcanopy_inst.uref_next_forcing
    pref_next: Array = mlcanopy_inst.pref_next_forcing
    co2ref_next: Array = mlcanopy_inst.co2ref_next_forcing
    swskyb_next: Array = mlcanopy_inst.swskyb_next_forcing
    swskyd_next: Array = mlcanopy_inst.swskyd_next_forcing
    lwsky_next: Array = mlcanopy_inst.lwsky_next_forcing

    for p in range(begp, endp + 1):  # Fortran: do p = begp, endp
        uref_next = uref_next.at[p].set(met_next.uref)
        tref_next = tref_next.at[p].set(met_next.tref)
        qref_next = qref_next.at[p].set(met_next.qref)
        pref_next = pref_next.at[p].set(met_next.pref)
        lwsky_next = lwsky_next.at[p].set(met_next.lwsky)
        swskyb_next = swskyb_next.at[p, ivis].set(met_next.solad[0])
        swskyd_next = swskyd_next.at[p, ivis].set(met_next.solai[0])
        swskyb_next = swskyb_next.at[p, inir].set(met_next.solad[1])
        swskyd_next = swskyd_next.at[p, inir].set(met_next.solai[1])
        co2ref_next = co2ref_next.at[p].set(met_next.co2_ppm)

    return mlcanopy_inst._replace(
        tref_next_forcing=tref_next,
        qref_next_forcing=qref_next,
        uref_next_forcing=uref_next,
        pref_next_forcing=pref_next,
        co2ref_next_forcing=co2ref_next,
        swskyb_next_forcing=swskyb_next,
        swskyd_next_forcing=swskyd_next,
        lwsky_next_forcing=lwsky_next,
    )
