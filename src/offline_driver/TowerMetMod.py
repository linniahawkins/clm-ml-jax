"""
JAX translation of TowerMetMod Fortran module.

Read tower meteorology forcing for offline CLM simulations.

The tower records a run (or a chunk of it) needs are read before its time
loop (:func:`read_tower_met_series`); :func:`derive_tower_met` derives missing
quantities (specific humidity, longwave radiation) and partitions solar
radiation into direct and diffuse components per waveband;
:func:`TowerMetCurr` installs one derived record into the CLM forcing
containers. (Fortran ``TowerMetNext`` filled the multilayer canopy's
next-step forcing; that is now part of the forcing window passed to
``MLCanopyFluxes``.)

Original Fortran module: TowerMetMod
Fortran lines 1-290
"""

import dataclasses
import datetime
from typing import List, NamedTuple, Sequence, Tuple

import jax.numpy as jnp
import netCDF4 as nc  # noqa: F401
import numpy as np
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
from offline_driver import TowerDataMod  # tower table, read when called (not copied at import)

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
    """Forcing derived from one tower record (Fortran TowerMetCurr/TowerMetNext).

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


def read_tower_met_series(ncfilename: str, records: Sequence[int]) -> List[TowerMetRaw]:
    """
    Read the given 0-based records of a tower netCDF file.

    Same variables, defaults and value conversion as the per-step
    ``readTowerMet`` (Fortran lines 218-280). Only the span of records
    requested is read, so a run can read its forcing a chunk at a time and
    no file is opened inside the time loop.
    """
    if not records:
        return []
    lo, hi = min(records), max(records) + 1
    with nc.Dataset(ncfilename, "r") as ds:
        for name in ("FSDS", "PRECTmms", "TBOT", "WIND"):
            if name not in ds.variables:
                handle_err(-1, name)
        n_file = ds.dimensions["time"].size
        if lo < 0 or hi > n_file:
            endrun(
                msg=f" TowerMet error: records {lo}..{hi - 1} outside {ncfilename} (0..{n_file - 1})"
            )

        def span(name: str):
            if name not in ds.variables:
                return None
            return ds.variables[name][lo:hi, 0, 0]

        columns = {
            "zbot": span("ZBOT"),
            "tbot": span("TBOT"),
            "rhbot": span("RH"),
            "qbot": span("QBOT"),
            "ubot": span("WIND"),
            "fsdsbot": span("FSDS"),
            "fldsbot": span("FLDS"),
            "pbot": span("PSRF"),
            "prect": span("PRECTmms"),
        }

    def value(values, r: int) -> float:
        return -999.0 if values is None else float(values[r - lo])

    return [TowerMetRaw(**{k: value(v, r) for k, v in columns.items()}) for r in records]


def tower_first_record(ncfilename: str, start: datetime.datetime, step_seconds: int) -> int:
    """
    0-based record of a tower file at date-time ``start``, from its time axis.

    The time variable must have units "days since <date>" and a constant
    interval of ``step_seconds``.
    """
    with nc.Dataset(ncfilename, "r") as ds:
        time = ds.variables["time"]
        origin = datetime.datetime.fromisoformat(time.units.split("since")[1].strip())
        first_day = float(time[0])
        n_file = ds.dimensions["time"].size
    days = (start - origin).total_seconds() / 86400.0 - first_day
    record = round(days * 86400.0 / step_seconds)
    if abs(record * step_seconds - days * 86400.0) > 1.0 or not 0 <= record < n_file:
        endrun(msg=f" TowerMet error: {start} is not a record time in {ncfilename}")
    return record


@dataclasses.dataclass(frozen=True)
class TowerFiles:
    """Tower forcing split in time over several files (e.g. one per month), read as one
    series of records. Record 0 is the first record of the first file; each file must
    start one interval after the previous one ends (no gaps, no overlap)."""

    paths: tuple[str, ...]
    offsets: tuple[int, ...]  # series index of each file's first record
    n_records: int
    origin: datetime.datetime  # time of record 0
    step_seconds: int

    @classmethod
    def open(cls, paths: Sequence[str], step_seconds: int) -> "TowerFiles":
        paths = tuple(str(p) for p in paths)
        if not paths:
            endrun(msg=" TowerMet error: no tower forcing files")
        offsets, total, origin = [], 0, None
        for path in paths:
            with nc.Dataset(path, "r") as ds:
                time = ds.variables["time"]
                base = datetime.datetime.fromisoformat(time.units.split("since")[1].strip())
                first = base + datetime.timedelta(days=float(time[0]))
                size = ds.dimensions["time"].size
            if origin is None:
                origin = first
            expected = origin + datetime.timedelta(seconds=total * step_seconds)
            if abs((first - expected).total_seconds()) > 1.0:
                endrun(
                    msg=f" TowerMet error: {path} starts at {first}, but the previous file "
                    f"ends at {expected} (files must follow each other without gaps)"
                )
            offsets.append(total)
            total += size
        return cls(paths, tuple(offsets), total, origin, step_seconds)

    def first_record(self, start: datetime.datetime) -> int:
        """Series record at date-time ``start``."""
        seconds = (start - self.origin).total_seconds()
        record = round(seconds / self.step_seconds)
        if abs(record * self.step_seconds - seconds) > 1.0 or not 0 <= record < self.n_records:
            endrun(msg=f" TowerMet error: {start} is not a record time in {self.paths[0]} ...")
        return record

    def read(self, records: Sequence[int]) -> List[TowerMetRaw]:
        """The given series records, each read from the file holding it."""
        out: dict = {}
        bounds = list(self.offsets) + [self.n_records]
        for r in records:
            if not 0 <= r < self.n_records:
                endrun(msg=f" TowerMet error: record {r} outside the forcing files")
        for i, path in enumerate(self.paths):
            mine = [r for r in records if bounds[i] <= r < bounds[i + 1]]
            if mine:
                raws = read_tower_met_series(path, [r - bounds[i] for r in mine])
                out.update(zip(mine, raws))
        return [out[r] for r in records]


def as_tower_files(files, step_seconds: int = 1800) -> TowerFiles:
    """A :class:`TowerFiles` from one path, a list of paths, or a ``TowerFiles``."""
    if isinstance(files, TowerFiles):
        return files
    return TowerFiles.open([files] if isinstance(files, (str, bytes)) else files, step_seconds)


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


def set_tower_location(it: int, begp: int, endp: int) -> None:
    """Set the gridcell latitude and longitude to tower ``it``'s (Fortran TowerMetCurr,
    lines 211-212, did this at every step; the values are constant)."""
    assert GridcellType.grc.latdeg is not None
    assert GridcellType.grc.londeg is not None
    latdeg = jnp.asarray(GridcellType.grc.latdeg)
    londeg = jnp.asarray(GridcellType.grc.londeg)
    _gridcell = np.asarray(patch.gridcell)
    for p in range(begp, endp + 1):
        g = int(_gridcell[p])
        latdeg = latdeg.at[g].set(float(TowerDataMod.tower_lat[it]))
        londeg = londeg.at[g].set(float(TowerDataMod.tower_lon[it]))
    GridcellType.grc = GridcellType.grc._replace(latdeg=latdeg, londeg=londeg)


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

    _column = np.asarray(patch.column)  # patch structure: static (NumPy) indices
    _gridcell = np.asarray(patch.gridcell)
    for p in range(begp, endp + 1):  # Fortran: do p = begp, endp
        c = int(_column[p])
        g = int(_gridcell[p])

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
        # (Fortran also sets the tower latitude and longitude here, lines 211-212; they are
        # constant, so set_tower_location sets them once before the time loop.)

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
