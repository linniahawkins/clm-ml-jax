"""
Grid domains (plan Phase 6b, step 2): gridcells and vegetated patches from a CLM surface
dataset, as CLM's subgrid setup does for its natural-vegetation and crop land units.

``run.yaml``::

    domain:
      type: grid
      min_patch_weight: 0.01        # of the gridcell's vegetated area (then renormalized)
      grid:
        surface_data: .../surfdata_0.9x1.25_hist_2000_16pfts_c251022.nc
        land_mask: .../<case>.clm2.h0a.2000-01.nc   # CLM history file: landmask == 1
        region: {lat: [25, 50], lon: [-125, -65]}   # optional box (degrees north / east)

What the loader takes from the surface data, and how (decisions 26-29 and the input
decisions in ``docs/design/global_input.md``):

- **Gridcells**: the surface-data cells with ``landmask == 1`` in the CLM history file (the
  CLM run whose soil moisture is prescribed) and some vegetation; their latitude and
  longitude (``LATIXY``, ``LONGXY``).
- **Patches**: one per PFT 1-16 with a non-zero area: natural PFTs 1-14 from
  ``PCT_NATVEG * PCT_NAT_PFT`` and the crop types 15-16 from ``PCT_CROP * PCT_CFT``. Bare
  ground (PFT 0) and the other land units are left out (decision 29): the weights are the
  patches' fractions of the gridcell's vegetated area. Patches below ``min_patch_weight``
  are dropped and the rest renormalized (as for towers).
- **Canopy top and bottom** (``MONTHLY_HEIGHT_TOP/BOT``): CLM's satellite-phenology values
  at the run's first step (:func:`clm_monthly_weights`); the layer structure is fixed at
  setup. (In the 2000 surface data the heights do not vary by month.) The bottom is kept in
  the domain but not used: CLM-ml's leaf-area profile runs from the ground to the top, and
  CLM uses the bottom height only for snow burial, which this model does not have (step 5).
- **Forcing height** per patch, CLM's rule: ``forcing_height + z0m + displa`` with
  ``z0m = z0mr * htop`` and ``displa = displar * htop`` (CLM's ``ZengWang2007`` roughness;
  ``forc_hgt_u_patch = forc_hgt_u + z0m + displa``, ``Sa_z = 30 m`` in CLM's data
  atmosphere), with ``z0mr`` and ``displar`` the PFT's ``z0m_to_height_ratio`` and
  ``displacement_to_height_ratio`` in params.yaml (e.g. the built-in ``clm6``).
- **Soil column** (one per patch, ``soil.columns: per_patch``): the top level of
  ``PCT_SAND``, ``PCT_CLAY``, ``ORGANIC``; ``SOIL_COLOR``; ``zbedrock``. Organic matter
  is capped at CLM-ml's ``organic_max`` (130 kg m-3): CLM caps the organic fraction at 1
  (``om_frac = min(organic / organic_max, 1)``), CLM-ml's setup formula has no cap (its
  towers never exceed it), and some peat soils of the surface data reach 318 kg m-3.
- **Root biomass**: not in a CLM surface dataset (CLM has no fine-root biomass in
  satellite-phenology mode); ``grid.root_biomass`` for every patch (default 500 g m-2, the
  value of every CLM-ml forest tower). Leaf and stem area profiles: the PFT defaults.

Fortran reference: CTSM surfrdMod.F90 (surfrd_veg_all), SatellitePhenologyMod.F90
(interpMonthlyVeg), FrictionVelocityMod.F90 / CanopyFluxesMod.F90 (forc_hgt_u_patch).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np

from clm_ml.io.yaml_schema import ConfigError

#: Days per month (CLM's interpMonthlyVeg, a no-leap table also in Gregorian runs)
NDAYPM = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: PFT index of each natural-vegetation (PCT_NAT_PFT 0-14) and crop (PCT_CFT 0-1) slot
N_NATPFT = 15  # natpft 0 (bare ground) to 14
CFT_PFTS = (15, 16)  # c3_crop, c3_irrigated

#: kg m-3: CLM-ml's organic_max (SoilStateInitTimeConstMod); organic matter is capped at it
ORGANIC_MAX = 130.0


@dataclasses.dataclass(frozen=True)
class RegionSection:
    lat: tuple[float, float] = (-90.0, 90.0)  # degrees north, [south, north]
    lon: tuple[float, float] = (-180.0, 360.0)  # degrees east, [west, east] (may wrap)


@dataclasses.dataclass(frozen=True)
class GridSection:
    """``run.yaml`` ``domain.grid``."""

    surface_data: str
    land_mask: str  # a CLM history file with ``landmask`` on the surface-data grid
    region: Optional[RegionSection] = None
    root_biomass: float = 500.0  # g m-2, every patch
    forcing_height: float = 30.0  # m above the zero-plane displacement + z0m (CLM: Sa_z)
    # Gridcells or patches beyond this many are dropped (for tests; null: all)
    max_gridcells: Optional[int] = None


def clm_monthly_weights(month: int, day: int) -> tuple[int, int, float, float]:
    """CLM's satellite-phenology time weights (``interpMonthlyVeg``) for a date: the two
    months (1-12) whose monthly values are interpolated and their weights. CLM evaluates
    them at the date one step after the end of the current step (``get_curr_date(offset =
    dtime)``); the weights change once a day, and a month's value is reached at the middle of
    its ``NDAYPM`` days.

    Fortran reference: SatellitePhenologyMod.F90, subroutine interpMonthlyVeg."""
    t = (day - 0.5) / NDAYPM[month - 1]
    it1 = int(t + 0.5)  # Fortran integer assignment (truncation; t + 0.5 > 0)
    it2 = it1 + 1
    m1, m2 = month + it1 - 1, month + it2 - 1
    if m1 < 1:
        m1 = 12
    if m2 > 12:
        m2 = 1
    w1 = (it1 + 0.5) - t
    return m1, m2, w1, 1.0 - w1


def satellite_phenology_date(step_start: dt.datetime, step_seconds: int) -> dt.datetime:
    """The date at which CLM evaluates the monthly vegetation for the step starting at
    ``step_start``: the end of the step plus one step (``get_curr_date`` with
    ``offset = dtime``)."""
    return step_start + dt.timedelta(seconds=2 * step_seconds)


def _in_region(lat, lon, region: Optional[RegionSection]):
    if region is None:
        return np.ones(lat.shape, dtype=bool)
    s, n = region.lat
    w, e = (float(x) % 360.0 for x in region.lon)
    lon360 = np.mod(lon, 360.0)
    if region.lon[1] - region.lon[0] >= 360.0:
        in_lon = np.ones(lat.shape, dtype=bool)
    elif w <= e:
        in_lon = (lon360 >= w) & (lon360 <= e)
    else:  # wraps through 0 E
        in_lon = (lon360 >= w) | (lon360 <= e)
    return (lat >= s) & (lat <= n) & in_lon


def load_grid_domain(
    section: GridSection,
    *,
    params,
    config,
    start: dt.datetime,
    step_seconds: int,
    min_patch_weight: float = 0.0,
    resolve=lambda p: Path(p),
):
    """The :class:`clm_ml.domain.Domain` of a grid run (module docstring). ``start`` and
    ``step_seconds`` give the first step (for the canopy heights, CLM's time weights);
    ``resolve`` turns run.yaml paths into absolute paths."""
    import netCDF4 as nc

    from clm_ml.domain import Domain, Gridcells, Patches, SoilColumns

    surf, mask_file = resolve(section.surface_data), resolve(section.land_mask)
    for key, path in (("surface_data", surf), ("land_mask", mask_file)):
        if not Path(path).exists():
            raise ConfigError(f"run.yaml: domain.grid.{key}: file not found: {path}")

    with nc.Dataset(mask_file) as ds:
        landmask = np.asarray(ds.variables["landmask"][:]) == 1
    with nc.Dataset(surf) as ds:

        def var(name):
            return np.ma.filled(ds.variables[name][:], np.nan).astype(float)

        lat2, lon2 = var("LATIXY"), var("LONGXY")
        if landmask.shape != lat2.shape:
            raise ConfigError(
                f"{mask_file}: landmask has shape {landmask.shape}, the surface data "
                f"{lat2.shape} (both must be on the same grid)"
            )
        natveg, crop = var("PCT_NATVEG"), var("PCT_CROP")
        nat_pft, cft = var("PCT_NAT_PFT"), var("PCT_CFT")
        if nat_pft.shape[0] != N_NATPFT or cft.shape[0] != len(CFT_PFTS):
            raise ConfigError(
                f"{surf}: expected {N_NATPFT} natural PFTs and {len(CFT_PFTS)} crop types "
                f"(a 16-PFT surface dataset); got {nat_pft.shape[0]} and {cft.shape[0]}"
            )
        # area of each PFT 1-16 in percent of the gridcell, (16, lat, lon)
        area = np.concatenate([natveg[None] * nat_pft[1:] / 100.0, crop[None] * cft / 100.0])
        area = np.where(np.isfinite(area), area, 0.0)
        vegetated = area.sum(axis=0)
        cells = landmask & (vegetated > 0.0) & _in_region(lat2, lon2, section.region)
        rows, cols = np.nonzero(cells)
        if section.max_gridcells is not None:
            rows, cols = rows[: section.max_gridcells], cols[: section.max_gridcells]
        if rows.size == 0:
            raise ConfigError("run.yaml: domain.grid: no vegetated land gridcell in the domain")

        # CLM's satellite-phenology heights at the first step
        when = satellite_phenology_date(start, step_seconds)
        m1, m2, w1, w2 = clm_monthly_weights(when.month, when.day)

        def at_cells(name, level):  # one level of a (level, lat, lon) field
            return np.ma.filled(ds.variables[name][level], np.nan)[rows, cols].astype(float)

        def monthly(name):  # (16, cells): PFTs 1-16 at the first step
            v = ds.variables[name]
            a = np.ma.filled(v[m1 - 1], np.nan)[1:17][:, rows, cols].astype(float)
            b = np.ma.filled(v[m2 - 1], np.nan)[1:17][:, rows, cols].astype(float)
            return w1 * a + w2 * b

        htop, hbot = monthly("MONTHLY_HEIGHT_TOP"), monthly("MONTHLY_HEIGHT_BOT")
        sand, clay = at_cells("PCT_SAND", 0), at_cells("PCT_CLAY", 0)
        organic = np.minimum(at_cells("ORGANIC", 0), ORGANIC_MAX)  # CLM: om_frac <= 1
        color = np.ma.filled(ds.variables["SOIL_COLOR"][:], 0)[rows, cols].astype(int)
        zbed = np.ma.filled(ds.variables["zbedrock"][:], np.nan)[rows, cols].astype(float)

    cell_area = area[:, rows, cols]  # (16, ng)
    frac = cell_area / cell_area.sum(axis=0, keepdims=True)
    keep = frac > 0.0
    if min_patch_weight > 0.0:
        keep &= frac >= min_patch_weight
        empty = ~keep.any(axis=0)
        if empty.any():  # keep each gridcell's largest patch
            keep[np.argmax(frac[:, empty], axis=0), np.flatnonzero(empty)] = True
    weight = np.where(keep, frac, 0.0)
    weight = weight / weight.sum(axis=0, keepdims=True)
    g_of, k_pft = np.nonzero(keep.T)  # gridcell-major order, then PFT
    pft = k_pft + 1
    n_p = pft.size

    ratios = ("z0m_to_height_ratio", "displacement_to_height_ratio")
    missing = params.missing_for(sorted(set(pft.tolist())), ratios)
    if missing:
        raise ConfigError(
            f"params ({params.source}): no value for {missing[:4]}: grid runs need each PFT's "
            "roughness ratios for the forcing height (CLM's rule), e.g. params: clm6"
        )
    z0mr = np.asarray(params.values[ratios[0]], dtype=float)[pft]
    displar = np.asarray(params.values[ratios[1]], dtype=float)[pft]
    top = htop[k_pft, g_of]
    if not np.all(np.isfinite(top)) or np.any(top <= 0.0):
        bad = int(np.sum(~(top > 0.0)))
        raise ConfigError(f"{surf}: {bad} patches have no canopy height (MONTHLY_HEIGHT_TOP)")
    zref = section.forcing_height + (z0mr + displar) * top

    lat, lon = lat2[rows, cols], lon2[rows, cols]
    names = tuple(f"{la:.2f}N_{lo:.2f}E" for la, lo in zip(lat, lon))
    gridcells = Gridcells(
        names=names,
        latitude=lat.astype(float),
        longitude=lon.astype(float),
        measurement_height=np.full(rows.size, float(section.forcing_height)),
        forcing_step_minutes=np.full(rows.size, step_seconds // 60),
        grid_index=np.stack([rows, cols], axis=1),
    )
    patches = Patches(
        gridcell_index=g_of.astype(int),
        column_index=np.arange(n_p),
        pft_index=pft.astype(int),
        weight=weight[k_pft, g_of],
        canopy_height=top,
        root_biomass=np.full(n_p, float(section.root_biomass)),
        leaf_area_profile_beta=np.full((n_p, 2), np.nan),
        stem_area_profile_beta=np.full((n_p, 2), np.nan),
        n_layers_within=np.zeros(n_p, dtype=int),
        n_layers_above=np.zeros(n_p, dtype=int),
        canopy_bottom=hbot[k_pft, g_of],
        measurement_height=zref,
    )
    columns = SoilColumns(
        texture_class=(None,) * n_p,
        sand_percent=sand[g_of],
        clay_percent=clay[g_of],
        organic_matter=organic[g_of],
        color_class=color[g_of],
        bedrock_depth=zbed[g_of],
    )
    domain = Domain(gridcells, patches, columns, towers={})
    missing = params.missing_for(sorted(set(pft.tolist())))
    if missing:
        raise ConfigError(f"params: no value for {missing[:5]} (used by the domain)")
    bad_soil = (sand + clay <= 0.0) | ~np.isfinite(sand + clay + organic + zbed)
    if bad_soil.any():
        raise ConfigError(
            f"{surf}: {int(bad_soil.sum())} gridcells have no top-level sand/clay, organic "
            f"matter or bedrock depth, e.g. {names[int(np.flatnonzero(bad_soil)[0])]}"
        )
    c = config.canopy_layers
    if c.n_layers_within is None:
        raise ConfigError(
            "domain.type grid: set model.yaml canopy_layers.n_layers_within/n_layers_above "
            "(constant layer counts, decision 28)"
        )
    if c.n_layers_within + c.n_layers_above > c.max_layers:
        raise ConfigError(
            f"model.yaml: canopy_layers: {c.n_layers_within} + {c.n_layers_above} layers "
            f"exceed max_layers {c.max_layers}"
        )
    return domain
