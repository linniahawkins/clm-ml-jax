"""
Model output: named variables with units, per patch and per gridcell.

:meth:`Outputs.from_legacy` turns the fields recorded from the legacy driver into the
variables below. Each variable says which legacy ``.out`` file column it corresponds to,
so results can be compared with earlier runs and with the Fortran model.

Dimensions: ``time``, ``patch``, ``layer`` (canopy-air layers, bottom first, padded to
``max_layers`` with NaN), ``leaf_class`` (sunlit, shaded), ``leaf_group`` (canopy total,
sunlit, shaded), ``band`` (vis, nir), ``soil_layer``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from clm_ml.params import PFT_NAMES

# Physical constants used only to derive output quantities (as in CLMml_driver.output).
_MMH2O, _MMDRY = 18.02e-3, 28.97e-3  # kg/mol, as in MLclm_varcon


@dataclasses.dataclass(frozen=True)
class VariableSpec:
    name: str
    units: str
    long_name: str
    dims: tuple[str, ...]  # after (time, patch)
    legacy: str  # where the value appears in the legacy .out files
    compute: Callable[["_Raw"], np.ndarray]  # -> (time, patch, *dims)


class _Raw:
    """Recorded legacy fields, with helpers for slicing profiles to the layer axis.

    ``xp`` is the array module: ``numpy`` for recorded runs, or ``jax.numpy`` inside a
    compiled chunk (output streams accumulate the variables during the scan). The variable
    definitions below use ``r.xp``, so each is written once for both."""

    def __init__(self, raw: Mapping[str, np.ndarray], max_layers: int, xp=np):
        self.raw = raw
        self.max_layers = max_layers
        self.xp = xp
        self.ncan = raw["mlcanopy.ncan_canopy"].astype(int)  # (nt, np)
        self.ntop = raw["mlcanopy.ntop_canopy"].astype(int)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.raw[f"mlcanopy.{name}"]

    def layer(self, values: np.ndarray, *, within_canopy_only: bool = False) -> np.ndarray:
        """Legacy 1-based profile (nt, np, nlevmlcan+1, ...) -> (nt, np, max_layers, ...),
        NaN above the top layer (or above the canopy, if ``within_canopy_only``)."""
        xp = self.xp
        out = xp.asarray(values[:, :, 1 : self.max_layers + 1], dtype=float)
        top = self.ntop if within_canopy_only else self.ncan
        k = xp.arange(1, self.max_layers + 1)
        invalid = k[None, None, :] > top[:, :, None]
        invalid = invalid.reshape(invalid.shape + (1,) * (out.ndim - 3))
        return xp.where(invalid, xp.nan, out)

    def leaf(self, values: np.ndarray) -> np.ndarray:
        """Legacy leaf profile (nt, np, nlev+1, nleaf+1) -> (nt, np, max_layers, 2),
        NaN in layers without leaves. Legacy leaf index 1 = sunlit, 2 = shaded."""
        xp = self.xp
        out = self.layer(values, within_canopy_only=True)[..., 1:3]
        has_leaf = self.layer(self["dpai_profile"], within_canopy_only=True) > 0  # NaN -> False
        return xp.where(has_leaf[..., None], out, xp.nan)

    def by_group(self, total: str, sun: str, sha: str, band: int | None = None) -> np.ndarray:
        parts = [self[total], self[sun], self[sha]]
        if band is not None:
            parts = [p[..., band] for p in parts]
        return self.xp.stack(parts, axis=-1)

    def at_layer(self, values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return self.xp.take_along_axis(values, index[:, :, None], axis=2)[:, :, 0]

    def latvap_tref(self) -> np.ndarray:
        """Molar latent heat of vaporization at the reference air temperature (J/mol),
        MLWaterVaporMod.LatVap (recorded by the legacy worker as derived.latvap_tref)."""
        if "derived.latvap_tref" in self.raw:
            return self.raw["derived.latvap_tref"]
        from clm_src_main.clm_varcon import hsub, hvap, tfrz
        from multilayer_canopy.MLclm_varcon import mmh2o

        t = self["tref_forcing"]
        return self.xp.where(t > tfrz, hvap, hsub) * mmh2o


def _shortwave_up(r: _Raw) -> np.ndarray:
    sky = r["swskyb_forcing"] + r["swskyd_forcing"]
    return (r["albcan_canopy"][..., 1:3] * sky[..., 1:3]).sum(axis=-1)


def _mid_layer(r: _Raw) -> np.ndarray:
    nbot, ntop = r["nbot_canopy"].astype(int), r.ntop
    return r.xp.maximum(1, nbot + (ntop - nbot + 1) // 2 - 1)


def _specific_humidity(r: _Raw) -> np.ndarray:
    e = r.layer(r["eair_profile"])
    p = r["pref_forcing"][:, :, None]
    ratio = _MMH2O / _MMDRY
    return ratio * e / (p - (1.0 - ratio) * e)


V = VariableSpec
VARIABLES: tuple[VariableSpec, ...] = (
    # --- canopy-scale fluxes (flux.out) -----------------------------------------------
    V("net_radiation", "W m-2", "Net radiation above the canopy", (), "flux.out col 2",
      lambda r: r["rnet_canopy"]),
    V("canopy_air_storage_flux", "W m-2", "Heat storage in canopy air", (), "flux.out col 3",
      lambda r: r["stflx_air_canopy"]),
    V("sensible_heat_flux", "W m-2", "Sensible heat flux above the canopy", (), "flux.out col 4",
      lambda r: r["shflx_canopy"]),
    V("latent_heat_flux", "W m-2", "Latent heat flux above the canopy", (), "flux.out col 5",
      lambda r: r["lhflx_canopy"]),
    V("gpp", "umol CO2 m-2 s-1", "Gross primary production", (), "flux.out col 6",
      lambda r: r["gppveg_canopy"]),
    V("friction_velocity", "m s-1", "Friction velocity", (), "flux.out col 7",
      lambda r: r["ustar_canopy"]),
    V("shortwave_up", "W m-2", "Reflected shortwave radiation", (), "flux.out col 8",
      _shortwave_up),
    V("longwave_up", "W m-2", "Upward longwave radiation above the canopy", (), "flux.out col 9",
      lambda r: r["lwup_canopy"]),
    V("canopy_top_air_temperature", "K", "Air temperature at the canopy top layer", (),
      "flux.out col 10", lambda r: r.at_layer(r["tair_profile"], r.ntop)),
    V("ground_heat_flux", "W m-2", "Soil heat flux (positive into soil)", (), "flux.out col 11",
      lambda r: r["gsoi_soil"]),
    V("soil_net_radiation", "W m-2", "Net radiation at the soil surface", (), "flux.out col 12",
      lambda r: r["rnsoi_soil"]),
    V("soil_sensible_heat_flux", "W m-2", "Sensible heat flux from the soil", (),
      "flux.out col 13", lambda r: r["shsoi_soil"]),
    V("soil_latent_heat_flux", "W m-2", "Latent heat flux from the soil", (), "flux.out col 14",
      lambda r: r["lhsoi_soil"]),
    V("transpiration_latent_heat_flux", "W m-2", "Latent heat flux from transpiration", (),
      "flux.out col 15", lambda r: r["trveg_canopy"] * r.latvap_tref()),
    V("canopy_evaporation_latent_heat_flux", "W m-2",
      "Latent heat flux from evaporation of intercepted water", (), "flux.out col 16",
      lambda r: r["evveg_canopy"] * r.latvap_tref()),
    V("rsl_beta", "1", "Roughness-sublayer beta: friction velocity / wind speed at canopy top",
      (), "flux.out col 17", lambda r: r["beta_canopy"]),
    V("vegetation_storage_flux", "W m-2", "Heat storage in vegetation", (), "flux.out col 18",
      lambda r: r["stflx_veg_canopy"]),
    # --- sunlit/shaded canopy totals (fsun.out) ---------------------------------------
    V("solar_zenith_angle", "degree", "Solar zenith angle", (), "fsun.out col 1",
      lambda r: r.xp.degrees(r["solar_zen_forcing"])),
    V("incident_visible_radiation", "W m-2", "Incident visible solar radiation", (),
      "fsun.out col 2", lambda r: r["swskyb_forcing"][..., 1] + r["swskyd_forcing"][..., 1]),
    V("leaf_area_index", "m2 m-2", "Leaf area index", (), "fsun.out col 3 (with sai)",
      lambda r: r["lai_canopy"]),
    V("stem_area_index", "m2 m-2", "Stem area index", (), "fsun.out col 3 (with lai)",
      lambda r: r["sai_canopy"]),
    V("leaf_area_index_by_leaf_class", "m2 m-2", "Sunlit and shaded leaf area index",
      ("leaf_class",), "fsun.out cols 4-5",
      lambda r: r.xp.stack([r["laisun_canopy"], r["laisha_canopy"]], axis=-1)),
    V("absorbed_visible_radiation", "W m-2", "Visible radiation absorbed by vegetation",
      ("leaf_group",), "fsun.out cols 6-8",
      lambda r: r.by_group("swveg_canopy", "swvegsun_canopy", "swvegsha_canopy", band=1)),
    V("gpp_by_leaf_group", "umol CO2 m-2 s-1", "Gross primary production",
      ("leaf_group",), "fsun.out cols 9-11",
      lambda r: r.by_group("gppveg_canopy", "gppvegsun_canopy", "gppvegsha_canopy")),
    V("vegetation_latent_heat_flux", "W m-2", "Latent heat flux from vegetation",
      ("leaf_group",), "fsun.out cols 12-14",
      lambda r: r.by_group("lhveg_canopy", "lhvegsun_canopy", "lhvegsha_canopy")),
    V("vegetation_sensible_heat_flux", "W m-2", "Sensible heat flux from vegetation",
      ("leaf_group",), "fsun.out cols 15-17",
      lambda r: r.by_group("shveg_canopy", "shvegsun_canopy", "shvegsha_canopy")),
    V("canopy_vcmax25", "umol m-2 s-1", "Canopy Vcmax at 25 C", ("leaf_group",),
      "fsun.out cols 18-20",
      lambda r: r.by_group("vcmax25veg_canopy", "vcmax25sun_canopy", "vcmax25sha_canopy")),
    V("canopy_stomatal_conductance", "mol H2O m-2 s-1", "Canopy stomatal conductance",
      ("leaf_group",), "fsun.out cols 21-23",
      lambda r: r.by_group("gsveg_canopy", "gsvegsun_canopy", "gsvegsha_canopy")),
    V("leaf_wind_speed", "m s-1", "Wind speed at leaves (leaf-area weighted)",
      ("leaf_group",), "fsun.out cols 24-26",
      lambda r: r.by_group("windveg_canopy", "windvegsun_canopy", "windvegsha_canopy")),
    V("mean_leaf_temperature", "K", "Leaf temperature (leaf-area weighted)",
      ("leaf_group",), "fsun.out cols 27-29",
      lambda r: r.by_group("tlveg_canopy", "tlvegsun_canopy", "tlvegsha_canopy")),
    V("leaf_air_temperature", "K", "Air temperature at leaves (leaf-area weighted)",
      ("leaf_group",), "fsun.out cols 30-32",
      lambda r: r.by_group("taveg_canopy", "tavegsun_canopy", "tavegsha_canopy")),
    # --- plant water (aux.out) ----------------------------------------------------------
    V("soil_water_stress", "1", "Soil water stress factor (btran)", (), "aux.out col 1",
      lambda r: r["btran_soil"]),
    V("top_leaf_specific_conductance", "mmol H2O m-2 s-1 MPa-1",
      "Leaf-specific conductance, canopy top layer", (), "aux.out col 2",
      lambda r: r.at_layer(r["lsc_profile"], r.ntop)),
    V("soil_water_potential", "MPa", "Weighted soil water potential", (), "aux.out col 3",
      lambda r: r["psis_soil"]),
    V("top_leaf_water_potential", "MPa", "Mean leaf water potential, canopy top layer", (),
      "aux.out col 4", lambda r: r.at_layer(r["lwp_mean_profile"], r.ntop)),
    V("mid_leaf_water_potential", "MPa", "Mean leaf water potential, mid-canopy layer", (),
      "aux.out col 5", lambda r: r.at_layer(r["lwp_mean_profile"], _mid_layer(r))),
    V("fraction_at_minimum_leaf_water_potential", "1",
      "Fraction of canopy at minimum leaf water potential", (), "aux.out col 6",
      lambda r: r["fracminlwp_canopy"]),
    # --- vertical profiles (profile.out, fluxprofile.out) -------------------------------
    V("layer_height", "m", "Height of layer center", ("layer",), "profile.out col 2",
      lambda r: r.layer(r["zs_profile"])),
    V("layer_top_height", "m", "Height of layer top interface", ("layer",),
      "fluxprofile.out col 2", lambda r: r.layer(r["zw_profile"])),
    V("plant_area_density", "m2 m-3", "Plant area density", ("layer",), "profile.out col 4",
      lambda r: r.layer(r["dpai_profile"], within_canopy_only=True)
      / r.layer(r["dz_profile"], within_canopy_only=True)),
    V("sunlit_fraction", "1", "Sunlit fraction of leaves", ("layer",), "profile.out col 3",
      lambda r: r.layer(r["fracsun_profile"], within_canopy_only=True)),
    V("air_temperature", "K", "Air temperature", ("layer",), "profile.out col 26",
      lambda r: r.layer(r["tair_profile"])),
    V("vapor_pressure", "Pa", "Water vapor pressure", ("layer",), "profile.out (via col 27)",
      lambda r: r.layer(r["eair_profile"])),
    V("specific_humidity", "kg kg-1", "Specific humidity", ("layer",),
      "profile.out col 27 (g/kg there)", _specific_humidity),
    V("wind_speed", "m s-1", "Wind speed", ("layer",), "profile.out col 25",
      lambda r: r.layer(r["wind_profile"])),
    V("aerodynamic_resistance", "s m-1", "Aerodynamic resistance between layers", ("layer",),
      "profile.out col 28",
      lambda r: r["rhomol_forcing"][:, :, None] / r.layer(r["gac_profile"])),
    V("sensible_heat_flux_profile", "W m-2", "Sensible heat flux at layer top", ("layer",),
      "fluxprofile.out col 3", lambda r: r.layer(r["shair_profile"])),
    V("latent_heat_flux_profile", "W m-2", "Latent heat flux at layer top", ("layer",),
      "fluxprofile.out col 4",
      lambda r: r.layer(r["etair_profile"]) * r.latvap_tref()[:, :, None]),
    V("momentum_flux_profile", "m2 s-2", "Momentum flux at layer top", ("layer",),
      "fluxprofile.out col 5", lambda r: r.layer(r["mflx_profile"])),
    V("direct_shortwave_down_profile", "W m-2", "Direct beam solar radiation", ("layer", "band"),
      "fluxprofile.out cols 6-7", lambda r: r.layer(r["swbeam_profile"])[..., 1:3]),
    V("diffuse_shortwave_down_profile", "W m-2", "Downward diffuse solar radiation",
      ("layer", "band"), "fluxprofile.out cols 8-9",
      lambda r: r.layer(r["swdwn_profile"])[..., 1:3]),
    V("diffuse_shortwave_up_profile", "W m-2", "Upward diffuse solar radiation",
      ("layer", "band"), "fluxprofile.out cols 10-11",
      lambda r: r.layer(r["swupw_profile"])[..., 1:3]),
    V("longwave_down_profile", "W m-2", "Downward longwave radiation (within canopy only)",
      ("layer",), "fluxprofile.out col 12 (0 above canopy there)",
      lambda r: r.layer(r["lwdwn_profile"], within_canopy_only=True)),
    V("longwave_up_profile", "W m-2", "Upward longwave radiation (within canopy only)",
      ("layer",), "fluxprofile.out col 13 (0 above canopy there)",
      lambda r: r.layer(r["lwupw_profile"], within_canopy_only=True)),
    # --- leaf profiles, sunlit and shaded (profile.out) ---------------------------------
    V("leaf_net_radiation", "W m-2", "Leaf net radiation (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 7-8", lambda r: r.leaf(r["rnleaf_leaf"])),
    V("leaf_sensible_heat_flux", "W m-2", "Leaf sensible heat flux (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 9-10", lambda r: r.leaf(r["shleaf_leaf"])),
    V("leaf_latent_heat_flux", "W m-2", "Leaf latent heat flux (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 11-12", lambda r: r.leaf(r["lhleaf_leaf"])),
    V("leaf_net_photosynthesis", "umol CO2 m-2 s-1", "Leaf net photosynthesis (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 13-14", lambda r: r.leaf(r["anet_leaf"])),
    V("leaf_absorbed_par", "umol photon m-2 s-1", "Leaf absorbed PAR (per leaf area)",
      ("layer", "leaf_class"), "profile.out cols 15-16", lambda r: r.leaf(r["apar_leaf"])),
    V("leaf_stomatal_conductance", "mol H2O m-2 s-1", "Leaf stomatal conductance",
      ("layer", "leaf_class"), "profile.out cols 17-18", lambda r: r.leaf(r["gs_leaf"])),
    V("leaf_water_potential", "MPa", "Leaf water potential", ("layer", "leaf_class"),
      "profile.out cols 19-20", lambda r: r.leaf(r["lwp_hist_leaf"])),
    V("leaf_temperature", "K", "Leaf temperature", ("layer", "leaf_class"),
      "profile.out cols 21-22", lambda r: r.leaf(r["tleaf_hist_leaf"])),
    V("leaf_vcmax25", "umol m-2 s-1", "Leaf Vcmax at 25 C", ("layer", "leaf_class"),
      "profile.out cols 23-24", lambda r: r.leaf(r["vcmax25_leaf"])),
)  # fmt: skip

VARIABLES_BY_NAME: Mapping[str, VariableSpec] = {v.name: v for v in VARIABLES}
# Variables that cannot be averaged across patches with different layer grids.
_PATCH_ONLY_DIMS = {"layer"}

COORDINATES = {
    "leaf_class": ["sunlit", "shaded"],
    "leaf_group": ["canopy", "sunlit", "shaded"],
    "band": ["vis", "nir"],
}


@dataclasses.dataclass(frozen=True)
class Outputs:
    """Per-patch output: ``data[name]`` has shape ``(time, patch, ...)``.

    Variables are also available as attributes (``outputs.latent_heat_flux``).
    """

    data: Mapping[str, np.ndarray]
    time: np.ndarray  # datetime64, end of each time step
    patch_gridcell: np.ndarray  # (patch,) int
    patch_weight: np.ndarray  # (patch,)
    patch_pft: tuple[str, ...]
    gridcell_names: tuple[str, ...]
    soil_temperature: np.ndarray | None = None  # (time, patch, soil_layer) K
    soil_depth: np.ndarray | None = None  # (patch, soil_layer) m, node depth
    raw: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict, repr=False)

    def __getattr__(self, name: str) -> np.ndarray:
        data = object.__getattribute__(self, "data")
        if name in data:
            return data[name]
        raise AttributeError(name)

    @classmethod
    def from_legacy(cls, raw: Mapping[str, np.ndarray], run) -> "Outputs":
        r = _Raw(raw, run.config.canopy_layers.max_layers)
        data = {v.name: np.asarray(v.compute(r), dtype=float) for v in VARIABLES}

        n_steps = raw["mlcanopy.ncan_canopy"].shape[0]
        step = dt.timedelta(seconds=run.run_file.time.step_seconds)
        time = np.array([run.start + (k + 1) * step for k in range(n_steps)], "datetime64[ns]")

        # Soil layers 1..nlevgrnd (legacy arrays are 1-based in this dimension).
        nlevgrnd = int(raw["static.nlevgrnd"])
        soil = raw["temperature.t_soisno_col"][:, :, 1 : nlevgrnd + 1]
        depth = raw["static.soil_depth"][:, 1 : nlevgrnd + 1]

        d = run.domain
        return cls(
            data=data,
            time=time,
            patch_gridcell=d.patches.gridcell_index.copy(),
            patch_weight=d.patches.weight.copy(),
            patch_pft=tuple(PFT_NAMES[i] for i in d.patches.pft_index),
            gridcell_names=d.gridcells.names,
            soil_temperature=np.asarray(soil, dtype=float),
            soil_depth=np.asarray(depth, dtype=float),
            raw=dict(raw),
        )

    @classmethod
    def from_records(
        cls, records: Mapping[str, np.ndarray], run, static: Mapping[str, np.ndarray]
    ) -> "Outputs":
        """One output stream's records (``{"time": ..., variable: (record, patch, ...)}``,
        from ``clm_ml.legacy.worker``). Raw legacy fields (``mlcanopy.<field>``, validation)
        go to ``raw``."""
        nlevgrnd = int(static["static.nlevgrnd"])
        soil = records.get("soil_temperature")
        d = run.domain
        return cls(
            data={
                n: np.asarray(v, dtype=float)
                for n, v in records.items()
                if n not in ("time", "soil_temperature") and "." not in n
            },
            time=np.asarray(records["time"], dtype="datetime64[ns]"),
            patch_gridcell=d.patches.gridcell_index.copy(),
            patch_weight=d.patches.weight.copy(),
            patch_pft=tuple(PFT_NAMES[i] for i in d.patches.pft_index),
            gridcell_names=d.gridcells.names,
            soil_temperature=None if soil is None else np.asarray(soil, dtype=float),
            soil_depth=np.asarray(static["static.soil_depth"])[:, 1 : nlevgrnd + 1],
            raw={n: v for n, v in records.items() if "." in n},
        )

    def split(self, file_per: str, run_start: dt.datetime) -> list:
        """``[(label, Outputs), ...]``, one per output file: the whole run (labelled by the run
        start date), or per year / month of the records (by the start of their interval)."""
        if file_per == "run":
            return [(f"{run_start:%Y-%m-%d}", self)]
        # A record's time is the end of its interval; one microsecond earlier is inside it.
        inside = self.time - np.timedelta64(1, "us")
        fmt = "%Y" if file_per == "year" else "%Y-%m"
        labels = np.array([dt.datetime.fromisoformat(str(t)[:26]).strftime(fmt) for t in inside])
        return [(label, self.take(labels == label)) for label in dict.fromkeys(labels)]

    def take(self, mask: np.ndarray) -> "Outputs":
        """The records selected by a boolean ``mask`` over time."""
        return dataclasses.replace(
            self,
            data={n: v[mask] for n, v in self.data.items()},
            time=self.time[mask],
            soil_temperature=None if self.soil_temperature is None else self.soil_temperature[mask],
            raw={n: v[mask] for n, v in self.raw.items()},
        )

    def select(self, names: Sequence[str]) -> "Outputs":
        unknown = sorted(set(names) - set(self.data))
        if unknown:
            raise KeyError(f"unknown output variable(s) {unknown}; see clm_ml.outputs.VARIABLES")
        return dataclasses.replace(self, data={n: self.data[n] for n in names})

    def gridcell_mean(self) -> dict[str, np.ndarray]:
        """Area-weighted mean over each gridcell's patches, for variables without a
        ``layer`` dimension (patches on a gridcell have different layer grids).
        Returns ``{name: array (time, gridcell, ...)}``."""
        n_g = len(self.gridcell_names)
        out = {}
        for name, values in self.data.items():
            if _PATCH_ONLY_DIMS & set(VARIABLES_BY_NAME[name].dims):
                continue
            w = self.patch_weight.reshape((1, -1) + (1,) * (values.ndim - 2))
            summed = np.zeros((values.shape[0], n_g) + values.shape[2:])
            np.add.at(summed, (slice(None), self.patch_gridcell), values * w)
            out[name] = summed
        return out

    def to_xarray(self, level: str = "patch"):
        """``xarray.Dataset`` with units and long names. ``level`` is ``"patch"`` or
        ``"gridcell"`` (area-weighted means; profile variables are omitted)."""
        import xarray as xr

        if level == "gridcell":
            data, unit_dim = self.gridcell_mean(), "gridcell"
        elif level == "patch":
            data, unit_dim = self.data, "patch"
        else:
            raise ValueError(f"level must be 'patch' or 'gridcell', got {level!r}")

        variables = {}
        for name, values in data.items():
            spec = VARIABLES_BY_NAME[name]
            variables[name] = xr.Variable(
                ("time", unit_dim) + spec.dims,
                values,
                attrs={"units": spec.units, "long_name": spec.long_name, "legacy": spec.legacy},
            )
        coords = {"time": self.time, **{k: v for k, v in COORDINATES.items()}}
        if level == "patch":
            coords.update(
                gridcell=("patch", self.patch_gridcell),
                weight=("patch", self.patch_weight),
                pft=("patch", list(self.patch_pft)),
            )
            if self.soil_temperature is not None:
                variables["soil_temperature"] = xr.Variable(
                    ("time", "patch", "soil_layer"),
                    self.soil_temperature,
                    attrs={"units": "K", "long_name": "Soil temperature"},
                )
                coords["soil_depth"] = (("patch", "soil_layer"), self.soil_depth, {"units": "m"})
        else:
            coords["gridcell"] = list(self.gridcell_names)
        ds = xr.Dataset(variables, coords=coords)
        ds.attrs["title"] = "CLM-ML-JAX multilayer canopy model output"
        ds.attrs["time_convention"] = "time is the end of each output interval"
        return ds

    def to_netcdf(self, path: Path, levels: Sequence[str] = ("patch", "gridcell")) -> list[Path]:
        """Write one file per requested level (``*_patch.nc``, ``*_gridcell.nc``)."""
        written = []
        for level in levels:
            out = Path(path).with_name(f"{Path(path).stem}_{level}.nc")
            self.to_xarray(level).to_netcdf(out)
            written.append(out)
        return written
