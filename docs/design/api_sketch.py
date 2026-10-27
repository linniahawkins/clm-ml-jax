"""
Proposed public API for ``clm_ml`` (design sketch v2: signatures only, no implementation).

Read alongside ``modernization_plan.md`` and the YAML files in ``examples/``.

How the pieces fit
------------------
    run.yaml ──► load_run() ──► Run
      ├─ model.yaml   ──► ModelConfig        (static: changing it recompiles)
      ├─ params.yaml  ──► Params             (per-PFT table; differentiable)
      ├─ sites        ──► Sites ──► Layers   (fixed max_layers + masks)
      ├─ forcing      ──► Forcing            (time, site, ...); differentiable
      └─ initial cond ──► State              (site, ...);       differentiable

    Run.execute() == Model.run(state, forcing, params) + write outputs

Conventions
-----------
- Every container is a frozen dataclass registered as a JAX pytree.
- ``static`` fields are compile-time constants (not differentiable).
- Shapes: ``nt`` = time steps, ``ns`` = sites, ``nl`` = ``max_layers``,
  ``nsoil`` = soil layers, ``npft`` = PFTs. Leaf arrays end in ``(2,)`` = [sunlit, shaded].
  Spectral arrays end in ``(2,)`` = [vis, nir].
- The physics is written for ONE column. ``Model`` adds the site axis with ``jax.vmap``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import jax

Array = jax.Array
PathLike = Union[str, Path]


# =============================================================================
# Top level: a run described by run.yaml
# =============================================================================


def load_run(path: PathLike, overrides: Sequence[str] = ()) -> "Run":
    """Read ``run.yaml`` and everything it references, validate it, and load forcing.

    ``overrides`` use dotted keys, as on the command line::

        load_run("run.yaml", overrides=["model.physics.integrator=rk2_midpoint",
                                        "time.end=2007-05-08T00:00"])

    Raises ``ConfigError`` naming the file, key, bad value and allowed values.
    """
    ...


@dataclasses.dataclass(frozen=True)
class RunSettings:
    """The experiment-level parts of run.yaml that are not model inputs."""

    start: str  # ISO datetime
    end: str
    output_directory: Path
    output_format: Literal["netcdf", "legacy_ascii"] = "netcdf"  # legacy_ascii: transitional
    output_variables: Union[Literal["all"], tuple[str, ...]] = "all"
    output_frequency: Literal["step", "hourly", "daily"] = "step"
    source_files: Mapping[str, Path] = dataclasses.field(default_factory=dict)  # provenance


@dataclasses.dataclass(frozen=True)
class Run:
    """Everything needed for one experiment, already loaded into memory."""

    model: "Model"
    params: "Params"
    forcing: "Forcing"
    initial_state: "State"
    settings: RunSettings

    def execute(self) -> "Outputs":
        """Run the model and write outputs as configured. Same as ``clm-ml run run.yaml``."""
        ...

    def summary(self) -> str:
        """What ``clm-ml check`` prints: sites, PFTs, layer counts, run length, memory estimate."""
        ...


class ConfigError(ValueError):
    """Invalid YAML content, e.g.
    ``model.yaml: numerics.integrator: 'rk5' is not one of [euler, ..., rk4]``."""


# =============================================================================
# ModelConfig  <-  model.yaml        (replaces MLclm_varctl globals)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PhysicsOptions:
    stomatal_conductance: Literal["medlyn", "ball_berry", "wue"] = "wue"
    water_stress_on_conductance: bool = True
    wue_solver: Literal["brent", "bisection"] = "bisection"
    photosynthesis_colimitation: bool = True
    temperature_acclimation: bool = True
    nitrogen_decay_kn: Optional[float] = None
    roughness_sublayer: Literal["harman_finnigan"] = "harman_finnigan"
    sparse_canopy: bool = True
    ground_conductance: Literal["extend_hf", "log_profile"] = "log_profile"
    flux_profile_solution: Literal["well_mixed", "implicit"] = "implicit"
    leaf_boundary_layer: Literal["laminar", "turbulent", "free_convection"] = "free_convection"
    solar_radiation: Literal["norman", "two_stream"] = "two_stream"
    leaf_optics_vary_with_height: bool = False
    longwave_radiation: Literal["norman"] = "norman"


@dataclasses.dataclass(frozen=True)
class NumericsOptions:
    substep_seconds: float = 300.0
    integrator: Literal["euler", "rk2_trapezoid", "rk2_midpoint", "rk2_ralston",
                        "rk3_heun", "rk3_ralston", "rk3_kutta", "rk4"] = "rk4"
    checkpoint_steps: bool = True


@dataclasses.dataclass(frozen=True)
class CanopyLayerOptions:
    max_layers: int = 50  # fixed array size for every site (CHATS7 uses 46)
    dz_tall: float = 0.5
    dz_short: float = 0.1
    tall_canopy_threshold: float = 2.0
    n_layers_above: Optional[int] = None
    n_layers_within: Optional[int] = None
    min_plant_area_per_layer: float = 0.01


@dataclasses.dataclass(frozen=True)
class SoilOptions:
    layer_structure: Literal["clm5_0", "clm4_5"] = "clm5_0"


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Contents of model.yaml. Hashable and static."""

    physics: PhysicsOptions = PhysicsOptions()
    numerics: NumericsOptions = NumericsOptions()
    canopy_layers: CanopyLayerOptions = CanopyLayerOptions()
    soil: SoilOptions = SoilOptions()

    @classmethod
    def from_yaml(cls, path: PathLike) -> "ModelConfig":
        ...

    def to_yaml(self, path: PathLike) -> None:
        """Write the complete config, all defaults included (useful for provenance)."""
        ...

    def replace(self, **dotted_changes: Any) -> "ModelConfig":
        """``config.replace(**{"physics.stomatal_conductance": "medlyn"})``"""
        ...


# =============================================================================
# Params  <-  params.yaml             (replaces MLpftcon / pftcon globals)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Params:
    """Per-PFT parameter table. Every float field is differentiable.

    Each field has shape ``(npft,)`` or ``(npft, 2)`` for [vis, nir] optics.
    Row order follows ``pft_names`` (CLM order, 0 = not_vegetated).
    """

    pft_names: tuple[str, ...]  # static

    # Photosynthesis / stomata
    vcmax25: Array
    g0_medlyn: Array
    g1_medlyn: Array
    g0_ball_berry: Array
    g1_ball_berry: Array
    iota: Array
    gs_min: Array
    psi50_gs: Array
    shape_gs: Array
    c3_photosynthesis: Array  # bool; structural, not differentiated

    # Plant hydraulics and roots
    plant_conductance: Array
    plant_capacitance: Array
    root_radius: Array
    root_density: Array
    root_resistivity: Array
    root_profile_beta: Array

    # Leaf structure and radiation
    leaf_dimension: Array
    leaf_angle_index: Array
    sla_top: Array
    leaf_emissivity: Array
    clumping_index: Array
    leaf_reflectance: Array  # (npft, 2)
    leaf_transmittance: Array  # (npft, 2)
    stem_reflectance: Array  # (npft, 2)
    stem_transmittance: Array  # (npft, 2)

    @classmethod
    def from_yaml(cls, path: PathLike) -> "Params":
        """Load a params file, resolving ``extends:`` chains (e.g. onto ``clm5_defaults``)."""
        ...

    def to_yaml(self, path: PathLike, *, only_changes_from: Optional[str] = None) -> None:
        """Write the table, e.g. after calibration. With ``only_changes_from="clm5_defaults"``
        writes a short override file like ``params_chats.yaml``."""
        ...

    def get(self, pft: str, name: str) -> Array:
        """``params.get("broadleaf_deciduous_temperate_tree", "vcmax25")``"""
        ...

    def set(self, pft: str, **values: Any) -> "Params":
        """Return a copy with values changed for one PFT (values may be JAX tracers)."""
        ...

    def for_sites(self, pft_index: Array) -> "SiteParams":
        """Gather each site's row. Differentiable: gradients flow back to the PFT table."""
        ...

    def describe(self) -> str:
        """Table of parameter names, units, and values for the PFTs in use."""
        ...


@dataclasses.dataclass(frozen=True)
class SiteParams:
    """Same fields as ``Params`` but shaped ``(ns,)`` / ``(ns, 2)``: one row per site.
    Internal: this is what the physics sees (one row, after vmap)."""


def perturb(params: Params, pft: str, **batched_values: Array) -> Params:
    """Build a batched ``Params`` for ensembles with ``jax.vmap``.

    ``perturb(params, "broadleaf_deciduous_temperate_tree", vcmax25=jnp.linspace(60, 140, 32))``
    returns ``Params`` whose fields have a leading axis of 32.
    """
    ...


# =============================================================================
# Sites and Layers                   (replaces TowerDataMod, MLinitVerticalMod, GridInfo)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Sites:
    """Site descriptions, one row per site (leading axis ``ns``)."""

    names: tuple[str, ...]  # static
    latitude: Array  # (ns,) degrees north
    longitude: Array  # (ns,) degrees east
    pft_index: Array  # (ns,) int, row into Params
    sand_percent: Array  # (ns,)
    clay_percent: Array  # (ns,)
    organic_matter: Array  # (ns,) kg m-3
    soil_color_class: Array  # (ns,) int
    bedrock_depth: Array  # (ns,) m
    reference_height: Array  # (ns,) m
    canopy_height: Array  # (ns,) m
    root_biomass: Array  # (ns,) g m-2
    leaf_area_profile_beta: Array  # (ns, 2)
    stem_area_profile_beta: Array  # (ns, 2)

    @classmethod
    def from_names(cls, names: Sequence[str], sites_file: Optional[PathLike] = None,
                   params: Optional[Params] = None) -> "Sites":
        """Look up sites in the built-in registry and/or a user sites.yaml.
        ``params`` is used to resolve PFT names to row indices."""
        ...

    @staticmethod
    def available() -> list[str]:
        """Names in the built-in registry."""
        ...


@dataclasses.dataclass(frozen=True)
class Layers:
    """Canopy vertical grid for every site, padded to a common ``max_layers``.

    Padded layers have zero thickness and zero plant area, are excluded from all sums,
    and are identity rows in tridiagonal solves, so padding does not change results.
    """

    max_layers: int  # static
    n_layers: Array  # (ns,) int, real air layers per site (was ncan)
    top_leaf_layer: Array  # (ns,) int, 0-based (was ntop)
    bottom_leaf_layer: Array  # (ns,) int, 0-based (was nbot)
    heights: Array  # (ns, nl) layer mid-point height (m)
    thickness: Array  # (ns, nl) m; 0 in padded layers
    is_air_layer: Array  # (ns, nl) bool mask
    has_leaf: Array  # (ns, nl) bool mask

    @classmethod
    def build(cls, sites: Sites, config: ModelConfig) -> "Layers":
        """Compute each site's layers and pad. Raises ``ConfigError`` naming the site
        if it needs more than ``config.canopy_layers.max_layers``."""
        ...


# =============================================================================
# Forcing                            (replaces TowerMetMod / clmDataMod per-step reads)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Forcing:
    """Everything prescribed from outside the model. Leading axes ``(nt, ns)``.
    All float fields are differentiable."""

    # Time (shared by all sites)
    time: Array  # (nt,) seconds since run start, end of each step
    calendar_day: Array  # (nt,) fractional day of year
    step_seconds: float  # static
    interpolation: Literal["none", "three_point"]  # static

    # Meteorology at reference height
    air_temperature: Array  # (nt, ns) K
    specific_humidity: Array  # (nt, ns) kg kg-1
    wind_speed: Array  # (nt, ns) m s-1
    air_pressure: Array  # (nt, ns) Pa
    co2: Array  # (nt, ns) umol mol-1
    o2: Array  # (nt, ns) mmol mol-1
    shortwave_direct: Array  # (nt, ns, 2) W m-2
    shortwave_diffuse: Array  # (nt, ns, 2) W m-2
    longwave_down: Array  # (nt, ns) W m-2
    rainfall: Array  # (nt, ns) kg m-2 s-1
    snowfall: Array  # (nt, ns) kg m-2 s-1
    solar_zenith: Array  # (nt, ns) rad, precomputed

    # Prescribed land state from CLM history
    lai: Array  # (nt, ns)
    sai: Array  # (nt, ns)
    soil_liquid_water: Array  # (nt, ns, nsoil) kg m-2, after adjustment
    soil_ice: Array  # (nt, ns, nsoil) kg m-2
    ground_albedo: Array  # (nt, ns, 2)

    acclimation_temperature: Array  # (ns,) K, mean over the record (was init_acclim)

    def window(self, t: int) -> "ForcingWindow":
        """Previous/current/next records around step ``t``, used by ``Model.step``."""
        ...

    def slice_time(self, start: int, stop: int) -> "Forcing":
        ...

    def select_sites(self, index: Sequence[int]) -> "Forcing":
        ...


@dataclasses.dataclass(frozen=True)
class ForcingWindow:
    previous: Forcing  # time axis removed
    current: Forcing
    next: Forcing


def load_forcing(forcing_section: Mapping[str, Any], sites: Sites, settings: RunSettings,
                 config: ModelConfig) -> Forcing:
    """Read every file named in run.yaml's ``forcing:`` section (filling ``{site}``)
    into memory. All netCDF I/O and unit conversion happens here, once."""
    ...


# =============================================================================
# State: prognostic variables only (the lax.scan carry)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CanopyState:
    air_temperature: Array  # (ns, nl) K
    vapor_pressure: Array  # (ns, nl) Pa
    co2: Array  # (ns, nl) umol mol-1
    leaf_temperature: Array  # (ns, nl, 2) K
    leaf_water_potential: Array  # (ns, nl, 2) MPa
    intercepted_water: Array  # (ns, nl) kg m-2
    ground_temperature: Array  # (ns,) K


@dataclasses.dataclass(frozen=True)
class SoilState:
    temperature: Array  # (ns, nsoil) K
    liquid_water: Array  # (ns, nsoil) kg m-2
    ice: Array  # (ns, nsoil) kg m-2


@dataclasses.dataclass(frozen=True)
class State:
    canopy: CanopyState
    soil: SoilState

    def save(self, path: PathLike) -> None:
        """Write a restart file (netCDF)."""
        ...

    @classmethod
    def load(cls, path: PathLike) -> "State":
        ...


# =============================================================================
# Outputs                            (replaces the six .out files)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Outputs:
    """Leading axes ``(nt, ns)`` from ``Model.run`` (``(ns,)`` from ``Model.step``).
    Unselected variables (``output.variables``) are ``None``."""

    # Canopy-scale fluxes (flux.out)
    net_radiation: Array  # W m-2
    canopy_air_storage_flux: Array
    sensible_heat_flux: Array
    latent_heat_flux: Array
    gpp: Array  # umol CO2 m-2 s-1
    friction_velocity: Array  # m s-1
    shortwave_up: Array
    longwave_up: Array
    canopy_top_air_temperature: Array  # K
    ground_heat_flux: Array
    soil_net_radiation: Array
    soil_sensible_heat_flux: Array
    soil_latent_heat_flux: Array
    transpiration_latent_heat: Array
    canopy_evaporation_latent_heat: Array
    soil_wetness_factor: Array
    vegetation_storage_flux: Array

    # Sunlit/shaded (fsun.out); trailing (3,) = [total, sunlit, shaded]
    leaf_area_sunlit_shaded: Array
    absorbed_par: Array
    gpp_by_leaf_class: Array
    latent_heat_by_leaf_class: Array
    sensible_heat_by_leaf_class: Array
    vcmax25_by_leaf_class: Array
    stomatal_conductance_by_leaf_class: Array
    leaf_temperature_by_leaf_class: Array

    # Profiles (profile.out, fluxprofile.out); NaN in padded layers when exported
    air_temperature_profile: Array  # (..., nl)
    vapor_pressure_profile: Array
    wind_speed_profile: Array
    leaf_temperature_profile: Array  # (..., nl, 2)
    sensible_heat_profile: Array
    latent_heat_profile: Array
    photosynthesis_profile: Array  # (..., nl, 2)

    # Soil (soiltemp.out)
    soil_temperature: Array  # (..., nsoil)

    # Conservation checks
    energy_balance_error: Array
    water_balance_error: Array

    def to_xarray(self, sites: Sites, forcing: Forcing, layers: Layers):
        """``xarray.Dataset`` with dims (time, site, layer, soil_layer, leaf_class, band),
        units and long names. Padded layers are NaN."""
        ...

    def to_netcdf(self, path: PathLike, sites: Sites, forcing: Forcing, layers: Layers) -> None:
        ...

    def write_legacy_ascii(self, directory: PathLike, sites: Sites, forcing: Forcing) -> list[Path]:
        """Six Fortran-format ``.out`` files per site, byte-compatible with today's output.
        Transitional: removed once the Fortran comparison test reads netCDF."""
        ...


# =============================================================================
# Model                              (replaces CLMml_drv, clm_drv, MLCanopyFluxes driver logic)
# =============================================================================


class Model:
    """Configured model for a set of sites. Holds only static structure; no mutable state."""

    config: ModelConfig
    sites: Sites
    layers: Layers  # built from sites + config

    def __init__(self, config: ModelConfig, sites: Sites) -> None:
        ...

    def initial_state(self, forcing: Forcing, *, source: Literal["clm_history"] = "clm_history"
                      ) -> State:
        """Cold-start canopy; soil from CLM history at forcing's first step."""
        ...

    def step(self, state: State, forcing: ForcingWindow, params: Params) -> tuple[State, Outputs]:
        """Advance every site by one forcing step (vmap of the one-column step). Pure."""
        ...

    def run(self, state: State, forcing: Forcing, params: Params, *,
            outputs: Union[Literal["all"], Sequence[str]] = "all",
            frequency: Literal["step", "hourly", "daily"] = "step",
            check: bool = False) -> Outputs:
        """Whole run as one compiled call: ``scan`` over time of ``vmap`` over sites.

        Differentiable w.r.t. ``state``, ``forcing`` and ``params``.
        """
        ...

    def run_with_final_state(self, state: State, forcing: Forcing, params: Params, **kwargs
                             ) -> tuple[State, Outputs]:
        """Like ``run`` but also returns the final state (chunked runs, restarts)."""
        ...


# =============================================================================
# Command line  (clm_ml/cli.py)
# =============================================================================
#
#   clm-ml run run.yaml [--set key=value ...]
#   clm-ml check run.yaml                     # validate + summary, no run
#   clm-ml convert nl.CHATS7.1day -o chats7/  # namelist -> run.yaml, model.yaml, params.yaml
#
# =============================================================================
# Process interface (for contributors)
# =============================================================================
#
# Each process in clm_ml/physics/ acts on ONE column with (max_layers,) arrays and masks:
#
#     def two_stream(step: StepState, inputs: StepInputs, params: SiteParams,
#                    layers: SiteLayers, config: ModelConfig) -> StepState:
#         """Two-stream solar radiative transfer.
#
#         Reads:  inputs.solar_zenith, inputs.shortwave_direct, step.lai_profile, layers.has_leaf
#         Writes: step.absorbed_shortwave_leaf, step.absorbed_shortwave_soil
#         Fortran reference: MLSolarRadiationMod.F90, subroutine TwoStream
#         """
#
#     SOLAR_RADIATION = {"norman": norman, "two_stream": two_stream}
