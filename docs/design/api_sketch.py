"""
Proposed public API for ``clm_ml`` (design sketch v3: signatures only, no implementation).

Read alongside ``modernization_plan.md`` (vocabulary in §3.1) and the YAML files in
``examples/``.

How the pieces fit
------------------
    run.yaml ──► load_run() ──► Run
      ├─ model.yaml   ──► ModelConfig   (static: changing it recompiles)
      ├─ params.yaml  ──► Params        (per-PFT table; differentiable)
      ├─ towers       ──► Domain        (gridcells, patches, soil columns + index maps)
      │                     └─► Layers  (per patch: fixed max_layers + masks)
      ├─ forcing      ──► Forcing       (time, gridcell | patch | column, ...); differentiable
      └─ initial cond ──► State         (patch | column, ...);                  differentiable

    Run.execute() == Model.run(state, forcing, params) + write outputs

Vocabulary
----------
- gridcell:    a location with its own atmospheric forcing.
- patch:       one PFT's canopy on a gridcell, with an area weight. THE UNIT OF COMPUTATION.
- soil column: the soil under a patch. Option A (default): one column per patch.
- tower:       a named flux-tower location in the registry; it defines a gridcell and its patches.

Conventions
-----------
- Every container is a frozen dataclass registered as a JAX pytree.
- ``static`` fields are compile-time constants (not differentiable).
- Shapes: ``nt`` = time steps, ``ng`` = gridcells, ``np`` = patches, ``nc`` = soil columns,
  ``nl`` = ``max_layers``, ``nsoil`` = soil layers, ``npft`` = PFTs.
  Leaf arrays end in ``(2,)`` = [sunlit, shaded]. Spectral arrays end in ``(2,)`` = [vis, nir].
- The physics is written for ONE patch. ``Model`` adds the patch axis with ``jax.vmap``,
  gathers gridcell forcing and PFT parameters to patches, and aggregates patch output to
  gridcells with an area-weighted ``segment_sum``.
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
    output_level: tuple[Literal["patch", "gridcell"], ...] = ("patch", "gridcell")
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
        """What ``clm-ml check`` prints: gridcells, patches (PFT, weight, layer count),
        run length, memory estimate."""
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
    max_layers: int = 50  # fixed array size for every patch (CHATS7 uses 46)
    dz_tall: float = 0.5
    dz_short: float = 0.1
    tall_canopy_threshold: float = 2.0
    n_layers_above: Optional[int] = None
    n_layers_within: Optional[int] = None
    min_plant_area_per_layer: float = 0.01


@dataclasses.dataclass(frozen=True)
class SoilOptions:
    layer_structure: Literal["clm5_0", "clm4_5"] = "clm5_0"
    # per_patch = option A (each patch has its own soil column; identical to today).
    # shared    = option B (patches on a gridcell share one column; later science change).
    columns: Literal["per_patch", "shared"] = "per_patch"


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

    def for_patches(self, pft_index: Array) -> "PatchParams":
        """Gather each patch's row. Differentiable: gradients flow back to the PFT table,
        summed over all patches that share a PFT."""
        ...

    def describe(self) -> str:
        """Table of parameter names, units, and values for the PFTs in use."""
        ...


@dataclasses.dataclass(frozen=True)
class PatchParams:
    """Same fields as ``Params`` but shaped ``(np,)`` / ``(np, 2)``: one row per patch.
    Internal: this is what the physics sees (one row, after vmap)."""


def perturb(params: Params, pft: str, **batched_values: Array) -> Params:
    """Build a batched ``Params`` for ensembles with ``jax.vmap``.

    ``perturb(params, "broadleaf_deciduous_temperate_tree", vcmax25=jnp.linspace(60, 140, 32))``
    returns ``Params`` whose fields have a leading axis of 32.
    """
    ...


# =============================================================================
# Domain: gridcells, patches, soil columns   (replaces TowerDataMod, CLM subgrid modules)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Gridcells:
    """Locations with their own atmospheric forcing. Leading axis ``ng``."""

    names: tuple[str, ...]  # static; tower names for tower runs
    latitude: Array  # (ng,) degrees north
    longitude: Array  # (ng,) degrees east
    measurement_height: Array  # (ng,) forcing (reference) height, m


@dataclasses.dataclass(frozen=True)
class Patches:
    """One PFT's canopy on a gridcell. Leading axis ``np``. The unit of computation."""

    gridcell_index: Array  # (np,) int, static: which gridcell each patch is on
    column_index: Array  # (np,) int, static: which soil column is under it (option A: arange)
    pft_index: Array  # (np,) int, row into Params
    weight: Array  # (np,) fraction of the gridcell; sums to 1 per gridcell; differentiable
    canopy_height: Array  # (np,) m
    root_biomass: Array  # (np,) g m-2
    leaf_area_profile_beta: Array  # (np, 2)
    stem_area_profile_beta: Array  # (np, 2)


@dataclasses.dataclass(frozen=True)
class SoilColumns:
    """Soil under patches. Leading axis ``nc`` (option A: ``nc == np``)."""

    sand_percent: Array  # (nc,)
    clay_percent: Array  # (nc,)
    organic_matter: Array  # (nc,) kg m-3
    color_class: Array  # (nc,) int
    bedrock_depth: Array  # (nc,) m


@dataclasses.dataclass(frozen=True)
class Domain:
    """Everything spatial: gridcells, their patches, and the soil columns under them."""

    gridcells: Gridcells
    patches: Patches
    columns: SoilColumns

    @classmethod
    def from_towers(cls, names: Sequence[str], towers_file: Optional[PathLike] = None, *,
                    params: Params, config: ModelConfig,
                    min_patch_weight: float = 0.0) -> "Domain":
        """Build a domain from built-in and/or user towers (``towers.yaml`` format).

        ``params`` resolves PFT names to rows. ``config.soil.columns`` decides whether each
        patch gets its own soil column (option A) or shares one per gridcell (option B).
        Patches below ``min_patch_weight`` are dropped and the rest renormalized.
        """
        ...

    @staticmethod
    def available_towers() -> list[str]:
        """Names in the built-in tower registry."""
        ...

    @property
    def n_gridcells(self) -> int:
        ...

    @property
    def n_patches(self) -> int:
        ...


@dataclasses.dataclass(frozen=True)
class Layers:
    """Canopy vertical grid for every patch, padded to a common ``max_layers``.

    Padded layers have zero thickness and zero plant area, are excluded from all sums,
    and are identity rows in tridiagonal solves, so padding does not change results.
    """

    max_layers: int  # static
    n_layers: Array  # (np,) int, real air layers per patch (was ncan)
    top_leaf_layer: Array  # (np,) int, 0-based (was ntop)
    bottom_leaf_layer: Array  # (np,) int, 0-based (was nbot)
    heights: Array  # (np, nl) layer mid-point height (m)
    thickness: Array  # (np, nl) m; 0 in padded layers
    is_air_layer: Array  # (np, nl) bool mask
    has_leaf: Array  # (np, nl) bool mask

    @classmethod
    def build(cls, domain: Domain, config: ModelConfig) -> "Layers":
        """Compute each patch's layers (canopy height up to its gridcell's measurement height)
        and pad. Raises ``ConfigError`` naming the tower and patch if it needs more than
        ``config.canopy_layers.max_layers``."""
        ...


# =============================================================================
# Forcing                            (replaces TowerMetMod / clmDataMod per-step reads)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Forcing:
    """Everything prescribed from outside the model. All float fields are differentiable.
    Meteorology is per gridcell; prescribed vegetation per patch; soil moisture per column."""

    # Time (shared)
    time: Array  # (nt,) seconds since run start, end of each step
    calendar_day: Array  # (nt,) fractional day of year
    step_seconds: float  # static
    interpolation: Literal["none", "three_point"]  # static

    # Meteorology at measurement height, per gridcell
    air_temperature: Array  # (nt, ng) K
    specific_humidity: Array  # (nt, ng) kg kg-1
    wind_speed: Array  # (nt, ng) m s-1
    air_pressure: Array  # (nt, ng) Pa
    co2: Array  # (nt, ng) umol mol-1
    o2: Array  # (nt, ng) mmol mol-1
    shortwave_direct: Array  # (nt, ng, 2) W m-2
    shortwave_diffuse: Array  # (nt, ng, 2) W m-2
    longwave_down: Array  # (nt, ng) W m-2
    rainfall: Array  # (nt, ng) kg m-2 s-1
    snowfall: Array  # (nt, ng) kg m-2 s-1
    solar_zenith: Array  # (nt, ng) rad, precomputed

    # Prescribed vegetation from CLM history, per patch
    lai: Array  # (nt, np)
    sai: Array  # (nt, np)

    # Prescribed soil state from CLM history, per soil column
    soil_liquid_water: Array  # (nt, nc, nsoil) kg m-2, after adjustment
    soil_ice: Array  # (nt, nc, nsoil) kg m-2
    ground_albedo: Array  # (nt, nc, 2)

    acclimation_temperature: Array  # (ng,) K, mean over the record (was init_acclim)

    def window(self, t: int) -> "ForcingWindow":
        """Previous/current/next records around step ``t``, used by ``Model.step``."""
        ...

    def slice_time(self, start: int, stop: int) -> "Forcing":
        ...


@dataclasses.dataclass(frozen=True)
class ForcingWindow:
    previous: Forcing  # time axis removed
    current: Forcing
    next: Forcing


def load_forcing(forcing_section: Mapping[str, Any], domain: Domain, settings: RunSettings,
                 config: ModelConfig) -> Forcing:
    """Read every file named in run.yaml's ``forcing:`` section (filling ``{tower}``)
    into memory. All netCDF I/O and unit conversion happens here, once."""
    ...


# =============================================================================
# State: prognostic variables only (the lax.scan carry)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CanopyState:
    """Per patch."""

    air_temperature: Array  # (np, nl) K
    vapor_pressure: Array  # (np, nl) Pa
    co2: Array  # (np, nl) umol mol-1
    leaf_temperature: Array  # (np, nl, 2) K
    leaf_water_potential: Array  # (np, nl, 2) MPa
    intercepted_water: Array  # (np, nl) kg m-2
    ground_temperature: Array  # (np,) K  (option B will need a rule relating this to the column)


@dataclasses.dataclass(frozen=True)
class SoilState:
    """Per soil column."""

    temperature: Array  # (nc, nsoil) K
    liquid_water: Array  # (nc, nsoil) kg m-2
    ice: Array  # (nc, nsoil) kg m-2


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
    """Per-patch output. Leading axes ``(nt, np)`` from ``Model.run`` (``(np,)`` from
    ``Model.step``). Unselected variables (``output.variables``) are ``None``.
    ``gridcell_mean()`` returns the same structure with ``np`` replaced by ``ng``."""

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

    # Profiles (profile.out, fluxprofile.out); NaN in padded layers when exported.
    # Profiles are per patch only (patches on a gridcell have different layer grids).
    air_temperature_profile: Array  # (..., nl)
    vapor_pressure_profile: Array
    wind_speed_profile: Array
    leaf_temperature_profile: Array  # (..., nl, 2)
    sensible_heat_profile: Array
    latent_heat_profile: Array
    photosynthesis_profile: Array  # (..., nl, 2)

    # Soil (soiltemp.out), per column
    soil_temperature: Array  # (..., nc, nsoil)

    # Conservation checks
    energy_balance_error: Array
    water_balance_error: Array

    def gridcell_mean(self, domain: Domain) -> "Outputs":
        """Area-weighted mean over each gridcell's patches (``segment_sum`` of
        ``weight * value``). Differentiable. Profile fields are dropped."""
        ...

    def to_xarray(self, domain: Domain, forcing: Forcing, layers: Layers):
        """``xarray.Dataset`` with dims (time, patch, layer, soil_layer, leaf_class, band),
        a ``gridcell`` coordinate on ``patch``, units and long names. Padded layers are NaN."""
        ...

    def to_netcdf(self, path: PathLike, domain: Domain, forcing: Forcing, layers: Layers) -> None:
        ...

    def write_legacy_ascii(self, directory: PathLike, domain: Domain, forcing: Forcing
                           ) -> list[Path]:
        """Six Fortran-format ``.out`` files per patch, byte-compatible with today's output.
        Transitional: removed once the Fortran comparison test reads netCDF."""
        ...


# =============================================================================
# Model                              (replaces CLMml_drv, clm_drv, MLCanopyFluxes driver logic)
# =============================================================================


class Model:
    """Configured model for a domain. Holds only static structure; no mutable state."""

    config: ModelConfig
    domain: Domain
    layers: Layers  # built from domain + config

    def __init__(self, config: ModelConfig, domain: Domain) -> None:
        ...

    def initial_state(self, forcing: Forcing, *, source: Literal["clm_history"] = "clm_history"
                      ) -> State:
        """Cold-start canopy; soil from CLM history at forcing's first step."""
        ...

    def step(self, state: State, forcing: ForcingWindow, params: Params) -> tuple[State, Outputs]:
        """Advance every patch by one forcing step. Gathers gridcell forcing and PFT
        parameters to patches, then vmaps the one-patch step. Pure."""
        ...

    def run(self, state: State, forcing: Forcing, params: Params, *,
            outputs: Union[Literal["all"], Sequence[str]] = "all",
            frequency: Literal["step", "hourly", "daily"] = "step",
            check: bool = False) -> Outputs:
        """Whole run as one compiled call: ``scan`` over time of ``vmap`` over patches.

        Differentiable w.r.t. ``state``, ``forcing`` and ``params`` (and patch weights via
        ``Outputs.gridcell_mean``).
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
# Each process in clm_ml/physics/ acts on ONE patch with (max_layers,) arrays and masks:
#
#     def two_stream(step: StepState, inputs: StepInputs, params: PatchParams,
#                    layers: PatchLayers, config: ModelConfig) -> StepState:
#         """Two-stream solar radiative transfer.
#
#         Reads:  inputs.solar_zenith, inputs.shortwave_direct, step.lai_profile, layers.has_leaf
#         Writes: step.absorbed_shortwave_leaf, step.absorbed_shortwave_soil
#         Fortran reference: MLSolarRadiationMod.F90, subroutine TwoStream
#         """
#
#     SOLAR_RADIATION = {"norman": norman, "two_stream": two_stream}
