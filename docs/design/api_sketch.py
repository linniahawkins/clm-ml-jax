"""
Proposed public API for ``clm_ml`` (design sketch v4: signatures only, no implementation).

Read alongside ``modernization_plan.md`` (vocabulary in §3.1) and the YAML files in
``examples/``.

How the pieces fit
------------------
    run.yaml ──► load_run() ──► Run
      ├─ model.yaml   ──► ModelConfig   (static: changing it recompiles)
      ├─ params.yaml  ──► Params        (per-PFT table; differentiable)
      ├─ towers       ──► Domain        (gridcells, patches, soil columns + index maps)
      │                     └─► Layers  (per patch: fixed max_layers + masks)
      ├─ forcing      ──► ForcingSource ──► Forcing chunks (time, gridcell | patch | column, ...)
      └─ initial cond ──► State         (patch | column, ...); differentiable, carried across chunks

    Run.execute() == for each chunk: state, out = Model.run(state, chunk, params); write out
    Memory depends on the chunk length, not the run length (plan §6.9).

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
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Union

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
class OutputStream:
    """One output file series (run.yaml ``output.streams``). Independent of the forcing
    interval, model step and chunk length."""

    name: str
    frequency: Literal["step", "hourly", "daily", "monthly", "yearly"] = "daily"
    statistic: Literal["mean", "instantaneous", "min", "max"] = "mean"
    variables: Union[Literal["all"], tuple[str, ...]] = "all"
    level: tuple[Literal["patch", "gridcell"], ...] = ("gridcell",)
    file_per: Literal["run", "year", "month"] = "year"


@dataclasses.dataclass(frozen=True)
class RunSettings:
    """The experiment-level parts of run.yaml that are not model inputs."""

    start: str  # ISO datetime
    end: str
    chunk: str = "1 month"  # execution.chunk: forcing read / output written per chunk
    restart_every: Optional[str] = "1 year"  # execution.restart_every (None: end of run only)
    output_directory: Path = Path("output")
    output_streams: tuple[OutputStream, ...] = (OutputStream(name="daily"),)
    source_files: Mapping[str, Path] = dataclasses.field(default_factory=dict)  # provenance


@dataclasses.dataclass(frozen=True)
class Run:
    """Everything needed for one experiment. Forcing is not loaded here: ``forcing_source``
    reads it a chunk at a time."""

    model: "Model"
    params: "Params"
    forcing_source: "ForcingSource"
    initial_state: "State"
    settings: RunSettings

    def execute(self) -> None:
        """Run the whole experiment: loop over chunks (reading the next chunk in the
        background), carry the state, write output streams per chunk and restart files at
        ``restart_every``. Same as ``clm-ml run run.yaml``."""
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
    # running_mean: 10-day (acclimation_days) running mean carried in State (decision 15).
    # run_mean: Fortran CLM-ml v2 (mean over the whole run); for validation against Fortran.
    acclimation_temperature: Literal["running_mean", "run_mean"] = "running_mean"
    acclimation_days: float = 10.0
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
    step_seconds: float = 1800.0  # model time step; forcing streams are interpolated to it
    substep_seconds: float = 300.0  # must divide step_seconds
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
# Forcing streams and chunks         (replaces TowerMetMod / clmDataMod per-step reads)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class StreamSpec:
    """One forcing stream from run.yaml ``forcing:`` (meteorology, vegetation, soil_water)."""

    files: str  # path pattern; {tower}, {year}, {month} are filled in
    source: Literal["file", "clm_history"] = "file"
    # How values between records are obtained (plan §6.9). Defaults reproduce CLM-ml v2 when
    # the data are already at the model step.
    interpolation: Literal["none", "linear", "three_point"] = "linear"
    shortwave: Literal["coszen", "interpolate", "constant"] = "coszen"  # meteorology only
    precipitation: Literal["constant", "interpolate"] = "constant"  # meteorology only


class ForcingSource:
    """Reads forcing streams and produces ``Forcing`` chunks interpolated to the model step.

    Each stream has its own time axis (half-hourly or 6-hourly meteorology, daily or monthly
    LAI/SAI and soil water). Only the files covering the requested time, and only the
    gridcells/patches of this process, are opened. Nothing is held between chunks except
    open file handles.
    """

    def __init__(self, streams: Mapping[str, StreamSpec], domain: "Domain",
                 config: ModelConfig) -> None:
        ...

    def chunk(self, start: str, n_steps: Optional[int] = None, days: Optional[float] = None
              ) -> "Forcing":
        """Forcing for ``n_steps`` model steps from ``start``, plus one record before and
        after (needed by the three-point time interpolation at chunk boundaries)."""
        ...

    def chunks(self, start: str, end: str, length: str) -> Iterator[Forcing]:
        """Consecutive chunks covering ``start``..``end``; the next chunk is read on a
        background thread while the caller runs the current one."""
        ...


@dataclasses.dataclass(frozen=True)
class Forcing:
    """One chunk of forcing, interpolated to the model step. All float fields are
    differentiable. Shapes lead with ``nt + 2`` (the chunk's steps plus one halo record at
    each end). Meteorology is per gridcell; vegetation per patch; soil water per column."""

    # Time (shared)
    time: Array  # (nt+2,) model-step end times, seconds since the run start
    calendar_day: Array  # (nt+2,) fractional day of year
    step_seconds: float  # static: the model step

    # Meteorology at measurement height, per gridcell
    air_temperature: Array  # (nt+2, ng) K
    specific_humidity: Array  # (nt+2, ng) kg kg-1
    wind_speed: Array  # (nt+2, ng) m s-1
    air_pressure: Array  # (nt+2, ng) Pa
    co2: Array  # (nt+2, ng) umol mol-1
    o2: Array  # (nt+2, ng) mmol mol-1
    shortwave_direct: Array  # (nt+2, ng, 2) W m-2
    shortwave_diffuse: Array  # (nt+2, ng, 2) W m-2
    longwave_down: Array  # (nt+2, ng) W m-2
    rainfall: Array  # (nt+2, ng) kg m-2 s-1
    snowfall: Array  # (nt+2, ng) kg m-2 s-1
    solar_zenith: Array  # (nt+2, ng) rad, precomputed

    # Prescribed vegetation, per patch (from daily/monthly or CLM history streams)
    lai: Array  # (nt+2, np)
    sai: Array  # (nt+2, np)

    # Prescribed soil water, per soil column
    soil_liquid_water: Array  # (nt+2, nc, nsoil) kg m-2, after adjustment
    soil_ice: Array  # (nt+2, nc, nsoil) kg m-2

    @property
    def n_steps(self) -> int:
        """Model steps in the chunk (excluding the two halo records)."""
        ...

    def window(self, t: int) -> "ForcingWindow":
        """Previous/current/next records around chunk step ``t`` (uses the halo records at
        the chunk edges)."""
        ...


@dataclasses.dataclass(frozen=True)
class ForcingWindow:
    previous: Forcing  # time axis removed
    current: Forcing
    next: Forcing


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
    acclimation_temperature: Array  # (np,) K, running mean of air temperature (decision 15)


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
    """Per-patch output records of one output stream: leading axes ``(n_records, np)``,
    where records are at the stream's frequency (``(np,)`` from ``Model.step``). Statistics
    are accumulated inside the scan; partial intervals are carried to the next chunk.
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
    rsl_beta: Array  # u*/U(h), roughness-sublayer beta (flux.out col 17; was mislabeled before)
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

    def initial_state(self, forcing: Forcing, *,
                      source: Literal["clm_history", "restart"] = "clm_history",
                      restart_file: Optional[PathLike] = None) -> State:
        """Cold-start canopy and soil from CLM history at the first step, or a restart file."""
        ...

    def step(self, state: State, forcing: ForcingWindow, params: Params) -> tuple[State, Outputs]:
        """Advance every patch by one forcing step. Gathers gridcell forcing and PFT
        parameters to patches, then vmaps the one-patch step. Pure."""
        ...

    def run(self, state: State, forcing: Forcing, params: Params, *,
            output_streams: Sequence[OutputStream] = (OutputStream(name="step",
                                                                    frequency="step"),),
            accumulators: Optional[Any] = None,
            check: bool = False) -> tuple[State, Outputs]:
        """One chunk as one compiled call: ``scan`` over the chunk's steps of ``vmap`` over
        patches. Returns the final state (for the next chunk) and completed output records.
        ``accumulators`` carries partial output intervals across chunks.

        Differentiable w.r.t. ``state``, ``forcing`` and ``params`` (and patch weights via
        ``Outputs.gridcell_mean``). For long runs, take gradients over a window of chunks
        starting from a saved state (plan §6.9).
        """
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
