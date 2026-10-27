"""
Proposed public API for ``clmml`` (design sketch: signatures only, no implementation).

This file is for review alongside ``modernization_plan.md``. Every name, field and
default here is a proposal. Comment on anything that reads oddly to you as a user.

Conventions
-----------
- Every container is a frozen dataclass registered as a JAX pytree.
- Fields marked ``static`` are compile-time constants: changing them recompiles and
  they are not differentiable. All other array fields are traced and differentiable.
- Units are SI unless noted, and are given in each field comment.
- Shapes: ``nt`` = tower time steps, ``nlay`` = canopy layers, ``nsoil`` = soil layers.
  Leaf arrays end in a ``(2,)`` axis ``[sunlit, shaded]``. Spectral arrays end in a
  ``(2,)`` axis ``[vis, nir]``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp

Array = jax.Array
PathLike = Union[str, Path]


# =============================================================================
# Config: scheme choices and numerics (static)
# Replaces: MLclm_varctl module globals and the namelist run options
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Config:
    """How the model is run. Hashable and static: changing it recompiles.

    Each string option lists the Fortran integer code it replaces, so existing
    namelists can be translated by ``clmml.legacy.from_namelist``.
    """

    # --- Photosynthesis and stomata ----------------------------------------
    stomatal_conductance: Literal["medlyn", "ball_berry", "wue"] = "wue"  # gs_type 0/1/2
    water_stress_on_conductance: bool = True  # gspot_type 1
    wue_solver: Literal["brent", "bisection"] = "bisection"  # gs_solver 1/2
    photosynthesis_colimitation: bool = True  # colim_type 1
    temperature_acclimation: bool = True  # acclim_type 1
    nitrogen_decay_kn: Optional[float] = None  # kn_val (None = computed)

    # --- Turbulence ---------------------------------------------------------
    roughness_sublayer: Literal["harman_finnigan"] = "harman_finnigan"  # turb_type 1
    sparse_canopy: bool = True  # sparse_canopy_type 1
    ground_conductance: Literal["extend_hf", "log_profile"] = "log_profile"  # HF_extension_type
    flux_profile_solution: Literal["well_mixed", "implicit"] = "implicit"  # flux_profile_type
    leaf_boundary_layer: Literal["laminar", "turbulent", "free_convection"] = "free_convection"

    # --- Radiation ----------------------------------------------------------
    solar_radiation: Literal["norman", "two_stream"] = "two_stream"  # light_type 1/2
    leaf_optics_vary_with_height: bool = False  # leaf_optics_type 0/1
    longwave_radiation: Literal["norman"] = "norman"  # longwave_type 1

    # --- Time stepping ------------------------------------------------------
    substep_seconds: float = 300.0  # dtime_ml; must divide the tower step evenly
    integrator: Literal["euler", "rk2_trapezoid", "rk2_midpoint", "rk2_ralston",
                        "rk3_heun", "rk3_ralston", "rk3_kutta", "rk4"] = "rk4"
    forcing_interpolation: Literal["none", "three_point"] = "three_point"  # met_type 0/3

    # --- Canopy layering ----------------------------------------------------
    dz_tall: float = 0.5  # layer thickness for tall canopies (m)
    dz_short: float = 0.1  # layer thickness for short canopies (m)
    tall_canopy_threshold: float = 2.0  # dz_param (m)
    n_layers_above: Optional[int] = None  # nlayer_above (None = from dz)
    n_layers_within: Optional[int] = None  # nlayer_within (None = from dz)
    min_plant_area_per_layer: float = 0.01  # dpai_min (m2/m2)

    # --- Memory / performance -------------------------------------------------
    checkpoint_steps: bool = True  # jax.checkpoint each step (less memory for grad)

    def replace(self, **changes) -> "Config":
        """Return a copy with some options changed."""
        ...


# =============================================================================
# Site: where and what is being simulated
# Replaces: TowerDataMod tables (tower_id, tower_lat, tower_pft, ...)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Site:
    """Fixed description of one tower site."""

    name: str
    latitude: float  # degrees north
    longitude: float  # degrees east
    pft: int  # CLM plant functional type index
    sand_percent: float
    clay_percent: float
    organic_matter: float  # kg/m3
    soil_color: int  # isoicol
    bedrock_depth: float  # zbed (m)
    reference_height: float  # tower measurement height (m)
    canopy_height: float  # top of canopy (m)
    root_biomass: float  # g/m2
    lai_beta_params: tuple[float, float]  # beta distribution shape of leaf area profile
    sai_beta_params: tuple[float, float]  # beta distribution shape of stem area profile
    utc_offset_hours: float = 0.0  # tower_time

    @classmethod
    def from_tower(cls, name: str) -> "Site":
        """Look up a built-in tower (e.g. ``"CHATS7"``, ``"US-UMB"``) in ``data/towers.csv``."""
        ...

    @classmethod
    def available_towers(cls) -> list[str]:
        """Names of all built-in towers."""
        ...


# =============================================================================
# Layers: derived vertical grid (static)
# Replaces: MLinitVerticalMod, GridInfo, ml_vert_init, ntop/nbot/ncan fields
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Layers:
    """Canopy vertical discretization, computed once from ``Site`` + ``Config``.

    Users normally do not build this directly; ``Model`` does it.
    """

    n_layers: int  # static: total canopy-air layers (was ncan)
    top_leaf_layer: int  # static: 0-based (was ntop)
    bottom_leaf_layer: int  # static: 0-based (was nbot)
    heights: Array  # (nlay,) layer mid-point heights (m)
    interface_heights: Array  # (nlay + 1,) layer edges (m), ground = 0
    has_leaf: Array  # (nlay,) bool mask, True where plant area > min_plant_area_per_layer
    n_soil_layers: int  # static


# =============================================================================
# Params: tunable, differentiable parameters
# Replaces: MLpftcon global, vcmaxpft_jax / g1_MED_jax special arguments
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Params:
    """Physical parameters for the site's PFT. Every field is differentiable.

    Scalars here are the value for *this site's* PFT, not per-PFT tables;
    ``default_params`` does the PFT lookup.
    """

    # Photosynthesis / stomata
    vcmax25: Array  # max carboxylation rate at 25 C, top of canopy (umol/m2/s)
    g0_medlyn: Array  # mol H2O/m2/s
    g1_medlyn: Array  # -
    g0_ball_berry: Array  # mol H2O/m2/s
    g1_ball_berry: Array  # -
    iota: Array  # stomatal water-use efficiency (umol CO2/mol H2O)
    gs_min: Array  # minimum stomatal conductance (mol H2O/m2/s)
    psi50_gs: Array  # leaf water potential at 50% conductance loss (MPa)
    shape_gs: Array  # -

    # Plant hydraulics
    plant_conductance: Array  # gplant (mmol H2O/m2 leaf/s/MPa)
    plant_capacitance: Array  # capac (mmol H2O/m2 leaf/MPa)
    root_radius: Array  # m
    root_density: Array  # g biomass/m3 root
    root_resistivity: Array  # MPa s g/mmol H2O

    # Radiation
    leaf_emissivity: Array  # -
    clumping_index: Array  # -

    def replace(self, **changes) -> "Params":
        """Return a copy with some parameters changed (values may be JAX tracers)."""
        ...

    def describe(self) -> str:
        """Table of parameter names, values, units, and valid ranges."""
        ...


def default_params(site: Site, preset: Literal["default", "chats"] = "default") -> Params:
    """Default parameters for the site's PFT.

    ``preset="chats"`` applies the CHATS overrides (was namelist ``pftcon_val = 1``).
    """
    ...


def perturb(params: Params, **batched_values: Array) -> Params:
    """Build a batched ``Params`` for ``jax.vmap`` ensembles.

    Example::

        batch = perturb(params, vcmax25=jnp.linspace(60, 140, 32))
        # batch.vcmax25.shape == (32,), other fields broadcast to (32,)
    """
    ...


# =============================================================================
# Forcing: all external inputs, loaded before the run
# Replaces: TowerMetMod / clmDataMod per-step reads, *_forcing / *_bef / *_next fields,
#           clm_time_manager globals
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Forcing:
    """Time series of everything prescribed from outside the model. Shapes lead with ``nt``."""

    # Time
    time: Array  # (nt,) seconds since start, end of each tower step
    calendar_day: Array  # (nt,) fractional day of year (1.0 = Jan 1 00Z)
    step_seconds: float  # static: tower step length, e.g. 1800.0

    # Tower meteorology at reference height
    air_temperature: Array  # (nt,) K
    specific_humidity: Array  # (nt,) kg/kg
    wind_speed: Array  # (nt,) m/s
    air_pressure: Array  # (nt,) Pa
    co2: Array  # (nt,) umol/mol
    o2: Array  # (nt,) mmol/mol
    shortwave_direct: Array  # (nt, 2) W/m2 [vis, nir]
    shortwave_diffuse: Array  # (nt, 2) W/m2 [vis, nir]
    longwave_down: Array  # (nt,) W/m2
    rainfall: Array  # (nt,) kg H2O/m2/s
    snowfall: Array  # (nt,) kg H2O/m2/s
    solar_zenith: Array  # (nt,) radians (precomputed from orbit + site)

    # Prescribed land state from CLM history (was clmData)
    lai: Array  # (nt,) m2/m2
    sai: Array  # (nt,) m2/m2
    soil_liquid_water: Array  # (nt, nsoil) kg/m2, after soil-moisture adjustment
    soil_ice: Array  # (nt, nsoil) kg/m2
    ground_albedo: Array  # (nt, 2) [vis, nir]

    # Acclimation (was init_acclim): mean air temperature over the record
    acclimation_temperature: Array  # () K

    @property
    def n_steps(self) -> int:
        ...

    def at_step(self, t: int) -> "ForcingWindow":
        """Previous/current/next records around step ``t`` (for ``Model.step``)."""
        ...

    def slice(self, start: int, stop: int) -> "Forcing":
        """Sub-period, e.g. for chunked runs."""
        ...


@dataclasses.dataclass(frozen=True)
class ForcingWindow:
    """Three consecutive tower records used by three-point time interpolation."""

    previous: Forcing  # each field has the leading nt axis removed
    current: Forcing
    next: Forcing


def load_tower_forcing(
    site: Site,
    tower_file: PathLike,
    clm_history_file: PathLike,
    soil_adjust_file: Optional[PathLike] = None,
    *,
    start: str,  # ISO date, e.g. "2007-05-01"
    days: Optional[float] = None,  # run length; None = to end of tower file
    clm_history_start: Optional[str] = None,  # default: read from file metadata
    soil_adjust_levels: int = 3,  # nlev_soil_adjust
) -> Forcing:
    """Read tower meteorology and CLM history data into memory as a ``Forcing``.

    All netCDF reading and unit conversion happens here, once.
    """
    ...


# =============================================================================
# State: prognostic variables only (the lax.scan carry)
# Replaces: prognostic parts of mlcanopy_type, temperature_inst, waterstatebulk_inst, ...
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CanopyState:
    air_temperature: Array  # (nlay,) K
    vapor_pressure: Array  # (nlay,) Pa
    co2: Array  # (nlay,) umol/mol
    leaf_temperature: Array  # (nlay, 2) K [sunlit, shaded]
    leaf_water_potential: Array  # (nlay, 2) MPa
    intercepted_water: Array  # (nlay,) kg H2O/m2
    ground_temperature: Array  # () K


@dataclasses.dataclass(frozen=True)
class SoilState:
    temperature: Array  # (nsoil,) K
    liquid_water: Array  # (nsoil,) kg/m2
    ice: Array  # (nsoil,) kg/m2


@dataclasses.dataclass(frozen=True)
class State:
    canopy: CanopyState
    soil: SoilState
    step_index: Array  # () int, number of tower steps completed


# =============================================================================
# Outputs: per-step results, stacked over time by Model.run
# Replaces: the six ASCII .out files and *_canopy diagnostic fields
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Outputs:
    """Model output. From ``Model.step`` the fields have no time axis; from ``Model.run``
    they have a leading ``(nt,)`` axis. Names follow CF conventions where one exists.
    """

    # Canopy-scale fluxes (flux.out)
    net_radiation: Array  # W/m2 (rnet)
    canopy_air_storage_flux: Array  # W/m2 (stflx_air)
    sensible_heat_flux: Array  # W/m2 (shflx)
    latent_heat_flux: Array  # W/m2 (lhflx)
    gpp: Array  # umol CO2/m2/s (gppveg)
    friction_velocity: Array  # m/s (ustar)
    shortwave_up: Array  # W/m2
    longwave_up: Array  # W/m2
    canopy_top_air_temperature: Array  # K
    ground_heat_flux: Array  # W/m2 (gsoi)
    soil_net_radiation: Array  # W/m2 (rnsoi)
    soil_sensible_heat_flux: Array  # W/m2 (shsoi)
    soil_latent_heat_flux: Array  # W/m2 (lhsoi)
    transpiration_latent_heat: Array  # W/m2
    canopy_evaporation_latent_heat: Array  # W/m2
    soil_wetness_factor: Array  # - (beta)
    vegetation_storage_flux: Array  # W/m2 (stflx_veg)

    # Sunlit/shaded totals (fsun.out), trailing (3,) axis = [total, sunlit, shaded]
    leaf_area_sunlit_shaded: Array  # (2,) m2/m2
    absorbed_par: Array  # (3,) W/m2
    gpp_by_leaf_class: Array  # (3,)
    latent_heat_by_leaf_class: Array  # (3,)
    sensible_heat_by_leaf_class: Array  # (3,)
    vcmax25_by_leaf_class: Array  # (3,)
    stomatal_conductance_by_leaf_class: Array  # (3,)
    leaf_temperature_by_leaf_class: Array  # (3,)

    # Vertical profiles (profile.out, fluxprofile.out)
    air_temperature_profile: Array  # (nlay,) K
    vapor_pressure_profile: Array  # (nlay,) Pa
    wind_speed_profile: Array  # (nlay,) m/s
    leaf_temperature_profile: Array  # (nlay, 2) K
    sensible_heat_profile: Array  # (nlay,) W/m2
    latent_heat_profile: Array  # (nlay,) W/m2
    photosynthesis_profile: Array  # (nlay, 2) umol/m2/s

    # Soil (soiltemp.out)
    soil_temperature: Array  # (nsoil,) K

    # Conservation checks (new)
    energy_balance_error: Array  # W/m2
    water_balance_error: Array  # kg/m2/s

    def to_xarray(self, forcing: Optional[Forcing] = None, layers: Optional[Layers] = None):
        """Return an ``xarray.Dataset`` with time/height coordinates, units and long names."""
        ...

    def write_legacy_ascii(self, directory: PathLike, site: Site, forcing: Forcing) -> list[Path]:
        """Write the six Fortran-format ``.out`` files (byte-compatible with today's output)."""
        ...


# =============================================================================
# Model: ties it together
# Replaces: CLMml_drv, clm_drv, MLCanopyFluxes driver logic, make_clm_ml_forward
# =============================================================================


class Model:
    """A configured single-site model. Holds only static structure (no mutable state).

    Example::

        model = Model(Site.from_tower("CHATS7"), Config())
        params = default_params(model.site, preset="chats")
        state0 = model.initial_state(forcing)
        result = model.run(state0, forcing, params)
    """

    site: Site
    config: Config
    layers: Layers  # built in __init__ from site + config

    def __init__(self, site: Site, config: Config = Config()) -> None:
        ...

    def initial_state(
        self,
        forcing: Forcing,
        *,
        soil_from_clm_history: bool = True,
    ) -> State:
        """Cold-start canopy state; soil temperature/moisture from CLM history at
        ``forcing``'s first step (was SoilInit + init_cold)."""
        ...

    def step(
        self,
        state: State,
        forcing: ForcingWindow,
        params: Params,
    ) -> tuple[State, Outputs]:
        """Advance one tower time step (including all sub-steps). Pure; jit-able; grad-able."""
        ...

    def run(
        self,
        state: State,
        forcing: Forcing,
        params: Params,
        *,
        outputs: Optional[Sequence[str]] = None,  # None = all; else names of Outputs fields
        check: bool = False,  # enable checkify runtime checks (NaNs, balance errors)
    ) -> Outputs:
        """Run over every step in ``forcing`` as one compiled ``lax.scan``.

        Differentiable w.r.t. ``state``, ``forcing`` and ``params``; vmap-able over any
        of them. Unselected outputs are filled with empty arrays to save memory.
        """
        ...

    def run_with_final_state(
        self, state: State, forcing: Forcing, params: Params, **kwargs
    ) -> tuple[State, Outputs]:
        """Like ``run`` but also returns the final state (for chunked/restarted runs)."""
        ...


# =============================================================================
# Legacy compatibility
# =============================================================================


class legacy:  # in the real package this is the module ``clmml.legacy``
    @staticmethod
    def from_namelist(path: PathLike) -> tuple[Config, Site, dict]:
        """Translate a Fortran ``clmML_inparm`` namelist into new-API objects.

        Returns ``(config, site, files)`` where ``files`` holds the input file paths and
        run dates, ready to pass to ``load_tower_forcing``.
        """
        ...

    @staticmethod
    def config_from_fortran_codes(**codes: int) -> Config:
        """E.g. ``config_from_fortran_codes(gs_type=2, light_type=2, runge_kutta_type=41)``."""
        ...


# =============================================================================
# Process interface (for contributors, not end users)
# =============================================================================
#
# Each process in clmml/physics/ is a pure function:
#
#     def two_stream(step: StepState, inputs: StepInputs, params: Params,
#                    layers: Layers, config: Config) -> StepState:
#         """Two-stream solar radiative transfer.
#
#         Reads:  inputs.solar_zenith, inputs.shortwave_direct, ..., step.lai_profile
#         Writes: step.absorbed_shortwave_leaf, step.absorbed_shortwave_soil, ...
#         Fortran reference: MLSolarRadiationMod.F90, subroutine TwoStream
#         """
#
# Variants are registered by name and chosen through Config:
#
#     SOLAR_RADIATION = {"norman": norman, "two_stream": two_stream}
#     STOMATAL_CONDUCTANCE = {"medlyn": medlyn, "ball_berry": ball_berry, "wue": wue}
#
# StepState (prognostic state + intermediate fluxes within a sub-step) and StepInputs
# (interpolated forcing for the sub-step) are internal types defined in Phase 3/5.
