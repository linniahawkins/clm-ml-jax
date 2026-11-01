"""
Model configuration: the contents of ``model.yaml``.

These settings choose physics schemes and numerics. They are *static*: changing any of
them changes the compiled program, and none of them are differentiable. Numeric values
that should be calibrated belong in ``params.yaml`` instead.

Each option's comment names the Fortran ``MLclm_varctl`` variable and code it replaces.
:data:`FORTRAN_CODES` holds that mapping in both directions.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Optional

from clm_ml.io.yaml_schema import ConfigError, from_dict, option, read_yaml, render, to_dict


@dataclasses.dataclass(frozen=True)
class PhysicsOptions:
    """Physics schemes. The defaults are CLM-ml v2 as run for CHATS (Bonan et al. 2026); the
    JAX model is validated against Fortran with these."""

    stomatal_conductance: Literal["medlyn", "ball_berry", "wue"] = option(
        "wue",
        "Stomatal conductance model: water-use-efficiency optimization (iota) with the plant "
        "hydraulics constraint (CLM-ml v2), or the Medlyn or Ball-Berry relation.",
        "gs_type 0/1/2",
    )
    water_stress_on_conductance: bool = option(
        True,
        "Reduce stomatal conductance as leaf water potential falls, "
        "gs = max(gs / (1 + (psi_leaf/psi50_gs)^shape_gs), gs_min) (Bonan et al. 2021, "
        "supplement eq. 14).",
        "gspot_type 1/0",
    )
    wue_solver: Literal["brent", "bisection"] = option(
        "bisection",
        "Root finder of the WUE optimization: bisection (CLM-ml v2). brent is refused (not "
        "translated as traceable code; decisions.md).",
        "gs_solver 1/2",
    )
    photosynthesis_colimitation: bool = option(
        True,
        "Co-limitation (smoothed minimum) of the Rubisco, light and product limited rates; false "
        "takes the plain minimum.",
        "colim_type 1/0",
    )
    temperature_acclimation: bool = option(
        True,
        "Acclimation of the photosynthetic temperature responses (Kattge and Knorr 2007) to the "
        "acclimation temperature below.",
        "acclim_type 1/0",
    )
    acclimation_temperature: Literal["running_mean", "run_mean"] = option(
        "running_mean",
        "The air temperature photosynthesis acclimates to: a running mean over "
        "acclimation_days, carried in the state (as CLM), or the mean over the whole run "
        "(Fortran CLM-ml v2; needs all forcing before the run starts: validation only).",
        "run_mean = init_acclim (CLMml_driver)",
    )
    acclimation_days: float = option(10.0, "Length of the acclimation running mean (days).", "")
    nitrogen_decay_kn: Optional[float] = option(
        None,
        "Leaf nitrogen (Vcmax25) decay coefficient through the canopy; null computes it from "
        "Vcmax25 at the canopy top, exp(0.00963 Vcmax25 - 2.43) (Bonan et al. 2021).",
        "kn_val (-999 = computed)",
    )
    roughness_sublayer: Literal["harman_finnigan"] = option(
        "harman_finnigan",
        "Turbulence above and within the canopy: Harman and Finnigan roughness sublayer with "
        "the psihat tables (the only scheme).",
        "turb_type 1",
    )
    sparse_canopy: bool = option(
        True,
        "Harman-Finnigan adjustments for sparse canopies: the displacement height and the "
        "canopy-top Prandtl number depend on the plant area index.",
        "sparse_canopy_type 1/0",
    )
    ground_conductance: Literal["extend_hf", "log_profile"] = option(
        "log_profile",
        "Aerodynamic conductance of the lowest canopy layer to the ground: a log wind profile "
        "(CLM-ml v2), or the Harman-Finnigan exponential profile extended down to the ground.",
        "HF_extension_type 2/1",
    )
    flux_profile_solution: Literal["well_mixed", "implicit"] = option(
        "implicit",
        "Coupled implicit solution of air temperature, vapour and leaf temperature in every "
        "layer. well_mixed is refused (Fortran calls it legacy, not supported).",
        "flux_profile_type 1/0",
    )
    leaf_boundary_layer: Literal["laminar", "turbulent", "free_convection"] = option(
        "free_convection",
        "Leaf boundary-layer conductance: laminar forced convection, turbulent forced "
        "convection, or forced plus free convection (CLM-ml v2).",
        "gb_type 1/2/3",
    )
    solar_radiation: Literal["norman", "two_stream"] = option(
        "two_stream",
        "Shortwave radiation through the canopy: two-stream, integrated over each layer. "
        "norman is refused (not yet supported).",
        "light_type 2/1",
    )
    leaf_optics_vary_with_height: bool = option(
        False,
        "Leaf optical properties varying with height in the canopy (not supported).",
        "leaf_optics_type 0/1",
    )
    longwave_radiation: Literal["norman"] = option(
        "norman",
        "Longwave radiation through the canopy: Norman (the only scheme).",
        "longwave_type 1",
    )


@dataclasses.dataclass(frozen=True)
class NumericsOptions:
    """Time stepping, solvers and run control."""

    step_seconds: int = option(
        1800,
        "Model time step (s). It must divide the tower forcing interval (run.yaml "
        "forcing.meteorology.interval_seconds); a shorter step interpolates the forcing.",
        "",
    )
    substep_seconds: float = option(
        300.0,
        "Canopy sub-step (s): each step is split into sub-steps, forcing interpolated within "
        "the step, output averaged over them.",
        "dtime_ml",
    )
    integrator: Literal[
        "euler",
        "rk2_trapezoid",
        "rk2_midpoint",
        "rk2_ralston",
        "rk3_heun",
        "rk3_ralston",
        "rk3_kutta",
        "rk4",
    ] = option(
        "rk4",
        "Time integration of the sub-step state (air temperature, vapour, leaf temperature, "
        "leaf water potential, intercepted water, ground temperature): 4th-order Runge-Kutta "
        "in CLM-ml v2 (v1 used Euler).",
        "runge_kutta_type",
    )
    checkpoint_steps: bool = option(
        True,
        "Recompute each step during gradients instead of storing it (less memory). Currently "
        "has no effect: steps are always checkpointed (open_items.md O65).",
        "",
    )
    on_patch_error: Literal["stop", "exclude"] = option(
        "stop",
        "A patch that fails a conservation check or whose state becomes unhealthy: stop the "
        "run naming it (as Fortran), or exclude it from then on (global runs; its output is "
        "fill values and gridcell sums use the remaining weights).",
        "",
    )
    max_failed_patch_fraction: float = option(
        0.01,
        "With exclude: stop the run when more than this fraction of patches has failed.",
        "",
    )
    ignore_checks: tuple[str, ...] = option(
        (),
        "Conservation checks that neither stop the run nor exclude a patch (their largest "
        "error is still reported), by (part of) their name in clm_ml.checks.BALANCE_CHECKS, "
        "e.g. 'CanopyFluxesDiagnostics 3'. Normally empty: every check passes for CHATS7 and "
        "the 1-degree grid runs.",
        "",
    )


@dataclasses.dataclass(frozen=True)
class CanopyLayerOptions:
    """The vertical layers of the canopy air space, from the ground to the forcing height."""

    max_layers: int = option(
        50,
        "Layer slots of every patch (one compiled program for all patches; unused slots are "
        "masked). CHATS7 uses 46.",
        "nlevmlcan",
    )
    dz_tall: float = option(
        0.5, "Layer thickness (m) for canopies taller than tall_canopy_threshold.", "dz_tall"
    )
    dz_short: float = option(0.1, "Layer thickness (m) for shorter canopies.", "dz_short")
    tall_canopy_threshold: float = option(
        2.0, "Canopy height (m) above which dz_tall is used.", "dz_param"
    )
    n_layers_above: Optional[int] = option(
        None,
        "Explicit number of layers above the canopy; null = from the spacing. Set with "
        "n_layers_within (both or neither); towers with short canopies under tall towers "
        "need it (5-10 within-canopy layers suffice for fluxes, Bonan et al. 2021).",
        "nlayer_above",
    )
    n_layers_within: Optional[int] = option(
        None,
        "Explicit number of layers within the canopy; null = from the spacing.",
        "nlayer_within",
    )
    min_plant_area_per_layer: float = option(
        0.01,
        "At setup, layers holding less than this fraction of the canopy's plant area are "
        "emptied and their area given to the other layers.",
        "dpai_min",
    )
    min_layer_plant_area: float = option(
        0.01,
        "Every leafy layer keeps at least this plant area (m2 m-2) at every step, the missing "
        "part added as stem area: thinner layers transmit more diffuse radiation than they "
        "receive (decisions.md; CLM-ml v1's dpai_min). 0 reproduces CLM-ml v2.",
        "",
    )


@dataclasses.dataclass(frozen=True)
class SoilOptions:
    """The soil under the patches (the CLM soil layer structure: 20 hydrologically active
    levels, 25 in all)."""

    columns: Literal["per_patch", "shared"] = option(
        "per_patch",
        "per_patch: each patch has its own soil column (option A). shared: patches on a "
        "gridcell share one column (option B; not yet implemented).",
        "",
    )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Contents of ``model.yaml``; the defaults reproduce CLM-ml v2/CHATS."""

    physics: PhysicsOptions = PhysicsOptions()
    numerics: NumericsOptions = NumericsOptions()
    canopy_layers: CanopyLayerOptions = CanopyLayerOptions()
    soil: SoilOptions = SoilOptions()

    def __post_init__(self):
        if self.soil.columns == "shared":
            raise ConfigError(
                "model.yaml: soil.columns: 'shared' (option B) is planned but not implemented; "
                "use 'per_patch'"
            )
        if self.physics.acclimation_days <= 0:
            raise ConfigError("model.yaml: physics.acclimation_days must be positive")
        if self.numerics.substep_seconds <= 0:
            raise ConfigError("model.yaml: numerics.substep_seconds must be positive")
        if not 0.0 <= self.numerics.max_failed_patch_fraction <= 1.0:
            raise ConfigError("model.yaml: numerics.max_failed_patch_fraction must be in [0, 1]")
        if self.canopy_layers.max_layers < 2:
            raise ConfigError("model.yaml: canopy_layers.max_layers must be at least 2")
        within, above = self.canopy_layers.n_layers_within, self.canopy_layers.n_layers_above
        if (within is None) != (above is None):
            raise ConfigError(
                "model.yaml: canopy_layers.n_layers_within and n_layers_above must be set "
                "together (or both left null to use dz_tall/dz_short)"
            )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ModelConfig":
        path = Path(path)
        return cls.from_dict(read_yaml(path), source=path.name)

    @classmethod
    def from_dict(cls, data: dict, source: str = "model.yaml") -> "ModelConfig":
        return from_dict(cls, data, source=source)

    def to_yaml(self, path: Path | str, generated: str = "") -> None:
        """Write the complete configuration, every key with its explanation
        (:func:`model_yaml_text`)."""
        Path(path).write_text(model_yaml_text(self, generated))

    def replace(self, **dotted_changes) -> "ModelConfig":
        """Return a copy with options changed, e.g.
        ``config.replace(**{"physics.stomatal_conductance": "medlyn"})``."""
        data = to_dict(self)
        for dotted, value in dotted_changes.items():
            section, _, name = dotted.partition(".")
            if section not in data or not name:
                raise ConfigError(f"replace: unknown option {dotted!r}")
            data[section][name] = value
        return ModelConfig.from_dict(data, source="replace()")

    def to_fortran_codes(self, interpolation: str) -> dict:
        """Values for the legacy ``MLclm_varctl`` module variables.

        ``interpolation`` comes from ``run.yaml`` (``forcing.meteorology.interpolation``), because it
        describes the forcing data rather than the physics.
        """
        p, n, c = self.physics, self.numerics, self.canopy_layers
        rk = INTEGRATOR_CODES[n.integrator]
        return {
            "gs_type": _GS_TYPE[p.stomatal_conductance],
            "gspot_type": int(p.water_stress_on_conductance),
            "gs_solver": {"brent": 1, "bisection": 2}[p.wue_solver],
            "colim_type": int(p.photosynthesis_colimitation),
            "acclim_type": int(p.temperature_acclimation),
            "kn_val": -999.0 if p.nitrogen_decay_kn is None else float(p.nitrogen_decay_kn),
            "turb_type": 1,
            "sparse_canopy_type": int(p.sparse_canopy),
            "HF_extension_type": {"extend_hf": 1, "log_profile": 2}[p.ground_conductance],
            "flux_profile_type": {"well_mixed": 0, "implicit": 1}[p.flux_profile_solution],
            "gb_type": {"laminar": 1, "turbulent": 2, "free_convection": 3}[p.leaf_boundary_layer],
            "light_type": {"norman": 1, "two_stream": 2}[p.solar_radiation],
            "leaf_optics_type": int(p.leaf_optics_vary_with_height),
            "longwave_type": 1,
            "dtime_ml": float(n.substep_seconds),
            "runge_kutta_type": rk,
            "nrk": rk // 10,
            "dz_tall": float(c.dz_tall),
            "dz_short": float(c.dz_short),
            "dz_param": float(c.tall_canopy_threshold),
            "nlayer_above": c.n_layers_above or 0,
            "nlayer_within": c.n_layers_within or 0,
            "dpai_min": float(c.min_plant_area_per_layer),
            "layer_pai_floor": float(c.min_layer_plant_area),
            "max_layers": int(c.max_layers),
            "met_type": {"none": 0, "three_point": 3}[interpolation],
        }

    def non_default_options(self) -> list[str]:
        """Dotted names of options that differ from the CLM-ml v2/CHATS defaults.

        The JAX translation has been validated against Fortran only with the defaults;
        other options are translated but not yet validated, so ``clm-ml check`` lists them.
        """
        default, mine = to_dict(ModelConfig()), to_dict(self)
        return [
            f"{section}.{name}"
            for section, values in mine.items()
            for name, value in values.items()
            if value != default[section][name]
            and section in ("physics", "numerics")
            and name != "checkpoint_steps"
        ]


#: Header of model.yaml in an experiment folder
HEADER = """\
model.yaml: how the physics and numerics work. Part of an experiment folder (run.yaml,
domain.yaml, model.yaml, params.yaml). Every setting is listed; the defaults are CLM-ml v2 as
run for CHATS (Bonan et al. 2026), the configuration validated against Fortran. Changing
anything here recompiles the model; none of these settings is differentiable (calibrated
numbers belong in params.yaml). {differs} {generated}"""


def model_yaml_text(config: ModelConfig, generated: str = "") -> str:
    """The complete model.yaml of ``config``: every section and key, each with its
    explanation, its choices and the Fortran variable it replaces, as comments
    (:func:`clm_ml.io.yaml_schema.render`). The header lists the settings that differ from
    the defaults."""
    default, mine = to_dict(ModelConfig()), to_dict(config)
    changed = [f"{sec}.{k}" for sec, vals in mine.items() for k, v in vals.items()
               if v != default[sec][k]]  # fmt: skip
    differs = (
        "Differs from the defaults in: " + ", ".join(changed) + "."
        if changed
        else "All settings are at their defaults."
    )
    return render(config, HEADER.format(differs=differs, generated=generated).strip())


_GS_TYPE = {"medlyn": 0, "ball_berry": 1, "wue": 2}
INTEGRATOR_CODES = {
    "euler": 10,
    "rk2_trapezoid": 21,
    "rk2_midpoint": 22,
    "rk2_ralston": 23,
    "rk3_heun": 31,
    "rk3_ralston": 32,
    "rk3_kutta": 33,
    "rk4": 41,
}

#: Fortran namelist/varctl codes -> readable model.yaml settings (used by ``clm-ml convert``).
FORTRAN_CODES = {
    "gs_type": ("physics.stomatal_conductance", {v: k for k, v in _GS_TYPE.items()}),
    "light_type": ("physics.solar_radiation", {1: "norman", 2: "two_stream"}),
    "runge_kutta_type": ("numerics.integrator", {v: k for k, v in INTEGRATOR_CODES.items()}),
    "flux_profile_type": ("physics.flux_profile_solution", {0: "well_mixed", 1: "implicit"}),
    "gb_type": (
        "physics.leaf_boundary_layer",
        {1: "laminar", 2: "turbulent", 3: "free_convection"},
    ),
}
