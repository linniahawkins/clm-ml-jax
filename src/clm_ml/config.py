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

import yaml

from clm_ml.io.yaml_schema import ConfigError, from_dict, read_yaml, to_dict


def _opt(default, doc: str, fortran: str = ""):
    """A model.yaml option: its default, what it means (``doc``, written into the generated
    ``configs/model.yaml``) and the Fortran ``MLclm_varctl`` variable it replaces."""
    return dataclasses.field(default=default, metadata={"doc": doc, "fortran": fortran})


@dataclasses.dataclass(frozen=True)
class PhysicsOptions:
    """Physics schemes. The defaults are CLM-ml v2 as run for CHATS (Bonan et al. 2026); the
    JAX model is validated against Fortran with these."""

    stomatal_conductance: Literal["medlyn", "ball_berry", "wue"] = _opt(
        "wue",
        "Stomatal conductance model: water-use-efficiency optimization (iota) with the plant "
        "hydraulics constraint (CLM-ml v2), or the Medlyn or Ball-Berry relation.",
        "gs_type 0/1/2",
    )
    water_stress_on_conductance: bool = _opt(
        True,
        "Reduce stomatal conductance as leaf water potential falls, "
        "gs = max(gs / (1 + (psi_leaf/psi50_gs)^shape_gs), gs_min) (Bonan et al. 2021, "
        "supplement eq. 14).",
        "gspot_type 1/0",
    )
    wue_solver: Literal["brent", "bisection"] = _opt(
        "bisection",
        "Root finder of the WUE optimization: bisection (CLM-ml v2). brent is refused (not "
        "translated as traceable code; decisions.md).",
        "gs_solver 1/2",
    )
    photosynthesis_colimitation: bool = _opt(
        True,
        "Co-limitation (smoothed minimum) of the Rubisco, light and product limited rates; false "
        "takes the plain minimum.",
        "colim_type 1/0",
    )
    temperature_acclimation: bool = _opt(
        True,
        "Acclimation of the photosynthetic temperature responses (Kattge and Knorr 2007) to the "
        "acclimation temperature below.",
        "acclim_type 1/0",
    )
    acclimation_temperature: Literal["running_mean", "run_mean"] = _opt(
        "running_mean",
        "The air temperature photosynthesis acclimates to: a running mean over "
        "acclimation_days, carried in the state (as CLM), or the mean over the whole run "
        "(Fortran CLM-ml v2; needs all forcing before the run starts: validation only).",
        "run_mean = init_acclim (CLMml_driver)",
    )
    acclimation_days: float = _opt(10.0, "Length of the acclimation running mean (days).", "")
    nitrogen_decay_kn: Optional[float] = _opt(
        None,
        "Leaf nitrogen (Vcmax25) decay coefficient through the canopy; null computes it from "
        "Vcmax25 at the canopy top, exp(0.00963 Vcmax25 - 2.43) (Bonan et al. 2021).",
        "kn_val (-999 = computed)",
    )
    roughness_sublayer: Literal["harman_finnigan"] = _opt(
        "harman_finnigan",
        "Turbulence above and within the canopy: Harman and Finnigan roughness sublayer with "
        "the psihat tables (the only scheme).",
        "turb_type 1",
    )
    sparse_canopy: bool = _opt(
        True,
        "Harman-Finnigan adjustments for sparse canopies: the displacement height and the "
        "canopy-top Prandtl number depend on the plant area index.",
        "sparse_canopy_type 1/0",
    )
    ground_conductance: Literal["extend_hf", "log_profile"] = _opt(
        "log_profile",
        "Aerodynamic conductance of the lowest canopy layer to the ground: a log wind profile "
        "(CLM-ml v2), or the Harman-Finnigan exponential profile extended down to the ground.",
        "HF_extension_type 2/1",
    )
    flux_profile_solution: Literal["well_mixed", "implicit"] = _opt(
        "implicit",
        "Coupled implicit solution of air temperature, vapour and leaf temperature in every "
        "layer. well_mixed is refused (Fortran calls it legacy, not supported).",
        "flux_profile_type 1/0",
    )
    leaf_boundary_layer: Literal["laminar", "turbulent", "free_convection"] = _opt(
        "free_convection",
        "Leaf boundary-layer conductance: laminar forced convection, turbulent forced "
        "convection, or forced plus free convection (CLM-ml v2).",
        "gb_type 1/2/3",
    )
    solar_radiation: Literal["norman", "two_stream"] = _opt(
        "two_stream",
        "Shortwave radiation through the canopy: two-stream, integrated over each layer. "
        "norman is refused (not yet supported).",
        "light_type 2/1",
    )
    leaf_optics_vary_with_height: bool = _opt(
        False,
        "Leaf optical properties varying with height in the canopy (not supported).",
        "leaf_optics_type 0/1",
    )
    longwave_radiation: Literal["norman"] = _opt(
        "norman",
        "Longwave radiation through the canopy: Norman (the only scheme).",
        "longwave_type 1",
    )


@dataclasses.dataclass(frozen=True)
class NumericsOptions:
    """Time stepping, solvers and run control."""

    step_seconds: Optional[int] = _opt(
        None,
        "Model time step (s); null = the forcing data interval. The forcing is interpolated "
        "to it; it must divide the data interval. run.yaml time.step_seconds sets the same.",
        "",
    )
    substep_seconds: float = _opt(
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
    ] = _opt(
        "rk4",
        "Time integration of the sub-step state (air temperature, vapour, leaf temperature, "
        "leaf water potential, intercepted water, ground temperature): 4th-order Runge-Kutta "
        "in CLM-ml v2 (v1 used Euler).",
        "runge_kutta_type",
    )
    checkpoint_steps: bool = _opt(
        True,
        "Recompute each step during gradients instead of storing it (less memory). Currently "
        "has no effect: steps are always checkpointed (open_items.md O65).",
        "",
    )
    on_patch_error: Literal["stop", "exclude"] = _opt(
        "stop",
        "A patch that fails a conservation check or whose state becomes unhealthy: stop the "
        "run naming it (as Fortran), or exclude it from then on (global runs; its output is "
        "fill values and gridcell sums use the remaining weights).",
        "",
    )
    max_failed_patch_fraction: float = _opt(
        0.01,
        "With exclude: stop the run when more than this fraction of patches has failed.",
        "",
    )
    ignore_checks: tuple[str, ...] = _opt(
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

    max_layers: int = _opt(
        50,
        "Layer slots of every patch (one compiled program for all patches; unused slots are "
        "masked). CHATS7 uses 46.",
        "nlevmlcan",
    )
    dz_tall: float = _opt(
        0.5, "Layer thickness (m) for canopies taller than tall_canopy_threshold.", "dz_tall"
    )
    dz_short: float = _opt(0.1, "Layer thickness (m) for shorter canopies.", "dz_short")
    tall_canopy_threshold: float = _opt(
        2.0, "Canopy height (m) above which dz_tall is used.", "dz_param"
    )
    n_layers_above: Optional[int] = _opt(
        None,
        "Explicit number of layers above the canopy; null = from the spacing. Set with "
        "n_layers_within (both or neither); towers with short canopies under tall towers "
        "need it (5-10 within-canopy layers suffice for fluxes, Bonan et al. 2021).",
        "nlayer_above",
    )
    n_layers_within: Optional[int] = _opt(
        None,
        "Explicit number of layers within the canopy; null = from the spacing.",
        "nlayer_within",
    )
    min_plant_area_per_layer: float = _opt(
        0.01,
        "At setup, layers holding less than this fraction of the canopy's plant area are "
        "emptied and their area given to the other layers.",
        "dpai_min",
    )
    min_layer_plant_area: float = _opt(
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

    columns: Literal["per_patch", "shared"] = _opt(
        "per_patch",
        "per_patch: each patch has its own soil column (option A). shared: patches on a "
        "gridcell share one column (option B; not yet implemented).",
        "",
    )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Contents of ``model.yaml``. Every key is optional; defaults reproduce CLM-ml v2/CHATS.
    The complete file with every default is ``configs/model.yaml`` (``clm-ml model-yaml``)."""

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

    def to_yaml(self, path: Path | str, header: str = "") -> None:
        """Write the complete configuration, every key with its explanation
        (:func:`model_yaml_text`)."""
        Path(path).write_text(model_yaml_text(self, header))

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


#: Header of the generated complete model.yaml (``configs/model.yaml``)
DEFAULT_HEADER = """\
Model configuration: how the physics and numerics work (model.yaml).
Every setting with its default value: CLM-ml v2 as run for CHATS (Bonan et al. 2026), the
configuration validated against Fortran. Changing anything here recompiles the model; none of
these settings is differentiable (calibrated numbers belong in params.yaml).
Generated from src/clm_ml/config.py by `clm-ml model-yaml` (do not edit by hand: change
config.py and regenerate; tests/clm_ml/test_configs.py checks it)."""


def model_yaml_text(config: ModelConfig, header: str = "") -> str:
    """The complete model.yaml of ``config``: every section and key, each with its
    explanation, its choices and the Fortran variable it replaces, as comments."""
    import textwrap

    def comment(text: str, indent: str) -> list[str]:
        width = 96 - len(indent)
        return [f"{indent}# {line}" for line in textwrap.wrap(text, width)]

    def value(v) -> str:
        if isinstance(v, tuple):
            v = list(v)
        return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()

    if not header:
        default, mine = to_dict(ModelConfig()), to_dict(config)
        changed = [f"{sec}.{k}" for sec, vals in mine.items() for k, v in vals.items()
                   if v != default[sec][k]]  # fmt: skip
        header = (
            DEFAULT_HEADER
            if not changed
            else (
                "Model configuration: how the physics and numerics work (model.yaml).\n"
                "Every setting. Differs from the defaults (configs/model.yaml) in: "
                + ", ".join(changed)
                + ".\nWritten by `clm-ml model-yaml`; changing anything here "
                "recompiles the model."
            )
        )
    lines = []
    for line in "\n".join(
        wrapped for para in header.splitlines() for wrapped in (textwrap.wrap(para, 94) or [""])
    ).splitlines():
        lines.append(f"# {line}".rstrip())
    for section in dataclasses.fields(config):
        options = getattr(config, section.name)
        lines += ["", f"{section.name}:"]
        lines += comment(" ".join(type(options).__doc__.split()), "  ")
        for f in dataclasses.fields(options):
            lines.append("")
            lines += comment(f.metadata.get("doc", ""), "  ")
            notes = []
            literal = _literal_choices(type(options), f.name)
            if literal:
                notes.append("choices: " + " | ".join(literal))
            if f.metadata.get("fortran"):
                notes.append("Fortran: " + f.metadata["fortran"])
            notes.append(f"default: {value(f.default)}")
            lines += comment("; ".join(notes) + ".", "  ")
            lines.append(f"  {f.name}: {value(getattr(options, f.name))}")
    return "\n".join(lines) + "\n"


def _literal_choices(cls, name: str) -> tuple:
    """The allowed values of a ``Literal[...]`` option (empty if not a Literal)."""
    import typing

    hint = typing.get_type_hints(cls)[name]
    if typing.get_origin(hint) is Literal:
        return typing.get_args(hint)
    return ()


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
