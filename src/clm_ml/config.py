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


@dataclasses.dataclass(frozen=True)
class PhysicsOptions:
    stomatal_conductance: Literal["medlyn", "ball_berry", "wue"] = "wue"  # gs_type 0/1/2
    water_stress_on_conductance: bool = True  # gspot_type 1/0
    wue_solver: Literal["brent", "bisection"] = "bisection"  # gs_solver 1/2
    photosynthesis_colimitation: bool = True  # colim_type 1/0
    temperature_acclimation: bool = True  # acclim_type 1/0
    nitrogen_decay_kn: Optional[float] = None  # kn_val (None = computed; was -999)
    roughness_sublayer: Literal["harman_finnigan"] = "harman_finnigan"  # turb_type 1
    sparse_canopy: bool = True  # sparse_canopy_type 1/0
    ground_conductance: Literal["extend_hf", "log_profile"] = "log_profile"  # HF_extension_type
    flux_profile_solution: Literal["well_mixed", "implicit"] = "implicit"  # flux_profile_type
    leaf_boundary_layer: Literal["laminar", "turbulent", "free_convection"] = (
        "free_convection"  # gb_type 1/2/3
    )
    solar_radiation: Literal["norman", "two_stream"] = "two_stream"  # light_type 1/2
    leaf_optics_vary_with_height: bool = False  # leaf_optics_type 0/1
    longwave_radiation: Literal["norman"] = "norman"  # longwave_type 1


@dataclasses.dataclass(frozen=True)
class NumericsOptions:
    substep_seconds: float = 300.0  # dtime_ml
    integrator: Literal[
        "euler",
        "rk2_trapezoid",
        "rk2_midpoint",
        "rk2_ralston",
        "rk3_heun",
        "rk3_ralston",
        "rk3_kutta",
        "rk4",
    ] = "rk4"  # runge_kutta_type
    checkpoint_steps: bool = True  # used from Phase 4 (whole-run compilation)


@dataclasses.dataclass(frozen=True)
class CanopyLayerOptions:
    max_layers: int = 50  # fixed array size for every patch (CHATS7 uses 46)
    dz_tall: float = 0.5  # m
    dz_short: float = 0.1  # m
    tall_canopy_threshold: float = 2.0  # dz_param (m)
    n_layers_above: Optional[int] = None  # nlayer_above (None = from dz)
    n_layers_within: Optional[int] = None  # nlayer_within (None = from dz)
    min_plant_area_per_layer: float = 0.01  # dpai_min (m2/m2)


@dataclasses.dataclass(frozen=True)
class SoilOptions:
    layer_structure: Literal["clm5_0", "clm4_5"] = "clm5_0"  # clm_phys
    # per_patch: each patch has its own soil column (option A; the model's current behavior).
    # shared: patches on a gridcell share one column (option B; not yet implemented).
    columns: Literal["per_patch", "shared"] = "per_patch"


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Contents of ``model.yaml``. Every key is optional; defaults reproduce CLM-ml v2/CHATS."""

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
        if self.numerics.substep_seconds <= 0:
            raise ConfigError("model.yaml: numerics.substep_seconds must be positive")
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

    def to_yaml(self, path: Path | str) -> None:
        """Write the complete configuration, defaults included (useful as a record of a run)."""
        with open(path, "w") as fh:
            yaml.safe_dump(to_dict(self), fh, sort_keys=False)

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

        ``interpolation`` comes from ``run.yaml`` (``forcing.interpolation``), because it
        describes the forcing data rather than the physics.
        """
        p, n, c = self.physics, self.numerics, self.canopy_layers
        rk = _RK_CODES[n.integrator]
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


_GS_TYPE = {"medlyn": 0, "ball_berry": 1, "wue": 2}
_RK_CODES = {
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
    "runge_kutta_type": ("numerics.integrator", {v: k for k, v in _RK_CODES.items()}),
    "flux_profile_type": ("physics.flux_profile_solution", {0: "well_mixed", 1: "implicit"}),
    "gb_type": (
        "physics.leaf_boundary_layer",
        {1: "laminar", 2: "turbulent", 3: "free_convection"},
    ),
}
