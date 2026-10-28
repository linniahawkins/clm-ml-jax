"""
Physical parameters per plant functional type (PFT): the contents of ``params.yaml``.

Every patch looks up its PFT's row, so parameters are shared by all patches of the same
PFT and gradients come back per PFT. All values are differentiable.

A params file usually *extends* another and lists only what changes::

    extends: clm5_defaults          # or a path to another params file
    pft:
      broadleaf_deciduous_temperate_tree:
        vcmax25: 125.0

Built-in files (``clm_ml/data/params/``): ``clm5_defaults`` (CLM5 / CLM-ml v1 values)
and ``chats`` (CHATS walnut orchard; Bonan, Burns & Patton 2026, Table 2).
"""

from __future__ import annotations

import dataclasses
from importlib import resources
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from clm_ml.io.yaml_schema import ConfigError, read_yaml

#: CLM PFT names, in CLM index order (index 0 = not_vegetated). Only PFTs 1-16 have
#: multilayer canopy parameters; CLM's crop PFTs (17-78) are not supported.
PFT_NAMES: tuple[str, ...] = (
    "not_vegetated",
    "needleleaf_evergreen_temperate_tree",
    "needleleaf_evergreen_boreal_tree",
    "needleleaf_deciduous_boreal_tree",
    "broadleaf_evergreen_tropical_tree",
    "broadleaf_evergreen_temperate_tree",
    "broadleaf_deciduous_tropical_tree",
    "broadleaf_deciduous_temperate_tree",
    "broadleaf_deciduous_boreal_tree",
    "broadleaf_evergreen_shrub",
    "broadleaf_deciduous_temperate_shrub",
    "broadleaf_deciduous_boreal_shrub",
    "c3_arctic_grass",
    "c3_non-arctic_grass",
    "c4_grass",
    "c3_crop",
    "c3_irrigated",
)


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    name: str  # params.yaml / Params field name
    units: str
    description: str
    legacy_table: str  # "MLpftcon" (multilayer_canopy.MLpftconMod) or "pftcon" (clm_src_main)
    legacy_field: str
    spectral: bool = False  # [vis, nir] pair


PARAM_SPECS: tuple[ParamSpec, ...] = (
    # Photosynthesis and stomatal conductance
    ParamSpec("vcmax25", "umol m-2 s-1", "Maximum carboxylation rate at 25 C, canopy top",
              "MLpftcon", "vcmaxpft"),
    ParamSpec("iota", "umol CO2 mol-1 H2O", "Marginal water-use efficiency (WUE stomatal model)",
              "MLpftcon", "iota_SPA"),
    ParamSpec("gs_min", "mol H2O m-2 s-1", "Minimum stomatal conductance",
              "MLpftcon", "gsmin_SPA"),
    ParamSpec("g0_ball_berry", "mol H2O m-2 s-1", "Ball-Berry minimum leaf conductance",
              "MLpftcon", "g0_BB"),
    ParamSpec("g1_ball_berry", "-", "Ball-Berry slope", "MLpftcon", "g1_BB"),
    ParamSpec("g0_medlyn", "mol H2O m-2 s-1", "Medlyn minimum leaf conductance",
              "MLpftcon", "g0_MED"),
    ParamSpec("g1_medlyn", "kPa0.5", "Medlyn slope", "MLpftcon", "g1_MED"),
    ParamSpec("psi50_gs", "MPa", "Leaf water potential at 50% loss of stomatal conductance",
              "MLpftcon", "psi50_gs"),
    ParamSpec("shape_gs", "-", "Shape parameter for stomatal closure with leaf water potential",
              "MLpftcon", "shape_gs"),
    ParamSpec("c3_photosynthesis", "1", "1 for C3 photosynthesis, 0 for C4 (structural)",
              "pftcon", "c3psn"),
    # Plant hydraulics and roots
    ParamSpec("plant_conductance", "mmol H2O m-2 s-1 MPa-1",
              "Stem hydraulic conductance per unit leaf area", "MLpftcon", "gplant_SPA"),
    ParamSpec("plant_capacitance", "mmol H2O m-2 MPa-1", "Plant capacitance per unit leaf area",
              "MLpftcon", "capac_SPA"),
    ParamSpec("root_radius", "m", "Fine root radius", "MLpftcon", "root_radius_SPA"),
    ParamSpec("root_density", "g m-3", "Fine root density (biomass per root volume)",
              "MLpftcon", "root_density_SPA"),
    ParamSpec("root_resistivity", "MPa s g mmol-1 H2O", "Hydraulic resistivity of root tissue",
              "MLpftcon", "root_resist_SPA"),
    ParamSpec("root_profile_beta", "-", "Root profile shape parameter", "pftcon", "rootprof_beta"),
    # Leaf structure and radiation
    ParamSpec("leaf_dimension", "m", "Leaf dimension (for boundary-layer conductance)",
              "pftcon", "dleaf"),
    ParamSpec("leaf_angle_index", "-", "Leaf angle departure from spherical (chi_l)",
              "pftcon", "xl"),
    ParamSpec("sla_top", "m2 gC-1", "Specific leaf area at canopy top", "pftcon", "slatop"),
    ParamSpec("leaf_emissivity", "-", "Leaf emissivity", "MLpftcon", "emleaf"),
    ParamSpec("clumping_index", "-", "Foliage clumping index", "MLpftcon", "clump_fac"),
    ParamSpec("leaf_reflectance", "-", "Leaf reflectance [vis, nir]", "pftcon", "rhol", True),
    ParamSpec("leaf_transmittance", "-", "Leaf transmittance [vis, nir]", "pftcon", "taul", True),
    ParamSpec("stem_reflectance", "-", "Stem reflectance [vis, nir]", "pftcon", "rhos", True),
    ParamSpec("stem_transmittance", "-", "Stem transmittance [vis, nir]", "pftcon", "taus", True),
)  # fmt: skip

SPECS_BY_NAME: Mapping[str, ParamSpec] = {s.name: s for s in PARAM_SPECS}


@dataclasses.dataclass(frozen=True)
class Params:
    """Per-PFT parameter table.

    ``values[name]`` has shape ``(n_pft,)``, or ``(n_pft, 2)`` for spectral [vis, nir]
    parameters, with rows in :data:`PFT_NAMES` order. PFTs without values are NaN.
    """

    values: Mapping[str, np.ndarray]
    source: str = ""  # file the table was loaded from (for provenance)

    def __getattr__(self, name: str) -> np.ndarray:
        if name in SPECS_BY_NAME:
            return self.values[name]
        raise AttributeError(name)

    @classmethod
    def from_yaml(cls, path_or_name: Path | str) -> "Params":
        """Load a params file (a path, or a built-in name such as ``"chats"``), resolving
        ``extends:`` chains."""
        merged, source = _load_merged(path_or_name, seen=())
        return cls._from_merged(merged, source)

    @classmethod
    def _from_merged(cls, merged: dict[str, dict[str, object]], source: str) -> "Params":
        values = {
            spec.name: np.full((len(PFT_NAMES), 2) if spec.spectral else len(PFT_NAMES), np.nan)
            for spec in PARAM_SPECS
        }
        for pft, entries in merged.items():
            row = pft_index(pft, source)
            for name, value in entries.items():
                spec = SPECS_BY_NAME.get(name)
                if spec is None:
                    raise ConfigError(
                        f"{source}: pft.{pft}.{name}: unknown parameter; "
                        f"allowed: {sorted(SPECS_BY_NAME)}"
                    )
                values[name][row] = _check_value(spec, value, f"{source}: pft.{pft}.{name}")
        return cls(values=values, source=source)

    def get(self, pft: str, name: str) -> np.ndarray:
        return self.values[name][pft_index(pft, "get()")]

    def set(self, pft: str, **changes) -> "Params":
        """Return a copy with values changed for one PFT."""
        row = pft_index(pft, "set()")
        values = {k: v.copy() for k, v in self.values.items()}
        for name, value in changes.items():
            if name not in SPECS_BY_NAME:
                raise ConfigError(f"set(): unknown parameter {name!r}")
            values[name][row] = value
        return Params(values=values, source=self.source)

    def canopy_params(self, pft: str):
        """The parameters of one PFT as the physics takes them
        (``multilayer_canopy.MLCanopyParamsMod.CanopyParams``, JAX scalars). Spectral
        ``[vis, nir]`` pairs become ``[0, vis, nir]`` (the physics indexes wavebands 1, 2)."""
        import jax.numpy as jnp

        from multilayer_canopy.MLCanopyParamsMod import CanopyParams

        row = pft_index(pft, "canopy_params()")
        fields = {}
        for name in CanopyParams._fields:
            value = jnp.asarray(self.values[name][row], dtype=jnp.float64)
            if SPECS_BY_NAME[name].spectral:
                value = jnp.concatenate([jnp.zeros(1), value])
            fields[name] = value
        return CanopyParams(**fields)

    def missing_for(self, pft_rows: list[int]) -> list[str]:
        """Parameters with no value (NaN) for any of the given PFT rows."""
        return sorted(
            f"{PFT_NAMES[row]}.{name}"
            for name, table in self.values.items()
            for row in pft_rows
            if np.isnan(table[row]).any()
        )

    def describe(self, pft_rows: Optional[list[int]] = None) -> str:
        rows = pft_rows if pft_rows is not None else range(1, len(PFT_NAMES))
        lines = []
        for row in rows:
            lines.append(PFT_NAMES[row])
            for spec in PARAM_SPECS:
                value = self.values[spec.name][row]
                shown = f"[{value[0]:g}, {value[1]:g}]" if spec.spectral else f"{value:g}"
                lines.append(f"  {spec.name:22s} {shown:>16s}  {spec.units}")
        return "\n".join(lines)


def pft_index(name: str, source: str) -> int:
    try:
        return PFT_NAMES.index(name)
    except ValueError:
        raise ConfigError(
            f"{source}: unknown PFT {name!r}; allowed: {', '.join(PFT_NAMES[1:])}"
        ) from None


def builtin_params_path(name: str) -> Path:
    return Path(str(resources.files("clm_ml") / "data" / "params" / f"{name}.yaml"))


def _check_value(spec: ParamSpec, value, where: str):
    if spec.spectral:
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ConfigError(f"{where}: expected [vis, nir], got {value!r}")
        for v in value:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"{where}: expected numbers, got {value!r}")
        return [float(v) for v in value]
    if spec.name == "c3_photosynthesis":
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _resolve(path_or_name: Path | str, relative_to: Optional[Path]) -> Path:
    text = str(path_or_name)
    if not text.endswith((".yaml", ".yml")) and "/" not in text:
        return builtin_params_path(text)
    path = Path(text)
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path


def _load_merged(path_or_name, seen: tuple, relative_to: Optional[Path] = None):
    path = _resolve(path_or_name, relative_to)
    if path in seen:
        raise ConfigError(f"{path.name}: circular 'extends' chain")
    data = read_yaml(path)
    unknown = sorted(set(data) - {"extends", "pft", "units", "description"})
    if unknown:
        raise ConfigError(f"{path.name}: unknown key(s) {unknown}; allowed: extends, pft, units")
    merged: dict[str, dict[str, object]] = {}
    if data.get("extends"):
        merged, _ = _load_merged(data["extends"], seen + (path,), relative_to=path.parent)
        merged = {pft: dict(entries) for pft, entries in merged.items()}
    pfts = data.get("pft") or {}
    if not isinstance(pfts, dict):
        raise ConfigError(f"{path.name}: pft: expected a mapping of PFT names to parameters")
    for pft, entries in pfts.items():
        pft_index(pft, path.name)
        if not isinstance(entries, dict):
            raise ConfigError(f"{path.name}: pft.{pft}: expected a mapping of parameters")
        merged.setdefault(pft, {}).update(entries)
    return merged, path.name
