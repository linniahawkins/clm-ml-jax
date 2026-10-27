"""
The spatial domain: gridcells, the patches on them, and the soil columns under the patches.

Vocabulary (modernization plan §3.1):

- **gridcell**: a location with its own atmospheric forcing.
- **patch**: one PFT's canopy on a gridcell, with an area ``weight``. The unit of
  computation.
- **soil column**: the soil under a patch. With ``soil.columns: per_patch`` (the only
  option implemented) every patch has its own column.
- **tower**: a named flux-tower location in ``towers.yaml``; it defines one gridcell and
  its patches.

Patches are a flat list. ``Patches.gridcell_index`` says which gridcell each one is on.
"""

from __future__ import annotations

import dataclasses
from importlib import resources
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from clm_ml.config import ModelConfig
from clm_ml.io.yaml_schema import ConfigError, from_dict, read_yaml
from clm_ml.params import PFT_NAMES, Params, pft_index

# --------------------------------------------------------------------------------------
# towers.yaml schema
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SoilSpec:
    texture_class: Optional[str] = None  # used when sand/clay are null
    sand_percent: Optional[float] = None
    clay_percent: Optional[float] = None
    organic_matter: Optional[float] = None  # kg m-3
    color_class: int = 15  # CLM soil color class, 1 (light) to 20 (dark)
    bedrock_depth: Optional[float] = None  # m


@dataclasses.dataclass(frozen=True)
class PatchSpec:
    pft: str
    weight: float = 1.0  # fraction of the gridcell
    canopy_height: Optional[float] = None  # m (null: from CLM history file)
    root_biomass: Optional[float] = None  # g m-2 (null: PFT default)
    leaf_area_profile_beta: Optional[tuple[float, float]] = None  # null: PFT default profile
    stem_area_profile_beta: Optional[tuple[float, float]] = None


@dataclasses.dataclass(frozen=True)
class TowerSpec:
    latitude: float  # degrees north
    longitude: float  # degrees east
    patches: tuple[PatchSpec, ...]
    measurement_height: Optional[float] = None  # forcing height, m (null: 30 m)
    forcing_step_minutes: int = 30
    soil: SoilSpec = SoilSpec()


def builtin_towers_path() -> Path:
    return Path(str(resources.files("clm_ml") / "data" / "towers.yaml"))


def load_towers(path: Path) -> dict[str, TowerSpec]:
    data = read_yaml(path)
    towers = {}
    for name, entry in data.items():
        tower = from_dict(TowerSpec, entry, source=path.name, key=str(name))
        if not tower.patches:
            raise ConfigError(f"{path.name}: {name}.patches: at least one patch is required")
        total = sum(p.weight for p in tower.patches)
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"{path.name}: {name}.patches: weights sum to {total:g}, not 1")
        for i, patch in enumerate(tower.patches):
            pft_index(patch.pft, f"{path.name}: {name}.patches[{i}].pft")
        towers[str(name)] = tower
    return towers


# --------------------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Gridcells:
    names: tuple[str, ...]
    latitude: np.ndarray  # (ng,)
    longitude: np.ndarray  # (ng,)
    measurement_height: np.ndarray  # (ng,) m
    forcing_step_minutes: np.ndarray  # (ng,) int


@dataclasses.dataclass(frozen=True)
class Patches:
    gridcell_index: np.ndarray  # (np,) int
    column_index: np.ndarray  # (np,) int; per_patch soil columns: arange(np)
    pft_index: np.ndarray  # (np,) int, row into Params
    weight: np.ndarray  # (np,)
    canopy_height: np.ndarray  # (np,) m, NaN = from CLM history
    root_biomass: np.ndarray  # (np,) g m-2, NaN = PFT default
    leaf_area_profile_beta: np.ndarray  # (np, 2), NaN = PFT default
    stem_area_profile_beta: np.ndarray  # (np, 2)


@dataclasses.dataclass(frozen=True)
class SoilColumns:
    texture_class: tuple[Optional[str], ...]
    sand_percent: np.ndarray  # (nc,) NaN = from texture class
    clay_percent: np.ndarray
    organic_matter: np.ndarray  # kg m-3
    color_class: np.ndarray  # (nc,) int
    bedrock_depth: np.ndarray  # (nc,) m


@dataclasses.dataclass(frozen=True)
class Domain:
    gridcells: Gridcells
    patches: Patches
    columns: SoilColumns
    towers: dict  # name -> TowerSpec, as loaded (for provenance and the legacy adapter)

    @property
    def n_gridcells(self) -> int:
        return len(self.gridcells.names)

    @property
    def n_patches(self) -> int:
        return len(self.patches.pft_index)

    @classmethod
    def from_towers(
        cls,
        names: Sequence[str],
        towers_file: Optional[Path] = None,
        *,
        params: Params,
        config: ModelConfig,
        min_patch_weight: float = 0.0,
    ) -> "Domain":
        """Build a domain from built-in towers and/or a user ``towers.yaml``
        (user entries take precedence over built-in ones with the same name)."""
        available = load_towers(builtin_towers_path())
        if towers_file is not None:
            available.update(load_towers(Path(towers_file)))
        missing = [n for n in names if n not in available]
        if missing:
            raise ConfigError(
                f"run.yaml: domain.towers: unknown tower(s) {missing}; "
                f"available: {sorted(available)}"
            )
        if not names:
            raise ConfigError("run.yaml: domain.towers: list at least one tower")

        g_rows, p_rows, c_rows = [], [], []
        for g, name in enumerate(names):
            tower = available[name]
            g_rows.append(tower)
            kept = [p for p in tower.patches if p.weight >= min_patch_weight]
            if not kept:
                raise ConfigError(
                    f"{name}: every patch is below domain.min_patch_weight={min_patch_weight}"
                )
            total = sum(p.weight for p in kept)
            for patch in kept:
                p_rows.append((g, patch, patch.weight / total))
                c_rows.append(tower.soil)

        def nan(v):
            return np.nan if v is None else v

        def pair(v):
            return (np.nan, np.nan) if v is None else v

        gridcells = Gridcells(
            names=tuple(names),
            latitude=np.array([t.latitude for t in g_rows]),
            longitude=np.array([t.longitude for t in g_rows]),
            measurement_height=np.array([nan(t.measurement_height) for t in g_rows]),
            forcing_step_minutes=np.array([t.forcing_step_minutes for t in g_rows]),
        )
        patches = Patches(
            gridcell_index=np.array([g for g, _, _ in p_rows]),
            column_index=np.arange(len(p_rows)),
            pft_index=np.array([PFT_NAMES.index(p.pft) for _, p, _ in p_rows]),
            weight=np.array([w for _, _, w in p_rows]),
            canopy_height=np.array([nan(p.canopy_height) for _, p, _ in p_rows]),
            root_biomass=np.array([nan(p.root_biomass) for _, p, _ in p_rows]),
            leaf_area_profile_beta=np.array([pair(p.leaf_area_profile_beta) for _, p, _ in p_rows]),
            stem_area_profile_beta=np.array([pair(p.stem_area_profile_beta) for _, p, _ in p_rows]),
        )
        columns = SoilColumns(
            texture_class=tuple(s.texture_class for s in c_rows),
            sand_percent=np.array([nan(s.sand_percent) for s in c_rows]),
            clay_percent=np.array([nan(s.clay_percent) for s in c_rows]),
            organic_matter=np.array([nan(s.organic_matter) for s in c_rows]),
            color_class=np.array([s.color_class for s in c_rows]),
            bedrock_depth=np.array([nan(s.bedrock_depth) for s in c_rows]),
        )
        domain = cls(gridcells, patches, columns, {n: available[n] for n in names})

        missing_params = params.missing_for(sorted(set(patches.pft_index.tolist())))
        if missing_params:
            raise ConfigError(f"params: no value for {missing_params[:5]} (used by the domain)")
        for i, n in enumerate(domain.layer_counts(config)):
            if n > config.canopy_layers.max_layers:
                raise ConfigError(
                    f"{names[patches.gridcell_index[i]]} patch {i} "
                    f"({PFT_NAMES[patches.pft_index[i]]}) needs {n} canopy layers but "
                    f"model.yaml canopy_layers.max_layers is {config.canopy_layers.max_layers}"
                )
        return domain

    def layer_counts(self, config: ModelConfig) -> list[int]:
        """Number of canopy-air layers per patch (ground to measurement height).

        Same rule as the legacy ``initVerticalStructure``. Patches whose canopy height comes
        from the CLM history file (NaN here) are counted when the run starts instead, and
        reported as 0.

        Fortran reference: MLinitVerticalMod.F90, subroutine initVerticalStructure.
        """
        c = config.canopy_layers
        counts = []
        for i, htop in enumerate(self.patches.canopy_height):
            zref = self.gridcells.measurement_height[self.patches.gridcell_index[i]]
            zref = 30.0 if np.isnan(zref) else zref  # legacy default for tower_ht = -999
            if np.isnan(htop):
                counts.append(0)
            elif c.n_layers_within is not None:
                counts.append(c.n_layers_within + c.n_layers_above)
            else:
                dz = c.dz_tall if htop > c.tall_canopy_threshold else c.dz_short
                ntop = round(htop / dz)
                nabove = round((zref - htop) / (htop / ntop))
                counts.append(ntop + nabove)
        return counts
