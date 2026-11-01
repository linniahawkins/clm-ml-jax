"""
The spatial domain: gridcells, the patches on them, and the soil columns under the patches.

Vocabulary (modernization plan §3.1):

- **gridcell**: a location with its own atmospheric forcing.
- **patch**: one PFT's canopy on a gridcell, with an area ``weight``. The unit of
  computation.
- **soil column**: the soil under a patch. With ``soil.columns: per_patch`` (the only
  option implemented) every patch has its own column.
- **tower**: a named flux-tower location in ``domain.yaml``; it defines one gridcell and
  its patches.

Patches are a flat list. ``Patches.gridcell_index`` says which gridcell each one is on.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from clm_ml.config import ModelConfig
from clm_ml.grid import GridSection
from clm_ml.io.yaml_schema import ConfigError, from_dict, option, read_yaml
from clm_ml.params import PFT_NAMES, Params, pft_index

# --------------------------------------------------------------------------------------
# domain.yaml schema
# --------------------------------------------------------------------------------------

HEADER = """\
domain.yaml: where the experiment runs: towers (location, forcing height, soil, patches) or a
grid (a CLM surface dataset). Part of an experiment folder (run.yaml, domain.yaml, model.yaml,
params.yaml). Every setting is listed; REQUIRED keys have no default. {generated}"""


@dataclasses.dataclass(frozen=True)
class SoilSpec:
    """The soil of the tower's column(s)."""

    texture_class: Optional[str] = option(
        None, "USDA texture class (e.g. silty clay loam), used when sand and clay are null."
    )
    sand_percent: Optional[float] = option(None, "Sand content (%).")
    clay_percent: Optional[float] = option(None, "Clay content (%).")
    organic_matter: Optional[float] = option(None, "Soil organic matter density (kg m-3).")
    color_class: int = option(15, "CLM soil colour class, 1 (light) to 20 (dark): soil albedo.")
    bedrock_depth: Optional[float] = option(None, "Depth to bedrock (m); null: none.")


@dataclasses.dataclass(frozen=True)
class PatchSpec:
    """One patch: one PFT's canopy on the tower's gridcell."""

    pft: str = option(doc="Plant functional type (a name from params.yaml, e.g. "
                      "broadleaf_deciduous_temperate_tree).")  # fmt: skip
    weight: float = option(1.0, "Fraction of the gridcell; the weights of a tower sum to 1.")
    canopy_height: Optional[float] = option(None, "Canopy height (m); null: from the CLM "
                                            "history file.")  # fmt: skip
    root_biomass: Optional[float] = option(None, "Root biomass (g m-2); null: the PFT default.")
    leaf_area_profile_beta: Optional[tuple[float, float]] = option(
        None, "Beta distribution (p, q) of leaf area with height; null: the PFT default."
    )
    stem_area_profile_beta: Optional[tuple[float, float]] = option(
        None, "Beta distribution (p, q) of stem area with height; null: the PFT default."
    )
    n_layers_within: Optional[int] = option(
        None,
        "Canopy-air layers within the canopy for this patch; null: model.yaml "
        "canopy_layers. Set with n_layers_above (e.g. a short canopy under a tall tower; "
        "5-10 layers within the canopy suffice for fluxes, Bonan et al. 2021).",
    )
    n_layers_above: Optional[int] = option(
        None, "Canopy-air layers above the canopy for this patch; null: model.yaml."
    )


@dataclasses.dataclass(frozen=True)
class TowerSpec:
    """One tower: a gridcell with its own forcing, its soil and its patches."""

    latitude: float = option(doc="Latitude, degrees north.")
    longitude: float = option(doc="Longitude, degrees east.")
    patches: tuple[PatchSpec, ...] = option(doc="The patches on the tower's gridcell.")
    measurement_height: Optional[float] = option(
        None, "Forcing (reference) height (m); null: 30 m.", fortran="tower_ht"
    )
    soil: SoilSpec = option(SoilSpec(), "The soil.")


@dataclasses.dataclass(frozen=True)
class DomainFile:
    """The contents of ``domain.yaml``."""

    type: Literal["towers", "grid"] = option(
        doc="towers: the towers listed below; grid: the gridcells of a CLM surface dataset."
    )
    towers: dict[str, TowerSpec] = option(
        doc="The towers, by name (each one gridcell with its own forcing).", applies="towers"
    )
    grid: GridSection = option(doc="The grid.", applies="grid")
    min_patch_weight: float = option(
        0.0,
        "Patches covering less than this fraction of their gridcell are left out (the others' "
        "weights renormalized to sum to 1).",
    )


def check_towers(towers: dict[str, TowerSpec], source: str) -> None:
    """Each tower has patches whose weights sum to 1, known PFTs and paired layer counts."""
    if not towers:
        raise ConfigError(f"{source}: towers: list at least one tower")
    for name, tower in towers.items():
        if not tower.patches:
            raise ConfigError(f"{source}: towers.{name}.patches: at least one patch is required")
        total = sum(p.weight for p in tower.patches)
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"{source}: towers.{name}.patches: weights sum to {total:g}, not 1")
        for i, patch in enumerate(tower.patches):
            where = f"{source}: towers.{name}.patches[{i}]"
            pft_index(patch.pft, f"{where}.pft")
            if (patch.n_layers_within is None) != (patch.n_layers_above is None):
                raise ConfigError(
                    f"{where}: n_layers_within and n_layers_above must be set together"
                )
            if patch.n_layers_within is not None and (
                patch.n_layers_within < 1 or patch.n_layers_above < 1
            ):
                raise ConfigError(f"{where}: n_layers_within and n_layers_above must be >= 1")


def load_towers(path: Path) -> dict[str, TowerSpec]:
    """A towers file (``{name: tower}``; the presets in ``clm_ml/presets/towers``)."""
    data = read_yaml(path)
    towers = {
        str(name): from_dict(TowerSpec, entry, source=path.name, key=str(name))
        for name, entry in data.items()
    }
    check_towers(towers, path.name)
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
    # Grid domains (Phase 6b): each gridcell's (row, column) on the surface-data grid
    grid_index: Optional[np.ndarray] = None  # (ng, 2) int


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
    n_layers_within: np.ndarray  # (np,) int, 0 = model.yaml canopy_layers
    n_layers_above: np.ndarray  # (np,) int, 0 = model.yaml canopy_layers
    # Grid domains (Phase 6b): canopy bottom (m, from the surface data; not used by the
    # leaf-area profile, see clm_ml.grid) and the forcing height of each patch (m, CLM's
    # 30 m + z0m + displacement height); None for towers (the gridcell's height)
    canopy_bottom: Optional[np.ndarray] = None  # (np,)
    measurement_height: Optional[np.ndarray] = None  # (np,)


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
    towers: dict  # name -> TowerSpec, as loaded (for provenance)

    @property
    def n_gridcells(self) -> int:
        return len(self.gridcells.names)

    @property
    def n_patches(self) -> int:
        return len(self.patches.pft_index)

    @classmethod
    def from_towers(
        cls,
        available: dict[str, TowerSpec],
        *,
        params: Params,
        config: ModelConfig,
        min_patch_weight: float = 0.0,
    ) -> "Domain":
        """Build a domain from towers (``domain.yaml`` ``towers``, in their order)."""
        check_towers(available, "domain.yaml")
        names = list(available)

        g_rows, p_rows, c_rows = [], [], []
        for g, name in enumerate(names):
            tower = available[name]
            g_rows.append(tower)
            kept = [p for p in tower.patches if p.weight >= min_patch_weight]
            if not kept:
                raise ConfigError(
                    f"domain.yaml: towers.{name}: every patch is below min_patch_weight="
                    f"{min_patch_weight}"
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
            n_layers_within=np.array([p.n_layers_within or 0 for _, p, _ in p_rows]),
            n_layers_above=np.array([p.n_layers_above or 0 for _, p, _ in p_rows]),
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

        Same rule as the legacy ``initVerticalStructure``, with the patch's explicit counts
        (``domain.yaml``) in place of model.yaml's when set. Patches whose canopy height comes
        from the CLM history file (NaN here) are counted when the run starts instead, and
        reported as 0.

        Fortran reference: MLinitVerticalMod.F90, subroutine initVerticalStructure.
        """
        c = config.canopy_layers
        counts = []
        own = self.patches.measurement_height
        for i, htop in enumerate(self.patches.canopy_height):
            zref = self.gridcells.measurement_height[self.patches.gridcell_index[i]]
            zref = zref if own is None else own[i]
            zref = 30.0 if np.isnan(zref) else zref  # legacy default for tower_ht = -999
            if np.isnan(htop):
                counts.append(0)
            elif self.patches.n_layers_within[i] > 0:
                counts.append(int(self.patches.n_layers_within[i] + self.patches.n_layers_above[i]))
            elif c.n_layers_within is not None:
                counts.append(c.n_layers_within + c.n_layers_above)
            else:
                dz = c.dz_tall if htop > c.tall_canopy_threshold else c.dz_short
                ntop = round(htop / dz)
                nabove = round((zref - htop) / (htop / ntop))
                counts.append(ntop + nabove)
        return counts
