"""
The spatial domain: gridcells, the patches on them, and the soil columns under the patches.

Vocabulary:

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
from typing import Literal, Optional, Union

import numpy as np

from clm_ml.config import LayerCounts, ModelConfig
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


TextureClass = Literal[
    "sand",
    "loamy sand",
    "sandy loam",
    "silty loam",
    "loam",
    "sandy clay loam",
    "silty clay loam",
    "clay loam",
    "sandy clay",
    "silty clay",
    "clay",
]  # fmt: skip  (clm_ml.setup.tables.SOIL_TEXTURES)


@dataclasses.dataclass(frozen=True)
class TextureSpec:
    """The soil texture: sand and clay percentages (hydraulic properties from CLM5's
    formulas), a USDA texture class (Clapp and Hornberger 1978 values), or the sand and clay
    of each layer of a CLM surface dataset (a texture per soil level, as CTSM maps them)."""

    given_as: Literal["percentages", "class", "layers"] = option(
        "percentages", "How the texture is given: percentages, class or layers."
    )
    sand_percent: Optional[float] = option(None, "Sand content (%).", required=True,
                                           when={"given_as": "percentages"})  # fmt: skip
    clay_percent: Optional[float] = option(None, "Clay content (%).", required=True,
                                           when={"given_as": "percentages"})  # fmt: skip
    texture_class: Optional[TextureClass] = option(
        None, "USDA texture class.", required=True, when={"given_as": "class"}
    )
    sand_percent_layers: Optional[tuple[float, ...]] = option(
        None, "Sand content (%) of each layer of a CLM surface dataset (PCT_SAND), top to "
        "bottom.", required=True, when={"given_as": "layers"},
    )  # fmt: skip
    clay_percent_layers: Optional[tuple[float, ...]] = option(
        None, "Clay content (%) of each layer of a CLM surface dataset (PCT_CLAY), top to "
        "bottom.", required=True, when={"given_as": "layers"},
    )  # fmt: skip


@dataclasses.dataclass(frozen=True)
class SoilSpec:
    """The soil of the tower's column(s)."""

    texture: Optional[TextureSpec] = option(None, "The soil texture.", required=True)
    organic_matter: Union[float, tuple[float, ...], None] = option(
        None,
        "Soil organic matter density (kg m-3): one value, in the top 0.5 m; or, with "
        "texture given_as layers, one value per layer (ORGANIC), at every level.",
        required=True,
    )
    color_class: int = option(15, "CLM soil colour class, 1 (light) to 20 (dark): soil albedo.")
    bedrock_depth: Optional[float] = option(None, "Depth to bedrock (m); null: no bedrock.")
    max_saturated_fraction: Optional[float] = option(
        None,
        "Largest fraction of the surface that can be saturated, 0 to 1 (FMAX of a CLM "
        "surface dataset at the tower's gridcell): surface runoff. Only used, and then "
        "required, with predicted soil water (model.yaml soil.water).",
    )
    slope: Optional[float] = option(
        None,
        "Topographic slope in degrees (SLOPE of a CLM surface dataset at the tower's "
        "gridcell; at least 0.2 is used, as in CLM): drainage and ponding. Only used, and "
        "then required, with predicted soil water.",
    )


@dataclasses.dataclass(frozen=True)
class AreaProfiles:
    """Beta distributions (p, q) of leaf and stem area with height in the canopy."""

    leaf_beta: tuple[float, float] = option(doc="Leaf area (p, q).")
    stem_beta: tuple[float, float] = option(doc="Stem area (p, q).")


@dataclasses.dataclass(frozen=True)
class PatchSpec:
    """One patch: one PFT's canopy on the tower's gridcell."""

    pft: str = option(doc="Plant functional type (a name from params.yaml, e.g. "
                      "broadleaf_deciduous_temperate_tree).")  # fmt: skip
    weight: float = option(1.0, "Fraction of the gridcell; the weights of a tower sum to 1.")
    canopy_height: Union[Literal["pft_default"], float] = option(
        "pft_default",
        "Canopy height (m), or pft_default (CLM's height of the PFT: 17 m needleleaf, 35/18/20 m "
        "broadleaf trees, 0.5 m shrubs and grasses).",
        "tower_canht; htop_pft",
    )
    root_biomass: Optional[float] = option(None, "Fine root biomass (g m-2).", "tower_root",
                                           required=True)  # fmt: skip
    area_profiles: Union[Literal["pft_default"], AreaProfiles] = option(
        "pft_default",
        "Leaf and stem area profiles: pft_default, or {leaf_beta: [p, q], stem_beta: [p, q]} "
        "(both, as Fortran).",
        "tower_pbeta_lai, tower_pbeta_sai",
    )
    canopy_layers: Union[Literal["from_model_yaml"], LayerCounts] = option(
        "from_model_yaml",
        "This patch's canopy-air layer counts: from_model_yaml (model.yaml canopy_layers), or "
        "{within: N, above: M}, which take precedence over model.yaml (e.g. a short canopy "
        "under a tall tower).",
    )


@dataclasses.dataclass(frozen=True)
class TowerSpec:
    """One tower: a gridcell with its own forcing, its soil and its patches."""

    latitude: float = option(doc="Latitude, degrees north.")
    longitude: float = option(doc="Longitude, degrees east.")
    measurement_height: float = option(30.0, "Forcing (reference) height (m).", "tower_ht")
    soil: Optional[SoilSpec] = option(None, "The soil.", required=True)
    patches: Optional[tuple[PatchSpec, ...]] = option(
        None, "The patches on the tower's gridcell.", required=True
    )


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
            counts = patch.canopy_layers
            if isinstance(counts, LayerCounts) and (counts.within < 1 or counts.above < 1):
                raise ConfigError(f"{where}.canopy_layers: within and above must be >= 1")
            if patch.root_biomass <= 0:
                raise ConfigError(f"{where}.root_biomass: must be positive")
        soil, where = tower.soil, f"{source}: towers.{name}.soil"
        fmax, slope = soil.max_saturated_fraction, soil.slope
        if fmax is not None and not 0.0 <= fmax <= 1.0:
            raise ConfigError(f"{where}.max_saturated_fraction: {fmax:g} is not within 0 to 1")
        if slope is not None and not 0.0 <= slope < 90.0:
            raise ConfigError(f"{where}.slope: {slope:g} degrees is not within 0 to 90")
        layered_organic = isinstance(soil.organic_matter, tuple)
        if soil.texture.given_as == "layers":
            n = len(soil.texture.sand_percent_layers)
            lengths = (len(soil.texture.clay_percent_layers),
                       len(soil.organic_matter) if layered_organic else -1)  # fmt: skip
            if n < 1 or any(m != n for m in lengths):
                raise ConfigError(
                    f"{where}: texture given_as layers needs sand_percent_layers, "
                    "clay_percent_layers and organic_matter with one value per layer each"
                )
        elif layered_organic:
            raise ConfigError(
                f"{where}.organic_matter: one value per layer only with texture given_as layers"
            )


def load_towers(path: Path) -> dict[str, TowerSpec]:
    """A towers file (``{name: tower}``; the compsets in ``clm_ml/compsets/towers``)."""
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
class LatLonMap:
    """Where a grid run's gridcells go in its latitude-longitude output: the
    axes of the map (the whole surface-data grid for a global run, its rows and columns
    inside the box of a region) and each gridcell's (row, column) on them."""

    lat: np.ndarray  # (nlat,) degrees north
    lon: np.ndarray  # (nlon,) degrees east, increasing
    index: np.ndarray  # (ng, 2) int


@dataclasses.dataclass(frozen=True)
class Gridcells:
    names: tuple[str, ...]
    latitude: np.ndarray  # (ng,)
    longitude: np.ndarray  # (ng,)
    measurement_height: np.ndarray  # (ng,) m
    # Grid domains: each gridcell's (row, column) on the surface-data grid
    grid_index: Optional[np.ndarray] = None  # (ng, 2) int
    # Grid domains on a regular latitude-longitude grid: the gridcell output's map
    latlon: Optional[LatLonMap] = None
    # Grid domains: written with the gridcell output, for totals. Area (km2)
    # and land fraction as CLM's history (land_mask file); the fraction of the land covered
    # by PFTs 1-16 in the surface data, which the patches represent
    area: Optional[np.ndarray] = None  # (ng,) km2
    land_fraction: Optional[np.ndarray] = None  # (ng,)
    vegetated_fraction: Optional[np.ndarray] = None  # (ng,)


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
    # Grid domains: canopy bottom (m, from the surface data; not used by the
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
    # the texture of each layer of a CLM surface dataset ((nc, n_layers), percent and
    # kg m-3), mapped to the soil levels at setup; None: one texture per column (above)
    sand_percent_layers: Optional[np.ndarray] = None
    clay_percent_layers: Optional[np.ndarray] = None
    organic_matter_layers: Optional[np.ndarray] = None
    # for predicted soil water ((nc,); NaN: not given): the largest saturated fraction
    # (CLM FMAX) and the topographic slope in degrees (CLM SLOPE)
    max_saturated_fraction: Optional[np.ndarray] = None
    slope: Optional[np.ndarray] = None


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
            return np.nan if v is None or isinstance(v, str) else v

        def profile(p, which):
            a = p.area_profiles
            return (np.nan, np.nan) if isinstance(a, str) else getattr(a, which)

        def counts(p, which):
            c = p.canopy_layers
            return 0 if isinstance(c, str) else getattr(c, which)

        def texture(s, which):
            t = s.texture
            if which == "texture_class":
                return t.texture_class if t.given_as == "class" else None
            return getattr(t, which) if t.given_as == "percentages" else np.nan

        layered = [s.texture.given_as == "layers" for s in c_rows]
        if any(layered) and not all(layered):
            raise ConfigError(
                "domain.yaml: soil texture given_as layers for some towers only; give every "
                "tower's texture by layers, or none"
            )
        if all(layered) and len({len(s.texture.sand_percent_layers) for s in c_rows}) > 1:
            raise ConfigError("domain.yaml: towers' texture layers differ in number")

        def organic(s):  # one value per column; NaN: given by layers
            return np.nan if isinstance(s.organic_matter, tuple) else float(s.organic_matter)

        def layers(which):
            if not all(layered):
                return None
            if which == "organic_matter":
                return np.array([s.organic_matter for s in c_rows], dtype=float)
            return np.array([getattr(s.texture, which + "_layers") for s in c_rows], dtype=float)

        gridcells = Gridcells(
            names=tuple(names),
            latitude=np.array([t.latitude for t in g_rows]),
            longitude=np.array([t.longitude for t in g_rows]),
            measurement_height=np.array([float(t.measurement_height) for t in g_rows]),
        )
        patches = Patches(
            gridcell_index=np.array([g for g, _, _ in p_rows]),
            column_index=np.arange(len(p_rows)),
            pft_index=np.array([PFT_NAMES.index(p.pft) for _, p, _ in p_rows]),
            weight=np.array([w for _, _, w in p_rows]),
            canopy_height=np.array([nan(p.canopy_height) for _, p, _ in p_rows]),
            root_biomass=np.array([float(p.root_biomass) for _, p, _ in p_rows]),
            leaf_area_profile_beta=np.array([profile(p, "leaf_beta") for _, p, _ in p_rows]),
            stem_area_profile_beta=np.array([profile(p, "stem_beta") for _, p, _ in p_rows]),
            n_layers_within=np.array([counts(p, "within") for _, p, _ in p_rows]),
            n_layers_above=np.array([counts(p, "above") for _, p, _ in p_rows]),
        )
        columns = SoilColumns(
            texture_class=tuple(texture(s, "texture_class") for s in c_rows),
            sand_percent=np.array([texture(s, "sand_percent") for s in c_rows], dtype=float),
            clay_percent=np.array([texture(s, "clay_percent") for s in c_rows], dtype=float),
            organic_matter=np.array([organic(s) for s in c_rows]),
            color_class=np.array([s.color_class for s in c_rows]),
            bedrock_depth=np.array([nan(s.bedrock_depth) for s in c_rows]),
            sand_percent_layers=layers("sand_percent"),
            clay_percent_layers=layers("clay_percent"),
            organic_matter_layers=layers("organic_matter"),
            max_saturated_fraction=np.array([nan(s.max_saturated_fraction) for s in c_rows]),
            slope=np.array([nan(s.slope) for s in c_rows]),
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

        Same rule as Fortran ``initVerticalStructure``: the patch's own counts
        (``domain.yaml``) when given, else model.yaml's explicit counts, else from the spacing;
        a patch without its own canopy height takes its PFT's default height.

        Fortran reference: MLinitVerticalMod.F90, subroutine initVerticalStructure.
        """
        return [n for n, _ in self.layer_counts_with_source(config)]

    def layer_counts_with_source(self, config: ModelConfig) -> list[tuple[int, str]]:
        """``(layers, where the count comes from)`` per patch (``clm-ml check``)."""
        from clm_ml.setup.tables import default_canopy_height

        c = config.canopy_layers
        out = []
        own = self.patches.measurement_height
        for i, htop in enumerate(self.patches.canopy_height):
            zref = self.gridcells.measurement_height[self.patches.gridcell_index[i]]
            zref = zref if own is None else own[i]
            zref = 30.0 if np.isnan(zref) else zref  # default height for tower_ht = -999
            if np.isnan(htop) or htop <= 0.0:
                htop = default_canopy_height(int(self.patches.pft_index[i]))
            if self.patches.n_layers_within[i] > 0:
                n = int(self.patches.n_layers_within[i] + self.patches.n_layers_above[i])
                out.append((n, "the patch's own canopy_layers"))
            elif isinstance(c.layer_counts, LayerCounts):
                out.append((c.layer_counts.within + c.layer_counts.above,
                            "model.yaml canopy_layers.layer_counts"))  # fmt: skip
            else:
                dz = c.dz_tall if htop > c.tall_canopy_threshold else c.dz_short
                ntop = round(htop / dz)
                nabove = round((zref - htop) / (htop / ntop))
                out.append((ntop + nabove, f"spacing {dz:g} m, canopy {htop:g} m, forcing "
                                           f"height {zref:g} m"))  # fmt: skip
        return out
