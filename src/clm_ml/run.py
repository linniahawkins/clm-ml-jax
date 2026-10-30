"""
A run: the contents of ``run.yaml`` and everything it references, loaded and validated.

``run.yaml`` replaces the Fortran namelist. It names the model and params files, the
towers, the dates, the forcing files, the initial conditions and the output.

``Run.execute()`` runs the legacy (Fortran-translated) physics in a separate process
(:mod:`clm_ml.legacy.worker`). A run holds any number of towers (gridcells) and patches
(Phase 6): every patch is set up as a one-patch legacy tower, and the time steps run batched
over the patches in one compiled program, with one clock for all gridcells (decision 24).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from clm_ml.config import ModelConfig
from clm_ml.domain import Domain
from clm_ml.grid import GridSection
from clm_ml.grid_forcing import GriddedForcingSection
from clm_ml.io.yaml_schema import ConfigError, apply_overrides, from_dict, read_yaml
from clm_ml.params import PFT_NAMES, Params

# --------------------------------------------------------------------------------------
# run.yaml schema
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TimeSection:
    start: str  # ISO date-time, e.g. 2007-05-01T00:00
    end: Optional[str] = None  # give exactly one of end, days, steps
    days: Optional[float] = None
    steps: Optional[int] = None
    step_seconds: int = 1800  # forcing data interval (the model step: model.yaml numerics)


@dataclasses.dataclass(frozen=True)
class DomainSection:
    type: Literal["towers", "grid"] = "towers"
    towers: tuple[str, ...] = ()
    towers_file: Optional[str] = None
    min_patch_weight: float = 0.0
    grid: Optional[GridSection] = None  # type grid (Phase 6b): clm_ml.grid


@dataclasses.dataclass(frozen=True)
class ClmHistorySection:
    file: str
    start: str  # first record of the history file (was clm_start_ymd / clm_start_tod)


#: A ``files`` entry: one path or a list; ``{tower}``, ``{year}`` and ``{month}`` (e.g.
#: ``{year}-{month:02d}.nc``) are expanded over the run's months (see :meth:`Run.files`).
Files = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class MeteorologySection:
    files: Files  # tower forcing netCDF, may be split in time
    # Within a model step (CLM-ml v2 sub-steps): three_point = met_type 3, none = met_type 0
    interpolation: Literal["none", "three_point"] = "three_point"
    # From a coarser data interval to the model step (decision 13; other variables linear)
    shortwave: Literal["coszen", "linear", "constant"] = "coszen"
    precipitation: Literal["constant", "linear"] = "constant"


@dataclasses.dataclass(frozen=True)
class SoilMoistureAdjustmentSection:
    file: str  # factors per CLM history time slice (CHATS)
    levels: int = 3  # was nlev_soil_adjust


@dataclasses.dataclass(frozen=True)
class LandDataSection:
    """Prescribed land data (decision 14): from the CLM history file, or from user files
    with daily or monthly records interpolated linearly between interval midpoints."""

    source: Literal["clm_history", "file"] = "clm_history"
    files: Files = ()  # netCDF: time ("days since ..."); LAI, SAI or H2OSOI
    interpolation: Literal["linear"] = "linear"


@dataclasses.dataclass(frozen=True)
class SoilWaterSection(LandDataSection):
    adjustment: Optional[SoilMoistureAdjustmentSection] = None


@dataclasses.dataclass(frozen=True)
class ForcingSection:
    # Towers: tower meteorology and the CLM history file (both required)
    meteorology: Optional[MeteorologySection] = None
    # The CLM history file: initial conditions, and LAI/SAI/soil water with source clm_history
    clm_history: Optional[ClmHistorySection] = None
    vegetation: LandDataSection = LandDataSection()  # LAI, SAI
    soil_water: SoilWaterSection = SoilWaterSection()  # volumetric soil water
    # Grid runs (Phase 6b): CRUJRA datm streams, CO2, monthly CLM history soil data
    # (clm_ml.grid_forcing); vegetation from the surface data
    gridded: Optional[GriddedForcingSection] = None


#: Earlier forcing keys and what replaced them (plan §4.1)
_OLD_FORCING_KEYS = {
    "interpolation": "forcing.meteorology.interpolation",
    "soil_moisture_adjustment": "forcing.soil_water.adjustment",
}


@dataclasses.dataclass(frozen=True)
class ExecutionSection:
    """How the run is divided into compiled chunks and how often it saves restarts."""

    # Compiled chunk length (decision 33): "N days" (fixed length, one compiled program; a
    # short last chunk is padded) or "1 month" (calendar months, each length compiles once)
    chunk: str = "10 days"
    restart_every: Optional[str] = None  # "N month(s)" / "N year(s)"; null: end of run only
    # Pad the domain to this many patches / gridcells (weight-0 copies, not in the output), so
    # domains of different sizes share one compiled program; null: the domain's own counts
    patch_count: Optional[int] = None
    gridcell_count: Optional[int] = None
    # Devices to spread the patches over (step 5: jax.shard_map; e.g. 4 on a Derecho GPU node)
    devices: int = 1


def parse_chunk(text: str, key: str) -> Optional[int]:
    """``"10 days"`` -> 10; ``"1 month"`` -> None (calendar months), decision 33."""
    parts = str(text).split()
    if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) >= 1:
        if parts[1] in ("day", "days"):
            return int(parts[0])
        if parts[1] in ("month", "months") and int(parts[0]) == 1:
            return None
    raise ConfigError(f"{key}: {text!r}: expected e.g. '10 days' or '1 month'")


def parse_every(text: Optional[str], key: str):
    """``"3 months"`` -> ``(3, "month")``; ``None`` -> ``None``."""
    if text is None:
        return None
    parts = str(text).split()
    units = {"month": "month", "months": "month", "year": "year", "years": "year"}
    if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) < 1 or parts[1] not in units:
        raise ConfigError(f"{key}: {text!r}: expected e.g. '1 month', '6 months' or '10 years'")
    return int(parts[0]), units[parts[1]]


@dataclasses.dataclass(frozen=True)
class InitialConditionsSection:
    source: Literal["clm_history", "restart"] = "clm_history"
    restart_file: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class OutputStreamSection:
    """One output stream (plan §4.1, decision 16): accumulated during the run."""

    name: str
    frequency: Literal["step", "hourly", "daily", "monthly", "yearly"] = "step"
    statistic: Literal["mean", "instantaneous", "min", "max"] = "mean"
    variables: tuple[str, ...] = ("all",)
    level: tuple[Literal["patch", "gridcell"], ...] = ("patch", "gridcell")
    file_per: Literal["run", "year", "month"] = "run"


@dataclasses.dataclass(frozen=True)
class OutputSection:
    directory: str = "output"
    streams: tuple[OutputStreamSection, ...] = (OutputStreamSection(name="step"),)
    # The six Fortran-format .out files (validation against Fortran only; runs the model a
    # second time, step-by-step output).
    legacy_ascii: bool = False


#: Phase 1 output keys and what replaced them (decision 16).
_OLD_OUTPUT_KEYS = {
    "format": "output.legacy_ascii: true (netCDF is always written)",
    "variables": "output.streams[].variables",
    "frequency": "output.streams[].frequency",
    "level": "output.streams[].level",
}


@dataclasses.dataclass(frozen=True)
class RunFile:
    """The ``run.yaml`` file as written (paths not yet resolved)."""

    time: TimeSection
    domain: DomainSection
    forcing: ForcingSection
    model: Optional[str] = None  # model.yaml path; null = all defaults
    params: str = "clm5_defaults"  # params file path or built-in name (clm5_defaults, chats)
    execution: ExecutionSection = ExecutionSection()
    initial_conditions: InitialConditionsSection = InitialConditionsSection()
    output: OutputSection = OutputSection()


# --------------------------------------------------------------------------------------
# Loaded run
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Run:
    """Everything needed for one experiment, loaded and validated."""

    run_file: RunFile
    base_dir: Path  # directory of run.yaml; relative paths are resolved against it
    config: ModelConfig
    params: Params
    domain: Domain
    start: dt.datetime
    n_steps: int
    source: str = "run.yaml"
    path_: Optional[Path] = None  # run.yaml itself (grid runs: the worker reloads it)
    overrides: tuple = ()

    @property
    def is_grid(self) -> bool:
        return self.run_file.domain.type == "grid"

    # -- paths -------------------------------------------------------------------------

    def path(self, text: str, tower: Optional[str] = None) -> Path:
        if tower is not None:
            text = text.replace("{tower}", tower)
        path = Path(text).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def files(self, entry, tower: str, key: str = "files") -> list[Path]:
        """The files of a ``files`` entry, in order. ``{year}`` and ``{month}`` are
        expanded for every month from one month before the run to one month after it (the
        neighbouring records interpolation needs); files for months inside the run must
        exist, the others are used when they exist."""
        entries = [entry] if isinstance(entry, str) else list(entry)
        end = self.start + dt.timedelta(seconds=(self.n_steps - 1) * self.step_seconds)  # last step
        months, (y, m) = [], (self.start.year, self.start.month)
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        while (y, m) <= ((end.year, end.month + 1) if end.month < 12 else (end.year + 1, 1)):
            months.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        run_months = set(months[1:-1])
        out = []
        for text in entries:
            if "{year" not in text and "{month" not in text:
                path = self.path(text.replace("{tower}", tower))
                if not path.exists():
                    raise ConfigError(f"{self.source}: {key}: file not found: {path}")
                out.append(path)
                continue
            for y, m in months:
                try:
                    name = text.format(tower=tower, year=y, month=m)
                except (KeyError, IndexError, ValueError) as err:
                    raise ConfigError(f"{self.source}: {key}: {text!r}: {err}") from None
                path = self.path(name)
                if path.exists():
                    out.append(path)
                elif (y, m) in run_months:
                    raise ConfigError(
                        f"{self.source}: {key}: file not found for {y}-{m:02d}: {path}"
                    )
        return list(dict.fromkeys(out))

    @property
    def output_directory(self) -> Path:
        return self.path(self.run_file.output.directory)

    @property
    def chunk_steps(self) -> Optional[int]:
        """Model steps per compiled chunk (``execution.chunk``; None: calendar months)."""
        days = parse_chunk(self.run_file.execution.chunk, "execution.chunk")
        return None if days is None else days * 86400 // self.step_seconds

    @property
    def step_seconds(self) -> int:
        """The model step (model.yaml numerics.step_seconds, default the forcing interval)."""
        return self.config.numerics.step_seconds or self.run_file.time.step_seconds

    @property
    def restart_path(self) -> Path:
        """Where this run writes its restart (the previous one is kept as ``.prev``)."""
        return self.output_directory / "restart" / f"{self.name}_restart.pkl"

    # -- running -----------------------------------------------------------------------

    def execute(
        self, *, write: bool = True, quiet: bool = False, return_outputs: bool = True
    ) -> dict:
        """Run the model and (if ``write``) write the output streams configured in run.yaml:
        one netCDF file per stream, level and ``file_per`` period, each written as soon as
        its period is complete.

        Returns ``{stream name: Outputs}`` (all records in memory), or ``{}`` with
        ``return_outputs=False``, so long runs keep only the files' records in memory.
        """
        from clm_ml.outputs import Outputs

        out = self.run_file.output
        out_dir = self.output_directory
        with tempfile.TemporaryDirectory() as tmp:
            raw_file = Path(tmp) / "streams.npz"
            spec = self.legacy_spec(None)
            spec["output_dir"] = str(out_dir) if write else None
            spec["return_records"] = return_outputs
            spec["restart_out"] = str(self.restart_path) if write else None
            self._run_worker(spec, raw_file, quiet)
            with np.load(raw_file) as data:
                arrays = {key: data[key] for key in data.files}
            if write and out.legacy_ascii:
                out_dir.mkdir(parents=True, exist_ok=True)
                legacy = dict(self.legacy_spec(out_dir), mode="legacy_ascii")
                self._run_worker(legacy, Path(tmp) / "unused.npz", quiet)

        if not return_outputs:
            return {}
        from clm_ml.outputs import domain_meta

        meta = dict(domain_meta(self.domain), soil_depth=arrays["static.soil_depth"])
        results = {}
        for stream in out.streams:
            prefix = f"{stream.name}/"
            records = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
            results[stream.name] = Outputs.from_records(records, meta)
        return results

    def stream_variables(self, stream) -> tuple[str, ...]:
        """A stream's variable names, with ``all`` expanded."""
        from clm_ml.outputs import VARIABLES

        names = []
        for v in stream.variables:
            names += [x.name for x in VARIABLES] + ["soil_temperature"] if v == "all" else [v]
        return tuple(dict.fromkeys(names))

    def _run_worker(self, spec: dict, out_file: Path, quiet: bool) -> None:
        spec_file = out_file.with_suffix(".json")
        spec_file.write_text(json.dumps(spec))
        src_dir = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src_dir), env.get("PYTHONPATH")]))
        subprocess.run(
            [sys.executable, "-m", "clm_ml.legacy.worker", str(spec_file), str(out_file)],
            cwd=src_dir,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL if quiet else None,
        )

    def patch_fortran_codes(self, patch: int, interpolation: str) -> dict:
        """model.yaml's Fortran option codes, with patch ``patch``'s explicit layer counts
        (towers.yaml ``n_layers_within``/``n_layers_above``) in place of model.yaml's when set."""
        codes = self.config.to_fortran_codes(interpolation)
        within = int(self.domain.patches.n_layers_within[patch])
        if within > 0:
            codes["nlayer_within"] = within
            codes["nlayer_above"] = int(self.domain.patches.n_layers_above[patch])
        return codes

    @property
    def name(self) -> str:
        """The run's name in file names: its tower, or ``domain`` for several gridcells."""
        names = self.domain.gridcells.names
        return names[0] if len(names) == 1 else "domain"

    def legacy_tower_name(self, patch: int) -> str:
        """Patch ``patch``'s entry in the legacy tower table: its gridcell's name, with
        ``#k`` (its place on the gridcell) when the gridcell has several patches."""
        gidx = self.domain.patches.gridcell_index
        g = int(gidx[patch])
        name = self.domain.gridcells.names[g]
        if int(np.sum(gidx == g)) == 1:
            return name
        return f"{name}#{int(np.sum(gidx[:patch] == g))}"

    def _patch_spec(self, patch: int) -> dict:
        """The per-patch part of the legacy spec: patch ``patch`` set up as a one-patch
        legacy tower (its gridcell's forcing files, its own tower values and layer counts)."""
        g = int(self.domain.patches.gridcell_index[patch])
        tower_name = self.domain.gridcells.names[g]
        f = self.run_file.forcing
        clm_start = _parse_datetime(f.clm_history.start, "forcing.clm_history.start")
        adjust = f.soil_water.adjustment
        met = f.meteorology
        tower_files = [str(p) for p in self.files(met.files, tower_name, "forcing.meteorology")]

        def land_files(key):
            section = getattr(f, key)
            if section.source != "file":
                return " "
            return [str(p) for p in self.files(section.files, tower_name, f"forcing.{key}")]

        namelist = {
            "tower_name": self.legacy_tower_name(patch),
            "start_ymd": int(f"{self.start:%Y%m%d}"),
            "start_tod": self.start.hour * 3600 + self.start.minute * 60 + self.start.second,
            "stop_option": "nsteps",
            "stop_n": self.n_steps,
            "clm_start_ymd": int(f"{clm_start:%Y%m%d}"),
            "clm_start_tod": clm_start.hour * 3600 + clm_start.minute * 60,
            "fin_tower": tower_files[0],
            "fin_tower_files": tower_files,
            "met_shortwave": met.shortwave,
            "met_precipitation": met.precipitation,
            "fin_clm": str(self.path(f.clm_history.file, tower_name)),
            "clm_phys": {"clm5_0": "CLM5_0", "clm4_5": "CLM4_5"}[self.config.soil.layer_structure],
            "fin_soil_adjust": str(self.path(adjust.file, tower_name)) if adjust else " ",
            "fin_vegetation": land_files("vegetation"),
            "fin_soil_water": land_files("soil_water"),
            "nlev_soil_adjust": adjust.levels if adjust else 0,
            "met_type": 3 if met.interpolation == "three_point" else 0,
            "dpai_min": self.config.canopy_layers.min_plant_area_per_layer,
            "pftcon_val": 0,
            "dtstep_seconds": self.step_seconds,
            "acclimation": self.config.physics.acclimation_temperature,
            "acclimation_days": self.config.physics.acclimation_days,
        }
        return {
            "namelist": namelist,
            "varctl": self.patch_fortran_codes(patch, met.interpolation),
            "tower_name": namelist["tower_name"],
            "tower": _legacy_tower_values(self.domain, patch),
        }

    def legacy_spec(self, legacy_ascii_dir: Optional[Path], patch: int = 0) -> dict:
        """Everything the legacy worker needs, as JSON-serializable values: patch ``patch``'s
        setup at the top level (one-patch models, ``clm_ml.model``), every patch's in
        ``patches``, and the domain's patch-gridcell map and weights in ``batch``. Grid runs
        (``grid``): the worker reloads the run file and builds its prototypes from the
        gridded data (``clm_ml.legacy.grid_setup``)."""
        tower_name = self.domain.gridcells.names[0]
        if self.is_grid:
            spec = {
                "grid": {"run_file": str(self.path_), "overrides": list(self.overrides)},
                "tower_name": "domain",
            }
            patches = []
        else:
            spec = self._patch_spec(patch)
            patches = [
                spec if p == patch else self._patch_spec(p) for p in range(self.domain.n_patches)
            ]
        return {
            **spec,
            "patches": patches,
            "batch": {
                "gridcell_index": self.domain.patches.gridcell_index.tolist(),
                "weight": self.domain.patches.weight.tolist(),
                "n_gridcells": self.domain.n_gridcells,
                "patch_count": self.run_file.execution.patch_count,
                "gridcell_count": self.run_file.execution.gridcell_count,
                "devices": self.run_file.execution.devices,
            },
            "file_prefix": self.name,
            "on_patch_error": self.config.numerics.on_patch_error,
            "max_failed_patch_fraction": self.config.numerics.max_failed_patch_fraction,
            "ignore_checks": list(self.config.numerics.ignore_checks),
            "params": {k: v.tolist() for k, v in self.params.values.items()},
            "params_source": self.params.source,
            "legacy_ascii_dir": str(legacy_ascii_dir) if legacy_ascii_dir else None,
            "max_layers": self.config.canopy_layers.max_layers,
            "streams": [
                {
                    "name": st.name,
                    "frequency": st.frequency,
                    "statistic": st.statistic,
                    "variables": list(self.stream_variables(st)),
                    "level": list(st.level),
                    "file_per": st.file_per,
                }
                for st in self.run_file.output.streams
            ],
            "meta": _domain_meta(self.domain),
            "run_start": self.start.isoformat(),
            "restart_every": parse_every(self.run_file.execution.restart_every, "restart_every"),
            "chunk_steps": self.chunk_steps,
            "restart_in": (
                str(self.path(self.run_file.initial_conditions.restart_file, tower_name))
                if self.run_file.initial_conditions.source == "restart"
                else None
            ),
        }

    def summary(self) -> str:
        """What ``clm-ml check`` prints."""
        d, c = self.domain, self.config
        end = self.start + dt.timedelta(seconds=self.n_steps * self.step_seconds)
        lines = [
            f"Run: {self.source}",
            f"  time       {self.start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}, "
            f"{self.n_steps} steps of {self.step_seconds} s",
            f"  params     {self.params.source}",
            f"  gridcells  {d.n_gridcells}   patches {d.n_patches}",
        ]
        shown_patches = 20 if self.is_grid else d.n_patches
        for i, n_layers in enumerate(d.layer_counts(c)[:shown_patches]):
            g = d.patches.gridcell_index[i]
            shown = "from CLM history" if n_layers == 0 else str(n_layers)
            lines.append(
                f"    patch {i}: {d.gridcells.names[g]}  {PFT_NAMES[d.patches.pft_index[i]]}  "
                f"weight {d.patches.weight[i]:g}  canopy layers {shown} "
                f"(max_layers {c.canopy_layers.max_layers})"
            )
        if shown_patches < d.n_patches:
            lines.append(f"    ... ({d.n_patches - shown_patches} more patches)")
        unvalidated = c.non_default_options()
        if unvalidated:
            lines.append(
                "  note: these options differ from the CLM-ml v2 defaults and have not been "
                f"validated against Fortran in the JAX translation: {', '.join(unvalidated)}"
            )
        lines.append("  OK")
        return "\n".join(lines)


def load_run(path: Path | str, overrides: tuple[str, ...] | list[str] = ()) -> Run:
    """Read ``run.yaml`` and everything it references, and validate it.

    ``overrides`` use dotted keys, as on the command line (``clm-ml run --set``)::

        load_run("run.yaml", ["model.physics.stomatal_conductance=medlyn", "time.days=7"])

    Keys starting with ``model.`` change the model configuration; others change run.yaml.
    """
    path = Path(path).resolve()
    data = read_yaml(path)
    model_overrides = [o[len("model.") :] for o in overrides if o.startswith("model.")]
    run_overrides = [o for o in overrides if not o.startswith("model.")]
    apply_overrides(data, run_overrides, source=path.name)
    old = sorted(set(data.get("output") or {}) & set(_OLD_OUTPUT_KEYS))
    if old:
        raise ConfigError(
            f"{path.name}: output: "
            + "; ".join(f"{k} was replaced by {_OLD_OUTPUT_KEYS[k]}" for k in old)
        )
    _check_forcing_keys(data.get("forcing"), path.name)
    run_file = from_dict(RunFile, data, source=path.name)
    base_dir = path.parent

    model_data = read_yaml(base_dir / run_file.model) if run_file.model else {}
    apply_overrides(model_data, model_overrides, source="model.yaml")
    config = ModelConfig.from_dict(model_data, source=run_file.model or "model.yaml")

    params_ref = run_file.params
    if params_ref.endswith((".yaml", ".yml")) and not Path(params_ref).is_absolute():
        params_ref = str(base_dir / params_ref)
    params = Params.from_yaml(params_ref)

    _check_supported(run_file, path.name)
    model_step = config.numerics.step_seconds or run_file.time.step_seconds
    if run_file.domain.type == "grid":
        return _load_grid_run(
            run_file, base_dir, config, params, path, tuple(overrides), model_step
        )
    towers_file = run_file.domain.towers_file
    domain = Domain.from_towers(
        run_file.domain.towers,
        (base_dir / towers_file) if towers_file else None,
        params=params,
        config=config,
        min_patch_weight=run_file.domain.min_patch_weight,
    )
    ex = run_file.execution
    for key, count, own in (
        ("patch_count", ex.patch_count, domain.n_patches),
        ("gridcell_count", ex.gridcell_count, domain.n_gridcells),
    ):
        if count is not None and count < own:
            raise ConfigError(f"{path.name}: execution.{key} ({count}) is less than {own}")
    if (ex.patch_count or domain.n_patches) > domain.n_patches and not (
        (ex.gridcell_count or domain.n_gridcells) > domain.n_gridcells
    ):
        raise ConfigError(
            f"{path.name}: execution.patch_count pads the patches: set gridcell_count above "
            f"{domain.n_gridcells} too (padding patches go on a padding gridcell)"
        )
    if ex.devices < 1:
        raise ConfigError(f"{path.name}: execution.devices must be at least 1")
    if ex.devices > 1 and (ex.patch_count or ex.gridcell_count):
        raise ConfigError(
            f"{path.name}: execution.patch_count/gridcell_count with devices > 1 is not "
            "supported (the device blocks are padded automatically)"
        )
    if run_file.output.legacy_ascii and domain.n_patches > 1:
        raise ConfigError(
            f"{path.name}: output.legacy_ascii: the legacy .out files (validation) hold one "
            f"patch; this domain has {domain.n_patches}"
        )
    # One clock for the whole run (decision 24): every gridcell's forcing has the run's step
    for g, name in enumerate(domain.gridcells.names):
        step_minutes = int(domain.gridcells.forcing_step_minutes[g])
        if run_file.time.step_seconds != step_minutes * 60:
            raise ConfigError(
                f"{path.name}: time.step_seconds is {run_file.time.step_seconds} but tower "
                f"{name} has forcing_step_minutes {step_minutes}"
            )

    if model_step <= 0 or run_file.time.step_seconds % model_step:
        raise ConfigError(
            f"model.yaml: numerics.step_seconds ({model_step}) must divide the forcing "
            f"interval (run.yaml time.step_seconds, {run_file.time.step_seconds} s)"
        )
    start, n_steps = _resolve_time(run_file.time, path.name, model_step)
    run = Run(run_file, base_dir, config, params, domain, start, n_steps, source=path.name,
              path_=path, overrides=tuple(overrides))  # fmt: skip
    forcing = run_file.forcing
    if forcing.meteorology is None or forcing.clm_history is None:
        raise ConfigError(
            f"{path.name}: forcing.meteorology and forcing.clm_history are required for towers"
        )
    for tower in domain.gridcells.names:  # every gridcell's forcing files ({tower} expanded)
        run.files(forcing.meteorology.files, tower, "forcing.meteorology.files")
        for key in ("vegetation", "soil_water"):
            section = getattr(forcing, key)
            if section.source == "file":
                if not section.files:
                    raise ConfigError(
                        f"{path.name}: forcing.{key}.files: required with source file"
                    )
                run.files(section.files, tower, f"forcing.{key}.files")
        files = {"forcing.clm_history.file": forcing.clm_history.file}
        if forcing.soil_water.adjustment:
            files["forcing.soil_water.adjustment.file"] = forcing.soil_water.adjustment.file
        for key, text in files.items():
            if not run.path(text, tower).exists():
                raise ConfigError(f"{path.name}: {key}: file not found: {run.path(text, tower)}")
    substeps = model_step / config.numerics.substep_seconds
    if substeps != int(substeps):
        raise ConfigError(
            f"model.yaml: numerics.substep_seconds ({config.numerics.substep_seconds:g}) must "
            f"divide the model step ({model_step} s)"
        )
    return run


def _load_grid_run(run_file, base_dir, config, params, path, overrides, model_step) -> Run:
    """A ``domain.type: grid`` run (Phase 6b): the domain from the surface data
    (:mod:`clm_ml.grid`), the gridded forcing (:mod:`clm_ml.grid_forcing`)."""
    from clm_ml.grid import load_grid_domain

    d, f, src = run_file.domain, run_file.forcing, path.name
    if d.grid is None:
        raise ConfigError(f"{src}: domain.grid: required with domain.type grid")
    if f.gridded is None:
        raise ConfigError(f"{src}: forcing.gridded: required with domain.type grid")
    if config.physics.acclimation_temperature != "running_mean":
        raise ConfigError(
            f"{src}: model.yaml physics.acclimation_temperature: grid runs need running_mean "
            "(the Fortran whole-run mean needs all of a run's forcing before it starts)"
        )
    if run_file.output.legacy_ascii:
        raise ConfigError(f"{src}: output.legacy_ascii: not available for grid runs")
    if run_file.initial_conditions.source != "clm_history" and not (
        run_file.initial_conditions.source == "restart"
    ):
        raise ConfigError(f"{src}: initial_conditions.source: clm_history or restart")
    if run_file.time.step_seconds != model_step:
        raise ConfigError(
            f"{src}: time.step_seconds: for grid runs, the model step ({model_step} s); the "
            "gridded streams are interpolated to it"
        )
    start, n_steps = _resolve_time(run_file.time, src, model_step)

    def resolve(text):
        p = Path(str(text)).expanduser()
        return p if p.is_absolute() else (base_dir / p).resolve()

    domain = load_grid_domain(
        d.grid, params=params, config=config, start=start, step_seconds=model_step,
        min_patch_weight=d.min_patch_weight, resolve=resolve,
    )  # fmt: skip
    ex = run_file.execution
    for key, count, own in (
        ("patch_count", ex.patch_count, domain.n_patches),
        ("gridcell_count", ex.gridcell_count, domain.n_gridcells),
    ):
        if count is not None and count < own:
            raise ConfigError(f"{src}: execution.{key} ({count}) is less than {own}")
    if ex.devices > 1 and (ex.patch_count or ex.gridcell_count):
        raise ConfigError(
            f"{src}: execution.patch_count/gridcell_count with devices > 1 is not supported"
        )
    substeps = model_step / config.numerics.substep_seconds
    if substeps != int(substeps):
        raise ConfigError(
            f"model.yaml: numerics.substep_seconds ({config.numerics.substep_seconds:g}) must "
            f"divide the model step ({model_step} s)"
        )
    return Run(run_file, base_dir, config, params, domain, start, n_steps, source=src,
               path_=path, overrides=overrides)  # fmt: skip


def _check_forcing_keys(forcing, source: str) -> None:
    """Reject the earlier forcing layout with the replacement, and let ``files`` be one
    path (normalized to a list, in place)."""
    if not isinstance(forcing, dict):
        return
    if isinstance(forcing.get("meteorology"), str):
        raise ConfigError(
            f"{source}: forcing.meteorology is now a section: "
            f"meteorology: {{files: {forcing['meteorology']}}}"
        )
    old = sorted(set(forcing) & set(_OLD_FORCING_KEYS))
    if old:
        raise ConfigError(
            f"{source}: forcing: "
            + "; ".join(f"{k} was replaced by {_OLD_FORCING_KEYS[k]}" for k in old)
        )
    for key in ("meteorology", "vegetation", "soil_water"):
        section = forcing.get(key)
        if isinstance(section, dict):
            if "file" in section:
                raise ConfigError(f"{source}: forcing.{key}.file was replaced by files")
            if isinstance(section.get("files"), str):
                section["files"] = [section["files"]]
            elif "files" in section and section["files"] is None:
                section["files"] = []


def _domain_meta(domain) -> dict:
    from clm_ml.outputs import domain_meta

    return domain_meta(domain)


#: Raw legacy fields allowed as stream variables (validation): "mlcanopy.<field>" etc.
_RAW_PREFIXES = ("mlcanopy.", "temperature.")


def _check_supported(run_file: RunFile, source: str) -> None:
    towers = list(run_file.domain.towers)
    if len(set(towers)) != len(towers):
        raise ConfigError(f"{source}: domain.towers: a tower is listed twice ({towers})")
    ic = run_file.initial_conditions
    if ic.source == "restart" and not ic.restart_file:
        raise ConfigError(
            f"{source}: initial_conditions.restart_file: required with source restart"
        )
    parse_chunk(run_file.execution.chunk, f"{source}: execution.chunk")
    parse_every(run_file.execution.restart_every, f"{source}: execution.restart_every")
    names = [st.name for st in run_file.output.streams]
    if len(set(names)) != len(names):
        raise ConfigError(f"{source}: output.streams: stream names must be unique ({names})")
    if not names:
        raise ConfigError(f"{source}: output.streams: at least one stream is needed")
    from clm_ml.outputs import VARIABLES_BY_NAME

    for st in run_file.output.streams:
        unknown = [
            v for v in st.variables
            if v not in ("all", "soil_temperature")
            and v not in VARIABLES_BY_NAME
            and not v.startswith(_RAW_PREFIXES)
        ]  # fmt: skip
        if unknown:
            raise ConfigError(
                f"{source}: output.streams[{st.name}].variables: unknown {unknown}; "
                "see clm_ml.outputs.VARIABLES"
            )


def _parse_datetime(text: str, key: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(str(text))
    except ValueError:
        raise ConfigError(f"run.yaml: {key}: {text!r} is not a date-time like 2007-05-01T00:00")


def _resolve_time(t: TimeSection, source: str, step_seconds: int) -> tuple[dt.datetime, int]:
    """Start and number of model steps (``step_seconds`` long)."""
    start = _parse_datetime(t.start, "time.start")
    given = [k for k in ("end", "days", "steps") if getattr(t, k) is not None]
    if len(given) != 1:
        raise ConfigError(f"{source}: time: give exactly one of end, days, steps (got {given})")
    if t.steps is not None:
        n = t.steps
    else:
        seconds = (
            (_parse_datetime(t.end, "time.end") - start).total_seconds()
            if t.end is not None
            else t.days * 86400.0
        )
        n = seconds / step_seconds
        if n != int(n):
            raise ConfigError(f"{source}: time: run length is not a whole number of steps")
    if int(n) < 1:
        raise ConfigError(f"{source}: time: the run must be at least one step long")
    return start, int(n)


def _legacy_tower_values(domain: Domain, patch: int = 0) -> dict:
    """Patch ``patch``'s tower values in ``TowerDataMod`` form (-999 = default): its
    gridcell's location and forcing, its soil column, and its own vegetation."""
    g, p, c = domain.gridcells, domain.patches, domain.columns
    gi, ci = int(p.gridcell_index[patch]), int(p.column_index[patch])

    def v(x):
        x = float(x)
        return -999.0 if np.isnan(x) else x

    def pair(x):
        return [-999.0, -999.0] if np.isnan(x).any() else [float(x[0]), float(x[1])]

    return {
        "tower_lat": v(g.latitude[gi]),
        "tower_lon": v(g.longitude[gi]),
        "tower_ht": v(g.measurement_height[gi]),
        "tower_time": int(g.forcing_step_minutes[gi]),
        "tower_tex": c.texture_class[ci] or "",
        "tower_sand": v(c.sand_percent[ci]),
        "tower_clay": v(c.clay_percent[ci]),
        "tower_organic": v(c.organic_matter[ci]),
        "tower_isoicol": int(c.color_class[ci]),
        "tower_zbed": v(c.bedrock_depth[ci]),
        "tower_pft": int(p.pft_index[patch]),
        "tower_canht": v(p.canopy_height[patch]),
        "tower_root": v(p.root_biomass[patch]),
        "tower_pbeta_lai": pair(p.leaf_area_profile_beta[patch]),
        "tower_pbeta_sai": pair(p.stem_area_profile_beta[patch]),
    }
