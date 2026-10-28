"""
A run: the contents of ``run.yaml`` and everything it references, loaded and validated.

``run.yaml`` replaces the Fortran namelist. It names the model and params files, the
towers, the dates, the forcing files, the initial conditions and the output.

In this phase (modernization plan, Phase 1) ``Run.execute()`` runs the legacy driver in a
separate process (:mod:`clm_ml.legacy.worker`), so results are identical to
``clm-ml-offline``. Supported for now: one tower with one patch, initial conditions from the
CLM history file, output every time step.
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


@dataclasses.dataclass(frozen=True)
class ClmHistorySection:
    file: str
    start: str  # first record of the history file (was clm_start_ymd / clm_start_tod)


@dataclasses.dataclass(frozen=True)
class SoilMoistureAdjustmentSection:
    file: str
    levels: int = 3  # was nlev_soil_adjust


@dataclasses.dataclass(frozen=True)
class ForcingSection:
    meteorology: str  # tower forcing netCDF; {tower} is replaced by the tower name
    clm_history: ClmHistorySection
    soil_moisture_adjustment: Optional[SoilMoistureAdjustmentSection] = None
    interpolation: Literal["none", "three_point"] = "three_point"  # met_type 0 / 3


@dataclasses.dataclass(frozen=True)
class ExecutionSection:
    """How the run is divided into compiled chunks and how often it saves restarts."""

    chunk: str = "1 month"  # decision 17: one compiled chunk per calendar month
    restart_every: Optional[str] = None  # "N month(s)" / "N year(s)"; null: end of run only


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

    # -- paths -------------------------------------------------------------------------

    def path(self, text: str, tower: Optional[str] = None) -> Path:
        if tower is not None:
            text = text.replace("{tower}", tower)
        path = Path(text).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def output_directory(self) -> Path:
        return self.path(self.run_file.output.directory)

    @property
    def step_seconds(self) -> int:
        """The model step (model.yaml numerics.step_seconds, default the forcing interval)."""
        return self.config.numerics.step_seconds or self.run_file.time.step_seconds

    @property
    def restart_path(self) -> Path:
        """Where this run writes its restart (the previous one is kept as ``.prev``)."""
        tower = self.domain.gridcells.names[0]
        return self.output_directory / "restart" / f"{tower}_restart.pkl"

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

    def legacy_spec(self, legacy_ascii_dir: Optional[Path]) -> dict:
        """Everything the legacy worker needs, as JSON-serializable values."""
        tower_name = self.domain.gridcells.names[0]
        f = self.run_file.forcing
        clm_start = _parse_datetime(f.clm_history.start, "forcing.clm_history.start")
        adjust = f.soil_moisture_adjustment
        namelist = {
            "tower_name": tower_name,
            "start_ymd": int(f"{self.start:%Y%m%d}"),
            "start_tod": self.start.hour * 3600 + self.start.minute * 60 + self.start.second,
            "stop_option": "nsteps",
            "stop_n": self.n_steps,
            "clm_start_ymd": int(f"{clm_start:%Y%m%d}"),
            "clm_start_tod": clm_start.hour * 3600 + clm_start.minute * 60,
            "fin_tower": str(self.path(f.meteorology, tower_name)),
            "fin_clm": str(self.path(f.clm_history.file, tower_name)),
            "clm_phys": {"clm5_0": "CLM5_0", "clm4_5": "CLM4_5"}[self.config.soil.layer_structure],
            "fin_soil_adjust": str(self.path(adjust.file, tower_name)) if adjust else " ",
            "nlev_soil_adjust": adjust.levels if adjust else 0,
            "met_type": 3 if f.interpolation == "three_point" else 0,
            "dpai_min": self.config.canopy_layers.min_plant_area_per_layer,
            "pftcon_val": 0,
            "dtstep_seconds": self.step_seconds,
            "acclimation": self.config.physics.acclimation_temperature,
            "acclimation_days": self.config.physics.acclimation_days,
        }
        return {
            "namelist": namelist,
            "varctl": self.config.to_fortran_codes(f.interpolation),
            "params": {k: v.tolist() for k, v in self.params.values.items()},
            "params_source": self.params.source,
            "tower_name": tower_name,
            "tower": _legacy_tower_values(self.domain),
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
        for i, n_layers in enumerate(d.layer_counts(c)):
            g = d.patches.gridcell_index[i]
            shown = "from CLM history" if n_layers == 0 else str(n_layers)
            lines.append(
                f"    patch {i}: {d.gridcells.names[g]}  {PFT_NAMES[d.patches.pft_index[i]]}  "
                f"weight {d.patches.weight[i]:g}  canopy layers {shown} "
                f"(max_layers {c.canopy_layers.max_layers})"
            )
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
    towers_file = run_file.domain.towers_file
    domain = Domain.from_towers(
        run_file.domain.towers,
        (base_dir / towers_file) if towers_file else None,
        params=params,
        config=config,
        min_patch_weight=run_file.domain.min_patch_weight,
    )
    if domain.n_patches != 1:
        raise ConfigError(
            f"{path.name}: this version runs one patch per run; "
            f"{run_file.domain.towers} has {domain.n_patches} (multi-patch runs: Phase 6)"
        )
    step_minutes = int(domain.gridcells.forcing_step_minutes[0])
    if run_file.time.step_seconds != step_minutes * 60:
        raise ConfigError(
            f"{path.name}: time.step_seconds is {run_file.time.step_seconds} but tower "
            f"{domain.gridcells.names[0]} has forcing_step_minutes {step_minutes}"
        )

    model_step = config.numerics.step_seconds or run_file.time.step_seconds
    if model_step <= 0 or run_file.time.step_seconds % model_step:
        raise ConfigError(
            f"model.yaml: numerics.step_seconds ({model_step}) must divide the forcing "
            f"interval (run.yaml time.step_seconds, {run_file.time.step_seconds} s)"
        )
    start, n_steps = _resolve_time(run_file.time, path.name, model_step)
    run = Run(run_file, base_dir, config, params, domain, start, n_steps, source=path.name)
    tower = domain.gridcells.names[0]
    files = {"forcing.meteorology": run_file.forcing.meteorology,
             "forcing.clm_history.file": run_file.forcing.clm_history.file}  # fmt: skip
    if run_file.forcing.soil_moisture_adjustment:
        files["forcing.soil_moisture_adjustment.file"] = (
            run_file.forcing.soil_moisture_adjustment.file
        )
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


def _domain_meta(domain) -> dict:
    from clm_ml.outputs import domain_meta

    return domain_meta(domain)


#: Raw legacy fields allowed as stream variables (validation): "mlcanopy.<field>" etc.
_RAW_PREFIXES = ("mlcanopy.", "temperature.")


def _check_supported(run_file: RunFile, source: str) -> None:
    if run_file.domain.type != "towers":
        raise ConfigError(f"{source}: domain.type: 'grid' is planned but not yet supported")
    if len(run_file.domain.towers) != 1:
        raise ConfigError(
            f"{source}: domain.towers: this version runs one tower per run "
            f"(got {list(run_file.domain.towers)}; multi-tower runs: Phase 6)"
        )
    ic = run_file.initial_conditions
    if ic.source == "restart" and not ic.restart_file:
        raise ConfigError(
            f"{source}: initial_conditions.restart_file: required with source restart"
        )
    if run_file.execution.chunk != "1 month":
        raise ConfigError(
            f"{source}: execution.chunk: only '1 month' (calendar-month chunks, decision 17) "
            "is supported"
        )
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


def _legacy_tower_values(domain: Domain) -> dict:
    """The single patch's tower values in ``TowerDataMod`` form (-999 = default)."""
    g, p, c = domain.gridcells, domain.patches, domain.columns

    def v(x):
        x = float(x)
        return -999.0 if np.isnan(x) else x

    def pair(x):
        return [-999.0, -999.0] if np.isnan(x).any() else [float(x[0]), float(x[1])]

    return {
        "tower_lat": v(g.latitude[0]),
        "tower_lon": v(g.longitude[0]),
        "tower_ht": v(g.measurement_height[0]),
        "tower_time": int(g.forcing_step_minutes[0]),
        "tower_tex": c.texture_class[0] or "",
        "tower_sand": v(c.sand_percent[0]),
        "tower_clay": v(c.clay_percent[0]),
        "tower_organic": v(c.organic_matter[0]),
        "tower_isoicol": int(c.color_class[0]),
        "tower_zbed": v(c.bedrock_depth[0]),
        "tower_pft": int(p.pft_index[0]),
        "tower_canht": v(p.canopy_height[0]),
        "tower_root": v(p.root_biomass[0]),
        "tower_pbeta_lai": pair(p.leaf_area_profile_beta[0]),
        "tower_pbeta_sai": pair(p.stem_area_profile_beta[0]),
    }
