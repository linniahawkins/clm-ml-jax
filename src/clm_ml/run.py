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
    step_seconds: int = 1800  # forcing time step


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
class InitialConditionsSection:
    source: Literal["clm_history", "restart"] = "clm_history"
    restart_file: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class OutputSection:
    directory: str = "output"
    format: tuple[Literal["netcdf", "legacy_ascii"], ...] = ("netcdf",)
    variables: tuple[str, ...] = ("all",)
    frequency: Literal["step", "hourly", "daily"] = "step"
    level: tuple[Literal["patch", "gridcell"], ...] = ("patch", "gridcell")


@dataclasses.dataclass(frozen=True)
class RunFile:
    """The ``run.yaml`` file as written (paths not yet resolved)."""

    time: TimeSection
    domain: DomainSection
    forcing: ForcingSection
    model: Optional[str] = None  # model.yaml path; null = all defaults
    params: str = "clm5_defaults"  # params file path or built-in name (clm5_defaults, chats)
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

    # -- running -----------------------------------------------------------------------

    def execute(self, *, write: bool = True, quiet: bool = False):
        """Run the model and (if ``write``) write the outputs configured in run.yaml.

        Returns :class:`~clm_ml.outputs.Outputs`.
        """
        from clm_ml.outputs import Outputs

        out = self.run_file.output
        out_dir = self.output_directory
        legacy_dir = out_dir if (write and "legacy_ascii" in out.format) else None
        with tempfile.TemporaryDirectory() as tmp:
            spec_file, raw_file = Path(tmp) / "spec.json", Path(tmp) / "raw.npz"
            spec_file.write_text(json.dumps(self.legacy_spec(legacy_dir)))
            src_dir = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src_dir), env.get("PYTHONPATH")]))
            subprocess.run(
                [sys.executable, "-m", "clm_ml.legacy.worker", str(spec_file), str(raw_file)],
                cwd=src_dir,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL if quiet else None,
            )
            with np.load(raw_file) as data:
                raw = {key: data[key] for key in data.files}

        outputs = Outputs.from_legacy(raw, self)
        if out.variables != ("all",):
            outputs = outputs.select(out.variables)
        if write and "netcdf" in out.format:
            out_dir.mkdir(parents=True, exist_ok=True)
            tower = self.domain.gridcells.names[0]
            outputs.to_netcdf(out_dir / f"{tower}_{self.start:%Y-%m-%d}.nc", levels=out.level)
        return outputs

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
        }
        return {
            "namelist": namelist,
            "varctl": self.config.to_fortran_codes(f.interpolation),
            "params": {k: v.tolist() for k, v in self.params.values.items()},
            "params_source": self.params.source,
            "tower_name": tower_name,
            "tower": _legacy_tower_values(self.domain),
            "legacy_ascii_dir": str(legacy_ascii_dir) if legacy_ascii_dir else None,
        }

    def summary(self) -> str:
        """What ``clm-ml check`` prints."""
        d, c = self.domain, self.config
        end = self.start + dt.timedelta(seconds=self.n_steps * self.run_file.time.step_seconds)
        lines = [
            f"Run: {self.source}",
            f"  time       {self.start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}, "
            f"{self.n_steps} steps of {self.run_file.time.step_seconds} s",
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

    start, n_steps = _resolve_time(run_file.time, path.name)
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
    substeps = run_file.time.step_seconds / config.numerics.substep_seconds
    if substeps != int(substeps):
        raise ConfigError(
            f"model.yaml: numerics.substep_seconds ({config.numerics.substep_seconds:g}) must "
            f"divide the forcing step ({run_file.time.step_seconds} s)"
        )
    return run


def _check_supported(run_file: RunFile, source: str) -> None:
    if run_file.domain.type != "towers":
        raise ConfigError(f"{source}: domain.type: 'grid' is planned but not yet supported")
    if len(run_file.domain.towers) != 1:
        raise ConfigError(
            f"{source}: domain.towers: this version runs one tower per run "
            f"(got {list(run_file.domain.towers)}; multi-tower runs: Phase 6)"
        )
    if run_file.initial_conditions.source != "clm_history":
        raise ConfigError(f"{source}: initial_conditions.source: 'restart' is not yet supported")
    if run_file.output.frequency != "step":
        raise ConfigError(f"{source}: output.frequency: only 'step' is supported so far")


def _parse_datetime(text: str, key: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(str(text))
    except ValueError:
        raise ConfigError(f"run.yaml: {key}: {text!r} is not a date-time like 2007-05-01T00:00")


def _resolve_time(t: TimeSection, source: str) -> tuple[dt.datetime, int]:
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
        n = seconds / t.step_seconds
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
