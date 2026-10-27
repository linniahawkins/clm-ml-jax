"""
Convert a Fortran-style ``clmML_inparm`` namelist into ``run.yaml`` + ``model.yaml``.

Used by ``clm-ml convert``. Paths in the namelist are interpreted the way
``clm-ml-offline`` does (``../input_files/...`` relative to ``src/offline_executable``) and
written relative to the output directory.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import yaml

from clm_ml.io.yaml_schema import ConfigError

_SRC_DIR = Path(__file__).resolve().parents[2]


def _resolve_legacy_path(raw: str, namelist_dir: Path) -> Path | None:
    """Same search order as ``offline_executable.main._resolve_path``: as given,
    relative to the namelist, then with leading ``../`` stripped under ``src/``."""
    raw = raw.strip()
    if not raw:
        return None
    p = Path(raw)
    for candidate in (p, namelist_dir / p):
        if candidate.exists():
            return candidate.resolve()
    parts = p.parts
    while parts and parts[0] == "..":
        parts = parts[1:]
    candidate = _SRC_DIR / Path(*parts)
    return candidate.resolve() if candidate.exists() else (namelist_dir / p).resolve()


def _datetime(ymd: int, tod: int) -> dt.datetime:
    day = dt.datetime.strptime(str(int(ymd)), "%Y%m%d")
    return day + dt.timedelta(seconds=int(tod))


def convert_namelist(namelist_path: Path, out_dir: Path) -> dict[str, Path]:
    """Write ``run.yaml`` and ``model.yaml`` for ``namelist_path`` into ``out_dir``.

    Returns the paths written. ``pftcon_val = 1`` becomes ``params: chats``.
    """
    import f90nml

    namelist_path = Path(namelist_path).resolve()
    nml = f90nml.read(str(namelist_path))
    if "clmml_inparm" not in nml:
        raise ConfigError(f"{namelist_path.name}: no &clmML_inparm group")
    g = nml["clmml_inparm"]
    known = {
        "tower_name", "start_ymd", "start_tod", "stop_option", "stop_n", "clm_start_ymd",
        "clm_start_tod", "fin_tower", "fin_clm", "clm_phys", "fin_soil_adjust",
        "nlev_soil_adjust", "dirout", "met_type", "dpai_min", "pftcon_val", "progress_nsteps",
    }  # fmt: skip
    unknown = sorted(set(g) - known)
    if unknown:
        raise ConfigError(f"{namelist_path.name}: cannot convert unknown setting(s) {unknown}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def rel(raw: str) -> str | None:
        """Relative to the output directory when nearby, otherwise absolute."""
        path = _resolve_legacy_path(str(raw), namelist_path.parent)
        if path is None:
            return None
        relative = os.path.relpath(path, out_dir)
        return str(path) if relative.startswith(os.path.join("..", "..", "..")) else relative

    tower = str(g["tower_name"]).strip()
    start = _datetime(g.get("start_ymd"), g.get("start_tod", 0))
    stop_option = str(g.get("stop_option", "ndays")).strip()
    stop_n = int(g.get("stop_n", 1))
    if stop_option == "ndays":
        time = {"start": start.isoformat(timespec="minutes"), "days": stop_n}
    elif stop_option == "nsteps":
        time = {"start": start.isoformat(timespec="minutes"), "steps": stop_n}
    else:
        raise ConfigError(f"{namelist_path.name}: stop_option {stop_option!r} not supported")
    time["step_seconds"] = 1800

    met_type = int(g.get("met_type", 0))
    if met_type not in (0, 3):
        raise ConfigError(f"{namelist_path.name}: met_type {met_type} is not supported (0 or 3)")
    forcing = {
        "meteorology": rel(g["fin_tower"]),
        "interpolation": "three_point" if met_type == 3 else "none",
        "clm_history": {
            "file": rel(g["fin_clm"]),
            "start": _datetime(g.get("clm_start_ymd"), g.get("clm_start_tod", 0)).isoformat(
                timespec="minutes"
            ),
        },
    }
    adjust = rel(g.get("fin_soil_adjust", ""))
    if adjust:
        forcing["soil_moisture_adjustment"] = {
            "file": adjust,
            "levels": int(g.get("nlev_soil_adjust", 0)),
        }

    run = {
        "model": "model.yaml",
        "params": "chats" if int(g.get("pftcon_val", 0)) == 1 else "clm5_defaults",
        "time": time,
        "domain": {"type": "towers", "towers": [tower]},
        "forcing": forcing,
        "initial_conditions": {"source": "clm_history"},
        "output": {
            "directory": rel(g.get("dirout", "output")) or "output",
            "format": ["netcdf"],
        },
    }
    model = {
        "canopy_layers": {"min_plant_area_per_layer": float(g.get("dpai_min", 0.01))},
        "soil": {
            "layer_structure": {"CLM5_0": "clm5_0", "CLM4_5": "clm4_5"}[
                str(g.get("clm_phys", "CLM5_0")).strip()
            ]
        },
    }

    header = f"# Converted from {namelist_path.name} by `clm-ml convert`.\n"
    written = {}
    for name, content in (("run.yaml", run), ("model.yaml", model)):
        path = out_dir / name
        with open(path, "w") as fh:
            fh.write(header)
            if name == "model.yaml":
                fh.write("# Only settings that came from the namelist are listed; everything\n")
                fh.write("# else uses the defaults (CLM-ml v2 / CHATS configuration).\n")
            yaml.safe_dump(content, fh, sort_keys=False)
        written[name] = path
    return written
