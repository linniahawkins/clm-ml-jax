"""
Bridge between the new YAML tables and the legacy (Fortran-translated) module tables.

The legacy code builds its PFT parameter tables (``pftconMod.pftcon``,
``MLpftconMod.MLpftcon``) and its tower table (``TowerDataMod``) as module globals at
import time, switching CHATS values on with ``MLclm_varctl.pftcon_val``. This module:

- reads those tables (:func:`dump_legacy_pft_tables` -- must run in a fresh process,
  because the tables are fixed at import),
- generates the shipped YAML files from them (``python -m clm_ml.legacy.tables --write``),
- converts :class:`~clm_ml.params.Params` back into legacy tables for the adapter
  (:func:`params_to_legacy`).

Fortran reference: pftconMod.F90 / MLpftconMod.F90 (InitRead), TowerDataMod.F90.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from clm_ml.params import PARAM_SPECS, PFT_NAMES, Params

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
N_PFT = len(PFT_NAMES)  # rows 0..16 carried by Params; legacy tables are larger (0..mxpft)

#: TowerDataMod array -> (towers.yaml section, key). "-999" in the legacy table means default.
TOWER_FIELDS = {
    "tower_lat": ("gridcell", "latitude"),
    "tower_lon": ("gridcell", "longitude"),
    "tower_ht": ("gridcell", "measurement_height"),
    "tower_time": ("gridcell", "forcing_step_minutes"),
    "tower_tex": ("soil", "texture_class"),
    "tower_sand": ("soil", "sand_percent"),
    "tower_clay": ("soil", "clay_percent"),
    "tower_organic": ("soil", "organic_matter"),
    "tower_isoicol": ("soil", "color_class"),
    "tower_zbed": ("soil", "bedrock_depth"),
    "tower_pft": ("patch", "pft"),
    "tower_canht": ("patch", "canopy_height"),
    "tower_root": ("patch", "root_biomass"),
    "tower_pbeta_lai": ("patch", "leaf_area_profile_beta"),
    "tower_pbeta_sai": ("patch", "stem_area_profile_beta"),
}


def dump_legacy_pft_tables(pftcon_val: int) -> dict[str, np.ndarray]:
    """Import the legacy PFT modules with ``pftcon_val`` and return their tables.

    Keys are ``"<table>.<field>"``. Call only in a fresh process (see module docstring).
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from multilayer_canopy import MLclm_varctl

    MLclm_varctl.pftcon_val = pftcon_val
    from clm_src_main import pftconMod
    from multilayer_canopy import MLpftconMod

    tables = {}
    for table_name, table in (("pftcon", pftconMod.pftcon), ("MLpftcon", MLpftconMod.MLpftcon)):
        for field in table._fields:
            tables[f"{table_name}.{field}"] = np.asarray(getattr(table, field))
    return tables


def params_to_legacy(params: Params, base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Overwrite rows of the legacy ``base`` tables with the values in ``params``.

    Rows where ``params`` has no value (NaN) keep the legacy value, so the untouched parts
    of the legacy tables (including their -999 placeholders) are reproduced exactly.
    Spectral parameters are stored in legacy columns 1 (vis) and 2 (nir); column 0 is the
    unused Fortran 1-based slot.
    """
    out = {key: value.copy() for key, value in base.items()}
    for spec in PARAM_SPECS:
        key = f"{spec.legacy_table}.{spec.legacy_field}"
        values = params.values[spec.name]
        for row in range(N_PFT):
            if np.isnan(values[row]).any():
                continue
            if spec.spectral:
                out[key][row, 1:3] = values[row]
            else:
                out[key][row] = values[row]
    return out


def legacy_to_params_dict(tables: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    """Legacy tables -> ``{pft_name: {param: value}}`` for PFTs 1..16."""
    out = {}
    for row in range(1, N_PFT):
        entries = {}
        for spec in PARAM_SPECS:
            table = tables[f"{spec.legacy_table}.{spec.legacy_field}"]
            if spec.spectral:
                entries[spec.name] = [float(table[row, 1]), float(table[row, 2])]
            elif spec.name == "c3_photosynthesis":
                entries[spec.name] = bool(table[row] == 1.0)
            else:
                entries[spec.name] = float(table[row])
        out[PFT_NAMES[row]] = entries
    return out


def dump_legacy_tower(name: str) -> dict:
    """One tower from ``TowerDataMod`` in ``towers.yaml`` form (``None`` for -999)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from offline_driver import TowerDataMod

    index = [str(t) for t in TowerDataMod.tower_id].index(name)
    entry = {"gridcell": {}, "soil": {}, "patch": {}}
    for array_name, (section, key) in TOWER_FIELDS.items():
        value = getattr(TowerDataMod, array_name)[index]
        if isinstance(value, str):
            entry[section][key] = value
            continue
        value = np.asarray(value)
        if key == "pft":
            entry[section][key] = PFT_NAMES[int(value)]
        elif value.ndim == 1:
            entry[section][key] = None if (value == -999).all() else [float(v) for v in value]
        elif value == -999:
            entry[section][key] = None
        elif np.issubdtype(value.dtype, np.integer):
            entry[section][key] = int(value)
        else:
            entry[section][key] = float(value)
    return entry


def _run_fresh(code: str) -> object:
    """Run ``code`` (which prints JSON) in a fresh interpreter with ``src/`` on the path."""
    src = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=src, capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def fresh_legacy_pft_params(pftcon_val: int) -> dict[str, dict[str, object]]:
    """:func:`legacy_to_params_dict` of the legacy tables, computed in a fresh process."""
    return _run_fresh(
        "import json; from clm_ml.legacy.tables import *; "
        f"print(json.dumps(legacy_to_params_dict(dump_legacy_pft_tables({pftcon_val}))))"
    )


def fresh_legacy_tower(name: str) -> dict:
    return _run_fresh(
        "import json; from clm_ml.legacy.tables import dump_legacy_tower; "
        f"print(json.dumps(dump_legacy_tower({name!r})))"
    )


def write_data_files() -> None:
    """Regenerate ``data/params/clm5_defaults.yaml``, ``data/params/chats.yaml`` and
    ``data/towers.yaml`` from the legacy tables."""
    defaults = fresh_legacy_pft_params(0)
    chats_full = fresh_legacy_pft_params(1)
    units = {spec.name: spec.units for spec in PARAM_SPECS}

    header = (
        "# Generated by `python -m clm_ml.legacy.tables --write` from the legacy tables\n"
        "# (pftconMod.py / MLpftconMod.py with pftcon_val = {val}). Do not edit by hand.\n"
    )
    params_dir = DATA_DIR / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    with open(params_dir / "clm5_defaults.yaml", "w") as fh:
        fh.write(header.format(val=0))
        fh.write("# CLM5 / CLM-ml v1 default parameters for PFTs 1-16.\n\n")
        yaml.safe_dump({"units": units, "pft": defaults}, fh, sort_keys=False, width=100)

    overrides = {}
    for pft, entries in chats_full.items():
        changed = {k: v for k, v in entries.items() if v != defaults[pft][k]}
        if changed:
            overrides[pft] = changed
    with open(params_dir / "chats.yaml", "w") as fh:
        fh.write(header.format(val=1))
        fh.write(
            "# CHATS walnut orchard overrides (Bonan, Burns & Patton 2026, Table 2):\n"
            "# Vcmax25 and iota from Rosati et al. (2006); plant hydraulics from Tyree et al.\n"
            "# (1993, 1994); psi50 from SPA-walnut; optics from Majasalmi & Bright (2019).\n\n"
        )
        yaml.safe_dump({"extends": "clm5_defaults", "pft": overrides}, fh, sort_keys=False)

    tower = fresh_legacy_tower("CHATS7")
    with open(DATA_DIR / "towers.yaml", "w") as fh:
        fh.write(
            "# Built-in towers. Generated by `python -m clm_ml.legacy.tables --write` from\n"
            "# TowerDataMod.py. Only CHATS7 (validated against Fortran CLM-ml v2) is shipped.\n"
            "# null means 'use the default' (was -999).\n\n"
        )
        yaml.safe_dump({"CHATS7": tower_yaml_entry(tower)}, fh, sort_keys=False)


def tower_yaml_entry(tower: dict) -> dict:
    """Legacy tower dict -> towers.yaml layout (gridcell fields at top level)."""
    entry = dict(tower["gridcell"])
    entry["soil"] = dict(tower["soil"])
    entry["patches"] = [dict(tower["patch"], weight=1.0)]
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="regenerate the data files")
    args = parser.parse_args()
    if not args.write:
        parser.error("nothing to do; pass --write")
    write_data_files()
    print(f"Wrote data files under {DATA_DIR}")


if __name__ == "__main__":
    main()
