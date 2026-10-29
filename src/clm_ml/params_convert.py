"""
Write a complete ``params.yaml`` (every parameter of every PFT 1-16, no ``extends``) from a
CLM parameter file or from another params file::

    clm-ml params ctsm60_params.c260410.nc -o params_clm6.yaml        # CLM file
    clm-ml params ctsm60_params.c260410.nc --base chats -o out.yaml   # other CLM-ml values
    clm-ml params chats -o chats_full.yaml                            # flatten a params file

From a CLM parameter file (netCDF with ``pftname``), the parameters CLM defines
(:data:`CLM_PARAMETER_FILE_FIELDS`) take the file's values; the CLM-ml-specific ones
(stomatal optimization, plant hydraulics, ``vcmax25``, ...), which a CLM file does not have,
take the ``--base`` params file's (default ``clm5_defaults``). The header of the written file
records the sources. The model reads only the YAML.

Fortran reference: CTSM pftconMod.F90 (InitRead).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from clm_ml.io.yaml_schema import ConfigError
from clm_ml.params import PARAM_SPECS, PFT_NAMES, Params

#: params.yaml name -> (CLM parameter-file variable(s), scale). Spectral parameters take a
#: (vis, nir) pair of variables. ``rootprof_beta`` has two variants; CLM's root profile for
#: water (``rooting_profile_varindex_water`` = 1), which CLM-ml's root profile is, is the
#: first.
CLM_PARAMETER_FILE_FIELDS: Mapping[str, tuple] = {
    "c3_photosynthesis": (("c3psn",), 1.0),
    "root_profile_beta": (("rootprof_beta",), 1.0),
    "leaf_dimension": (("dleaf",), 1.0),
    "leaf_angle_index": (("xl",), 1.0),
    "sla_top": (("slatop",), 1.0),
    "leaf_reflectance": (("rholvis", "rholnir"), 1.0),
    "leaf_transmittance": (("taulvis", "taulnir"), 1.0),
    "stem_reflectance": (("rhosvis", "rhosnir"), 1.0),
    "stem_transmittance": (("tausvis", "tausnir"), 1.0),
    "g1_medlyn": (("medlynslope",), 1.0),
    "g0_medlyn": (("medlynintercept",), 1.0e-6),  # umol H2O m-2 s-1 -> mol H2O m-2 s-1
    "g1_ball_berry": (("mbbopt",), 1.0),
    "z0m_to_height_ratio": (("z0mr",), 1.0),
    "displacement_to_height_ratio": (("displar",), 1.0),
}


def read_clm_parameter_file(path: Path) -> dict[str, dict]:
    """``{pft name: {params.yaml name: value}}`` for PFTs 1-16 of a CLM parameter file
    (:data:`CLM_PARAMETER_FILE_FIELDS`). The file's PFT names must match
    :data:`clm_ml.params.PFT_NAMES` row by row."""
    import netCDF4 as nc

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"CLM parameter file not found: {path}")
    n = len(PFT_NAMES)
    with nc.Dataset(path) as ds:
        names = [str(x).strip() for x in nc.chartostring(ds.variables["pftname"][:n])]
        if names != list(PFT_NAMES):
            raise ConfigError(f"{path}: pftname rows 0-{n - 1} are {names}, not {PFT_NAMES}")

        def read(var):
            if var not in ds.variables:
                raise ConfigError(f"{path}: variable {var} not found")
            v = np.asarray(ds.variables[var][:], dtype=float)
            return v[0, :n] if v.ndim == 2 else v[:n]  # rootprof_beta: (variants, pft)

        values: dict[str, dict] = {name: {} for name in PFT_NAMES[1:]}
        for key, (variables, scale) in CLM_PARAMETER_FILE_FIELDS.items():
            columns = [read(v) * scale for v in variables]
            for row, name in enumerate(PFT_NAMES[1:], start=1):
                if key == "c3_photosynthesis":
                    values[name][key] = bool(round(columns[0][row]) == 1)
                elif len(columns) == 2:
                    values[name][key] = [float(columns[0][row]), float(columns[1][row])]
                else:
                    values[name][key] = float(columns[0][row])
    return values


def complete_table(source: str | Path, base: str | Path = "clm5_defaults") -> tuple[dict, str]:
    """``({pft name: {name: value}}, description)``: every parameter of every PFT from
    ``source`` (a CLM parameter file ``.nc``, or a params file / built-in name), the values a
    CLM file lacks from ``base``."""
    text = str(source)
    if text.endswith(".nc"):
        clm = read_clm_parameter_file(Path(text))
        table = _table(Params.from_yaml(base))
        for pft, entries in clm.items():
            table[pft].update(entries)
        taken = ", ".join(CLM_PARAMETER_FILE_FIELDS)
        about = (
            f"From the CLM parameter file {Path(text).resolve()}: {taken}.\n"
            f"Everything else (CLM-ml-specific parameters) from {base}."
        )
        return table, about
    params = Params.from_yaml(text)
    return _table(params), f"The params file {params.source} with its extends chain flattened."


def _table(params: Params) -> dict:
    """``{pft name: {name: value}}`` of a :class:`Params` (NaN entries left out)."""
    table = {}
    for row, pft in enumerate(PFT_NAMES[1:], start=1):
        entries = {}
        for spec in PARAM_SPECS:
            v = params.values[spec.name][row]
            if np.isnan(v).any():
                continue
            if spec.spectral:
                entries[spec.name] = [float(v[0]), float(v[1])]
            elif spec.name == "c3_photosynthesis":
                entries[spec.name] = bool(v == 1.0)
            else:
                entries[spec.name] = float(v)
        table[pft] = entries
    return table


def write_params_yaml(source, out: Path, base="clm5_defaults") -> Path:
    """Write the complete params file of ``source`` (:func:`complete_table`) to ``out``."""
    import yaml

    table, about = complete_table(source, base)
    order = [s.name for s in PARAM_SPECS]
    table = {pft: {k: e[k] for k in order if k in e} for pft, e in table.items()}
    units = {s.name: s.units for s in PARAM_SPECS}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import textwrap

    lines = [f"Generated by `clm-ml params {source}` (clm_ml.params_convert).",
             "A complete table: every parameter of PFTs 1-16. Edit or extend it like any",
             "params.yaml."]  # fmt: skip
    for paragraph in about.splitlines():
        lines += textwrap.wrap(paragraph, 96, break_on_hyphens=False, break_long_words=False)
    header = "".join(f"# {line}\n" for line in lines)
    with open(out, "w") as fh:
        fh.write(header + "\n")
        yaml.safe_dump({"units": units, "pft": table}, fh, sort_keys=False, width=100)
    Params.from_yaml(out)  # it must load
    return out
