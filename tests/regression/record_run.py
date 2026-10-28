"""
Run the CHATS7 reference case and record its output at full precision, with noise twins.

The six legacy ``.out`` files hold 3 decimal places, too coarse to catch refactoring mistakes.
This runs the case through ``clm_ml`` (the same driver as ``clm-ml-offline``; Phase 1 showed
the two routes are bit-for-bit identical) and saves every field the output writer reads, as
float64, at every time step.

It also records two **noise twins**: the same run with ``leaf_emissivity × (1 ± 1e-12)``. Their
differences from the reference measure the model's own sensitivity to round-off, which the
regression test uses as its tolerance for changes that alter round-off
(docs/design/numerics_and_regression.md §5–6).

Regenerate the reference after an intended change in results, or after a change that alters
only round-off and passed the noise-twin test::

    python tests/regression/record_run.py --update-reference
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

# Reference case: CHATS7 from 2007-05-01 00Z, 50 half-hour steps (one full diurnal cycle).
REFERENCE_NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.50steps"
REFERENCE_FILE = REFERENCE_DIR / "CHATS7_50steps.npz"
TWIN_FILES = {
    "plus": REFERENCE_DIR / "CHATS7_50steps_twin_plus.npz",
    "minus": REFERENCE_DIR / "CHATS7_50steps_twin_minus.npz",
}
TWIN_PERTURBATION = 1e-12  # relative change of leaf_emissivity in the twins
TWIN_PFT = "broadleaf_deciduous_temperate_tree"  # CHATS7's PFT

# Prefixes of the recorded fields kept in the reference: every mlcanopy field the legacy
# output writer reads, soil temperature, and the calendar day.
RECORDED_PREFIXES = ("mlcanopy.", "temperature.", "calendar_day")


def write_portable_namelist(source: Path, dest: Path, output_dir: Path) -> Path:
    """Copy a namelist, making input paths absolute and redirecting output to ``output_dir``."""
    import f90nml

    nml = f90nml.read(str(source))
    group = nml["clmML_inparm"]
    for key in ("fin_tower", "fin_clm", "fin_soil_adjust"):
        raw = str(group[key]).strip()
        if raw and not Path(raw).is_absolute():
            group[key] = str((SRC_DIR / Path(*Path(raw).parts[1:])).resolve())
    group["dirout"] = str(output_dir) + "/"
    nml.write(str(dest), force=True)
    return dest


def run_case(workdir: Path, *, params: str | None = None, legacy_ascii: bool = False):
    """Run the reference case through ``clm_ml`` in ``workdir``.

    Returns ``(run, outputs)``. ``params`` overrides the params file (e.g. for twins).
    """
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import yaml

    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.run import load_run

    workdir = Path(workdir)
    convert_namelist(REFERENCE_NAMELIST, workdir)
    run_yaml = workdir / "run.yaml"
    data = yaml.safe_load(run_yaml.read_text())
    formats = ["netcdf", "legacy_ascii"] if legacy_ascii else ["netcdf"]
    data["output"] = {"directory": str(workdir / "out"), "format": formats}
    if params is not None:
        data["params"] = params
    run_yaml.write_text(yaml.safe_dump(data))
    run = load_run(run_yaml)
    return run, run.execute(quiet=True)


def recorded_fields(raw) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in raw.items() if k.startswith(RECORDED_PREFIXES)}


def _record_one(name: str, out_file: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        params = None
        if name != "reference":
            sign = 1.0 if name == "plus" else -1.0
            params = str(_twin_params_file(Path(tmp), sign))
        _, outputs = run_case(Path(tmp) / "case", params=params)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_file, **recorded_fields(outputs.raw), **_provenance(name))
    return f"wrote {out_file}"


def _twin_params_file(directory: Path, sign: float) -> Path:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from clm_ml.params import Params

    emissivity = float(Params.from_yaml("chats").get(TWIN_PFT, "leaf_emissivity"))
    path = directory / "twin_params.yaml"
    path.write_text(
        "extends: chats\n"
        f"pft:\n  {TWIN_PFT}:\n"
        f"    leaf_emissivity: {emissivity * (1.0 + sign * TWIN_PERTURBATION)!r}\n"
    )
    return path


def _provenance(name: str) -> dict[str, np.ndarray]:
    import jax

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "meta.git_commit": np.asarray(commit),
        "meta.jax_version": np.asarray(jax.__version__),
        "meta.created_utc": np.asarray(
            datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        ),
        "meta.case": np.asarray(name),
    }


def update_reference() -> None:
    """Record the reference and both twins (three runs, in parallel processes)."""
    jobs = {"reference": REFERENCE_FILE, **TWIN_FILES}
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for message in pool.map(_record_one, jobs.keys(), jobs.values()):
            print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--update-reference",
        action="store_true",
        help="regenerate the reference and its noise twins under tests/regression/reference/",
    )
    args = parser.parse_args()
    if not args.update_reference:
        parser.error("nothing to do; pass --update-reference")
    update_reference()


if __name__ == "__main__":
    main()
