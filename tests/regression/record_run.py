"""
Run the offline model and record its output state at full precision.

The six ``.out`` files are written with 3 decimal places, which is too coarse to
catch refactoring mistakes. This script runs ``offline_executable.main`` exactly as
the CLI does, but wraps ``CLMml_driver.output`` so that every field the output
writer reads is also saved as float64 at every time step.

Run as a separate process (the model keeps state in module globals, so it must not
share an interpreter with other tests)::

    python tests/regression/record_run.py NAMELIST OUT.npz

To regenerate the committed reference after an *intended* change in results::

    python tests/regression/record_run.py --update-reference
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

# Reference case: CHATS7 from 2007-05-01 00Z, 50 half-hour steps (one full diurnal cycle).
REFERENCE_NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.50steps"
REFERENCE_FILE = REFERENCE_DIR / "CHATS7_50steps.npz"

# Every mlcanopy_inst field read by CLMml_driver.output() (i.e. everything that reaches
# the .out files), plus soil temperature. Keys in the .npz are "mlcanopy.<field>".
# fmt: off
MLCANOPY_FIELDS = (
    "albcan_canopy", "anet_leaf", "apar_leaf", "beta_canopy", "btran_soil", "dpai_profile",
    "dz_profile", "eair_profile", "etair_profile", "evveg_canopy", "fracminlwp_canopy",
    "fracsun_profile", "gac_profile", "gppveg_canopy", "gppvegsha_canopy", "gppvegsun_canopy",
    "gs_leaf", "gsoi_soil", "gsveg_canopy", "gsvegsha_canopy", "gsvegsun_canopy", "lai_canopy",
    "laisha_canopy", "laisun_canopy", "lhflx_canopy", "lhleaf_leaf", "lhsoi_soil", "lhveg_canopy",
    "lhvegsha_canopy", "lhvegsun_canopy", "lsc_profile", "lwdwn_profile", "lwp_hist_leaf",
    "lwp_mean_profile", "lwup_canopy", "lwupw_profile", "mflx_profile", "nbot_canopy",
    "ncan_canopy", "ntop_canopy", "pref_forcing", "psis_soil", "rhomol_forcing", "rnet_canopy",
    "rnleaf_leaf", "rnsoi_soil", "sai_canopy", "shair_profile", "shflx_canopy", "shleaf_leaf",
    "shsoi_soil", "shveg_canopy", "shvegsha_canopy", "shvegsun_canopy", "solar_zen_forcing",
    "stflx_air_canopy", "stflx_veg_canopy", "swbeam_profile", "swdwn_profile", "swskyb_forcing",
    "swskyd_forcing", "swupw_profile", "swveg_canopy", "swvegsha_canopy", "swvegsun_canopy",
    "tair_profile", "taveg_canopy", "tavegsha_canopy", "tavegsun_canopy", "tleaf_hist_leaf",
    "tlveg_canopy", "tlvegsha_canopy", "tlvegsun_canopy", "tref_forcing", "trveg_canopy",
    "ustar_canopy", "vcmax25_leaf", "vcmax25sha_canopy", "vcmax25sun_canopy",
    "vcmax25veg_canopy", "wind_profile", "windveg_canopy", "windvegsha_canopy",
    "windvegsun_canopy", "zs_profile", "zw_profile",
)
# fmt: on
TEMPERATURE_FIELDS = ("t_soisno_col",)


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


def record(namelist: Path, out_file: Path) -> None:
    """Run the model in this process and save the recorded fields to ``out_file``."""
    sys.path.insert(0, str(SRC_DIR))
    # Import main first: it enables float64 before any model module creates arrays.
    import offline_executable.main as main_mod

    records: dict[str, list[np.ndarray]] = {}
    calday: list[float] = []

    def recording_output(original_output):
        def wrapper(curr_calday, tower_num, *rest):
            mlcanopy_inst, temperature_inst = rest[-2], rest[-1]
            calday.append(float(curr_calday))
            for name in MLCANOPY_FIELDS:
                records.setdefault(f"mlcanopy.{name}", []).append(
                    np.asarray(getattr(mlcanopy_inst, name))
                )
            for name in TEMPERATURE_FIELDS:
                records.setdefault(f"temperature.{name}", []).append(
                    np.asarray(getattr(temperature_inst, name))
                )
            return original_output(curr_calday, tower_num, *rest)

        return wrapper

    # CLMml_driver imports clm_instMod, which must not be imported before
    # InitializeRealize (see main.py), so install the recorder right after it runs.
    original_initialize = main_mod.InitializeRealize

    def initialize_then_install_recorder(bounds):
        original_initialize(bounds)
        import offline_driver.CLMml_driver as driver_mod

        driver_mod.output = recording_output(driver_mod.output)

    main_mod.InitializeRealize = initialize_then_install_recorder
    sys.argv = ["clm-ml-offline", str(namelist)]
    main_mod.main()

    if not calday:
        raise RuntimeError("model ran but no output steps were recorded")
    arrays = {key: np.stack(values) for key, values in records.items()}
    arrays["calendar_day"] = np.asarray(calday)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **arrays, **_provenance())


def _provenance() -> dict[str, np.ndarray]:
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
    }


def run_in_subprocess(namelist: Path, out_file: Path, workdir: Path) -> None:
    """Make a portable copy of ``namelist`` in ``workdir`` and record a run in a fresh process."""
    output_dir = workdir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    portable = write_portable_namelist(namelist, workdir / namelist.name, output_dir)
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(portable), str(out_file)],
        cwd=SRC_DIR,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("namelist", nargs="?", type=Path)
    parser.add_argument("out_file", nargs="?", type=Path)
    parser.add_argument(
        "--update-reference",
        action="store_true",
        help=f"Regenerate {REFERENCE_FILE.relative_to(PROJECT_ROOT)} from {REFERENCE_NAMELIST.name}",
    )
    args = parser.parse_args()

    if args.update_reference:
        with tempfile.TemporaryDirectory() as tmp:
            run_in_subprocess(REFERENCE_NAMELIST, REFERENCE_FILE, Path(tmp))
        print(f"Wrote {REFERENCE_FILE}")
    elif args.namelist and args.out_file:
        record(args.namelist, args.out_file)
    else:
        parser.error("give NAMELIST and OUT_FILE, or --update-reference")


if __name__ == "__main__":
    main()
