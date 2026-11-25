"""Build and run the CTSM test harness: CLM's own soil water Fortran, called for one idealized
soil column inside CTSM's unit-test framework.

The CTSM checkout is never changed. The script makes a copy of its ``src/`` in a work
directory on scratch, adds the harness's files, makes the changes listed below, checks that
nothing else differs from the checkout, and builds and runs CTSM's unit tests (with the
harness's) through CIME's ``run_tests.py`` in a batch job (``qcmd``).

    python tests/fortran_validation/ctsm_harness/build.py [--work-dir DIR] [--ctsm DIR]

The three kinds of change to the copy of ``src/``:

- source files added to the test build's lists (``SOURCE_LISTS``): CTSM's unit-test build
  compiles only the files its own tests use;
- routines and variables made visible to the drivers (``VISIBILITY``): ``private`` becomes
  ``public`` on the listed lines, no other character changes;
- the harness's own files (``fortran/``): a stand-in for ``controlMod`` and the test directory.
"""

from __future__ import annotations

import argparse
import difflib
import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORTRAN = HERE / "fortran"

CTSM_TAG = "ctsm5.4.004"
DEFAULT_CTSM = Path(os.environ.get("CTSM_ROOT", "/glade/work/linnia/ctsm5.4.004"))
DEFAULT_WORK = Path(os.environ.get("SCRATCH", "/glade/derecho/scratch/linnia")) / "ctsm_harness"
PBS_ACCOUNT = "P93300041"

# Lines appended to a CMake list of CTSM's test build: {file: (the line after which to add,
# the lines added)}. Paths are relative to src/.
SOURCE_LISTS = {
    "biogeophys/CMakeLists.txt": (
        "  SoilWaterRetentionCurveMod.F90",
        [
            "  SoilWaterRetentionCurveClappHornberg1978Mod.F90",
            "  SoilWaterMovementMod.F90",
            "  SoilWaterPlantSinkMod.F90",
            "  SurfaceWaterMod.F90",
            "  TridiagonalMod.F90",
        ],
    ),
    "biogeophys/test/CMakeLists.txt": (
        "add_subdirectory(WaterType_test)",
        ["add_subdirectory(ctsm_harness_test)"],
    ),
    "unit_test_stubs/main/CMakeLists.txt": (
        "  GetGlobalValuesMod_stub.F90",
        ["  controlMod_stub.F90"],
    ),
}

# Lines on which "private" becomes "public": {file: [the line as in the checkout]}.
VISIBILITY = {
    "biogeophys/SoilWaterMovementMod.F90": [
        "  private :: soilwater_moisture_form",
        "  private :: compute_hydraulic_properties",
        "  private :: compute_moisture_fluxes_and_derivs",
        "  private :: compute_RHS_moisture_form",
        "  private :: compute_LHS_moisture_form",
        "  private :: IceImpedance",
        "  type, private :: params_type",
        "  type(params_type), private ::  params_inst",
    ],
}

# The harness's files: {path under fortran/: path under src/}.
ADDED = {
    "controlMod_stub.F90": "unit_test_stubs/main/controlMod_stub.F90",
    "ctsm_harness_test": "biogeophys/test/ctsm_harness_test",
}
PATHS_MODULE = "biogeophys/test/ctsm_harness_test/HarnessPathsMod.F90"
# The harness's directory in the build
HARNESS_BUILD = "__command_line_test__/__command_line_test__/clm_biogeophys_test/ctsm_harness_test"


def check_checkout(ctsm: Path) -> None:
    """Stop unless ``ctsm`` is an unmodified checkout of the reference tag."""
    tag = subprocess.run(
        ["git", "-C", str(ctsm), "describe", "--tags"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if tag != CTSM_TAG:
        sys.exit(f"{ctsm} is at '{tag}', not {CTSM_TAG}")
    changed = subprocess.run(
        ["git", "-C", str(ctsm), "status", "--short", "src"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if changed:
        sys.exit(f"{ctsm}/src differs from {CTSM_TAG}:\n{changed}")


def _edit_lines(path: Path, edit) -> None:
    lines = path.read_text().split("\n")
    path.write_text("\n".join(edit(lines)))


def _index_of(lines: list[str], line: str, path: Path) -> int:
    if lines.count(line) != 1:
        sys.exit(f"{path}: expected exactly one line '{line}', found {lines.count(line)}")
    return lines.index(line)


def prepare_sources(ctsm: Path, work: Path) -> Path:
    """Make ``work/ctsm`` (links to the checkout, and a changed copy of ``src/``); return the
    copy of ``src/``."""
    tree = work / "ctsm"
    tree.mkdir(parents=True, exist_ok=True)
    for entry in ctsm.iterdir():
        link = tree / entry.name
        if entry.name in ("src", ".git") or link.is_symlink():
            continue
        link.symlink_to(entry)
    src = tree / "src"
    if src.exists():
        shutil.rmtree(src)
    # copy2 keeps the files' times, so an unchanged file is not compiled again
    shutil.copytree(ctsm / "src", src, copy_function=shutil.copy2)

    for rel, (anchor, added) in SOURCE_LISTS.items():

        def add(lines, anchor=anchor, added=added, rel=rel):
            i = _index_of(lines, anchor, src / rel)
            return lines[: i + 1] + added + lines[i + 1 :]

        _edit_lines(src / rel, add)

    for rel, targets in VISIBILITY.items():

        def publish(lines, targets=targets, rel=rel):
            for target in targets:
                i = _index_of(lines, target, src / rel)
                lines[i] = target.replace("private", "public")
            return lines

        _edit_lines(src / rel, publish)

    for name, rel in ADDED.items():
        source, target = FORTRAN / name, src / rel
        if source.is_dir():
            shutil.copytree(source, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(source, target)

    for sub in ("cases", "out"):
        (work / sub).mkdir(exist_ok=True)
    (src / PATHS_MODULE).write_text(
        "module HarnessPathsMod\n"
        "  ! Written by build.py: the harness's work directory (cases/ and out/).\n"
        "  implicit none\n"
        "  public\n"
        f"  character(len=*), parameter :: harness_dir = '{work}'\n"
        "end module HarnessPathsMod\n"
    )
    return src


def verify_sources(ctsm: Path, src: Path) -> None:
    """Stop unless the copy differs from the checkout's ``src/`` only by the harness's own
    files, by lines added to the listed CMake files, and by ``private`` -> ``public``.

    The comparison is made on the files themselves, not on what ``prepare_sources`` meant to
    do, so it also catches a mistake in that function.
    """
    problems: list[str] = []
    harness_files = {PATHS_MODULE}
    for name, rel in ADDED.items():
        source = FORTRAN / name
        if source.is_dir():
            harness_files |= {f"{rel}/{p.name}" for p in source.iterdir()}
        else:
            harness_files.add(rel)

    def walk(cmp: filecmp.dircmp, rel: str) -> None:
        for name in cmp.left_only:
            problems.append(f"missing from the copy: {rel}{name}")
        for name in cmp.right_only:
            path = src / rel / name
            new = [str(p.relative_to(src)) for p in path.rglob("*")] if path.is_dir() else []
            for added in new or [f"{rel}{name}"]:
                if added not in harness_files and not (src / added).is_dir():
                    problems.append(f"unexpected file in the copy: {added}")
        for name in cmp.common_files:
            file = f"{rel}{name}"
            if filecmp.cmp(ctsm / "src" / file, src / file, shallow=False):
                continue
            old = (ctsm / "src" / file).read_text().split("\n")
            new = (src / file).read_text().split("\n")
            if file in SOURCE_LISTS:
                # only whole lines added
                for line in difflib.ndiff(old, new):
                    if line.startswith(("- ", "? ")):
                        problems.append(f"{file}: a line of the checkout changed: {line}")
            elif len(old) != len(new):
                problems.append(f"{file}: number of lines changed")
            else:
                for i, (a, b) in enumerate(zip(old, new), start=1):
                    if a != b and a.replace("private", "public") != b:
                        problems.append(f"{file}:{i}: more than visibility changed:\n  {a}\n  {b}")
        for name, sub in cmp.subdirs.items():
            walk(sub, f"{rel}{name}/")

    walk(filecmp.dircmp(ctsm / "src", src, ignore=[]), "")
    if problems:
        sys.exit("The harness's copy of src/ is not the checkout's:\n" + "\n".join(problems))


def run_tests(work: Path, src: Path, only_harness: bool = False) -> None:
    """Build CTSM's unit tests with the harness in a batch job and run them (all of them, or
    the harness's alone)."""
    command = [
        "qcmd", "-A", PBS_ACCOUNT, "-q", "develop", "-l", "walltime=01:00:00",
        "-l", "select=1:ncpus=8:mem=20GB", "--",
        "../cime/scripts/fortran_unit_testing/run_tests.py",
        "--build-dir", str(work / "build"), "--make-j", "8",
    ]  # fmt: skip
    if only_harness:
        command += ["--ctest-args=-R ctsm_harness"]
    # pFUnit's main program includes the list of test modules, a dependency make does not
    # see: without this, a test file added since the last build is compiled but not run.
    main_program = work / "build" / HARNESS_BUILD / "ctsm_harness_driver.F90"
    if main_program.exists():
        main_program.touch()
    log = work / "last_run.log"
    with open(log, "w") as out:
        result = subprocess.run(command, cwd=src, stdout=out, stderr=subprocess.STDOUT)
    text = log.read_text()
    if result.returncode != 0 or "100% tests passed" not in text:
        sys.exit(f"The build or a test failed: see {log}\n" + "\n".join(text.split("\n")[-25:]))
    print([line for line in text.split("\n") if "tests passed" in line][-1].strip())


def build(ctsm: Path = DEFAULT_CTSM, work: Path = DEFAULT_WORK, only_harness: bool = False):
    """Prepare the sources, check them, build and run. Returns the work directory."""
    check_checkout(ctsm)
    src = prepare_sources(ctsm, work)
    verify_sources(ctsm, src)
    print(f"Sources in {src}: only the harness's changes differ from {CTSM_TAG}.")
    run_tests(work, src, only_harness)
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ctsm", type=Path, default=DEFAULT_CTSM, help="the CTSM checkout")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="scratch directory")
    parser.add_argument(
        "--only-harness", action="store_true", help="run the harness's tests, not CTSM's too"
    )
    args = parser.parse_args()
    build(args.ctsm.resolve(), args.work_dir.resolve(), args.only_harness)


if __name__ == "__main__":
    main()
