"""Make the golden data of CLM's soil water routines: write the cases, run CLM's Fortran on
them in the test harness (``build.py``), and write ``golden_IO/*.json``.

    python tests/fortran_validation/ctsm_harness/generate_golden.py [routine ...]
        [--work-dir DIR] [--ctsm DIR]

Without routine names, every routine of ``ROUTINES`` is regenerated. The cases are idealized
sweeps: random draws with a fixed seed over the ranges a routine can meet, plus hand-picked
edge values. Regenerating gives the same cases and, from the same CTSM tag and compiler, the
same values.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
from build import CTSM_TAG, DEFAULT_CTSM, DEFAULT_WORK, build

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden_IO"
SEED = 20261124
N_RANDOM = 300


def _log_uniform(rng, low: float, high: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), n))


def _saturation(rng, n: int) -> np.ndarray:
    """Relative saturation over CLM's bounds [0.01, 1]: half of the draws uniform, half
    uniform in the logarithm (the dry end, where the curve is steep)."""
    s = np.where(rng.random(n) < 0.5, rng.uniform(0.01, 1.0, n), _log_uniform(rng, 0.01, 1.0, n))
    return np.clip(s, 0.01, 1.0)


def soil_hk_cases() -> list[dict]:
    """Hydraulic conductivity: saturation from the dry bound to saturated; ice impedance 1
    (unfrozen) or ``10**(-e_ice*icefrac)`` with ``e_ice = 6``; saturated conductivity from
    clay to sand (mm/s); Clapp-Hornberger ``b`` from organic soil to heavy clay."""
    rng = np.random.default_rng(SEED)
    n = N_RANDOM
    s = _saturation(rng, n)
    imped = np.where(rng.random(n) < 0.5, 1.0, 10.0 ** (-6.0 * rng.uniform(0.0, 1.0, n)))
    hksat = _log_uniform(rng, 1.0e-5, 1.0e-1, n)
    bsw = rng.uniform(2.5, 25.0, n)
    cases = [
        {"s": float(a), "imped": float(b), "hksat": float(c), "bsw": float(d)}
        for a, b, c, d in zip(s, imped, hksat, bsw)
    ]
    for s_edge in (0.01, 1.0):
        for bsw_edge in (2.5, 6.9, 25.0):
            for imped_edge in (1.0, 1.0e-6):
                cases.append({"s": s_edge, "imped": imped_edge, "hksat": 0.0035, "bsw": bsw_edge})
    return cases


def soil_suction_cases() -> list[dict]:
    """Matric potential: saturation from the dry bound to saturated; saturated suction (mm)
    and Clapp-Hornberger ``b`` over CLM's soils."""
    rng = np.random.default_rng(SEED + 1)
    n = N_RANDOM
    s = _saturation(rng, n)
    sucsat = _log_uniform(rng, 10.0, 1000.0, n)
    bsw = rng.uniform(2.5, 25.0, n)
    cases = [
        {"s": float(a), "sucsat": float(b), "bsw": float(c)} for a, b, c in zip(s, sucsat, bsw)
    ]
    for s_edge in (0.01, 1.0):
        for bsw_edge in (2.5, 6.9, 25.0):
            for sucsat_edge in (10.0, 230.0, 1000.0):
                cases.append({"s": s_edge, "sucsat": sucsat_edge, "bsw": bsw_edge})
    return cases


class Routine(NamedTuple):
    fortran: str  # Module.F90::Subroutine
    cases: Callable[[], list[dict]]


# The harness's name of each routine (its case file, output file and driver).
ROUTINES = {
    "soil_hk": Routine("SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_hk", soil_hk_cases),
    "soil_suction": Routine(
        "SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_suction", soil_suction_cases
    ),
}


def golden_file(name: str) -> Path:
    module, routine = ROUTINES[name].fortran.split(".F90::")
    return GOLDEN_DIR / f"test_{module}_{routine}.json"


def _namelist_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_namelist_value(v) for v in value)
    # repr of a float has the 17 digits that identify the number; Fortran wants a "d" exponent
    return repr(value).replace("e", "d") if isinstance(value, float) else repr(value)


def write_cases(path: Path, cases: list[dict]) -> None:
    """One namelist group ``&case ... /`` per case."""
    with open(path, "w") as f:
        for case in cases:
            f.write("&case\n")
            for key, value in case.items():
                f.write(f" {key} = {_namelist_value(value)}\n")
            f.write("/\n")


def read_outputs(path: Path) -> list[dict]:
    """The outputs of every case: ``name=value`` scalars, and ``name(j)=value`` lines
    gathered into lists in the order of ``j`` (CLM's 1-based level index)."""
    cases: list[dict] = []
    for line in path.read_text().splitlines():
        key, text = (part.strip() for part in line.split("="))
        if key == "case":
            assert int(text) == len(cases) + 1, f"{path}: case {text} out of order"
            cases.append({})
            continue
        value = float(text) if any(ch in text for ch in ".EeNn") else int(text)
        if key.endswith(")"):
            name, index = key[:-1].split("(")
            array = cases[-1].setdefault(name, [])
            assert int(index) == len(array) + 1, f"{path}: {key} out of order"
            array.append(value)
        else:
            cases[-1][key] = value
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("routines", nargs="*", default=list(ROUTINES), help="default: all")
    parser.add_argument("--ctsm", type=Path, default=DEFAULT_CTSM, help="the CTSM checkout")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="scratch directory")
    args = parser.parse_args()
    work = args.work_dir.resolve()

    # Every routine's cases are written (a driver without a case file is skipped), so that a
    # run always exercises all the drivers; only the named routines' JSON files are rewritten.
    (work / "cases").mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    cases = {name: routine.cases() for name, routine in ROUTINES.items()}
    for name in ROUTINES:
        write_cases(work / "cases" / f"{name}.nml", cases[name])
        (work / "out" / f"{name}.out").unlink(missing_ok=True)

    build(args.ctsm.resolve(), work, only_harness=True)

    for name in args.routines:
        outputs = read_outputs(work / "out" / f"{name}.out")
        if len(outputs) != len(cases[name]):
            raise SystemExit(f"{name}: {len(cases[name])} cases, {len(outputs)} outputs")
        data = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "executable": f"ctsm_harness:{name}",
            "routine": ROUTINES[name].fortran,
            "ctsm_tag": CTSM_TAG,
            "cases": [{"inputs": i, "outputs": o} for i, o in zip(cases[name], outputs)],
        }
        with open(golden_file(name), "w") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        print(f"{golden_file(name).name}: {len(outputs)} cases")


if __name__ == "__main__":
    main()
