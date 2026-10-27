"""
Timing test: wall-clock run time of the offline driver.

Runs ``offline_executable.main`` on a namelist in a fresh process and reports:

- startup: process start until the first time step begins (imports, initialization,
  input file reads)
- first step: wall time of time step 1 (includes most JIT compilation)
- steady step: median wall time of steps 2..N
- seconds per simulated day, from the steady step time

Results are printed and appended to ``timing/results.jsonl`` so phases of the
modernization plan can be compared against this baseline.

Usage::

    python timing/time_offline.py                      # CHATS7, 50 steps
    python timing/time_offline.py --namelist src/offline_executable/nl.CHATS7.1day
    python timing/time_offline.py --cold-cache         # ignore persistent JIT cache
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.regression.record_run import SRC_DIR, write_portable_namelist  # noqa: E402

DEFAULT_NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.50steps"
RESULTS_FILE = Path(__file__).resolve().parent / "results.jsonl"

STEP_LINE = re.compile(r"timestep (\d+)/(\d+) \| step=([\d.]+)s")


def run(namelist: Path, cold_cache: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "out").mkdir()
        portable = write_portable_namelist(namelist, tmp / namelist.name, tmp / "out")
        env = dict(os.environ)
        if cold_cache:
            env["JAX_COMPILATION_CACHE_DIR"] = str(tmp / "jax_cache")

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "offline_executable.main", str(portable)],
            cwd=SRC_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        step_times: dict[int, float] = {}
        first_step_end = None
        n_steps = None
        log = []
        for line in proc.stdout:
            log.append(line)
            match = STEP_LINE.search(line)
            if match:
                step, n_steps, dt = int(match[1]), int(match[2]), float(match[3])
                step_times[step] = dt
                if step == 1:
                    first_step_end = time.perf_counter() - t0
        proc.wait()
        total = time.perf_counter() - t0
        if proc.returncode != 0:
            sys.stdout.writelines(log[-30:])
            raise SystemExit(f"model run failed with exit code {proc.returncode}")

    import jax

    steady = [dt for step, dt in sorted(step_times.items()) if step > 1]
    steady_step = statistics.median(steady)
    step_seconds = 1800  # CHATS7 forcing step; reported per simulated day below
    return {
        "namelist": namelist.name,
        "n_steps": n_steps,
        "cold_cache": cold_cache,
        "startup_s": round(first_step_end - step_times[1], 2),
        "first_step_s": round(step_times[1], 2),
        "steady_step_median_s": round(steady_step, 3),
        "steady_step_min_s": round(min(steady), 3),
        "steady_step_max_s": round(max(steady), 3),
        "total_s": round(total, 1),
        "s_per_simulated_day": round(steady_step * 86400 / step_seconds, 1),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "date": time.strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--namelist", type=Path, default=DEFAULT_NAMELIST)
    parser.add_argument("--cold-cache", action="store_true", help="use an empty JAX cache")
    parser.add_argument("--no-save", action="store_true", help="do not append to results.jsonl")
    args = parser.parse_args()

    result = run(args.namelist.resolve(), args.cold_cache)
    print(json.dumps(result, indent=2))
    if not args.no_save:
        with RESULTS_FILE.open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        print(f"Appended to {RESULTS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
