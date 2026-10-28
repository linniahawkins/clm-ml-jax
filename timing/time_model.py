"""
Timing test of the compiled model (clm_ml.model.Model) on CHATS7:

- forward: one compiled chunk of N steps, run twice; the second run is the steady cost
  (no compilation);
- gradient: d(mean latent heat over the chunk)/d(parameters), first call (compilation
  included) and second call.

Peak memory is the process's maximum resident set size. Appends one JSON line to
``timing/results_model.jsonl``. Used before and after Phase 5b (mask-aware layers).

    python timing/time_model.py [--steps 8] [--label "before 5b"] [--no-grad]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _peak_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, kB on Linux


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-grad", action="store_true")
    parser.add_argument("--set", action="append", default=[], help="run.yaml/model.yaml override")
    args = parser.parse_args()

    import jax

    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.model import Model
    from clm_ml.output_streams import StreamSpec
    from clm_ml.run import load_run

    tmp = Path(tempfile.mkdtemp())
    convert_namelist(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day", tmp)
    run = load_run(tmp / "run.yaml", args.set)
    t0 = time.perf_counter()
    model = Model.from_run(run)
    setup_s = time.perf_counter() - t0
    state, params = model.initial_state(), model.default_params()
    first, n = 1, args.steps
    forcing = model.forcing(first, n)
    stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

    def forward(p):
        _, out = model.run_chunk(state, forcing, first, p, streams=[stream])
        return out["s"]["latent_heat_flux"].mean()

    result = {"label": args.label, "steps": n, "overrides": args.set, "setup_s": round(setup_s, 1)}
    for name, fn in (("forward", forward), ("gradient", jax.grad(forward))):
        if name == "gradient" and args.no_grad:
            continue
        times = []
        for _ in range(2):
            t = time.perf_counter()
            jax.block_until_ready(fn(params))
            times.append(time.perf_counter() - t)
        result[f"{name}_first_s"] = round(times[0], 2)
        result[f"{name}_steady_s"] = round(times[1], 3)
        result[f"{name}_steady_per_step_ms"] = round(1000 * times[1] / n, 1)
    result.update(
        peak_rss_gb=round(_peak_gb(), 2),
        device=str(jax.devices()[0]),
        jax_version=jax.__version__,
        machine=platform.machine(),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    print(json.dumps(result, indent=1))
    with open(ROOT / "timing" / "results_model.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
