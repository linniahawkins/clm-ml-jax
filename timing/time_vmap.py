"""
Timing test of ``vmap`` over patches (plan Phase 6, step 1b): the compiled CHATS7 chunk run on
N identical copies of the patch, stacked along a new leading axis, without changing the physics.

Measures, for one N per process (so peak device memory is per N):

- forward: one compiled chunk of ``--steps`` steps over N patches, run twice (the second run is
  the steady cost); patch-steps per second;
- gradient (``--grad``): d(sum over patches of mean latent heat)/d(parameters), with the
  parameters shared by all patches (as PFT parameters are), so the gradient adds up over them;
- peak device memory (``memory_stats``; GPU only) and peak resident set size;
- check: every patch equals the unbatched run (largest relative difference).

Appends one JSON line to ``timing/results_vmap.jsonl``.

    python timing/time_vmap.py --patches 1024 [--steps 8] [--grad] [--label ...]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--patches", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--grad", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    import offline_executable.main as M
    from clm_ml import output_streams as S
    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.model import Model
    from clm_ml.output_streams import StreamSpec
    from clm_ml.run import load_run

    tmp = Path(tempfile.mkdtemp())
    convert_namelist(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day", tmp)
    model = Model.from_run(load_run(tmp / "run.yaml", []))
    state, params = model.initial_state(), model.default_params()
    first, n, npatch = 1, args.steps, args.patches
    forcing = model.forcing(first, n)

    # The chunk's pieces, as Model.run_chunk builds them
    stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))
    max_layers = model.run.config.canopy_layers.max_layers
    runner = M.make_stream_runner(
        model._legacy, model._template, model._filt, (stream,), max_layers
    )
    shapes = M.output_shapes(model._legacy, model._template, stream.variables, max_layers)
    starts = M.step_starts(model._legacy, first, n)
    rc = S.chunk_records(starts, first, model._legacy.time_of(0).dtime, stream.frequency)
    buffers = (S.init_buffers("mean", len(rc.keys), {v: shapes[v] for v in stream.variables}, jnp),)
    slots = (jnp.asarray(rc.index),)

    def one_patch(state, forcing, template, params):
        _, (b,), errors = runner(state, forcing, slots, buffers, params, template)
        lh = b["sum"]["latent_heat_flux"] / b["count"].reshape(-1, 1)
        return lh, errors

    def stack(tree):
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (npatch,) + jnp.shape(x)).copy(), tree)

    batched = jax.jit(jax.vmap(one_patch, in_axes=(0, 0, 0, None)))
    xs = (stack(state), stack(forcing), stack(model._template))
    jax.block_until_ready(xs)

    def forward(p):
        return batched(*xs, p)

    def loss(p):
        return forward(p)[0].mean(axis=0).sum()  # sum over patches of the chunk-mean flux

    result = {"label": args.label, "patches": npatch, "steps": n}
    fns = [("forward", forward)] + ([("gradient", jax.jit(jax.grad(loss)))] if args.grad else [])
    for name, fn in fns:
        times = []
        for _ in range(2):
            t = time.perf_counter()
            out = jax.block_until_ready(fn(params))
            times.append(time.perf_counter() - t)
        result[f"{name}_first_s"] = round(times[0], 2)
        result[f"{name}_steady_s"] = round(times[1], 4)
        result[f"{name}_steady_per_step_ms"] = round(1000 * times[1] / n, 2)
        result[f"{name}_patch_steps_per_s"] = round(npatch * n / times[1], 1)
        if name == "forward":
            lh, errors = out
            ref, _ = jax.jit(one_patch)(state, forcing, model._template, params)
            ref = np.asarray(ref)
            diff = np.abs(np.asarray(lh) - ref[None]) / np.maximum(np.abs(ref[None]), 1e-30)
            result["max_rel_diff_vs_single"] = float(diff.max())
            result["finite"] = bool(np.isfinite(np.asarray(lh)).all())
        else:
            leaves = jax.tree.leaves(out)
            result["gradient_finite"] = bool(all(np.isfinite(np.asarray(x)).all() for x in leaves))

    dev = jax.devices()[0]
    stats = dev.memory_stats() if hasattr(dev, "memory_stats") else None
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(
        peak_device_gb=round(stats["peak_bytes_in_use"] / 1e9, 3) if stats else None,
        peak_rss_gb=round(peak_rss / 1e9 if sys.platform == "darwin" else peak_rss / 1e6, 2),
        device=str(dev),
        jax_version=jax.__version__,
        machine=platform.machine(),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    print(json.dumps(result, indent=1))
    with open(ROOT / "timing" / "results_vmap.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
