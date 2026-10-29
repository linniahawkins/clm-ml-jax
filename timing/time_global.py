"""
Timing test of a synthetic global-size domain (plan Phase 6, step 6): many gridcells with
several patches each, set up with the vectorized setup (``clm_ml.legacy.vector_setup``) and
run as one compiled chunk spread over the devices.

The domain is built from CHATS7's forcing (every gridcell reads the same files, so there is
one forcing prototype per PFT) with every per-patch input varied at random: latitude and
longitude, sand, clay, organic matter, soil colour, bedrock depth, PFT (three C3 and one C4),
canopy height, root biomass, explicit layer counts (5-15 within, 5-15 above) and patch
weights. Records setup time, first-step time, forcing read time, chunk compile and steady
times, patch-steps per second and peak device memory in ``timing/results_global.jsonl``.

    python timing/time_global.py [--patches 262144] [--per-gridcell 4] [--devices 4]
                                 [--steps 48] [--label ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PFTS = (  # (name, C3, canopy height range m, root biomass g m-2)
    ("broadleaf_deciduous_temperate_tree", True, (8.0, 20.0), 500.0),
    ("needleleaf_evergreen_temperate_tree", True, (8.0, 20.0), 800.0),
    ("broadleaf_deciduous_boreal_tree", True, (5.0, 15.0), 500.0),
    ("c4_grass", False, (0.3, 1.0), 400.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--patches", type=int, default=262144)
    parser.add_argument("--per-gridcell", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import yaml

    import offline_executable.main as M
    from clm_ml import output_streams as S
    from clm_ml.legacy import vector_setup, worker
    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.params import PFT_NAMES
    from clm_ml.run import load_run

    n_p, k = args.patches, args.per_gridcell
    n_g = n_p // k
    rng = np.random.default_rng(0)
    result = {"label": args.label, "patches": n_p, "gridcells": n_g, "devices": args.devices}

    # prototypes: one tower per PFT, CHATS7 forcing (one forcing key)
    tmp = Path(tempfile.mkdtemp())
    convert_namelist(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day", tmp)
    chats = yaml.safe_load((ROOT / "src" / "clm_ml" / "data" / "towers.yaml").read_text())["CHATS7"]
    towers = {}
    for i, (pft, _, hr, root) in enumerate(PFTS):
        t = copy.deepcopy(chats)
        t["patches"] = [{"pft": pft, "weight": 1.0, "canopy_height": float(np.mean(hr)),
                         "root_biomass": root, "n_layers_within": 10, "n_layers_above": 10}]  # fmt: skip
        towers[f"PROTO{i}"] = t
    (tmp / "towers.yaml").write_text(yaml.safe_dump(towers))
    d = yaml.safe_load((tmp / "run.yaml").read_text())
    d["domain"].update(towers=list(towers), towers_file="towers.yaml")
    (tmp / "run.yaml").write_text(yaml.safe_dump(d))
    spec = load_run(tmp / "run.yaml").legacy_spec(None)
    proto_keys = [vector_setup.forcing_key(spec, ps) + (int(ps["tower"]["tower_pft"]),)
                  for ps in spec["patches"]]  # fmt: skip
    proto_specs = {key: {**spec, **ps} for key, ps in zip(proto_keys, spec["patches"])}

    # the synthetic domain: every per-patch input at random
    kind = rng.integers(0, len(PFTS), size=n_p)
    kind[::k] = 0  # every gridcell has a walnut-like C3 tree
    gidx = np.repeat(np.arange(n_g), k)
    w = rng.random(n_p) + 0.1
    w = w / np.bincount(gidx, weights=w)[gidx]
    lat = rng.uniform(-60.0, 75.0, n_g)[gidx]
    lon = rng.uniform(-180.0, 180.0, n_g)[gidx]
    sand = rng.uniform(5.0, 80.0, n_g)
    clay = rng.uniform(5.0, 95.0 - sand)
    tower = {
        "tower_lat": lat,
        "tower_lon": lon,
        "tower_tex": [""] * n_p,
        "tower_sand": sand[gidx],
        "tower_clay": clay[gidx],
        "tower_organic": rng.uniform(0.0, 100.0, n_g)[gidx],
        "tower_isoicol": rng.integers(1, 21, n_g)[gidx],
        "tower_zbed": rng.uniform(0.5, 10.0, n_g)[gidx],
        "tower_pft": np.array([PFT_NAMES.index(PFTS[j][0]) for j in kind]),
        "tower_canht": np.array([rng.uniform(*PFTS[j][2]) for j in kind]),
        "tower_root": np.array([PFTS[j][3] for j in kind]),
        "tower_pbeta_lai": np.full((n_p, 2), -999.0),
        "tower_pbeta_sai": np.full((n_p, 2), -999.0),
    }
    within = rng.integers(5, 16, n_p)
    above = rng.integers(5, 16, n_p)
    keys = [proto_keys[j] for j in kind]
    names = [f"g{g}#{i % k}" for i, g in enumerate(gidx)]

    t0 = time.perf_counter()
    batch = vector_setup.setup_arrays(
        proto_specs, keys, tower, within, above, gidx, w, n_g, names, host=True
    )
    batch = worker.laid_out(batch, {"devices": args.devices})
    result["setup_s"] = round(time.perf_counter() - t0, 1)
    print("setup", result["setup_s"], "s", flush=True)

    names_out = ("latent_heat_flux", "gpp")
    max_layers = 50
    t0 = time.perf_counter()
    template, _ = M.batched_first_step(batch, names_out, max_layers)
    jax.block_until_ready(template)
    result["first_step_s"] = round(time.perf_counter() - t0, 1)
    print("first step", result["first_step_s"], "s", flush=True)
    from clm_ml.legacy.state import state_from_instances

    state = state_from_instances(template)

    first, n = 1, args.steps
    t0 = time.perf_counter()
    xg, xp = M.read_batch_inputs(batch, first, n)
    jax.block_until_ready((xg, xp))
    result["read_s"] = round(time.perf_counter() - t0, 1)

    stream = S.StreamSpec("h", "hourly", "mean", names_out)
    gvars = [M.gridcell_variables(stream.variables, ("patch", "gridcell"))]
    runner = M.make_batch_runner(batch, template, batch.init_filt, [stream], max_layers, gvars)
    starts = M.step_starts(batch.run, first, n)
    rc = S.chunk_records(starts, first, batch.run.time_of(0).dtime, "hourly")
    shapes = {v: (batch.n_patches,) for v in names_out}
    shapes.update({f"gridcell:{v}": (batch.n_gridcells,) for v in gvars[0]})
    params = batch.params()
    weight = jnp.asarray(batch.weight)
    alive = jnp.ones(batch.n_patches, dtype=bool)
    gi = jnp.asarray(batch.runner_gidx)
    times = []
    for _ in range(2):
        buffers = (S.init_buffers("mean", len(rc.keys), shapes, jnp),)
        t0 = time.perf_counter()
        state2, out, errors = runner(
            state, xg, xp, (jnp.asarray(rc.index),), buffers, params, weight, alive, gi
        )
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    lh = np.asarray(M.real_part(out[0], batch)["sum"]["latent_heat_flux"])
    result.update(
        chunk_first_s=round(times[0], 1),
        chunk_steady_s=round(times[1], 2),
        steady_per_step_s=round(times[1] / n, 3),
        patch_steps_per_s=round(n_p * n / times[1], 1),
        finite_fraction=float(np.isfinite(lh).mean()),
        steps=n,
    )
    stats = [dev.memory_stats() for dev in jax.devices()[: args.devices]]
    peak = [s["peak_bytes_in_use"] / 1e9 for s in stats if s]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    result.update(
        peak_device_gb=[round(x, 2) for x in peak] or None,
        peak_rss_gb=round(peak_rss, 1),
        device=str(jax.devices()[0]),
        jax_version=jax.__version__,
        machine=platform.machine(),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    print(json.dumps(result, indent=1))
    with open(ROOT / "timing" / "results_global.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
