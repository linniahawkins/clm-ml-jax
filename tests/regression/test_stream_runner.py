"""Output streams accumulated inside a compiled chunk agree with the variables computed
afterwards (numpy) from each step's StepState, and a daily-mean stream equals the mean of
the per-step values."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

HERE = Path(__file__).resolve().parent

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from clm_ml.io.namelist import convert_namelist
from clm_ml.run import load_run
from clm_ml.model import Model
from clm_ml import driver as D, step as S
from clm_ml import output_streams as OS
from clm_ml.outputs import step_fields
from clm_ml.outputs import VARIABLES, _Raw

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
model = Model.from_run(load_run(tmp / "run.yaml"))
sim = model._sim
first, n = 1, 49
state, forcing = model.initial_state(), model.forcing(first, n)

# each step's StepState (patch 0), then the variables computed afterwards with numpy
one = lambda t: jax.tree.map(lambda a: jnp.asarray(a)[0], t)
xg, xp = forcing
xs = S.step_inputs(jax.tree.map(lambda a: a[:, 0], xg), jax.tree.map(lambda a: a[:, 0], xp))
constants, params = one(jax.tree.map(jnp.asarray, sim.constants)), one(sim.patch_params())

def body(s, x):
    s = D.advance_acclimation(sim, s, x.tref)
    r = S.step(s, x, constants, params, sim.tables, options=sim.options, c3=bool(sim.c3[0]))
    return r.state, r

_, steps = jax.jit(lambda s, xs: jax.lax.scan(body, s, xs))(one(state), xs)
fields = jax.vmap(lambda st: step_fields(st, constants.layers))(steps)  # (step, ...)
r = _Raw({{k: np.asarray(v)[:, None] for k, v in fields.items()}}, 50)  # (step, patch, ...)
ref = {{v.name: np.asarray(v.compute(r), float)[:, 0] for v in VARIABLES}}

names = tuple(v.name for v in VARIABLES) + ("soil_temperature",)
specs = (OS.StreamSpec("step", "step", "instantaneous", names),
         OS.StreamSpec("daily", "daily", "mean", ("latent_heat_flux",)))
_, out = model.run_chunk(state, forcing, first, streams=specs)
worst, nan_mismatch = 0.0, []
for name, a in ref.items():
    b = np.asarray(out["step"][name])[:, 0]
    v = np.isfinite(a)
    if not np.array_equal(v, np.isfinite(b)):
        nan_mismatch.append(name)
    elif v.any():
        worst = max(worst, float(np.max(np.abs(a[v] - b[v])) / (np.max(np.abs(a[v])) or 1)))
lh = np.asarray(out["step"]["latent_heat_flux"])[:, 0]
daily = np.asarray(out["daily"]["latent_heat_flux"])[:, 0]
print("RESULT" + json.dumps({{"worst": worst, "nan_mismatch": nan_mismatch,
                             "daily": float(daily[0]), "daily_numpy": float(lh[:47].mean()),
                             "n_daily": int(daily.shape[0])}}))
"""


def test_streams_match_post_hoc_outputs():
    root = HERE.parents[1]
    script = SCRIPT.format(
        src=str(root / "src"),
        root=str(root),
        namelist=str(root / "data" / "namelists" / "nl.CHATS7.50steps"),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])
    assert not result["nan_mismatch"], result["nan_mismatch"]
    # Different compiled programs: round-off only (noise twins differ by ~5e-11).
    assert result["worst"] < 1e-10, result["worst"]
    assert result["n_daily"] == 2  # 47 steps on 1 May, 2 on 2 May
    assert abs(result["daily"] - result["daily_numpy"]) <= 1e-12 * abs(result["daily_numpy"])
