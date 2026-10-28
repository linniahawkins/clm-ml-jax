"""Output streams accumulated inside a compiled chunk agree with the variables computed
afterwards (numpy) from each step's containers, and a daily-mean stream equals the mean of
the per-step values. The model is initialized in a fresh process (module-level tables)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

HERE = Path(__file__).resolve().parent

SCRIPT = """
import json, sys, tempfile, datetime as dt
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import offline_executable.main as M
from tests.regression.record_run import write_portable_namelist
import jax, jax.numpy as jnp, numpy as np
from clm_ml.legacy.state import state_from_instances
from clm_ml import output_streams as S
from clm_ml.outputs import VARIABLES, _Raw

tmp = Path(tempfile.mkdtemp()); (tmp / "out").mkdir()
nml = write_portable_namelist(Path({namelist!r}), tmp / "nl", tmp / "out")
run = M.setup_run(M.read_namelist(str(nml)))
st, filt = M.advance(run, run.insts, run.filt, M.step_inputs(run, run.load_chunk(0, 1), 0),
                     initialize=True)
template, state0 = st, state_from_instances(st)
first, n = 1, 49
chunk = run.load_chunk(first, n)
xs = M.stack_step_inputs([M.step_inputs(run, chunk, k) for k in range(first, first + n)])
names = tuple(v.name for v in VARIABLES) + ("soil_temperature",)
_, (_, mlc, _) = M.make_chunk_runner(run, template, filt)(state0, xs)
r = _Raw({{f"mlcanopy.{{f}}": np.asarray(getattr(mlc, f)) for f in mlc._fields}}, 50)
ref = {{v.name: np.asarray(v.compute(r), float)[:, 0] for v in VARIABLES}}

def starts(k0, nn):
    out = []
    for k in range(k0, k0 + nn):
        t = run.time_of(k)
        out.append(dt.datetime(t.year, t.month, t.day) + dt.timedelta(seconds=t.tod - t.dtime))
    return out

specs = (S.StreamSpec("step", "step", "instantaneous", names),
         S.StreamSpec("daily", "daily", "mean", ("latent_heat_flux",)))
recs = [S.chunk_records(starts(first, n), first, 1800, s.frequency) for s in specs]
shapes = M.output_shapes(run, template, names, 50)
buffers = tuple(S.init_buffers(s.statistic, len(rc.keys), {{v: shapes[v] for v in s.variables}},
                               jnp) for s, rc in zip(specs, recs))
runner = M.make_stream_runner(run, template, filt, specs, 50)
_, bufs, _ = runner(state0, xs, tuple(jnp.asarray(rc.index) for rc in recs), buffers)
step_vals = S.finish("instantaneous", bufs[0])
worst, nan_mismatch = 0.0, []
for name, a in ref.items():
    b = np.asarray(step_vals[name])[:, 0]
    v = np.isfinite(a)
    if not np.array_equal(v, np.isfinite(b)):
        nan_mismatch.append(name)
    elif v.any():
        worst = max(worst, float(np.max(np.abs(a[v] - b[v])) / (np.max(np.abs(a[v])) or 1)))
lh = np.asarray(step_vals["latent_heat_flux"])[:, 0]
daily = float(np.asarray(S.finish("mean", bufs[1])["latent_heat_flux"])[0, 0])
print("RESULT" + json.dumps({{"worst": worst, "nan_mismatch": nan_mismatch,
                             "daily": daily, "daily_numpy": float(lh[:47].mean()),
                             "counts": np.asarray(bufs[1]["count"]).tolist()}}))
"""


def test_streams_match_post_hoc_outputs():
    root = HERE.parents[1]
    script = SCRIPT.format(
        src=str(root / "src"),
        root=str(root),
        namelist=str(root / "src" / "offline_executable" / "nl.CHATS7.50steps"),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])
    assert not result["nan_mismatch"], result["nan_mismatch"]
    # Different compiled programs: round-off only (noise twins differ by ~5e-11).
    assert result["worst"] < 1e-10, result["worst"]
    assert result["counts"] == [47.0, 2.0]
    assert abs(result["daily"] - result["daily_numpy"]) <= 1e-12 * abs(result["daily_numpy"])
