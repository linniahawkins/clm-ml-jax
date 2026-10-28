"""
Parameter ensemble: jax.vmap of a compiled chunk over a batch of CanopyParams.

A member with the default parameters must reproduce the single run, and every member must
give finite results. (Plan Phase 4: a 32-member, one-day vcmax25 ensemble runs in ~11 s on
CPU; the members differ in the daytime as expected.) The model is initialized in a fresh
process (module-level tables).
"""

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
import offline_executable.main as M
from tests.regression.record_run import write_portable_namelist
import jax, jax.numpy as jnp, numpy as np
from clm_ml.legacy.state import state_from_instances
from multilayer_canopy.MLCanopyParamsMod import patch_setup

tmp = Path(tempfile.mkdtemp()); (tmp / "out").mkdir()
nml = write_portable_namelist(Path({namelist!r}), tmp / "nl", tmp / "out")
run = M.setup_run(M.read_namelist(str(nml)))
inputs0 = M.step_inputs(run, run.load_chunk(0, 1), 0)
st, filt = M.advance(run, run.insts, run.filt, inputs0, initialize=True)
template, state0 = st, state_from_instances(st)
n = 8
chunk = run.load_chunk(1, n)
xs = M.stack_step_inputs([M.step_inputs(run, chunk, k) for k in range(1, n + 1)])
runner = M.make_chunk_runner(run, template, filt, outputs="state")
_, params = patch_setup(0, template.mlcanopy)
factors = jnp.array([1.0, 0.8, 0.9, 1.2])
ens = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + jnp.shape(x)), params)
ens = ens._replace(vcmax25=params.vcmax25 * factors)
final, _ = jax.jit(jax.vmap(runner, in_axes=(None, None, 0)))(state0, xs, ens)
single, _ = runner(state0, xs)
diff = max(
    float(np.max(np.abs(np.asarray(a)[0] - np.asarray(b))) / (np.max(np.abs(np.asarray(b))) or 1))
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(single))
)
finite = all(bool(np.isfinite(np.asarray(a)).all()) for a in jax.tree.leaves(final))
print("RESULT" + json.dumps({{"member0_vs_single": diff, "finite": finite}}))
"""


def test_parameter_ensemble():
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
    assert result["finite"], "non-finite ensemble state"
    assert result["member0_vs_single"] <= 1e-13, result
