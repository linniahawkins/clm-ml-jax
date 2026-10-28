"""clm_ml.model.Model: a compiled chunk in this process, differentiable with respect to the
parameters (checked against a finite difference). Slow: compiles a gradient."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import yaml
import jax
from clm_ml.legacy.namelist import convert_namelist
from clm_ml.run import load_run
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
model = Model.from_run(load_run(tmp / "run.yaml"))
state, params = model.initial_state(), model.default_params()
first, n = 1, 8
forcing = model.forcing(first, n)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

def loss(p):
    _, out = model.run_chunk(state, forcing, first, p, streams=[stream])
    return out["s"]["latent_heat_flux"].mean()

g = float(jax.grad(loss)(params).vcmax25)
eps = 1e-3 * float(params.vcmax25)
fd = (float(loss(params._replace(vcmax25=params.vcmax25 + eps)))
      - float(loss(params._replace(vcmax25=params.vcmax25 - eps)))) / (2 * eps)

refused = False
other = tmp / "other"; other.mkdir()
convert_namelist(Path({namelist!r}), other)
d = yaml.safe_load((other / "run.yaml").read_text()); d["time"]["steps"] = 10
(other / "run.yaml").write_text(yaml.safe_dump(d))
try:
    Model.from_run(load_run(other / "run.yaml"))
except RuntimeError:
    refused = True
print("RESULT" + json.dumps({{"grad": g, "fd": fd, "refused": refused}}))
"""


@pytest.mark.slow
def test_model_chunk_gradient_matches_finite_difference():
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        root=str(ROOT),
        namelist=str(ROOT / "src" / "offline_executable" / "nl.CHATS7.50steps"),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])
    assert result["refused"], "a second, different run in the same process must be refused"
    assert result["grad"] != 0.0
    assert abs(result["grad"] - result["fd"]) <= 0.05 * abs(result["fd"]), result
