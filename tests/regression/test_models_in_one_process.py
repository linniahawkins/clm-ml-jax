"""
Several models of the same tower in one Python process (plan Phase 5a, decision 20): each
model holds its own options and parameters, so a model with Medlyn stomatal conductance
created after a WUE model does not change the WUE model's results, and gives the same
results as when it is alone in its process. A model of another tower is refused (the CLM
subgrid structure is still module state until Phase 6).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r})
import yaml
from clm_ml.legacy.namelist import convert_namelist
from clm_ml.run import load_run
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux", "gpp"))

def run(model):
    first, n = 1, 4
    _, out = model.run_chunk(model.initial_state(), model.forcing(first, n), first,
                             streams=[stream])
    return {{k: [float(x) for x in v.ravel()] for k, v in out["s"].items()}}

result = {{}}
for name in {models!r}:
    overrides = ["model.physics.stomatal_conductance=" + name]
    result[name] = run(Model.from_run(load_run(tmp / "run.yaml", overrides)))
if len({models!r}) > 1:
    result["wue_again"] = run(Model.from_run(load_run(tmp / "run.yaml")))
    towers = yaml.safe_load(open({towers!r}))
    (tmp / "towers.yaml").write_text(yaml.safe_dump({{"OTHER": towers["CHATS7"]}}))
    d = yaml.safe_load((tmp / "run.yaml").read_text())
    d["domain"].update(towers=["OTHER"], towers_file="towers.yaml")
    (tmp / "other.yaml").write_text(yaml.safe_dump(d))
    try:
        Model.from_run(load_run(tmp / "other.yaml"))
        result["other_tower_refused"] = False
    except RuntimeError:
        result["other_tower_refused"] = True
print("RESULT" + json.dumps(result))
"""


def _run(models):
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        namelist=str(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day"),
        towers=str(ROOT / "src" / "clm_ml" / "data" / "towers.yaml"),
        models=list(models),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


def test_two_models_of_one_tower_in_one_process():
    together = _run(["wue", "medlyn"])
    alone = _run(["medlyn"])
    for var in ("latent_heat_flux", "gpp"):
        # the second model does not disturb the first
        np.testing.assert_array_equal(together["wue_again"][var], together["wue"][var])
        # a model gives the same results with or without another model in its process
        np.testing.assert_array_equal(together["medlyn"][var], alone["medlyn"][var])
    # the options are the models' own: the schemes differ
    assert not np.allclose(together["medlyn"]["latent_heat_flux"],
                           together["wue"]["latent_heat_flux"])  # fmt: skip
    assert together["other_tower_refused"]
