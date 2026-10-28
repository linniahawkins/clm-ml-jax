"""
CanopyParams built from params.yaml (``clm_ml.Params.canopy_params``) equal those the model
builds from the legacy PFT tables in a CHATS7 run (``MLCanopyParamsMod.patch_setup``).

The model is initialized in a fresh process (module-level tables).
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
import json, sys
sys.path.insert(0, {tests!r}); sys.path.insert(0, {root!r})
from tests.regression import process_io as P
out = {{}}
def hook(F):
    out["F"] = F
P._run_model(1, hook)
import numpy as np
from clm_ml.params import PFT_NAMES, Params
from clm_src_main.PatchType import patch
from multilayer_canopy.MLCanopyParamsMod import patch_setup
import clm_src_main.clm_instMod as I
p = 0
_, from_tables = patch_setup(p, I.mlcanopy_inst)
pft = PFT_NAMES[int(patch.itype[p])]
from_yaml = Params.from_yaml("chats").canopy_params(pft)
diffs = {{}}
for name in from_tables._fields:
    a = np.atleast_1d(np.asarray(getattr(from_tables, name)))
    b = np.atleast_1d(np.asarray(getattr(from_yaml, name)))
    if a.size > 1:  # spectral: compare wavebands 1, 2 (index 0 unused)
        a, b = a[1:3], b[1:3]
    diffs[name] = float(np.max(np.abs(a - b)))
print("RESULT" + json.dumps({{"pft": pft, "diffs": diffs}}))
"""


def test_params_yaml_equals_legacy_tables():
    script = SCRIPT.format(tests=str(HERE), root=str(HERE.parents[1]))
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])
    assert result["pft"] == "broadleaf_deciduous_temperate_tree"
    mismatched = {k: v for k, v in result["diffs"].items() if v != 0.0}
    assert not mismatched, f"params.yaml differs from the legacy tables: {mismatched}"
