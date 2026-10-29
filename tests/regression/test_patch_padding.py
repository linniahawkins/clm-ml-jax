"""
Phase 6 step 3 test: padding the patch and gridcell counts (``execution.patch_count``,
``gridcell_count``) lets domains of different sizes share one compiled program.

Domain A has two gridcells and three patches; domain B one gridcell and one patch. Both are
padded to 4 patches and 3 gridcells (padding: weight-0 copies on a padding gridcell, not in
the output). In one process:

- B's chunk run through A's compiled program equals B's own run, and A's program does not
  compile again (the patch-gridcell map, weights and all per-patch data are inputs);
- the padded B equals B without padding (only its own patch and gridcell in the output).
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
import json, sys, tempfile, copy
from pathlib import Path
sys.path.insert(0, {src!r})
import yaml
import numpy as np
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec
from clm_ml.legacy.namelist import convert_namelist
from clm_ml.run import load_run

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
chats = yaml.safe_load(open({towers!r}))["CHATS7"]
walnut = chats["patches"][0]
needle = {{"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.4, "canopy_height": 15.0,
          "root_biomass": 800.0, "n_layers_within": 8, "n_layers_above": 6}}
a = copy.deepcopy(chats); a["patches"] = [dict(walnut, weight=0.6), needle]
b = copy.deepcopy(chats); b.update(latitude=45.0, longitude=-100.0)
b["soil"].update(bedrock_depth=1.0, color_class=5)
b["patches"] = [dict(walnut, weight=1.0, n_layers_within=10, n_layers_above=8)]
(tmp / "towers.yaml").write_text(yaml.safe_dump({{"SITEA": a, "SITEB": b}}))
base = yaml.safe_load((tmp / "run.yaml").read_text())

def model(towers, pad):
    d = copy.deepcopy(base)
    d["domain"].update(towers=towers, towers_file="towers.yaml")
    if pad:
        d["execution"] = {{"patch_count": 4, "gridcell_count": 3}}
    name = "_".join(towers) + ("_pad" if pad else "")
    (tmp / f"{{name}}.yaml").write_text(yaml.safe_dump(d))
    return Model.from_run(load_run(tmp / f"{{name}}.yaml"))

ma, mb, mb0 = model(["SITEA", "SITEB"], True), model(["SITEB"], True), model(["SITEB"], False)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux", "gpp", "net_radiation"))
first, n = 12, 6

def run(m, program=None):
    _, out = m.run_chunk(m.initial_state(), m.forcing(first, n), first, streams=[stream],
                         program=program)
    return {{k: np.asarray(v).tolist() for k, v in out["s"].items()}}

result = {{"a": run(ma), "n_patches": [ma.n_patches, mb.n_patches, mb0.n_patches]}}
runner = next(iter(ma._runners.values()))
result["b_own"] = run(mb)
result["compiled_before"] = runner._cache_size()
result["b_via_a"] = run(mb, program=ma)
result["compiled_after"] = runner._cache_size()
result["b_unpadded"] = run(mb0)
print("RESULT" + json.dumps(result))
"""


def test_padded_domains_share_one_program():
    script = SCRIPT.format(
        src=str(ROOT / "src"),
        namelist=str(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day"),
        towers=str(ROOT / "src" / "clm_ml" / "data" / "towers.yaml"),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
    r = json.loads(line[len("RESULT") :])
    assert r["n_patches"] == [4, 4, 1]  # padded batches; the unpadded B has its one patch
    assert r["compiled_after"] == r["compiled_before"]  # no new compilation
    for name, own in r["b_own"].items():
        own = np.asarray(own, dtype=float)
        # the outputs hold the real patches and gridcells only
        assert own.shape[1] == 1, name
        assert np.asarray(r["a"][name]).shape[1] == (2 if name.startswith("gridcell:") else 3)
        via = np.asarray(r["b_via_a"][name], dtype=float)
        unpadded = np.asarray(r["b_unpadded"][name], dtype=float)
        scale = float(np.max(np.abs(own)))
        assert np.max(np.abs(via - own)) <= 1e-9 * scale + 1e-12, name
        assert np.max(np.abs(unpadded - own)) <= 1e-9 * scale + 1e-12, name
