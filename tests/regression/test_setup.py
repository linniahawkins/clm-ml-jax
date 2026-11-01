"""
Phase 7.4 test: the setup as pure functions (``clm_ml.setup``) equals the legacy setup bit for
bit: every patch's constants (``clm_ml.step.PatchConstants``) and state before the first step,
and the tower forcing read by the ported readers (``clm_ml.tower_forcing``).

Seven patches on three gridcells with every towers.yaml input varied (as
``test_vector_setup``): two forcing heights, four PFTs (C3 and C4), sand/clay given or from
the texture class, organic matter, soil colour, bedrock depth, canopy height given or the PFT
default, root biomass, leaf/stem profiles given or the PFT default, explicit or default layer
counts. The legacy reference is each patch's legacy setup with its first-step initialization
(``worker.setup_batch_legacy``), read through ``clm_ml.legacy.step.patch_constants`` and
``clm_ml.legacy.state.state_from_instances``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import sys, tempfile, copy
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, yaml, jax
jax.config.update("jax_enable_x64", True)
from clm_ml.legacy.namelist import convert_namelist
from clm_ml.run import load_run
from clm_ml.setup.patches import build_patches
from clm_ml.setup.towers import gridcell_sources, first_values
tmp = Path(tempfile.mkdtemp())
convert_namelist(ROOT / "src/offline_executable/nl.CHATS7.1day", tmp)
base = yaml.safe_load((tmp / "run.yaml").read_text())
chats = yaml.safe_load((ROOT / "src/clm_ml/data/towers.yaml").read_text())["CHATS7"]
walnut = chats["patches"][0]
def tower(lat, lon, ht, soil, patches):
    t = copy.deepcopy(chats); t.update(latitude=lat, longitude=lon, measurement_height=ht); t["soil"].update(soil); t["patches"] = patches; return t
towers = {{
  "G1": tower(38.49, -121.84, 23.0, {{}}, [dict(walnut, weight=0.5),
        {{"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.3, "canopy_height": 15.0, "root_biomass": 800.0, "n_layers_within": 8, "n_layers_above": 6}},
        {{"pft": "c4_grass", "weight": 0.2, "canopy_height": 0.5, "root_biomass": 400.0, "n_layers_within": 5, "n_layers_above": 10}}]),
  "G2": tower(45.0, -100.0, 23.0, {{"bedrock_depth": 1.0, "color_class": 5, "sand_percent": 40.0, "clay_percent": 20.0, "organic_matter": 20.0}},
        [dict(walnut, weight=0.7, canopy_height=12.0, root_biomass=600.0, leaf_area_profile_beta=[3.0, 1.5], stem_area_profile_beta=[2.0, 1.5]),
         {{"pft": "broadleaf_deciduous_boreal_tree", "weight": 0.3, "root_biomass": 500.0, "n_layers_within": 10, "n_layers_above": 4}}]),
  "G3": tower(10.0, 20.0, 21.0, {{"texture_class": "loam", "sand_percent": None, "clay_percent": None, "bedrock_depth": 3.0, "color_class": 10}},
        [dict(walnut, weight=0.6), {{"pft": "needleleaf_evergreen_temperate_tree", "weight": 0.4, "canopy_height": 9.0, "root_biomass": 700.0}}]),
}}
(tmp / "towers.yaml").write_text(yaml.safe_dump(towers))
d = copy.deepcopy(base); d["domain"].update(towers=list(towers), towers_file="towers.yaml")
(tmp / "r.yaml").write_text(yaml.safe_dump(d))
run = load_run(tmp / "r.yaml")

# the new setup first, in a fresh process (no legacy module state)
sources = gridcell_sources(run)
first = first_values(run, sources)
gi = run.domain.patches.gridcell_index
per_patch = {{k: v[gi] for k, v in first.items()}}
new = build_patches(run.domain, run.params, run.config, per_patch, acclimation_steps=1.0)
chunks = [s.chunk(0, 48) for s in sources]

from clm_ml.legacy import worker
from clm_ml.legacy.state import state_from_instances
from clm_ml.legacy.step import patch_constants
spec = run.legacy_spec(None)
legacy = worker.setup_batch_legacy(spec)
order = np.asarray(legacy.real_index)
old_c = [patch_constants(legacy.inits[i][0]) for i in order]
old_s = [state_from_instances(legacy.inits[i][0]) for i in order]
bad = 0
def compare(name, a, b):
    global bad
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        bad += 1
        print("DIFF", name, a.shape, b.shape, a.dtype, b.dtype)
for p in range(len(order)):
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(old_c[p]), jax.tree.leaves(jax.tree.map(lambda x: x[p], new.constants))):
        compare(f"patch {{p}} constants{{jax.tree_util.keystr(path)}}", a, b)
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(old_s[p]), jax.tree.leaves(jax.tree.map(lambda x: x[p], new.state))):
        compare(f"patch {{p}} state{{jax.tree_util.keystr(path)}}", a, b)
c3_old = [legacy.grids[i].c3 for i in order]
print("c3 equal:", c3_old == new.c3.tolist())
# the forcing: the ported readers against the legacy run of each gridcell's first patch
firsts = [int(np.flatnonzero(gi == g)[0]) for g in range(run.domain.n_gridcells)]
xbad = 0
for g, p in enumerate(firsts):
    old = legacy.runs[order[p]].load_chunk(0, 48).forcing
    for part in ("tower", "clm_history"):
        for k, v in vars(getattr(old, part)).items():
            w = getattr(getattr(chunks[g], part), k)
            if (v is None) != (w is None) or (v is not None and not np.array_equal(v, w)):
                xbad += 1; print("FORCING DIFF", g, part, k)
print("setup fields differing:", bad, "; forcing fields differing:", xbad)
"""


def test_setup_equals_legacy_setup():
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(root=str(ROOT))], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-3000:]
    out = run.stdout
    assert "setup fields differing: 0 ; forcing fields differing: 0" in out, out[-3000:]
    assert "c3 equal: True" in out, out[-3000:]
