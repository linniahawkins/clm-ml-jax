"""
Phase 6 step 6 test: the vectorized setup (``clm_ml.legacy.vector_setup``) equals the legacy
setup of every patch (``clm_ml.legacy.worker.setup_batch_legacy``) bit for bit.

Seven patches on three gridcells, every towers.yaml input varied: two forcing heights (two
forcing keys), four PFTs (C3 and C4), sand/clay given or from the texture class, organic
matter, soil colour, bedrock depth, canopy height given or the PFT default, root biomass,
leaf/stem profiles given or the PFT default, explicit or default layer counts. Every field of
the first-step-initialized containers and the first step's inputs are compared, in domain
order. The vectorized setup runs first in a fresh process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import sys, tempfile, copy, time
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, yaml, jax
from clm_ml.legacy.namelist import convert_namelist
from clm_ml.run import load_run
from clm_ml.legacy import worker, vector_setup
import offline_executable.main as M
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
spec = load_run(tmp / "r.yaml").legacy_spec(None)
# the vectorized setup first, in a fresh process (it must not rely on state a legacy setup
# leaves behind: an early version read the soil layer counts before they were set)
t0 = time.perf_counter(); vec = vector_setup.setup_batch(spec); t1 = time.perf_counter()
legacy = worker.setup_batch_legacy(spec); t2 = time.perf_counter()
print(f"setup: vectorized {{t1-t0:.1f}}s, legacy {{t2-t1:.1f}}s for {{len(spec['patches'])}} patches", flush=True)
L = jax.tree.map(lambda a: np.asarray(a)[legacy.real_index], M.stack_patches([insts for insts, _, _ in legacy.inits]))
V = vec.init_stack[0]
bad = 0
for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(L), jax.tree.leaves(V)):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
        bad += 1
        if a.shape == b.shape:
            diff = np.abs(a.astype(float) - b.astype(float)); diff[np.isnan(diff)] = 0
            where = np.argwhere(~np.isclose(a.astype(float), b.astype(float), rtol=0, atol=0, equal_nan=True))
            print("DIFF", jax.tree_util.keystr(path), a.dtype, b.dtype, "max", diff.max(), "patches", sorted(set(where[:, 0].tolist()))[:8])
        else:
            print("SHAPE", jax.tree_util.keystr(path), a.shape, b.shape)
LX = jax.tree.map(lambda a: np.asarray(a)[legacy.real_index], M.stack_patches([(i.now, i.met, i.clm_record, i.forcing_bef, i.forcing_next) for *_, i in legacy.inits]))
xbad = sum(not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) for a, b in zip(jax.tree.leaves(LX), jax.tree.leaves(vec.init_stack[1])))
print("fields differing:", bad, "of", len(jax.tree.leaves(L)), "; first-step inputs differing:", xbad)
print("grids equal:", [legacy.grids[i].c3 for i in legacy.real_index] == [g.c3 for g in vec.grids])
"""


def test_vectorized_setup_equals_legacy_setup():
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(root=str(ROOT))], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-3000:]
    out = run.stdout
    assert "fields differing: 0 of" in out, out[-3000:]
    assert "first-step inputs differing: 0" in out, out[-3000:]
    assert "grids equal: True" in out, out[-3000:]
