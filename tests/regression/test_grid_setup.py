"""
Phase 6b step 3 test: the setup of a grid run (``clm_ml.legacy.grid_setup``: prototypes per
PFT, every forcing- and land-data-dependent field computed per patch from its gridcell's
gridded data) equals the legacy setup of each patch from tower-format files holding that
gridcell's gridded data (``grid_setup.write_tower_files``), bit for bit.

A few 1° gridcells (several PFTs, C3 and C4) on 1 July 2000 (no snow, so the tower files,
which have no snowfall, hold the same precipitation), CO2 constant at the tower value (383
ppm). Every field of the first-step-initialized containers and the first step's inputs are
compared, except the first step's forcing height ``met.hgt`` (towers: the patch's forcing
height; grid runs: CLM's 30 m, unused after setup).

Needs the Phase 6b input files on NCAR's GLADE file system (skipped elsewhere).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
HIST = (
    "/glade/derecho/scratch/slevis/archive/wetlands.n04.ctsm5.4.044_f09BgcCrop_flat_174_Hist/"
    "lnd/hist/wetlands.n04.ctsm5.4.044_f09BgcCrop_flat_174_Hist.clm2.h0a"
)
TRENDY = "/glade/campaign/cgd/tss/projects/TRENDY2026/inputs"
SURFACE = (
    "/glade/campaign/cesm/cesmdata/inputdata/lnd/clm2/surfdata_esmf/ctsm5.4.0/"
    "surfdata_0.9x1.25_hist_2000_16pfts_c251022.nc"
)

needs_data = pytest.mark.skipif(
    not (Path(SURFACE).exists() and Path(TRENDY).exists() and Path(f"{HIST}.2000-07.nc").exists()),
    reason="Phase 6b input files (NCAR GLADE) not available",
)


def grid_run_files(tmp: Path, region: dict, start: str, days: float = 1, co2: str = "{ppm: 383}",
                   extra_model: str = "") -> Path:  # fmt: skip
    """A small grid run (run.yaml, model.yaml, params.yaml) in ``tmp``."""
    (tmp / "model.yaml").write_text(
        "canopy_layers: {n_layers_within: 10, n_layers_above: 10, max_layers: 20}\n" + extra_model
    )
    (tmp / "run.yaml").write_text(f"""
model: model.yaml
params: clm6
time: {{start: {start}, days: {days}, step_seconds: 1800}}
domain:
  type: grid
  grid:
    surface_data: {SURFACE}
    land_mask: {HIST}.2000-01.nc
    region: {region}
forcing:
  gridded:
    meteorology:
      tpqw: {TRENDY}/three_stream/clmforc.CRUJRAv4.10_0.5x0.5.TPQWL.{{year}}.nc
      solar: {TRENDY}/three_stream/clmforc.CRUJRAv4.10_0.5x0.5.Solr_Diffuse.{{year}}.nc
      precipitation: {TRENDY}/three_stream/clmforc.CRUJRAv4.10_0.5x0.5.Prec.{{year}}.nc
      mask: {TRENDY}/three_stream/domain.crujra.v2.5_0.5x0.5.c240722.nc
    co2: {co2}
    soil: {{files: "{HIST}.{{year}}-{{month:02d}}.nc"}}
output:
  directory: {tmp / "out"}
  streams: [{{name: h, frequency: hourly, variables: [latent_heat_flux, gpp]}}]
""")
    return tmp / "run.yaml"


SCRIPT = """
import sys, tempfile
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from tests.regression.test_grid_setup import grid_run_files
from clm_ml.run import load_run
from clm_ml.legacy import grid_setup, worker
tmp = Path(tempfile.mkdtemp())
path = grid_run_files(tmp, {{"lat": [38.5, 40.5], "lon": [-90.0, -88.5]}}, "2000-07-01T00:00")
run = load_run(path)
spec = run.legacy_spec(None)
grid, inputs = grid_setup.setup_grid_batch(spec, run=run, workdir=tmp / "proto")
import offline_executable.main as M
d = run.domain
print("patches", d.n_patches, "gridcells", d.n_gridcells, "C4", int((d.patches.pft_index == 14).sum()))
met = inputs.met_at([0, 1])
print("snow", float(np.max(met["snow"])))
# the legacy setup of every patch from its gridcell's data as tower files
from clm_ml.legacy.grid_setup import SOIL_LEVELS, tower_spec, write_tower_files
patches = []
for p in range(d.n_patches):
    g = int(d.patches.gridcell_index[p])
    tw, hi = write_tower_files(inputs, g, 2, tmp / "towers", SOIL_LEVELS["clm5_0"], lai_patch=p)
    patches.append(tower_spec(run, p, f"P{{p}}", tw, hi))
legacy = worker.setup_batch_legacy(dict(spec, patches=patches))
real = legacy.real_index  # the legacy batch is laid out in C3/C4 blocks
L = jax.tree.map(lambda a: np.asarray(a)[real], M.stack_patches([insts for insts, _, _ in legacy.inits]))
LX = jax.tree.map(lambda a: np.asarray(a)[real], M.stack_patches([(i.now, i.met, i.clm_record, i.forcing_bef, i.forcing_next) for *_, i in legacy.inits]))
G, GX = grid.init_stack
bad = 0
for (path, a), b in zip(jax.tree_util.tree_leaves_with_path((L, LX)), jax.tree.leaves((G, GX))):
    a, b = np.asarray(a), np.asarray(b)
    key = jax.tree_util.keystr(path)
    if key.endswith(".hgt"):
        continue
    if a.shape != b.shape or not np.array_equal(a.astype(float), b.astype(float), equal_nan=True):
        bad += 1
        if a.shape == b.shape:
            diff = np.abs(a.astype(float) - b.astype(float)); diff[np.isnan(diff)] = 0
            print("DIFF", key, "max", diff.max())
        else:
            print("SHAPE", key, a.shape, b.shape)
print("leaves", len(jax.tree.leaves((L, LX))), "differing:", bad)
print("grids equal:", [legacy.grids[i].c3 for i in real] == [g.c3 for g in grid.grids])
"""


@needs_data
def test_grid_setup_equals_legacy_setup_of_its_data():
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(root=str(ROOT))], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-4000:]
    out = run.stdout
    assert "snow 0.0" in out, out[-3000:]
    assert "differing: 0" in out, out[-3000:]
    assert "grids equal: True" in out, out[-3000:]


#: Run both ways and keep the run directories (``tmp``): the grid run and tower runs of its
#: patches whose files hold the gridded data
RUN_SCRIPT = """
import sys, json, yaml
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from tests.regression.test_grid_setup import grid_run_files
from clm_ml.run import load_run
from clm_ml.legacy import grid_setup
from clm_ml.params import PFT_NAMES
tmp = Path({tmp!r})
path = grid_run_files(tmp, {{"lat": [39.5, 40.5], "lon": [-90.0, -88.0]}}, "2000-07-01T00:00")
run = load_run(path)
d = run.domain
inputs = grid_setup.grid_inputs(run)
towers = {{}}
for p in range(d.n_patches):
    g = int(d.patches.gridcell_index[p])
    tw, hi = grid_setup.write_tower_files(inputs, g, run.n_steps, tmp / "towers" / str(p),
                                          grid_setup.SOIL_LEVELS["clm5_0"], lai_patch=p)
    (tmp / "towers" / f"P{{p}}_met.nc").symlink_to(tw); (tmp / "towers" / f"P{{p}}_hist.nc").symlink_to(hi)
    c = d.columns
    towers[f"P{{p}}"] = {{"latitude": float(d.gridcells.latitude[g]), "longitude": float(d.gridcells.longitude[g]),
        "measurement_height": float(d.patches.measurement_height[p]), "forcing_step_minutes": 30,
        "soil": {{"sand_percent": float(c.sand_percent[p]), "clay_percent": float(c.clay_percent[p]),
                 "organic_matter": float(c.organic_matter[p]), "color_class": int(c.color_class[p]),
                 "bedrock_depth": float(c.bedrock_depth[p])}},
        "patches": [{{"pft": PFT_NAMES[int(d.patches.pft_index[p])], "weight": 1.0,
                     "canopy_height": float(d.patches.canopy_height[p]), "root_biomass": 500.0}}]}}
(tmp / "towers.yaml").write_text(yaml.safe_dump(towers))
r = yaml.safe_load(path.read_text())
r["domain"] = {{"type": "towers", "towers": list(towers), "towers_file": "towers.yaml"}}
r["forcing"] = {{"meteorology": {{"files": str(tmp / "towers" / "{{tower}}_met.nc")}},
                "clm_history": {{"file": str(tmp / "towers" / "{{tower}}_hist.nc"),
                                "start": "2000-07-01T00:00"}}}}
(tmp / "tower_run.yaml").write_text(yaml.safe_dump(r))
for name, f in (("grid", path), ("tower", tmp / "tower_run.yaml")):
    out = load_run(f).execute(write=False)["h"]
    np.savez(tmp / f"{{name}}_out.npz", **{{v: np.asarray(out.data[v]) for v in ("latent_heat_flux", "gpp")}})
print("ran")
"""

#: One run's chunk inputs per patch and step (the meteorology gathered to the patches, the
#: CLM history record as the chunk forms it)
INPUTS_SCRIPT = """
import sys
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from clm_ml.run import load_run
from clm_ml.legacy import worker
import offline_executable.main as M
tmp = Path({tmp!r})
batch = worker.setup_batch(load_run(tmp / {run_file!r}).legacy_spec(None))
xg, xp = M.read_batch_inputs(batch, 1, 47)
if batch.land_fn is not None:
    xw, land = xp
    gidx = jax.numpy.asarray(batch.runner_gidx)
    xp = jax.jit(jax.vmap(lambda w: batch.land_fn(w, land, gidx)))(xw)
real = np.asarray(batch.real_index); gi = np.asarray(batch.runner_gidx)[real]
out = {{}}
for name, tree, idx in (("xg", xg[1:], gi), ("xp", xp, real)):  # xg[0]: the step times
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        a = np.asarray(leaf)
        out[name + jax.tree_util.keystr(path)] = a[:, idx] if a.ndim >= 2 else a
np.savez(tmp / {out!r}, **out)
print("saved")
"""


def _python(script):
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-4000:]
    return run.stdout


@needs_data
def test_grid_run_equals_tower_runs_of_its_data(tmp_path):
    """A day of a grid run against tower runs of its patches whose files hold the gridded
    data (meteorology at the model step, vegetation and soil water as the grid run forms
    them): (1) every chunk input of every patch and step is the same (the forcing height
    ``met.hgt`` excepted: unused after setup; vegetation and soil water to the last bit); (2) the outputs are equal to round-off
    (<= 1e-10 relative; measured <= 2e-13) through the night. After sunrise they differ (up to ~10 W m-2 in latent heat by evening): these
    1° patches amplify round-off (a 1e-15 relative change of LAI alone moves a tower run by
    ~8 W m-2 within hours; open issue in docs/design/global_input.md), and the two
    programs (vegetation and soil water formed inside the chunk vs read per step) round
    differently."""
    import numpy as np

    _python(RUN_SCRIPT.format(root=str(ROOT), tmp=str(tmp_path)))
    for run_file, out in (("run.yaml", "grid_in.npz"), ("tower_run.yaml", "tower_in.npz")):
        _python(INPUTS_SCRIPT.format(root=str(ROOT), tmp=str(tmp_path), run_file=run_file, out=out))
    gin, tin = np.load(tmp_path / "grid_in.npz"), np.load(tmp_path / "tower_in.npz")
    assert sorted(gin.files) == sorted(tin.files)
    # the meteorology exactly; the vegetation and soil water to the last bit (XLA fuses the
    # interpolation's multiply-adds, NumPy does not)
    differing = [
        k for k in tin.files
        if not k.endswith(".hgt")
        and not (np.array_equal(gin[k], tin[k]) if k.startswith("xg")
                 else np.allclose(gin[k], tin[k], rtol=1e-15, atol=0.0))
    ]  # fmt: skip
    assert not differing, differing
    g, t = np.load(tmp_path / "grid_out.npz"), np.load(tmp_path / "tower_out.npz")
    for v in ("latent_heat_flux", "gpp"):
        night = np.max(np.abs(g[v][:12] - t[v][:12])) / np.max(np.abs(t[v]))  # 00-12 UTC
        assert night <= 1e-10, (v, night)
        assert np.max(np.abs(g[v] - t[v])) <= 0.1 * np.max(np.abs(t[v])), v


THIN_LAYER_SCRIPT = """
import sys
from pathlib import Path
ROOT = Path({root!r}); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import yaml, numpy as np
from tests.regression.test_grid_setup import grid_run_files
from clm_ml.run import load_run
tmp = Path({tmp!r})
path = grid_run_files(tmp, {{"lat": [41.5, 42.5], "lon": [-109.0, -108.5]}}, "2000-01-01T00:00",
                      days=16, co2="{{ppm: 369}}",
                      extra_model='numerics: {{ignore_checks: ["CanopyFluxesDiagnostics 3"]}}\\n')
r = yaml.safe_load(path.read_text())
r["output"]["streams"] = [{{"name": "s", "frequency": "step", "statistic": "instantaneous",
    "level": ["patch"], "variables": ["mlcanopy.td_profile", "mlcanopy.dpai_profile",
    "mlcanopy.tleaf_leaf", "stem_area_added", "stem_area_index", "leaf_area_index"]}}]
path.write_text(yaml.safe_dump(r))
run = load_run(path)
pft = [str(x) for x in run.domain.patches.pft_index]
out = run.execute(write=False)["s"]
d = {{k: np.asarray(v) for k, v in {{**out.data, **out.raw}}.items()}}
td, dpai, tl = d["mlcanopy.td_profile"], d["mlcanopy.dpai_profile"], d["mlcanopy.tleaf_leaf"]
leafy = (dpai > 0) & (dpai < 1e30)
print("max td of leafy layers", float(td[leafy].max()))
print("min dpai of leafy layers", float(dpai[leafy].min()))
print("leaf T range", float(tl[..., 1:3][leafy].min()), float(tl[..., 1:3][leafy].max()))
print("stem area added max by patch", np.round(d["stem_area_added"].max(axis=0), 4).tolist(), pft)
"""


@needs_data
def test_thin_layers_keep_minimum_plant_area(tmp_path):
    """Issue 14: a stems-only shrub (LAI 0, SAI 0.108) in Wyoming, January 2000, whose bottom
    layer held 0.0013 m2 m-2: CLM-ml's 9-angle diffuse transmittance exceeded 1 there and
    the leaf temperature ran away at a calm sunrise (13 January). With every leafy layer
    kept at min_layer_plant_area (0.01, added as stems) the run completes (patch errors stop
    the run), every leafy layer transmits less than it receives, and leaf temperatures stay
    physical."""
    out = _python(THIN_LAYER_SCRIPT.format(root=str(ROOT), tmp=str(tmp_path)))
    value = lambda key: [float(x) for x in out.split(key)[1].split("\n")[0].split()]  # noqa: E731
    assert value("max td of leafy layers")[0] < 1.0, out
    assert value("min dpai of leafy layers")[0] >= 0.01 - 1e-12, out
    t_min, t_max = value("leaf T range")
    assert 200.0 < t_min and t_max < 330.0, out
    added = out.split("stem area added max by patch")[1].split("]")[0]
    assert max(float(x) for x in added.strip(" [").split(",")) > 0.0, out
