"""
Phase 6 step 4 test (decision 22): C3 and C4 patches in one compiled program.

One gridcell with a C3 patch (the CHATS7 walnut, weight 0.7) and a C4 patch (C4 grass, 0.5 m,
explicit layer counts, weight 0.3). The two form groups inside one program, each running only
its own photosynthesis pathway. The run of the gridcell matches each patch run alone, and the
gridcell value is the area-weighted sum.
"""

from __future__ import annotations

import copy
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
VARS = ["latent_heat_flux", "sensible_heat_flux", "gpp", "net_radiation"]


def _execute(tmp: Path, name: str, towers: dict):
    sys.path.insert(0, str(ROOT / "src"))
    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.run import load_run

    work = tmp / name
    work.mkdir()
    convert_namelist(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day", work)
    (work / "towers.yaml").write_text(yaml.safe_dump(towers))
    d = yaml.safe_load((work / "run.yaml").read_text())
    d["domain"].update(towers=list(towers), towers_file="towers.yaml")
    d["output"] = {
        "directory": str(work / "out"),
        "streams": [{"name": "step", "frequency": "step", "statistic": "mean",
                     "variables": VARS, "level": ["patch", "gridcell"]}],
    }  # fmt: skip
    (work / "run.yaml").write_text(yaml.safe_dump(d))
    return load_run(work / "run.yaml").execute(write=False, quiet=True)["step"]


def _close(actual, expected, name):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(actual - expected)) <= 1e-9 * scale + 1e-12, name


def test_c3_and_c4_patches_in_one_program(tmp_path):
    chats = yaml.safe_load((ROOT / "src" / "clm_ml" / "data" / "towers.yaml").read_text())["CHATS7"]
    walnut = dict(chats["patches"][0], weight=0.7)
    grass = {"pft": "c4_grass", "weight": 0.3, "canopy_height": 0.5, "root_biomass": 400.0,
             "n_layers_within": 5, "n_layers_above": 10}  # fmt: skip
    mixed = copy.deepcopy(chats)
    mixed["patches"] = [walnut, grass]
    jobs = {"mixed": {"MIXED": mixed}}
    for name, patch in (("C3", walnut), ("C4", grass)):
        t = copy.deepcopy(chats)
        t["patches"] = [dict(patch, weight=1.0)]
        jobs[name] = {name: t}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {k: pool.submit(_execute, tmp_path, k, v) for k, v in jobs.items()}
        out = {k: f.result() for k, f in futures.items()}
    shutil.rmtree(tmp_path, ignore_errors=True)

    mixed = out["mixed"]
    for name in VARS:
        _close(mixed.data[name][:, 0], out["C3"].data[name][:, 0], f"{name} C3")
        _close(mixed.data[name][:, 1], out["C4"].data[name][:, 0], f"{name} C4")
        expected = 0.7 * out["C3"].data[name][:, 0] + 0.3 * out["C4"].data[name][:, 0]
        _close(mixed.gridcell_mean()[name][:, 0], expected, f"{name} gridcell")
    # the C4 pathway takes effect: daytime GPP differs from the C3 patch's
    assert not np.allclose(mixed.data["gpp"][:, 0], mixed.data["gpp"][:, 1], rtol=0.1)
