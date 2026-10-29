"""
Phase 6 step 5 test: patches spread over several devices (``execution.devices``,
``jax.shard_map``), on CPU with simulated devices (``--xla_force_host_platform_device_count``).

A domain of three gridcells and four patches, C3 and C4: ``SITEA`` (a walnut and a C4 grass),
``SITEB`` (a walnut, another location and soil) and ``SITEC`` (a needleleaf tree). Spread over
four devices (one gridcell per device block, one C3 and one C4 slot per block, the fourth
block all padding), it gives the same patch and gridcell outputs, in domain order, as on one
device.
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
VARS = ["latent_heat_flux", "sensible_heat_flux", "gpp", "net_radiation"]


def _towers() -> dict:
    chats = yaml.safe_load((ROOT / "src" / "clm_ml" / "data" / "towers.yaml").read_text())["CHATS7"]
    walnut = chats["patches"][0]
    grass = {"pft": "c4_grass", "weight": 0.3, "canopy_height": 0.5, "root_biomass": 400.0,
             "n_layers_within": 5, "n_layers_above": 10}  # fmt: skip
    needle = {"pft": "needleleaf_evergreen_temperate_tree", "weight": 1.0, "canopy_height": 15.0,
              "root_biomass": 800.0, "n_layers_within": 8, "n_layers_above": 6}  # fmt: skip
    a = copy.deepcopy(chats)
    a["patches"] = [dict(walnut, weight=0.7), grass]
    b = copy.deepcopy(chats)
    b.update(latitude=45.0, longitude=-100.0)
    b["soil"].update(bedrock_depth=1.0, color_class=5)
    b["patches"] = [dict(walnut, weight=1.0, n_layers_within=10, n_layers_above=8)]
    c = copy.deepcopy(chats)
    c["patches"] = [needle]
    return {"SITEA": a, "SITEB": b, "SITEC": c}


def _execute(tmp: Path, devices: int):
    sys.path.insert(0, str(ROOT / "src"))
    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.run import load_run

    work = tmp / f"devices{devices}"
    work.mkdir()
    convert_namelist(ROOT / "src" / "offline_executable" / "nl.CHATS7.1day", work)
    (work / "towers.yaml").write_text(yaml.safe_dump(_towers()))
    d = yaml.safe_load((work / "run.yaml").read_text())
    d["domain"].update(towers=["SITEA", "SITEB", "SITEC"], towers_file="towers.yaml")
    d["execution"] = {"devices": devices}
    d["output"] = {
        "directory": str(work / "out"),
        "streams": [{"name": "step", "frequency": "step", "statistic": "mean",
                     "variables": VARS, "level": ["patch", "gridcell"]}],
    }  # fmt: skip
    (work / "run.yaml").write_text(yaml.safe_dump(d))
    return load_run(work / "run.yaml").execute(write=False, quiet=True)["step"]


def test_patches_spread_over_devices(tmp_path, monkeypatch):
    # the worker processes inherit the flag: four CPU devices
    flags = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    monkeypatch.setenv("XLA_FLAGS", flags.strip())
    one, four = _execute(tmp_path, 1), _execute(tmp_path, 4)
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert list(four.patch_gridcell) == [0, 0, 1, 2]
    for name in VARS:
        a, b = one.data[name], four.data[name]
        assert a.shape == b.shape == (48, 4), name
        scale = float(np.max(np.abs(a)))
        assert np.max(np.abs(b - a)) <= 1e-9 * scale + 1e-12, name
        ga, gb = one.gridcell_mean()[name], four.gridcell_mean()[name]
        assert ga.shape == gb.shape == (48, 3), name
        assert np.max(np.abs(gb - ga)) <= 1e-9 * scale + 1e-12, name
