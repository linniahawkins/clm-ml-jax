"""Forcing is read in chunks with halo records; chunks reproduce a single whole-run read."""

import json
import subprocess
import sys

import pytest

from clm_ml.forcing import chunk_run_steps

from .conftest import SRC_DIR


def test_chunk_run_steps_interior_and_run_edges():
    # Interior chunk: halos are the neighbouring steps.
    assert chunk_run_steps(10, 3, 4) == [2, 3, 4, 5, 6, 7]
    # First and last chunk: the halo repeats the edge step (Fortran behaviour).
    assert chunk_run_steps(10, 0, 3) == [0, 0, 1, 2, 3]
    assert chunk_run_steps(10, 7, 3) == [6, 7, 8, 9, 9]
    # One chunk covering the whole run.
    assert chunk_run_steps(3, 0, 3) == [0, 0, 1, 2, 2]


@pytest.mark.parametrize("args", [(10, 8, 3), (10, -1, 2), (10, 0, 0)])
def test_chunk_outside_run_is_rejected(args):
    with pytest.raises(ValueError):
        chunk_run_steps(*args)


_SCRIPT = r"""
import datetime, json, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from offline_driver import clmSoilOptionMod
clmSoilOptionMod.clm_phys = "CLM5_0"
clmSoilOptionMod.nlev_soil_adjust = 3
from clm_src_main.clm_varpar import clm_varpar_init
clm_varpar_init()
from clm_ml.legacy.forcing import load_forcing
from offline_driver.TowerMetMod import tower_first_record

tower = "input_files/tower-forcing/CHATS7/2007-05.nc"
clm = "input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc"
adjust = "input_files/clm5_0/CHATS7/CHATS_soil_moisture_correction_29Jul25.nc"
start = datetime.datetime(2007, 5, 1)
first = tower_first_record(tower, start, 1800)
n_run = 200

def load(first_step, n_steps):
    return load_forcing(tower, clm, adjust, first_record=first, run_n_steps=n_run,
                        first_step=first_step, n_steps=n_steps,
                        history_index=lambda k: 1441 + k, tower_height=23.0)

def arrays(f):
    out = {}
    for group in (f.tower, f.clm_history):
        for name, value in vars(group).items():
            if value is not None:
                out[f"{type(group).__name__}.{name}"] = np.asarray(value)
    return out

whole = arrays(load(0, n_run))
problems = []
for first_step, n_steps in [(0, 70), (70, 70), (140, 60)]:
    chunk = arrays(load(first_step, n_steps))
    for name, values in chunk.items():
        # whole-run array index of run step k is k + 1 (index 0 is the leading halo)
        expected = whole[name][first_step : first_step + n_steps + 2]
        if not np.array_equal(values, expected):
            problems.append(f"{name} differs in chunk starting at {first_step}")
records = {
    "start": tower_first_record(tower, start, 1800),
    "next_day": tower_first_record(tower, start + datetime.timedelta(days=1), 1800),
}
print(json.dumps({"problems": problems, "records": records}))
"""


def test_chunked_reads_equal_one_whole_run_read():
    """Three chunks, including halos, are identical to the matching slices of one read."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], cwd=SRC_DIR, capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["problems"] == []
    # The tower file starts at 2007-05-01 00Z with 30-minute records.
    assert report["records"] == {"start": 0, "next_day": 48}
