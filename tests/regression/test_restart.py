"""
Restarts and fixed-length chunks (decision 33): a run resumed from a mid-run restart gives
the same results, bit for bit, as the uninterrupted run, including an output record (a daily
mean) that spans the restart; and a run in 23-step chunks, whose short last chunk is padded,
gives the same results as the run in one chunk: bit for bit at every step, to round-off
(1e-15) for a record spanning chunks.

The 50-step CHATS7 run in 23-step chunks: steps 1-23, 24-46 and 47-49 (padded to 23 steps).
A restart is written after every chunk; the end-of-run restart rotates the one after the
second chunk to ``.prev``, which the resumed run reads. Each run is a fresh process
(module-level tables).
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

HERE = Path(__file__).resolve().parent

SCRIPT = """
import pickle, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {root!r})
import offline_executable.main as M
from tests.regression.record_run import write_portable_namelist
from clm_ml.output_streams import StreamSpec

M.restart_boundary = lambda start, end, every: True
tmp = Path(tempfile.mkdtemp()); (tmp / "out").mkdir()
nml = write_portable_namelist(Path({namelist!r}), tmp / "nl", tmp / "out")
run = M.setup_run(M.read_namelist(str(nml)))
streams = [StreamSpec("step", "step", "instantaneous", ("latent_heat_flux", "air_temperature")),
           StreamSpec("daily", "daily", "mean", ("latent_heat_flux",))]
records = [[], []]
restart_path = Path({restart!r})
restart = M.read_restart(Path(str(restart_path) + ".prev")) if {resume!r} else None
M.run_streams(run, streams, 50, lambda i, r: records[i].extend(r), restart=restart,
              restart_path=None if {resume!r} else restart_path, chunk_steps={chunk!r})
with open({out!r}, "wb") as fh:
    pickle.dump(records, fh)
"""


def _run(tmp_path, resume: bool, chunk=23):
    root = HERE.parents[1]
    out = tmp_path / ("resumed.pkl" if resume else f"full_{chunk}.pkl")
    script = SCRIPT.format(
        src=str(root / "src"),
        root=str(root),
        namelist=str(root / "src" / "offline_executable" / "nl.CHATS7.50steps"),
        restart=str(tmp_path / "restart.pkl"),
        resume=resume,
        chunk=chunk,
        out=str(out),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    with open(out, "rb") as fh:
        return pickle.load(fh)


def test_resume_from_mid_run_restart_reproduces_the_full_run(tmp_path):
    full = _run(tmp_path, resume=False)
    assert (tmp_path / "restart.pkl.prev").exists()  # the restart after the second chunk
    resumed = _run(tmp_path, resume=True)

    # The record of step 46 was still pending (not yet emitted) when the restart was
    # written, so the resumed run completes it: it returns steps 46-49.
    step_full, step_resumed = full[0], resumed[0]
    assert len(step_full) == 50 and [r[0] for r in step_resumed] == list(range(46, 50))
    for (k1, e1, v1), (k2, e2, v2) in zip(step_full[46:], step_resumed):
        assert (k1, e1) == (k2, e2)
        for name in v1:
            np.testing.assert_array_equal(v1[name], v2[name], err_msg=f"step {k1} {name}")

    # Day 1 (steps 0-47) spans the restart: its partial mean was saved and completed.
    assert [r[0] for r in full[1]] == [r[0] for r in resumed[1]]
    for (_, _, v1), (_, _, v2) in zip(full[1], resumed[1]):
        np.testing.assert_array_equal(v1["latent_heat_flux"], v2["latent_heat_flux"])


def test_padded_fixed_length_chunks_equal_one_chunk(tmp_path):
    chunked = _run(tmp_path, resume=False)  # 23-step chunks, the last one padded
    one = _run(tmp_path, resume=False, chunk=None)  # one chunk (calendar month)
    # every step bit for bit; a record spanning chunks (the daily mean) to round-off, since
    # its partial sums from the two chunks are added on the host (another order of addition)
    for stream_chunked, stream_one, rtol in zip(chunked, one, (0.0, 1e-15)):
        assert [r[:2] for r in stream_chunked] == [r[:2] for r in stream_one]
        for (k, _, v1), (_, _, v2) in zip(stream_chunked, stream_one):
            for name in v1:
                np.testing.assert_allclose(
                    v1[name], v2[name], rtol=rtol, atol=0, err_msg=f"{k} {name}"
                )
