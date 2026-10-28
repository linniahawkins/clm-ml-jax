"""
Restarts: a run resumed from a mid-run restart gives the same results, bit for bit, as the
uninterrupted run, including an output record (a daily mean) that spans the restart.

CHATS7 forcing covers one month, so a monthly run has one chunk; the test replaces the month
schedule with two chunks (steps 1-23 and 24-49) and writes a restart after every chunk. The
end-of-run restart rotates the mid-run one to ``.prev``, which the resumed run reads. Each run
is a fresh process (module-level tables).
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

M.month_chunks = lambda run, first: [c for c in [(1, 23), (24, 26)] if c[0] >= first]
M.restart_boundary = lambda end, every: True
tmp = Path(tempfile.mkdtemp()); (tmp / "out").mkdir()
nml = write_portable_namelist(Path({namelist!r}), tmp / "nl", tmp / "out")
run = M.setup_run(M.read_namelist(str(nml)))
streams = [StreamSpec("step", "step", "instantaneous", ("latent_heat_flux", "air_temperature")),
           StreamSpec("daily", "daily", "mean", ("latent_heat_flux",))]
records = [[], []]
restart_path = Path({restart!r})
restart = M.read_restart(Path(str(restart_path) + ".prev")) if {resume!r} else None
M.run_streams(run, streams, 50, lambda i, r: records[i].extend(r), restart=restart,
              restart_path=None if {resume!r} else restart_path)
with open({out!r}, "wb") as fh:
    pickle.dump(records, fh)
"""


def _run(tmp_path, resume: bool):
    root = HERE.parents[1]
    out = tmp_path / ("resumed.pkl" if resume else "full.pkl")
    script = SCRIPT.format(
        src=str(root / "src"),
        root=str(root),
        namelist=str(root / "src" / "offline_executable" / "nl.CHATS7.50steps"),
        restart=str(tmp_path / "restart.pkl"),
        resume=resume,
        out=str(out),
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    with open(out, "rb") as fh:
        return pickle.load(fh)


def test_resume_from_mid_run_restart_reproduces_the_full_run(tmp_path):
    full = _run(tmp_path, resume=False)
    assert (tmp_path / "restart.pkl.prev").exists()  # the restart after the first chunk
    resumed = _run(tmp_path, resume=True)

    # The record of step 23 was still pending (not yet emitted) when the restart was
    # written, so the resumed run completes it: it returns steps 23-49.
    step_full, step_resumed = full[0], resumed[0]
    assert len(step_full) == 50 and [r[0] for r in step_resumed] == list(range(23, 50))
    for (k1, e1, v1), (k2, e2, v2) in zip(step_full[23:], step_resumed):
        assert (k1, e1) == (k2, e2)
        for name in v1:
            np.testing.assert_array_equal(v1[name], v2[name], err_msg=f"step {k1} {name}")

    # Day 1 (steps 0-47) spans the restart: its partial mean was saved and completed.
    assert [r[0] for r in full[1]] == [r[0] for r in resumed[1]]
    for (_, _, v1), (_, _, v2) in zip(full[1], resumed[1]):
        np.testing.assert_array_equal(v1["latent_heat_flux"], v2["latent_heat_flux"])
