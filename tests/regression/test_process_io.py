"""
Routine-level regression: each physics routine reproduces its recorded outputs.

``reference/process_io.pkl.gz`` holds real inputs and outputs of the routines that had separate
"normal" and "traceable" versions before Phase 3c, recorded at dusk, night and midday of the
CHATS7 reference day from the code before the merge (``process_io.py --record``). Replaying the
same inputs through the current routines must reproduce the outputs to round-off. This is the
identical-input check of docs/design/numerics_and_regression.md §1: precise where the
end-to-end noise-twin test is coarse.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.regression.process_io import PROCESS_IO_FILE, ROUTINES

pytestmark = pytest.mark.regression

ROUNDOFF = 1e-12  # largest allowed difference, relative to each field's magnitude


@pytest.fixture(scope="module")
def replay_report(tmp_path_factory):
    if not PROCESS_IO_FILE.exists():
        pytest.fail(f"Missing {PROCESS_IO_FILE}; run tests/regression/process_io.py --record")
    out = tmp_path_factory.mktemp("process_io") / "replay.json"
    script = PROCESS_IO_FILE.parents[1] / "process_io.py"
    subprocess.run([sys.executable, str(script), "--replay", str(out)], check=True)
    return json.loads(out.read_text())


@pytest.mark.parametrize("routine", ROUTINES)
def test_routine_reproduces_recorded_outputs(routine, replay_report):
    calls = replay_report[routine]
    assert calls, f"no recorded calls for {routine}"
    for call in calls:
        worst = {f: d for f, d in call["max_rel_diff"].items() if d > ROUNDOFF}
        assert not worst, f"{routine} (step {call['step']}): fields beyond round-off: {worst}"
