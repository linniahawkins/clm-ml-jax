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

# Largest allowed difference, relative to each field's magnitude. A one-unit change in the
# last digit of a solved temperature (2e-16 relative) appears as up to ~1e-11 in fields that
# are differences of nearly equal values (e.g. the sensible heat flux between layers,
# -cp (T(ic+1) - T(ic)) gac; storage terms (T - T_bef)); measured in Phase 5b, where
# computing on all layer slots changed the last digit of the solved air temperatures.
# 1e-10 is the level above which the regression policy looks for a defect
# (numerics_and_regression.md §6).
ROUNDOFF = 1e-10


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
