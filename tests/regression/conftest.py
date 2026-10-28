"""One 50-step CHATS7 run per test session, shared by the regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from tests.regression.record_run import REFERENCE_FILE, TWIN_FILES, recorded_fields, run_case


def _load(path):
    if not path.exists():
        pytest.fail(f"Missing {path}; run tests/regression/record_run.py --update-reference")
    with np.load(path) as data:
        return {k: data[k] for k in data.files if not k.startswith("meta.")}


@pytest.fixture(scope="session")
def reference():
    return _load(REFERENCE_FILE)


@pytest.fixture(scope="session")
def twins():
    return {name: _load(path) for name, path in TWIN_FILES.items()}


@pytest.fixture(scope="session")
def current(tmp_path_factory):
    """``(run, outputs, output_directory)`` of the reference case with the current code."""
    workdir = tmp_path_factory.mktemp("regression")
    run, outputs = run_case(workdir, legacy_ascii=True)
    return run, outputs, workdir / "out"


@pytest.fixture(scope="session")
def current_fields(current):
    return recorded_fields(current[1].raw)
