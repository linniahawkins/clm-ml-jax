"""
End-to-end regression test: the model must reproduce its own committed output.

This is the guardrail for the modernization refactor (docs/design/modernization_plan.md,
Phase 0). Refactoring should not change results, so a fresh 50-step CHATS7 run is
compared field by field against ``reference/CHATS7_50steps.npz`` at ``RTOL``.

The Fortran comparison test (2.5 % tolerance on 3-decimal ASCII output) is far too
loose to catch refactoring mistakes; this test is not.

If results change *on purpose* (a science change or bug fix), regenerate the reference
and say so in the PR::

    python tests/regression/record_run.py --update-reference
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.regression.record_run import (
    MLCANOPY_FIELDS,
    REFERENCE_FILE,
    REFERENCE_NAMELIST,
    TEMPERATURE_FIELDS,
    run_in_subprocess,
)

RTOL = 1e-6

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE_FILE.exists():
        pytest.fail(f"Missing {REFERENCE_FILE}; run record_run.py --update-reference")
    with np.load(REFERENCE_FILE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("regression")
    out_file = workdir / "current.npz"
    run_in_subprocess(REFERENCE_NAMELIST, out_file, workdir)
    with np.load(out_file) as data:
        return {key: data[key] for key in data.files}


FIELDS = (
    ["calendar_day"]
    + [f"mlcanopy.{name}" for name in MLCANOPY_FIELDS]
    + [f"temperature.{name}" for name in TEMPERATURE_FIELDS]
)


@pytest.mark.parametrize("field", FIELDS)
def test_field_matches_reference(field, reference, current):
    ref = reference[field]
    new = current[field]
    assert new.shape == ref.shape, f"{field}: shape {new.shape} != reference {ref.shape}"

    if np.issubdtype(ref.dtype, np.integer):
        np.testing.assert_array_equal(new, ref, err_msg=field)
        return

    # Relative tolerance, with an absolute floor scaled to the field's magnitude so that
    # values crossing zero (e.g. night-time fluxes) are not held to an impossible bar.
    finite = np.isfinite(ref) & (np.abs(ref) < 1e30)  # skip spval (1e36) fill values
    scale = np.max(np.abs(ref[finite])) if finite.any() else 0.0
    np.testing.assert_allclose(
        new, ref, rtol=RTOL, atol=RTOL * scale, equal_nan=True, err_msg=field
    )
