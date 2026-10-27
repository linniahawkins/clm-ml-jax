"""Shared setup for clm_ml tests: put src/ on the path (works without reinstalling)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CHATS_NAMELISTS = SRC_DIR / "offline_executable"


@pytest.fixture
def chats_run_dir(tmp_path):
    """A converted nl.CHATS7.50steps (run.yaml + model.yaml) in a temporary directory."""
    from clm_ml.legacy.namelist import convert_namelist

    convert_namelist(CHATS_NAMELISTS / "nl.CHATS7.50steps", tmp_path)
    return tmp_path
