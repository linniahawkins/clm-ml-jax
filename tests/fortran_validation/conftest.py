"""
pytest configuration for CLM-ML-JAX Fortran validation tests.

These tests compare JAX function outputs against golden reference values
captured from verified Fortran builds of the CLM-ml model.

Golden data lives in:  clm-ml-fortran/golden_IO/<function>.json
JAX source lives in:   src/

Session setup:
  - JAX 64-bit precision enabled (must happen before any JAX import)
  - src/ added to sys.path
  - RSL psihat lookup tables initialized when possible (needed by GetPsiRSL tests)
"""

import sys
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
GOLDEN_DIR = PROJECT_ROOT / "tests" / "fortran_validation" / "golden_IO"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest  # noqa: E402

# Session-level flag: True if RSL tables were loaded successfully.
RSL_AVAILABLE: bool = False


@pytest.fixture(scope="session", autouse=True)
def initialize_rsl_tables():
    """
    Initialize RSL psihat lookup tables.

    Required for GetPsiRSL tests.  Fails gracefully if netCDF4 is
    unavailable (binary incompatibility) — those tests are then skipped.
    """
    global RSL_AVAILABLE
    try:
        # netCDF4 may emit a RuntimeWarning about numpy ABI mismatch on
        # some HPC environments; suppress it so it doesn't abort the session.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            from clm_ml.io.rsl_tables import default_rsl_tables

            default_rsl_tables()
        RSL_AVAILABLE = True
    except Exception as exc:
        RSL_AVAILABLE = False
        # Don't fail the session — GetPsiRSL tests will skip themselves.
        warnings.warn(
            f"RSL lookup table init failed ({type(exc).__name__}: {exc}). "
            "GetPsiRSL tests will be skipped.",
            stacklevel=1,
        )


@pytest.fixture
def require_rsl():
    """Skip the test if RSL tables weren't loaded successfully."""
    if not RSL_AVAILABLE:
        pytest.skip("RSL psihat lookup tables unavailable (netCDF4 load failed)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "fortran_validation: tests that compare JAX output against Fortran golden data",
    )
    config.addinivalue_line(
        "markers",
        "requires_rsl: test requires RSL psihat lookup tables (netCDF4)",
    )
