"""Every module in src/ imports without ImportError.

A module imported only inside functions can hold a stale name (e.g. a removed function)
that the linter does not see until the model runs; this test finds it in seconds.
"""

import json
import subprocess
import sys

from .conftest import SRC_DIR

PACKAGES = ["clm_ml"]

_SCRIPT = """
import importlib, json, pkgutil, sys
import jax
jax.config.update("jax_enable_x64", True)
errors = {}
for pkg in sys.argv[1:]:
    module = importlib.import_module(pkg)
    for info in pkgutil.walk_packages(module.__path__, pkg + "."):
        try:
            importlib.import_module(info.name)
        except ImportError as exc:
            errors[info.name] = str(exc)
print(json.dumps(errors))
"""


def test_all_modules_import():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, *PACKAGES],
        cwd=SRC_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    errors = json.loads(result.stdout.strip().splitlines()[-1])
    assert errors == {}, errors
