"""Every module in src/ imports without ImportError.

Many legacy modules mark their imports ``# noqa: F401``, which hides a stale name
(e.g. a removed function) from the linter until the model is run. This test finds them in
seconds. It runs in a fresh process because importing legacy modules changes global state.
"""

import json
import subprocess
import sys

from .conftest import SRC_DIR

PACKAGES = [
    "clm_ml",
    "clm_src_biogeophys",
    "clm_src_cpl",
    "clm_src_main",
    "clm_src_utils",
    "multilayer_canopy",
    "offline_driver",
    "offline_executable",
]

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
