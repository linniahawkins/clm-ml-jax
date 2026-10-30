"""Run the gradient tier (tests marked ``gradient``) on CPU, with the flag that lets CPU gradients
run.

Issue 16 (JAX 0.9.2): compiled CPU gradients of a CHATS7 chunk segfault in LLVM-optimized code;
with LLVM optimization off they run. The flag changes round-off, so it is set here for the
gradient tests only (the regression tests refuse to run with it). See
``docs/design/phase7_progress.md`` ("Testing in Phase 7")::

    python tests/gradient_tier.py            # all gradient tests (8, ~10 min on a login node)
    python tests/gradient_tier.py --focused  # the chunk gradient test only (~5 min), after
                                             # moving a routine with a hand-written gradient

Extra arguments go to pytest. On GPU, run the same tests without this script
(``jobs/grad_gpu.pbs``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_GRADIENT_XLA_FLAGS = (
    "--xla_cpu_parallel_codegen_split_count=1 --xla_backend_optimization_level=0"
)
FOCUSED = "tests/test_model_api.py::test_model_chunk_gradient_matches_finite_difference"
# Files holding the tests marked ``gradient`` (named, not collected from all of tests/: other
# modules set legacy module globals at import)
GRADIENT_FILES = ["tests/test_model_api.py", "tests/test_differentiability.py"]


def main(argv: list[str]) -> int:
    focused = "--focused" in argv
    extra = [a for a in argv if a != "--focused"]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = " ".join(filter(None, [env.get("XLA_FLAGS", ""), CPU_GRADIENT_XLA_FLAGS]))
    targets = [FOCUSED] if focused else GRADIENT_FILES
    cmd = [sys.executable, "-m", "pytest", *targets, "-m", "gradient"]
    cmd += ["-p", "no:cacheprovider", "-o", "addopts=", "-q", *extra]
    print("XLA_FLAGS=" + env["XLA_FLAGS"], " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
