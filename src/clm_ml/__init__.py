"""CLM-ML-JAX: multilayer canopy model (CLM-ml v2) in JAX.

See docs/design/modernization_plan.md. Public API is re-exported here as it is built.
"""

import os as _os

# XLA's CPU compiler splits a program into pieces compiled in parallel and links them with
# LLVM's JIT; for the large batched programs (e.g. C3 and C4 patch groups in one chunk) the
# linker's recursion overflowed a compile thread's stack: an intermittent segmentation fault
# during compilation (JAX 0.9.2 on Derecho, Phase 6; backtrace in llvm::orc
# RTDyldObjectLinkingLayer::emit). One piece avoids it (5 of 5 compilations clean; without it
# about half of the compilations of these programs crashed). XLA reads the flag when it
# starts, so it is set on import, unless the user's XLA_FLAGS already choose it.
_SPLIT = "--xla_cpu_parallel_codegen_split_count=1"
if "xla_cpu_parallel_codegen_split_count" not in _os.environ.get("XLA_FLAGS", ""):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "") + " " + _SPLIT).strip()
