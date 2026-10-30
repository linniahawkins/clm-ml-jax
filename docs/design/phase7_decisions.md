# Phase 7 decision log

Decisions Claude makes while doing the Phase 7 work packages, kept for posterity: anything the
scope (`phase7_scope.md`) and the maintainer's decisions (plan §0, decision 35; scope D1-D12)
do not already settle. Examples: names and file layout, how an adapter works, the order of a
move, a test skipped or added, a deviation from the scope and why, an option considered and
rejected. Science choices are not made here: they go to the maintainer (and plan §0).

Add an entry in the same commit as the change. Number entries in order (C1, C2, ...), never
renumber; if a later decision replaces one, say so in both.

Format:

```
### C<n>. <short title> (<package>, <date>, <commit>)
Context: what needed deciding.
Decision: what was chosen.
Why: the reason (measured where possible).
Alternatives: what was considered and why not.
```

## Decisions

### C0. Minimum testing for Phase 7 (setup, 2026-10-30)
Context: the maintainer asked to cut testing to the bare minimum to speed up Phase 7.
Decision: per commit only lint, the bit-for-bit + noise-twin test and the moved routines'
golden tests; a focused CPU gradient test when a routine with a hand-written gradient moves;
the CPU gradient tier and touched feature tests at package ends; everything else, including
one GPU gradient job, at the end of Phase 7 (`phase7_progress.md`). Replaces D12's "GPU
gradient job at package ends".
Why: Phase 7 does not change arithmetic, and the bit-for-bit test compares every field after
every step; gradients can change without the forward values changing only where gradients
are hand-written (implicit-function solvers, `stop_gradient`), so those moves get a gradient
check.
Alternatives: the Phase 6 tiers (fast tier after each edit, feature tests per commit, GPU at
package ends): safer, slower; GPU at package ends dropped because the CPU tier with
optimization off checks the same gradients without the queue.

### C1. Dead code removed whole, including `controlMod` and the CIME duplicates (7.0, 2026-10-30)
Context: scope §4 asks to delete the dead modules of §2 and to "remove `controlMod` writes and
`clm_instRest`".
Decision: deleted `offline_driver/controlMod.py` as a whole (not only the writes to it), the
`clm_instRest` function, all of `cime_src_share_util/` (including its `shr_orb_mod.py`, a
duplicate of `clm_share.shr_orb_mod`, which stays) and `utils/` (`gpu_config.py`), plus the
modules listed in §2.
Why: nothing calls `controlMod.control()` (the namelist is read by `offline_executable/main.py`)
and nothing reads its three module variables; `clm_instRest` had only commented-out calls.
Checked with a search of `src`, `tests` and `timing` for imports and attribute uses.
Alternatives: keep `controlMod` as an unused Fortran mirror: no use, and it imports legacy
module globals.

### C2. The legacy view lives in `src/clm_ml/legacy/view.py` and feeds the outputs too (7.0, 2026-10-30)
Context: the legacy-view adapter (scope §3.1) maps the state to the reference's names and
1-based shapes; where it lives and who uses it.
Decision: `legacy_view(insts) -> {"mlcanopy.<f>": ..., "temperature.<f>": ...}` (every field of
the two containers, legacy layout) in `clm_ml/legacy/view.py`, plus `raw_fields` (the
`mlcanopy` fields with a time axis, as `outputs._Raw` takes them). The three places in
`offline_executable/main.py` that read raw fields from the containers (`step_variables` inside
the compiled chunk, `output_shapes`, `step_values` for the first step) now read them only
through the view. Its test is the bit-for-bit test (the recorded `fields` stream reads every
reference field through it; a missing field fails the run) plus
`test_legacy_view_gives_reference_fields` (the recorded field list equals the reference's).
Not yet through the view: the legacy `.out` writer (`write_output`, fed the containers by the
chunk runner); 7.1 must route it through the view or retire it first.
Why: in 7.1 the containers are replaced by the step state; with one function between state
and every output and reference field, only the view changes and the bit-for-bit test checks it
at every step. It is in `src/` (not `tests/`) because the output variables use it in runs.
Alternatives: a test-only adapter (scope's wording): would leave the output variables reading
the containers, so they would need a second adapter in 7.1. A separate unit test on containers
built without a run: needs a legacy setup (a run), so it would duplicate the end-to-end test.

### C3. Gradient tier: `gradient` marker, `tests/gradient_tier.py`, a guard, the GPU job in the repo (7.0, 2026-10-30)
Context: scope 7.0 asks for a pytest marker or script for the CPU gradient tier and the GPU
gradient job with a warm cache.
Decision: marker `gradient` on the 8 tests of the CPU tier (the Model chunk gradient test and
`test_differentiability.py`); `python tests/gradient_tier.py [--focused]` runs them with
`JAX_PLATFORMS=cpu` and the optimization-off `XLA_FLAGS`, naming the two files (not collecting
all of `tests/`, whose other modules set legacy globals at import). `tests/regression/conftest.py`
fails the regression tests, and `record_run.py --update-reference` refuses to record, if that
flag is set (it changes round-off). The GPU job is now
`jobs/grad_gpu.pbs` in the repo (paths overridable by environment; the same 10 tests as the
scratch job, selected by marker; walltime 1.5 h; fails loudly if the worktree can't be made), with its persistent GPU compilation cache in scratch
(`jax_compile_cache_gpu`, kept apart from the CPU cache).
Why: one command, no flag to remember, and no way to run the bit-for-bit test with the wrong
code generation by accident. The job was only in scratch.
Alternatives: a `conftest` hook setting `XLA_FLAGS` for marked tests: too late (JAX reads the
flag at backend start, and the tests' subprocesses inherit the environment).
