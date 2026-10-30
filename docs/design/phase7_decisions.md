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

### C4. The bit-for-bit test stays a hard gate; tests and the reference run on CPU (7.1, 2026-10-30)
Context: 7.0 found the code no longer bit for bit with the committed reference (75 of 88
fields, round-off) and blamed the compilation cache. The maintainer chose to re-record the
reference with a fresh cache, then (when fresh compiles turned out to differ from each other)
option 2: start 7.1 with the noise twin as the gate and bit for bit advisory.
Decision: neither. The Derecho login nodes (derecho3, derecho7) have two A100s and JAX uses
them by default, so the 7.0 probes and the first probes of this session ran on the **GPU**.
Probes (`/glade/derecho/scratch/linnia/clm-ml-jax/probe7b`, `probe7c`): on the GPU, about one
fresh compile in five gives a different program (54 fields differ from step 0 by ~1e-11, up to
6e-10 relative; `--xla_gpu_deterministic_ops` does not change it; runs from one cache are
reproducible; Python hash seeds play no part). **On the CPU (`JAX_PLATFORMS=cpu`) two fresh
compiles are identical to each other and to the committed reference on all 88 fields.** So the
reference stays as it is, the bit-for-bit test remains the per-commit gate, and every test and
recording runs on CPU: `tests/conftest.py` sets `JAX_PLATFORMS=cpu` unless chosen (that file
appeared untracked in the working tree on 2026-10-30, not written by Claude; committed as it
is), `record_run.py --update-reference` refuses another platform, and `jobs/grad_gpu.pbs` now
sets `JAX_PLATFORMS=cuda` (with the conftest it would otherwise have tested the CPU).
Why: stronger than option 2 at no cost, and it is the safety net the scope assumes. All GPU
variants pass the noise-twin test (worst 63 % of its allowance), so GPU results stay checkable
with the noise twin.
Alternatives: option 2 as chosen (weaker for no reason once the cause was found); running
the tests on a fresh compilation cache (adds ~1 min per commit, unnecessary on CPU).
Open: the GPU's compile-to-compile variation (diagnosis not finished: the ~412 small setup
programs get different cache keys in every fresh process, on CPU too, without changing CPU
results); matters for GPU reproducibility, not for Phase 7.

### C5. The carried state: 0-based, no patch axis, adapters by blocks (7.1, 2026-10-30)
Context: scope 7.1 asks for the step state as 0-based arrays without the patch axis, with
adapters to the legacy containers while the physics still uses them.
Decision: `clm_ml.state.State` keeps its structure (`canopy`, `soil`; plan §6.7 keeps soil per
column for option B) and names, with new shapes: layer fields `(layer,)` (legacy index
`i + 1`), leaf fields `(layer, leaf)` (leaf 0 sunlit, 1 shaded), scalars `()`, soil
temperature `(soil_level,)` (legacy `t_soisno_col[1 : nlevgrnd + 1]`). Batched runs add one
leading patch axis (`(patch, ...)` instead of `(patch, 1, ...)`); `batched_state(template)`
builds it. The adapters (`clm_ml/legacy/state.py`) describe each field as a block of its legacy
array (`LAYOUTS`: first index per axis after the patch axis); reading slices the block, writing
uses `lax.dynamic_update_slice` with constant starts into the template. The entries outside
the blocks (index 0 of the layer and leaf axes, unused soil levels) come from the template:
the reference shows them constant fill values (or 0) after the first step, and the
bit-for-bit test passes (all 88 fields), so they carry nothing from step to step. An axis
`level` (`max_layers + 1`, 0 = the ground) is used for the five radiation profiles whose
legacy index 0 is the ground (it varies in time) and for `zw` (layer tops above the ground).
Restart format version 4 (the old pickles have the old shapes).
Why: the scope's layout (D1) with the least change at the boundary; whole-array copies of
fill entries would keep legacy structure in the state.
Alternatives: scatter writes (`.at[...].set`): avoided because scatters into the scan carry
misbehaved on CPU (numerics doc §6); a separate `ground` field for the radiation profiles:
two fields per flux instead of one axis, and the legacy view would need to join them anyway.

### C6. `StepState` = state + diagnostics in 7.1; diagnostics keep the legacy names (7.1, 2026-10-30)
Context: scope 7.1 lists the step state as prognostic state, within-step carry, inputs and
diagnostics, with diagnostics cut to what outputs and checks use (D6).
Decision: `StepState(state, diagnostics)`. `Diagnostics` holds exactly the reference fields
that are not prognostic or layer grid (80 fields: every one is read by an output variable,
the `.out` writer or both; checked against `clm_ml/outputs.py`) plus `balance_error` (the
checks). The within-step carry (`*_bef`, RK increments) and the step's inputs join when the
physics that uses them moves (7.2): until then the step runs on the containers and nothing
would read them. Diagnostic names stay the legacy names (`lhflx_canopy`, ...) until the
outputs are defined on the new state (7.6), with 0-based shapes; the five radiation profiles
use the `level` axis (C5). `PatchLayers` holds the counts as `n_layers` (ncan), `n_within`
(ntop: a count of layers 0..n_within-1) and `first_leafy` (nbot - 1, 0-based), with masks as
properties. Every field no longer returned is in `phase7_removed_diagnostics.md` (134: the
other 133 `mlcanopy` fields and `t_ref2m_patch`), with the modules that set and read it,
generated by a source search (`probe7c/make_removed.py`); the approximate "read by" column
tells 7.2 which are work values.
Why: one rename instead of two (7.2 processes declare reads/writes in these names; 7.6
renames once with the outputs); fields with no reader would be speculative structure.
Alternatives: readable diagnostic names now (80 names to choose twice, since the outputs keep
the legacy names until 7.6); defining carry/inputs types now (unused and untestable in 7.1).

### C7. Everything after a step reads its `StepState` through the legacy view (7.1, 2026-10-30)
Context: scope 7.1: the legacy view maps the new state to the reference names; the legacy
`.out` writer goes through the view (or is retired) first.
Decision: `legacy_view(template, step, layers)` puts the `StepState` and `PatchLayers` into
the template's containers (`instances_with_step_state`) and returns the 88 reference fields
(`view.FIELDS`); `view_of_instances` (first step, output shapes) and `view_series` (a chunk's
steps) wrap it. In compiled chunks the step returns `(state, output variables, errors)` with
the variables computed from the view of its `StepState`; `make_chunk_runner` returns each
step's `StepState` (not containers); the legacy `.out` writer (`main()`) gets the view's
fields as attribute namespaces, so `CLMml_driver.output` is unchanged. Validation streams can
name only the view's fields.
Why: the bit-for-bit test then checks the step state, the adapters and the view at every step
(it passes on all 88 fields, as do the `.out` files and the stream-runner test). The writer is
kept (not retired) because the Fortran comparison needs it until 7.6.
Alternatives: retire `clm-ml-offline` now and write `.out` from netCDF (scope puts it in 7.6).

### C8. Independent review of 7.1: fixes (7.1, 2026-10-30)
Context: the reviewer agent (diff only) found two defects outside the per-commit tests and
several risks.
Decision:
- `td_profile` (diffuse transmittance per layer) is kept as a diagnostic: the issue-14 test
  (`test_thin_layers_keep_minimum_plant_area`) checks it, and D6 keeps what checks use. The
  view now gives every state field under its legacy name as well (`tleaf_leaf`, `cair_profile`,
  `lwp_leaf`, `h2ocan_profile`, `tg_soil`, `t_a10_patch`, which that test and validation may
  name), and `clm-ml check` accepts only the view's fields as raw stream variables (a typo or
  a removed field fails at check time, not after setup).
- `put` checks the block's shape: `dynamic_update_slice` silently moves a block that does not
  fit (e.g. a `level` array given for a `layer` field in 7.2).
- `timing/time_global.py` builds the batched state with `batched_state`.
- New fast test `tests/clm_ml/test_legacy_adapters.py` (fake containers, ~2 s): 0-based shapes,
  blocks at the right legacy entries, the round trip containers -> `StepState` -> containers,
  template entries kept outside the blocks, `batched_state`, the shape check, the layer masks,
  and that the view covers every reference field. Old-layout fixtures in `test_acclimation`
  and `test_restart_utils` updated.
- The `.out` writer keeps JAX arrays (reviewer's nit: numpy once per chunk): `CLMml_driver.output` reads `t_soisno_col[1, ic]` (Fortran's column 1) on a column axis of length 1, which JAX clamps to column 0 and numpy rejects; the writer goes in 7.6.
- A batched 3-patch run (C3 and C4 groups, different layer counts) compared bit for bit with
  the 7.0 code (`probe7d/mixed.py`): the carried state's dropped fill entries were checked
  only by the one-patch reference before.
Not done: an explicit field -> layout table in place of the name-based rule (reviewer's
readability nit); the rule covers 5 exceptions and the new test exercises every field.
Why: the defects were real failures (KeyError in the issue-14 test; a crash in the global
timing test); the guards make 7.2's moves fail loudly instead of silently.
