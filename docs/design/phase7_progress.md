# Phase 7 progress log

Branch `phase7/restructure` (from `phase6/patches` at the end of Phase 6b). Scope, work packages
and the maintainer's decisions (D1-D12): `docs/design/phase7_scope.md`. Decisions Claude makes
while working: `docs/design/phase7_decisions.md`. Fields removed from the state (D6):
`docs/design/phase7_removed_diagnostics.md` (created in 7.1).

One entry per commit: what changed, which tests ran and their result, any round-off change and
its reference update. Each session ends at a commit with "Next" updated.

## How to work (agreed with the maintainer, 2026-10-30)

- **One session per work package** (7.2 split into sessions of 2-3 processes). Start from
  "Next" below; end at a commit with this log and "Next" updated. Don't rely on compaction to
  carry a package across sessions.
- **Stop at the end of each package** and report to the maintainer (tests, timing, round-off
  changes, decisions logged) before the next one. Within a package, commit and push when the
  tests below pass; stop only for science questions or a slowdown above ~1.5x.
- **Independent review** (a reviewer agent that sees the diff, not Claude's reasoning) before
  the last commit of each package.
- **Log every decision** you make that the scope does not already settle (naming, layout,
  an adapter, a test skipped, a deviation) in `phase7_decisions.md`, in the same commit.

## Testing in Phase 7 (minimum, maintainer 2026-10-30)

Phase 7 changes structure, not arithmetic, and the bit-for-bit CHATS7 test compares every
remaining state and output field after every step. So the testing is cut to:

| When | Tests | Time |
|---|---|---|
| Each commit | `black`/`ruff`; bit-for-bit + noise twin (`tests/regression/test_end_to_end_regression.py -m regression`, one run); the Fortran golden tests of the routines moved in that commit; `tests/clm_ml` only if `src/clm_ml/` config/IO code changed | ~1-2 min |
| A commit that moves a routine with a hand-written gradient (implicit-function solvers: Obukhov length, stomatal conductance bisection, any `stop_gradient` construction) | plus `test_model_api.py::test_model_chunk_gradient_matches_finite_difference` on CPU with optimization off (command below) | ~5 min |
| End of a package | the CPU gradient tier (below), and the feature tests the package touched (7.3: `test_coarse_forcing`, `test_land_data_files`, `test_grid_setup`, `test_stream_runner`; 7.4: `test_vector_setup`, `test_grid_setup`, `test_patches_batched`, `test_c3_c4_groups`, `test_sites_one_program`; 7.5: `test_restart`) | ~10-20 min |
| End of Phase 7 | all of `tests/regression`, the slow tests, the Fortran comparison, and **one** GPU gradient job (`jobs/grad_gpu.pbs`, develop queue) | ~1.5 h |

Not run per commit any more: the fast tier after every edit, `test_clm_ml_equivalence`, the
other feature tests, GPU jobs. Accepted risk: a break in a feature not tested until the package
end is found later; commits stay small (one process each) so `git bisect` finds it quickly.

CPU gradient tier (issue 16: the default CPU build crashes running gradients; with LLVM
optimization off it runs; round-off differs, so never use this flag for the regression tests):

```bash
python tests/gradient_tier.py              # the 8 tests marked `gradient`, ~11 min (sets the flags)
python tests/gradient_tier.py --focused    # the chunk gradient test only, after a hand-written gradient moves
qsub -v REV=<commit>,TAG=<name> jobs/grad_gpu.pbs   # GPU gradient job (develop queue, warm GPU cache)
```

## Log

### 7.0 Preparation (2026-10-30)

- **Dead code removed (C1):** `cime_src_share_util/`, `utils/`, `lnd_comp_mct`, `histFileMod`,
  `restUtilMod`, `fileutils`, `clm_share.shr_file_mod`/`shr_kind_mod`, CLM's
  `SoilTemperatureMod`, `WaterFluxType`, `WaterStateType`, `offline_driver/CLMml.py`, the whole
  of `controlMod` (its writes in `main.py` and `test_differentiability.py`) and `clm_instRest`.
- **Legacy view (C2):** `clm_ml/legacy/view.py`; the output variables and validation streams read
  the state only through it (the legacy `.out` writer not yet). New test
  `test_legacy_view_gives_reference_fields`.
- **Gradient tier (C3):** marker `gradient`, `tests/gradient_tier.py`, guards against the
  optimization-off flag in the regression tests and `record_run.py`, `jobs/grad_gpu.pbs` in the
  repo. CLAUDE.md gotchas updated (index layout, options copied at import, the view).
- **Tests:** lint; `tests/clm_ml` 129 pass (11.8 min on a loaded login node); fast Fortran golden
  160 pass (4.2 min); CPU gradient tier 8 pass (11.3 min); noise twin 88 pass. Independent
  review: no defects; nits fixed.
- **Bit for bit: this package is identical to HEAD `aa61279` on all 88 fields** (probe: same
  50-step run with each tree, same cache). **But HEAD itself is not bit for bit with the
  committed reference** (75 of 88 fields; relative differences up to ~1e-9 in
  `shair_profile`, 7e-16 in `pref_forcing`; noise twin passes). Probes
  (`/glade/derecho/scratch/linnia/clm-ml-jax/probe7/`): not this package (HEAD fails the same
  way); not the calendar commit `b45bd9a` (`15fbd08`, before it, gives values identical to
  HEAD); not `legacy_ascii` (identical with and without); runs are reproducible. A fresh
  compile (empty compilation cache) differs from cached programs in one field
  (`t_soisno_col`, 1.7e-13 K, soil level 6). So the reference (recorded at `607fe1f` + the
  `3dcea3c` change) was made by compiled programs that today's compilation does not reproduce;
  the bit-for-bit result depends on the compilation-cache state. **Reference not re-recorded:
  the maintainer's decision** (see Next).

### 7.1 The step state (started 2026-10-30)

- **Safety net (C4):** the 7.0 bit-for-bit failure was the login node's GPUs (JAX's default
  there), not the compilation cache. On CPU, fresh compiles are identical to each other and to
  the committed reference (all 88 fields); on the GPU about one fresh compile in five differs
  (~1e-11, within the noise twins). Reference **not** re-recorded; bit for bit stays the gate.
  `tests/conftest.py` (CPU unless chosen), `record_run.py` refuses another platform,
  `jobs/grad_gpu.pbs` selects `cuda`. Tests: lint; bit-for-bit + noise twin 177 pass (1.5 min).
- **Carried state 0-based (C5):** `clm_ml.state.State` without the patch axis, 0-based;
  adapters by blocks in `clm_ml/legacy/state.py` (`batched_state` for batched templates);
  restart version 4. `clm_ml/state.py` also defines `Diagnostics`, `StepState` and
  `PatchLayers` (used from the next commit). Tests: lint; bit-for-bit + noise twin 177 pass.
- **Step results as `StepState` (C6, C7):** the step returns `StepState(state, diagnostics)`;
  output variables, validation streams, the first step and the legacy `.out` writer read it
  only through the legacy view (`legacy_view(template, step, layers)`). Removed fields (D6):
  `phase7_removed_diagnostics.md` (134; 133 after C8). Tests: lint; bit-for-bit + noise twin +
  `test_clm_ml_equivalence` + `test_stream_runner` 180 pass.
- **Independent review fixes (C8):** `td_profile` kept (issue-14 check); the view gives all
  state fields too; `clm-ml check` validates raw stream names against the view; `put` checks
  block shapes; `time_global.py` fixed; new `tests/clm_ml/test_legacy_adapters.py`. Tests: lint;
  bit-for-bit + noise twin + equivalence + thin-layer (issue 14) 180 pass; adapter, acclimation
  and restart-utility unit tests pass.

- **Mixed C3/C4 batch (risk 5):** a 3-patch batched run (C3 and C4 groups) at 7.0 and at
  7.1: all 68 recorded arrays identical; `anet_leaf` differs only in fill values (>= 1e30) of
  unused slots, every real value identical.
- **Package end (2026-10-30, CPU):** gradient tier 8 pass (5.7 min); all of
  `tests/regression -m regression` 207 pass (53 min, fresh compiles); `tests/clm_ml` 137 pass;
  Fortran golden (not slow) 160 pass. Timing (`time_model.py --steps 8`, CPU, gradient XLA
  flags, 7.0 and 7.1 alternated twice; `timing/results_model.jsonl`, labels `p71-*`):
  forward 24.1-24.8 -> 26.8-28.4 ms/step (~1.1x, the adapters), gradient 585-606 -> 604
  ms/step (no change), peak memory 4.4-4.5 -> 4.3-4.6 GB. **Package 7.1 done.**

### 7.2 Physics by process (started 2026-10-30)

- **Process 1, `clm_ml/physics/math.py` (C9):** from `MLMathToolsMod.py` (deleted; the 10
  importing modules and the golden tests import the new module, no shim). `tridiag` 0-based
  as a `lax.scan` (replaces the unrolled 1-based `tridiag` and `tridiag_scan`; the pivot
  guard as `min_pivot`, passed by the solar radiation and soil temperature solves). Unused
  container-threading root finders not moved. Found: `wue_solver: brent` is never used (open
  issue 18 in `global_input.md`, for process 7). New `tests/clm_ml/test_physics_math.py`.
  Tests: lint; bit-for-bit + noise twin 177 pass (2.0 min); golden `MLMathToolsMod` 41 pass;
  new unit tests and `test_imports` pass; focused CPU gradient test (`tridiag_2eq`'s
  `custom_vjp` moved) 1 pass (3.2 min).
- **Process 2, `clm_ml/physics/soil.py` (C10): round-off change, reference re-recorded.**
  The six soil routines as array functions on one column (0-based levels); legacy adapters
  with the old names and `DECLARED` reads/writes in `clm_ml/legacy/soil.py` (`clm_driver`
  changes only its imports); `SoilWaterMovementMod`, `SurfaceResistanceMod`,
  `MLSoilTemperatureMod` deleted. Not bit for bit end to end (55 fields, worst 1.65x the twin
  spread) although every routine is bit-identical in isolation: XLA compiles the chunk
  differently (probes in `probe72`, C10). Reference and twins re-recorded (D7). New
  `tests/clm_ml/test_physics_soil.py` (declared reads/writes per adapter). Tests: lint;
  before re-recording: noise twin 88 pass, bit for bit 55 fail; after: bit-for-bit + noise
  twin 177 pass (1.2 min); `test_imports`, new unit tests pass.
- **Soil thermal properties once per step (C11):** `SoilTemperature` uses the `tk`, `cv` of
  `clm_drv`'s `SoilThermProp` call (same inputs, checked in the code). Tests: lint;
  bit-for-bit + noise twin 177 pass (1.6 min); soil unit tests pass.

## Next (resume here)

**Package 7.2 in progress. Session 1 (2026-10-30) did processes 1-2; last commit = the one
that wrote this.** Next session: **processes 3-4** (scope table):
3. `clm_ml/physics/atm_forcing.py` (`MLGetAtmForcingMod`: GetAtmForcing, TimeInterpolation3),
   ready for 7.3. Process-replay routine `GetAtmForcing` (`tests/regression/process_io.py`).
4. `clm_ml/physics/solar_radiation.py` (two-stream; Norman registered as unsupported, D8): the
   150-term sum as a scan in the same order. Process replay `SolarRadiation`.

Same pattern as process 2 (C10): physics on plain 0-based arrays in `clm_ml/physics/`, legacy
adapters with the old names and a `DECLARED` reads/writes table in `clm_ml/legacy/`, a test on
fake containers holding only the declared fields. Expect round-off changes when loops become
array operations even if every routine is bit-identical in isolation (C10): probe the routine
alone against the old code first (`probe72/compare.py` pattern, old files from git), then, if
only the whole-chunk compile differs and the noise twin passes, re-record the reference in the
same commit (D7) and log it. Constants through `clm_ml/physics/constants.py`.

Open for the maintainer: C10 (reference re-recorded for the soil move, round-off only), open
issue 18 (`wue_solver: brent` never used; decide at process 7). Earlier: C4, the removed
diagnostics, C5-C8. Timing baseline for the end of 7.2: `p71-*` in
`timing/results_model.jsonl` (forward ~27-28 ms/step, gradient ~604 ms/step).

### Practical notes for the session (learned in 7.0-7.1)

- **Environment:** `export PATH=/glade/work/linnia/conda-envs/clm-ml-jax/bin:$PATH` (the
  system Python has no numpy; black, ruff, pytest live there).
- **CPU only:** login nodes have two A100s and JAX picks them by default. `tests/conftest.py`
  forces the CPU for pytest; every probe script must set `JAX_PLATFORMS=cpu` itself. On the
  GPU, fresh compiles differ by ~1e-11 (C4; open issue outside Phase 7).
- **Per-commit command** (~1.5-3 min with a warm cache, ~10 min cold):
  `python -m pytest tests/regression/test_end_to_end_regression.py -m regression -q -p no:cacheprovider -o addopts=""`
  plus the moved routines' golden tests and `black --check --line-length 100 src tests timing && ruff check src tests timing`.
- **Package end** (7.1 took): gradient tier 6 min, all `tests/regression` 53 min,
  `tests/clm_ml` 0.5 min, golden 5 s. Script to copy:
  `/glade/derecho/scratch/linnia/clm-ml-jax/logs7/package_end.sh` (sequential, logs in the
  same directory). Timing before/after: `logs7/timing.sh` (a `git worktree` of the old commit
  in scratch; `time_model.py` imports its own tree's `src`).
- **Committing docs:** `docs/` is matched by an ignore rule: use `git add -u docs` for tracked
  files and `git add -f` for new ones, and check `git status` after committing.
- **Two layouts until process 10 of 7.2:** the physics still runs on the legacy containers
  (1-based, patch axis of length 1); the step's input/output is the 0-based `StepState`
  (`clm_ml/state.py`), converted in `clm_ml/legacy/state.py`; outputs read through
  `clm_ml/legacy/view.py`. When a moved process changes what the step returns, change the
  view, never the reference. The legacy container leaves the step at process 10
  (`canopy.py`).
- **Known quirk:** the legacy `.out` writer reads soil level 1 of a column axis of length 1;
  it works only because JAX clamps out-of-range indices (C8). Don't convert the view to numpy.
- **Never** scatter into a scan carry with array indices, add `jax.debug.callback`, or set the
  gradient XLA flags for the regression tests (CLAUDE.md gotchas).
