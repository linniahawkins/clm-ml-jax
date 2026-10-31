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

## Next (resume here)

**Package 7.1 done (2026-10-30, last commit `bdde552`); reported to the maintainer, who asked
to start 7.2 in the next session.** Open review points they may still raise: C4 (bit for bit
holds on CPU, so the reference was not re-recorded and option 2, the noise twin as the only
gate, was not needed), the 133 removed diagnostics (`phase7_removed_diagnostics.md`), C5-C8.

### Package 7.2: physics by process into `clm_ml/physics/` (scope §4)

One commit per process in the scope's table order, each **bit for bit on CPU**; 2-3 processes
per session (How to work, above). Each process gets: variants registered by name (plan §6.6),
a docstring with `Fortran reference:` and its reads/writes, a test that the declared writes
are what changes, and its golden tests repointed (`tests/fortran_validation/test_golden_jax.py`
imports routines by module name; never edit the `golden_IO/*.json`).

**First session: processes 1-2.**
1. `clm_ml/physics/math.py` from `multilayer_canopy/MLMathToolsMod.py` (root finders
   `hybrid`/`zbrent`/`bisection`, `quadratic`, `tridiag`, `tridiag_2eq`, `tridiag_scan`,
   beta/gamma functions; 12 source files import it). Scope: `tridiag` as a scan (removes 48
   unrolled ops). `tridiag_2eq` has a **hand-written gradient** (`custom_vjp`), so this commit
   also runs `python tests/gradient_tier.py --focused`. Golden: `test_MLMathToolsMod_*` (7).
   Before deciding how the old module names keep working during the move (re-export shim
   vs. updating every import), log the choice as a C decision.
2. `clm_ml/physics/soil.py` (clmData, SoilAlbedo, SurfaceResistance, SoilThermProp,
   SoilWater, SoilTemperature): array operations instead of ~500 scalar updates (D5);
   `SoilThermProp` computed once only if its two calls see the same inputs (check first,
   with a probe).

Before the first 7.2 commit, take the **timing baseline** at `bdde552` (already recorded:
`p71-clm-ml-jax-*` in `timing/results_model.jsonl`, forward ~27-28 ms/step, gradient
~604 ms/step) and compare at the end of 7.2 (discuss > 1.5x, scope risk table).

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
- **Two layouts until step 10 of 7.2:** the physics still runs on the legacy containers
  (1-based, patch axis of length 1); the step's input/output is the 0-based `StepState`
  (`clm_ml/state.py`), converted in `clm_ml/legacy/state.py`; outputs read through
  `clm_ml/legacy/view.py`. When a moved process changes what the step returns, change the
  view, never the reference. The legacy container leaves the step at process 10
  (`canopy.py`).
- **Known quirk:** the legacy `.out` writer reads soil level 1 of a column axis of length 1;
  it works only because JAX clamps out-of-range indices (C8). Don't convert the view to numpy.
- **Never** scatter into a scan carry with array indices, add `jax.debug.callback`, or set the
  gradient XLA flags for the regression tests (CLAUDE.md gotchas).
