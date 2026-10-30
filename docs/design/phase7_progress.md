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
  `phase7_removed_diagnostics.md` (134). Tests: lint; bit-for-bit + noise twin +
  `test_clm_ml_equivalence` + `test_stream_runner` 180 pass.
- **Independent review fixes (C8):** `td_profile` kept (issue-14 check); the view gives all
  state fields too; `clm-ml check` validates raw stream names against the view; `put` checks
  block shapes; `time_global.py` fixed; new `tests/clm_ml/test_legacy_adapters.py`. Tests: lint;
  bit-for-bit + noise twin + equivalence + thin-layer (issue 14) 180 pass; adapter, acclimation
  and restart-utility unit tests pass.

## Next (resume here)

**Package 7.0 done; waiting for the maintainer.** Decide first, before 7.1:
1. **The bit-for-bit safety net** (log 7.0): the committed reference no longer matches the code
   bit for bit (round-off only). Proposal: re-record the reference at the 7.0 commit with an
   empty compilation cache, and make the bit-for-bit test (and `record_run.py`) always use a
   fresh, test-only cache directory, so the compared programs never depend on what other runs
   left in the shared cache; check that two recordings in separate processes are identical.
2. Then **package 7.1 (the step state)**, scope §4: `StepState` (0-based, no patch axis),
   boundary adapters, diagnostics cut (D6) with `phase7_removed_diagnostics.md`; the legacy
   view maps the new state to the reference names; route the legacy `.out` writer through the
   view (or retire it) first.
