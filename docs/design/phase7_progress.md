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
| End of a package | the CPU gradient tier (below), and the feature tests the package touched (7.3: `test_coarse_forcing`, `test_land_data_files`, `test_grid_setup`, `test_stream_runner`, `test_patches_batched`; 7.4: `test_vector_setup`, `test_grid_setup`, `test_patches_batched`, `test_c3_c4_groups`, `test_sites_one_program`; 7.5: `test_restart`) | ~10-20 min |
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

- **Process 3, `clm_ml/physics/atm_forcing.py` (C12):** `atm_forcing` on one patch (0-based
  bands), time interpolation registered by name (`none`, `three_point`; two-point dropped:
  Fortran stops for it); `clm_forcing_to_ml` moved there; legacy adapter `GetAtmForcing` and
  the legacy-layout forcing window in `clm_ml/legacy/atm_forcing.py`; `MLGetAtmForcingMod`
  deleted. Bit for bit. New `tests/clm_ml/test_physics_atm_forcing.py`. Tests: lint;
  bit-for-bit + noise twin 177 pass (1.6 min); process replay 7 pass; new unit tests and
  `test_imports` pass.

- **Process 4, `clm_ml/physics/solar_radiation.py` (C13):** `canopy_optics`, `two_stream`
  (the 150-term absorption sum as a `lax.scan`, same order), `absorbed_par` on one patch
  (0-based `PatchLayers`, bands, leaves); `SOLAR_RADIATION` registry with `norman` refused
  (D8), the legacy Norman translation kept unchanged in the adapter
  `clm_ml/legacy/solar_radiation.py`; `clm_ml/physics/layers.py` (mask helpers, 0-based);
  `legacy.state.mlcanopy_layers`; `MLSolarRadiationMod` deleted. Bit for bit. New
  `tests/clm_ml/test_physics_solar_radiation.py`. Tests: lint; bit-for-bit + noise twin 177
  pass (1.5 min); process replay 7 pass; new unit tests, `test_imports`,
  `test_legacy_adapters` pass.

- **Process 5, `longwave.py`, `canopy_water.py`, `leaf_heat_capacity.py`,
  `plant_hydraulics.py` (C14): round-off change, reference re-recorded.** Physics on one
  patch (0-based layers, leaves, soil levels); leaf water potential of sun and shade in one
  call over the leaf axis (`MLCanopyFluxesMod` calls it once); adapters with the old names
  and `DECLARED` reads/writes in `clm_ml/legacy/{longwave,canopy_water,plant_hydraulics}.py`;
  the four legacy modules deleted. Every routine bit-identical under `jit` in isolation; the
  chunk differs in 53 fields, worst 1.28x the twin spread (C10 pattern). Found: soil ice read
  one level too deep in `SoilResistance` (no effect while ice is 0; open issue 19, kept).
  Golden tests `WettedFraction`, `LeafHeatCapacity` repointed to the physics. New
  `tests/clm_ml/test_physics_longwave.py`, `test_physics_canopy_water_hydraulics.py`.
  Tests: lint; repointed golden 11 pass; new unit tests 16 pass, `test_imports` pass; before
  re-recording: noise twin and process replay pass, bit for bit 53 fail; after: bit-for-bit
  + noise twin 177 pass (1.1 min).

- **Process 6, `clm_ml/physics/turbulence.py` (C15):** the Harman-Finnigan RSL turbulence
  on one patch (0-based layers), the psihat tables an argument (`RSLTables`; file moved to
  `src/clm_ml/data/psihat.nc`, read by `clm_ml/io/rsl_tables.py`; the `MLclm_varcon` arrays,
  module caches, `rslfile` and `LookupPsihatINI` gone), still a constant of the compiled
  chunk until process 10 (maintainer: fix there, "Fixes to make at process 10" below). Canopy-top values rounded as
  stored (`optimization_barrier`: without it XLA folded `ustar / beta`, 1 ulp off in 2 of 6
  recorded calls; probe `probe74/compare.py`). Ground conductance by name (other
  `HF_extension_type` values now refused). Adapter `clm_ml/legacy/turbulence.py`;
  `MLCanopyTurbulenceMod` deleted. Bit for bit. Golden tests repointed; new
  `tests/clm_ml/test_physics_turbulence.py`. Tests: lint; golden turbulence 59 pass, all
  fast golden 160 pass; new unit tests 15 pass, `test_imports` pass; bit-for-bit + noise twin
  177 pass (1.8 min); process replay 7 pass; focused CPU gradient test 1 pass (2.6 min).

- **Process 7, `leaf_boundary_layer.py`, `photosynthesis.py` (C16): round-off change,
  reference re-recorded.** Sunlit and shaded leaves in one call (layers on axis 0, leaves on
  axis 1, array code: one kernel instead of the four Medlyn/Ball-Berry/WUE x acclimation
  copies); stomatal models and boundary layer regimes by name; parameters as numbers;
  solvers `solve_ci`, `solve_gs_bisection` (same arithmetic, implicit-function gradients).
  Issue 18 settled by the maintainer: `wue_solver: brent` refused (physics and `clm_ml.run`;
  plan "After Phase 8: unsupported variants"). Adapters `clm_ml/legacy/leaf_boundary_layer.py`,
  `clm_ml/legacy/photosynthesis.py` (`il = None` both leaves, or one leaf for the replay);
  both legacy modules deleted. Per leaf under `jit` bit-identical to the old code; the
  (layer, 2) call compiles differently: 53 fields differ, worst 1.24x the twin spread
  (`probe77`). Process replay: fields zero to round-off in the recording now compared
  absolutely (night `aj`, 2.2e-16 vs 0; the old code compiled gives the same). Golden tests
  repointed; new `tests/clm_ml/test_physics_photosynthesis.py` (23). Tests: lint; fast golden
  160 pass; new unit tests 23 pass, `tests/clm_ml` 216 pass (`run.py` changed); before
  re-recording: noise twin pass, bit for bit 53 fail; process replay 7 pass; focused CPU
  gradient test 1 pass (2.2 min); after: bit-for-bit + noise twin 177 pass (1.2 min).

- **Process 8, `clm_ml/physics/flux_profile.py`, `water_vapor.py` (C17): round-off change,
  reference re-recorded.** Implicit flux-profile solution, leaf fluxes (both leaves at once)
  and soil fluxes on one patch (0-based layers, leaves on axis 1), the legacy kernel's
  arithmetic; `sat_vap`/`lat_vap` in `water_vapor.py` (all importers repointed);
  `FLUX_PROFILE` with `well_mixed` refused (physics and `clm_ml.run`; Fortran stops for it
  too); Fortran's `ErrorCheck01`/`ErrorCheck02` as `implicit_errors` (run by the unit test;
  never run in the time step, before or now: open issue 20). Adapter
  `clm_ml/legacy/flux_profile.py` (WellMixed translation kept there, then `endrun`);
  `MLFluxProfileSolutionMod`, `MLSoilFluxesMod`, `MLLeafFluxesMod`, `MLWaterVaporMod`
  deleted. On the recorded calls every written field is bit-identical to the old adapter
  except `etair`/`stair` (up to 1.8e-13 relative: the compiled fusion recomputes
  `eair = q pref` next to the subtraction; `optimization_barrier` does not survive to
  fusion; `probe78`); end to end 53 fields differ, worst 1.33x the twin spread. Golden
  `SatVap`/`LatVap` repointed. New `tests/clm_ml/test_physics_flux_profile.py` (6). Tests:
  lint; fast golden 160 pass; `tests/clm_ml` 223 pass (`run.py` changed); before
  re-recording: noise twin pass, bit for bit 53 fail; process replay 7 pass; after:
  bit-for-bit + noise twin 177 pass (1.3 min); focused CPU gradient test 1 pass (2.3 min; a first try was killed, out of memory while the reference was recorded alongside).

- **Process 9, `clm_ml/physics/nitrogen_profile.py`, `integrators.py` (C18):** the
  exponential nitrogen profile (both leaves at once, the integration error returned and
  recorded as `NITROGEN` by the adapter) and the Runge-Kutta update on the six integrated
  variables (`Integrated`) with the tableaus as numpy constants by model.yaml name
  (`TABLEAUS`; `config.INTEGRATOR_CODES`, was `_RK_CODES`). Adapters
  `clm_ml/legacy/nitrogen_profile.py`, `clm_ml/legacy/integrators.py`
  (`RungeKuttaUpdate(irk, tableau, inst, grid)`, `integrator_tableau(opts)`);
  `MLCanopyNitrogenProfileMod`, `MLRungeKuttaMod` deleted. Process replay translates the
  recorded `RungeKuttaUpdate` arguments. Golden tests repointed; the 5
  `CanopyNitrogenProfile` golden cases now run (were skipped). Bit for bit in isolation
  and end to end (a first version, identical in isolation, moved the shaded leaves'
  Vcmax25 by 1 ulp in the chunk: masks restored to the legacy order, `probe79`). New
  `tests/clm_ml/test_physics_nitrogen_integrators.py` (15). Tests: lint; fast golden 165
  pass; `tests/clm_ml` 238 pass (`config.py` changed); first try bit for bit 55 fail
  (fixed, above); bit-for-bit + noise twin 177 pass (1.8 min); process replay 7 pass;
  focused CPU gradient test 1 pass (2.8 min).

- **Process 10a, `clm_ml/physics/canopy.py` (C19): round-off change, reference
  re-recorded.** Process 10 planned and split into 10a-10d (C19). 10a: the step-level pieces
  of `MLCanopyFluxesMod` on 0-based arrays: `plant_area`, `soil_surface_humidity`, the
  sub-step averages (`SubstepFluxes`, `accumulate`, `average`), `canopy_totals` (Fortran
  `CanopyFluxesDiagnostics`; only what outputs, checks and the CLM coupling read: 5 totals
  and 23 layer profiles no longer computed, listed in `phase7_removed_diagnostics.md`),
  `merge_sun_shade`; adapters in `clm_ml/legacy/canopy.py` (`DECLARED`); the old helpers
  deleted (`MLCanopyFluxesMod` 2075 -> ~1000 lines). The adapter keeps the unused legacy
  index 0 of the leaf fields as recorded (averaged fill values; leaf index 0 of the history
  fields). Eager bit for bit with the old driver over steps 1-7 (`probe710/eager.py`); the
  compiled chunk differs in 4 fields at the last bit (`albcan`, `gppveg`, `anet_leaf`,
  `rnet`; with an `optimization_barrier` on the averages, without it 54 fields). New
  `tests/clm_ml/test_physics_canopy.py` (12). Tests: lint; `tests/clm_ml` + fast golden 415
  pass; before re-recording: noise twin 88 pass, bit for bit 4 fail; after: bit-for-bit +
  noise twin 177 pass (1.0 min); only the 4 fields changed in the reference.

- **Process 10b, the canopy sub-step on plain arrays (C20): round-off change, reference
  re-recorded.** `clm_ml/physics/canopy.py`: `substep` (forcing, solar radiation, nitrogen
  profile, the Runge-Kutta stages as one `lax.scan` calling the physics of processes 3-9
  directly, the sums), the carry `Substep` (0-based, legacy field names, `SUBSTEP_KINDS`),
  `SubstepInputs`, `SubstepOptions`; `clm_ml/legacy/canopy.py`: `substep_options` (code
  mapping and checks; Norman and well-mixed stop before the step), container <-> carry
  conversions. `MLCanopyFluxes` reads the container once and writes it once; step-by-step
  runs call one compiled sub-step. The step no longer writes the unused legacy index 0
  (C19's junk replication removed); 27 work values no longer written
  (`phase7_removed_diagnostics.md`). The adapters of processes 3-9 are no longer called by
  the step (kept for the process replay). Op by op bit for bit with the old driver over
  steps 2-3 (index 0 and dropped values masked; `probe710b/exact_masked.py`); compiled:
  60 fields at round-off (twin-moved within 1.65x the spread; twin-invariant solar fields
  a few ulps). New tests in `tests/clm_ml/test_physics_canopy.py` (14). Tests: lint;
  `tests/clm_ml` + fast golden 417 pass; process replay 7 pass; focused CPU gradient test
  1 pass (2.4 min); before re-recording: noise twin 85 pass, 3 fail (index 0 only); after:
  bit-for-bit + noise twin 177 pass (1.0 min).

- **Process 10c, the time step on the step state (C21): round-off change, reference
  re-recorded.** `clm_ml/step.py`: `step(state, inputs, constants, params, options, c3) ->
  StepState` (the soil routines of `clm_drv`, `GetCLMVar`, the canopy set-up, the sub-steps,
  averages, totals, merge, soil temperature) on `StepInputs`, `PatchConstants`,
  `StepOptions`; `clm_ml/legacy/step.py`: `patch_constants(template)`, `step_options`,
  `step_inputs`; `offline_executable.main.state_step` used by `make_chunk_runner`,
  `_patch_step` (now with the constants, formed once per chunk), `make_batch_runner` and
  the step-by-step loop. `instances_with_state`/`step_state_from_instances` left the time
  step (first step and view keep the containers). Each step's checks start from zero (was
  the first step's errors); `mlcan_to_clm = 1` refused. Op by op bit for bit with the old
  step in every state and diagnostic field over steps 2-3 (`probe710c/eager.py`); compiled:
  67 fields at round-off (twin-moved within 1.13x the spread; 14 twin-invariant by 1-2 ulps:
  solar zenith angle, solar fields, `psis_soil`; barriers did not help). New
  `tests/clm_ml/test_step.py` (3). Tests: lint; `tests/clm_ml` + fast golden 420 pass;
  before re-recording: noise twin pass, bit for bit 67 fail; after: bit-for-bit + noise
  twin 177 pass (1.0 min); `test_stream_runner` + `test_ensemble` 2 pass; process replay 7
  pass; focused CPU gradient test 1 pass (2.2 min). Forward timing CPU 15.6 ms/step
  (`p710c-check`; `p71` 24-25).

- **Process 10d, the psihat tables as an input of the compiled step (C22): bit for bit.**
  `RSLTables` is an argument of `canopy.substep`, `clm_ml.step.step`, `state_step`'s step
  and the compiled chunks (`make_chunk_runner`, `make_stream_runner`, `make_batch_runner`
  with `in_axes=None`, `_sharded` replicated; required keyword `tables`), from
  `LegacyRun.rsl_tables` (setup default `default_rsl_tables()`); `Model.run_chunk` takes an
  optional `tables`. The first step keeps the package tables as a constant until 7.4. The
  lowered chunk has the two tables as parameters, no 276x41 constant
  (`probe710d/consts.py`). Tests: lint; `tests/clm_ml` + fast golden 420 pass; bit-for-bit +
  noise twin 177 pass (1.4 min); `test_stream_runner` + `test_ensemble` 2 pass; focused CPU
  gradient test 1 pass (2.3 min).

- **Package-end tests and review fixes (C23).** On `20ce6f3`: CPU gradient tier 8 pass
  (6.8 min); all of `tests/regression` 207 pass (31.7 min); `tests/clm_ml` 255 pass; fast
  golden 165 pass (logs `logs7/p72_*.log`). Independent review by two reviewers (physics
  modules; step, driver, runners): one bug, `leaf_boundary_layer: laminar | turbulent`
  stopped every run (the conductances lost their leaf axis in 10b); fixed in
  `physics/leaf_boundary_layer.py`, unit test checks the shapes; CHATS7 with `gb_type` 1
  and 2 agrees with `bdde552` to 1.4e-11 over 30 steps. Two observations recorded for the
  maintainer (C23). Tests: lint; photosynthesis unit tests + fast golden 188 pass;
  bit-for-bit + noise twin 177 pass (default regime unchanged).

- **Timing, 7.2 done (2026-10-31).** `timing/time_model.py --steps 8` on CPU with the
  gradient flags (LLVM optimization off, issue 16), the end of 7.1 (`bdde552`, worktree)
  and HEAD alternated twice (`p72-*` in `timing/results_model.jsonl`, `logs7/timing72.sh`):
  forward 26.8/25.9 -> 30.7/30.8 ms/step (1.16x slower; below the 1.5x threshold);
  gradient 640/567 -> 405/376 ms/step (1.5x faster); setup 32 -> 19 s; peak memory
  4.3-4.4 -> 3.6-3.9 GB. (With the default flags the forward step was 15.6 ms at 10c.)

### 7.3 The chunk's forcing as 0-based arrays (2026-10-31)

- **Package 7.3 (C24): bit for bit.** The chunk inputs are `clm_ml.step.MetInputs` (per
  gridcell: calendar days, the step's meteorology, the previous and next steps'
  `ForcingRecord`s) and `clm_ml.step.LandInputs` (per patch: leaf and stem area, soil water,
  adjustment factor), NumPy on the host, stacked per step; `clm_ml.step.step_inputs` joins
  them per patch inside the chunk. Producers in `clm_ml/chunk_inputs.py` (`met_inputs`,
  `forcing_inputs`); tower runs through `main.chunk_inputs`/`patch_inputs`/
  `read_batch_inputs`, grid runs through `GridChunkReader` and `grid_forcing.land_record`.
  `clm_ml.legacy.step.step_inputs` and `main.stack_step_inputs` deleted; the first step keeps
  its legacy records until 7.4. Tests: lint; `tests/clm_ml` + fast golden 420 pass (one
  acclimation test updated: `advance_acclimation` takes the air temperature); bit-for-bit +
  noise twin 177 pass (1.5 min, unchanged reference). Package end: `test_coarse_forcing`,
  `test_land_data_files`, `test_grid_setup`, `test_stream_runner`, `test_patches_batched`,
  `test_ensemble` pass (11.3 min); CPU gradient tier 8 pass (6.1 min; logs `logs7/p73_*`).

- **7.3 review fixes.** Independent review of `5c946f5`: nothing serious; the previous
  forcing record in NumPy (no device work on the reader thread), grid input test by field
  name, a test helper removed (C24). Tests: lint; `tests/clm_ml` + fast golden 420 pass;
  bit-for-bit + noise twin 177 pass; `test_grid_setup` + `test_stream_runner` 4 pass.
  **Package 7.3 done.**

### 7.4 Setup without module globals (started 2026-10-31)

- **7.4a, the legacy view without the template (C25): bit for bit.**
  `clm_ml.legacy.view.legacy_view(step, layers)` fills the entries the state does not hold
  with `spval` (probe `probe74/fills.py`: every such entry of the CHATS7 template is `spval`)
  instead of copying them from the template; `view_series(steps, layers)`. Tests: lint;
  `tests/clm_ml` + fast golden 420 pass; bit-for-bit + noise twin 177 pass;
  `test_stream_runner` pass.

- **7.4b, the constants are setup values (C26): bit for bit.** `PatchConstants` lost the
  sub-step carry's start values and the padding slots the step kept (`substep`, `dlai`,
  `dsai`, `dpai`, `lwp_mean`, `tleaf_hist`, `lwp_hist`, `hk_l`, `smp_l`); the step starts
  the carry from `physics.canopy.initial_substep` (`spval`, errors 0) with the state put in,
  and keeps `spval` where it computes nothing. Tests: lint; `tests/clm_ml` + fast golden 420
  pass; bit-for-bit + noise twin 177 pass (no re-recording: the reference already holds
  `spval` in those slots); focused CPU gradient test pass (2.3 min).

- **7.4c, the setup as pure functions (C27): bit for bit against the legacy setup.** New
  `clm_ml/setup/` (`soil`, `canopy`, `tables`, `patches.build_patches`, `towers`), ported
  readers `clm_ml/io/tower.py` and `clm_ml/io/clm_history.py` (explicit soil structure, no
  module globals), `clm_ml/tower_forcing.py` (the tower forcing source on them),
  `clm_ml/calendar.py` (the time manager with an explicit calendar). Not yet used by runs.
  New `tests/regression/test_setup.py`: seven patches on three gridcells, every towers.yaml
  input varied: 266 constant and state arrays and every forcing array of 48 steps equal to
  the legacy setup's. Found two legacy setup bugs, kept bit for bit and recorded for the
  maintainer (`global_input.md` issues 21, 22). Tests: lint; `tests/clm_ml` + fast golden 420
  pass; bit-for-bit + noise twin 177 pass; `test_setup` pass.

- **7.4d, tower runs on the new setup, in-process (C28): reference re-recorded
  (round-off).** New `clm_ml/driver.py` (`Simulation`, `simulation(run)`, `make_runner`,
  `run_streams`, checks, restarts; from `offline_executable.main` without containers or
  template: the constants are an input of the compiled chunk, C23 b), `clm_ml/checks.py`
  (conservation checks, `ModelError`). `Run.execute` runs tower runs in this process (grid
  runs keep the worker until 7.4e; `legacy_ascii` keeps it until 7.4f); `clm_ml.model.Model`
  on `Simulation`. The first step is an ordinary step of the first chunk (the setup gives
  the state before it; acclimation counts start at 0). Before re-recording: noise twin
  pass; 68 fields not bit for bit: 53 the twins move, within 1.43x the twin spread; 15 they
  do not, by 1-2 ulps (solar zenith angle at 7 daytime steps and the solar fields after it,
  as in C21; soil temperature of the first step, 1.7e-15). Recording the reference and both
  twins: 15 s (legacy path: minutes). Tests: lint; `tests/clm_ml` + fast golden 420 pass;
  bit-for-bit + noise twin 177 pass (37 s); focused CPU gradient test pass (1.9 min).

- **7.4e, grid runs on the new setup (C29).** `clm_ml/setup/grid.py` (first-step values
  per patch from its gridcell's gridded data, `GridChunkReader` for `Simulation`, the
  in-chunk land data with the run's soil levels): no prototypes, no tower-format files, no
  worker. `Run.execute` runs every run in this process. `test_grid_setup` rewritten
  without the legacy code: the grid setup equals the tower setup of its data (constants,
  initial state, C3/C4, 48 steps of chunk inputs) bit for bit; the runs agree through the
  night; thin layers test unchanged. Tests: lint; `tests/clm_ml` + fast golden 420 pass;
  bit-for-bit + noise twin 177 pass; `test_grid_setup` 3 pass (2.2 min).

- **7.4f, the legacy setup, worker, containers and Fortran-mirroring packages deleted
  (C30).** `src/` holds `clm_ml` (and data); `clm-ml-offline` retired (`output.legacy_ascii`
  written by the run itself, `clm_ml/io/legacy.py`: the `.out` files of a CHATS7 day equal
  the legacy driver's but for the sign of night-time zero GPP); constants (Python floats),
  orbit, `CanopyParams`, options moved into `clm_ml`. Tests: adapter tests and legacy-vs-
  legacy tests deleted; `test_setup` against a recorded copy of the legacy setup
  (`reference/setup_7patches.npz`); the tests that used the legacy driver rewritten on
  `clm_ml.driver`/`Model`; parallel runs in tests made sequential (in-process netCDF is not
  thread-safe); `test_devices` runs each case in its own process (device count fixed at JAX
  start); timing scripts on the new API. Tests: lint; `tests/clm_ml` + fast golden 375
  pass; all of `tests/regression` 199 pass (19.7 min); CPU gradient tier 8 pass (6.7 min).

- **7.4 review fixes and timing (C31). Package 7.4 done.** Two independent reviewers (setup
  and readers; driver, Model and run). Fixed: the output shapes were recomputed on every
  `Model.run_chunk` (1.4 s per call: the first timing showed forward 234 ms/step); now found
  once per stream set; `program=` refuses a program with other options; output files are
  closed (their records written) when a run stops on an error; the legacy `.out` files are
  appended when a run continues from a restart; the checks transfer the state once per chunk;
  a CLM history whose soil levels do not match `soil.layer_structure`, or a time slice outside
  the file, is an error (it was silently truncated or wrapped). Timing (`time_model.py
  --steps 8`, CPU, gradient flags; `logs7/timing74.sh`, `p74-*` in
  `timing/results_model.jsonl`): end of 7.3 (`8d93889`) forward 33.4/31.0 ms/step, gradient
  542/395 ms/step, setup 21 s, peak 3.8-3.9 GB; after the fix forward 25.1/25.2 ms/step
  (1.3x faster), gradient 373/365 ms/step, setup 11-12 s, peak 3.1-3.2 GB. Tests: lint;
  `tests/clm_ml` + fast golden 378 pass; bit-for-bit + noise twin, `test_setup`,
  `test_grid_setup`, `test_coarse_forcing`, `test_land_data_files`,
  `test_sites_one_program`, `test_one_program_any_layers`, `test_patch_exclusion` 188 pass.

### 7.5 Restarts as netCDF (2026-11-01)

- **Package 7.5 (C32): netCDF restarts.** `clm_ml/io/restart.py` (format 6: state with units,
  alive patches, the records in progress of every stream with typed keys, layout, calendar,
  the output files in progress; `.prev` rotation and fallback); the run loop and `Run` use it
  (`<output>/restart/<run>_restart.nc`). Independent review applied (stream identity checked
  on resume, named dimensions and units, clear refusal of pickle restarts, warning on
  `.prev` fallback). Tests: lint; `tests/clm_ml` + fast golden 378 pass (new
  `test_restart_file`, 4); `test_restart` (resume bit for bit through the file),
  bit-for-bit + noise twin, `test_patch_exclusion` 180 pass; CPU gradient tier 8 pass
  (6.0 min). **Package 7.5 done.**

### 7.6 Cleanup (2026-11-01)

- **7.6a, outputs from the step state, reference in the new names, data out of `src/`
  (C33, C34).** `outputs.step_fields`/`_Raw` 0-based; validation fields
  `diagnostics.*`/`state.*`/`layers.*`; reference re-recorded in those names (values equal
  to the old reference's 88 legacy fields at every step, checked before re-recording);
  `clm_ml.legacy` removed (`io/namelist.py`; legacy layout only in the `.out` writer, whose
  files are byte-identical); `data/input_files`, `data/namelists`; one `pyproject.toml`;
  stale docstrings, README, CONTRIBUTING, CLAUDE.md updated. Tests: lint; `tests/clm_ml` +
  fast golden 370 pass; all of `tests/regression` 214 pass + the grid thin-layer test after
  its field names were updated (18.6 min).

## Next (resume here)

**Package 7.2 done (2026-10-31; sessions 1-11, processes 1-10d, package-end tests,
review fixes C23, timing above); last commit = the one that wrote this.** Stopped to report
to the maintainer. **7.3 discussed with the maintainer (2026-10-31): D3 revised (plan
decision 36)**: offline runs know all forcing in advance, so the forcing stays prepared on
the host, a chunk ahead and overlapped with the GPU; the meteorology is not moved into the
chunk (measured: GPU time is the physics, the forcing ~2-4 GB of a 21 GB peak). Next session:
**package 7.3 as revised** (`phase7_scope.md`, 7.3, ~1 session): the chunk inputs as stacked
0-based arrays shaped like `clm_ml.step.StepInputs` (meteorology per gridcell, land data per
patch, previous/next records without the patch axis, 0-based bands) from the tower readers,
`GridChunkReader`/`land_fn` and `Model.forcing`; `clm_ml.legacy.step.step_inputs` leaves the
time step; first step unchanged until 7.4; bit for bit expected. Package-end feature tests:
`test_coarse_forcing`, `test_land_data_files`, `test_grid_setup`, `test_stream_runner`,
`test_patches_batched`; timing baseline `p72-clm-ml-jax-*`. C23: the maintainer agreed to fix
the template-as-constant in 7.4 and to leave the Euler tableau as it is. Other open items:
the list below.

Same pattern as processes 2-9 (C10, C12-C18): physics on plain 0-based arrays in `clm_ml/physics/` (mask helpers in `physics/layers.py`, layer grid `PatchLayers`), legacy
adapters with the old names and a `DECLARED` reads/writes table in `clm_ml/legacy/`, a test on
fake containers holding only the declared fields. Expect round-off changes when loops become
array operations even if every routine is bit-identical in isolation (C10): probe the routine
alone against the old code first (`probe72/compare.py` pattern, old files from git), then, if
only the whole-chunk compile differs and the noise twin passes, re-record the reference in the
same commit (D7) and log it. Constants through `clm_ml/physics/constants.py`.
If the routine alone differs by an ulp because an intermediate no longer round-trips through
the container, XLA may be folding divisions across it: pass the values Fortran stores through
`jax.lax.optimization_barrier` (C15). The barrier only blocks HLO simplification: if the
compiled fusion recomputes a value next to its use (C17: check the optimized HLO,
`jit(f).lower(...).compile().as_text()`, for `opt-barrier`), it does not help; then it is the
C10 pattern (round-off, re-record).
Before re-recording, compare the run with the reference relative to the twin spread
(`probe78/ratio.py`): a field the twins do not move that differs (C18: the moved routine's
own output, e.g. `vcmax25_leaf`) points at the moved arithmetic itself, often fixable by
keeping the legacy order of masks and operations.

### Fixes to make at process 10 (maintainer's instructions)

Items deferred to process 10 (the step driver, where the containers leave the step), each
done and ticked in the process-10 commit:

- [x] **psihat tables as a run-time input of the compiled step** (done in 10d, C22) (maintainer, 2026-10-31;
  C15). Today the legacy adapter passes `clm_ml.io.rsl_tables.default_rsl_tables()`, so the
  tables are a constant of the compiled chunk. At process 10: pass `RSLTables` into the step
  as an input (one copy per chunk, not per patch or per step: broadcast, `in_axes=None` under
  `vmap`, replicated under `shard_map`) through `make_chunk_runner`, `make_stream_runner`,
  `make_batch_runner`, the sharded wrapper (`_sharded`), the batched first step and
  `clm_ml.model.Model`; drop `default_rsl_tables()` from the time step (keep it only as the
  setup's default). Test: bit for bit (the values are the same; a constant becoming an
  argument may change XLA's folding: probe as in C10/C15 if not) and the focused gradient test.

Open for the maintainer: C10 and C14 (reference re-recorded for the soil and process-5 moves,
round-off only), C15 (the `optimization_barrier`; the psihat tables as a step input done in 10d, C22), open issue 19 (soil ice index in `SoilResistance`, latent), C16 (reference re-recorded for process 7, round-off only; process replay compares fields zero to round-off absolutely). C17 (reference re-recorded for process 8, round-off only; `well_mixed` refused by `clm_ml.run`), C18 (process 9, bit for bit; `kn_val` still a static option), C19 (process 10 split into 10a-10d; 10a re-recorded the reference, 4 fields at the last bit; 28 diagnostics no longer computed), C20 (10b re-recorded the reference, round-off; the step no longer writes the unused legacy index 0, so the reference's index-0 entries are now setup fill values; Norman/well-mixed stop before the step; 27 more work values not written), open issue 20 (Fortran's implicit-solution checks never run in the step: record them in `BALANCE_CHECKS`?), C21 (10c re-recorded the reference, round-off; each step's checks start from zero instead of the first step's errors; `mlcan_to_clm = 1` refused), C22 (10d, bit for bit: the psihat tables an input of the compiled chunks, `tables=` required; the first step keeps them as a constant until 7.4). Earlier: C4, the removed
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
- **Since process 10c the time step is `clm_ml.step.step` on the 0-based state** (constants from
  the template, `clm_ml.legacy.step`); only the first step and the legacy view use the
  containers. Before 10c (kept for the record): **two layouts:** the physics ran on the legacy containers
  (1-based, patch axis of length 1); the step's input/output is the 0-based `StepState`
  (`clm_ml/state.py`), converted in `clm_ml/legacy/state.py`; outputs read through
  `clm_ml/legacy/view.py`. When a moved process changes what the step returns, change the
  view, never the reference. The legacy container leaves the step at process 10
  (`canopy.py`).
- **Known quirk:** the legacy `.out` writer reads soil level 1 of a column axis of length 1;
  it works only because JAX clamps out-of-range indices (C8). Don't convert the view to numpy.
- **Never** scatter into a scan carry with array indices, add `jax.debug.callback`, or set the
  gradient XLA flags for the regression tests (CLAUDE.md gotchas).
