# Phase 7 scope (decisions taken 2026-10-30; start not yet agreed)

Phase 7 turns the translated Fortran structure into the target architecture of the plan (§6):
physics organized by process under `clm_ml/physics/`, a per-step state in place of the legacy
containers, setup without module globals, forcing interpolated inside the compiled chunk,
restarts as netCDF, and only `clm_ml/` left in `src/`. Every step gives the same answer as
before (bit for bit on CPU, or round-off only where stated).

Sources: the plan (Phase 7, §6), two code surveys of 2026-10-30 (time step; setup, I/O and
tests) and the Phase 6 review. Numbers marked ~ are estimates.

## 1. Decisions already taken (maintainer, 2026-10-30)

| | Decision |
|---|---|
| D1 | **0-based indexing** for layers, leaves, bands, soil levels and RK stages. |
| D2 | **Restarts as netCDF.** |
| D3 | **Meteorology built inside the compiled chunk** (open issue 7). |
| D4 | Orbital parameters per year: **not** done (decision 34's known departure stays). |
| D5 | **No new unrolled Python loops** in compiled code; remove the large existing ones where it preserves the arithmetic. |
| D6 | **Diagnostics: keep only what outputs and checks use.** Every removed field is listed (name, meaning, the routine that computed it) in `docs/design/phase7_removed_diagnostics.md` so it can be added back. |
| D7 | **Round-off-changing simplifications allowed** (sun/shade as one vmapped call, vectorized soil loops, one `SoilThermProp` call), each as its own commit with a recorded reference update and the noise-twin test. |
| D8 | **WellMixed flux-profile, dataset profiles, Norman solar: kept, registered as unsupported** (a clear error when chosen); fixing them is a later task (listed in the plan after Phase 8). |
| D9 | **`gs_solver` and `mlcan_to_clm` options kept.** |
| D10 | **C3/C4 groups unchanged** (decision 22); revisited with the GPU work after Phase 8. |
| D11 | **Worker subprocess dropped** once setup has no module globals (7.4): runs and tests in-process. |
| D12 | **Gradient testing:** CPU gradient tier with `--xla_backend_optimization_level=0` after each process move (§7 D); GPU gradient job at package ends with a warm compilation cache (§7 A, C). |
| — | Earlier: process interface as a name-to-function registry with declared reads/writes (§6.6, decision 21); bare ground, issue 10 and GPU performance after Phase 8. |

## 2. What the code looks like today (facts from the surveys)

- **Time step.** `MLCanopyFluxes` (623 lines) drives ~25 routines; sub-steps and RK stages are
  already `lax.scan`, solver iterations `fori_loop`. Soil routines (`clmData`, `SoilAlbedo`,
  `calc_soilevap_resis`, `SoilThermProp` (called twice), `SoilWater`, `SoilTemperature`) are
  scalar per-layer Python loops.
- **State.** The legacy `mlcanopy_type` has 235 fields; the step uses 232. By role: 7
  prognostic (+ soil temperature and the acclimation mean), 13 carried within a step (`*_bef`,
  RK increments), ~33 inputs rewritten each step, 17 setup constants, ~164 diagnostics and work
  values. Eleven other legacy containers contribute ~40 fields.
- **Indexing.** The patch index in the run path is already 0 (an axis of length 1 that `vmap`
  makes unnecessary). 1-based: layer axis (`max_layers + 1`, index 0 unused except as the
  ground level in 7 fields), leaf axis (`isun = 1`, `isha = 2`), band axis (`ivis = 1`,
  `inir = 2`), soil levels, RK stages; `ncan/ntop/nbot` stored 1-based. ~900 index sites,
  ~25 % of them in `MLCanopyFluxesMod`; ~1 000 patch-axis sites that disappear rather than
  shift. Already mixed today (`tridiag_scan` 0-based, `tridiag` 1-based, parts of `clmData`).
- **Setup.** Per (forcing key, PFT) a legacy prototype setup (`setup_run`: `InitializeRealize`,
  14 container singletons, `TowerDataMod`, `pftcon` tables, `ColumnType`/`PatchType`), then
  `vector_setup` recomputes every per-patch field vectorized, and the first step builds the
  template. A worker subprocess is needed only because of the module globals.
- **Forcing.** Built on the host per chunk at the model step; each meteorology row is sent three
  times (previous/current/next records): ~39 numbers per gridcell-step for grid runs.
- **Program size.** The compiled 10-day global chunk is 2.3 GB of GPU code. Duplication comes
  most likely from: the C3/C4 groups (the whole step traced twice in mixed batches; decision
  22), sun and shade photosynthesis traced separately, ~500 scalar soil operations, and a
  150-term unrolled two-stream sum.
- **Tests tied to the legacy layout.** The end-to-end reference (`.npz` of raw 1-based
  `mlcanopy.*` fields), the `process_io` replay pickle (7 routines on pickled legacy
  containers), `outputs._Raw` (output variables computed from raw 1-based fields),
  `test_differentiability.py` (hand-built legacy setup). The Fortran golden tests hardly depend
  on the layout (one adapter, `tridiag`). Dead modules: `lnd_comp_mct`, `histFileMod`,
  `restUtilMod`, `fileutils`, `cime_src_share_util/*`, `clm_share.shr_file_mod`/`shr_kind_mod`,
  CLM's `SoilTemperatureMod`, `WaterFluxType`, `WaterStateType`, `offline_driver/CLMml.py`,
  `utils/gpu_config.py`.

## 3. How the work keeps the answer the same

The bit-for-bit end-to-end test is the safety net, so it must survive the change of names and
layout:

1. **A legacy view of the new state** (test adapter): maps the new state and diagnostics back to
   the reference's field names and 1-based shapes (`mlcanopy.tair_profile` etc.). The reference
   stays the Phase 6 one, and every work package is checked bit for bit against it. At the end
   of Phase 7 the reference is re-recorded in the new names (same values; the adapter goes).
2. **Process-level replay** (`process_io`): re-recorded per process in the new structure when
   that process moves (inputs/outputs converted once through the legacy view), so each moved
   process is compared with its own pre-move behaviour on identical input.
3. **Golden Fortran tests**: import paths and the `tridiag` adapter change; the JSON files never
   change.
4. **Gradients**: CPU gradient tier (optimization off, §7 D) after each process move; GPU tier
   at package ends.
5. **Fortran comparison** (`.out` files): a small netCDF-to-`.out` writer in `clm_ml/io/legacy.py`
   replaces `clm-ml-offline` for this workflow.

## 4. Work packages, in order

Each package: one or more commits on `phase7/restructure`, before-commit tests (fast tier,
bit-for-bit + noise twin, the package's own tests), CPU gradient tier where physics moved (GPU tier at the package end),
documentation in the same change.

**7.0 Preparation** (~2 days)
- Delete the dead modules (list in §2); remove `controlMod` writes and `clm_instRest`.
- The legacy-view adapter and a test that it reproduces today's reference from today's
  containers exactly.
- The CPU gradient tier (pytest marker + command with the flag) and the GPU gradient job with
  a warm cache.
- Update CLAUDE.md gotchas that no longer hold (options copied at import, patch index 1).

**7.1 The step state** (~1 week) — §6.6 interface, 0-based from the start (D1)
- `StepState` (`clm_ml/state.py`): prognostic state, within-step carry, inputs, diagnostics, as
  NamedTuples of 0-based arrays without the patch axis; `PatchLayers` (setup constants and
  masks); `PatchParams` (exists as `CanopyParams`).
- Boundary adapters old container ⇄ `StepState` so processes can move one at a time while the
  rest still uses the container (transition only; removed in 7.4).
- Diagnostics (D6): only fields used by outputs and checks; removed ones listed in
  `phase7_removed_diagnostics.md`. The legacy view fills removed fields from the reference so
  the bit-for-bit test still compares everything that remains.

**7.2 Processes into `clm_ml/physics/`** (~3-4 weeks; one commit per process, each bit for bit)

| Order | Process (from) | Notes |
|---|---|---|
| 1 | `math.py` (MLMathToolsMod) | `tridiag` as a scan (removes 48 unrolled ops) |
| 2 | `soil.py` (clmData, SoilAlbedo, SurfaceResistance, SoilThermProp, SoilWater, SoilTemperature) | array operations instead of ~500 scalar updates (D5); `SoilThermProp` computed once if its two calls see the same inputs (check first) |
| 3 | `atm_forcing.py` (GetAtmForcing, TimeInterpolation3) | ready for 7.3 |
| 4 | `solar_radiation.py` (two-stream; Norman stays registered as unsupported) | the 150-term sum as a scan in the same order |
| 5 | `longwave.py`, `canopy_water.py`, `leaf_heat_capacity`, `plant_hydraulics.py` | sun/shade water potential over the leaf axis |
| 6 | `turbulence.py` (MLCanopyTurbulenceMod, 1 500 lines; psihat table packaged as data) | largest; RSL lookup tables as an input, not a module global |
| 7 | `leaf_boundary_layer.py`, `photosynthesis.py` | sun and shade as one vmapped call over the leaf axis (D5; halves the largest kernel) — round-off only if XLA fuses differently: check |
| 8 | `flux_profile.py` (implicit solution, soil fluxes; WellMixed registered as unsupported, D8) | |
| 9 | `nitrogen_profile.py`, `integrators.py` (RK) | |
| 10 | `canopy.py`: the step driver (MLCanopyFluxes, accumulation, diagnostics) | the legacy container leaves the step here |

Each process gets: variants registered by name (§6.6), a docstring with `Fortran reference:` and
its reads/writes, a test that the declared writes are what changes, its golden tests repointed.

**7.3 Forcing inside the chunk (D3)** (~1-2 weeks)
- Tower: derived records sent once per chunk (`n + 2` rows), previous/current/next by slicing in
  the scan; the different arithmetic of the previous and next records kept (bit for bit).
- Grid: the 6-hourly CRUJRA records the chunk needs plus per-step interpolation indices and
  weights; linear, nearest and coszen interpolation and the CDEPS derivation (`datm_derive`)
  in `jax.numpy` inside the chunk; per-step calday/declination from the host. Equal to today's
  host path to round-off; the grid regression test compares the inputs.
- CLM history record for towers as records + weights (as `land_record` for grids).
- Memory per chunk falls from ~39 numbers per gridcell-step to ~8 per gridcell-record.

**7.4 Setup without module globals** (~2-3 weeks)
- Pure functions: soil layer structure, soil properties and roots, vertical structure and
  initial profiles (already vectorized in `vector_setup`, freed of their prototype bases), PFT
  values from params.yaml (`rootprof_beta`, `c3psn`, canopy heights, PAD betas move there),
  initial soil state from the history file, the first step as an ordinary compiled step.
- Then the prototypes, `InitializeRealize`, the container singletons, `TowerDataMod`, the
  worker subprocess (runs in-process) and the import-order gotchas go away.
- Tests: the new setup against today's template (bit for bit), towers and grid.

**7.5 Restarts as netCDF (D2)** (~3-4 days)
- Schema: dimensions patch, gridcell, layer, leaf, soil level; the prognostic state with units;
  per stream the record in progress (sums/values, count, key, end time); `alive`, layout,
  `chunk_anchor`, writers' files in progress; model calendar and time; format version.
- Keep the `.prev` rotation and fallback; resume rebuilds setup deterministically (the template
  is not stored). Tests: resume bit for bit (as now), file round trip, schema check.

**7.6 Cleanup (7b)** (~1 week)
- Remaining legacy modules deleted; data out of `src/` (`importlib.resources`); one
  `pyproject.toml`; `clm-ml-offline` retired (`clm-ml convert` + `clm-ml run`; `.out` writer for
  the Fortran comparison); `outputs` defined on the new state; reference re-recorded in the new
  names; timing scripts on the new API; docs.

Rough total: ~10-14 weeks of sessions. Largest risks: 7.2 (turbulence, photosynthesis, the
driver) and 7.4 (setup).

## 5. Done when

`multilayer_canopy/` and the other Fortran-mirroring packages are gone (not just unimported);
`src/` holds only `clm_ml/`; the step is `clm_ml/physics/` processes on `StepState`; restarts are
netCDF; forcing is interpolated in the chunk; all tests pass (CPU tiers, GPU gradient tier,
Fortran comparison unchanged); the program-size and compile-time numbers recorded before and
after (no GPU optimization beyond D5).

## 6. Risks and how they are handled

| Risk | Handling |
|---|---|
| A refactor changes the answer silently | bit-for-bit through the legacy view after every commit; process replay per process |
| Gradients break | CPU gradient tier (optimization off, §7 D) after every process move; GPU tier at package ends |
| Boundary adapters during 7.2 slow the step | transition only; timing test before/after 7.2, discuss > 1.5x (standing rule) |
| vmapping sun/shade or vectorizing soil loops changes round-off | allowed only with the noise-twin test and a recorded reference update, as a separate commit |
| Setup rewrite misses a forcing-dependent field | the template comparison test (towers and grid, the Phase 6b review locations) |
| Phase too large to review | one process per commit; a progress log as in Phase 6 |

## 7. GPU queue time

Measured 2026-10-30: the GPU develop queue (`gpudev`) started jobs in ~10-20 min; the main GPU
queue took hours; the full gradient tier takes ~45 min to run, a focused gradient test ~5 min
with a warm compilation cache.

Options (recommended: D for day-to-day gradient checks, plus A + C for the GPU tier at package
ends; B and E only if D turns out too slow):

- **A. Gradient tier per package, not per edit.** Forward correctness is checked bit for bit on
  CPU after every edit; gradients once per moved process (focused test, ~5 min of GPU) and the
  full tier at each package end.
- **B. A GPU test runner held for a work block.** One develop-queue job (1 GPU, a few hours)
  runs a small loop that picks up test requests (commit + test list) from a scratch directory,
  runs them in a worktree and writes the results. One queue wait per session instead of one per
  check; cost: the GPU-hours the job holds (~1 per hour).
- **C. Warm compilation cache** for the GPU gradient programs (they are small enough to cache):
  unchanged programs reload in ~1 min instead of ~7.
- **D. CPU gradient tier with LLVM optimization off (issue 16 work-around, found 2026-10-30).**
  Probes (`probe/issue16/`): the 1-step CHATS7 gradient segfaults while running on CPU (the
  forward pass is fine) with the default settings, single-threaded execution, a 2 GB thread
  stack, no compilation cache and AVX2-only code generation, but **runs with
  `XLA_FLAGS="--xla_cpu_parallel_codegen_split_count=1 --xla_backend_optimization_level=0"`**:
  the crash is in LLVM's optimized CPU code. With that flag the chunk gradient test and all
  of `tests/test_differentiability.py` pass on a login node (8 tests, 9.7 min, no queue).
  The flag changes round-off (loss 119.32881205729213 vs 119.32881205727297), so it is for
  gradient tests only, never for the bit-for-bit or noise-twin tests. Proposal: a CPU
  gradient tier run by a separate pytest invocation with the flag after each process move;
  the GPU tier at package ends (GPU gradients are what calibration uses). Still to do: report
  the crash to JAX with a minimal reproducer.
- **E. Login-node GPUs** (derecho7 has two A100 80 GB): instant for 5-minute checks, if NCAR's
  login-node policy allows it; not for long runs.

## 8. Open decisions

All answered 2026-10-30 (D6-D12 in §1). Remaining: the maintainer's go-ahead to start 7.0.
