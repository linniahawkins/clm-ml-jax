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

### C9. Process 1: `clm_ml/physics/math.py`; imports updated, no shim; dead root finders dropped (7.2, 2026-10-30)
Context: `multilayer_canopy/MLMathToolsMod.py` moves to `clm_ml/physics/math.py` (scope 7.2
process 1, `tridiag` as a scan); how the old module name keeps working, and what moves.
Decision:
- **No re-export shim:** the 10 importing modules and the golden tests import
  `clm_ml.physics.math`; `MLMathToolsMod.py` is deleted. `test_imports` catches a stale import.
- **Moved:** `is_concrete`, `bisection_traceable`, `quadratic`, `tridiag_2eq` (with its
  `custom_vjp`, unchanged), the beta/gamma functions. **Not moved (deleted):** `hybrid`,
  `zbrent`, `bisection` (Fortran's container-threading signature) and their `_scalar` copies,
  `quadratic_py`: nothing calls them (only `noqa` imports and docstrings named them). Found
  while checking this: `wue_solver: brent` is never used (open issue 18 in
  `global_input.md`, for process 7).
- **`tridiag`** is 0-based (D1) and a `lax.scan` (forward elimination and back substitution),
  replacing both the 1-based unrolled `tridiag` and `tridiag_scan`. The unrolled one guarded
  every pivot with `max(|bet|, 1e-10)` (not in Fortran); the scan does that only with
  `min_pivot=1e-10`, which the solar radiation and soil temperature calls pass (as before);
  longwave calls it without (as `tridiag_scan` did). The golden test runs Fortran's unguarded
  form; `tests/clm_ml/test_physics_math.py` checks both against a dense solve and that the
  guard changes nothing when pivots exceed it.
- `_beta_function_incomplete_cf` raises `RuntimeError` on non-convergence instead of calling
  the legacy `endrun` (`SystemExit`), so the module imports nothing from the legacy packages.
- No variant registry for this module: it has no process variants (pure helpers).
Why: a shim would be one more legacy name to remove in 7.6, and the callers are few; the
dead finders could not run in a compiled step anyway (Python `while` on the container).
Alternatives: a shim module re-exporting the names (keeps every import working, but leaves
two names for each function until 7.6); keeping the guard always on (would change longwave
round-off if a pivot ever fell below 1e-10, and departs further from Fortran).

### C10. Process 2: `clm_ml/physics/soil.py` as array functions; legacy adapters in `clm_ml/legacy/soil.py` (7.2, 2026-10-30)
Context: the soil routines of the step (`clmData`, `SoilAlbedo`, `calc_soilevap_resis`,
`SoilThermProp`, `SoilWater`, `SoilTemperature`) move to `clm_ml/physics/soil.py` with array
operations (D5) while the rest of the step still runs on the legacy containers.
Decision:
- **Physics:** six functions on one soil column, 0-based soil levels and bands (D1), arguments
  and results only (`soil_water_from_history`, `soil_albedo`, `soil_evaporative_resistance`,
  `soil_thermal_properties`, `soil_hydraulic_properties`, `soil_temperature`); each docstring
  names the legacy fields it reads and writes. Per-level Python loops replaced by array
  operations (same expression per element); the tridiagonal assembly keeps each row type's
  own expression (selected with `where`); the energy-conservation sum is a `lax.scan` in
  Fortran's order (bit for bit).
- **Adapters:** `clm_ml/legacy/soil.py` keeps the six legacy names and arguments (so
  `clm_driver` only changes its imports), slices the column out of the containers, calls the
  physics and writes the results back to the same entries. `DECLARED` lists each adapter's
  reads and writes; `tests/clm_ml/test_physics_soil.py` runs every adapter on fake containers
  holding only the declared fields (an undeclared read fails; exactly the declared writes
  change). This is the "declared writes" test of the scope for processes that still run on
  the containers. The old modules `SoilWaterMovementMod`, `SurfaceResistanceMod`,
  `MLSoilTemperatureMod` are deleted; `SoilAlbedo` leaves `SurfaceAlbedoMod` and `clmData`
  leaves `clmDataMod` (their setup and file-reading parts stay until 7.4).
- **Constants:** `clm_ml/physics/constants.py` re-exports the `clm_varcon` values (one place
  for the processes; the definitions move there in 7.6).
- **Snow layers:** the adapter no longer reads `ColumnType.col.snl`: Fortran `clm_drv` sets
  `snl = 0` before `SoilThermProp`, so the snow water is always added to the top level's heat
  capacity (the setup's `snl` is 0 too, so nothing changes).
- **Legacy quirk kept:** `h2osoi_vol_col` has `nlevgrnd` entries indexed 1..nlevgrnd, so the
  bottom level never had an entry (the old element writes past the end were silently dropped
  by JAX). The adapter writes levels 1..nlevgrnd-1, as before. The old `clmData` then read
  that missing entry (JAX clamps reads: the level above) for the bottom level's liquid water;
  the physics uses the bottom level's own value. Identical for `CLM5_0` histories (the
  bedrock levels are 0) — the only setting in use; it would differ for `CLM4_5`.
- No variant registry: the soil routines have no alternatives in CLM-ml v2.
- **Round-off change, reference re-recorded (D7, "vectorized soil loops"):** the move is not
  bit for bit end to end: 55 of 88 fields differ by round-off from step 1 (`psis_soil`, 1 ulp)
  and step 6 (`t_soisno_col`, 4e-13 K), worst 1.65x the noise twins' spread (allowed 2x;
  noise twin passes on all fields). Probes (`/glade/derecho/scratch/linnia/clm-ml-jax/probe72`,
  old routines from `a0af038` against the new ones on identical inputs): **every routine is
  bit-identical in isolation** (eager, `jit`, `jit` + `vmap` over patches, traced `dtime`),
  except the soil-temperature energy error under `jit` with a constant `dtime` (1e-14 W m-2,
  XLA folding the constant; identical with `dtime` traced, as in runs). Swapping old routines
  back into the full run: old `SoilWater` alone changes nothing; old `clmData` alone removes
  the step-1 difference and old `SoilThermProp`/`SoilTemperature` the step-6 one. So the
  differences come from XLA compiling the whole chunk differently once ~500 element updates
  become a few array operations (fusion and the order of the sums that read these arrays, e.g.
  `psis_soil` in plant hydraulics), not from any changed formula. Keeping the per-element
  updates for a bit-for-bit move first would add a commit of code that D5 removes (and is not
  certain to be bit for bit: the adapters' slicing changes the graph too). Reference and twins re-recorded on CPU with this
  commit (`record_run.py --update-reference`).
Why: the physics written once on plain arrays is what 7.2 aims for; the adapters keep the
legacy step unchanged until process 10 and are deleted with it.
Alternatives: adapters inside the old modules (keeps six legacy files alive for thin
wrappers); declaring reads/writes only in docstrings (not testable).

### C11. Soil thermal properties computed once per step (7.2, 2026-10-30)
Context: scope 7.2 process 2: `SoilThermProp` once if its two calls (Fortran `clm_drv` before
the canopy fluxes, and `SoilTemperature` after them) see the same inputs; check first.
Decision: `clm_drv` passes its `tk`, `cv` to `SoilTemperature`, which no longer recomputes
them (`clm_ml.legacy.soil.SoilTemperature(..., dtime, tk, cv)`).
Why: the inputs are the same arrays: between the calls only `SoilWater` (writes `hk_l`,
`smp_l`) and `MLCanopyFluxes` (returns only `mlcanopy`) run, and neither writes
`t_soisno`, soil water, `h2osfc`, `h2osno` or the soil properties; the second call's `thk_col`
write repeated the first's values. Confirmed by the bit-for-bit test (177 pass): the step's
soil temperature, the only user of the second call, is unchanged.
Alternatives: keep Fortran's two calls (twice the work for the same numbers).

### C12. Process 3: `clm_ml/physics/atm_forcing.py`; the legacy forcing window in `clm_ml/legacy/atm_forcing.py` (7.2, 2026-10-31)
Context: `MLGetAtmForcingMod` (`GetAtmForcing`, `TimeInterpolation2/3`, the forcing record
and window types, `clm_forcing_to_ml`) moves to `clm_ml/physics/` (scope 7.2 process 3,
"ready for 7.3").
Decision:
- **Physics:** `atm_forcing(bef, cur, nxt, zref, t_bef, t_cur, t_next, t_ml, *,
  interpolation)` on one patch, 0-based shortwave bands, returning `AtmForcing` (the 15
  `<name>_forcing` fields); inputs are per-patch `ForcingRecord`s. The time interpolation is a
  variant registered by name, `TIME_INTERPOLATION = {"none", "three_point"}` (model.yaml
  `interpolation`; Fortran `met_type` 0 and 3), applied field by field (`tree_map`; the
  shortwave bands as one array instead of two scalar calls). `clm_forcing_to_ml` (Fortran
  `GetCLMVar`'s unit conversion) moves to the physics module too.
- **Two-point interpolation dropped:** `TimeInterpolation2` (Fortran `met_type = 2`) is not
  moved. In CLM-ml v2 it stops the model (`endrun` before the interpolation), and the
  configuration only accepts 0 and 3 (`clm_ml.config`, `legacy.namelist`). The adapter
  raises `ValueError` for any other `met_type` (before: `endrun` for 2, an undefined-name
  error for anything else).
- **Adapter and legacy types:** `clm_ml/legacy/atm_forcing.py` keeps `GetAtmForcing` with its
  legacy arguments (so `MLCanopyFluxes` only changes its import), `DECLARED` reads/writes,
  and the legacy-layout `AtmForcingRecord`/`AtmForcingWindow` (patch axis, 1-based band
  columns) used by the forcing builders; they go when the forcing is built inside the chunk
  (7.3) and the containers leave the step (process 10). `MLGetAtmForcingMod.py` is deleted, no
  shim (as C9). `tests/regression/process_io.py` reads the pickled window of the old module
  from the new one (`_Unpickler.MOVED`), so the process replay keeps its recording.
- **Constants:** `cpd`, `cpw`, `lapse_rate`, `mmdry`, `mmh2o`, `rgas`, `wind_forc_min`
  re-exported from `MLclm_varcon` by `clm_ml/physics/constants.py` (as C10).
- **Bit for bit** end to end (177 pass); the process replay of `GetAtmForcing` passes on the
  recorded inputs. New `tests/clm_ml/test_physics_atm_forcing.py` (declared reads/writes on a
  fake container, piecewise-linear interpolation, derived air properties).
Why: the physics on plain per-patch values is what 7.3 needs (the forcing interpolated
inside the chunk); the legacy window stays only as long as the host builds it.
Alternatives: keep the window types in the physics module (their layout is the legacy one,
replaced in 7.3); keep `TimeInterpolation2` (unreachable code, and Fortran refuses it).

### C13. Process 4: `clm_ml/physics/solar_radiation.py`; Norman kept in the legacy adapter; `clm_ml/physics/layers.py` (7.2, 2026-10-31)
Context: `MLSolarRadiationMod` moves to `clm_ml/physics/` (scope 7.2 process 4: two-stream,
Norman registered as unsupported per D8, the 150-term sum as a scan in the same order).
Decision:
- **Physics on one patch, 0-based:** `canopy_optics` (leaf/stem optics, leaf angle, `kb`,
  `tb`, `td`, `tbi`, sunlit fraction, two-stream parameters; returns `CanopyOptics`, 0 outside
  the leafy layers, plus the sunlit-fraction check), `two_stream` (both sweeps as before;
  returns `SolarFluxes` with levels = ground + layers, leaves 0 sun / 1 shade, and the solar
  conservation error), `absorbed_par`. The layer grid is a `clm_ml.state.PatchLayers` (the
  adapter builds it with the new `clm_ml.legacy.state.mlcanopy_layers`). Parameters enter as
  plain arrays (0-based bands), not the `CanopyParams` container.
- **The 150-term sum** (vegetation, sunlit and shaded absorption, 50 slots x 3) is a
  `lax.scan` bottom to top with the same additions (`(veg + sun) + sha`).
- **Registry:** `SOLAR_RADIATION = {"two_stream": two_stream, "norman": norman}`; `norman`
  raises `NotImplementedError` (D8). The legacy Norman translation (`_Norman`, never
  validated, stops at the first CHATS7 step) is moved unchanged into the adapter module
  `clm_ml/legacy/solar_radiation.py` and still runs for `light_type = 1` (which `clm_ml.run`
  refuses), so the code is kept for the fix after Phase 8 without porting unvalidated,
  unrolled code into the physics package. The optics it needs (`rho`, `tau`, `omega`) come
  from `canopy_optics`.
- **Adapter:** `SolarRadiation` keeps its legacy arguments; writes leafy-layer values, 0 in the
  other layers in use, keeps padding (as before); `DECLARED` reads/writes (the layer-grid
  fields `dz/zs/zw_profile` are declared reads because `PatchLayers` holds them).
  `leaf_optics_type = 1` still stops through `endrun` (unchanged).
- **`clm_ml/physics/layers.py`:** `keep`, `pad_with_top`, `leafy_or_top_leafy` on 0-based
  `PatchLayers` slots, for the processes that follow; `MLLayersMod` (1-based) stays for the
  legacy code until process 10 (a small duplication for the transition).
- **Constants** `rpi`, `J_to_umol`, `chil_min/max`, `kb_max` through `physics/constants.py`.
- **Bit for bit** end to end (177 pass); the process replay of `SolarRadiation` passes.
  `MLSolarRadiationMod.py` deleted, no shim. New `tests/clm_ml/test_physics_solar_radiation.py`
  (declared reads/writes, energy conservation, Beer's law onto the ground, zero outside the
  leaves, Norman refused).
- No gradient test: nothing with a hand-written gradient moved (package-end tier covers it).
Why: the physics on plain arrays is the aim of 7.2; keeping Norman verbatim honours D8
("kept") without spending effort on a scheme to be redone and validated later.
Alternatives: port Norman to 0-based in the physics module now (unvalidated code, its
unrolled 2 x max_layers solve, against "no new unrolled loops"); delete it (D8 keeps it).

### C14. Process 5: `longwave.py`, `canopy_water.py`, `leaf_heat_capacity.py`, `plant_hydraulics.py`; leaf water potential over the leaf axis (7.2, 2026-10-31)
Context: scope 7.2 process 5 (longwave, canopy water, leaf heat capacity, plant hydraulics;
sun/shade water potential over the leaf axis).
Decision:
- **Physics on one patch, 0-based** (layers on axis 0, leaves 0 sun / 1 shade, soil levels
  0-based), each process returning plain arrays or a `NamedTuple`:
  `longwave.norman` (-> `LongwaveFluxes`; registry `LONGWAVE_RADIATION = {"norman": ...}`,
  the name model.yaml `physics.longwave_radiation` already uses), `canopy_water.
  wetted_fraction`, `interception` (-> `Interception`), `evaporation`;
  `plant_hydraulics.plant_resistance`, `soil_resistance` (-> `SoilResistance`),
  `leaf_water_potential`; `leaf_heat_capacity.leaf_heat_capacity`. Parameters enter as
  numbers, not the `CanopyParams` container; the time step `dtime` as an argument.
- **Leaf heat capacity in its own module** (`physics/leaf_heat_capacity.py`; the scope names
  the function, not a file): it is a leaf thermal property, unrelated to canopy water or
  hydraulics. Its adapter `LeafHeatCapacity` lives with the hydraulics adapters
  (`clm_ml/legacy/plant_hydraulics.py`), which are called with it at the start of each step
  (one adapter module fewer for a 15-line transition shim).
- **Leaf water potential over the leaf axis:** one call computes the sunlit and shaded leaves
  ((layer, leaf) arrays); the legacy adapter `LeafWaterPotential` drops its leaf argument
  and `MLCanopyFluxesMod` calls it once instead of twice (the two calls were independent).
- **Longwave** keeps the legacy one-system formulation (all slots, sky row after the real
  rows, identity rows; `tridiag` without `min_pivot`, as before, C9). `lwleaf` is returned
  per leaf (equal for sun and shade, as Norman gives) so later consumers need no broadcast.
  The adapter writes levels 0..ntop and keeps the levels above (unchanged behaviour).
- **Adapters** in `clm_ml/legacy/{longwave,canopy_water,plant_hydraulics}.py` with the old
  names and `DECLARED` reads/writes; the `jax.jit` wrappers of the legacy routines were
  dropped (they run inside the compiled chunk). The old modules are deleted, no shim.
- **Found, not fixed:** `SoilResistance` reads the ice of soil level `j + 1` for level `j`
  (legacy translation; Fortran reads level `j`). No effect today (ice is always 0), kept
  unchanged and recorded as open issue 19 in `global_input.md` for the maintainer.
- **Golden tests repointed:** `MLCanopyWaterMod_WettedFraction` now calls
  `canopy_water.wetted_fraction` (it re-implemented the formula in Python before),
  `MLLeafHeatCapacityMod_LeafHeatCapacity` calls `leaf_heat_capacity`. No plant-hydraulics
  golden data exist. New tests `tests/clm_ml/test_physics_longwave.py` (declared
  reads/writes, energy conservation, radiative equilibrium of an isothermal canopy) and
  `tests/clm_ml/test_physics_canopy_water_hydraulics.py` (declared reads/writes of the seven
  adapters, water conservation of interception, evaporation, wetted-fraction limit, uptake
  fractions summing to 1 above bedrock, leaf water potential relaxing to its steady state).
- **Round-off change, reference re-recorded (D7):** every moved routine is bit-identical to
  the old code under `jax.jit` on the same inputs (probe `probe73/compare.py`; eager calls
  differ in last bits only because the old routines carried their own `jax.jit`), but the
  compiled chunk is not: 53 fields differ from step 0, worst 1.28x the noise-twin spread
  (`gs_leaf`, 3.8e-12; `probe73/ratio.py`), noise twin and process replay pass. As in C10,
  XLA compiles the chunk differently; reference and twins re-recorded in this commit.
- No gradient test: nothing with a hand-written gradient moved.
Why: the aim of 7.2 (physics on plain arrays, files by process); the leaf axis halves the
water-potential calls as the scope asks.
Alternatives: leaf heat capacity inside `canopy_water.py` or `plant_hydraulics.py` (not
the same process); fixing the ice index now (a science change inside a refactor).

### C15. Process 6: `clm_ml/physics/turbulence.py`; the psihat tables as an argument, read from the package; values rounded as stored (7.2, 2026-10-31)
Context: scope 7.2 process 6 (`MLCanopyTurbulenceMod`, 1 500 lines; "psihat table packaged as
data, RSL lookup tables as an input, not a module global"). The legacy module filled 6
arrays of `MLclm_varcon` and 16 module caches at initialization (`LookupPsihatINI`), which
every psihat lookup read.
Decision:
- **Physics on one patch, 0-based** (`PatchLayers` slots): the Monin-Obukhov functions,
  `lookup_psihat`, `prandtl_schmidt`, `psi_rsl` (and `psi_rsl_at` over heights),
  `beta_stability`, `obukhov_terms` (one function for the residual and the write-back, which
  the legacy code had twice), `obukhov_length` (25 secant steps, implicit-function
  gradient, the same arithmetic), `roughness_length`, `wind_profile`,
  `aerodynamic_conductance` (-> `Conductances`) and the process `harman_finnigan`
  (-> `Turbulence`); registry `ROUGHNESS_SUBLAYER = {"harman_finnigan": ...}` (model.yaml
  `physics.roughness_sublayer`). Fortran names became descriptive ones
  (`GetBeta` -> `beta_stability`, `GetPrSc` -> `prandtl_schmidt`, `GetPsiRSL` -> `psi_rsl`),
  each docstring keeping its `Fortran reference:`.
- **The psihat tables:** `RSLTables` (momentum and scalar `PsihatTable`s: grids and values,
  1-D grids instead of Fortran's `(nZ, 1)`/`(1, nL)` arrays) is an **argument** of every
  routine that interpolates psihat. The file moved to `src/clm_ml/data/psihat.nc` (package
  data, `pyproject.toml`); `clm_ml/io/rsl_tables.py` reads it (`read_rsl_tables(path)`,
  checking each table's shape against its grids instead of Fortran's fixed 276 x 41) and
  `default_rsl_tables()` caches the package's tables per process (immutable). The
  `MLclm_varcon` table arrays, `nZ`/`nL`, `clm_varctl.rslfile` and `LookupPsihatINI` are
  gone; `clm_initializeMod` reads the tables at initialization (fails early if missing).
- **Not yet an input of the compiled step:** the legacy adapter passes
  `default_rsl_tables()`, so the tables are a constant of the compiled chunk, as before.
  Making them a runtime input means changing every chunk runner's signature
  (`make_chunk_runner`, `make_stream_runner`, `make_batch_runner`, the sharded wrapper,
  `clm_ml.model`); that belongs with the step's new signature when the containers leave the
  step (process 10) and the setup without globals (7.4). The physics already takes them as
  an argument, so that change touches only the driver.
  **Maintainer (2026-10-31): document it as a fix to make at process 10**; listed in
  `phase7_progress.md`, "Fixes to make at process 10".
- **Values rounded as stored:** `harman_finnigan` passes the canopy-top terms
  (`obu`, `beta`, `zdisp`, `PrSc`, `ustar`, `gac_to_hc`) through
  `jax.lax.optimization_barrier` before using them. Without it, XLA folded
  `u(h) = ustar / beta` with the division inside `ustar` (`(a / b) / beta -> a / (b beta)`),
  1 ulp off the legacy value in 2 of the 6 recorded calls (`probe74/compare.py`); with it,
  every recorded call is bit-identical, eager and compiled. Fortran stores these values
  too (`ustar(p)`, ...), and the legacy code wrote them to the container. No effect on
  gradients (the barrier is the identity for differentiation).
- **Ground conductance by name:** `aerodynamic_conductance(..., ground_conductance)` with
  `GROUND_CONDUCTANCE = ("extend_hf", "log_profile")` (model.yaml
  `physics.ground_conductance`, Fortran `HF_extension_type` 1, 2); any other name raises
  `ValueError`. Fortran has no default case (gac0 keeps its last value) and the
  configuration accepts only 1 and 2, so the adapter now raises for other values instead of
  carrying the previous conductance (unreachable through `clm_ml.run`, as C12's two-point
  interpolation).
- **Adapter:** `clm_ml/legacy/turbulence.py::CanopyTurbulence` with the legacy arguments
  (the process replay and `MLCanopyFluxesMod` unchanged apart from the import); profiles
  written in the layers in use, padding kept (as before); the `eta >= eta_max` warning
  printed in step-by-step runs, from the physics' `eta`; `DECLARED` reads/writes. The
  physics call is one `jax.jit` (as the legacy `_HF2008` was), so step-by-step runs don't
  dispatch thousands of small operations. `MLCanopyTurbulenceMod.py` deleted, no shim.
- **Golden tests repointed** (phim/phic, psim/psic, GetBeta, GetPrSc, GetPsiRSL: 59 pass).
  New `tests/clm_ml/test_physics_turbulence.py`: declared reads/writes, the Obukhov length is
  the residual's root or the Lc/L limit (stable to unstable), the wind profile gives `uref`
  at `zref` and is continuous at the canopy top, the roughness length is its residual's root,
  conductances and eddy diffusivities, the lookup on and beyond the grid, the Obukhov
  length's implicit gradient against finite differences.
- **Bit for bit** end to end (177 pass, no reference change).
Why: the scope's aim (physics on plain arrays, the tables as data and an argument); keeping
the tables a compile-time constant until the step's signature changes avoids touching five
runners twice.
Alternatives: thread the tables through the chunk runners now (twice the driver work, see
above); tables per patch in `CanopyParams` (vmapped: 180 kB copied per patch); keep the
1 ulp difference and re-record the reference (avoidable, and the stored-value rounding is
what Fortran does).

### C16. Process 7: `leaf_boundary_layer.py`, `photosynthesis.py`; sunlit and shaded leaves in one call; `wue_solver: brent` refused (7.2, 2026-10-31)
Context: scope 7.2 process 7 (`MLLeafBoundaryLayerMod`, `MLLeafPhotosynthesisMod`, 2 000
lines with four near-copies of the layer kernel: Medlyn/Ball-Berry and WUE, each with and
without acclimation; "sun and shade as one vmapped call over the leaf axis ... round-off only
if XLA fuses differently: check"); open issue 18 to settle with the maintainer.
Decision:
- **Physics on one patch, 0-based, layers on axis 0 and leaves on axis 1**, as array
  (broadcasting) code rather than a `vmap` of a scalar kernel: one kernel instead of four (the
  acclimation parameters are Python floats without acclimation, so they stay constants of the
  compiled program as the closed-over values of the old kernels were).
  `leaf_boundary_layer.leaf_boundary_layer` (-> `BoundaryLayerConductance`; regimes by name,
  `REGIMES = ("laminar", "turbulent", "free_convection")`, the names model.yaml
  `physics.leaf_boundary_layer` uses); `photosynthesis.leaf_photosynthesis`
  (-> `Photosynthesis`, the 21 `*_leaf` fields) built from `ft`, `fth`, `fth25`,
  `temperature_parameters`, `leaf_rates`, `electron_transport`, `realized_rate` (the pathway
  and co-limitation as static switches instead of `jnp.where` on the `c3psn` value),
  `photosynthesis_at_gs` (CiFuncGs), `stomata_efficiency`, the solvers `solve_ci` (40 secant
  steps + Newton step) and `solve_gs_bisection` (20 halvings + Newton step), unchanged in
  arithmetic, each docstring with its `Fortran reference:`. Stomatal models by name
  (`STOMATAL_CONDUCTANCE = ("medlyn", "ball_berry", "wue")`). Parameters enter as numbers
  (`iota`, `gs_min`, `psi50_gs`, `shape_gs`, `g0`, `g1`; C14), `stomatal_parameters` picks
  `g0`, `g1` of the model from the PFT's `CanopyParams`. The photosynthesis pads its inputs
  with the top layer's (takes `PatchLayers`), as the legacy routine did; the boundary layer
  computes on every slot without padding, as before.
- **One masking rule for absorbed PAR:** the old Medlyn/Ball-Berry kernels did not replace
  `apar` by 1 in layers without plant area, the WUE kernels did; now all do. Forward values
  are unchanged (every result of such a layer is zeroed, and the water-stress pass reads the
  container's `apar`); it only removes a difference between copies.
- **Issue 18 (maintainer, 2026-10-31): `wue_solver: brent` is refused**, not translated:
  `WUE_SOLVER = ("bisection", "brent")` with `brent()` raising `NotImplementedError` (D8
  pattern), and `clm_ml.run` stops with a `ConfigError` for `brent` with
  `stomatal_conductance: wue` (Fortran ignores `gs_solver` for the other models, so does the
  JAX). Listed in the plan under "After Phase 8: unsupported variants".
- **Adapters** `clm_ml/legacy/leaf_boundary_layer.py::LeafBoundaryLayer` (both leaves; it
  replaces `LeafBoundaryLayerBoth`) and `clm_ml/legacy/photosynthesis.py::LeafPhotosynthesis`
  (`il = None`: both leaves in one call, used by `MLCanopyFluxesMod`; `il = isun/isha`: one
  leaf, as the process replay records it); results written in the layers in use, padding
  kept (as `restore_padding` did); `btran_soil = 1`, `g0_canopy`/`g1_canopy` for Medlyn and
  Ball-Berry only; invalid option codes raise `ValueError` (Fortran `endrun`); `DECLARED`
  reads/writes. Each physics call is one `jax.jit` (as the old kernels were). Both legacy
  modules deleted, no shim. New constants re-exported by `physics/constants.py`; `SatVap`
  still comes from `MLWaterVaporMod` (moves with the flux-profile solution or 7.6).
- **Golden tests repointed** (`ft`, `fth`, `fth25`, `RealizedRate`, `LeafBoundaryLayer`; 160
  fast golden tests pass). New `tests/clm_ml/test_physics_photosynthesis.py`: declared
  reads/writes of both adapters (all three stomatal models), one leaf at a time = both,
  regimes ordered and floored, diffusion equation, Ball-Berry and Medlyn satisfy their models
  (Fortran's error checks; without co-limitation, because the water-stress pass recomputes
  photosynthesis with `ac` and `aj` solved separately), the WUE optimum is the root of the
  efficiency check, low light gives `gs_min`, the water-stress rule, C4 and the options,
  `brent` refused, and d gs / d vcmax25 against finite differences for WUE and Medlyn (the
  implicit-function gradients).
- **Round-off change, reference re-recorded (D7):** under `jax.jit`, the new code called one
  leaf at a time is bit-identical to the old code on the six recorded calls, and the boundary
  layer is identical (probe `probe77/compare.py`). Computing both leaves in one (layer, 2)
  array compiles differently (last bits of `je`, which the WUE efficiency check, a
  difference of net photosynthesis 0.001 apart in gs, amplifies to ~3e-12 in gs): 53 fields
  differ from step 0, worst 1.24x the noise-twin spread (`lhvegsha_canopy`;
  `probe77/ratio.py`), noise twin and process replay pass. Reference and twins re-recorded
  in this commit.
- **Process replay: fields that are zero to round-off compared absolutely.** The replay of
  `LeafPhotosynthesis` at night (step 14) failed with "relative difference 1.0" in `aj_leaf`
  and `agross_leaf`: the recording (old code, run step by step) has 0, the new adapter
  2.2e-16, because `aj = root + rd` with `root = -rd` in the dark; the old code compiled
  gives the same 2.2e-16 (`probe77/replay14.py`: new = old under `jit` in every call), so
  the running model did not change, only the adapter now compiles the whole routine.
  Relative to a field whose largest value is 1e-16, any round-off reads as 1.
  `process_io._max_rel_diff` now compares a field smaller than `ZERO_FIELD = 1e-12`
  everywhere in the recorded call by its absolute difference; all other fields as before.
Why: the scope's aim (physics on plain arrays, one kernel over the leaf axis instead of two
calls and four copies); refusing `brent` is the maintainer's choice for issue 18.
Alternatives: keep one call per leaf (bit for bit, but not the scope's leaf-axis call and
twice the photosynthesis program); `vmap` of a scalar kernel (the old form; broadcasting
keeps Python-float constants constant without care about how `vmap` passes unbatched
arguments); a traceable Brent now (not validated against Fortran; maintainer chose refusal).

### C17. Process 8: `clm_ml/physics/flux_profile.py` and `water_vapor.py`; `well_mixed` refused (7.2, 2026-10-31)
Context: scope 7.2 process 8 (`MLFluxProfileSolutionMod`, with `MLSoilFluxesMod`; WellMixed
registered as unsupported, D8); the progress log left open where `SatVap`/`LatVap`
(`MLWaterVaporMod`) go, and `MLLeafFluxesMod` (called by Fortran's implicit solution and by
WellMixed) is not named in the scope.
Decision:
- **`clm_ml/physics/water_vapor.py`**: `sat_vap`, `lat_vap` (same polynomials, same Horner
  nesting, same `where` branches; elementwise on any shape). `SatVap_py` (a Python-float copy
  nothing used) not moved. All importers repointed (photosynthesis, `MLCanopyFluxesMod`, the
  tower forcing reader, the `.out` writer, the golden tests `SatVap`/`LatVap`); `hsub`,
  `hvap` re-exported by `physics/constants.py`.
- **`clm_ml/physics/flux_profile.py`** on one patch (0-based layers, leaves on axis 1):
  `leaf_fluxes` (Fortran `LeafFluxes`, elementwise, both leaves at once), `soil_fluxes`
  (`SoilFluxes`), `implicit` (`ImplicitFluxProfileSolution`; inputs grouped as `Canopy`,
  `Ground`, `Reference`; result `FluxProfile` with `LeafFluxes`, `SoilFluxes` and the
  implicit solution's own fluxes `ImplicitFluxes`). The arithmetic is the legacy kernel's,
  which was already array code: padding with the top layer, identity rows after the top
  layer, `tridiag_2eq` (its `custom_vjp` unchanged). The leaf fluxes are `leaf_fluxes` at
  the solved profiles, as Fortran calls `LeafFluxes` there (the legacy kernel had inlined
  the same formulas). `FLUX_PROFILE = {"implicit", "well_mixed"}`, `well_mixed` raising
  `NotImplementedError` (D8 pattern; Fortran itself stops after WellMixed: "legacy code, not
  supported"), and `clm_ml.run` stops with a `ConfigError` for
  `physics.flux_profile_solution: well_mixed` (as for `norman`, `brent`).
- **Fortran's checks of the implicit solution as a function, not in the step:**
  `implicit_errors` computes the 15 checks of `ErrorCheck01` (implicit solution vs
  `LeafFluxes`/`SoilFluxes`) and `ErrorCheck02` (conservation in each layer, the canopy and
  at the ground) as array code; the unit test runs them. The legacy translation ran them only
  with a debug flag (`DEBUG_FPS_CHECKS = False`, host loops), so the time step never did;
  this is unchanged (adding them to `BALANCE_CHECKS` changes the state: open issue 20 in
  `global_input.md`, for the maintainer). The flag and its host loops are gone.
- **Adapter** `clm_ml/legacy/flux_profile.py::FluxProfileSolution` (the old name, imported by
  `MLCanopyFluxesMod`; one `jax.jit` per patch as before): results in the layers in use,
  padding kept; the unused leaf index 0 set to 0 in the layers in use (as the legacy code
  wrote it); `cair` set in levels 0..ncan-1 as translated (numerics doc §4.5); `DECLARED`
  reads/writes. The WellMixed legacy translation is kept in the adapter (runs, then
  `endrun`, as Fortran), with its leaf and soil fluxes from `physics.leaf_fluxes`/
  `soil_fluxes` (the legacy `LeafFluxes`/`SoilFluxes` had the same formulas) so that
  `MLLeafFluxesMod` could go. `MLFluxProfileSolutionMod`, `MLSoilFluxesMod`,
  `MLLeafFluxesMod`, `MLWaterVaporMod` deleted, no shim.
- **Round-off change, reference re-recorded (D7):** on the six recorded calls the new
  adapter equals the old one in every written field except `etair_profile` and
  `stair_profile` (all layers, up to 1.8e-13 relative), eager and under `jit`
  (`probe78/compare.py`). The old values equal numpy evaluated in source order on the
  output `eair`; the new ones do not (`probe78/detail2.py`: un-jitted physics matches numpy,
  jitted does not). The compiled HLO recomputes `eair = q pref` inside the fusion that forms
  `-(eair(i+1) - eair(i)) / pref * gac` (no container write in between any more), where the
  multiply and the subtraction are compiled together; the difference of two close vapor
  pressures amplifies the last-bit change. An `optimization_barrier` (C15) does not help:
  it is gone from the optimized HLO before fusion. End to end 53 fields differ from step 0,
  worst 1.33x the noise-twin spread (`lwp_mean_profile`; `probe78/ratio.py`); noise twin and
  process replay pass. Reference and twins re-recorded in this commit.
Why: the scope's aim (physics on plain arrays by process, variants by name); keeping the
legacy arithmetic in the same order keeps every field but the vertical vapor flux bit for bit
in isolation; `well_mixed` refused because Fortran refuses it.
Alternatives: keep `MLLeafFluxesMod` for WellMixed only (a dead module); recording the
implicit-solution checks in `BALANCE_CHECKS` now (changes the state and the reference's
`balance_error`; a decision for the maintainer, issue 20); fighting the fusion with
container-like writes (fragile: the old code matched numpy only because of where XLA cut
the fusions).

### C18. Process 9: `clm_ml/physics/nitrogen_profile.py` and `integrators.py` (7.2, 2026-10-31)
Context: scope 7.2 process 9 (`MLCanopyNitrogenProfileMod`, `MLRungeKuttaMod`); the scope
does not say how the Runge-Kutta tableaus are chosen or what the integrated state looks like.
Decision:
- **`clm_ml/physics/nitrogen_profile.py`** on one patch (0-based layers, leaves on axis 1):
  `canopy_top_rates` (Vcmax, Jmax, Rd, Kp at the canopy top; C3/C4 and acclimation as static
  choices, Python branches instead of `jnp.where` on a constant), `decay_coefficient` (kn;
  `kn_val` stays a static option, as before: making it a differentiable parameter is a
  change of interface, not of structure), `nitrogen_scale` (Fortran's per-layer formulas,
  elementwise), `canopy_nitrogen_profile` (all layers at once, top down, the legacy
  arithmetic and order: the exclusive cumulative sum on the reversed axis, the 1e-30
  floors); the integration error is returned and the adapter records it as `NITROGEN`. The
  invalid-option `endrun` branches are gone (model.yaml offers only the valid ones; the
  adapter raises on other codes).
- **`clm_ml/physics/integrators.py`**: the tableaus as numpy constants registered by the
  model.yaml names (`TABLEAUS`, `numerics.integrator`; `euler` has one stage and no update,
  as Fortran never calls `RungeKuttaUpdate` for it); `config._RK_CODES` renamed
  `INTEGRATOR_CODES` (public: the adapter maps Fortran codes to names with it).
  `runge_kutta_update(stage, tableau, layers, bef, cur, increments)` on the six integrated
  variables (`Integrated`: air temperature, vapor pressure, intercepted water, leaf
  temperature, leaf water potential, ground temperature), the legacy arithmetic (earlier
  stages summed, then this stage's term: the legacy order, not Fortran's one-by-one sum,
  numerics doc §4 unchanged); it returns the state, this stage's increments and the stage
  index; the adapter writes the increments into the container's stage axis as before.
- **Adapters** `clm_ml/legacy/nitrogen_profile.py::CanopyNitrogenProfile` (one `jax.jit` of
  the physics per call, as the legacy routine was jitted) and
  `clm_ml/legacy/integrators.py::RungeKuttaUpdate(irk, tableau, inst, grid)` with
  `integrator_tableau(opts)` (replaces `RungeKuttaIni`; the unused `c` and filter arguments
  dropped); `DECLARED` reads/writes. The legacy leaf axis keeps its unused index 0 updated
  with the leaves, as before. Both legacy modules deleted, no shim.
- **Process replay**: the recorded `RungeKuttaUpdate(irk, a, b, c, num_filter, filter,
  inst)` calls are replayed as `(irk, Tableau(a, b, c), inst)` (`process_io._new_arguments`).
- **Golden tests**: `RungeKuttaIni` reads `TABLEAUS["rk4"]`; `NitrogenScale` calls
  `nitrogen_scale` (it had re-implemented the formulas in the test); the
  `CanopyNitrogenProfile` golden cases (5, skipped until now: "requires full container
  initialization") run against `canopy_nitrogen_profile`.
- **Bit for bit** in isolation (old HEAD files vs new adapters, every container field, eager
  and `jit`, the six recorded RK calls at stages 1-5, the nitrogen profile on the three
  recorded states with the default options, without acclimation, with
  `leaf_optics_type = 1` and with `kn_val = 0.3`; `probe79/compare.py`). A first version
  differed by 1 ulp with `leaf_optics_type = 1` (scaling factor from the unmasked `fn`);
  that branch uses the masked `fn` as before. **End to end** a version that formed
  `fn_sha = fn - fn_sun` before masking (identical in isolation) changed the shaded leaves'
  `vcmax25_leaf` by 1 ulp inside the compiled chunk (117 values, from step 0; 55 fields
  differed, worst 1.54x the twin spread; `probe79/cur.npz`), a field the twins do not move:
  the subtraction compiled differently once it followed the multiplications directly
  (probably a fused multiply-add; not checked in the HLO). `canopy_nitrogen_profile` masks
  `fn` and `fn_sun` before combining them, as the legacy code: bit for bit end to end,
  reference unchanged. The physics is therefore split as `nitrogen_fractions` (fn, fn_sun)
  and `nitrogen_scale` (the factors; used by the golden test).
Why: the scope's aim (physics on plain arrays by process, variants by name); the legacy
arithmetic kept so nothing moves.
Alternatives: tableaus as a pytree argument of the step (they are constants of the method,
which is static); a differentiable `kn` (interface change, for the maintainer if wanted).

### C19. Process 10 split into four commits; 10a: `clm_ml/physics/canopy.py` (7.2, 2026-10-31)
Context: scope 7.2 process 10 (the step driver; the legacy container leaves the step; the
psihat fix, C15). The progress log asked for a plan and a split first. `MLCanopyFluxes`
today: the step's set-up on the containers (CLM variables, plant area, plant hydraulics,
soil humidity), the sub-step loop of processes 3-9 through their container adapters, the
averages over the sub-steps, `CanopyFluxesDiagnostics`, the sunlit/shaded merge; around it
`clm_drv` (the soil routines) and the boundary conversions template + `State` -> containers
-> `StepState` in `offline_executable/main.py`.
Decision: **four sub-steps, each its own commit (and session if needed):**
- **10a (this commit):** the step-level pieces of `MLCanopyFluxesMod` on 0-based arrays in
  `clm_ml/physics/canopy.py`: `plant_area` (with the issue-14 floor), `soil_surface_humidity`,
  the sub-step averages (`SubstepFluxes`, `zero_sums`, `accumulate`, `average`,
  `canopy_top_latent_heat`), `canopy_totals` (Fortran `CanopyFluxesDiagnostics`; inputs the
  averages and `CanopyLeaves`), `merge_sun_shade`. Adapters in `clm_ml/legacy/canopy.py`
  with `DECLARED` reads/writes; `_MLAccumulateFluxes`, `_MLScaleAndWriteBack`,
  `_CanopyFluxesDiagnostics` and the inline plant-area, humidity and merge code deleted.
- **10b:** the sub-step on plain arrays: `canopy.substep` calling the physics of processes
  3-9 directly on a 0-based within-step carry (the RK stages, the forcing interpolation, the
  sums); `MLCanopyFluxes` converts its container to that carry once per step and back once.
  The process replay (`process_io`, which replays the container adapters) is re-recorded in
  the new structure or retired then (scope §3.2), and the adapters of processes 3-9 go or
  shrink to what the first step still needs.
- **10c:** the step on `StepState`: the soil routines of `clm_drv`, `_GetCLMVar` (forcing,
  solar zenith) and the step's set-up as one step function on `State` + step inputs +
  per-patch constants (a 0-based pytree extracted once from the template; setup stays legacy
  until 7.4); `instances_with_state`/`step_state_from_instances` leave the time step (kept
  for the first step and the view); the legacy view fills the entries the state does not
  hold (index 0, padding) from the template.
- **10d:** the psihat tables as a run-time input of the compiled step (C15, maintainer).
Then the package-end tests (CPU gradient tier, independent review, timing against `p71-*`).

10a details:
- **D6:** `canopy_totals` returns only what outputs, checks and the CLM coupling read; the
  sunlit/shaded longwave and water-vapor totals, `etveg` and the 23 layer source and mean
  profiles are no longer computed (listed in `phase7_removed_diagnostics.md`).
- **Unused legacy slots kept as recorded:** the reference stores the raw legacy arrays,
  index 0 included, and later steps take index 0 from the first step's containers. The
  legacy averages covered index 0 of the leaf fields (six copies of the 1e36 fill average to
  9.999999999999999e35), and the merge copied leaf index 0 into the `*_hist_leaf` fields.
  The adapter therefore accumulates the leaf fields' full legacy slabs (`zero_based` gives
  the physics its 0-based part) and writes leaf index 0 of the history fields. Junk, but
  recorded junk: changing it would change the reference for no physical reason. It goes
  when the reference is re-recorded in the new names (7.6).
- **Round-off change, reference re-recorded (D7):** eager (op by op) the new
  `MLCanopyFluxes` equals the old one in every container field over CHATS7 steps 1-7, except
  the dropped diagnostics (`probe710/eager.py`). The compiled chunk differs in 4 fields from
  the start: `albcan_canopy` (12 values), `gppveg_canopy`, `anet_leaf` (3 values at step 0),
  `rnet_canopy` (3 values), all at the last bit (worst 2e-16 relative; noise twin passes:
  `albcan` is not moved by the twins but is within the test's relative floor). Probably
  fused multiply-adds now formed between the averaging and the totals, where the legacy
  container stores separated them (C18 pattern; not checked in the HLO). An
  `optimization_barrier` on the averages before the totals (as Fortran stores them) is kept:
  without it 54 fields differed at round-off (worst 1.7x the twin spread); reading the
  averages back from the container instead (the legacy data path) also gave 53 fields.
  Reproducing the old packed accumulators in the adapter was not tried: 10b changes the
  sub-step structure anyway. Reference and twins re-recorded in this commit (only the 4
  fields change).
Why: the progress log's plan-first instruction; one reviewable commit per sub-step, each
bit for bit or a recorded round-off change; 10a moves the pieces that are independent of the
sub-step carry, so 10b and 10c only change the data flow.
Alternatives: one large commit for the whole driver (unreviewable, hard to bisect); keeping
the removed diagnostics (D6 says drop them when their process moves); faking index 0 in the
view now (the view takes index 0 from the template, which is the first step's containers:
the first step must produce the recorded values until 7.4).

### C20. Process 10b: the canopy sub-step on plain arrays (7.2, 2026-10-31)
Context: C19's plan; the sub-step still ran processes 3-9 through their container adapters
(every routine reading and writing the 235-field container in the scan carry).
Decision:
- **`clm_ml/physics/canopy.py::substep(carry, calday, inputs, params, *, options, c3)`**:
  one sub-step on 0-based arrays: `save_start` (the `*_bef` values), `_radiation` (forcing
  interpolation, solar radiation, nitrogen profile), the Runge-Kutta stages as one
  `lax.scan` over `_stage` (wetted fraction, longwave, net radiation, turbulence, leaf
  boundary layer, photosynthesis, implicit flux-profile solution, leaf water potential,
  interception and evaporation, the state update), then the sub-step's `SubstepFluxes`
  added to the sums. The physics of processes 3-9 are called directly, with each legacy
  adapter's write rules (layers in use, padding kept; `cair` in layers 0..ncan-2, numerics
  doc §4.5) and checks recorded in the carried `balance_error`.
- **The carry `canopy.Substep`**: the values a sub-step reads or writes besides the step's
  constants, named as the legacy container fields (renamed with the outputs in 7.6), with
  their kinds (`SUBSTEP_KINDS`: scalar, band, layer, level, leaf, stages, ...); every slot.
  **`SubstepInputs`**: the values fixed over the step's sub-steps (layer grid, forcing
  window and its calendar days, plant area, ground albedo, soil inputs, leaf heat capacity,
  leaf-specific conductance, soil water potential). **`SubstepOptions`**: the static
  choices by model.yaml name; `clm_ml.legacy.canopy.substep_options(opts)` maps the
  Fortran codes and makes the legacy adapters' checks. Norman solar radiation and the
  well-mixed flux profile now stop before the step (`NotImplementedError`) instead of
  running their legacy translations and then stopping (D8; the translations stay in
  `clm_ml.legacy.solar_radiation`/`flux_profile`, unused).
- **`MLCanopyFluxes`** reads the container once before the sub-steps
  (`substep_from_container`, `substep_inputs`) and writes it once after them
  (`container_with_substep`). Compiled chunks trace `canopy.substep` in the sub-step
  `lax.scan` (with `jax.checkpoint`, as before); step-by-step runs call it compiled once
  (`MLCanopyFluxesMod._substep`, one program for the run). The turbulence's `eta >=
  eta_max` warning, printed only in step-by-step runs, is gone (it never printed in
  compiled chunks).
- **D6:** values no process, average, total or output reads are no longer written
  (turbulence `Lc`, `taf`, `qaf`, `PrSc`, `gac_to_hc`, `uaf`; `eg`; 18 photosynthesis
  fields; `g0`, `g1`): listed in `phase7_removed_diagnostics.md`.
- **The unused legacy index 0 is no longer written by the step** (layer slot 0 of non-level
  fields, leaf 0, band 0): the 0-based carry has no such slots. The container keeps the
  setup's values there, so the view's index-0 entries (from the template, C19) are setup
  fill values, as 7.1 intended ("constant fill values, taken from the template"). C19's
  replication of the averaged junk (leaf fields' full slabs, `*_hist_leaf` leaf 0) is
  removed: `SubstepFluxes` and the averages are 0-based. The reference's index-0 entries
  change with the re-recording (e.g. `shleaf_leaf`, `lhleaf_leaf`, `tleaf_hist_leaf`:
  0 or averaged values -> the 1e36 fill).
- **Process replay unchanged:** it replays the container adapters of processes 3-9, which
  the step no longer calls; they still check the same physics against the recorded
  pre-Phase-7 calls, so they stay (re-exported by `MLCanopyFluxesMod`, where the replay
  looks routines up by name) until 10c decides their end with the containers.
- **Round-off change, reference re-recorded (D7):** run op by op (`jax.disable_jit`, no
  compilation in either), the new `MLCanopyFluxes` equals the old one in every container
  slot over CHATS7 steps 2-3, except the unused index 0 and the dropped values above
  (`probe710b/exact_masked.py`). The compiled chunk differs in 60 of 88 recorded fields
  (index 0 masked): those the twins move within 1.65x the twin spread (from step 0); the
  twin-invariant solar fields (`apar_leaf`, `swveg*`, `swdwn`, `swupw`, `albcan`) by a few
  ulps (at most 2.2e-14 relative) from step 2, with identical inputs (forcing, zenith
  angle, plant area): probably XLA hoisting the sub-step-invariant optics out of the
  sub-step loop now that the step's constants are not part of the carried container (not
  checked in the HLO); an `optimization_barrier` on the interpolated forcing changed
  nothing (not kept). Before re-recording the noise-twin test failed only on index-0
  entries (`shleaf_leaf`, `lhleaf_leaf`, `tleaf_hist_leaf`: finite in the reference, the
  1e36 fill now). Reference and twins re-recorded in this commit; focused CPU gradient
  test passes.
Why: the scope's process 10 (the step on plain arrays), in the order of C19; one carry
instead of the container makes 10c (the step on `StepState`) a change of the step's
boundary only.
Alternatives: a carry of separate NamedTuples per process (more types, same values);
keeping the legacy index-0 values (they cannot be computed without legacy-shaped arrays);
retiring the process replay now (it still tests the physics on real inputs).

### C21. Process 10c: the time step on the step state (7.2, 2026-10-31)
Context: C19's plan; after 10b the compiled time step still rebuilt the legacy containers
from the template and the state every step (`instances_with_state`), ran `clm_drv` and
`MLCanopyFluxes` on them and read the results back (`step_state_from_instances`).
Decision:
- **`clm_ml/step.py::step(state, inputs, constants, params, *, options, c3) -> StepState`**:
  one patch's step, pure, built from the physics of processes 2-10 in the order of
  `clm_drv` (history LAI/SAI and soil water, soil albedo, evaporative resistance, thermal
  and hydraulic properties; `GetCLMVar`'s forcing, O2 and solar zenith angle; plant area,
  soil and plant resistance, leaf heat capacity, soil surface humidity; the sub-steps as a
  `lax.scan` with `jax.checkpoint`; averages, totals, merge; soil temperature). Inputs:
  `StepInputs` (the step's tower/gridcell meteorology, CLM history record, previous/next
  forcing records, calendar days; 0-based), `PatchConstants` (layer grid, the sub-step
  carry's start values besides the state, the padding slots the step keeps, PFT root
  values, soil column properties, soil colour albedos, location), `StepOptions` (static:
  the sub-step options, step length, layer PAI floor, totals option, history soil-water
  settings, `nlevsoi`, orbital parameters). It lives at the package top (like `state.py`):
  it composes processes, it is not one.
- **`clm_ml/legacy/step.py`** (transition): `patch_constants(template)` (0-based values
  from the containers after the first step; formed once per chunk, outside the scan, in
  `make_chunk_runner`, `make_stream_runner` and `make_batch_runner` (vmapped per C3/C4
  group)), `step_options(opts, step_seconds)`, `step_inputs(now, met, clm_record, bef,
  next)`. `offline_executable.main.state_step` wraps them (with the acclimation update) for
  the compiled chunks, the batched runner (`_patch_step` now takes the constants) and the
  step-by-step loop of `clm-ml-offline` (now one compiled step called per step).
  `instances_with_state` and `step_state_from_instances` left the time step; the first
  (initializing) step, the batched first step and the legacy view still use the
  containers (until 7.4); the view takes index 0 and other slots the state does not hold
  from the template, as before.
- **Each step's conservation errors start from zero.** The legacy step started from the
  template's `balance_error`, i.e. the first step's errors, so every step reported at least
  those (which had passed the checks). Not recorded in the reference; can only make a
  check report a step's own error.
- **`mlcan_to_clm = 1` refused** (`step_options`): it wrote CLM containers nothing reads
  offline (and failed on JAX arrays). `nlevsoi < nlevgrnd` is required: `SoilResistance`
  reads the ice of level j at j + 1 (C14, open issue 19), which must exist.
- **The legacy adapters of processes 3-9 and the process replay stay** until the first
  step leaves the containers (7.4): they test the same physics on the recorded
  pre-Phase-7 calls. `clm_driver.clm_drv` and `MLCanopyFluxes` now run only the first step
  (and `make_clm_ml_forward` of `test_differentiability`).
- **Round-off change, reference re-recorded (D7):** op by op (`jax.disable_jit`) the new
  step equals the old container step in every `State` and `Diagnostics` leaf over CHATS7
  steps 2-3, except `balance_error` (above) (`probe710c/eager.py`). The compiled chunk
  differs in 67 fields: 53 the twins move, within 1.13x the twin spread; 14 they do not, by
  1-2 ulps (at most 2.3e-14 relative, `apar_leaf`): the solar zenith angle in 7 daytime
  steps (sin/cos/acos) and the solar fields after it, and `psis_soil` (a sum over soil
  levels), with identical inputs. `optimization_barrier` on the solar zenith angle and on
  the soil hydraulic properties changed nothing (not kept): XLA fuses the transcendental
  and the sums differently now that the containers are gone (C10/C17 pattern). Noise twin
  passed before re-recording. Forward timing on CPU 15.6 ms/step (`p710c-check`; `p71`:
  24-25 ms/step); gradient timing at the package end.
Why: C19's 10c; the step is now a function of the step state, so 7.3 (forcing in the
chunk) and 7.4 (setup) only change where its inputs and constants come from.
Alternatives: splitting 10c into soil routines first and the canopy set-up second (each
would still need the container boundary; one probe covered the whole step); keeping the
template's error floor (no reason; not recorded); computing the constants inside each step
(slicing the template under the step's `vmap` and checkpoint every step).

### C22. Process 10d: the psihat tables as an input of the compiled step (7.2, 2026-10-31)

- **What:** the roughness-sublayer psihat tables (`RSLTables`) are an argument of
  `clm_ml.physics.canopy.substep` and `clm_ml.step.step` (next to `params`), of
  `state_step`'s step and of every compiled chunk: `make_chunk_runner`,
  `make_stream_runner`, `make_batch_runner` (not batched under `vmap`: `in_axes=None`),
  the sharded wrapper (`_sharded`, replicated) and `clm_ml.model.Model.run_chunk`
  (new optional `tables`, default the run's). The time step no longer calls
  `default_rsl_tables()`.
- **Where the tables come from:** `LegacyRun.rsl_tables`, set at setup to
  `default_rsl_tables()` (the package's `psihat.nc`, read once per process). The first
  (initializing) step, legacy `MLCanopyFluxes`, keeps `default_rsl_tables()` as a constant
  until the setup leaves the containers (7.4), as planned.
- **`tables` is a required keyword argument of the compiled chunks**, not a default:
  `jax.jit` bakes a Python default value that the caller does not pass into the program as
  a constant (checked), which is what this process removes. Callers (the run loops,
  `Model`, `test_stream_runner`, `test_ensemble`, `timing/time_vmap.py`,
  `timing/time_global.py`) pass `tables=run.rsl_tables`. Under `jax.vmap` keyword
  arguments are always mapped, so the ensemble test closes over the tables.
- **Bit for bit:** the CHATS7 bit-for-bit + noise twin test passes unchanged (no
  re-recording). The lowered chunk takes the two 276x41 tables as parameters and holds no
  276x41 constant (`probe710d/consts.py`).
Why: maintainer's instruction (process 10 fixes, C15): the tables are data, so another
table file needs no recompilation and the tables can be differentiated or swapped.
Alternatives: a Python wrapper filling in the default (hides the input; loses the jitted
function's methods used by tests, `_cache_size`); tables inside `CanopyParams` (they are
not per-PFT parameters; they would be gathered and batched per patch).

### C23. Package 7.2 review fixes (7.2, 2026-10-31)

Independent review of `bdde552..HEAD` (two reviewers, the physics modules against the
deleted originals and the step/driver/runners against the old driver; each saw the code
only). One bug, fixed; the rest recorded:
- **`leaf_boundary_layer: laminar | turbulent` stopped every run (fixed).** In these
  regimes the conductances do not depend on the leaf temperature, so
  `physics/leaf_boundary_layer.py` returned (layer, 1) instead of (layer, leaf); the old
  adapter broadcast them over sunlit and shaded leaves (`.at[p, 1:, leaves].set`), the
  sub-step carry of 10b (C20) does not, and the Runge-Kutta scan refused the shape (in the
  first step already). Now broadcast to the leaf temperature's shape inside the routine
  (Fortran calls it once per leaf). The unit test missed it because JAX clamps the
  out-of-range index `[:, 1]`; it now checks the shapes. CHATS7 with `gb_type` 1 and 2
  agrees with `bdde552` to 1.4e-11 relative over 30 steps (night to 15:00;
  `probe710d/dump.py`, `cmp.py`); the default regime is unchanged (bit for bit).
- **Recorded, not changed:** (a) the `euler` tableau has `b = [0]`: `runge_kutta_update`
  with it would reset the state; `canopy._stage` returns before calling it for Euler, so
  the model is right; (b) `run_streams` and `main()`'s chunk path call the runners without
  `template` (and `make_chunk_runner` without `params`), so jit compiles the template, and
  the patch constants formed from it, into the program as constants, contrary to the
  comment in `make_stream_runner` (since Phase 5b, not 7.2; `Model.run_chunk` passes
  them). Results are unaffected; it costs a recompilation per patch structure. **Maintainer
  (2026-10-31): agreed, fix (b) in 7.4** (setup without globals), where the template goes
  away; (a) left as it is.
- Checked by the reviewers and found equal to the old code (round-off): every physics
  module; the alternative photosynthesis options (C3/C4 x colimitation x Medlyn /
  Ball-Berry / WUE, 3.8e-12); Euler, RK2, RK3, Medlyn without acclimation/colimitation/
  water stress, `HF_extension_type 1` with a dense canopy and `kn_val 0.3` (against
  `bdde552` over 26 steps, 1.8e-11); the process order, the 13 conservation checks, the
  options reaching the physics, the `_sharded` argument order and `vmap` axes.

### C24. Package 7.3: the chunk's inputs as 0-based arrays (7.3, 2026-10-31)

- **Types** (`clm_ml/step.py`): the compiled step still takes one flat `StepInputs` per
  patch; a chunk holds its inputs split by what they belong to: **`MetInputs`** (calendar
  days, the step's meteorology, the previous and next steps' `ForcingRecord`s; per
  gridcell) and **`LandInputs`** (leaf and stem area, soil water, the adjustment factor;
  per patch), each stacked along a leading step axis. `step_inputs(met, land)` joins them
  per patch inside the chunk (after the gridcell-to-patch gather). Shortwave bands 0-based
  `(..., 2)`, no legacy patch axis, no `numrad + 1` band slot.
- **Producers** (`clm_ml/chunk_inputs.py`): `met_inputs(bef, cur, nxt, calday, calday_beg)`
  from derived-meteorology dicts (`TowerMet` names) and `forcing_inputs(forcing, ...)` from a
  tower `clm_ml.forcing.Forcing` chunk, both vectorized over the chunk with NumPy on the host
  (the previous record with the expressions of `physics.atm_forcing.clm_forcing_to_ml`,
  the step's current record, in NumPy: review, so the reader thread never computes on a GPU). Tower runs: `main.chunk_inputs` (one tower: gridcell and
  patch axes of length 1), `main.patch_inputs` (one patch: `StepInputs` stacked per step,
  for `make_chunk_runner`/`make_stream_runner`), `main.read_batch_inputs` (runs joined
  along the gridcell and patch axes, then gathered to the batch's slots; NumPy on the host,
  placed on the devices by `place_inputs` as before). Grid runs: `GridChunkReader` builds
  `MetInputs` with `met_inputs` (calendar days `(step,)` without a gridcell axis, broadcast
  in the chunk as before); `grid_forcing.land_record` (the chunk's `land_fn`) returns
  `LandInputs`.
- **No adjustment factor = 1.** The legacy record held `None` and the adapter wrote 1.0;
  `LandInputs.h2osoi_factor` is always an array (ones), the value the step used before.
- **Left on the legacy records until 7.4:** the first (initializing) step (`main.StepInputs`,
  `main.step_inputs`, `PatchBatch.inits`/`init_stack`, `grid_setup.first_step_inputs`), which
  runs on the containers. `clm_ml.legacy.step.step_inputs` and `main.stack_step_inputs`
  are deleted; `advance_acclimation` takes the step's air temperature.
- **Bit for bit:** the CHATS7 bit-for-bit + noise twin test passed unchanged (no
  re-recording); the grid input test compares the new arrays of a grid run and of the
  equivalent towers (calendar days excluded, as the step times were before).
Why: scope 7.3 as revised (D3, plan decision 36): the forcing stays prepared on the host;
only its layout changes, so the compiled step no longer converts legacy records.
Alternatives: nesting `MetInputs`/`LandInputs` inside `StepInputs` (every physics access
would change for no gain); keeping one `StepInputs` per patch in the chunk (sends the
meteorology once per patch instead of once per gridcell: the Phase 6 layout keeps it per
gridcell).
- **Review (independent, 7.3):** nothing serious; three minor points applied: the previous
  record in NumPy (above), the grid input test drops the calendar days by name, a no-op test
  helper removed.

### C25. Package 7.4 plan, and 7.4a: the legacy view without the template (7.4, 2026-10-31)

Probes before starting (`probe74/consts.py`, `which.py`, `initvals.py`, `fills.py`; CHATS7,
47 steps after the first):
- The time step reads the template (containers after the first step) only through
  `PatchConstants`. With the constants of the *initialized* containers (before the first
  step's physics: `spval` in every slot no setup routine writes) instead, the later steps
  change only in slots the outputs never show: the levels above the canopy of the longwave
  profiles and the leaf fluxes of padding and above-canopy layers (kept from the start
  values; the output variables mask them, the `.out` writer writes 0 there), and the
  state's padding slots. Every value in use is bit for bit the same.
- Of the sub-step carry's start values only `pref_forcing` and `wind_profile` are not
  `spval` after initialization (pressure of the first record, initial wind); no step reads
  them before writing them.
- The first step run by `clm_ml.step.step` from the initialized containers differs from the
  legacy container step by round-off (compiled vs eager), and in `balance_error`.
- The entries of the legacy view outside the state are all `spval` (7.4a).
Plan (in commits): (a) the view without the template; (b) the padding and start values the
step kept from `PatchConstants` become fill values inside the step (constants shrink to
setup values); (c) the setup as pure functions (`clm_ml/setup/`) with ported readers,
tested against the legacy initialized containers; (d) runs on the new setup in-process, the
first step an ordinary step of the first chunk; (e) grid runs without prototypes; (f) the
legacy setup, worker and containers deleted.
7.4a: `legacy_view(step, layers)` builds each legacy array by padding the 0-based value with
`spval` (no template argument; the legacy snow levels of `t_soisno_col` by soil structure).
Bit for bit.

### C26. 7.4b: the step's constants are setup values (7.4, 2026-10-31)
- `PatchConstants` holds only what the setup computes (layer grid, area fractions, heights,
  roots, soil column, soil albedos, location). The sub-step carry starts from
  `physics.canopy.initial_substep(n_layers, stage_slots, n_checks)`: `spval` in every slot
  (the legacy containers' fill; stage slots = the method's stages + 1, the legacy
  `nrk + 1`) and zero errors, with the prognostic state put in. Slots the step does not
  compute (layers above the top in use for the plant area, mean leaf water potential and
  the output leaf temperature/water potential; soil levels below bedrock for the hydraulic
  conductivity and matric potential) hold `spval` (`step._fill`).
- Before, these values came from the template (containers after the first step). C25's
  probes showed no value in use depends on them; the CHATS7 reference already held `spval`
  in every recorded slot concerned, so the change is bit for bit.
Why: the setup can now produce the constants directly (7.4c), and per-patch memory drops
(the carry's ~120 start arrays per patch are no longer inputs of the chunk).
Alternatives: zeros instead of `spval` (would change recorded fill slots; `spval` keeps the
legacy meaning "not set").
