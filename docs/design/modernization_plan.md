# CLM-ML-JAX Modernization Plan

**Status:** v4. Phases 0–2 done (branches `phase0/safety-net`, `phase1/config-api`,
`phase2/forcing-time`). v3 added the gridcell/patch structure (§3.1). v4 makes memory
independent of run length (forcing streams, chunked runs, output streams; §6.9), makes the
model time step configurable, and adopts a 10-day running-mean acclimation temperature.

**Companion files:**
- [`api_sketch.py`](api_sketch.py): the proposed Python API, as typed signatures with docstrings.
- [`examples/`](examples/): example `run.yaml`, `model.yaml`, `params_chats.yaml` and
  `towers.yaml` files, written as a user would write them. `run_global_example.yaml` shows the
  target form of a 150-year global run (hypothetical; it does not run yet).
- [`timing.md`](timing.md): timing-test results.
- [`numerics_and_regression.md`](numerics_and_regression.md): decisions on iterative solvers,
  differentiability, the model's round-off sensitivity, and the regression-testing policy.

---

## 0. Decisions recorded

| # | Question | Decision |
|---|---|---|
| 1 | Package name | Import as **`import clm_ml`**. The install name stays `clm-ml-jax` (the repo name). The command-line tool is `clm-ml`. |
| 2 | Option names | **Readable strings** (`stomatal_conductance: wue`). Fortran integer codes are accepted only by the namelist converter. |
| 3 | Spatial scope | **One tower up to global.** A global run is many gridcells with no lateral exchange between them. |
| 4 | Subgrid structure | Each gridcell holds one or more **patches** (one PFT each, with an area weight). **The patch is the unit of computation:** physics is written for one patch, and all patches in a run are batched with `vmap`. Patches are stored as a flat list, with an index giving each patch's gridcell. Gridcell output is the area-weighted sum of its patches. |
| 5 | Soil under patches | **Option A (default): each patch has its own soil column.** This is what the code does today (p == c), so results are unchanged. Option B (patches share one soil column, as in CLM's natural-vegetation land unit) is a later science option behind a `model.yaml` setting (§6.7). |
| 6 | Vertical layers | A **fixed `max_layers` (default 50) with masks** for every patch. Padded layers must not change results. |
| 7 | Gradients | With respect to **parameters, meteorological forcing, and state variables** (initial state). |
| 8 | Configuration | **YAML files**: `model.yaml` (physics and numerics), `params.yaml` (parameters by PFT), and `run.yaml` (the experiment, replacing the namelist). They are loaded into typed Python objects, which are what JAX runs on and differentiates. |
| 9 | Parameters | A **per-PFT table**. Each patch looks up its PFT's values, so gradients come out per PFT. |
| 10 | Config framework | **Plain YAML plus our own validation** (dataclasses with clear error messages). No Hydra. |
| 11 | Namelists | Kept as a **one-time converter**: `clm-ml convert nl.CHATS7.1day` writes the YAML files. |
| 12 | Run length and memory | **Memory must not depend on run length.** The design target is a global run with 150 years of 6-hourly forcing. Forcing is read from **streams** in **chunks** of time (for example a month), each with one extra record at either end. A run is a loop over chunks with the state carried between them. The next chunk is read in the background, and output is written per chunk (§6.9). |
| 13 | Model time step | **Configurable** (`numerics.step_seconds`, default 1800 s) and independent of how often forcing data are available. Every forcing stream is interpolated in time to the model step. When the step equals the forcing interval (CHATS7), results are unchanged. |
| 14 | Prescribed land data | LAI, SAI and soil water come from **user-supplied daily or monthly streams**. The CLM history file at the forcing interval, as used for CHATS7, remains supported. |
| 15 | Acclimation temperature | **10-day running mean** of air temperature, carried in the model state. This is a science change that alters CHATS7 results slightly, so the reference is regenerated when it is adopted (Phase 4b). The Fortran CLM-ml v2 definition (mean over the whole run, which needs all forcing before the run starts) stays available as an option, for validation against Fortran only. |
| 16 | Output | **Configurable and independent of the forcing and model step.** Several output streams, each with its own variables, frequency, statistic (mean, instantaneous, min, max), level (patch, gridcell) and file split. Statistics are accumulated during the run, so per-step values are never all held in memory. |
| 17 | Chunk length | **Calendar months**, aligned to month starts (a chunk is the steps whose start time falls in one month); a run shorter than a month is one chunk. Chunks therefore have up to four full-month lengths plus partial first/last months; each distinct length compiles once (§6.9). As in jax-gcm, output is accumulated inside the compiled chunk, and the chunk length only sets forcing memory, restart and file frequency, and health checks. |

| 18 | Phase 5 approach | **Masks in place**: the existing modules keep their files, 1-based arrays and golden tests while becoming mask-aware. Moving processes to `clm_ml/physics/`, 0-based indexing and Fortran-filter removal move to Phase 7 (decided after Phase 4b). |
| 19 | Different layer counts | **Masks**, so one compiled program serves every patch (simpler batching and sharding, gradients across mixed patches). Not grouping patches by layer structure. Cost: compute on padded layers (e.g. a 20-layer grass in 50 slots). |
| 20 | Configuration in the code | **Passed explicitly** (Phase 5a), not module globals: options static, tunable numbers in the differentiable parameters. Allows several models per process and in-process tests. |
| 21 | Process interface (§6.6) | **Deferred to Phase 7** (not needed for batching). |
| 22 | C3/C4 photosynthesis with mixed patches | **A static choice per group of patches**, not per-patch data. Phase 6 sorts patches into a C3 group and a C4 group *inside one compiled program* (group sizes are known when the run is set up), so each patch computes only its own pathway. Computing both pathways for every patch (tried in Phase 5b) wastes about half the photosynthesis work and memory on a GPU, where patches in a batch run in lockstep. Masks remain the choice for layer counts, which take many values; C3/C4 takes two. Extends to other per-PFT code switches as long as their combinations stay few. |
| 23 | Design target of Phase 6 | **Global runs at about 0.5° with several patches per gridcell** (roughly 6×10⁴ land gridcells and a few ×10⁵ patches), not collections of different towers. Tower runs stay supported as a special case (a domain of one or a few gridcells), but where the two pull apart the global run decides. Steps 2–5 are revisited after the step 1b `vmap` measurements (maintainer, Phase 6 on Derecho). |
| 24 | Time in a run with many gridcells | **One clock for the whole run**: all gridcells share the calendar, start, model step and chunking, as in a global run. What differs between gridcells (latitude, longitude, hence solar geometry and local time) is per-gridcell data. Towers with different periods are separate runs. (`StepTime` values can already be traced, so per-gridcell calendars stay possible later, but nothing is designed for them.) |
| 25 | A patch that fails a conservation check | **Recorded per patch** (patch, gridcell, check, step), not as one largest error per run. Default: stop and name the failing patches, as Fortran does. Option (`numerics.on_patch_error: exclude`, for global runs): the patch is excluded from then on (fill values in its output, left out of gridcell sums with the remaining weights renormalized, zero gradient contribution through a safe `where`), and the run stops only if more than a set fraction of patches fail. A NaN in one patch must never reach a gridcell sum or a shared PFT-parameter gradient. Provisional until step 3. |

Smaller decisions: see §8.

---

## 1. Goals and non-goals

### Goals

1. **A file-driven, Python-friendly interface.** A run is described by three YAML files and
   started with one command (`clm-ml run run.yaml`). The same files load in Python, where a
   scientist can run ensembles or take gradients in a few lines.
2. **Explicit state, no hidden globals.** Everything the model depends on goes in as an argument,
   and everything it changes comes back as a return value. This is what makes `jit`, `grad` and
   `vmap` work reliably.
3. **One patch up to millions with the same code.** Physics is written for one patch. A tower
   run, a tower with mixed vegetation, and a global run differ only in how many patches there are.
4. **One code path.** Today there is a "normal" mode (Python loops, file output, diagnostics) and a
   "differentiable" mode (`lax.scan`, diagnostics skipped). These become a single path that is
   both fast and differentiable.
5. **Swappable process schemes**, chosen by name in `model.yaml`.
6. **Keep scientific fidelity.** The refactored model must reproduce the current JAX output (to
   `rtol=1e-6`) and therefore the Fortran output, at every phase. Deliberate science changes
   (decision 15) are separate, documented steps with a regenerated reference.
7. **Memory independent of run length.** A 150-year run needs the same memory as a one-month run
   with the same domain.

### Non-goals (for this effort)

- Changing the science or adding new physics. (Option B soil sharing is planned as a separate,
  validated science change after Phase 6.)
- **Loading gridded/global input data.** The architecture supports any number of gridcells and
  patches, and `run.yaml` reserves `domain.type: grid`. Reading gridded datasets (CLM surface
  datasets for patch weights and PFTs, plus reanalysis forcing) is a follow-on project. The
  forcing-stream interface it plugs into is part of this plan (§6.9), so gridded readers only
  have to implement it.
- Coupling to CESM/CTSM. The `clm_src_cpl` code stays as a reference only.
- Rewriting the numerics of each routine, beyond what fixed layers and masks require (Phase 5b).

---

## 2. Where we are today (summary of the audit)

| Area | Current design | Problem it causes |
|---|---|---|
| Code organization | One Python file per Fortran module (`clm_src_main/`, `MLCanopyFluxesMod.py`, …), with ~1,760 "Fortran line" references | You have to know the Fortran CLM layout to navigate. The top-level package names (`clm_src_main`, `utils`, …) are generic and can collide with other installed packages. |
| Model state | ~15 `*_inst` objects held as module globals in `clm_instMod` and reassigned after every step (25 `global` statements in total) | Invisible to `jit`/`grad`/`vmap`. You cannot run two simulations side by side. |
| Time | `clm_time_manager.itim`, `start_date_ymd`, … are module globals that the driver mutates (including a documented "hack" that temporarily overwrites the start date) | Time cannot be traced, so the outer loop cannot be compiled. |
| Run options | `MLclm_varctl` module globals (`gs_type = 2`, …) plus a Fortran namelist read from stdin | Options are "magic integers" set by mutating module state. |
| Parameters | `MLpftcon`/`pftcon` global singletons. Differentiable parameters are threaded through as special-case arguments (`vcmaxpft_jax`, `g1_MED_jax`). CHATS values are switched on with `pftcon_val = 1`. | Only parameters someone has manually plumbed through can be differentiated. Tower-specific values are hard-coded. |
| Interfaces | `MLCanopyFluxes` takes 20 arguments, mostly typed `Any`. `mlcanopy_type` has hundreds of fields mixing inputs, forcing, prognostic state, scratch space and diagnostics. | It is hard to see what a routine reads or writes. |
| Forcing | Tower and CLM-history netCDF files are re-read inside the time loop. For 3-point interpolation, the state carries `*_bef/_cur/_next` copies of forcing. | File I/O inside the loop blocks compilation. Forcing is mixed into model state. |
| Time loop | The outer loop is a Python `for` with ASCII writes each step. The inner 300 s sub-step loop is a Python `for` in normal mode and a `lax.scan` in differentiable mode. | It is slow, and two code paths can silently diverge. |
| Subgrid hierarchy | CLM's gridcell → column → patch indices exist (`patch.gridcell`, `patch.column`), but the canopy loops over patches in Python with Fortran-style filters, and the standalone soil temperature assumes one patch per column (`MLSoilTemperatureMod`: "p == c"). Arrays are 1-based. | Patches can't be batched, and the patch/column relationship is implicit. |
| Layers | The number of canopy layers (`ncan`, `ntop`, `nbot`) depends on canopy height, and is used as a Python loop bound. | Patches with different canopy heights have different array shapes, so they **cannot be batched together**. This is the main blocker for multi-patch and global runs. |
| Output | Six fixed-format ASCII files (`flux.out`, `fsun.out`, …) | Not self-describing (no names, units or time coordinate). |
| Run length (found after Phase 2) | Phase 2 reads all forcing for the whole run before it starts. The tower reader always starts at the file's first record. CLM history data are read for every model step. The acclimation temperature is the mean air temperature over the whole run. The model step equals the forcing interval. Outputs for every step are kept in memory. | Fine for a tower month (~10 MB of forcing), impossible for a 150-year global run (~140 GB of 6-hourly meteorology at 1°, terabytes of per-step soil water). Addressed by decisions 12–16 and §6.9. |
| Validation | Strong: 160+ routine-level golden tests at `rtol=1e-9`, plus end-to-end Fortran comparison (2.5% / r ≥ 0.9999) and differentiability tests. Phase 0 added an end-to-end regression test at `rtol=1e-6`. | — |

---

## 3. Design principles

### 3.1 Vocabulary: gridcells, patches, soil columns, towers

| Term | Meaning | Holds |
|---|---|---|
| **gridcell** | A location with its own atmospheric forcing | Latitude/longitude, meteorological forcing |
| **patch** | One PFT's canopy on a gridcell, covering a fraction (`weight`) of it. **The unit of computation.** | PFT, weight, canopy height, leaf/stem area profiles, canopy state |
| **soil column** | The soil under one or more patches | Soil texture, color, bedrock depth, soil temperature and moisture |
| **tower** | A named flux-tower location in the built-in registry (`data/towers.yaml`). It defines one gridcell and its patches, plus observation metadata such as the measurement height. | — |

- A **tower run** today is 1 gridcell, 1 patch and 1 soil column. A **tower with mixed
  vegetation** is 1 gridcell with several patches. A **global run** is many gridcells, each with
  several patches.
- With **option A** (default), soil column = patch, one-to-one. With **option B**, a soil column
  serves several patches.
- **Patches are a flat list.** Each patch carries `gridcell_index` and `column_index`. This
  handles gridcells with different numbers of patches without padding a `(gridcell, patch)` grid.
  CLM has up to 15 natural PFTs plus crops per gridcell, most of them with tiny weights.

### 3.2 Principles

1. **Pure functions.** Every physics function has the shape `f(inputs) -> outputs`. It reads no
   globals and mutates nothing.
2. **Files describe, objects run.** Each YAML file loads into a typed Python object. Validation
   happens at load time, with messages that name the file and key. JAX only ever sees the objects.
3. **Separate the kinds of data.** Each kind gets its own container, because JAX treats each one
   differently:

   | Container | Comes from | What it holds | Shape | JAX treatment |
   |---|---|---|---|---|
   | `ModelConfig` | `model.yaml` | Scheme choices, time steps, layer settings, `max_layers`, soil coupling | — | **Static.** Changing it recompiles. |
   | `Domain` | `run.yaml` → `towers.yaml` (built-in or user's own) | `Gridcells`, `Patches` (with gridcell/column indices) and `SoilColumns` | `(n_gridcell,)`, `(n_patch,)`, `(n_column,)` | Arrays. Counts and indices are static. |
   | `Layers` | Derived from `Patches` + `ModelConfig` | Per-patch layer heights and **masks** inside a fixed `max_layers` | `(n_patch, max_layers)` | `max_layers` is static. Heights and masks are arrays. |
   | `Params` | `params.yaml` | **Per-PFT table** of physical parameters | `(n_pft,)` | **Traced and differentiable** |
   | `ForcingSource` | `run.yaml` → `forcing:` streams | Readers for each forcing stream (meteorology, vegetation, soil water), each with its own time axis | — | Not traced. Produces `Forcing` chunks. |
   | `Forcing` | A `ForcingSource`, one chunk at a time | Meteorology per gridcell, plus prescribed vegetation per patch and soil water per column, **interpolated to the model step** for one chunk of time (plus one record either side) | `(time, gridcell, …)`, `(time, patch, …)`, `(time, column, …)` | **Traced and differentiable** |
   | `State` | `run.yaml` → initial conditions or a restart file | Prognostic variables only: canopy per patch (including the running-mean acclimation temperature), soil per column | `(patch, …)`, `(column, …)` | **Traced and differentiable**. It is the `lax.scan` carry, and is carried from chunk to chunk. |
   | `Outputs` | Produced by the run | Fluxes, profiles and diagnostics per patch, accumulated into each output stream's statistic. Gridcell means come from area weighting. | `(output time, patch, …)` → `(output time, gridcell, …)` | Accumulated inside `scan`, written per chunk |

4. **Write physics for one patch, batch with `vmap`.** Each process is written as if there is one
   patch, with arrays of shape `(max_layers,)`. The model applies `jax.vmap` over the patch axis.
   jax-gcm does the same with `vectorize_columns`. Splitting patches across GPUs is then a
   sharding annotation on that axis, not a code change.
5. **Gather in, aggregate out.** Each patch gets its forcing by indexing gridcell arrays
   (`forcing[gridcell_index]`) and its parameters by indexing the PFT table (`params[pft]`).
   Patch results are combined into gridcell values with an area-weighted `segment_sum`. All of
   these are ordinary differentiable JAX operations.
6. **Fixed layers plus masks.** Every patch gets `max_layers` canopy layers. A per-patch mask says
   which layers are real air layers, and another says which contain leaves. Physics uses masks,
   never patch-specific loop bounds. Padded layers must have no effect: they are zero-weighted in
   sums and decoupled (identity rows) in the tridiagonal solves.
7. **A chunk is one compiled function; a run is a loop over chunks.**
   `run_chunk = scan over time of (vmap over patches of step)`, where
   `step(state, forcing_window, params) -> (state, outputs)` acts on one patch. The Python
   loop over chunks carries the state, reads the next chunk in the background, and writes
   output. Memory depends on the chunk length, not the run length.
8. **Adapters first, rewrites later.** Put the new interface around the existing translated code
   first. Then move internals across one process at a time, each step verified by tests.
9. **Fortran traceability moves to docstrings.** Each refactored function keeps a
   `Fortran reference:` line in its docstring.
10. **Few dependencies.** Containers are standard `dataclasses` registered as JAX pytrees. New
    runtime dependencies: `pyyaml`, `xarray`, `netCDF4` (added in Phase 0).

---

## 4. Configuration files

Full examples are in [`examples/`](examples/). Every key has a default, except where marked
required, so most files stay short.

### 4.1 `run.yaml`: the experiment (replaces the namelist)

```yaml
model: model.yaml                 # path, relative to this file
params: params_chats.yaml
time:
  start: 2007-05-01T00:00         # required
  end:   2007-05-02T00:00         # or: days: / steps:
domain:
  type: towers                    # towers | grid (reserved for later)
  towers: [CHATS7]                # names from the built-in registry or towers_file
  towers_file: null               # optional: your own towers.yaml
  min_patch_weight: 0.0           # drop patches covering less than this fraction of a gridcell
execution:
  chunk: 1 month                  # forcing read and output written per chunk; sets memory use
  restart_every: 1 year           # write a restart file at these boundaries (null: end only)
forcing:                          # streams; each has its own time axis, interpolated to the model step
  meteorology:
    files: input_files/tower-forcing/{tower}/{year}-{month:02d}.nc   # may be split in time
    interpolation: three_point    # within a step's sub-steps: none | three_point (CLM-ml v2)
    shortwave: coszen             # coarser data -> model step: coszen | linear | constant
    precipitation: constant       # constant (conserves totals) | linear
  clm_history:                    # initial conditions; LAI/SAI/soil water with source clm_history
    file: input_files/clm5_0/{tower}/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc
    start: 2007-04-01T00:00
  vegetation:                     # LAI, SAI per patch
    source: clm_history           # clm_history | file
    files: null                   # with source: file, e.g. data/lai_{year}.nc
    interpolation: linear         # for daily/monthly data
  soil_water:
    source: clm_history
    interpolation: linear
    adjustment:
      file: input_files/clm5_0/{tower}/CHATS_soil_moisture_correction_29Jul25.nc
      levels: 3                   # was nlev_soil_adjust
initial_conditions:
  source: clm_history             # clm_history | restart
  restart_file: null
output:                           # independent of forcing and model step
  directory: output/
  streams:
    - name: halfhourly
      frequency: step             # step | hourly | daily | monthly | yearly
      statistic: mean             # mean | instantaneous | min | max
      variables: all              # or a list of names
      level: [patch, gridcell]
      file_per: run               # run | year | month
    - name: monthly
      frequency: monthly
      statistic: mean
      variables: [latent_heat_flux, sensible_heat_flux, gpp]
      level: [gridcell]
      file_per: year
```

Each stream's time axis comes from its files. Data at the model step (such as CHATS7's
half-hourly tower and CLM history files) are used directly. Coarser data (6-hourly
meteorology, daily soil water, monthly LAI) are interpolated as configured; see §6.9.

### 4.2 `model.yaml`: physics and numerics

```yaml
physics:
  stomatal_conductance: wue               # medlyn | ball_berry | wue
  water_stress_on_conductance: true
  wue_solver: bisection                   # brent | bisection
  photosynthesis_colimitation: true
  temperature_acclimation: true
  acclimation_temperature: running_mean   # running_mean | run_mean (Fortran CLM-ml v2; validation only)
  acclimation_days: 10                    # averaging period of the running mean
  nitrogen_decay_kn: null                 # null = computed
  roughness_sublayer: harman_finnigan
  sparse_canopy: true
  ground_conductance: log_profile         # extend_hf | log_profile
  flux_profile_solution: implicit         # well_mixed | implicit
  leaf_boundary_layer: free_convection    # laminar | turbulent | free_convection
  solar_radiation: two_stream             # norman | two_stream
  leaf_optics_vary_with_height: false
  longwave_radiation: norman
numerics:
  step_seconds: 1800                      # model time step; forcing is interpolated to it
  substep_seconds: 300                    # multilayer canopy substep; must divide step_seconds
  integrator: rk4                         # euler | rk2_* | rk3_* | rk4
  checkpoint_steps: true                  # less memory when taking gradients
canopy_layers:
  max_layers: 50                          # fixed array size for every patch
  dz_tall: 0.5
  dz_short: 0.1
  tall_canopy_threshold: 2.0
  n_layers_above: null
  n_layers_within: null
  min_plant_area_per_layer: 0.01          # dpai_min
soil:
  layer_structure: clm5_0                 # clm5_0 | clm4_5 (was clm_phys)
  columns: per_patch                      # per_patch (option A) | shared (option B, later)
```

### 4.3 `params.yaml`: parameters by PFT

The package ships `clm5_defaults.yaml` with every parameter for every PFT, in this format:

```yaml
units:
  vcmax25: umol m-2 s-1
  ...
pft:
  broadleaf_deciduous_temperate_tree:
    vcmax25: 57.7
    g1_medlyn: 4.45
    ...
```

An experiment's file **extends** the defaults and lists only what changes. This replaces
`pftcon_val = 1`:

```yaml
extends: clm5_defaults
pft:
  broadleaf_deciduous_temperate_tree:   # CHATS walnut orchard
    vcmax25: 125.0              # Rosati et al. (2006)
    plant_conductance: 7.0      # Tyree et al. (1993)
    iota: 375.0                 # Rosati et al. (2006)
    root_resistivity: 14.0      # Tyree et al. (1994)
    psi50_gs: -1.60             # SPA-walnut
    leaf_angle_index: 0.53      # walnut
    leaf_reflectance: [0.06, 0.42]
    leaf_transmittance: [0.04, 0.43]
    stem_reflectance: [0.21, 0.49]
```

Validation rejects unknown PFT names, unknown parameter names, and missing values for any PFT
used by a patch in the run. Ranges from the units table are checked as warnings.

**Parameters are per PFT, not per patch or gridcell.** Two patches with the same PFT share
parameter values. That is what makes per-PFT calibration across many towers possible.

### 4.4 `towers.yaml`: tower descriptions

The built-in tower table (now in `TowerDataMod.py`) moves to `clm_ml/data/towers.yaml`. Users can
supply their own file in the same format. Each tower describes its gridcell, soil and patches.

```yaml
CHATS7:
  latitude: 38.49
  longitude: -121.84
  measurement_height: 23.0        # forcing (reference) height, m
  forcing_step_minutes: 30
  soil: {texture_class: silty clay loam, sand_percent: 10.0, clay_percent: 35.0,
         organic_matter: 50.0, color_class: 15, bedrock_depth: 2.0}
  patches:
    - pft: broadleaf_deciduous_temperate_tree
      weight: 1.0
      canopy_height: 10.0
      root_biomass: 500.0
      leaf_area_profile_beta: [2.6, 1.3]
      stem_area_profile_beta: [1.8, 1.3]
```

- Patch weights on a gridcell must sum to 1, which validation checks.
- With option A, every patch gets its own soil column with the tower's soil properties.
- `null` means "use the default", replacing the Fortran `-999` sentinel.

See [`examples/towers_example.yaml`](examples/towers_example.yaml), which includes a
hypothetical two-patch tower.

### 4.5 Commands

```bash
clm-ml run run.yaml
clm-ml run run.yaml --set model.physics.stomatal_conductance=medlyn --set time.end=2007-05-08
clm-ml check run.yaml                    # validate all files, report patches and layer counts, no run
clm-ml convert nl.CHATS7.1day -o chats7/ # namelist -> run.yaml, model.yaml, params.yaml
clm-ml-offline < nl.CHATS7.1day          # legacy entry point, kept during transition
```

---

## 5. Python API (summary)

Full signatures are in [`api_sketch.py`](api_sketch.py).

```python
import clm_ml

run = clm_ml.load_run("run.yaml")      # reads and validates run + model + params + towers
run.execute()                          # same as `clm-ml run run.yaml`: loops over chunks, writes output

# Or drive the chunk loop yourself (this is what execute() does):
state = run.initial_state
for chunk in run.forcing_source.chunks(run.start, run.end, run.chunk):   # next chunk read in background
    state, out = run.model.run(state, chunk, run.params)                 # one compiled scan
    run.writer.write(out)                                                # output streams, per chunk
```

For a short run, a single chunk can be inspected directly:

```python
chunk = run.forcing_source.chunk(run.start, days=31)
state, out = run.model.run(run.initial_state, chunk, run.params)
ds = out.to_xarray()                   # dims (time, patch, layer, ...), with units and gridcell index
```

Gradients with respect to parameters, forcing and initial state, over a time window (for long
runs, windows start from saved restart states; see §6.9):

```python
model, params, state0 = run.model, run.params, run.initial_state
forcing = run.forcing_source.chunk(run.start, days=31)

def loss(params, forcing, state0):
    _, out = model.run(state0, forcing, params)
    return jnp.mean((out.gridcell_mean().latent_heat_flux - obs_lh) ** 2)

g_params, g_forcing, g_state = jax.grad(loss, argnums=(0, 1, 2))(params, forcing, state0)
g_params.vcmax25          # shape (n_pft,): gradient per plant functional type
g_forcing.air_temperature # shape (time, gridcell)
```

Ensembles:

```python
batch = clm_ml.perturb(params, pft="broadleaf_deciduous_temperate_tree",
                       vcmax25=jnp.linspace(60, 140, 32))
lh = jax.vmap(lambda p: model.run(state0, forcing, p)[1].latent_heat_flux)(batch)  # (32, time, patch)
```

Changing settings in Python without editing files:

```python
params = params.set("broadleaf_deciduous_temperate_tree", vcmax25=110.0)
model = clm_ml.Model(model.config.replace(**{"physics.integrator": "rk2_midpoint"}), model.domain)
```

---

## 6. Target architecture

### 6.1 Package layout

```
src/clm_ml/
  __init__.py            # public API (load_run, Model, ModelConfig, Params, ...)
  run.py                 # Run: load_run(), execute()
  model.py               # Model: initial_state, step, run (scan over time, vmap over patches)
  config.py              # ModelConfig + YAML loading/validation
  domain.py              # Domain, Gridcells, Patches, SoilColumns; tower registry loader
  layers.py              # Layers: fixed-size vertical grid + masks (from MLinitVerticalMod)
  params.py              # Params (per-PFT table), YAML loading, extends/overrides, perturb()
  forcing.py             # Forcing, loaders, time interpolation
  state.py               # CanopyState, SoilState, State
  outputs.py             # Outputs, gridcell_mean, to_xarray, netCDF / legacy ASCII writers
  constants.py
  physics/               # one-patch, mask-aware processes
    solar_radiation.py  longwave.py  turbulence.py  leaf_boundary_layer.py
    photosynthesis.py   plant_hydraulics.py  canopy_water.py  soil.py
    integrators.py      math.py
  io/
    yaml_schema.py       # shared validation helpers, error formatting
    netcdf.py
    legacy.py            # namelist converter, Fortran option-code table, .out writers
  cli.py                 # `clm-ml run|check|convert`
  data/
    towers.yaml          # built-in towers (moved from TowerDataMod.py)
    params/clm5_defaults.yaml
    rsl_lookup_tables/psihat.nc
```

The existing packages (`clm_src_*`, `multilayer_canopy`, `offline_driver`, …) stay in place during
the transition, and the new API calls into them through adapters. Each process moves into
`clm_ml/physics/` as it is refactored (Phase 7, decision 18), and old modules are deleted only
when nothing imports them.

### 6.2 One time step (for one patch)

The science order is the same as today. Each box is a pure function:

```
step(state, forcing_window, params_for_this_patch, layers_for_this_patch)
 ├─ interpolate prescribed CLM veg/soil data at this step     (was clmData)
 ├─ soil_properties(...)                                      (SoilAlbedo, calc_soilevap_resis,
 │                                                             SoilThermProp, SoilWater)
 ├─ canopy_setup(...)                                         (PAD profiles, SoilResistance,
 │                                                             PlantResistance, LeafHeatCapacity)
 ├─ scan over sub-steps (e.g. 6 x 300 s):                     (one path; was two)
 │    ├─ atm_forcing = interpolate(forcing_window, t_sub)     (GetAtmForcing)
 │    ├─ solar_radiation[config.solar_radiation]
 │    ├─ nitrogen_profile
 │    └─ integrator[config.integrator]: for each RK stage
 │         wetted_fraction → longwave → turbulence → leaf_boundary_layer
 │         → photosynthesis(sun, shade) → flux_profile_solution
 │         → leaf_water_potential → interception/evaporation
 ├─ average sub-step fluxes
 ├─ canopy diagnostics                                        (always computed now)
 └─ soil_temperature(...)                                     (option A: this patch's own column)
returns (new_state, outputs_t)
```

`Model.run` (one chunk) wraps this as follows (option A):

```
scan over the chunk's time steps:                # carry = (State, output accumulators)
    forcing_p = forcing_window[gridcell_index]   # gather gridcell forcing to patches
    params_p  = params.for_patches(pft)          # gather PFT parameters to patches
    vmap over patches:
        step(state[p], forcing_p[p], params_p[p], layers[p])
    accumulate outputs into each output stream's statistic (mean, min, max, ...)
gridcell output = segment_sum(weight * patch output, gridcell_index)
returns (final State, completed output records)
```

The run loop around it (§6.9) carries the `State` from chunk to chunk.

`forcing_window` holds the previous, current and next records. That removes the
`*_bef/_cur/_next` forcing copies from model state.

### 6.3 Static vs traced: what can change without recompiling

These **recompile** (they are static): anything in `model.yaml` (including `max_layers` and
soil coupling), the number of gridcells, patches and soil columns, the index arrays that connect
them, and the number of time steps.

These do **not** recompile, and all are differentiable: parameter values, forcing values, and
initial state. Patch properties (canopy height, profile shapes), patch weights and soil properties
are also arrays and can change without recompiling, provided every patch still fits within
`max_layers`. PFT indices and layer masks are integers and booleans, so they are not
differentiated.

This replaces today's `DIFFERENTIABLE_MODE` flag and `GridInfo` workaround with a single rule.

### 6.4 Fixed layers and masks

- `Layers` is computed per patch at load time, with the same algorithm as `MLinitVerticalMod`.
  It is then **padded to `max_layers`**. Patches on the same gridcell share the measurement
  height but can have different canopy heights, so their layer counts can differ.
- Fields (each per patch): `n_layers` (int), `top_leaf_layer`/`bottom_leaf_layer` (int),
  `heights`/`thickness` `(max_layers,)`, `is_air_layer` `(max_layers,)` bool, and
  `has_leaf` `(max_layers,)` bool.
- **Padding rule.** Padded layers sit above the domain top. They have zero thickness and zero
  plant area, are excluded from every sum, and are identity rows in the tridiagonal systems. So
  a padded patch gives the same answer as the unpadded run. The regression test checks this.
- If any patch needs more than `max_layers`, `clm-ml check`/`load_run` fails with a message giving
  the tower and patch, the layers needed, and the configured maximum.
- **Cost.** Short canopies compute on unused layers. If this matters for global runs, a later
  optimization can group patches into buckets with similar layer counts, each compiled
  separately.

### 6.5 Parameters by PFT

- `Params` fields have shape `(n_pft,)`, or `(n_pft, 2)` for spectral optics. PFT order is fixed
  by the defaults file (CLM ordering, `not_vegetated` = 0).
- `params.for_patches(patches.pft)` returns per-patch values with shape `(n_patch,)`. The physics
  only ever sees one patch's values.
- Includes both today's `MLpftcon` fields and the CLM `pftcon` fields the canopy model uses (leaf
  dimension, leaf angle, leaf/stem optics, root profile, SLA, C3/C4 flag). The C3/C4 flag is a
  model structure choice, so it is stored as an integer and is not differentiated.
- **Future extension** (not in this plan): allow a parameter to vary per patch or per gridcell
  through a netCDF override file. The `for_patches` gather is the single place that would change.

### 6.6 Process interface

*Implemented in Phase 7a (decision 21); Phase 5 keeps the legacy container.*

Each process is a plain function on one patch:

```python
def process(step: StepState, inputs: StepInputs, params: PatchParams, layers: PatchLayers,
            config: ModelConfig) -> StepState
```

- Variants are registered by name, for example
  `SOLAR_RADIATION = {"norman": norman, "two_stream": two_stream}`.
  `model.yaml` selects one.
- Each process's docstring declares which fields it **reads** and **writes**. A test checks the
  declared write set against what actually changes.

**Why not full jax-gcm-style composable terms?** This model has one fixed process order with a
few variant choices per process, so a name-to-function registry gives most of the benefit with
much less machinery. It can be upgraded later without changing the YAML or the public API.

### 6.7 Soil under patches: options A and B

| | Option A: `soil.columns: per_patch` (default) | Option B: `soil.columns: shared` (later) |
|---|---|---|
| Soil columns | One per patch (`n_column = n_patch`) | One per gridcell (or per land unit), shared by its patches |
| Coupling | None. The whole step is `vmap` over patches. | Canopy: `vmap` over patches. Soil temperature: `vmap` over columns, driven by the area-weighted ground heat flux of the column's patches (`segment_sum`). |
| Science | Identical to today (standalone CLM-ml: p == c) | **New.** Ground temperature is currently part of each patch's canopy solution. Sharing a column needs a rule for reconciling per-patch ground temperatures with one column temperature. It needs its own validation. |
| When | Phases 1–6 | Separate science change after Phase 6 |

Soil moisture is prescribed today, so sharing soil water between patches does not arise yet. It
will matter if the model ever predicts soil moisture (root water uptake by several patches from
one column).

### 6.8 Errors and checks

- Load-time validation of all YAML. Errors name the file, key, bad value and allowed values.
- `clm-ml check` validates everything and prints a summary (gridcells, patches, PFTs, layer
  counts, run length, estimated memory) without running.
- `endrun` calls inside traced code are replaced by an opt-in `check: true` mode using
  `jax.experimental.checkify` (for NaNs and balance errors).
  **Done for Fortran's conservation checks (Phase 5a):** leaf and soil energy balance,
  longwave conservation and the canopy nitrogen integral record their largest error in
  `mlcanopy.balance_error` (`MLCanopyFluxesType.BALANCE_CHECKS`, with Fortran's tolerances)
  instead of a host callback inside compiled code. The error is checked on the host
  (`check_balance`): after each step that runs on concrete values (the first step,
  step-by-step runs) and after each compiled chunk (the chunk's largest error), where the run
  stops naming the check, the error and the dates. Difference from Fortran: a compiled run
  stops at the end of the chunk, not at the failing step. Callbacks made the compiled chunk
  impossible to store in JAX's compilation cache (and paused the device every step).
- Energy- and water-balance errors are included in `Outputs` by default, per patch and per
  gridcell.

### 6.9 Forcing streams, chunked runs and output streams

**Why.** The design target is a global run with 150 years of 6-hourly forcing: 219,000
records. At about 1° there are roughly 20,000 land gridcells:

| Data | Whole run | One month |
|---|---|---|
| Meteorology, about 8 variables, float32 | ~140 GB | ~80 MB |
| Soil water, daily, 20 layers × ~3 patches per gridcell, float32 | ~260 GB | ~150 MB |
| LAI/SAI, monthly | small | small |

So forcing is read in chunks, and memory is set by the chunk length.

**Forcing streams.** Each kind of forcing is a stream with its own files and time axis:
meteorology (per gridcell), vegetation (LAI/SAI per patch) and soil water (per column). A
`ForcingSource` combines the streams and has one job: `chunk(start, n_steps) -> Forcing`, with
the data interpolated to the model time step. It also returns one record before the chunk's
first step and one after its last, which the three-point interpolation needs at chunk
boundaries. Readers open only the files covering the requested time (files may be split by
month or year) and only the gridcells assigned to this process (see sharding below).

**Time interpolation to the model step** (decision 13). The defaults follow CLM's data
atmosphere and reproduce CLM-ml v2 when data are at the model step:

| Variable | Default rule |
|---|---|
| Temperature, humidity, wind, pressure, longwave | Linear between records; `three_point` (CLM-ml v2's centered rule) when data are at the model step |
| Shortwave | Scaled by the cosine of the solar zenith angle within each interval (`coszen`), so the diurnal cycle survives 6-hourly data and the interval mean is conserved |
| Precipitation | Held constant over the interval, which conserves totals |
| LAI, SAI (monthly or daily) | Linear between record midpoints |
| Soil water (daily or monthly) | Linear between record midpoints |
| CO₂ | Constant (as now), or a stream if supplied |

Each rule is a named option in `run.yaml`, so these are choices, not hard-coded behavior.

**The run loop.**

```
state = initial state (or restart file)
prefetch chunk 1 on a background thread
for each chunk:
    forcing = wait for the prefetched chunk; start reading the next one
    state, outputs = model.run(state, forcing, params)     # one compiled scan
    write completed output records (per stream, per file split)
    at restart boundaries: write state to a restart file
```

- **Two separate time scales** (as in jax-gcm, `jcm/model.py` and `jcm/runners.py`):
  - **Output interval, inside the compiled scan.** A chunk is a *nested* scan: an outer loop
    over output records, and inside it an inner loop over the model steps between records.
    The inner loop accumulates each output stream's statistic (running sum for a mean,
    min/max, or the instantaneous value) in the scan's carry and keeps no per-step values,
    so memory scales with the number of output records in the chunk, not the number of
    steps. `jax.checkpoint` wraps the inner step (as jax-gcm does) so gradients over a chunk
    store only the state at each step, not every intermediate value. With several output
    streams of different frequencies, the inner loop runs at the model step and each stream
    emits a record when its own interval completes.
  - **Chunk length, outside it.** The chunk is host bookkeeping: how much forcing is in
    memory, how often restart files and output files are written, and how often the state's
    health is checked. jax-gcm runs the whole integration as one compiled call by default
    (`chunk_days: 0`) and uses 5-day chunks (`run_chunked`) for long runs.
- **Chunk length** is a run setting (`execution.chunk`). Default: the whole run when it is
  short (a tower month), otherwise a few days to a month, limited by forcing memory (table
  above). Chunk boundaries should fall on output-file and forcing-file boundaries.
  **Decided (decision 17): calendar months**, aligned to month starts; a chunk is the steps
  whose start time falls in one month. Full months have 28–31 days and partial first/last
  months other lengths, so at most six lengths each compile once.
- **When compilation happens.** A compiled chunk program is reused for every chunk with the
  same input shapes and model options; new data do not trigger compilation. It recompiles
  only for a different chunk length, different array sizes (number of patches or layers), a
  different `model.yaml` option, or a new process (and JAX's on-disk compilation cache can
  skip even that). A short last chunk is padded to the full length, so it reuses the
  program; the extra steps are discarded. CHATS7 example: the first one-day chunk takes about
  35 s (about 24 s of it compiling), each later one about 0.7 s of physics.
- **Reading overlaps computing.** The next chunk is read while the current one runs. jax-gcm
  does the same (its design note `chunked_run_io_overlap.md`).
- **Storage precision.** Forcing is stored and transferred as float32 and computed in float64.
- **Between chunks (host side, as jax-gcm's `run_chunked`):**
  - **Health check** of the state (NaN, out-of-range temperatures); the run stops at the first
    failing chunk and says which variable and where, instead of writing garbage for decades.
  - **Restarts** at chunk boundaries (`execution.restart_every`). Written beside the old one
    and renamed over it, keeping the previous one (`.prev`) as a fallback, so a run killed
    mid-write can still resume. A resumed run continues from the last completed chunk; it
    refuses to overwrite output of a run it cannot resume.
  - **Partial output intervals** (e.g. a monthly mean in progress) are saved with the
    restart, so a resumed run produces the same monthly means.
- **Acclimation temperature.** The 10-day running mean (decision 15) is part of the `State`,
  updated every step, so it needs no look-ahead over the forcing. The legacy whole-run mean
  needs a pre-pass over all forcing, so it is only allowed for short runs (validation against
  Fortran).
- **Sharding for global runs.** Patches are split across devices (Phase 6). Each process reads
  only the gridcells its patches belong to, so no single process holds global forcing.

**Output streams** (decision 16). Each stream accumulates its statistic during the scan
(the inner loop above): running sums for means, running min/max, or the instantaneous value at
the output time. Only
completed output records leave the device. They are written per chunk to files split as
configured. Output frequency is independent of the forcing interval, model step and chunk
length. An output interval that spans chunks carries its partial accumulators to the next
chunk. Per-step output (`frequency: step`) is allowed for short runs and validation.

**Gradients on long runs.** Back-propagating through 219,000 steps is not feasible, even with
`jax.checkpoint`. Calibration therefore differentiates over windows (one chunk, or a few),
starting from restart states saved by a forward run. The model's own API (`Model.run` on a
chunk) supports this directly.

---

## 7. Migration plan

Each phase ends with all tests passing and a mergeable PR. **The model must give the same answer
at the end of every phase.**

### Phase 0: Safety net ✅

- End-to-end regression test against the current JAX output at `rtol=1e-6`: CHATS7, 50 steps,
  86 fields, reference committed as `.npz`. It has its own CI job.
- Packaging fixed (packages are found under `src/`). Python ≥ 3.10. `netCDF4`, `pyyaml` and
  `xarray` are runtime dependencies.
- Baseline timing test (`timing/time_offline.py`) recorded in `docs/design/timing.md`: 214 s per
  simulated day on an Apple Silicon CPU. **GPU timing is deferred until after Phase 4.** Until
  then, the code dispatches many small operations from Python, which a GPU handles poorly.

### Phase 1: Configuration files and the new API as a wrapper (no physics changes) ✅

**As built** (`src/clm_ml/`): `config.py` (model.yaml), `params.py` (per-PFT table with
`extends:`), `domain.py` (towers.yaml → gridcells/patches/soil columns), `run.py`
(run.yaml, `load_run`, `Run.execute`), `outputs.py` (64 named variables with units →
netCDF at patch and gridcell level), `cli.py` (`clm-ml run|check|convert`), and
`legacy/` (the adapter: `worker.py` runs the unchanged legacy driver in a fresh process after
installing all options, PFT tables and the tower; `tables.py` generates the shipped YAML from
the legacy tables; `namelist.py` converts namelists). Acceptance: the converted CHATS7
50-step case run through `clm-ml` is bit-for-bit identical to `clm-ml-offline` in all 86
recorded fields and in all six `.out` files (`tests/regression/test_clm_ml_equivalence.py`).

Planned scope:


- Create `src/clm_ml/` with `ModelConfig`, `Domain`, `Params`, `Run`, the YAML loaders and
  validation, and the CLI (`run`, `check`, `convert`).
- Write `data/params/clm5_defaults.yaml` from `MLpftconMod`/`pftconMod`, and
  `data/towers.yaml` from `TowerDataMod`. Add a test that the YAML values equal the current Python
  tables exactly.
- `Run.execute()` translates the new objects into the existing globals, calls the existing driver,
  and collects outputs into arrays. It writes netCDF (with names and units) and/or the legacy
  `.out` files byte-for-byte.
- In this phase, **one tower with one patch per run**. Multi-patch and multi-tower `run.yaml`
  files are validated but rejected with a clear "not yet supported" error.
- `clm-ml convert` turns every namelist in `offline_executable/` into YAML, and a test runs each
  converted set.

**Done when:** `clm-ml run` on the converted CHATS7 files gives identical output to
`clm-ml-offline`, the regression test passes, and a tutorial notebook runs.

### Phase 2: Forcing and time become data ✅

**As built**, in three steps, each bit-for-bit identical to `clm-ml-offline`:

- **2a, forcing as data.** `TowerMetMod` and `clmDataMod` read every record a run needs
  once, before the time loop. The tower derivations that `TowerMetCurr` and `TowerMetNext`
  each repeated now live once, in `derive_tower_met`. `clm_ml.forcing.Forcing` holds the
  forcing as arrays shaped `(time, gridcell)` and `(time, patch)`. `ModelAdvance`/`clm_drv`
  receive a CLM-history record instead of file names. The unused duplicate driver
  `CLMml_drv` was removed.
- **2b, time as data.** The time manager has pure functions (`date_after`, `calday_of`,
  `step_time` → `StepTime`), and the physics (`MLCanopyFluxes`, `CanopyTurbulence`,
  `SoilTemperature`) receives time as an argument. `main.py` computes the time axis once.
  The start-date "hack" is replaced by a direct calculation, and the loop no longer
  mutates `itim`.
- **2c, forcing window.** The 24 `*_bef/_cur/_next_forcing` fields are gone from
  `mlcanopy_type`. `GetAtmForcing` takes an `AtmForcingWindow`, built each step from
  `Forcing`. The "previous" record uses the same conversion as `_GetCLMVar`
  (`clm_forcing_to_ml`), and "next" the same values as the former `TowerMetNext`, so
  results are unchanged.

Not done here, by design: the time manager still holds the static start date and step
length (read by the legacy output writer), and the driver still runs in `main.py` until
Phase 3 moves it into `clm_ml`.

**Follow-up 2d ✅: forcing read as a chunk.** `Forcing` is one chunk of `n_steps` steps
plus a halo record at each end (`n_steps + 2` records). Interior halos are the neighbouring
records; at the run's first and last steps the halo repeats the edge record, as in Fortran
(`clm_ml.forcing.chunk_run_steps`). `load_forcing(first_record, run_n_steps, first_step,
n_steps, history_index, …)` reads only the records a chunk needs. `TowerMetMod.tower_first_record`
finds the run's start in the tower file from its time axis, instead of assuming the file
starts at the run start. The CLM history index and step times are computed per step rather
than as whole-run lists. Tower runs still load one chunk covering the run, so results are
unchanged (bit-for-bit). `tests/clm_ml/test_forcing_chunks.py` checks that three chunks,
including halos, equal the matching slices of one whole-run read. The stream interface,
time interpolation and the chunk loop follow in Phase 4.

Planned scope:


- The loaders read all tower and CLM-history records into a `Forcing` pytree before the run,
  including derived quantities (solar zenith angle, CO₂, emissivity). Meteorology has shape
  `(time, gridcell, …)`, prescribed vegetation `(time, patch, …)`, and soil moisture
  `(time, column, …)`. These axes already exist, with size 1 for now.
- Replace the `clm_time_manager` globals with time arrays. Remove the start-date "hack".
- Remove `*_bef/_cur/_next` forcing fields from the state, in favor of the forcing window.

**Done when:** no file is opened during the run, and the regression test passes.

### Phase 3: Explicit state and a pure `step`

- Define `State` with only the prognostic variables: `CanopyState` with shape
  `(patch, max_layers, …)` and `SoilState` with shape `(column, nsoil)`. Find the prognostic
  fields by recording which `mlcanopy_type` fields are read before they are written in a step.
  The adapter slices the padded arrays down to the patch's `n_layers` for the old code.
- Write `Model.step` for one patch. Internally it unpacks into the existing `*_inst` objects,
  calls the existing physics sequence, and packs the results back. **No module globals are read
  or written.**
- Delete the `clm_instMod` globals, `global filter`, `DIFFERENTIABLE_MODE`, and the
  non-diff/diff branching in `MLCanopyFluxes`. Keep one physics path; its sub-step loop is a
  Python loop in normal runs and `lax.scan` when traced, until Phase 4 compiles whole chunks
  ([`numerics_and_regression.md`](numerics_and_regression.md) §1).
- Move first-step initialization (`ml_vert_init`, RK coefficient globals) into `Model` and
  `initial_state`.
- Pass `Params` everywhere, replacing the `vcmaxpft_jax`/`g1_MED_jax` threading.
- Make the acclimation temperature part of `State` (still initialized from the whole-run mean in
  this phase, so results are unchanged; Phase 4b switches to the 10-day running mean).

**Done when:** `jax.grad` works with respect to `Params`, `Forcing` and `State`, the regression
test passes, and no `global` statement remains in the run path.

Progress:
- 3a ✅ model state carried explicitly through the time step (`ClmInstances`).
- 3b ✅ explicit first-step initialization; no Runge–Kutta or flag globals.
- 3c ✅ one implementation per process (normal/traceable duplicates and `DIFFERENTIABLE_MODE`
  deleted; round-off-only change, checked routine by routine and with the noise twin).
- 3d ✅ parameters passed explicitly: each routine takes a `CanopyParams` (the patch's PFT
  values, `params.yaml` names, a pytree of JAX scalars) instead of reading the global
  `MLpftcon`/`pftcon` tables; `vcmaxpft_jax`/`g1_MED_jax` removed. The C3/C4 flag selects a
  code path, so it is static, in `GridInfo`. `MLCanopyFluxes` builds both from the tables
  (`patch_setup`); 3e builds them from `clm_ml.Params`. Remaining table reads are in
  initialization only (`MLinitVerticalMod`, `SoilStateInitTimeConst`).
- 3e (in progress): gradients through a full CLM step work ✅ (vectorized per-layer loops and
  Runge–Kutta stages as a scan; numerics doc §1; with respect to state and parameters, checked
  against finite differences). `clm_ml.state.State` ✅ (canopy air temperature, vapor
  pressure, CO₂, leaf temperature and water potential, intercepted water, ground and soil
  temperature): the time loop rebuilds each step from a fixed template plus `State`, bit for bit
  (fields found by NaN-poisoning each field before a step). Acclimation temperature in `State`
  ✅ (still the whole-run mean; Phase 4b). Gradients with respect to state, parameters and
  forcing ✅, each checked against a finite difference. `Params.canopy_params(pft)` builds the
  physics parameters from `params.yaml` (identical to the legacy tables, tested); `params` can be
  passed through `ModelAdvance`/`clm_drv`/`MLCanopyFluxes` ✅.
- **Moved to Phase 4:** the `clm_ml.Model` API (`initial_state`, `step`, `run`) and moving
  first-step initialization into it. Phase 4 builds the run loop as one compiled `scan` per
  chunk, which fixes the shape of these functions; building `Model.step` around the current
  per-step loop first would be done twice. The substance is in place: a step is a function of
  `(State, forcing, params)` (the loop rebuilds every step from a template plus `State`), and
  gradients with respect to all three work. No `global` statement executes in the time loop;
  the remaining ones run once at initialization (Phase 4 initial state, Phase 7 cleanup).

**Phase 3 status: done** (with the `Model` API moved to Phase 4, above).

### Phase 4: Chunked, compiled runs (one patch)

Progress:
- ✅ Driver split into `setup_run` / `step_inputs` / `advance` (`offline_executable/main.py`).
- ✅ A whole CLM step traces (structural lookups as NumPy constants, no globals written in the
  step, time and O₂ as data, orbital functions on traced calendar days).
- ✅ **Compiled chunks are the run loop**: after the first (initializing) step, `chunk_steps`
  steps (namelist; 0 = step by step; calendar months by default since decision 17, first
  implemented as 48 = one day) run as one `jit(scan)`. 50 CHATS7
  steps: 47 s including compilation (was ~135 s); a compiled step takes ~15 ms (was ~2.7 s). A
  short last chunk is padded to reuse the compiled scan. Two chunks of 4 equal one chunk of 8
  bit for bit (done-when 1). Compiled vs step by step: ≤ 6.6e-11 relative (round-off, within
  the noise twins); reference regenerated in compiled mode. **Interim:** each chunk still
  returns every step's full canopy container (~0.21 MB per step per patch, ~10 MB per day per
  patch) so the legacy text output can be written; fine for a tower, impossible globally.
  Replaced by the nested scan with output accumulation (§6.9).
- ✅ Forcing is read per chunk (with halo records), and the whole-run acclimation mean is summed
  over the forcing read chunk by chunk (same order, bit for bit). Peak memory footprint is flat
  with run length: 4.3 GB for 2 days, 4.2 GB for 20 days of CHATS7 (one month of forcing
  available). The resident set grew from 2.2 to 3.5 GB over the 20 days: to check on a longer
  run (possible slow leak). Per step, writing the legacy text output (~80 ms) now dominates
  the compiled physics (~15 ms); output streams will replace it.
- ✅ **Parameter ensembles**: the compiled chunk takes the patch's `CanopyParams`, and
  `jax.vmap` over a batch runs an ensemble (done-when 3, CPU part). 32 members with
  `vcmax25` × 0.8–1.2, one day (48 steps): 11 s after compiling (~7 ms per member-step), 3.1 GB
  peak; the default member reproduces the single run; at midday higher `vcmax25` gives cooler
  sunlit leaves (296.0, 295.9, 295.8 K for ×0.8, 1.0, 1.2), as expected from more
  transpiration. Test: `tests/regression/test_ensemble.py`.
- ✅ **Output streams** (decision 16, §6.9): `run.yaml` `output.streams` (name, frequency
  step/hourly/daily/monthly/yearly, statistic mean/instantaneous/min/max, variables, level,
  `file_per` run/year/month); Phase 1 keys are rejected with their replacements. Each step's
  variables (one definition for numpy and jax.numpy, `clm_ml.outputs`) are accumulated inside
  the compiled monthly chunk into one buffer slot per output record (`clm_ml.output_streams`;
  the equivalent of jax-gcm's nested scan, for several streams at once); records spanning
  chunks are completed on the host. The `clm_ml` run no longer records every step; the first
  (initializing) step is included on the host. `Run.execute()` returns one `Outputs` per
  stream and writes one netCDF file per stream, level and period. Raw legacy fields
  (`mlcanopy.<field>`) can be requested for validation; the regression uses that.
  Round-off only (different compiled program; reference regenerated).
  ✅ Each file is written as soon as its period (run, year, month) is complete
  (`StreamFileWriter`, in the worker); `Run.execute(return_outputs=False)` keeps no records in
  memory beyond the current file's.
- ✅ `jax.checkpoint` on the step of the compiled chunk (as jax-gcm).
- ✅ **Between chunks:** a health check of the state (finite; air, leaf, ground and soil
  temperatures within 150–350 K at every meaningful position) stops the run with the variable,
  its range and the chunk's dates. **Restarts** (`execution.restart_every`, e.g. `1 month`,
  `10 years`, and always at the end of the run) hold the state, its time and the output
  bookkeeping (partial records, records not yet in a file); written beside the old one and
  renamed over it, keeping `.prev`, and read with a fallback to `.prev`. Format: a versioned
  Python pickle (`{output}/restart/{tower}_restart.pkl`); **to become netCDF in Phase 7**.
  `initial_conditions.source: restart` resumes: a restart time inside the run continues it
  (results equal the uninterrupted run bit for bit, tested with a daily mean spanning the
  restart); a restart time equal to the run start begins a new run from that state (e.g.
  after a spin-up). The layer grid is rebuilt by running the initialization step, whose
  results are discarded. Note: a new run from a restart keeps the restart's acclimation
  temperature (the whole-run mean of the earlier run) until Phase 4b's running mean.
- ✅ **`ForcingSource`** (`clm_ml.forcing`): any object with `chunk(first_step, n_steps) ->
  Forcing` (with halos, at the model step). `clm_ml.legacy.forcing.TowerForcingSource` reads
  the tower meteorology and CLM history files; readers for coarser streams (6-hourly
  meteorology, daily or monthly LAI/SAI/soil water with interpolation to the model step,
  decision 13–14) implement the same interface (Phase 4b). The next chunk is read on a
  background thread while the current one runs.
- ✅ **`clm_ml.model.Model`** (in this process, for notebooks and calibration):
  `Model.from_run(run)`, `initial_state()`, `default_params()` (from params.yaml),
  `chunks()`, `forcing(first, n)`, and `run_chunk(state, forcing, first, params, streams)`
  → `(state, {stream: records})`, differentiable with respect to state, parameters and
  forcing (test: d(mean latent heat over 8 steps)/d(vcmax25) agrees with a finite
  difference). One model per process (legacy globals); a second run is refused.
  **XLA runtime bug:** with JAX 0.6.2, gradients over a chunk of ≥ 2 steps crash
  (segmentation fault) in XLA's default CPU ("thunk") runtime; the older runtime
  (`XLA_FLAGS=--xla_cpu_use_thunk_runtime=false`) computes them, with `jax.checkpoint` on the
  step. `clm_ml.model` selects it on import (warning if JAX has already started); normal
  runs keep the default runtime. Cost: an 8-step chunk gradient takes ~7 min and ~13 GB
  (mostly compilation). Re-test with newer JAX.
- ✅ **Memory with run length** (streams path, daily-mean stream, no records kept): 2 days
  3.56 GB peak footprint (3.32 GB resident), 31 days 3.68 GB (2.88 GB resident); the earlier
  growth of resident memory came from the per-step containers and text output, now gone.
  Only one month of CHATS7 forcing exists, so runs of several months (several chunks) are
  not tested here.
- ✅ **Timing test (CPU)**, `timing/results.jsonl`: the 50-step run takes 56 s (Phase 0:
  246 s), 39 s per simulated day (Phase 0: 214 s); this path still writes the legacy text
  output. A compiled step takes ~15 ms; a 31-day run with a daily stream takes 82 s.

- ✅ **Forcing split in time and the §4.1 `forcing:` layout** (follow-up, after Phase 4b):
  - `forcing.meteorology: {files, interpolation, shortwave, precipitation}`.
    `interpolation` is CLM-ml v2's rule within a step's sub-steps (`none` | `three_point`,
    met_type 0 / 3); `shortwave` (`coszen` | `linear` | `constant`) and `precipitation`
    (`constant` | `linear`) choose how a coarser data interval is brought to the model step
    (the other variables are always linear between interval midpoints).
  - `forcing.vegetation` / `forcing.soil_water: {source, files, interpolation}`; the CHATS
    adjustment moved to `forcing.soil_water.adjustment: {file, levels}`.
  - `files` is one path or a list. `{tower}`, `{year}` and `{month}` (e.g.
    `{year}-{month:02d}.nc`) are expanded for every month from one month before the run to
    one month after it; the files of months inside the run must exist (an error names the
    missing month), the neighbouring months' files are used if present (halo records).
  - Tower files are read as one series (`TowerMetMod.TowerFiles`): each file must start one
    interval after the previous one ends, or the run stops naming the gap. Land-data files
    are merged by time (records must not repeat; their intervals may be irregular).
  - **Deviation from the §4.1 sketch:** the CLM history file keeps its own key,
    `forcing.clm_history: {file, start}`, instead of being given as `vegetation.files` /
    `soil_water.files` with `source: clm_history`, because it also supplies the initial
    conditions and the code reads one history file for all three. The adjustment file is
    `file` (one file, indexed by CLM history time slice), not `files`.
  - The earlier keys (`forcing.meteorology` as a path, `forcing.interpolation`,
    `forcing.soil_moisture_adjustment`, `vegetation.file`) are rejected with a message naming
    the replacement; `clm-ml convert` writes the new layout.
  - Tests: daily-split CHATS7 tower files read the same records as the single file; a missing
    day is rejected; land-data files split in time read as one series; config errors; end to
    end, the CLM-history LAI/SAI as monthly files named by a `{year}-{month:02d}` template plus
    meteorology split into 31 daily files reproduce the default run (within 1e-6). CHATS7 is
    unchanged (bit for bit).

**Phase 4 status: done, except GPU timing** (done-when 3, GPU part; deferred, no GPU here).
Done-when 1 (a chunked run is bit for bit identical to one chunk: two chunks of 4 vs one of 8,
and a resumed run vs an uninterrupted one), done-when 2 (memory flat with run length, within
the one month of forcing available) and the CPU part of done-when 3 (32-member ensemble,
timing) are met.

- `Model.run(state, forcing_chunk, params) -> (state, outputs)` = `jit(scan(step))` over one
  chunk, with `jax.checkpoint` on the step.
- **Forcing streams** (§6.9): the `ForcingSource` interface, with readers for tower files and
  CLM history files, files split in time, and one-record halos at chunk boundaries.
- **The run loop:** chunks with background reading of the next chunk, restart files at chunk
  boundaries (`initial_conditions.source: restart`).
- **Output streams:** variables, frequency, statistic, level and file split, accumulated inside
  the scan and written per chunk.
- **`run.yaml` moves to the stream form of §4.1** (`execution`, `forcing` streams, `output.streams`).
  The Phase 1 keys (`time.step_seconds`, `forcing.meteorology` as a single file, a single
  `output` section) are rejected with a message naming their replacements, and
  `clm-ml convert` writes the new form.
- `vmap` over a **parameter ensemble** for one patch.
- Timing test against Phase 0, now including GPU. Watch compile time in particular (`main.py`
  notes ~290 s today).

**Done when:**
- A chunked run (for example 1-day chunks) is bit-for-bit identical to the same run in one
  chunk.
- Peak memory is flat with run length: a 1-year tower run uses about the same memory as a
  1-month run.
- A 32-member ensemble works, and timing-test results (CPU and GPU) are recorded.

### Phase 4b: Science changes for long runs

Separate commits, each documented, each with a regenerated regression reference and a recorded
comparison against the Fortran output:

- **10-day running-mean acclimation temperature** (decision 15), as the default.
  `acclimation_temperature: run_mean` keeps the Fortran definition for validation.
  ✅ Done. `model.yaml` `physics.acclimation_temperature: running_mean` (default) |
  `run_mean`, `physics.acclimation_days` (10). The running mean follows CLM's `runmean`
  accumulation: `mean += (T - mean) / min(n, N)` with `n` the steps so far and `N` the steps
  in `acclimation_days` (the plain mean of all steps for the first 10 days, then a running mean
  with a 10-day time scale). `T` is the tower air temperature (as in Fortran's whole-run mean);
  the first step starts the mean; it is updated before each later step. The mean and its step
  count (`State.canopy.acclimation_steps`) are in the state, so restarts continue it exactly.
  `run_mean` reproduces the previous reference bit for bit. Fortran comparison, May 2007
  (112 columns; the fraction-at-minimum-leaf-water-potential column is excluded: Fortran is 0
  at every step and the running mean touches the floor at one step, which the relative-error
  metric, scaled by the Fortran maximum, turns into 3763%):

  | Acclimation | Max relative error | Mean relative error | Columns with r < 0.9999 |
  |---|---|---|---|
  | `run_mean` (Fortran definition) | 0.16 % | 0.012 % | 7 |
  | `running_mean` (new default) | 0.95 % | 0.080 % | 22 |

  All columns stay within the 2.5 % tolerance; the Fortran comparison test should be run with
  `acclimation = 'run_mean'`. Reference and twins regenerated (science change).
- **Configurable model step with time interpolation** (decision 13): `numerics.step_seconds`,
  and the interpolation rules of §6.9 (linear, `coszen` shortwave, constant precipitation,
  midpoint interpolation of daily/monthly LAI, SAI and soil water). With CHATS7 (data at the
  model step) results are unchanged. New tests: 6-hourly forcing derived from the CHATS7
  half-hourly data, interpolated back to 30 minutes, conserves daily totals; monthly LAI is
  interpolated as specified.
  ✅ Done. `model.yaml` `numerics.step_seconds` (null: the forcing data interval, which
  `run.yaml` `time.step_seconds` gives); it must divide the data interval, and the
  substep must divide it. `clm_ml.forcing_interpolation`: a record stamped `t` is the mean
  over `[t, t + Δf]`, located at its midpoint; air temperature, humidity, wind, pressure,
  longwave and height are linear between midpoints (evaluated at the step midpoint); shortwave
  is scaled by `cosz(step) / cosz(interval)` (mean of `max(cos zenith, 0)`, the model's orbital
  functions), conserving each interval's mean (held constant if the sun is down for the whole
  interval); precipitation is held constant. Raw file variables are interpolated, then
  derived per step (humidity from RH, longwave, solar partitioning), as CLM's data atmosphere.
  The CLM history slice is the one whose interval (read from the file's time axis) contains
  the step's end. With equal intervals nothing is interpolated: CHATS7 results bit for bit
  unchanged, so the Fortran comparison is that of the item above.
  Tests: 6-hourly means of the CHATS7 data interpolated back to 30 min conserve daily and
  6-hourly shortwave and precipitation to 1e-12, put no sunlight at night, follow the sun
  (correlation with the observed half-hourly shortwave 0.92, vs 0.79 holding the 6-hour mean
  constant) and interpolate temperature linearly; end to end, 3 CHATS7 days forced by 6-hourly
  data at a 30-min step vs half-hourly forcing: daily net radiation within ~2 %, latent heat
  within ~7 %, GPP within ~6 %, sensible heat (the small residual) within ~11 W m-2.
  Daily or monthly LAI, SAI and soil water: next item.
- **Daily/monthly prescribed land data** (decision 14): readers for user-supplied LAI, SAI and
  soil water files.
  ✅ Done. `run.yaml`:

  ```yaml
  forcing:
    vegetation: {source: file, file: lai_monthly.nc}   # LAI, SAI (m2 m-2)
    soil_water: {source: file, file: h2osoi_daily.nc}  # H2OSOI (m3 m-3), (time, level)
  ```

  `source: clm_history` (the default) keeps the CLM history file, so CHATS7 is unchanged.
  The file has `time` in `days since <date>`, each record stamped at the start of its
  interval (intervals may be irregular: months); extra axes of size 1 (one gridcell) are
  dropped. Values are linear between interval midpoints, evaluated at each model step's
  midpoint, held beyond the first and last record (`forcing_interpolation.linear_between_
  midpoints`). Only the records bracketing a chunk are read, so memory does not depend on the
  file's length. `H2OSOI` replaces the CLM history soil water layer for layer (at least the
  model's `nlevsoi` = 20 levels); the CHATS soil-moisture adjustment factor still multiplies
  it. The leaf-area profile is rebuilt from the step's LAI and SAI, as with the CLM history.
  Tests (`tests/clm_ml/test_land_data.py`, `tests/regression/test_land_data_files.py`):
  midpoint interpolation with monthly and daily records, the windowed reader; end to end, the
  CHATS7 CLM-history LAI, SAI and soil water written as user files reproduce the default run
  (inputs to 1e-12, fluxes within 1e-6), and a monthly LAI file (April 1, May 2, June 3)
  gives the expected leaf area index at every step (to 1e-6: the output is the layers' sum,
  and the beta-distribution fractions sum to 1 only to ~1e-7).

### Phase 5: Explicit configuration and mask-aware layers (revised after Phase 4)

**Why Phase 5 exists.** Phase 6 runs many patches with `jax.vmap`, which needs every patch to use
the *same* compiled program. Today each patch's layer counts (`ncan`, `ntop`, `nbot`) are Python
integers inside the physics (`GridInfo`), so a patch with a different layer count compiles its own
program and cannot be batched with others. Phase 5 turns layer counts into data and every
layer loop into masked array operations. It also removes the module-global options, which are
the reason each model run needs a fresh Python process.

**Revised scope** (decisions 18–21, after Phase 4b): the original plan moved every process into
`clm_ml/physics/`, rewrote it with 0-based indexing and a new process interface, one PR per
process. Only the masks are needed for batching. Phase 5 therefore converts the code **in
place**: same files, same 1-based arrays, same Fortran golden tests. Moving files, 0-based
indexing and the §6.6 process interface move to Phase 7 (decision 18, 21). Reasons: jax-gcm's
"wrap, don't rewrite"; a renumbering of ~14,000 lines invites exactly the layer-range bugs
found in Phase 4 (the Runge–Kutta update skipped the top layer); the golden tests are 1-based.

#### Phase 5a: Explicit configuration (no module-global options)

Options and tables are copied into ~20 modules when they are imported
(`from multilayer_canopy.MLclm_varctl import gs_type`), so `clm_ml.legacy.worker` must set them in a
fresh process before the physics is imported; a process holds one model (`clm_ml.model.Model`
refuses a second), and every end-to-end test pays ~40–60 s of process start-up and compilation.

- **Inventory first:** every module-level option, table and size read at import (`MLclm_varctl`,
  `clm_varctl`, `MLpftconMod`/`pftconMod` tables and their `Init` functions, `TowerDataMod`,
  `clm_varpar` sizes such as `nlevgrnd`, `clm_instMod` instances still read as globals). Record it
  here with what replaces each.
- **Inventory** (done; AST scan of every import and attribute read, `src/` outside `clm_ml`):

  | Group | Globals | Read by | Replacement |
  |---|---|---|---|
  | Fixed indices and sizes | `isun`, `isha`, `nleaf`, `ivis`, `inir`, `numrad`, `mxpft`, `iulog` | many | **Keep**: true constants, identical for every model |
  | Scheme options | `MLclm_varctl`: `gs_type`, `gs_solver`, `gspot_type`, `colim_type`, `acclim_type`, `gb_type`, `turb_type`, `HF_extension_type`, `sparse_canopy_type`, `flux_profile_type`, `light_type`, `longwave_type`, `leaf_optics_type`, `runge_kutta_type`, `nrk`, `met_type`, `mlcan_to_clm` | 12 physics modules, copied at import | One static `MLOptions` argument (built from `ModelConfig`) |
  | Numerics | `dtime_ml` (sub-step) | 5 physics modules | In `MLOptions` (static) |
  | Layer structure | `dz_tall`, `dz_short`, `dz_param`, `dpai_min`, `nlayer_within`, `nlayer_above`; `nlevmlcan` = 100 | `MLinitVerticalMod`; 8 modules (size) | Arguments of the layer setup; `nlevmlcan` becomes `max_layers` in 5b |
  | Tunable numbers among the options | `kn_val` (leaf nitrogen decay coefficient) | `MLCanopyNitrogenProfileMod` | Parameter (differentiable); in `MLOptions` until then if needed |
  | PFT tables | `pftcon`, `MLpftcon` (and their `Init`, replaced by the worker) | `MLCanopyNitrogenProfileMod`, `MLCanopyParamsMod`, `SoilStateInitTimeConstMod`, `clm_initializeMod` | `Params` → `CanopyParams` (already the run path for most canopy parameters); the rest passed to the setup functions |
  | Tower table | `TowerDataMod.tower_*` (height, lat/lon, canopy height, beta parameters, PFT, roots, soil texture, colour, bedrock, time step) | `CLMml_driver`, `TowerMetMod`, `SurfaceAlbedoMod`, `SoilStateInitTimeConstMod`, `initVerticalMod`, `initGridCellsMod` | One tower record (from `Domain`) passed to the setup functions |
  | Soil layer structure | `clm_phys` → `nlevgrnd`, `nlevsoi`, `nlevsno` (`clm_varpar_init`), `nlev_soil_adjust` | 14 modules (sizes), `clmDataMod` | Static structure argument; the sizes are the same for CLM5.0 runs |
  | Run control and time | `controlMod`, `clm_time_manager.dtstep`/start date | `main`, `CLMml_driver` (host side) | Already data in the run path (Phase 2: `StepTime`); remove the host globals |
  | Model instances | `clm_instMod` | `main` only | Already explicit (Phase 3) |

  The time step reads only the scheme options, numerics, `kn_val`, `MLpftcon` and the fixed
  constants; the tables and soil structure are read by the setup code run once before the time
  loop. Order of work: scheme options and numerics (the traced step) first, then the setup
  code (layers, tower table, PFT tables), then the soil structure and host-side globals.
- **Options** (scheme choices, numerics) become one frozen, hashable object built from
  `ModelConfig`, passed down as an argument and treated as *static* by `jit` (changing an option
  recompiles, as decided for `model.yaml`).
- **Numbers** in the PFT and tower tables that are tunable become part of the parameters
  (differentiable, per PFT: `Params`/`CanopyParams`), never static. Structural values (sizes, PFT
  indices) are static.
- Adapters first: modules keep their names and Fortran-reference docstrings; functions gain an
  argument instead of reading a global. `Run.execute()` may keep its worker process (isolation for
  the command line), but nothing requires it.
- **Tests:** pure refactor, so **bit for bit**. New: two models with different options (e.g. WUE
  and Medlyn stomatal conductance) in one process each reproduce their separate-process runs.
  The regression tests then run the model **in-process** and share compiled chunks through
  session fixtures; record the new suite time (numerics doc §6.1).

Progress:
- ✅ **Scheme options and numerics in the time step**: `MLclm_varctl.MLOptions` (a hashable
  named tuple; `DEFAULT_OPTIONS` = the Fortran defaults) is built once by `setup_run` (from
  `model.yaml` through the worker, or the Fortran defaults plus the namelist's `met_type` and
  `dpai_min` for `clm-ml-offline`), kept in `LegacyRun.opts`, and passed as a keyword `opts`
  through `ModelAdvance` → `clm_drv` → `MLCanopyFluxes` → every process (radiation,
  turbulence, boundary layer, photosynthesis, flux–profile solution, hydraulics, canopy
  water, Runge–Kutta, forcing interpolation, layer structure). Compiled functions take it as
  a static argument; option-dependent kernel factories (boundary layer, photosynthesis) are
  cached per option value. The Runge–Kutta update takes its stage count from the tableau.
  Golden-test adapters pass options as arguments (the module-override helper is gone);
  the process-IO replay passes the CHATS7 options. Bit for bit.
- ✅ **Compiled programs cacheable** (test and start-up speed): profiling one CHATS7 run
  (57 s) showed ~39 s compiling ~540 programs: ~500 small ones from the eagerly run first
  step (below the 10 s threshold of the on-disk compilation cache) and four large ones,
  including the compiled chunk, that the cache refused because they contained host callbacks
  (the Fortran error checks). With the checks recorded instead (§6.8) and every program
  cached (`jax_persistent_cache_min_compile_time_secs = 0`; capped at 2 GB, least recently
  used removed, when the `filelock` package is installed), a second run of the same case in a
  new process takes **20 s instead of 58 s** and compiles nothing. Bit for bit.
- ✅ The Runge–Kutta increment arrays have room for the largest method (`MAX_RK_STAGES` = 4),
  so the container no longer depends on the option.
- ✅ **Several models of the same tower in one process**: the run holds its patch's
  `CanopyParams` (read from the PFT tables at setup; `LegacyRun.canopy_params`) and uses them
  for every step; tower values are read from the tower table when called, not copied at
  import; `worker.prepare_process` can run again for another model; `Model.from_run` accepts
  a second model of the same tower (and soil layer structure) and refuses another tower.
  Test (`tests/regression/test_models_in_one_process.py`): a WUE model, then a Medlyn model,
  then the WUE model again in one process: the WUE results are bit for bit unchanged, the
  Medlyn results equal those of a Medlyn model alone in its process, and the schemes differ.
- **Scope decided (after measuring):** "same tower, many models". The time step also reads
  CLM's subgrid structure from module state (`PatchType.patch`, `ColumnType.col` soil layers,
  `GridcellType.grc` latitude/longitude, soil-colour albedo tables), set up per tower; making
  it explicit is the `Domain` work of Phase 6/7. Phase 5a ends with several models of the
  *same* tower (different options or parameters) in one process; models of different towers
  in one process wait for Phase 6.

**Done when:** no physics module reads an option or table from a module global (CLM subgrid
structure excepted, Phase 6); several models of the same tower run in one process; the
regression suite's runs reuse compiled programs (in-process or through the compilation
cache).

#### Phase 5b: Mask-aware layers, in place

- **Step 0, audit:** list every remaining use of `ncan`/`ntop`/`nbot` as a loop bound, slice or
  branch (~37 loops, ~250 references), and every `is_concrete` guard. Several processes were
  already vectorized over layers in Phases 3–4 to cut gradient memory (solar radiation,
  turbulence, the flux–profile solution); order the work from the audit, simplest first, with the
  tridiagonal solver made padding-safe early (identity rows in padded layers).
- **Array size = `max_layers`:** the legacy containers are sized `nlevmlcan = 100`; outputs are cut
  to `max_layers` (50). Size every layer array by `max_layers` (CHATS7: 46 used of 50), which
  about halves the layer work.
- **Layer counts become data** of each patch's layer structure (with masks: within canopy, above
  canopy, in use); loops over a Python range become masked array operations or a `lax.scan` over
  `max_layers`. Padded layers hold neutral values (zero leaf and stem area) and are excluded from
  every sum; outputs are NaN there (as today).
- **Guards:** checks that need concrete numbers (`is_concrete`) are skipped under `vmap`; their
  physical content moves to the between-chunk health checks (plan §6.8) or a debug mode.
- **Gradient hygiene:** a masked `jnp.where` differentiates both branches, so an expression
  that divides by an empty layer's zero leaf area gives NaN gradients even when the forward value
  is right. Use the safe-`where` pattern (make the unused branch's inputs harmless).
- **Tests for each PR** (testing tiers, numerics doc §6.1):
  - the routine's Fortran golden tests, JSON unchanged;
  - bit-for-bit and noise-twin tests. Masked sums may change the order of additions; then the
    noise-twin test plus the identical-input routine check (§1 of the numerics doc) must pass,
    and the reference is regenerated, saying so in the commit;
  - a **padding test**: the same patch at its exact layer count and at 2× agrees (within noise);
  - a **gradient-with-padding test**: gradients are finite and agree with the unpadded ones.
- **Performance:** the CHATS7 timing test and the memory of a gradient over 8 steps (Phase 4:
  ~7 min, ~13 GB, mostly compilation) are recorded at the start and end of Phase 5b; a slowdown
  of more than ~1.5× is discussed before going on.

Progress:
- ✅ **Step 0, baselines** (`timing/time_model.py`, `timing/results_model.jsonl`; CHATS7, CPU,
  compile cache warm): compiled step 2.8 ms (47-step chunk); gradient over 8 steps 15.6 ms per
  step, 5.9 GB peak; timing test (`time_offline.py`, legacy text output) 7.2 s per simulated
  day (Phase 4: 39 s, before the compile cache).
- ✅ **Step 1, array size = `max_layers`**: the canopy container is created with
  `model.yaml` `canopy_layers.max_layers` slots (`MLOptions.max_layers`, passed through
  `InitializeRealize` → `initialize2` → `clm_instInit` → `create_mlcanopy`) instead of the
  Fortran parameter `nlevmlcan = 100`; routines take the slot count from their arrays. A
  canopy that needs more layers stops with a message naming `max_layers`.
  - The padding test found one read past the top layer (`_implicit_fps_jit` took
    `tair[2:n+2]` and overwrote the extra element): fixed, same values.
  - **Padding test** (`tests/regression/test_padding.py`): CHATS7 with 46 (no padding), 50
    and 100 slots agree on every output variable within 1e-9 relative. A gradient version
    (finite gradients, equal across sizes) is marked slow.
  - Round-off only: on the shared slots the 50-slot run differs from the 100-slot reference by
    at most 3.6e-11 relative (≤ 1.07 × the twin noise; sums over arrays of different length
    group their additions differently); the identical-input routine replay passes. The
    reference and twins were regenerated (with 50 slots). The 100-slot reference held real
    values in unused slots only for `lsc_profile` (computed for every slot, never read).
  - Restart format version 2 (layer arrays sized by `max_layers`); a restart whose arrays do
    not match the run (e.g. another `max_layers`) is refused with the field and shapes.
- ✅ **Step 2a, the patch's structure is an input of the compiled chunk**: the stream runner
  takes the template containers (layer heights, leaf area per layer, layer counts, soil
  properties, ...) as an argument instead of baking them in as constants, and the layer
  indices (`GridInfo`) are found once outside the compiled code and passed down
  (`advance(..., grid=)` → `ModelAdvance` → `clm_drv` → `MLCanopyFluxes`). One conversion of
  container data to NumPy in `clm_drv` (the vegetated-fraction filter) now uses only the
  array's shape. Bit for bit. The program is still specialized by the layer indices
  (`GridInfo.ncan/ntop/nbot`, static); steps 3–4 replace them with masks.
- ✅ **Step 3a, masks** (`multilayer_canopy/MLLayersMod.py`): `layers(inst, p)` gives the
  patch's counts as data and masks (`in_use` 1..ncan, `in_canopy` 1..ntop, `above_canopy`,
  `leafy` nbot..ntop). Routines compute on every slot; `pad_with_top` gives padding slots
  the inputs of layer `ncan` (always above the canopy and leafless), so padding computes a
  harmless leafless layer instead of arithmetic on fill values (1e36) that would give
  infinite values and NaN gradients; `restore_padding` puts the padding of every field a
  routine changed back as it was. Converted: Runge–Kutta update, photosynthesis (bit for
  bit).
- ✅ **Step 3b, longwave radiation**: the Norman system always has `2 * max_layers + 2`
  rows: the real rows in Fortran order, the sky row right after them (no upper coupling),
  then identity rows for unused leafy-layer slots, so each real row and the real solution
  are computed as before. Round-off only (the canopy total of absorbed longwave is a sum over
  all slots): noise-twin test and identical-input replay pass; reference regenerated.
- ✅ **Step 3c, solar radiation**: optical properties computed on every slot and kept in the
  leafy layers; the downward product of beam transmittance is a scan from the top with
  layers above the canopy multiplying by 1 (exact). Two-stream: both sweeps scan every slot,
  leafy layers computed as before and other slots passing the sweep's values through.
  Norman (solar): the same fixed-size system as longwave. The solar conservation checks
  (two-stream 1e-6, Norman 1e-3) are recorded like the other Fortran checks (they used to
  run only on concrete values). Round-off only (noise-twin and identical-input replay pass);
  reference regenerated.
  **Known issue, not from 5b:** the Norman solar option (`solar_radiation: norman`, not the
  CHATS default, never validated against Fortran in the JAX translation) stops at the first
  CHATS7 step with a conservation error, with the code before and after this step; to be
  investigated separately.
- ✅ **Step 3d, turbulence**: wind profile, momentum flux and aerodynamic conductances are
  formulas over every slot, chosen per layer by mask (above the canopy, within it, its top
  layer); the height above the top layer is zref; the Obukhov solver and roughness length
  were already per patch. Round-off only; reference regenerated.
- ✅ **Step 3e, flux–profile solution**: the coupled air temperature/humidity system is built
  over every slot; padding slots get the top layer's inputs and identity rows after layer
  `ncan` (whose upward coupling is zero), so the real solution is unchanged; results go to
  layers 1..ncan. The solved temperatures change in the last digit (2e-16), which appears
  as up to 7e-12 in fields that are differences of nearly equal values (sensible heat flux
  between layers, storage terms); the identical-input replay threshold is therefore 1e-10
  (the policy's defect threshold) instead of 1e-12. Reference regenerated. Found while
  converting: the CO₂ profile reset covers levels 0..ncan−1 instead of 1..ncan (no effect;
  numerics doc §4.5, not changed). Compile time grows with `max_layers` here (the solve is
  unrolled over every slot): the padding test now takes ~9 min.
- ✅ **Step 3f, the canopy driver** (`MLCanopyFluxesMod`): leaf/stem area profile update,
  copy of the previous state, sub-step flux accumulation and write-back, sun/shade merge and
  the canopy diagnostics work on every slot with masks. The remaining Fortran checks that ran
  only on concrete values (plant area profile; the four energy balances of
  `CanopyFluxesDiagnostics`) are recorded like the others (`BALANCE_CHECKS` now has 11).
  `GridInfo` no longer holds the layer counts (only the patch index and C3/C4 flag); old
  process-IO recordings are read with a compatibility class. Round-off only; reference
  regenerated.
- ✅ **Step 4, concrete-only checks**: the last Fortran stops that ran only on concrete values
  (soil-temperature energy conservation; sunlit/shaded fractions positive) are recorded like
  the others (13 checks), so they now also run in compiled chunks. The `HF2008` warning
  (`eta >= eta_max`) is a Fortran warning, not a stop: still printed only for concrete
  values. Bit for bit.
- ✅ **Step 5, done-when test** (`tests/regression/test_one_program_any_layers.py`): CHATS7
  with its default 46 layers and with explicit counts (10 within + 10 above = 20 layers),
  same `max_layers`: the 20-layer model's chunk runs through the 46-layer model's compiled
  program **without compiling again** (`_cache_size` unchanged) and gives its own results
  within 1e-9 of each variable (measured ~1e-11: separately compiled programs round
  differently).
- **C3/C4 as data: tried, measured, reverted (decision 22).** Computing both pathways for
  every patch and selecting each patch's own (commit `29cb2fb`) let one program serve C3 and
  C4 patches, but doubles the photosynthesis work for every patch on a GPU (patches in a batch
  run in lockstep, so a per-patch branch computes both), and the first implementation made
  the gradient 25× slower (below). C3/C4 is therefore a static choice again
  (`GridInfo.c3`); Phase 6 groups patches by pathway inside one compiled program (decision
  22). Reverted in commit "Phase 5b: C3/C4 static per group"; code, tests and regression
  reference are back to commit `33dea92` (bit for bit).
- **Timing after 5b** (`timing/results_model.jsonl`, CHATS7, CPU), against the step-0
  baselines: compiled step 4.3 ms (2.8 ms; 1.5×); gradient over 8 steps 386 ms per step
  (15.6 ms; **25×**), peak 4.4 GB (5.9 GB); timing test 8.2 s per simulated day (7.2 s).
  A diagnostic run with the C3 pathway only (not committed) gives 3.4 ms (1.2×) and 21 ms per
  gradient step (1.4×): the masks cost ~1.2–1.4×; the 25× comes from the C3/C4 merge, which
  selects between two complete canopy containers on each of ~60 photosynthesis calls per
  step. **Over the 1.5× limit: to discuss before going on.** Gradient compilation takes
  ~11 min (the before-5b first call, 35 s, probably came from the compilation cache, so
  compile times are not comparable yet).
- Gradient padding test (step 1 code): gradients finite for 46, 50 and 100 slots; they
  differ by up to 5e-9 relative (round-off amplified by the implicit-function derivatives
  of the solvers), so the gradient comparison uses 1e-7 (the finite-difference checks pass
  at 5 %).

**Done when:** the run path has no Python-integer layer bounds; **one compiled step serves
patches with different layer counts** (CHATS7 at `max_layers` 46, 50 and 100 agree within noise;
CHATS7 with explicit `n_layers_within`/`n_layers_above` that differ from the default runs with the
same compiled program and matches its own separate run); gradients are finite with padding; the
timing results are recorded.

**Phase 5b end-of-phase results** (final code: masks, C3/C4 static):
- Timing (CHATS7, CPU, against the step-0 baselines): compiled step 3.3 ms (2.8 ms, 1.18×);
  gradient over 8 steps 21.6 ms per step (15.6 ms, 1.38×), peak 5.1 GB (5.9 GB); timing test
  8.2 s per simulated day (7.2 s, 1.14×). Within the 1.5× limit. Gradient compilation ~11.5 min
  (CPU).
- Full regression folder: 192 passed (24 min with a cold compilation cache).
- Slow tests: differentiability (7, including the finite-difference checks) and `Model` API
  passed. The gradient padding test ran out of memory on the 17 GB CPU machine: gradient
  compilation memory grows with `max_layers`, because the longwave and flux–profile
  tridiagonal solves are unrolled over every slot (202 and 100 rows at 100 slots; before 5b,
  over the patch's real layers). The test now compares 46 and 60 slots, one process each;
  **not re-run on CPU (maintainer's decision): to be run on GPU after Phase 6.** Follow-up
  for Phase 6: write those two solves as `lax.scan` loops, so program size, compile time
  and memory do not grow with `max_layers` (round-off check required).
- Fortran comparison, May 2007, `acclimation = run_mean`, 112 columns (the
  fraction-at-minimum-leaf-water-potential column excluded as in Phase 4b): largest relative
  error 0.074 % (Phase 4: 0.16 %), mean 0.0075 % (0.012 %), 2 columns with correlation
  below 0.9999 (7): `flux` column 15 (r = 0.9968) and `aux` column 3 (r = 0.99990). All
  within the 2.5 % tolerance.

### Phase 6: Many patches and gridcells

**Where:** Phase 6 continues on NCAR **Derecho** with GPUs (maintainer's decision after
Phase 5, branch `phase6/patches`). The batching choices (patch-axis layout, acceptable padding
waste, C3/C4 group sizes, sharding) are made from GPU measurements. See "Handoff to Derecho"
below for setup and for the open items carried over from the CPU sessions.

**Status:**
- ✅ **Step 1, tridiagonal solves as `lax.scan`** (from Phase 5b): the longwave system (rows
  built as arrays, solved by `MLMathToolsMod.tridiag_scan`) and the coupled flux–profile solve
  (`tridiag_2eq`: elimination and back substitution as scans; its custom gradient rule as
  array operations) no longer unroll over the layer slots. Same arithmetic per row; round-off
  only (noise-twin and identical-input replay pass); reference regenerated. Effect (CHATS7,
  CPU, 8-step gradient): gradient compilation 694 s → 273 s, gradient 21.6 → 16.5 ms per
  step (before 5b: 15.6 ms), compiled step 3.3 → 3.2 ms; the forward padding test takes
  3 min instead of 10. (The Norman solar solver, not the default, is still unrolled.)

**Goal (decision 23):** global runs at about 0.5° with several patches per gridcell, one clock
for all gridcells (decision 24). Steps 0 and 2–6 below were revised for that goal after step
1b and agreed with the maintainer. Progress log: `docs/design/phase6_progress.md`.

**Step 1b, `vmap` throughput (measure before refactoring).** The design rests on one bet: that
one-patch physics batched with `vmap` keeps a GPU busy. A single patch does not: on an A100 the
compiled CHATS7 step takes 366 ms forward and 2.0 s for the gradient (8-step chunk), against
3.2 ms and 16.5 ms on the laptop CPU. The step is thousands of small operations in sequence
(scans over 50 layer slots, 25 Obukhov iterations, 20 bisection iterations, 6 sub-steps), so a
GPU waits on launch overhead. `timing/time_vmap.py` runs the existing compiled chunk on N
identical CHATS7 copies stacked along a new axis (no physics change) and records patch-steps
per second, compile time and peak device memory for forward and gradient
(`timing/results_vmap.jsonl`).

Results (one A100 40 GB, JAX 0.9.2, 8-step chunk, forward; every batched patch equals the
unbatched run exactly):

| Patches | Step (s) | Patch-steps/s | Peak device memory (GB) |
|---|---|---|---|
| 1 | 0.37 | 2.7 | 0.02 |
| 64 | 0.40 | 160 | 0.04 |
| 512 | 0.42 | 1 222 | 0.16 |
| 4 096 | 0.51 | 8 067 | 1.3 |
| 16 384 | 0.91 | 17 974 | 5.4 |
| 65 536 | 2.30 | 28 456 | 21.5 |

- The batching bet holds: up to ~4 000 patches cost almost nothing extra, and memory is
  ~0.33 MB per patch (~10⁵ patches fit on one A100).
- Two costs: a fixed ~0.37 s per step (a chain of thousands of small GPU kernels, each with
  launch overhead) and ~33 µs per extra patch-step, so a full A100 approaches ~30 000
  patch-steps/s. One laptop CPU core does ~300 for one patch, so one A100 is worth roughly one
  100-core CPU node. Estimated FP64 use is a few percent of the A100's peak: the step is
  bound by many small unfused operations, not arithmetic.
- A 0.5° global run (a few ×10⁵ patches, 30-min step, ~5×10⁹ patch-steps per year) would take
  ~2 GPU-days per simulated year at this rate.
- Compile time grows with N (110 s at N = 1, 460 s at N = 512); compiled programs are per
  patch count, so global runs pad the patch count to a fixed size per device.
- Conclusion: keep the design (`vmap` over patches, masks, one program); the cost is the
  number and size of the operations inside one step, which is step 1c.

**Step 1c, GPU performance of the step (measured; optimization deferred).** Arithmetic-
preserving changes only (round-off at most); science-adjacent changes (fewer solver iterations,
fewer sub-steps, another time integrator) are excluded by the maintainer. The maintainer
deferred the optimization itself ("modernize first, optimize the GPU later"); the
measurements below are the recorded starting point (`docs/design/phase6_progress.md` has the
details and file locations).
- Profile (`nsys`, one patch, one step): ~31 600 kernel launches per step, 2.3 µs of GPU work
  each on average; the GPU is busy 16 % of the wall time. The host cost is in about equal parts
  kernel launches, ~25 000 small device-to-device copies (loop-carried values of the scans and
  solver loops) and small CUDA-graph launches. XLA already captures straight-line code in CUDA
  graphs; loops are not captured.
- CUDA graphs over loops (`--xla_gpu_enable_command_buffer=...,CONDITIONAL,WHILE`,
  `--xla_gpu_graph_min_graph_size=1`) are *slower* with JAX 0.9.2 (N = 1: 530 ms per step
  against 367 ms; N = 4 096: 632 against 508 ms). Adding
  `--xla_gpu_command_buffer_unroll_loops=true` brings N = 1 back to the default (368 ms). No
  gain from flags alone.
- When the optimization resumes: fewer, larger operations inside the loops (a parallel
  tridiagonal solve: ~6 rounds instead of 50 sequential rows; fusing the per-layer leaf loops),
  guided by the profile.

**Agreed plan for steps 0 and 2–6 (maintainer, after step 1b).** Memory is not the obstacle
for a global run (~0.33 MB per patch: ~300 000 patches are ~25 GB per GPU on a 4-GPU node),
so the slim carried state stays in Phase 7. The obstacle is setup: 14–26 s per patch in the
legacy Python code with module state. Testing on Derecho: fast and before-commit tests on the
login node; slow and end-of-phase tests as PBS jobs (CPU: project P93300041; GPU: UCLB0069).

**Step 0, housekeeping (separate commits).**
- 0a. Regenerate the regression reference on Derecho: JAX 0.9.2 and the Derecho CPU change
  results by ~1e-16 (noise-twin passes, bit-for-bit fails on 65 fields), so the bit-for-bit
  test is only useful with a Derecho reference. The laptop (JAX 0.6.2) no longer matches bit
  for bit (accepted by the maintainer).
- 0b. float32 inputs to float64 (open issue below): a precision fix towards Fortran, ~1e-8;
  regenerated reference and a Fortran comparison run.
- 0c. The XLA CPU-runtime flag (`clm_ml.model`) only for JAX < 0.7: JAX 0.9.2 ignores it, and
  gradients over chunks do not crash with 0.9.2.

**Step 2, per-patch data: one compiled program for any patch.** The time step still reads
module state that is set up once per tower (audit at the end of Phase 5b):

| State | Read by | Varies between patches? | Change |
|---|---|---|---|
| Soil column geometry `ColumnType.col`: `z`, `dz`, `zi`, `nbedrock`, `snl` | `MLSoilTemperatureMod`, `MLPlantHydraulicsMod` (root/soil resistance), `SurfaceResistanceMod`, `SoilWaterMovementMod`, `clmDataMod`, `clm_driver` | yes (bedrock depth per site) | per-patch data; `nbedrock` is a Python loop bound today, so it becomes a **soil-layer mask** (as the canopy layers in 5b) |
| Gridcell latitude/longitude `GridcellType.grc` (set by `TowerMetMod.set_tower_location`) | solar geometry | yes (per gridcell) | per-gridcell data, gathered to patches |
| Soil colour `SurfaceAlbedoMod.isoicol` (the tables `albsat`/`albdry` are constants) | `SoilAlbedo` | yes | per-patch data |
| Index maps `PatchType.patch.column`, `.gridcell`, `.itype` | several | structure | identity per patch once each patch is a batch entry; the PFT is in the parameters, C3/C4 in the group (step 4) |
| Tower table `TowerDataMod` (soil texture, organic matter, etc.) | setup only (`SoilStateInitTimeConstMod`, `CLMml_driver`) | yes | read per patch at setup (already only at setup) |

- A per-patch record (pytree) with what the step reads and is fixed for a site: soil
  geometry with a **soil-layer mask** from the bedrock depth (as the canopy layers in 5b;
  float64, fixing the float32 cast in `MLPlantHydraulicsMod`), latitude/longitude (per
  gridcell, gathered to patches), and the soil's saturated/dry albedos looked up from the
  colour class at setup (the step no longer sees the class).
- `MLOptions` split: options that select code (static) and setup-only settings (layer spacing
  and counts, `dpai_min`, `max_layers`), so patches with different layer settings share one
  program.
- Explicit layer counts per patch (`n_layers_within`/`n_layers_above` in `towers.yaml`, later
  from the grid input), with the model-wide `model.yaml` values as defaults.
- Test: one compiled program runs two test patches that differ in bedrock depth, latitude,
  longitude, soil colour and layer counts, each equal to its own single run; CHATS7 alone
  stays bit for bit.

**Step 3, many gridcells and patches in one run (`vmap`).**
- Setup: each patch with the existing legacy code in a Python loop, stacked along a patch
  axis (enough for tests and up to ~10³ patches; step 6 replaces it).
- Forcing with a gridcell axis, gathered to patches; one clock (decision 24).
- Parameters: a table with one row per PFT, gathered per patch; gradients per PFT add up over
  its patches.
- Output: patch streams as now; gridcell streams as the area-weighted sum of patches formed at
  every step inside the scan (so min, max and instantaneous are right); profiles patch-level only.
- Errors per patch (decision 25): default stop naming the patches; option `exclude`.
- The patch count padded to a fixed size with zero-weight copies, so a new domain does not
  recompile; restarts with the patch axis.
- Remove the one-patch, one-tower and one-tower-per-process restrictions (`run.py`,
  `clm_ml.model._PREPARED`).
- Tests: 2 gridcells × 2 patches, every per-patch field different, equal to each patch run
  alone (round-off); permuting patches permutes outputs only; gridcell output equals the
  weighted sum for every statistic; gradient with respect to patch weights.

**Step 4, C3/C4 groups in one program** (decision 22): the C3 patches and the C4 patches form
two groups of fixed size, each batched with `vmap` and running only its own pathway
(`GridInfo.c3` static per group); outputs are combined and PFT-parameter gradients add up
across groups. Test: a gridcell with a C3 and a C4 patch matches each patch run alone. The same
group mechanism can later bucket `max_layers` sizes if padding waste matters.

**Step 5, several devices:** patches sorted by gridcell, no gridcell split between devices (so
gridcell sums need no communication); the patch axis is split across devices with
`jax.sharding`, PFT-parameter gradients summed across them. Test on CPU with simulated devices
(`--xla_force_host_platform_device_count`); timing on 4 A100s.

**Step 6, setup at global scale (new):** a batched (`vmap`) initialization over all patches
(vertical structure from canopy height and layer counts, soil initialization, the first
initializing step) in place of the per-patch legacy setup. Test: a synthetic global-size
domain built from CHATS7 (~3×10⁵ patches, varied latitude/longitude, soils and layer counts)
is set up in minutes and runs on one Derecho GPU node; memory and timing recorded.

**Follow-on, not Phase 6:** a loader for real global input (`domain.type: grid`: gridcells,
PFT patches and weights from a CLM surface dataset; gridded forcing, LAI and soil water). It
needs science choices: parameters for every PFT (only the CHATS7 walnut set is validated), the
layer rule for global canopies, and the datasets. Phase 6 documents how it plugs in.

**Done when:** multi-gridcell and multi-patch runs match single-patch runs; runs work across
4 GPUs and with simulated CPU devices; the synthetic global-size domain sets up and runs on a
GPU node.

#### Handoff to Derecho (written at the end of the CPU sessions)

**Setup:**
- Clone/pull the repository, branch `phase6/patches` (everything is pushed).
- `CLAUDE.md` is local and gitignored: copy it from the laptop
  (`~/Desktop/LEAP/clm-ml-jax/CLAUDE.md`) into the Derecho checkout. It holds the working
  rules and the handoff notes; this plan holds the full status and decisions.
- Python environment with JAX for CUDA (e.g. `jax[cuda12]` at the version used so far, 0.6.2,
  or newer), `pip install -e ".[dev]"` (black, ruff, pytest, pytest-cov). Float64 is enabled
  by the model (A100 float64 is fine).
- Compilation cache: `export JAX_COMPILATION_CACHE_DIR=<scratch>/jax_compile_cache` (default
  `~/.cache/jax_compile_cache`; home quotas are small). Install `filelock` so the cache is
  capped at 2 GB (least recently used removed).
- Work in an interactive GPU job (e.g. `qsub -I -q develop -l select=1:ngpus=1 ...`; check
  NCAR's documentation for current queue names and limits). First compilations take minutes
  (gradients ~5 min on CPU).

**Deferred to Derecho (from the CPU sessions), with results (2026-10-28, A100, JAX 0.9.2):**
1. ✅ **GPU timing:** one patch, forward 366–384 ms per step, gradient 2.0 s per step (8-step
   chunk; laptop CPU 3.2 and 16.5 ms); legacy offline run 3.05 s per step (laptop 0.17 s).
   Many patches: step 1b.
2. ✅ **Gradient padding test** (46 vs 60 slots): passes on the A100 (18 min).
3. ✅ **XLA CPU "thunk" runtime bug:** JAX 0.9.2 no longer supports the workaround flag (the
   new runtime is the only one), and a 2-step CPU gradient runs without crashing (step 0c).
4. ✅ **Compile time on GPU:** empty cache 100 s forward and 430 s gradient (8-step chunk);
   from the cache 13 s and 83 s.

**Open issues found during Phases 4–5 (not fixed; listed so they are not lost):**
- **Norman solar radiation** (`solar_radiation: norman`, not the CHATS default, never
  validated in the JAX translation) stops at the first CHATS7 step with a conservation error,
  before and after Phase 5b. Its tridiagonal solve is also still unrolled.
- **CO₂ profile reset** covers levels 0..ncan−1 instead of Fortran's 1..ncan (no effect while
  `cair` is constant): numerics doc §4.5.
- **`MLOptions` mixes run-time options and setup-only fields** (`dz_tall`, `dz_short`,
  `dz_param`, `nlayer_within/above`, `dpai_min`, `max_layers`). Compiled functions take the
  whole `MLOptions` as a static argument, so two models that differ only in layer settings
  compile separate programs, although the time step does not read those fields (seen in
  `test_one_program_any_layers.py`). Split the setup-only fields off before batching patches
  with different layer settings.
- **The `HF2008` warning** (`eta >= eta_max`, a Fortran warning, not a stop) is printed only
  when values are concrete; it is silent in compiled chunks.
- **Fortran comparison test** skips the files with several rows per step (`fsun`, `profile`,
  `fluxprofile`: 6 tests skipped) because it matches rows by time only.
- `offline_driver/controlMod.control` is not used by the run path (Phase 7 cleanup).
- **float32 inputs in a float64 model** (found on Derecho, Phase 6): `CanopyStateType`
  creates `elai_patch`, `esai_patch`, `htop_patch`, `frac_veg_nosno_patch` as float32, and
  `SurfaceAlbedoType` the albedos and `coszen_col`. LAI and SAI from the forcing are written into
  them each step (`clmDataMod`, JAX warns "cannot safely cast value from float64 to float32")
  and read by `MLCanopyFluxes` (`elai_patch[p]`); ground albedo enters the radiation. Fortran
  uses double precision. Effect ~1e-8 relative on these inputs, beyond the noise-twin band,
  and lower gradient precision with respect to LAI. Fix (float64) as its own commit with a
  regenerated reference.

**JAX lessons from these sessions (for writing tests and code):**
- **Never add host callbacks** (`jax.debug.callback`, `jax.debug.print`) to compiled code: a
  program that contains one is never stored in the compilation cache. Record errors in
  `mlcanopy.balance_error` (`MLCanopyFluxesType.record_balance_error`, `BALANCE_CHECKS`) and
  check them on the host (`check_balance`).
- A compiled function retraces when the **call signature** changes: passing an argument
  explicitly vs. letting it default, or a **weakly typed** value (`jnp.asarray(0.0)`) where a
  regular float64 array was passed before. Build replacement values with the original's dtype
  (`jnp.zeros_like(x)`).
- Under `vmap` (and on a GPU in general) a per-patch branch computes **both** branches; select
  on the outputs only, never on whole containers (the C3/C4 experiment: 25× slower gradients).
- Masked computations must feed unused entries finite inputs (`MLLayersMod.pad_with_top`):
  arithmetic on fill values (1e36) gives NaN gradients even when the result is discarded.
- Avoid Python loops over layers or rows in traced code (they unroll: compile time and memory
  grow with the loop length); use `lax.scan` or array operations.

**How to validate (commands that are only in this session's history):**
- Fortran comparison (end of a phase, after science-relevant changes): run May 2007 with the
  Fortran acclimation definition and compare the legacy `.out` files:

  ```python
  # write a namelist with absolute paths and the Fortran acclimation
  from tests.regression.record_run import write_portable_namelist
  nl = write_portable_namelist(Path("src/offline_executable/nl.CHATS7.05.2007"), out / "nl", out)
  # then add  acclimation = 'run_mean'  to the clmML_inparm group (f90nml)
  ```

  then `cd src && clm-ml-offline <out>/nl` (~4 min on CPU) and
  `CLM_ML_COMPARE_DIR=<out> python -m pytest tests/test_fortran_python_comparison.py -m slow`.
  Report max/mean relative error and the number of columns with correlation < 0.9999 over the
  112 columns, excluding `aux` column 2 (fraction at minimum leaf water potential, a metric
  artifact). Last result (end of Phase 5): max 0.074 %, mean 0.0075 %, 2 columns below 0.9999
  (`flux` col 15 r = 0.9968, `aux` col 3 r = 0.99990); Phase 4: 0.16 %, 0.012 %, 7. The two
  correlation tests fail on these columns (they did in Phase 4 too).
- After a round-off-only change: noise-twin test + identical-input replay
  (`tests/regression/test_process_io.py`, threshold 1e-10) + padding test, then
  `python tests/regression/record_run.py --update-reference` and say so in the commit.

### Phase 7: Restructure and cleanup

**7a. Restructure by process** (moved here from Phase 5; decisions 18, 21):
- Move each process from `multilayer_canopy/ML*Mod.py` into `clm_ml/physics/`, organized by
  physical process and scheme name (jax-gcm), one PR per process, keeping the `Fortran reference:`
  docstrings.
- The process interface of §6.6: a per-step state/diagnostics structure in place of the legacy
  `mlcanopy_type` container, variants registered by name, and each process declaring what it reads
  and writes (with a test).
- 0-based indexing and removal of Fortran-style filters: decide at the start of Phase 7 whether
  the benefit is worth the risk; if done, the golden tests get an index adapter (their JSON files
  stay unchanged). These are arithmetic-preserving changes, so each PR passes the bit-for-bit test.

**7b. Cleanup:**

- Delete unused Fortran-mirroring modules: `clm_src_cpl`, `decompMod`, `spmdMod`, `ncdio_pio`,
  `histFileMod`, `restUtilMod`, duplicated `cime_src_share_util`/`clm_share`, the CLM subgrid
  modules (`initSubgridMod`, `PatchType`, `ColumnType`, `GridcellType`, now replaced by
  `Domain`), and the remaining `*Type.py` containers.
- Move data files out of `src/`. Package lookup tables via `importlib.resources`. Keep example
  data under `examples/data/`.
- Remove the duplicated `src/pyproject.toml` and `src/pytest.ini`.
- Retire `clm-ml-offline` in favor of `clm-ml convert` + `clm-ml run` (see §8, question 4).
- **Restart files as netCDF** (maintainer request). Phase 4 writes them as a versioned Python
  pickle (`{output}/restart/{tower}_restart.pkl`): the state plus output bookkeeping (partial
  records, records not yet in a file). Change to netCDF with named, documented variables (state
  fields with their dimensions and units; the bookkeeping as additional variables/groups), keep
  the `.prev` rotation and fallback, and keep the resume test bit for bit.

**Done when:** `multilayer_canopy/` is no longer imported by the run path, `src/` contains only
`clm_ml/`, restarts are netCDF, and all tests pass.

### Phase 8: Documentation and release

- A getting-started guide (YAML-first), a full reference for every YAML key (generated from the
  dataclasses so it can't drift), a gradient-calibration example, an ensemble example, a
  multi-tower/multi-patch example, and a "for Fortran CLM-ml users" page mapping namelist keys
  and option codes to YAML.
- An API reference generated from docstrings.
- Update `CONTRIBUTING.md`: "add a process variant" replaces "mirror the Fortran module".
- Version 1.0.0.

### After Phase 8: option B (shared soil columns)

A separate science change: implement `soil.columns: shared` (§6.7), decide how per-patch ground
temperatures relate to the shared column, and validate it against observations at towers with
mixed vegetation before it becomes available for global runs.

### Rough effort

| Phase | Size | Notes |
|---|---|---|
| 0 | Small | Done |
| 1 | Medium | YAML schema, loaders, CLI, converter. No physics changes. |
| 2 | Medium | The 3-point interpolation needs care |
| 3 | Large | Untangling `mlcanopy_type` state from scratch fields |
| 4 | Medium | Chunk loop, forcing streams, output streams, restarts |
| 4b | Medium | Science changes with new references and validation |
| 5a | Medium | Explicit configuration; pure refactor (bit for bit); makes tests faster |
| 5b | Large | Masks in place, from the audit; many small PRs |
| 6 | Medium | Small if Phase 5 is done well; needs a GPU for timing |
| 7 | Large | 7a restructure by process (moved from Phase 5) + 7b mechanical cleanup |
| 8 | Small to medium | |

Milestones for users:
- After **Phase 1**: YAML configuration and the new CLI.
- After **Phase 4**: fast, differentiable single-patch runs and ensembles, of any length.
- After **Phase 4b**: configurable model step, 10-day acclimation, daily/monthly land data.
- After **Phase 5a**: several models (e.g. scheme comparisons) in one Python session.
- After **Phase 6**: many towers, mixed-vegetation gridcells, ready for global work.

---

## 8. Smaller decisions

| # | Question | Answer |
|---|---|---|
| 1 | Minimum Python version | **3.10** |
| 2 | Default `max_layers` | **50.** CHATS7 needs 46 layers with the default spacing: a 10 m canopy at 0.5 m gives 20 layers inside it, plus 26 above it up to the 23 m measurement height. Patches needing more fail at load time with a message naming them, and can raise `max_layers` in `model.yaml` (the Fortran limit was `nlevmlcan = 100`). **Resolved by row 9:** only CHATS7 is built in. (For reference: with default spacing, 9 of the 15 legacy towers need more than 50 (US-Ha1 60, US-Me2 64, US-UMB 92, US-MMS 96; US-ARM 300, beyond even the Fortran limit). Bonan et al. (2021) found 5–10 within-canopy layers reproduce the 50 cm benchmark, so explicit `n_layers_within`/`n_layers_above`, or coarser above-canopy spacing, may be the better default for multi-tower and global runs.) |
| 3 | Regression tolerance | **Revised in Phase 3:** bit-for-bit for refactors that should not change the arithmetic, and a **noise-twin** test for changes that alter round-off. The twins differ by ≤ 4e-11, so the noise-twin test is tight; an apparent evening-transition sensitivity (up to 4.5%) turned out to be a solver bug, now fixed. See [`numerics_and_regression.md`](numerics_and_regression.md) §5–6. |
| 4 | Legacy lifetime | Undecided. Default: keep `clm-ml-offline` until Phase 7, and keep the namelist converter permanently. Revisit at Phase 7. |
| 5 | Soil model scope | Undecided. Default: soil routines go behind the same process interface, migrated last in Phase 7 (decision 18). Making them swappable is not a goal yet. Revisit at Phase 7. (Soil column sharing is decided separately: option A now, option B later; see §6.7.) |
| 6 | Legacy `.out` format | **Not needed after validation.** It is kept only as long as the Fortran comparison test needs it, and is replaced by netCDF output once that test reads netCDF. `output.format` options become `netcdf` only. |
| 7 | GPU timing | **Deferred until after Phase 4.** GPU doesn't change the design. Note that the model runs in float64: data-center GPUs (A100/H100) handle this well, but consumer GPUs run float64 30–60× slower. A `precision` option could be considered in Phase 4, with its own validation. |
| 8 | Minimum patch weight | Default `0.0` (keep all patches). For global runs, dropping patches below, for example, 1% of a gridcell is a science choice left to `run.yaml` (`domain.min_patch_weight`). Remaining weights are renormalized to sum to 1. |
| 9 | Built-in towers | **CHATS7 only.** It is the only tower validated against Fortran, and the only one with input data in the repo. The other 14 `TowerDataMod` entries are not carried into `towers.yaml`. Users can describe other towers in their own `towers.yaml` (`domain.towers_file`), setting explicit layer counts where the default spacing gives too many layers (short canopies under tall towers). This also settles the `max_layers` follow-up in row 2. |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactoring silently changes results | Tight regression test (Phase 0) on every PR, plus the golden tests and the Fortran comparison |
| Padding changes results (for example, padded layers leaking into solvers or sums) | Padding test in every Phase 5b PR. Identity rows in the tridiagonal solves. Masks in every reduction. |
| Patch/gridcell aggregation errors (wrong weights, wrong index) | Phase 6 multi-patch test: a two-patch run equals the weighted sum of single-patch runs. Validation that weights sum to 1. |
| `mlcanopy_type` mixes state and scratch fields in non-obvious ways | Derive `State` by recording which fields are read before being written in a step |
| YAML files drift from the code (keys renamed, defaults changed) | The schema is generated from the dataclasses. `clm-ml check` runs in CI on every example. The key reference docs are generated from the same source. |
| Whole-run compilation is slow to compile (already ~290 s) | Keep the persistent compilation cache. Use `scan`, not unrolled loops. Run the timing test. Fall back to per-step `jit` if needed. |
| Gradient memory for long or many-patch runs | `jax.checkpoint` on the step, chunked runs, output selection and averaging inside the scan |
| Wasted compute on padded layers in global runs | Measure in Phase 6. Group patches by layer count if needed. |
| Value-dependent Python `if`s inside physics break tracing | Move them to initialization, or use `jnp.where`/`lax.cond`. Most were already handled for the diff-mode path. |
| Long-lived branch drifts from `main` | Every phase is merged on its own |
| Errors at chunk boundaries (halo records, carried state, partial output intervals) | Phase 4 test: chunked run is bit-for-bit identical to the unchunked run, with output intervals spanning chunks |
| Reading forcing becomes the bottleneck for global runs | Background reading of the next chunk, float32 storage, reading only local gridcells; measured in the timing test |
| Interpolation choices change results | Defaults reproduce CLM-ml v2 when data are at the model step; every rule is a named option; conservation tests (daily totals) |
| Gradients over very long runs | Windowed gradients from restart states (§6.9) |

---

## 10. Glossary

- **Gridcell / patch / soil column / tower:** see §3.1.
- **Pure function:** a function whose result depends only on its arguments, and which changes
  nothing outside itself. JAX's transformations assume this.
- **Pytree:** any nested structure of containers (dataclasses, tuples, dicts) holding arrays. JAX
  can `jit`, `grad` and `vmap` through pytrees as a whole.
- **Static vs traced:** *static* values (such as `max_layers` or a scheme choice) are fixed at
  compile time, and changing them triggers a recompile. *Traced* values (such as parameters or
  forcing) can change freely and can be differentiated.
- **`lax.scan`:** JAX's compiled loop. It carries `state` forward step by step and stacks each
  step's outputs.
- **`vmap`:** runs a one-patch function on many patches at once, as a single batched computation.
- **Gather / `segment_sum`:** a gather looks up a value for each patch from a smaller table (its
  gridcell's forcing, its PFT's parameters). A `segment_sum` adds patch values back into their
  gridcells (here, weighted by area). Both are differentiable.
- **Forcing stream / chunk / halo:** a stream is one kind of forcing data with its own files and
  time axis. A chunk is the part of a run held in memory at once. The halo is the one extra
  record at each end of a chunk, needed by the time interpolation.
- **Mask:** a true/false array saying which entries are real. Padded layers are false and are
  ignored by the physics.
- **Sharding:** splitting an array's patch axis across several GPUs so each one computes a share
  of the patches.
- **Adapter:** a thin layer that translates between the new interface and the old code, so the
  inside can change later without changing what users see.
- **Regression test:** a test that checks that output has not changed compared to a saved
  reference.
