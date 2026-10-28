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
- Rewriting the numerics of each routine, beyond what fixed layers and masks require (Phase 5).

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
    interpolation: three_point    # none | linear | three_point (CLM-ml v2, centered)
    shortwave: coszen             # spread within an interval: coszen | interpolate | constant
    precipitation: constant       # fluxes held constant over the interval (conserves totals)
  vegetation:                     # LAI, SAI per patch
    source: clm_history           # clm_history | file
    files: input_files/clm5_0/{tower}/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc
    interpolation: linear         # for daily/monthly data
  soil_water:
    source: clm_history
    files: input_files/clm5_0/{tower}/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc
    interpolation: linear
    adjustment:
      files: input_files/clm5_0/{tower}/CHATS_soil_moisture_correction_29Jul25.nc
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
`clm_ml/physics/` as it is refactored, and old modules are deleted only when nothing imports them.

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

- **Chunk length** is a run setting (`execution.chunk`). The compiled scan is reused for every
  chunk of the same length. Calendar months have different lengths, so either compile one
  scan per distinct length (at most four) or pad the last steps with masked steps; this is
  decided in Phase 4.
- **Reading overlaps computing.** The next chunk is read while the current one runs. jax-gcm
  does the same (its design note `chunked_run_io_overlap.md`).
- **Storage precision.** Forcing is stored and transferred as float32 and computed in float64.
- **Restarts** are written at chunk boundaries (`execution.restart_every`). A run can resume
  from any restart.
- **Acclimation temperature.** The 10-day running mean (decision 15) is part of the `State`,
  updated every step, so it needs no look-ahead over the forcing. The legacy whole-run mean
  needs a pre-pass over all forcing, so it is only allowed for short runs (validation against
  Fortran).
- **Sharding for global runs.** Patches are split across devices (Phase 6). Each process reads
  only the gridcells its patches belong to, so no single process holds global forcing.

**Output streams** (decision 16). Each stream accumulates its statistic during the scan
(running sums for means, running min/max, or the instantaneous value at the output time). Only
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
  (fields found by NaN-poisoning each field before a step). Remaining: `Model.step` in
  `clm_ml`, `Params` from `params.yaml` into `CanopyParams`, acclimation temperature in `State`,
  gradients with respect to forcing.

### Phase 4: Chunked, compiled runs (one patch)

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
- **Configurable model step with time interpolation** (decision 13): `numerics.step_seconds`,
  and the interpolation rules of §6.9 (linear, `coszen` shortwave, constant precipitation,
  midpoint interpolation of daily/monthly LAI, SAI and soil water). With CHATS7 (data at the
  model step) results are unchanged. New tests: 6-hourly forcing derived from the CHATS7
  half-hourly data, interpolated back to 30 minutes, conserves daily totals; monthly LAI is
  interpolated as specified.
- **Daily/monthly prescribed land data** (decision 14): readers for user-supplied LAI, SAI and
  soil water files.

### Phase 5: Mask-aware, one-patch processes

This phase is what enables batching patches. The old internals use `ncan`/`ntop`/`nbot` as Python
loop bounds, so patches with different layer counts cannot be batched until each process is
rewritten to use masks.

- Move each process from `multilayer_canopy/ML*Mod.py` into `clm_ml/physics/`, one PR per process.
  Each one is rewritten to work on one patch with `(max_layers,)` arrays and masks, 0-based
  indexing, and no Fortran-style filters.
- Suggested order, simplest first: math utilities (make the tridiagonal solver padding-safe) →
  water vapor → leaf heat capacity → canopy water → longwave → solar radiation → leaf boundary
  layer → plant hydraulics → photosynthesis/stomata → turbulence/flux-profile → soil.
- Each PR passes three tests: the golden tests for that routine (with the JSON files unchanged),
  the end-to-end regression test, and a **padding test** (the same patch run with `max_layers` at
  its exact layer count and at 2×; results must agree).

**Done when:** `multilayer_canopy/` is no longer imported by the run path, and all three kinds of
tests pass.

### Phase 6: Many patches and gridcells

- Turn on `vmap` over the patch axis, with forcing gathered from gridcells and outputs
  aggregated back (`segment_sum`). Remove the one-patch restriction from Phase 1.
- Load forcing for several towers (the `{tower}` placeholders in file paths).
- Tests:
  - **Multi-tower:** running CHATS7 and US-UMB together must match running each alone.
  - **Multi-patch:** a two-patch tower must match each patch run alone, with the gridcell output
    equal to the weighted sum.
  - **Weights:** gradients with respect to patch weights work.
- Device sharding over the patch axis (`jax.sharding`) so large patch counts spread across GPUs.
  Each process reads only the forcing for its own gridcells (§6.9). Timing test: patches per
  second on one GPU.
- Document how a gridded/global run will plug in: a `domain.type: grid` loader that builds the
  `Domain` (gridcells, PFT patches and weights from a CLM surface dataset) and `Forcing` from
  gridded data. Implementing it is the follow-on project.

**Done when:** multi-tower and multi-patch runs match single-patch runs, and sharding works on a
multi-GPU machine (or on CPU with simulated devices in CI).

### Phase 7: Cleanup

- Delete unused Fortran-mirroring modules: `clm_src_cpl`, `decompMod`, `spmdMod`, `ncdio_pio`,
  `histFileMod`, `restUtilMod`, duplicated `cime_src_share_util`/`clm_share`, the CLM subgrid
  modules (`initSubgridMod`, `PatchType`, `ColumnType`, `GridcellType`, now replaced by
  `Domain`), and the remaining `*Type.py` containers.
- Move data files out of `src/`. Package lookup tables via `importlib.resources`. Keep example
  data under `examples/data/`.
- Remove the duplicated `src/pyproject.toml` and `src/pytest.ini`.
- Retire `clm-ml-offline` in favor of `clm-ml convert` + `clm-ml run` (see §8, question 4).

**Done when:** `src/` contains only `clm_ml/`, and all tests pass.

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
| 5 | Large | The biggest phase. Many small PRs, which can be parallelized. |
| 6 | Medium | Small if Phase 5 is done well |
| 7 | Medium | Mechanical |
| 8 | Small to medium | |

Milestones for users:
- After **Phase 1**: YAML configuration and the new CLI.
- After **Phase 4**: fast, differentiable single-patch runs and ensembles, of any length.
- After **Phase 4b**: configurable model step, 10-day acclimation, daily/monthly land data.
- After **Phase 6**: many towers, mixed-vegetation gridcells, ready for global work.

---

## 8. Smaller decisions

| # | Question | Answer |
|---|---|---|
| 1 | Minimum Python version | **3.10** |
| 2 | Default `max_layers` | **50.** CHATS7 needs 46 layers with the default spacing: a 10 m canopy at 0.5 m gives 20 layers inside it, plus 26 above it up to the 23 m measurement height. Patches needing more fail at load time with a message naming them, and can raise `max_layers` in `model.yaml` (the Fortran limit was `nlevmlcan = 100`). **Resolved by row 9:** only CHATS7 is built in. (For reference: with default spacing, 9 of the 15 legacy towers need more than 50 (US-Ha1 60, US-Me2 64, US-UMB 92, US-MMS 96; US-ARM 300, beyond even the Fortran limit). Bonan et al. (2021) found 5–10 within-canopy layers reproduce the 50 cm benchmark, so explicit `n_layers_within`/`n_layers_above`, or coarser above-canopy spacing, may be the better default for multi-tower and global runs.) |
| 3 | Regression tolerance | **Revised in Phase 3:** bit-for-bit for refactors that should not change the arithmetic, and a **noise-twin** test for changes that alter round-off. The twins differ by ≤ 4e-11, so the noise-twin test is tight; an apparent evening-transition sensitivity (up to 4.5%) turned out to be a solver bug, now fixed. See [`numerics_and_regression.md`](numerics_and_regression.md) §5–6. |
| 4 | Legacy lifetime | Undecided. Default: keep `clm-ml-offline` until Phase 7, and keep the namelist converter permanently. Revisit at Phase 7. |
| 5 | Soil model scope | Undecided. Default: soil routines go behind the same process interface, migrated last in Phase 5. Making them swappable is not a goal yet. Revisit at Phase 5. (Soil column sharing is decided separately: option A now, option B later; see §6.7.) |
| 6 | Legacy `.out` format | **Not needed after validation.** It is kept only as long as the Fortran comparison test needs it, and is replaced by netCDF output once that test reads netCDF. `output.format` options become `netcdf` only. |
| 7 | GPU timing | **Deferred until after Phase 4.** GPU doesn't change the design. Note that the model runs in float64: data-center GPUs (A100/H100) handle this well, but consumer GPUs run float64 30–60× slower. A `precision` option could be considered in Phase 4, with its own validation. |
| 8 | Minimum patch weight | Default `0.0` (keep all patches). For global runs, dropping patches below, for example, 1% of a gridcell is a science choice left to `run.yaml` (`domain.min_patch_weight`). Remaining weights are renormalized to sum to 1. |
| 9 | Built-in towers | **CHATS7 only.** It is the only tower validated against Fortran, and the only one with input data in the repo. The other 14 `TowerDataMod` entries are not carried into `towers.yaml`. Users can describe other towers in their own `towers.yaml` (`domain.towers_file`), setting explicit layer counts where the default spacing gives too many layers (short canopies under tall towers). This also settles the `max_layers` follow-up in row 2. |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactoring silently changes results | Tight regression test (Phase 0) on every PR, plus the golden tests and the Fortran comparison |
| Padding changes results (for example, padded layers leaking into solvers or sums) | Padding test in every Phase 5 PR. Identity rows in the tridiagonal solves. Masks in every reduction. |
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
