# CLM-ML-JAX Modernization Plan

**Status:** Draft v3. Phase 0 is done (branch `phase0/safety-net`). v3 adds the gridcell/patch
structure (§3.1).

**Companion files:**
- [`api_sketch.py`](api_sketch.py): the proposed Python API, as typed signatures with docstrings.
- [`examples/`](examples/): example `run.yaml`, `model.yaml`, `params_chats.yaml` and
  `towers.yaml` files, written as a user would write them.
- [`timing.md`](timing.md): timing-test results.

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
   `rtol=1e-6`) and therefore the Fortran output, at every phase.

### Non-goals (for this effort)

- Changing the science or adding new physics. (Option B soil sharing is planned as a separate,
  validated science change after Phase 6.)
- **Loading gridded/global input data.** The architecture supports any number of gridcells and
  patches, and `run.yaml` reserves `domain.type: grid`. Reading gridded datasets (CLM surface
  datasets for patch weights and PFTs, plus reanalysis forcing) is a follow-on project.
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
   | `Forcing` | `run.yaml` → netCDF files | Meteorology per gridcell, plus prescribed CLM vegetation per patch and soil moisture per column | `(time, gridcell, …)`, `(time, patch, …)`, `(time, column, …)` | **Traced and differentiable**, loaded fully before the run |
   | `State` | `run.yaml` → initial conditions | Prognostic variables only: canopy per patch, soil per column | `(patch, …)`, `(column, …)` | **Traced and differentiable**. It is the `lax.scan` carry. |
   | `Outputs` | Produced by the run | Fluxes, profiles and diagnostics per patch. Gridcell means come from area weighting. | `(time, patch, …)` → `(time, gridcell, …)` | Returned and stacked by `scan` |

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
7. **A run is one compiled function.**
   `run = scan over time of (vmap over patches of step)`, where
   `step(state, forcing_window, params) -> (state, outputs)` acts on one patch.
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
  end:   2007-05-02T00:00         # or: days: 1
  step_seconds: 1800              # forcing time step
domain:
  type: towers                    # towers | grid (reserved for later)
  towers: [CHATS7]                # names from the built-in registry or towers_file
  towers_file: null               # optional: your own towers.yaml
  min_patch_weight: 0.0           # drop patches covering less than this fraction of a gridcell
forcing:
  meteorology: input_files/tower-forcing/{tower}/2007-05.nc    # {tower} is filled per tower
  interpolation: three_point      # none | three_point (met_type 0 / 3)
  clm_history:
    file: input_files/clm5_0/{tower}/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc
    start: 2007-04-01T00:00       # was clm_start_ymd / clm_start_tod
  soil_moisture_adjustment:
    file: input_files/clm5_0/{tower}/CHATS_soil_moisture_correction_29Jul25.nc
    levels: 3                     # was nlev_soil_adjust
initial_conditions:
  source: clm_history             # clm_history | restart
  restart_file: null
output:
  directory: output_files/
  format: netcdf                  # netcdf (legacy_ascii only during validation; see §8)
  variables: all                  # or a list of names
  frequency: step                 # step | hourly | daily
  level: [patch, gridcell]        # write patch output, gridcell means, or both
```

### 4.2 `model.yaml`: physics and numerics

```yaml
physics:
  stomatal_conductance: wue               # medlyn | ball_berry | wue
  water_stress_on_conductance: true
  wue_solver: bisection                   # brent | bisection
  photosynthesis_colimitation: true
  temperature_acclimation: true
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
  substep_seconds: 300
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

run = clm_ml.load_run("run.yaml")      # reads and validates run + model + params + towers, loads forcing
result = run.execute()                 # same as `clm-ml run run.yaml`
ds = result.to_xarray()                # dims (time, patch, layer, ...), with units and gridcell index
gc = result.gridcell_mean().to_xarray()  # area-weighted, dims (time, gridcell, ...)
```

Gradients with respect to parameters, forcing and initial state:

```python
model, params, forcing, state0 = run.model, run.params, run.forcing, run.initial_state

def loss(params, forcing, state0):
    out = model.run(state0, forcing, params).gridcell_mean()
    return jnp.mean((out.latent_heat_flux - obs_lh) ** 2)

g_params, g_forcing, g_state = jax.grad(loss, argnums=(0, 1, 2))(params, forcing, state0)
g_params.vcmax25          # shape (n_pft,): gradient per plant functional type
g_forcing.air_temperature # shape (time, gridcell)
```

Ensembles:

```python
batch = clm_ml.perturb(params, pft="broadleaf_deciduous_temperate_tree",
                       vcmax25=jnp.linspace(60, 140, 32))
lh = jax.vmap(lambda p: model.run(state0, forcing, p).latent_heat_flux)(batch)  # (32, time, patch)
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

`Model.run` wraps this as follows (option A):

```
scan over time:                                  # carry = State
    forcing_p = forcing_window[gridcell_index]   # gather gridcell forcing to patches
    params_p  = params.for_patches(pft)          # gather PFT parameters to patches
    vmap over patches:
        step(state[p], forcing_p[p], params_p[p], layers[p])
gridcell output = segment_sum(weight * patch output, gridcell_index)
```

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

### Phase 1: Configuration files and the new API as a wrapper (no physics changes)

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

### Phase 2: Forcing and time become data

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
  non-diff/diff branching in `MLCanopyFluxes`. Keep one path based on `lax.scan`.
- Move first-step initialization (`ml_vert_init`, RK coefficient globals) into `Model` and
  `initial_state`.
- Pass `Params` everywhere, replacing the `vcmaxpft_jax`/`g1_MED_jax` threading.

**Done when:** `jax.grad` works with respect to `Params`, `Forcing` and `State`, the regression
test passes, and no `global` statement remains in the run path.

### Phase 4: Whole-run compilation (one patch)

- `Model.run` = `jit(scan(step))` over all time steps, with `jax.checkpoint` on the step.
- Output selection and output frequency (`output.variables`, `output.frequency`). Averaging to
  hourly/daily happens inside the scan, so long runs don't store every step.
- Chunked runs and restart files (`initial_conditions.source: restart`).
- `vmap` over a **parameter ensemble** for one patch.
- Timing test against Phase 0, now including GPU. Watch compile time in particular (`main.py`
  notes ~290 s today).

**Done when:** 1-day and 31-day runs are each one compiled call, a 32-member ensemble works, and
timing-test results (CPU and GPU) are recorded.

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
  Timing test: patches per second on one GPU.
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
| 4 | Small to medium | Mostly timing tests and compile-time tuning |
| 5 | Large | The biggest phase. Many small PRs, which can be parallelized. |
| 6 | Medium | Small if Phase 5 is done well |
| 7 | Medium | Mechanical |
| 8 | Small to medium | |

Milestones for users:
- After **Phase 1**: YAML configuration and the new CLI.
- After **Phase 4**: fast, differentiable single-patch runs and ensembles.
- After **Phase 6**: many towers, mixed-vegetation gridcells, ready for global work.

---

## 8. Smaller decisions

| # | Question | Answer |
|---|---|---|
| 1 | Minimum Python version | **3.10** |
| 2 | Default `max_layers` | **50.** CHATS7 needs 46 layers with the default spacing: a 10 m canopy at 0.5 m gives 20 layers inside it, plus 26 above it up to the 23 m measurement height. Patches needing more fail at load time with a message naming them, and can raise `max_layers` in `model.yaml` (the Fortran limit was `nlevmlcan = 100`). **Follow-up needed:** with default spacing, 9 of the 15 built-in towers need more than 50 (US-Ha1 60, US-Me2 64, US-UMB 92, US-MMS 96; US-ARM 300, beyond even the Fortran limit). Bonan et al. (2021) found 5–10 within-canopy layers reproduce the 50 cm benchmark, so explicit `n_layers_within`/`n_layers_above`, or coarser above-canopy spacing, may be the better default for multi-tower and global runs. To confirm before Phase 1. |
| 3 | Regression tolerance | **`rtol=1e-6`** against the current JAX output |
| 4 | Legacy lifetime | Undecided. Default: keep `clm-ml-offline` until Phase 7, and keep the namelist converter permanently. Revisit at Phase 7. |
| 5 | Soil model scope | Undecided. Default: soil routines go behind the same process interface, migrated last in Phase 5. Making them swappable is not a goal yet. Revisit at Phase 5. (Soil column sharing is decided separately: option A now, option B later; see §6.7.) |
| 6 | Legacy `.out` format | **Not needed after validation.** It is kept only as long as the Fortran comparison test needs it, and is replaced by netCDF output once that test reads netCDF. `output.format` options become `netcdf` only. |
| 7 | GPU timing | **Deferred until after Phase 4.** GPU doesn't change the design. Note that the model runs in float64: data-center GPUs (A100/H100) handle this well, but consumer GPUs run float64 30–60× slower. A `precision` option could be considered in Phase 4, with its own validation. |
| 8 | Minimum patch weight | Default `0.0` (keep all patches). For global runs, dropping patches below, for example, 1% of a gridcell is a science choice left to `run.yaml` (`domain.min_patch_weight`). Remaining weights are renormalized to sum to 1. |

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
- **Mask:** a true/false array saying which entries are real. Padded layers are false and are
  ignored by the physics.
- **Sharding:** splitting an array's patch axis across several GPUs so each one computes a share
  of the patches.
- **Adapter:** a thin layer that translates between the new interface and the old code, so the
  inside can change later without changing what users see.
- **Regression test:** a test that checks that output has not changed compared to a saved
  reference.
