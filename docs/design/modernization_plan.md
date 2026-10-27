# CLM-ML-JAX Modernization Plan

**Status:** Draft v2 for review. No code changes until this plan and the API are approved.
**Branch:** `design/modern-api-plan`

**Companion files:**
- [`api_sketch.py`](api_sketch.py): the proposed Python API, as typed signatures with docstrings.
- [`examples/`](examples/): example `run.yaml`, `model.yaml`, `params_chats.yaml` and
  `sites.yaml` files, written as a user would write them.

---

## 0. Decisions recorded

| # | Question | Decision |
|---|---|---|
| 1 | Package name | Import as **`import clm_ml`**. The install name stays `clm-ml-jax` (the repo name). The command-line tool is `clm-ml`. |
| 2 | Option names | **Readable strings** (`stomatal_conductance: wue`). Fortran integer codes are accepted only by the namelist converter. |
| 3 | Spatial scope | **One site up to global.** A global run is many independent sites (no lateral exchange between grid cells). The design uses a **site dimension, a fixed number of canopy layers, and masks** from the start. |
| 4 | Gradients | With respect to **parameters, meteorological forcing, and state variables** (initial state). |
| 5 | Configuration | **YAML files**: `model.yaml` (physics and numerics), `params.yaml` (parameters by PFT), and `run.yaml` (the experiment, replacing the namelist). They are loaded into typed Python objects, which are what JAX runs on and differentiates. |
| 6 | Parameters | A **per-PFT table**. Each site looks up its PFT's values, so gradients come out per PFT. |
| 7 | Config framework | **Plain YAML plus our own validation** (dataclasses with clear error messages). No Hydra. |
| 8 | Namelists | Kept as a **one-time converter**: `clm-ml convert nl.CHATS7.1day` writes the YAML files. |

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
3. **One site up to many sites with the same code.** Physics is written for one column. Running
   many sites is batching (`vmap`), and later splitting sites across GPUs.
4. **One code path.** Today there is a "normal" mode (Python loops, file output, diagnostics) and a
   "differentiable" mode (`lax.scan`, diagnostics skipped). These become a single path that is
   both fast and differentiable.
5. **Swappable process schemes**, chosen by name in `model.yaml`.
6. **Keep scientific fidelity.** The refactored model must reproduce the current JAX output (to
   near machine precision) and therefore the Fortran output, at every phase.

### Non-goals (for this effort)

- Changing the science or adding new physics.
- **Loading gridded/global forcing.** The architecture supports many sites, and `run.yaml`
  reserves `domain.type: grid`. Reading gridded datasets (for example, CLM surface datasets and
  reanalysis forcing) is a follow-on project.
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
| Parameters | `MLpftcon`/`pftcon` global singletons. Differentiable parameters are threaded through as special-case arguments (`vcmaxpft_jax`, `g1_MED_jax`). CHATS values are switched on with `pftcon_val = 1`. | Only parameters someone has manually plumbed through can be differentiated. Site-specific values are hard-coded. |
| Interfaces | `MLCanopyFluxes` takes 20 arguments, mostly typed `Any`. `mlcanopy_type` has hundreds of fields mixing inputs, forcing, prognostic state, scratch space and diagnostics. | It is hard to see what a routine reads or writes. |
| Forcing | Tower and CLM-history netCDF files are re-read inside the time loop. For 3-point interpolation, the state carries `*_bef/_cur/_next` copies of forcing. | File I/O inside the loop blocks compilation. Forcing is mixed into model state. |
| Time loop | The outer loop is a Python `for` with ASCII writes each step. The inner 300 s sub-step loop is a Python `for` in normal mode and a `lax.scan` in differentiable mode. | It is slow, and two code paths can silently diverge. |
| Layers and sites | The number of canopy layers (`ncan`, `ntop`, `nbot`) depends on canopy height, and is used as a Python loop bound. Arrays are 1-based with a "patch" dimension and Fortran-style filters. | Sites with different canopy heights have different array shapes, so they **cannot be batched together**. This is the main blocker for multi-site and global runs. |
| Output | Six fixed-format ASCII files (`flux.out`, `fsun.out`, …) | Not self-describing (no names, units or time coordinate). |
| Validation | Strong: 160+ routine-level golden tests at `rtol=1e-9`, plus end-to-end Fortran comparison (2.5% / r ≥ 0.9999) and differentiability tests. | Needs one addition: a tight end-to-end regression against the current JAX output (Phase 0). |

---

## 3. Design principles

1. **Pure functions.** Every physics function has the shape `f(inputs) -> outputs`. It reads no
   globals and mutates nothing.
2. **Files describe, objects run.** Each YAML file loads into a typed Python object. Validation
   happens at load time, with messages that name the file and key. JAX only ever sees the objects.
3. **Separate the kinds of data.** Each kind gets its own container, because JAX treats each one
   differently:

   | Container | Comes from | What it holds | JAX treatment |
   |---|---|---|---|
   | `ModelConfig` | `model.yaml` | Scheme choices, time steps, layer settings, `max_layers` | **Static.** Changing it recompiles. |
   | `Sites` | `run.yaml` → built-in or user `sites.yaml` | Per-site location, PFT, soil texture, heights, profile shapes | Arrays with a leading site axis |
   | `Layers` | Derived from `Sites` + `ModelConfig` | Per-site layer heights and **masks** inside a fixed `max_layers` | `max_layers` is static. Heights and masks are arrays. |
   | `Params` | `params.yaml` | **Per-PFT table** of physical parameters | **Traced and differentiable** |
   | `Forcing` | `run.yaml` → netCDF files | Meteorology plus prescribed CLM vegetation/soil, shape `(time, site, …)` | **Traced and differentiable**, loaded fully before the run |
   | `State` | `run.yaml` → initial conditions | Prognostic variables only, shape `(site, …)` | **Traced and differentiable**. It is the `lax.scan` carry. |
   | `Outputs` | Produced by the run | Fluxes, profiles and diagnostics, shape `(time, site, …)` | Returned and stacked by `scan` |

4. **Write physics for one column, batch over sites.** Each process is written as if there is one
   site, with arrays of shape `(max_layers,)`. The model applies `jax.vmap` over the site axis.
   jax-gcm does the same with `vectorize_columns`. Splitting sites across GPUs is then a sharding
   annotation on that axis, not a code change.
5. **Fixed layers plus masks.** Every site gets `max_layers` canopy layers. A per-site mask says
   which layers are real air layers, and another says which contain leaves. Physics uses masks,
   never site-specific loop bounds. Padded layers must have no effect: they are zero-weighted in
   sums and decoupled (identity rows) in the tridiagonal solves.
6. **Parameters by PFT.** `Params` holds a table with a value per PFT for every field. Inside a
   run, each site's values are gathered from its PFT (`params.for_sites(sites.pft)`). Gathering is
   differentiable, so gradients come back per PFT and sites that share a PFT add up.
7. **A run is one compiled function.**
   `run = scan over time of (vmap over sites of step)`, where
   `step(state, forcing_window, params) -> (state, outputs)` acts on one column.
8. **Adapters first, rewrites later.** Put the new interface around the existing translated code
   first. Then move internals across one process at a time, each step verified by tests.
9. **Fortran traceability moves to docstrings.** Each refactored function keeps a
   `Fortran reference:` line in its docstring.
10. **Few dependencies.** Containers are standard `dataclasses` registered as JAX pytrees. New
    runtime dependencies: `pyyaml`, `xarray`, `netCDF4` (currently dev-only but needed at
    runtime).

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
  step_seconds: 1800              # tower / forcing time step
domain:
  type: towers                    # towers | grid (reserved for later)
  sites: [CHATS7]                 # names from the built-in registry or sites_file
  sites_file: null                # optional: your own sites.yaml
forcing:
  meteorology: input_files/tower-forcing/{site}/2007-05.nc     # {site} is filled per site
  interpolation: three_point      # none | three_point (met_type 0 / 3)
  clm_history:
    file: input_files/clm5_0/{site}/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc
    start: 2007-04-01T00:00       # was clm_start_ymd / clm_start_tod
  soil_moisture_adjustment:
    file: input_files/clm5_0/{site}/CHATS_soil_moisture_correction_29Jul25.nc
    levels: 3                     # was nlev_soil_adjust
initial_conditions:
  source: clm_history             # clm_history | restart
  restart_file: null
output:
  directory: output_files/
  format: netcdf                  # netcdf (legacy_ascii only during validation; see §8)
  variables: all                  # or a list of names
  frequency: step                 # step | hourly | daily
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
  max_layers: 50                          # fixed array size for every site
  dz_tall: 0.5
  dz_short: 0.1
  tall_canopy_threshold: 2.0
  n_layers_above: null
  n_layers_within: null
  min_plant_area_per_layer: 0.01          # dpai_min
soil:
  layer_structure: clm5_0                 # clm5_0 | clm4_5 (was clm_phys)
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

A site or experiment file **extends** the defaults and lists only what changes. This replaces
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
```

Validation rejects unknown PFT names, unknown parameter names, and missing values for any PFT
used by a site in the run. Ranges from the units table are checked as warnings.

### 4.4 `sites.yaml`: site descriptions

The built-in tower table (now in `TowerDataMod.py`) moves to `clm_ml/data/sites.yaml`. Users can
supply their own file in the same format:

```yaml
CHATS7:
  latitude: 38.49
  longitude: -121.84
  pft: broadleaf_deciduous_temperate_tree
  soil: {texture_class: silty clay loam, sand_percent: 10.0, clay_percent: 35.0,
         organic_matter: 50.0, color_class: 15, bedrock_depth: 2.0}
  reference_height: 23.0
  canopy_height: 10.0
  root_biomass: 500.0
  leaf_area_profile_beta: [2.6, 1.3]
  stem_area_profile_beta: [1.8, 1.3]
  forcing_step_minutes: 30
```

`null` means "use the default", replacing the Fortran `-999` sentinel. See
[`examples/sites_example.yaml`](examples/sites_example.yaml).

### 4.5 Commands

```bash
clm-ml run run.yaml
clm-ml run run.yaml --set model.physics.stomatal_conductance=medlyn --set time.end=2007-05-08
clm-ml check run.yaml                    # validate all files, report layer counts, no run
clm-ml convert nl.CHATS7.1day -o chats7/ # namelist -> run.yaml, model.yaml, params.yaml
clm-ml-offline < nl.CHATS7.1day          # legacy entry point, kept during transition
```

---

## 5. Python API (summary)

Full signatures are in [`api_sketch.py`](api_sketch.py).

```python
import clm_ml

run = clm_ml.load_run("run.yaml")      # reads and validates run + model + params + sites, loads forcing
result = run.execute()                 # same as `clm-ml run run.yaml`
ds = result.to_xarray()                # dims (time, site, layer, ...), with units
```

Gradients with respect to parameters, forcing and initial state:

```python
model, params, forcing, state0 = run.model, run.params, run.forcing, run.initial_state

def loss(params, forcing, state0):
    out = model.run(state0, forcing, params)
    return jnp.mean((out.latent_heat_flux - obs_lh) ** 2)

g_params, g_forcing, g_state = jax.grad(loss, argnums=(0, 1, 2))(params, forcing, state0)
g_params.vcmax25          # shape (n_pft,): gradient per plant functional type
g_forcing.air_temperature # shape (time, site)
```

Ensembles:

```python
batch = clm_ml.perturb(params, pft="broadleaf_deciduous_temperate_tree",
                       vcmax25=jnp.linspace(60, 140, 32))
lh = jax.vmap(lambda p: model.run(state0, forcing, p).latent_heat_flux)(batch)  # (32, time, site)
```

Changing settings in Python without editing files:

```python
params = params.set("broadleaf_deciduous_temperate_tree", vcmax25=110.0)
model = clm_ml.Model(model.config.replace(integrator="rk2_midpoint"), model.sites)
```

---

## 6. Target architecture

### 6.1 Package layout

```
src/clm_ml/
  __init__.py            # public API (load_run, Model, ModelConfig, Params, ...)
  run.py                 # Run: load_run(), execute()
  model.py               # Model: initial_state, step, run (scan over time, vmap over sites)
  config.py              # ModelConfig + YAML loading/validation
  sites.py               # Sites + registry
  layers.py              # Layers: fixed-size vertical grid + masks (from MLinitVerticalMod)
  params.py              # Params (per-PFT table), YAML loading, extends/overrides, perturb()
  forcing.py             # Forcing, loaders, time interpolation
  state.py               # CanopyState, SoilState, State
  outputs.py             # Outputs, to_xarray, netCDF / legacy ASCII writers, variable metadata
  constants.py
  physics/               # one-column, mask-aware processes
    solar_radiation.py  longwave.py  turbulence.py  leaf_boundary_layer.py
    photosynthesis.py   plant_hydraulics.py  canopy_water.py  soil.py
    integrators.py      math.py
  io/
    yaml_schema.py       # shared validation helpers, error formatting
    netcdf.py
    legacy.py            # namelist converter, Fortran option-code table, .out writers
  cli.py                 # `clm-ml run|check|convert`
  data/
    sites.yaml           # built-in towers (moved from TowerDataMod.py)
    params/clm5_defaults.yaml
    rsl_lookup_tables/psihat.nc
```

The existing packages (`clm_src_*`, `multilayer_canopy`, `offline_driver`, …) stay in place during
the transition, and the new API calls into them through adapters. Each process moves into
`clm_ml/physics/` as it is refactored, and old modules are deleted only when nothing imports them.

### 6.2 One time step (for one column)

The science order is the same as today. Each box is a pure function:

```
step(state, forcing_window, params_for_this_site, layers_for_this_site)
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
 └─ soil_temperature(...)
returns (new_state, outputs_t)
```

`Model.run` wraps this as follows:

```
scan over time:                       # carry = State (site, ...)
    vmap over sites:
        step(state[s], forcing_window[:, s], params.for_sites(pft)[s], layers[s])
```

`forcing_window` holds the previous, current and next tower records. That removes the
`*_bef/_cur/_next` forcing copies from model state.

### 6.3 Static vs traced: what can change without recompiling

These **recompile** (they are static): anything in `model.yaml` (including `max_layers`), the
number of sites, and the number of time steps.

These do **not** recompile, and all are differentiable: parameter values, forcing values, and
initial state. Site properties (heights, soil texture, PFT) are also arrays and can change without
recompiling, provided every site still fits within `max_layers`. PFT indices and layer masks are
integers and booleans, so they are not differentiated.

This replaces today's `DIFFERENTIABLE_MODE` flag and `GridInfo` workaround with a single rule.

### 6.4 Fixed layers and masks

- `Layers` is computed per site at load time, with the same algorithm as `MLinitVerticalMod`.
  It is then **padded to `max_layers`**.
- Fields (each per site): `n_layers` (int), `top_leaf_layer`/`bottom_leaf_layer` (int),
  `heights`/`thickness` `(max_layers,)`, `is_air_layer` `(max_layers,)` bool, and
  `has_leaf` `(max_layers,)` bool.
- **Padding rule.** Padded layers sit above the domain top. They have zero thickness and zero
  plant area, are excluded from every sum, and are identity rows in the tridiagonal systems. So
  a padded single site gives the same answer as the unpadded run. The regression test checks this.
- If any site needs more than `max_layers`, `clm-ml check`/`load_run` fails with a message giving
  the site, the layers needed, and the configured maximum.
- **Cost.** Short canopies compute on unused layers. If this matters for global runs, a later
  optimization can group sites into buckets with similar layer counts, each compiled separately.

### 6.5 Parameters by PFT

- `Params` fields have shape `(n_pft,)`, or `(n_pft, 2)` for spectral optics. PFT order is fixed
  by the defaults file (CLM ordering, `not_vegetated` = 0).
- `params.for_sites(sites.pft)` returns per-site values with shape `(n_site,)`. The physics only
  ever sees per-site values.
- Includes both today's `MLpftcon` fields and the CLM `pftcon` fields the canopy model uses (leaf
  dimension, leaf angle, leaf/stem optics, root profile, SLA, C3/C4 flag). The C3/C4 flag is a
  model structure choice, so it is stored as an integer and is not differentiated.
- **Future extension** (not in this plan): allow a parameter to vary per site or per grid cell
  through a netCDF override file. The `for_sites` gather is the single place that would change.

### 6.6 Process interface

Each process is a plain function on one column:

```python
def process(step: StepState, inputs: StepInputs, params: SiteParams, layers: SiteLayers,
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

### 6.7 Errors and checks

- Load-time validation of all YAML. Errors name the file, key, bad value and allowed values.
- `clm-ml check` validates everything and prints a summary (sites, PFTs, layer counts, run
  length, estimated memory) without running.
- `endrun` calls inside traced code are replaced by an opt-in `check: true` mode using
  `jax.experimental.checkify` (for NaNs and balance errors).
- Energy- and water-balance errors are included in `Outputs` by default.

---

## 7. Migration plan

Each phase ends with all tests passing and a mergeable PR. **The model must give the same answer
at the end of every phase.**

### Phase 0: Safety net

- **Add an end-to-end regression test against the current JAX output** at tight tolerance
  (`rtol=1e-6`). Use `nl.CHATS7.50steps` and commit the reference output as `.npz`. The
  Fortran comparison (2.5%) is too loose to catch refactoring mistakes.
- Fix `requires-python` (current JAX needs ≥ 3.10). Add `pyyaml` and `xarray`, and move
  `netCDF4` to runtime dependencies.
- Record a baseline timing test (`timing/time_offline.py`): compile time and wall time per
  simulated day on CPU and GPU. Results go in `docs/design/timing.md`.

**Done when:** the regression test passes on `main`, and timing-test numbers are recorded.

### Phase 1: Configuration files and the new API as a wrapper (no physics changes)

- Create `src/clm_ml/` with `ModelConfig`, `Sites`, `Params`, `Run`, the YAML loaders and
  validation, and the CLI (`run`, `check`, `convert`).
- Write `data/params/clm5_defaults.yaml` from `MLpftconMod`/`pftconMod`, and
  `data/sites.yaml` from `TowerDataMod`. Add a test that the YAML values equal the current Python
  tables exactly.
- `Run.execute()` translates the new objects into the existing globals, calls the existing driver,
  and collects outputs into arrays. It writes netCDF (with names and units) and/or the legacy
  `.out` files byte-for-byte.
- In this phase, **one site per run**. Multi-site `run.yaml` files are validated but rejected
  with a clear "not yet supported" error.
- `clm-ml convert` turns every namelist in `offline_executable/` into YAML, and a test runs each
  converted set.

**Done when:** `clm-ml run` on the converted CHATS7 files gives identical output to
`clm-ml-offline`, the regression test passes, and a tutorial notebook runs.

### Phase 2: Forcing and time become data

- The loaders read all tower and CLM-history records into a `Forcing` pytree before the run,
  including derived quantities (solar zenith angle, CO₂, emissivity). Shape is
  `(time, site, …)`, already with the site axis (size 1 for now).
- Replace the `clm_time_manager` globals with time arrays. Remove the start-date "hack".
- Remove `*_bef/_cur/_next` forcing fields from the state, in favor of the forcing window.

**Done when:** no file is opened during the run, and the regression test passes.

### Phase 3: Explicit state and a pure `step`

- Define `State` (`CanopyState`, `SoilState`) with only the prognostic variables, shape
  `(site, max_layers, …)`. Find the prognostic fields by recording which `mlcanopy_type` fields are
  read before they are written in a step. The adapter slices the padded arrays down to the
  site's `n_layers` for the old code.
- Write `Model.step` for one column. Internally it unpacks into the existing `*_inst` objects,
  calls the existing physics sequence, and packs the results back. **No module globals are read
  or written.**
- Delete the `clm_instMod` globals, `global filter`, `DIFFERENTIABLE_MODE`, and the
  non-diff/diff branching in `MLCanopyFluxes`. Keep one path based on `lax.scan`.
- Move first-step initialization (`ml_vert_init`, RK coefficient globals) into `Model` and
  `initial_state`.
- Pass `Params` everywhere, replacing the `vcmaxpft_jax`/`g1_MED_jax` threading.

**Done when:** `jax.grad` works with respect to `Params`, `Forcing` and `State`, the regression
test passes, and no `global` statement remains in the run path.

### Phase 4: Whole-run compilation (single site)

- `Model.run` = `jit(scan(step))` over all time steps, with `jax.checkpoint` on the step.
- Output selection and output frequency (`output.variables`, `output.frequency`). Averaging to
  hourly/daily happens inside the scan, so long runs don't store every step.
- Chunked runs and restart files (`initial_conditions.source: restart`).
- `vmap` over a **parameter ensemble** at one site.
- Timing test against Phase 0. Watch compile time in particular (`main.py` notes ~290 s today).

**Done when:** 1-day and 31-day runs are each one compiled call, a 32-member ensemble works, and
timing-test results are recorded.

### Phase 5: Mask-aware, one-column processes

This phase is what enables multi-site runs. The old internals use `ncan`/`ntop`/`nbot` as Python
loop bounds, so sites with different layer counts cannot be batched until each process is
rewritten to use masks.

- Move each process from `multilayer_canopy/ML*Mod.py` into `clm_ml/physics/`, one PR per process.
  Each one is rewritten to work on one column with `(max_layers,)` arrays and masks, 0-based
  indexing, and no patch dimension.
- Suggested order, simplest first: math utilities (make the tridiagonal solver padding-safe) →
  water vapor → leaf heat capacity → canopy water → longwave → solar radiation → leaf boundary
  layer → plant hydraulics → photosynthesis/stomata → turbulence/flux-profile → soil.
- Each PR passes three tests: the golden tests for that routine (with the JSON files unchanged),
  the end-to-end regression test, and a **padding test** (the same site run with `max_layers` at
  its exact layer count and at 2×; results must agree).

**Done when:** `multilayer_canopy/` is no longer imported by the run path, and all three kinds of
tests pass.

### Phase 6: Multi-site runs

- Turn on `vmap` over the site axis. Remove the "one site per run" restriction from Phase 1.
- Load forcing for several towers (the `{site}` placeholders in file paths).
- Multi-site test: running CHATS7 and US-UMB together must match running each alone.
- Device sharding over the site axis (`jax.sharding`) so large site counts spread across GPUs.
  Timing test: sites per second on one GPU.
- Document how a gridded/global run will plug in (a `domain.type: grid` loader that builds
  `Sites` and `Forcing` from gridded datasets). Implementing it is the follow-on project.

**Done when:** a multi-site run matches single-site runs, and sharding works on a multi-GPU
machine (or on CPU with simulated devices in CI).

### Phase 7: Cleanup

- Delete unused Fortran-mirroring modules: `clm_src_cpl`, `decompMod`, `spmdMod`, `ncdio_pio`,
  `histFileMod`, `restUtilMod`, duplicated `cime_src_share_util`/`clm_share`, and the remaining
  `*Type.py` containers.
- Move data files out of `src/`. Package lookup tables via `importlib.resources`. Keep example
  data under `examples/data/`.
- Remove the duplicated `src/pyproject.toml` and `src/pytest.ini`.
- Retire `clm-ml-offline` in favor of `clm-ml convert` + `clm-ml run` (see §8, question 4).

**Done when:** `src/` contains only `clm_ml/`, and all tests pass.

### Phase 8: Documentation and release

- A getting-started guide (YAML-first), a full reference for every YAML key (generated from the
  dataclasses so it can't drift), a gradient-calibration example, an ensemble example, a
  multi-site example, and a "for Fortran CLM-ml users" page mapping namelist keys and option
  codes to YAML.
- An API reference generated from docstrings.
- Update `CONTRIBUTING.md`: "add a process variant" replaces "mirror the Fortran module".
- Version 1.0.0.

### Rough effort

| Phase | Size | Notes |
|---|---|---|
| 0 | Small | Tests and packaging |
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
- After **Phase 4**: fast, differentiable single-site runs and ensembles.
- After **Phase 6**: many sites at once, ready for global work.

---

## 8. Smaller decisions

| # | Question | Answer |
|---|---|---|
| 1 | Minimum Python version | **3.10** |
| 2 | Default `max_layers` | **50.** CHATS7 needs 46 layers with the default spacing: a 10 m canopy at 0.5 m gives 20 layers inside it, plus 26 above it up to the 23 m reference height. Sites needing more fail at load time with a message naming the site, and can raise `max_layers` in `model.yaml` (the Fortran limit was `nlevmlcan = 100`). **Follow-up needed:** with default spacing, 9 of the 15 built-in towers need more than 50 (US-Ha1 60, US-Me2 64, US-UMB 92, US-MMS 96; US-ARM 300, beyond even the Fortran limit). Bonan et al. (2021) found 5–10 within-canopy layers reproduce the 50 cm benchmark, so explicit `n_layers_within`/`n_layers_above`, or coarser above-canopy spacing, may be the better default for multi-site runs. To confirm before Phase 1. |
| 3 | Regression tolerance | **`rtol=1e-6`** against the current JAX output |
| 4 | Legacy lifetime | Undecided. Default: keep `clm-ml-offline` until Phase 7, and keep the namelist converter permanently. Revisit at Phase 7. |
| 5 | Soil model scope | Undecided. Default: soil routines go behind the same process interface, migrated last in Phase 5. Making them swappable is not a goal yet. Revisit at Phase 5. |
| 6 | Legacy `.out` format | **Not needed after validation.** It is kept only as long as the Fortran comparison test needs it, and is replaced by netCDF output once that test reads netCDF. `output.format` options become `netcdf` only. |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactoring silently changes results | Tight regression test (Phase 0) on every PR, plus the golden tests and the Fortran comparison |
| Padding changes results (for example, padded layers leaking into solvers or sums) | Padding test in every Phase 5 PR. Identity rows in the tridiagonal solves. Masks in every reduction. |
| `mlcanopy_type` mixes state and scratch fields in non-obvious ways | Derive `State` by recording which fields are read before being written in a step |
| YAML files drift from the code (keys renamed, defaults changed) | The schema is generated from the dataclasses. `clm-ml check` runs in CI on every example. The key reference docs are generated from the same source. |
| Whole-run compilation is slow to compile (already ~290 s) | Keep the persistent compilation cache. Use `scan`, not unrolled loops. Run the timing test. Fall back to per-step `jit` if needed. |
| Gradient memory for long or many-site runs | `jax.checkpoint` on the step, chunked runs, output selection and averaging inside the scan |
| Wasted compute on padded layers in global runs | Measure in Phase 6. Group sites by layer count if needed. |
| Value-dependent Python `if`s inside physics break tracing | Move them to initialization, or use `jnp.where`/`lax.cond`. Most were already handled for the diff-mode path. |
| Long-lived branch drifts from `main` | Every phase is merged on its own |

---

## 10. Glossary

- **Pure function:** a function whose result depends only on its arguments, and which changes
  nothing outside itself. JAX's transformations assume this.
- **Pytree:** any nested structure of containers (dataclasses, tuples, dicts) holding arrays. JAX
  can `jit`, `grad` and `vmap` through pytrees as a whole.
- **Static vs traced:** *static* values (such as `max_layers` or a scheme choice) are fixed at
  compile time, and changing them triggers a recompile. *Traced* values (such as parameters or
  forcing) can change freely and can be differentiated.
- **`lax.scan`:** JAX's compiled loop. It carries `state` forward step by step and stacks each
  step's outputs.
- **`vmap`:** runs a one-site (one-column) function on many sites at once, as a single batched
  computation.
- **Mask:** a true/false array saying which entries are real. Padded layers are false and are
  ignored by the physics.
- **Sharding:** splitting an array's site axis across several GPUs so each one computes a share of
  the sites.
- **Adapter:** a thin layer that translates between the new interface and the old code, so the
  inside can change later without changing what users see.
- **Regression test:** a test that checks that output has not changed compared to a saved
  reference.
