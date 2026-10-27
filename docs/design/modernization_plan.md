# CLM-ML-JAX Modernization Plan

**Status:** Draft for review. No code changes until this plan and the API are approved.
**Branch:** `design/modern-api-plan`
**Companion file:** [`api_sketch.py`](api_sketch.py) contains the proposed public API as typed
signatures with docstrings. Review the two files together.

---

## 1. Goals and non-goals

### Goals

1. **A Python-first interface.** A scientist should be able to run a site, change a parameter,
   run an ensemble, or take a gradient in a few lines of Python. They should not need to edit a
   namelist or know the Fortran module structure.
2. **Explicit state, no hidden globals.** Everything the model depends on goes in as an argument,
   and everything it changes comes back as a return value. This is what makes `jit`, `grad` and
   `vmap` work reliably.
3. **One code path.** Today there is a "normal" mode (Python loops, file output, diagnostics) and a
   "differentiable" mode (`lax.scan`, diagnostics skipped). These should become a single path
   that is both fast and differentiable.
4. **Swappable process schemes.** Choosing Medlyn vs Ball-Berry vs WUE stomatal conductance, or
   Norman vs two-stream radiation, should be a readable configuration choice. Adding a new scheme
   should not require editing the driver.
5. **Keep scientific fidelity.** The refactored model must reproduce the current JAX output (to
   near machine precision) and therefore the Fortran output, at every phase.

### Non-goals (for this effort)

- Changing the science or adding new physics.
- Global or gridded runs. Single-site runs are the target. Multi-site runs come from `vmap`
  (see §7, question 3).
- Coupling to CESM/CTSM. The `clm_src_cpl` code stays as a reference only.
- Rewriting the numerics of each routine. Internals are refactored gradually, behind tests.

---

## 2. Where we are today (summary of the audit)

| Area | Current design | Problem it causes |
|---|---|---|
| Code organization | One Python file per Fortran module (`clm_src_main/`, `MLCanopyFluxesMod.py`, …), with ~1,760 "Fortran line" references | You have to know the Fortran CLM layout to navigate. The top-level package names (`clm_src_main`, `utils`, …) are generic and can collide with other installed packages. |
| Model state | ~15 `*_inst` objects held as module globals in `clm_instMod` and reassigned after every step (25 `global` statements in total) | Invisible to `jit`/`grad`/`vmap`. You cannot run two simulations side by side in one Python process. |
| Time | `clm_time_manager.itim`, `start_date_ymd`, … are module globals that the driver mutates (including a documented "hack" that temporarily overwrites the start date) | Time cannot be traced, so the outer loop cannot be compiled. |
| Run options | `MLclm_varctl` module globals (`gs_type = 2`, `light_type = 2`, `runge_kutta_type = 41`, …) plus a Fortran namelist read from stdin | Options are "magic integers". Changing them means mutating module state before import order matters. |
| Parameters | `MLpftcon` global singleton. Differentiable parameters are threaded through as special-case arguments (`vcmaxpft_jax`, `g1_MED_jax`). | Only parameters someone has manually plumbed through can be differentiated. |
| Interfaces | `MLCanopyFluxes` takes 20 arguments, mostly typed `Any`. `mlcanopy_type` has hundreds of fields mixing inputs, forcing, prognostic state, scratch space and diagnostics. | It is hard to see what a routine reads or writes, and every routine can touch everything. |
| Forcing | Tower and CLM-history netCDF files are re-read inside the time loop (`TowerMetCurr`, `TowerMetNext`, `clmData`). For 3-point interpolation, the state carries `*_bef/_cur/_next` copies of forcing. | File I/O inside the loop blocks compilation. Forcing is mixed into model state. |
| Time loop | The outer loop is a Python `for` with ASCII writes each step. The inner 300 s sub-step loop is a Python `for` in normal mode and a `lax.scan` in differentiable mode. | It is slow (no whole-run compilation), and two code paths can silently diverge. |
| Indexing | 1-based arrays with an unused index 0. A "patch" dimension (`np = endp + 1`) and Fortran-style filters, even though a run has one patch. | Off-by-one risk, wasted memory, and Python loops over filter indices. |
| Output | Six fixed-format ASCII files (`flux.out`, `fsun.out`, …). | Not self-describing (no names, units or time coordinate), and hard to analyze or differentiate. |
| Validation | Strong: 160+ routine-level golden tests at `rtol=1e-9`, plus end-to-end Fortran comparison (2.5% / r ≥ 0.9999) and differentiability tests. | Needs one addition: a tight end-to-end regression against the current JAX output (see Phase 0). |

The physics itself is in good shape, and the routine-level golden tests are a real asset. The work
is mostly in the "plumbing": how state, time, options and I/O flow.

---

## 3. Design principles

These come from jax-gcm and general JAX practice, adapted to a single-site land model.

1. **Pure functions.** Every physics function has the shape `f(inputs) -> outputs`. It reads no
   globals and mutates nothing.
2. **Separate the kinds of data.** Each kind gets its own container, because each is treated
   differently by JAX:

   | Container | What it holds | JAX treatment | Today's equivalent |
   |---|---|---|---|
   | `Config` | Scheme choices, time steps, layer settings | **Static.** Changing it recompiles. Not differentiable. | `MLclm_varctl` globals, namelist |
   | `Site` | Location, PFT, soil texture, canopy/tower heights, LAI/SAI shape parameters | Mostly static (it decides the layer count) | `TowerDataMod` tables |
   | `Layers` | Derived grid: number of canopy layers, top/bottom foliage layer, heights | **Static** integers and arrays computed once | `GridInfo`, `ml_vert_init`, `MLinitVerticalMod` |
   | `Params` | Tunable physical parameters (Vcmax25, g1, hydraulic conductances, emissivity, clumping, …) | **Traced and differentiable** | `MLpftcon`, `vcmaxpft_jax`, `g1_MED_jax` |
   | `Forcing` | Time series of tower meteorology plus prescribed CLM vegetation and soil data | Traced and differentiable, loaded fully **before** the run | `TowerMetMod`, `clmDataMod`, `*_forcing` fields |
   | `State` | Prognostic variables only: canopy air T/q/CO₂ profiles, leaf temperatures, leaf water potential, intercepted water, soil temperature/moisture | Traced and differentiable. It is the `lax.scan` carry. | Parts of `mlcanopy_type`, `temperature_inst`, … |
   | `Outputs` | Fluxes, profiles and diagnostics per time step | Returned from each step, stacked by `scan` | `*_canopy` diagnostic fields, the `.out` files |

3. **A model run is one compiled function.**
   `run = scan(step, State₀, Forcing)`, where `step(state, forcing_t, params) -> (state, outputs)`.
4. **Processes have a uniform interface.** Each physical process (radiation, turbulence,
   photosynthesis, …) is a function with a consistent signature, and `Config` selects which
   variant runs. This is a lighter version of jax-gcm's `PhysicsTerm` composition (see §5.4).
5. **Adapters first, rewrites later.** Put the new interface *around* the existing translated code
   first. Then move internals across one process at a time, each step verified by tests. jax-gcm
   did the same with its translated SPEEDY physics.
6. **Fortran traceability moves to docstrings.** Each refactored function keeps a
   `Fortran reference:` line in its docstring, rather than mirroring Fortran file and variable
   names.
7. **No new heavy dependencies.** Containers are standard `dataclasses` registered as JAX pytrees.
   No Flax or Equinox is needed for this model's size. `xarray` becomes a runtime dependency for
   output.

---

## 4. Proposed user-facing API (summary)

Full signatures are in [`api_sketch.py`](api_sketch.py). The everyday experience would look like
this:

```python
import clmml

# 1. Describe the site and load forcing (all file I/O happens here, once)
site = clmml.Site.from_tower("CHATS7")
forcing = clmml.load_tower_forcing(
    site,
    tower_file="input_files/tower-forcing/CHATS7/2007-05.nc",
    clm_history_file="input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc",
    soil_adjust_file="input_files/clm5_0/CHATS7/CHATS_soil_moisture_correction_29Jul25.nc",
    start="2007-05-01", days=1,
)

# 2. Choose schemes (readable names instead of integer codes)
config = clmml.Config(stomatal_conductance="wue", solar_radiation="two_stream",
                      integrator="rk4", substep_seconds=300.0)

# 3. Build the model, get parameters and the initial state
model = clmml.Model(site, config)
params = clmml.default_params(site, preset="chats")   # replaces pftcon_val = 1
state0 = model.initial_state(forcing)

# 4. Run: one compiled call; outputs are arrays with a time axis
result = model.run(state0, forcing, params)
ds = result.to_xarray()                   # named variables, units, time coordinate
ds["latent_heat_flux"].plot()
result.write_legacy_ascii("output_files/")  # identical .out files, for Fortran comparison
```

Gradient-based calibration:

```python
def loss(params):
    out = model.run(state0, forcing, params)
    return jnp.mean((out.latent_heat_flux - obs_lh) ** 2)

grads = jax.grad(loss)(params)          # gradient w.r.t. every parameter
print(grads.vcmax25, grads.g1_medlyn)
```

Ensembles and sensitivity analysis:

```python
params_batch = clmml.perturb(params, vcmax25=jnp.linspace(60, 140, 32))
lh = jax.vmap(lambda p: model.run(state0, forcing, p).latent_heat_flux)(params_batch)
```

Advanced use (custom loops, data assimilation, coupling):

```python
state, out_t = model.step(state, forcing.at_step(t), params)
```

Backward compatibility for existing workflows:

```bash
clm-ml-offline input_files/nl.CHATS7.1day          # still works, now built on the new API
```

```python
config, site, files = clmml.legacy.from_namelist("nl.CHATS7.1day")
```

---

## 5. Target architecture

### 5.1 Package layout

```
src/clmml/
  __init__.py            # public API re-exports (Model, Config, Site, Params, ...)
  model.py               # Model: initial_state, step, run
  config.py              # Config (frozen dataclass, string-valued scheme choices)
  site.py                # Site, built-in tower registry (data/towers.csv)
  layers.py              # Layers: canopy vertical discretization (from MLinitVerticalMod)
  params.py              # Params pytree, PFT defaults, presets, perturb()
  forcing.py             # Forcing pytree, load_tower_forcing, time interpolation
  state.py               # CanopyState, SoilState, State
  outputs.py             # Outputs pytree, to_xarray, write_legacy_ascii, variable metadata
  constants.py           # physical constants (from clm_varcon / MLclm_varcon)
  physics/
    __init__.py          # process registry: name -> implementation
    solar_radiation.py   # norman, two_stream
    longwave.py
    turbulence.py        # Harman & Finnigan RSL, flux-profile solution
    leaf_boundary_layer.py
    photosynthesis.py    # Farquhar + stomatal conductance variants (medlyn, ball_berry, wue)
    plant_hydraulics.py
    canopy_water.py      # interception, wetted fraction, evaporation
    soil.py              # soil fluxes, thermal properties, soil temperature
    integrators.py       # Euler / RK2 / RK3 / RK4 tables
    math.py              # tridiagonal, quadratic, beta distribution, ...
  legacy/
    namelist.py          # from_namelist(): .nl file -> (Config, Site, files)
    ascii_output.py      # bit-compatible .out writers
  cli.py                 # clm-ml-offline entry point
  data/towers.csv        # site table (moved out of TowerDataMod.py)
```

The existing packages (`clm_src_*`, `multilayer_canopy`, `offline_driver`, …) stay in place during
the transition. The new API calls into them through adapters. Each process moves into
`clmml/physics/` as it is refactored, and old modules are deleted only when nothing imports them.

### 5.2 One time step

Today, one 30-minute CLM step runs `clm_drv` followed by `MLCanopyFluxes`. The new structure keeps
the same science order, but each box is a pure function:

```
step(state, forcing_window, params)            # one 30-min tower step
 ├─ interpolate prescribed CLM veg/soil data at this step     (was clmData)
 ├─ soil_properties(state.soil, ...)                          (SoilAlbedo, calc_soilevap_resis,
 │                                                             SoilThermProp, SoilWater)
 ├─ canopy_setup(...)                                         (PAD profiles, SoilResistance,
 │                                                             PlantResistance, LeafHeatCapacity)
 ├─ scan over 6 x 300 s sub-steps:                            (one path; was two)
 │    ├─ atm_forcing = interpolate(forcing_window, t_sub)     (GetAtmForcing, met_type 3)
 │    ├─ solar_radiation[config.solar_radiation]
 │    ├─ nitrogen_profile
 │    └─ integrator[config.integrator]: for each RK stage
 │         wetted_fraction → longwave → turbulence → leaf_boundary_layer
 │         → photosynthesis(sun, shade) → flux_profile_solution
 │         → leaf_water_potential → interception/evaporation
 ├─ average sub-step fluxes                                   (_MLAccumulateFluxes/_MLScaleAndWriteBack)
 ├─ canopy diagnostics                                        (always computed now)
 └─ soil_temperature(...)
returns (new_state, outputs_t)
```

`forcing_window` contains the previous, current and next tower records. That removes the need to
carry `*_bef/_cur/_next` forcing copies in model state.

### 5.3 Static vs traced: what can change without recompiling

Changing any of these **recompiles** (they are static): `Config`, the `Site` structure, `Layers`
(number of canopy layers), and run length.

Changing any of these does **not** recompile, and all are differentiable: `Params`, `Forcing`
values, and the initial `State`.

This replaces today's `DIFFERENTIABLE_MODE` flag and `GridInfo` workaround with a single rule.

### 5.4 Process interface

Each process is a plain function with the same signature:

```python
def process(state: StepState, inputs: StepInputs, params: Params, layers: Layers,
            config: Config) -> StepState
```

- `StepState` is the working state inside a sub-step (prognostic state plus intermediate fluxes).
- Scheme variants are registered by name, for example
  `SOLAR_RADIATION = {"norman": norman, "two_stream": two_stream}`.
  `Config(solar_radiation="two_stream")` selects one.
- Each process declares in its docstring which fields it **reads** and which it **writes**. A
  test checks the declared write set against what actually changes (a light version of jax-gcm's
  `requires`/`provides` validation).

**Why not full jax-gcm-style composable terms?** jax-gcm needs to mix and match dozens of
schemes across several physics packages. This model has one fixed process order with a few
variant choices per process. A name-to-function registry gives most of the benefit with much less
machinery. If arbitrary user-supplied process lists become a goal later, the registry can be
upgraded without changing the public API.

### 5.5 Arrays and indexing

- 0-based indexing, with no unused slot 0.
- No patch dimension. A `Model` simulates one column. Multiple sites or ensembles use
  `jax.vmap`.
- Canopy profiles have shape `(n_layers,)`. Leaf arrays have shape `(n_layers, 2)`, where the
  last axis is `[sunlit, shaded]`. Spectral arrays have a trailing `(2,)` axis for `[vis, nir]`.
- Layers without vegetation are handled with a mask array (`layers.has_leaf`) instead of
  `ntop`/`nbot` loop bounds. This removes the Python loops over layers and filters.

### 5.6 Errors and checks

`endrun` calls inside traced code cannot raise errors under `jit`. They will be replaced by:

- Input validation in constructors (for example, `Config` rejects
  `substep_seconds` that does not divide 1800).
- An opt-in `model.run(..., check=True)` using `jax.experimental.checkify` for runtime checks
  such as energy-balance closure and NaN detection.
- Energy- and water-balance error diagnostics included in `Outputs` by default.

---

## 6. Migration plan

Each phase ends with all tests passing and a mergeable PR. **The model must give the same answer
at the end of every phase**, so if something breaks, we know which phase caused it.

### Phase 0: Safety net (before any refactoring)

- **Add an end-to-end regression test against the current JAX output**, at tight tolerance
  (proposed `rtol=1e-10`). Use the existing `nl.CHATS7.50steps` namelist. Commit the reference
  output as `.npz`. The Fortran comparison (2.5%) is too loose to catch refactoring mistakes. This
  test is the main guardrail for the rest of the plan.
- Fix `requires-python`. JAX ≥ 0.4.30 requires Python ≥ 3.10 (the local environment has JAX
  0.6.2). Proposed minimum: Python 3.10 (see question 5).
- Move `netCDF4` from dev to runtime dependencies (the loaders need it). Add `xarray`.
- Record a baseline benchmark: compile time and wall time per simulated day on CPU and on GPU.

**Done when:** the regression test passes on `main`, and benchmark numbers are recorded in
`docs/design/benchmarks.md`.

### Phase 1: New API as a wrapper (no physics changes)

- Create `src/clmml/` with `Config`, `Site`, `Params`, `Forcing`, `Outputs` and `Model` exactly
  as in `api_sketch.py`.
- Internally, `Model.run` translates the new objects into the existing globals, calls the existing
  driver step by step, and collects outputs into arrays instead of files.
- `Outputs.to_xarray()` attaches names, units and long names, derived from the variables written
  to the six `.out` files.
- `write_legacy_ascii()` reproduces the six files byte-for-byte.
- Rewrite the `clm-ml-offline` CLI on top of `clmml.legacy.from_namelist` and `Model.run`.
- Move the tower site table from `TowerDataMod.py` to `data/towers.csv`.

**Done when:** the CLI produces identical files, the regression test passes through the new API,
and a short tutorial notebook runs. At this point, users already get the new interface.

### Phase 2: Forcing and time become data

- `load_tower_forcing` reads all required tower and CLM-history records into one `Forcing` pytree
  before the run, including derived quantities (solar zenith angle, CO₂, emissivity).
- Replace the `clm_time_manager` globals with time arrays in `Forcing`: calendar day, seconds of
  day, and sub-step offsets.
- Remove the start-date "hack" in `CLMml_drv`. The CLM-history time index is computed once in the
  loader.
- Remove `*_bef/_cur/_next` forcing fields from the state. The step receives a three-record
  forcing window instead.

**Done when:** no file is opened during `run`, and the regression test passes.

### Phase 3: Explicit state and a pure `step`

- Define `State` (`CanopyState`, `SoilState`) with only the prognostic variables. Identify them by
  listing what `_copy_bef_state`, `RungeKuttaUpdate` and `SoilTemperature` carry across steps.
- Write `Model.step(state, forcing_window, params) -> (state, outputs)`. Internally it unpacks
  into the existing `*_inst` objects (an adapter), calls `clm_drv`'s physics sequence, and packs
  the results back. **No module globals are read or written.**
- Delete `clm_instMod` globals, `global filter`, `DIFFERENTIABLE_MODE` and the non-diff/diff
  branching in `MLCanopyFluxes`. Keep one path based on `lax.scan`. Diagnostics are computed in
  that path too.
- Move first-step initialization (`ml_vert_init`, `zref == spval` checks, RK coefficient globals)
  into `Model.__init__` and `initial_state`.
- Replace `vcmaxpft_jax`/`g1_MED_jax` threading with `Params` passed everywhere.

**Done when:** `jax.grad` works through `step` with respect to any `Params` field, the regression
test passes, and `grep -r "global "` in `src/` returns nothing in the run path.

### Phase 4: Whole-run compilation

- `Model.run` = `jax.jit(lax.scan(step, ...))` over all tower steps. Apply `jax.checkpoint`
  (rematerialization) on the step for memory-efficient gradients, keeping the current
  `CLM_ML_NO_CHECKPOINT`-style opt-out as a `Config` option.
- Add `Outputs` selection (`model.run(..., outputs=[...])`) so long runs do not store every
  profile.
- Add chunked runs for long simulations (for example, one month at a time, carrying `State`).
- Benchmark against the Phase 0 baseline. Watch compile time in particular: `main.py` notes
  ~290 s today.

**Done when:** a 1-day and a 31-day run are each one compiled call, `vmap` over a 32-member
parameter ensemble works, and benchmark results are recorded.

### Phase 5: Process interface and parameters

- Move each process from `multilayer_canopy/ML*Mod.py` into `clmml/physics/` behind the uniform
  interface (§5.4), one PR per process. Suggested order, simplest first: math utilities →
  water vapor/constants → leaf heat capacity → canopy water → longwave → solar radiation → leaf
  boundary layer → plant hydraulics → photosynthesis/stomata → turbulence/flux-profile → soil.
- Turn integer option codes into named choices (`gs_type=2` → `"wue"`). Keep a mapping table in
  `clmml.legacy` for namelists.
- Define `Params` fields with names, units, valid ranges and per-PFT defaults. Presets
  (`"default"`, `"chats"`) replace `pftcon_val`.
- Move the routine-level golden tests to call the new functions. Keep the JSON golden files
  unchanged.

**Done when:** `multilayer_canopy/` is no longer imported by the run path, all golden tests pass
against the new functions, and the regression test passes.

### Phase 6: Internal cleanup

- Convert to 0-based indexing, remove the patch dimension, and use masks instead of
  `ntop`/`nbot` loops (§5.5). Do one process at a time, behind the regression test.
- Delete unused Fortran-mirroring modules: `clm_src_cpl`, `decompMod`, `spmdMod`, `ncdio_pio`,
  `histFileMod`, `restUtilMod`, duplicated `cime_src_share_util`/`clm_share`, and the remaining
  `*Type.py` containers.
- Move data files out of `src/` (`input_files/`, `output_files/`, `rsl_lookup_tables/`). Package
  lookup tables via `importlib.resources`. Keep example data under `examples/data/`.
- Fix the duplicated `src/pyproject.toml` and `src/pytest.ini`.

**Done when:** `src/` contains only `clmml/`, and the regression and golden tests pass.

### Phase 7: Documentation and release

- A getting-started guide, a gradient-calibration example, an ensemble example, and a
  "for Fortran CLM-ml users" page mapping old names and options to the new ones.
- An API reference generated from docstrings.
- Update `CONTRIBUTING.md`: "add a process variant" replaces "mirror the Fortran module".
- Version 1.0.0. The `clm-ml-offline` CLI and namelist support remain for compatibility.

### Rough effort

| Phase | Size | Notes |
|---|---|---|
| 0 | Small | Mostly test and config work |
| 1 | Medium | Mostly new code. It touches no physics. |
| 2 | Medium | The 3-point interpolation needs care |
| 3 | Large | The hardest phase. Untangling `mlcanopy_type` state from scratch fields. |
| 4 | Small to medium | Mostly benchmarking and compile-time tuning |
| 5 | Large | Many small PRs. Can be parallelized. |
| 6 | Medium | Mechanical, but touches everything |
| 7 | Small to medium | |

Phases 0–1 give users the new interface quickly. Phases 2–4 deliver the speed and
differentiability benefits. Phases 5–7 are about long-term maintainability.

---

## 7. Decisions needed from you (please answer before Phase 0)

1. **Package/import name.** I propose `clmml` (`import clmml`) and keeping `clm-ml-jax` as the
   distribution name. Alternatives: `clm_ml`, `mlcanopy`.
2. **Option names.** Should options use readable strings (`stomatal_conductance="wue"`), as
   proposed, or keep the Fortran integer codes (`gs_type=2`)? The proposal keeps the integer codes
   available only through `clmml.legacy`.
3. **Multi-site runs.** Different sites have different numbers of canopy layers, which changes
   array shapes. Do you need to `vmap` across *sites* (requiring padding to a common layer count
   with masks)? Or is `vmap` across *parameters/ensembles* at one site enough for now? The proposal
   assumes the latter and leaves room for the former.
4. **Differentiable inputs.** Which quantities must gradients be available for: PFT parameters
   only, also physical constants, forcing, or initial state? The proposal makes `Params`, `Forcing`
   and `State` all differentiable, and keeps constants fixed.
5. **Minimum Python version.** Python 3.10 is proposed (current JAX needs it). jax-gcm requires
   3.11.
6. **xarray as a required dependency?** Proposed: yes.
7. **Legacy support lifetime.** Keep the namelist CLI and `.out` files indefinitely, or deprecate
   them after 1.0?
8. **Regression tolerance.** Is `rtol=1e-10` against current JAX output the right bar? XLA may
   reorder floating-point operations when whole-run compilation is introduced (Phase 4). That
   phase may need `1e-8`, with the Fortran comparison still passing.
9. **Scope of the soil model.** `SoilTemperature` and the CLM soil routines currently sit outside
   the canopy model. Should they be first-class processes, as proposed, or stay as a fixed
   "lower boundary"?

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactoring silently changes results | Tight regression test (Phase 0) on every PR, plus the golden tests and the Fortran comparison |
| `mlcanopy_type` mixes state and scratch fields in non-obvious ways | In Phase 3, derive the `State` field list by instrumenting one step and recording which fields are read before they are written |
| Whole-run compilation is slow to compile (already ~290 s) | Keep the persistent compilation cache. Use `scan` (compiled once per step body), not unrolled loops. Benchmark in Phase 4. Fall back to per-step `jit` if needed. |
| Gradient memory for month-long runs | `jax.checkpoint` on the step, chunked runs, and selectable outputs |
| Value-dependent Python `if`s inside physics (for example, `zref == spval`) break tracing | Move them to initialization, or replace them with `jnp.where`/`lax.cond`. Most were already handled for the diff-mode path. |
| Long-lived branch drifts from `main` | Every phase is merged on its own. No long-lived refactor branch. |
| Contributors used to the Fortran-mirroring layout | Phase 7 mapping page. Every function keeps its `Fortran reference:` docstring line. |

---

## 9. Glossary

- **Pure function:** a function whose result depends only on its arguments, and which changes
  nothing outside itself. JAX's transformations assume this.
- **Pytree:** any nested structure of containers (dataclasses, tuples, dicts) holding arrays. JAX
  can `jit`, `grad` and `vmap` through pytrees as a whole.
- **Static vs traced:** *static* values (such as the layer count or a scheme choice) are fixed at
  compile time, and changing them triggers a recompile. *Traced* values (such as parameters or
  forcing) can change freely and can be differentiated.
- **`lax.scan`:** JAX's compiled loop. It runs `step` repeatedly while carrying `state` forward,
  and stacks each step's outputs.
- **Adapter:** a thin layer that translates between the new interface and the old code. This lets
  the inside change later without changing what users see.
- **Regression test:** a test that checks that output has not changed compared to a saved
  reference.
