# How CLM-ML-JAX works

This page describes the model as it is in the code: how it is organized, what runs when,
and how the JAX version differs from the Fortran CLM-ml v2 it translates. Paths are relative
to `src/clm_ml/` unless they start with the repository's top folders (`configs/`, `docs/`,
...). The development record behind every choice is in `docs/design/` (the decision register
`decisions.md`, open items `open_items.md`, and the longer `model_overview.md`).

**Contents:** 1. What the model computes · 2. How the code is organized · 3. Skeleton: what
runs when · 4. Three kinds of value · 5. How the JAX version differs from Fortran ·
6. Gradients · 7. Known limitations · 8. Where to read more

---

## 1. What the model computes

CLM-ML-JAX is CLM-ml v2, the multilayer canopy model of Bonan et al. (2021, 2026), written in
JAX. For every **patch** (one plant functional type's canopy on a gridcell) it solves the
canopy air space from the ground to the forcing height in layers (0.5 m for canopies taller
than 2 m, 0.1 m otherwise, or explicit layer counts): air temperature, water vapour, sunlit
and shaded leaf temperature and leaf water potential in each layer, intercepted water, ground
temperature and soil temperature. The processes are two-stream shortwave and Norman longwave
radiation, Harman–Finnigan roughness-sublayer turbulence, leaf boundary layer, Farquhar
photosynthesis with water-use-efficiency stomatal optimization constrained by plant
hydraulics (Medlyn and Ball–Berry are alternatives), an implicit flux–profile solution,
interception and soil heat diffusion.

The model runs **offline**: meteorology (tower or gridded), leaf and stem area and soil
water are prescribed. As an option the soil water is predicted instead, with CLM's soil
hydrology (section 5b). Each 30-minute model step is split into six 5-minute sub-steps advanced
with 4th-order Runge–Kutta. Patches do not exchange anything with each other, so the same
code runs one tower or the whole globe; gridcell output is the area-weighted sum of its
patches.

The vocabulary used throughout: a **gridcell** has its own atmospheric forcing; a **patch** is
one PFT's canopy on a gridcell with an area weight, and is the unit of computation; a **soil
column** is the soil under a patch (one per patch); a **tower** is a named flux-tower location
in `domain.yaml` that defines a gridcell and its patches.

---

## 2. How the code is organized

### 2a. An experiment is a folder of four files

A run reads nothing but these four files and the data files they name. Every setting is
written out, with its explanation, in the files themselves (`clm-ml new` writes them from a
compset; `configs/` holds the compsets written out).

| File | Says | Becomes in Python |
|---|---|---|
| `run.yaml` | **when and with what**: dates and calendar, forcing files and their interval, land-data sources (leaf/stem area, soil water), initial conditions, output streams, chunk length, restarts, devices | `run.RunFile` |
| `domain.yaml` | **where**: towers (location, forcing height, soil, patches) or a grid (a CLM surface dataset and a region) | `domain.Domain` (gridcells, patches, soil columns) |
| `model.yaml` | **how**: the parameterization of each process (`processes`), numerics (model step, sub-step, integrator), canopy layers, soil | `config.ModelConfig` |
| `params.yaml` | **which plants**: every parameter of every PFT 1–16 | `params.Params` |

`experiment.load_experiment(folder)` reads and checks the four files (unknown, missing or
misplaced keys are errors that name the file and key) and returns an
`experiment.Experiment`. `Experiment.execute()` runs it. The command line (`cli.py`:
`clm-ml new | check | run | upgrade | convert | params`) calls the same functions.

### 2b. The source tree

The code is organized by role and, inside `processes/`, by physical or biological process
(not by the Fortran module it came from).

```
src/clm_ml/
  experiment.py     load an experiment folder, check it, run it (Experiment.execute)
  run.py domain.py config.py params.py grid.py
                    the four files' contents as typed Python objects
  io/yaml_schema.py every setting declared once (explanation, default, Fortran name);
                    the YAML files are generated from these declarations
  compsets/          the compsets `clm-ml new` copies (towers, PFT parameter sets, experiments)

  setup/            before the run, once (NumPy): each patch's layer grid, leaf/stem area
                    fractions, soil column, roots, and the initial state
                    (towers.py, grid.py, canopy.py, soil.py, patches.py)

  driver.py         the run: Simulation (everything set up), the compiled chunk
                    (make_runner), the loop over chunks on the host (run_streams)
  step.py           one model step of one patch (step), and its inputs
                    (MetInputs, LandInputs, PatchConstants)
  state.py          State (what is carried from step to step), Diagnostics (what a step
                    reports), PatchLayers (the layer grid and its masks)
  processes/        the science, as pure functions of arrays:
    canopy.py         one sub-step (substep), its Runge-Kutta stages, sums and totals
    atm_forcing.py    forcing within the step (3-point time interpolation, minimum wind)
    solar_radiation.py longwave.py  radiation
    turbulence.py     Harman-Finnigan roughness sublayer (with the psihat tables)
    leaf_boundary_layer.py photosynthesis.py plant_hydraulics.py nitrogen_profile.py
    flux_profile.py   the implicit solution for air T, vapour, leaf T, ground T
    canopy_water.py   interception, wetted fraction of the leaves, evaporation
    soil.py           soil water from the prescribed record, soil properties, soil temperature,
                      freezing and thawing of soil water (CLM's)
    soil_water.py surface_runoff.py drainage.py hydrology.py
                      predicted soil water (CLM's soil hydrology): the Richards equation,
                      runoff and ponded water, water table and drainage, the frost table
                      and the perched water table, the assembled step
    integrators.py    Euler and Runge-Kutta tableaus and the state update
    constants.py orbit.py math.py layers.py water_vapor.py leaf_heat_capacity.py

  tower_forcing.py grid_forcing.py forcing.py forcing_interpolation.py chunk_inputs.py
                    reading forcing and land data a chunk at a time, on the host
  output_streams.py outputs.py   output variables, accumulation, netCDF files
  io/               readers (tower, CLM history), restart files, the legacy .out writer,
                    the namelist converter
  checks.py         the conservation checks (recorded in the step, tested after each chunk)
  model.py          Model: compiled chunks and gradients for calibration
  data/psihat.nc    the roughness-sublayer lookup table
```

---

## 3. Skeleton: what runs when

A run has two parts. The **host** (ordinary Python and NumPy on the CPU) sets things up,
reads forcing, checks results and writes files. The **compiled chunk** (one program that
JAX traces once and compiles with XLA for the CPU or GPU) advances all patches over many
steps. The loops nest like this; the left column says where each runs.

```
HOST      Experiment.execute()                                   experiment.py
HOST      ├─ load the four files, check them                     experiment.load_experiment
HOST      ├─ set up every patch once                             driver.simulation -> setup/
HOST      │    layer grid (heights, masks), leaf/stem area fractions, soil column,
HOST      │    roots -> PatchConstants;  initial State (from the CLM history or a restart)
HOST      │    patches ordered into C3 and C4 groups, padded, split over devices
HOST      └─ run_streams: for each chunk (default 10 days)       driver.run_streams
HOST           ├─ forcing for the next chunk read on a background thread
HOST           │    tower files or CRUJRA streams -> MetInputs (per gridcell, per step)
HOST           │    CLM history or daily/monthly files -> LandInputs (per patch, per step)
COMPILED       ├─ run_chunk = jit(scan over the chunk's steps)    driver.make_runner
COMPILED       │   for each model step (30 min):
COMPILED       │   ├─ meteorology gathered from gridcells to patches
COMPILED       │   ├─ vmap over patches (each C3/C4 group):       driver.patch_step
COMPILED       │   │   step(state, inputs, constants, params, tables)   step.step
COMPILED       │   │   ├─ acclimation temperature: 10-day running mean
COMPILED       │   │   ├─ soil water from the prescribed record; soil albedo,
COMPILED       │   │   │  evaporative resistance, thermal and hydraulic properties
COMPILED       │   │   ├─ canopy forcing, solar zenith angle, plant area per layer
COMPILED       │   │   ├─ soil and plant resistance, leaf heat capacity
COMPILED       │   │   ├─ scan over 6 sub-steps (5 min):          processes/canopy.substep
COMPILED       │   │   │   ├─ forcing interpolated in time (previous, current, next record)
COMPILED       │   │   │   ├─ two-stream solar radiation, Vcmax25 profile
COMPILED       │   │   │   ├─ scan over 4 Runge-Kutta stages:     canopy._stage
COMPILED       │   │   │   │   Norman longwave -> net radiation
COMPILED       │   │   │   │   -> RSL turbulence (wind, u*, conductances, Obukhov length)
COMPILED       │   │   │   │   -> leaf boundary layer -> interception, wetted fraction
COMPILED       │   │   │   │   -> photosynthesis and stomata
COMPILED       │   │   │   │   -> implicit flux-profile solution (air T, vapour, leaf T,
COMPILED       │   │   │   │      ground T) -> leaf water potential -> leaf evaporation
COMPILED       │   │   │   │   -> Runge-Kutta update of the canopy state
COMPILED       │   │   │   └─ fluxes added to the step's sums
COMPILED       │   │   ├─ averages over the sub-steps, canopy totals, energy checks
COMPILED       │   │   ├─ sunlit/shaded leaves merged to the layer mean
COMPILED       │   │   ├─ soil temperature (heat diffusion); with soil.freeze_thaw the
COMPILED       │   │   │  freezing and thawing of soil water and the melt of snow
COMPILED       │   │   └─ with predicted soil water: runoff, infiltration, the Richards
COMPILED       │   │      equation, water table, drainage      processes/hydrology
COMPILED       │   │   -> new State + Diagnostics
COMPILED       │   ├─ every output stream accumulated into its record
COMPILED       │   │  (memory grows with output records, not with steps)
COMPILED       │   └─ gridcell values: area-weighted sums of the patches
HOST           ├─ conservation and health checks (stop, or exclude failing patches)
HOST           ├─ completed output records -> netCDF files (one per stream, level, period)
HOST           └─ restart file at the chosen boundaries and at the end
```

What is carried from one step to the next (`state.State`, the restart file's content):

| Field | Per patch | Units |
|---|---|---|
| `canopy.air_temperature`, `canopy.vapor_pressure`, `canopy.co2` | each layer | K, Pa, µmol mol⁻¹ |
| `canopy.leaf_temperature`, `canopy.leaf_water_potential` | each layer, sunlit/shaded | K, MPa |
| `canopy.intercepted_water` | each layer | kg m⁻² |
| `canopy.ground_temperature` | one value | K |
| `canopy.acclimation_temperature`, `canopy.acclimation_steps` | one value | K, steps |
| `soil.temperature` | 25 soil levels | K |
| `soil.liquid_water`, `soil.surface_water`, `soil.water_table_depth` | 20 soil levels, one value, one value | kg m⁻², mm, m |

Everything else a step computes is in `state.Diagnostics` (about 95 fields with the
CLM-ml names), which the output variables (`outputs.VARIABLES`, 78 named variables) are
computed from.

**Layers and masks.** Every patch has the same number of layer slots (`max_layers`, 50
by default). A patch with 46 layers (CHATS7) uses the bottom 46; the slots above are padding,
filled with the top layer's values while computing and masked out of the results (NaN in the
output files). Every patch therefore has arrays of the same shape, which is what lets one
compiled program run every patch at once, whatever its canopy height.

**Chunks.** A run is a loop over chunks of fixed length (default 10 days; a short last chunk
is padded with inactive steps). The chunk length is host bookkeeping: how much forcing is in
memory, how often the checks run, when files are written. It does not change results. Memory
does not grow with run length, so a 150-year global run needs no more memory than a 10-day
one.

---

## 4. Three kinds of value

JAX compiles the model into a program before running it. That splits every input into one of
three kinds, and the split decides what you can change without recompiling and what you can
take gradients with respect to.

- **Static: chooses the code or the array sizes.** Everything in `model.yaml` (schemes,
  model step, sub-step, integrator, `max_layers`), the C3/C4 pathway of each group of patches,
  the number of steps in a chunk, the output stream definitions. Changing one compiles a new
  program (about 20–60 s on the CPU; compiled programs are cached on disk, so a repeated
  setting starts quickly). Static values cannot have gradients.
- **Data: any value, same program.** Forcing, leaf and stem area, soil water, the per-patch
  constants (layer grid, soil column, roots, location), patch weights, the psihat tables,
  the initial state. Another tower, soil, canopy height or date runs through the program
  already compiled.
- **Differentiable: data that `jax.grad` can follow.** The PFT parameters of `params.yaml`
  (`vcmax25`, `iota`, `psi50_gs`, optics, ...), the state, the forcing and the patch weights.
  A few PFT parameters act only at setup (root profile, default canopy height and area
  profiles) and so have no gradient through a run.

---

## 5. How the JAX version differs from Fortran

The JAX model computes what CLM-ml v2 computes. The Fortran results are reproduced routine
by routine (160+ tests at a relative tolerance of 1e-9) and end to end: CHATS7, May 2007,
with every output column within 2.5 % of Fortran and correlation r ≥ 0.9999 (largest
relative difference 0.07 %, `tests/test_fortran_python_comparison.py`).
What differs is mostly how the code is written. The few science departures are listed in 5b.

### 5a. How the code is written

| Fortran CLM-ml v2 | CLM-ML-JAX | Why |
|---|---|---|
| Module variables and derived types (`mlcanopy_inst`, `pftcon`, ...) read and written anywhere | Pure functions: everything a function uses is an argument, everything it changes is returned. The state is an immutable `State` | JAX can only compile and differentiate functions without hidden inputs; it also makes each process testable on its own |
| Loops over patches (`do fp = 1, num_filter`) and over layers (`do ic = 1, ncan`) | The step is written for **one patch** and `jax.vmap` runs it over all patches at once; layers are whole arrays with masks | One program for one tower or 70 000 patches, on CPU or GPU |
| Each patch has its own number of layers (`ncan`) | Fixed `max_layers` slots per patch, with masks | Arrays of equal shape are needed to batch patches; padding never changes the results |
| Time loop in the driver, one step at a time | `jax.lax.scan` over the steps of a chunk, compiled once; sub-steps and Runge–Kutta stages are scans too | The compiled program is reused for every chunk; scans can be differentiated |
| Iterative solvers stop at a tolerance (Obukhov length: hybrid secant/Brent; stomatal conductance: bisection; intercellular CO₂: hybrid) | Fixed numbers of iterations that stop updating when converged; gradients by the implicit function theorem, which does not change the forward value | Compiled code needs a known number of iterations; differentiating through every iteration gives wrong or exploding gradients. Within the validation tolerance |
| 1-based indexing (layers `1..ncan`, bands `1..2`, leaves `isun=1, isha=2`) | 0-based everywhere (layer 0 at the ground; band 0 visible, 1 near-infrared; leaf 0 sunlit, 1 shaded) | Python convention; only the legacy `.out` writer rebuilds the 1-based layout |
| Namelist (`nl.*`) with integer option codes | Four YAML files with named choices (`stomatal_conductance: wue`); `clm-ml convert` translates a namelist | Readable, complete and checked: a typo is an error, not a silent default |
| `.out` text files, every step | Output streams: chosen variables, frequency (step to yearly), statistic (mean, instantaneous, min, max), patch or gridcell level (a grid run's gridcell files on the latitude-longitude grid), one netCDF file per run, year or month; `output.legacy_ascii` still writes the `.out` files for one patch | Configurable output whose memory does not grow with run length |
| Conservation checks stop the run at the failing step (`endrun`) | The step records each check's largest error; the host tests them after each chunk and stops, or leaves the failing patch out (`on_patch_error`) | A host call inside compiled code pauses the device every step and prevents caching. Fortran's 15 checks of the implicit flux-profile solution are not run |
| Forcing read record by record from whole-run arrays | Forcing streams read a chunk at a time (with one record either side for the time interpolation), on a background thread | Global, multi-decade runs need memory independent of run length |
| One tower per executable | Any number of towers or a grid in one run; patches grouped by C3/C4 pathway; split over several GPUs with `jax.shard_map` | Spatial scale from one tower to global |
| Double precision | float64 | Same precision as Fortran |

### 5b. Science departures (all recorded in `docs/design/decisions.md`)

- **Acclimation temperature**: a 10-day running mean of air temperature,
  carried in the state. Fortran uses the mean over the whole run, which needs all of the
  forcing before the run starts. `temperature_acclimation: run_mean` keeps Fortran's choice
  for validation (the May 2007 comparison uses it).
- **Minimum plant area in every leafy layer**: each leafy layer keeps at least
  0.01 m² m⁻², the missing part added as stem area. Without it, layers with almost no leaf
  area make the longwave solution unstable (diffuse transmittance above 1). Setting it to 0
  reproduces CLM-ml v2. CHATS7 is unchanged.
- **Canopy-top latent heat**: averaged per sub-step, so the reported value is
  right when the air crosses 0 °C. This changes a diagnostic only.
- **Wetted fraction of the leaves**: taken from the intercepted water left at
  the end of each 5-minute sub-step, so leaves cannot evaporate more water than they hold.
  Fortran takes it from the water at the start of each Runge–Kutta stage: almost dry leaves
  (after rain, in drizzle, with dew) then alternate between wet and dry, their water turns
  negative and latent heat flickers by several W m⁻². `numerics.wetted_fraction: explicit`
  keeps Fortran's choice (the May 2007 comparison uses it). Dry days are unchanged.
- **A stable limit on zeta for short canopies**: the PFT parameter
  `zeta_max_stable` limits how stable the surface layer can become, as CLM's `zetamaxstable`:
  0.5 for shrubs, grasses and crops, no limit (1e30) for trees. Without it the turbulence of
  short canopies collapses on calm nights (friction velocity near zero) and switches on and
  off within a time step. It raises their night-time friction velocity and warms their
  canopy air at night. Trees, and so CHATS7, are unchanged; 1e30 for every PFT reproduces
  CLM-ml v2.
- **No water stress at or above zero leaf water potential**: with dew, leaf
  water potential rises slightly above zero; the stress factor is then 1. Values are
  unchanged for the default parameters.
- **Configurable model step** with forcing interpolated to it, and **daily or monthly leaf
  area, stem area and soil water** files. The default (30-minute step, CLM history input) is
  the Fortran setup.
- **Predicted soil water** (`model.yaml` `soil.water: predicted`; the default
  `prescribed` is CLM-ml v2). The soil water is carried from step to step and moved by
  CLM's soil hydrology (CTSM 5.4), translated routine by routine and checked against CLM's
  Fortran: runoff from the saturated fraction of the surface, infiltration, ponded surface
  water, the Richards equation with CLM's adaptive sub-steps, the water table, and
  drainage. The canopy model gives it the rain reaching the ground (throughfall and
  drip), the evaporation or dew at the ground and the transpiration. With predicted
  soil water the canopy sees the soil levels weighted by their soil-to-stem conductance
  (the water potential of the roots, as in CLM's plant hydraulics), not by what each
  level can still supply as in CLM-ml v2: dry levels with many roots then count, and the
  stomata close in a drought. Each level gives its share of the transpiration, and the
  roots also move water from wetter to drier levels (`soil.root_redistribution`, on by
  default; off: no level receives water from the roots). Two more choices, both CLM-ml's
  by default: `soil.root_conductance` (`clm_ml`: CLM-ml's soil and root resistances;
  `clm`: CLM's, in which a level's roots conduct less the deeper the level and the drier
  the root, and the top level gives no water) and `soil.root_biomass` (`prescribed`: the
  patch's fine root biomass; `from_leaf_area`: CLM's, leaf area index / `sla_top` ×
  `root_to_leaf_ratio` × 2). CLM's roots use four parameters of `params.yaml`
  (`root_to_leaf_ratio`, `root_conductance_max`, `root_psi50`, `root_ck`; `clm-ml
  upgrade` adds them to older files at CLM6's values). It starts from the soil water a
  prescribed run has at its first step, or from a restart. What it does not have: a
  snowpack, irrigation, and any exchange between soil columns; soil water freezes only
  with the next option. Its results are not CLM's soil moisture: the canopy
  that dries the soil is a different one. It needs two more inputs per tower (see
  [Your own tower](your_own_tower.md)); grids read them from the surface data.
- **Soil water that freezes and thaws** (`model.yaml` `soil.freeze_thaw: true`, with
  predicted soil water; off by default). CLM's treatment (CTSM 5.4), translated and
  checked against CLM's Fortran as the soil water was. After the soil temperature of a
  step is solved, a level above 0 °C that holds ice melts and a level below 0 °C freezes;
  while its water changes phase the level stays at 0 °C, and the heat this takes or gives
  is counted in the soil's energy balance. Some water stays liquid below 0 °C (more in
  clay than in sand, less the colder), as in CLM. Ice slows the movement of water
  (infiltration, flow between levels, drainage), so rain on frozen ground ponds and runs
  off. Where thawed soil lies over frozen soil the water above can saturate: CLM's frost
  table and perched water table are computed, the perched zone drains
  (`perched_baseflow_scalar` in the `hydrology` section of `params.yaml`), and the
  saturated fraction of the surface follows the perched water table. Vapour is taken
  from the top level's liquid water and ice in their proportion, and dew on frozen ground
  is frost. The run starts from the ice of the CLM history (`SOILLIQ`, `SOILICE`) where
  the file has them; otherwise the water is divided by the soil temperature. Where this
  differs from CLM:
  - *There is no snowpack.* Snow that reaches the ground lies there as a store of water
    (`snow_on_ground`) until the top soil level's heat melts it, as CLM treats snow too
    thin for a snow layer. It has no albedo and does not insulate, so the soil in winter
    is colder and freezes deeper than under snow, and the surface is darker.
  - *The canopy is CLM-ml's.* The ground's latent heat flux uses the heat of vaporization
    also when ice sublimes, and the humidity of the soil pores at the surface is computed
    from the liquid water alone (CLM: liquid and ice), which makes a frozen surface drier.
  - *Roots.* CLM-ml's roots (the default) take no water from a level with any ice (CLM-ml
    v2's rule). CLM's roots (`soil.root_conductance: clm`) keep such a level, as CLM
    does: its ice lowers its conductivity and its liquid water is held at a very negative
    potential.
  - *The pond does not freeze* (it has no temperature of its own).
  Towers: by default all precipitation is rain, at any temperature (CLM-ml v2); set
  `run.yaml` `forcing.meteorology.snowfall: by_temperature` for CLM's division (snow at
  0 °C and below, rain at 2 °C and above). Grids always divide it.
  Two things to know. CLM's freezing overshoots (a freezing level takes the water that
  may stay liquid at its temperature before the change, then warms): while a fine soil
  freezes, results are sensitive to differences in the last digit (two runs that start
  1e-12 apart can differ by hundredths of a degree for a while), and exact derivatives
  grow without bound. Derivatives therefore use a stable approximation by default
  (`numerics.phase_change_derivative: stabilized`); forward results are not affected.
- **Runs across a new year**: the time of a step's forcing record is the middle of the
  step. Fortran takes it as the mean of the calendar days at the step's start and end,
  which is half a year off in the step that ends at midnight of 1 January (the calendar
  day starts again at 1); here that step's end is its start plus a step. Every other step
  is unchanged. Fortran's runs do not cross a year.
- **Not available** (refused with an error): Norman shortwave
  radiation, the well-mixed canopy air, profiles read from a dataset, and the Brent solver
  for stomatal conductance.

The model limitations of CLM-ml v2 itself (too-warm near-ground air in strongly stable
conditions, daytime latent heat biased high, prescribed soil moisture by default) are
unchanged; see
`open_items.md` §7.

---

## 6. Gradients

`Experiment.execute()` runs the model and writes files. For calibration and sensitivity
studies, `clm_ml.Model` exposes the pieces and one differentiable compiled chunk:

```python
import jax
import clm_ml
from clm_ml.output_streams import StreamSpec

model = clm_ml.Model.from_experiment(clm_ml.load_experiment("configs/chats7_1day"))
state = model.initial_state()            # after the first step
params = model.default_params()          # one row per PFT of the domain
first, n = model.chunks()[0]
forcing = model.forcing(first, n)
daily = StreamSpec("daily", "daily", "mean", ("latent_heat_flux",))

def loss(p):
    _, records = model.run_chunk(state, forcing, first, p, streams=[daily])
    return records["daily"]["latent_heat_flux"].mean()

grads = jax.grad(loss)(params)           # d(loss)/d(every PFT parameter)
```

That is **reverse mode**: one number (here the mean latent heat flux) differentiated with
respect to every parameter at once. **Forward mode** answers the opposite question: how every
output, at every step, changes along one chosen direction (here a 1 % change of Vcmax25),
in one run:

```python
import jax.numpy as jnp

halfhourly = StreamSpec("hh", "step", "mean", ("latent_heat_flux", "gpp"))

def outputs(p):
    _, records = model.run_chunk(state, forcing, first, p, streams=[halfhourly])
    return records["hh"]

direction = jax.tree.map(jnp.zeros_like, params)._replace(vcmax25=0.01 * params.vcmax25)
values, sensitivities = jax.jvp(outputs, (params,), (direction,))
# sensitivities["latent_heat_flux"]: change of each half-hour's latent heat for +1 % Vcmax25
```

Use reverse mode for calibration (one loss, many parameters) and forward mode for time series
of sensitivities to a few parameters or forcing directions (`jax.jacfwd` stacks several).

`run_chunk` is differentiable, in reverse and forward mode, with respect to the state, the
forcing, the PFT parameters and the patch weights. Gradients through iterative solvers come
from the implicit function theorem, so they never change the forward values.

**Gradients over many days: two derivative rules.** Two quantities are taken from the start of
each Runge-Kutta stage and the fluxes react strongly to them: the leaf water potential the
water stress of the stomata reads, and the Obukhov length (solved from the canopy-top air).
Near the stress threshold, and in stability transitions, the model alternates from stage to
stage, and its exact derivative grows without bound over days (a true slope, but useless as
a sensitivity). Two settings in `model.yaml` replace those two derivatives with bounded ones;
the forward results are the same, bit for bit, either way:

| Setting (`numerics`) | Default | What it does | Cost |
|---|---|---|---|
| `water_stress_derivative` | `stabilized` | The derivative of the leaf water potential consistent with the transpiration it gives (`exact`: the exact derivative of the forward values) | none |
| `obukhov_derivative` | `exact` | `stabilized`: where the stability loop overshoots, the derivative of the Obukhov length consistent with the air the stage gives | the gradient takes 1.7 times as long, with half as much memory again |

Set `obukhov_derivative: stabilized` for sensitivities over days where stability
flips often (measured: the wet-season cerrado); a tower like CHATS7 does not need it. With
these rules a gradient is the slope of a smoother model than the one that runs: 30-day mean
sensitivities were within about 7 % (Great Plains, July) and 12 % (cerrado, January) of the
response to a 1 % change of a parameter, where the exact derivative was off by factors of
thousands; a finite difference at a single affected step disagrees by design. Check a new
case against finite differences of time means. Tested for one parameter (`iota`) and latent
heat, in two 1° boxes, one month each.

---

## 7. Known limitations

What the model does not do, or does differently from what a user might expect:

- **Model limitations of CLM-ml v2** (science questions, unchanged): near-ground air several
  degrees too warm in strongly stable conditions (first-order closure cannot represent
  non-local mixing in the canopy); daytime latent heat biased high at CHATS (about
  70 W m⁻²); soil heat flux biased high; soil moisture prescribed by default.
- **No snowpack, and frozen soil only as an option:** there is no snowpack on the ground
  (no snow albedo, no insulation of the soil). By default all soil water is liquid, and
  with predicted soil water snowfall reaching the ground enters the soil as liquid water
  at once. With `soil.freeze_thaw` soil water freezes and thaws as in CLM and snow lies on
  the ground until it melts; without the snowpack, winter soil is colder than it should
  be. Tower forcing has no snowfall unless `forcing.meteorology.snowfall` is set.
- **No bare ground:** every patch has a canopy; a patch with almost no leaves is a sparse
  canopy, not bare soil.
- **Solvers** run a fixed number of iterations instead of Fortran's tolerances, and Fortran's
  checks of the implicit flux-profile solution are not run; results agree with Fortran within
  the validation tolerance.
- **Transpiration and canopy-evaporation latent heat** outputs use the last sub-step's latent
  heat of vaporization; on steps where the air crosses 0 °C they are slightly off.
- **Not differentiable:** the nitrogen-profile coefficient (`nitrogen_profile`, a model.yaml
  setting) and the PFT parameters used only at setup (root profile, C3/C4 pathway, default
  canopy heights and leaf/stem profiles) give zero gradient through `Model`.
- **Gradients** over many days of water-stressed weather use derivative rules that are not
  the exact derivative of the forward values (section 6).
- **Grid runs:** orbital parameters are those of the start year; real 1° patches amplify
  round-off (a 1e-15 change in leaf area can move latent heat by several W m⁻² in some
  patches after sunrise); a patch that fails a check still counts in its chunk's gridcell
  means.
- **Output caveats:** layers above a patch's top layer are NaN; `soil_water_stress` is 1
  with the WUE stomatal scheme (which uses leaf water potential instead);
  `stem_area_added` is the stem area added to keep every leafy layer above the minimum plant
  area.
- `Model.chunks()` starts at step 1 (the setup runs step 0).

The open issues are tracked in
[`docs/design/open_items.md`](repo:docs/design/open_items.md).

## 8. Where to read more

- `README.md`: install and run CHATS7.
- `examples/run_tower.py`, `examples/run_global.py`: a tower run and a global run from Python.
- `configs/README.md`: the ready-made experiment folders.
- `docs/design/model_overview.md`: the same material in more detail, with every input and
  state field, and diagrams of the data flow and of one step.
- `docs/design/decisions.md`: every design and science decision, numbered.
- `docs/design/open_items.md`: open issues, known departures and limitations.
- `docs/design/numerics_and_regression.md`: solvers, round-off and the regression tests.
- `CONTRIBUTING.md`: adding a process or scheme, and the tests every change must pass.
