# CLAUDE.md

Guidance for Claude (and other contributors) working in this repository.

## Working with the maintainer

- The maintainer is a **land-surface scientist, not a software engineer**. Explain software
  concepts (pytrees, static vs traced, adapters, etc.) in plain language, with the scientific
  reason they matter. Don't assume familiarity with software-engineering jargon.
- **Think before coding.** State assumptions. If there are several reasonable interpretations,
  present them instead of picking one silently. Push back when something conflicts with the
  science or with an earlier decision (for example, a setting that would reject a supported tower).
- Decisions made with the maintainer are recorded in `docs/design/modernization_plan.md` §0 and §8.
  Don't re-open them without a new reason.
- **Standing preferences** (from the working sessions):
  - Commit and push when the tests required by the testing tiers pass (authorized); never
    open PRs; one branch per phase (`phase7/restructure` now). Commit footer:
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the model in use).
  - Don't modify the maintainer's Python environment (no `pip install` without asking).
  - Stop before starting a new phase to discuss it; during a phase, a measured slowdown of
    more than ~1.5× (timing test or gradient) is discussed before going on.
  - Explain plainly, with the scientific reason; keep reports short.
  - Lint with `black --check --line-length 100 src tests timing && ruff check src tests timing`
    (dev extras). If `pytest-cov` is missing, run pytest with `-p no:cacheprovider -o addopts=""`.
- **Bugs and science (maintainer, Phase 6b):** diagnose first with probes in scratch, *no code
  changes*, and bring the diagnosis to the maintainer before changing code; no shortcuts,
  hacks or dropping patches/things that fail; the model is a scientific tool (accurate and
  precise). Where a modelling choice comes up, copy CLM (CTSM 5.4 source in
  `/glade/work/linnia/ctsm5.4.004`; CLM-ml v1/v2 Fortran sources: see the progress log).
  Science changes are the maintainer's decision; record each as a decision (plan §0) and
  every open issue in `docs/design/global_input.md`.
- **Resume here:** `docs/design/phase7_progress.md`, section "Next" (one session per work
  package; end each session at a commit with "Next" updated; stop and report at the end of
  each package).
- **Phase 7 documents:** scope and the maintainer's decisions D1-D12
  (`docs/design/phase7_scope.md`); **log every decision you make** that the scope does not
  settle in `docs/design/phase7_decisions.md` (C1, C2, ...) in the same commit; removed state
  fields in `docs/design/phase7_removed_diagnostics.md`.
- **Phase 7 testing is the minimum in `phase7_progress.md`** ("Testing in Phase 7"; replaces
  the tiers under "Rules for changes" for Phase 7): per commit lint + bit-for-bit/noise twin +
  the moved routines' golden tests; CPU gradient test when a hand-written gradient moves;
  gradient tier and touched feature tests at package ends; everything else at phase end.
- **Where the work is:** NCAR Derecho (CPU login node; GPUs through PBS, account UCLB0069,
  develop queue `gpudev`). Phase 6 history: `docs/design/phase6_progress.md`.
- Terminology: say **"timing test"** for wall-clock performance measurements, not "benchmark".
  (In the papers, "benchmark" means the reference multilayer simulation; keep that meaning.)

## What this model is

**CLM-ML-JAX** is a JAX translation of **CLM-ml v2**, the multilayer canopy model developed for
the Community Land Model (CLM) by Gordon Bonan (NSF NCAR). The Fortran reference is
[gbonan/CLM-ml_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS). The JAX version adds GPU
execution, automatic differentiation (for calibration and sensitivity analysis), and `vmap`
ensembles, while reproducing the Fortran results.

It runs **offline at flux towers**: prescribed 30-min meteorology drives the canopy, with leaf
and stem area, soil moisture and initial soil temperature read from a CLM history file.

### Scientific background (key papers)

1. **Bonan, Patton, Finnigan, Baldocchi & Harman (2021)**, *Moving beyond the incorrect but
   useful paradigm: reevaluating big-leaf and multilayer plant canopies to model
   biosphere-atmosphere fluxes – a review*. Agric. For. Meteorol. 306, 108435.
   doi:10.1016/j.agrformet.2021.108435. Introduces **CLM-ml v1**, generalized to any number of
   layers (1 to N), and compares one-layer (1L) with multilayer (ML) canopies at seven AmeriFlux
   forest sites (56 site-years, e.g. US-UMB with 42 layers, US-Me2 with 28).
   - ML reduces biases relative to 1L, especially in friction velocity, sensible heat and
     nighttime fluxes. The main reasons are the vertical profiles of air temperature, humidity
     and wind (the "well-mixed assumption" tests), and of leaf water potential (upper-canopy
     water stress).
   - **5–10 within-canopy layers reproduce the benchmark ML canopy**, with little improvement
     beyond 10 layers, for the dense forests tested. The benchmark uses 50 cm resolution
     (28–54 within-canopy layers).
2. **Bonan, Burns & Patton (2026)**, *Beyond surface fluxes: Observational and computational
   needs of multilayer canopy models – A walnut orchard test case*. Agric. For. Meteorol. 378,
   110960. doi:10.1016/j.agrformet.2025.110960. Data: github.com/gbonan,
   doi:10.5281/zenodo.17426258. Introduces **CLM-ml v2** and the **CHATS** test case. **This is
   the configuration this repo reproduces.**

Earlier model papers: Bonan et al. (2014) for leaf gas exchange, stomatal optimization and plant
hydraulics; Bonan et al. (2018) for roughness-sublayer turbulence and the flux–profile solution,
using Harman & Finnigan (2007, 2008).

### Model physics (CLM-ml v2)

- **Vertical structure.** Canopy air space from the ground to the forcing (reference) height,
  divided into layers of `dz_tall = 0.5 m` (canopies over 2 m) or `dz_short = 0.1 m`. Leaf and
  stem area profiles follow beta distributions. There are separate sunlit and shaded leaves in
  each layer.
- **Flux–profile solution.** Implicit coupled tridiagonal equations for air temperature, water
  vapor and leaf temperature at every layer, plus ground temperature. This avoids CLM5's
  iterative solver and its ad-hoc "fixers".
- **Turbulence.** First-order closure with the **Harman & Finnigan roughness sublayer (RSL)**:
  an exponential wind profile within the canopy, and RSL-modified similarity functions above it
  (ψ̂ from the lookup table `rsl_lookup_tables/psihat.nc`).
  - β = u*/U(h) uses the Harman (2012) form, Eq. 2 of the 2026 paper, with CHATS fit a₁=0.89,
    a₂=−0.07, a₃=2.19, β_N=0.35.
  - Applicability bounds −2 ≤ L_c/L ≤ 1 (L_c = (c_d a)⁻¹, canopy length scale).
  - Log-profile aerodynamic conductance in the lowest layer.
- **Leaf processes.**
  - Leaf energy balance, including a heat-storage term.
  - Farquhar photosynthesis with an exponential leaf nitrogen (Vcmax) profile.
  - **Water-use-efficiency stomatal optimization** (parameter ι), constrained by **plant
    hydraulics** (SPA-style). Stomata close as leaf water potential ψ_l falls, through
    f(ψ_l) = [1 + (ψ_l/ψ₅₀)^α]⁻¹. v2 solves for stomatal conductance by bisection.
  - Medlyn and Ball–Berry stomatal conductance are alternatives.
  - Leaf boundary-layer conductance has a minimum of 0.2 mol m⁻² s⁻¹; there is no minimum wind
    speed.
- **Radiation.** Two-stream shortwave (Norman as an alternative), integrated over each layer,
  and Norman longwave.
- **Time stepping.**
  - The 30-min forcing step is split into six **5-min sub-steps** (`dtime_ml = 300 s`).
  - Forcing is **interpolated with three-point time interpolation** (previous, current and next
    records; `met_type = 3`), centered in each interval.
  - State advances with **4th-order Runge–Kutta** (`runge_kutta_type = 41`); v1 used Euler.
  - The variables advanced are air temperature, vapor pressure, leaf temperature (sunlit and
    shaded), leaf water potential, intercepted water and ground temperature.
  - Output is averaged over the sub-steps to 30 min.
  - Forcing wind speed has a minimum of 0.5 m s⁻¹, and aerodynamic resistance within a layer is
    capped at 500 s m⁻¹.
- **Soil.** Soil moisture is a **prescribed boundary condition** (from CLM, adjusted to
  observations). Soil temperature is computed with a heat-diffusion model specific to the
  multilayer canopy, which differs from standard CLM.

### The CHATS test case (CHATS7, May 2007)

- Walnut orchard (*Juglans regia* 'Chandler'), Dixon, California (38.49°N, 121.84°W), part of
  the Canopy Horizontal Array Turbulence Study (CHATS).
- Canopy height 10 m. Forcing at 23 m (radiation at 16 m). LAI 2.04, SAI 0.74. Beta parameters
  are (2.6, 1.3) for leaves and (1.8, 1.3) for stems. Drag coefficient c_d = 0.25, giving
  L_c ≈ 14.8 m.
- Soil: silty clay loam (10% sand, 35% clay), soil color class 15, depth to bedrock 2 m.
  CO₂ is 383 ppm. PFT 7 (`broadleaf_deciduous_temperate_tree`).
- **With the default spacing, CHATS7 has 46 canopy-air layers**: 20 within the canopy and 26
  above it.
- **CHATS parameter overrides** (`pftcon_val = 1`; Table 2 of the 2026 paper; in code in
  `MLpftconMod.py` and `pftconMod.py`):

  | Parameter | Default | CHATS |
  |---|---|---|
  | Vcmax25 | 57.7 | 125 |
  | ι | 750 | 375 |
  | ψ₅₀ | −2.3 | −1.6 |
  | α | 40 | 40 |
  | k_p | 4 | 7 |
  | R_r* | 25 | 14 |
  | Leaf reflectance (vis, nir) | 0.10, 0.45 | 0.06, 0.42 |
  | Leaf transmittance (vis, nir) | 0.05, 0.25 | 0.04, 0.43 |
  | Stem reflectance (vis, nir) | 0.16, 0.39 | 0.21, 0.49 |
  | χ_l (leaf angle) | 0.25 | 0.53 |

- **Known model limitations** (don't "fix" these without discussion; they are science
  questions):
  - Biases in strongly stable conditions: near-ground air several degrees too warm. First-order
    closure cannot represent non-local within-canopy mixing.
  - Daytime latent heat flux biased high (by about 70 W m⁻²).
  - Soil heat flux biased high (but the point measurement is likely unrepresentative).
  - Soil moisture is prescribed rather than predicted.

## Repository layout (current)

The code mirrors the Fortran module structure one file per module. This is being modernized;
see below.

```
src/
  multilayer_canopy/   ML*Mod.py: canopy physics (radiation, turbulence, photosynthesis, ...)
  clm_src_biogeophys/  CLM soil/surface routines and *Type state containers
  clm_src_main/        CLM state containers (clm_instMod globals), constants, driver (clm_driver.py)
  clm_src_utils/       time manager (module globals), orbital parameters
  clm_src_cpl/         coupler stubs (InitializeRealize, ModelAdvance)
  offline_driver/      tower table (TowerDataMod), forcing readers, CLMml_driver (output writer)
  offline_executable/  main.py (the `clm-ml-offline` entry point) and namelists nl.*
  input_files/         CHATS7 tower forcing and CLM history netCDF
  rsl_lookup_tables/   psihat.nc
tests/
  fortran_validation/  160+ routine-level tests against Fortran golden JSON (rtol 1e-9)
  regression/          end-to-end regression against committed reference (rtol 1e-6)
  test_fortran_python_comparison.py  end-to-end vs Fortran .out files (slow; 2.5% / r ≥ 0.9999)
  test_differentiability.py          jax.grad through the canopy model (slow)
timing/                time_offline.py (timing test) and results.jsonl
docs/design/           modernization plan, API sketch, example YAML, timing results
```

### Gotchas in the current code

- **Enable float64 before importing any model module.** Modules create `jnp.float64` arrays at
  import time. `offline_executable/main.py` sets `jax_enable_x64` at the top. Import it (or set
  the flag yourself) first, or arrays silently become float32.
- **Do not import `clm_instMod` (or anything that imports it, such as
  `offline_driver.CLMml_driver`) before `InitializeRealize` runs.** It captures `nlevgrnd = -1`
  and later fails with a negative-shape error.
- State lives in **module globals** (`clm_instMod.*_inst`, `MLclm_varctl.*`). Running the
  model mutates them, so run end-to-end tests in a **subprocess**.
- **Forcing and time are data** (Phase 2). All tower and CLM-history records are read
  before the time loop (`clm_ml.legacy.forcing.load_forcing` → `clm_ml.forcing.Forcing`).
  The physics receives `StepTime` (`clm_time_manager.step_time`) instead of reading the
  time manager, and receives the previous/next-step forcing as an `AtmForcingWindow`
  argument. Don't reintroduce file reads or `clm_time_manager.itim` in the run path.
- Profile axes (layers, leaves `isun=1`/`isha=2`, bands `ivis=1`/`inir=2`, soil levels, RK
  stages) are **1-based** with an unused index 0 (the ground in a few layer fields). The
  legacy containers keep a "patch" axis of length 1; in the run path the patch index is 0
  (`begp = endp = 0`). Phase 7.1 moves to 0-based arrays without the patch axis (D1).
- **The legacy view (Phase 7):** the regression reference stores raw legacy fields
  (`mlcanopy.<field>`, 1-based, patch axis). `clm_ml/legacy/view.py::legacy_view` maps the
  state to them; the output variables (`outputs._Raw`) and validation streams read the
  state only through it (the legacy `.out` writer still reads the containers:
  `write_output`). When the state changes, change the view, never the reference.
- Since Phase 3c each physics routine has **one** traceable implementation taking a `GridInfo`
  (since Phase 5b only the patch index and the static C3/C4 flag; layer counts are data, see
  below). Guard checks that need concrete numbers with
  `MLMathToolsMod.is_concrete(...)`. `MLCanopyFluxes` drives one `_substep` function with a
  Python loop in normal runs and `lax.scan` + `jax.checkpoint` when traced. Deviations of
  the running model from Fortran are listed in `docs/design/numerics_and_regression.md` §4;
  don't "fix" them inside a refactor.
- **Runs are compiled fixed-length chunks** (decision 33, `execution.chunk`, default 10 days;
  a short last chunk padded with inactive steps, so one program per run;
  `offline_executable/main.py`: `run_streams`, `chunk_schedule`, `make_batch_runner`);
  forcing goes to the devices split by patch (`place_inputs`); output streams accumulate
  inside the scan (`clm_ml/output_streams.py`, variables defined once for numpy and jax.numpy in `clm_ml/outputs.py`). Restarts and health
  checks between chunks. `clm_ml.model.Model` runs chunks in-process for gradients.
- **XLA CPU gradient crash (issue 16, JAX 0.9.2):** running compiled CPU gradients of a CHATS7
  chunk segfaults in LLVM-optimized code. Run CPU gradient tests with
  `XLA_FLAGS="--xla_cpu_parallel_codegen_split_count=1 --xla_backend_optimization_level=0"`
  (gradient tests only: round-off differs; `tests/gradient_tier.py` sets it, and the
  regression tests and `record_run.py` refuse to run with it). (JAX 0.6.2 had a different
  gradient crash;
  `clm_ml.model` sets `--xla_cpu_use_thunk_runtime=false` only for JAX < 0.7.)
- **Never scatter into a scan's carry with array indices** (`.at[idx].set` inside the chunk's
  `lax.scan`): with JAX 0.9.2 on CPU it gave wrong states and intermittent segfaults (Phase 6
  step 4). Patch groups are contiguous slices joined by concatenation (numerics doc §6).
- **CPU compile crash (JAX 0.9.2):** large batched programs segfaulted intermittently while
  compiling (LLVM JIT linker recursion overflowing a thread stack); `clm_ml/__init__.py` sets
  `XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1` on import unless already chosen.
- **Read `clm_varpar` layer counts at call time** (`clm_varpar.nlevsoi`), not with
  `from clm_varpar import nlevsoi` in code that may run before the first setup: that copies -1
  (the vectorized setup did; `test_vector_setup.py` now runs it first in a fresh process).
- The `-999` value in tower and parameter tables means "use the default".
- **Options reach the physics only as `opts` (Phase 5a);** the module variables of
  `MLclm_varctl` are setup inputs. `pftcon_val` is still copied at import by `pftconMod` and
  `MLpftconMod` (`from ... import pftcon_val`), so `clm_ml.legacy.worker` sets the options,
  PFT tables (and their `Init` functions, which `initialize2` calls again) and the tower
  table *before* those modules are imported, in a fresh process (goes in Phase 7.4).
- The standalone soil temperature (`MLSoilTemperatureMod`) assumes **one patch per soil column**
  (p == c). Multiple PFTs sharing a column is not supported by the current code.
- **Options are passed explicitly (Phase 5a):** `MLclm_varctl.MLOptions` (built from model.yaml
  by `setup_run`, kept in `LegacyRun.opts`) goes to every routine as `opts=`; compiled
  functions take it as a static argument. Don't read options from `MLclm_varctl` module
  variables in physics code. The time step receives `opts.step_options()` (setup-only layer
  fields at defaults, Phase 6). A process can hold models of **any towers** with one soil
  layer structure (`clm_ml.model.Model`).
- **Per-site data are in the containers (Phase 6 step 2):** `ClmInstances.site`
  (`clm_src_main.SiteType`: soil z/dz/zi, bedrock index, soil colour, lat/lon) is passed to
  the time-step routines; don't read `ColumnType.col`/`GridcellType.grc`/`isoicol` there.
- **Runs are batches of patches (Phase 6 steps 3-6):** `offline_executable.main.PatchBatch`
  (one-patch legacy runs as prototypes, `make_batch_runner` vmaps the step over patches,
  C3/C4 as contiguous groups, `shard_map` over devices with `execution.devices`, padding with
  `execution.patch_count/gridcell_count`); setup by `clm_ml.legacy.vector_setup` (bit for bit
  the per-patch legacy setup, `worker.setup_batch_legacy`); the first step is batched and
  compiled (`batched_first_step`). Outputs carry `gridcell:<var>` sums formed in the scan.
  Failures per patch: `numerics.on_patch_error: stop | exclude`.
- **Layer counts are data (Phase 5b):** every profile array has `max_layers` slots
  (model.yaml `canopy_layers.max_layers`, default 50); `MLLayersMod.layers(inst, p)` gives
  the counts and masks (`in_use` 1..ncan, `in_canopy`, `above_canopy`, `leafy`). Compute on
  every slot, give padding the top layer's inputs (`pad_with_top`), write back with masks
  (`keep`, `restore_padding`). Tridiagonal solves pad with identity rows after the top layer
  and run as `lax.scan` (`tridiag_scan`, `tridiag_2eq`). No Python loops over layers in the
  time step. The template containers are an *input* of the compiled chunk.
- **Fortran error checks are recorded, not raised (Phase 5a/5b):** `record_balance_error`
  stores the largest error of each check (`MLCanopyFluxesType.BALANCE_CHECKS`, 13 checks with
  Fortran's tolerances) and `check_balance` stops the run on the host after each eager step or
  compiled chunk. **Never add `jax.debug.callback`/`print` to compiled code:** it makes the
  program uncacheable.
- **Compilation cache:** every compiled program is cached on disk
  (`JAX_COMPILATION_CACHE_DIR`, default `~/.cache/jax_compile_cache`; 2 GB cap with
  `filelock`). A repeated run in a new process is ~3× faster.
- **C3/C4 is a static choice per group of patches** (decision 22), not per-patch data.
- **Grid runs (Phase 6b, `domain.type: grid`):** `clm_ml/grid.py` (domain from the surface
  data), `clm_ml/grid_forcing.py` (CRUJRA/CO2/LAI/soil water as CDEPS and CTSM handle them),
  `clm_ml/legacy/grid_setup.py` (prototypes per PFT from tower-format files of one gridcell,
  every forcing-dependent field per patch, `GridChunkReader`). Vegetation and soil water are
  interpolated inside the compiled chunk (`PatchBatch.land_fn`). Decisions G1-G20 and open
  issues in `docs/design/global_input.md`: when a choice comes up, copy CLM (source in
  `/glade/work/linnia/ctsm5.4.004`). Example configs `docs/design/examples/*grid*`.
- **Don't import legacy record modules (`clmDataMod`, `TowerMetMod`, ...) before the first
  legacy setup** in grid setup code either: the grid setup builds its first-step data after
  the prototypes (`setup_arrays(first=callable)`).
- **CPU compilation cache bug (JAX 0.9.2):** loading some large cached CPU programs back
  crashes in LLVM's object loader (deterministic); the batched first step is compiled without
  the cache on CPU (`main.without_persistent_cache_on_cpu`). A deterministic segfault right
  after "Initialization complete." on CPU is this; try an empty cache directory.
- **Login nodes have GPUs** (derecho3, derecho7: two A100s each) and JAX uses them by
  default: tests and the reference recording force the CPU (`tests/conftest.py`,
  `record_run.py`); probes must set `JAX_PLATFORMS=cpu` too. On the GPU, two fresh compiles of
  the same program can differ by ~1e-11 (decision C4 in `phase7_decisions.md`), so bit for bit
  is a CPU test.
- **Real 1° patches amplify round-off** (1e-15 in LAI -> ~8 W m-2 after sunrise): grid
  regression tests compare inputs exactly and outputs only through the night.

## Commands

```bash
pip install -e ".[dev]"                     # dev extras include pytest-cov, which pytest.ini needs (--no-cov)

clm-ml convert src/offline_executable/nl.CHATS7.1day -o chats7/   # namelist -> run.yaml + model.yaml
clm-ml check chats7/run.yaml               # validate + summary, no run
clm-ml params ctsm60_params.nc -o p.yaml   # CLM parameter file -> complete params.yaml (built-in clm6 = CTSM 6.0)
clm-ml run chats7/run.yaml [--set key=value]   # run via the clm_ml API (netCDF output)
clm-ml-offline src/offline_executable/nl.CHATS7.1day      # legacy run (writes src/output_files/*.out)
python -m pytest tests/clm_ml -q                           # clm_ml unit tests (fast)
python -m clm_ml.legacy.tables --write     # regenerate data/params/*.yaml + data/towers.yaml from legacy tables

python -m pytest tests/fortran_validation -m "not slow" -q            # routine-level golden tests (fast)
python -m pytest tests/regression -m regression -q                    # all end-to-end regression tests (~8 min CPU; end of phase)
python -m pytest tests/regression/test_end_to_end_regression.py tests/regression/test_clm_ml_equivalence.py -m regression -q   # bit-for-bit + noise twin only (~1 min; before each commit)
python -m pytest tests/test_fortran_python_comparison.py -m slow -q   # vs Fortran output
python tests/gradient_tier.py [--focused]                  # CPU gradient tier (tests marked `gradient`, LLVM optimization off; issue 16)
qsub -v REV=<commit>,TAG=<name> jobs/grad_gpu.pbs          # GPU gradient tier (develop queue, warm GPU compile cache)

python tests/regression/record_run.py --update-reference   # reference + 2 noise twins; only after intended or round-off-only changes; say so in the PR
python timing/time_offline.py                              # timing test; appends to timing/results.jsonl
python timing/time_model.py [--steps 8] [--no-grad]        # compiled chunk + gradient timing/memory; results_model.jsonl

black --check --line-length 100 src/ tests/ && ruff check src/ tests/   # what CI runs
```

Python ≥ 3.10. CI tests 3.10 and 3.12.

## Rules for changes

- **Regression policy** (`docs/design/numerics_and_regression.md` §6):
  - refactors that should not change the arithmetic must pass the **bit-for-bit** test;
  - every change must pass the **noise-twin** test, i.e. stay within the model's own round-off
    sensitivity, measured by two twin runs perturbed by 1e-12.
  A change that legitimately alters round-off (traceable code, compilation, GPU) is expected
  to fail bit-for-bit. Regenerate the reference with `--update-reference` and say so.
  Science changes also regenerate it, explicitly.
- **Test in tiers** (numerics doc §6.1); end-to-end tests cost ~40–60 s per model run, so don't
  run the whole suite after every change:
  - after each edit: `tests/clm_ml` + `tests/fortran_validation -m "not slow"` (~10 s), plus the
    test being written;
  - before each commit: the bit-for-bit and noise-twin tests (one shared run, ~1 min), plus the
    changed feature's own end-to-end test;
  - at the end of a phase: all of `tests/regression` (~8 min) and the slow tests (~20 min).
  Every compiled program goes into JAX's on-disk cache (`~/.cache/jax_compile_cache`), so a
  repeated run in a new process takes ~20 s; never add host callbacks (`jax.debug.callback`)
  to compiled code: they make the program uncacheable (record errors instead, §6.8 of the plan).
  New end-to-end tests simulate as little as needed (usually one day) in as few runs as possible.
- **Noise twins differ by ≤ 4e-11**, so the noise-twin test is tight. An apparent
  evening-transition sensitivity (up to 4.5%) was a solver bug: the fixed-step secant took a
  ~100 m step when two converged residuals were exactly equal (fixed: no step when f1 == f0;
  numerics doc §5). If a round-off-only change moves results by more than ~1e-10, look for a
  similar discrete defect (compare routines on identical input, §1) instead of widening tolerances.
- **Iterative solvers:** the forward value must be the solver's own answer (Fortran algorithm
  and stopping rule). Implicit-function gradients are applied with
  `x0 - (step - stop_gradient(step))` so they never change forward values. Known deviation:
  the Obukhov solver is 25 fixed secant steps, not Fortran's hybrid secant/Brent with
  tol = 0.1 m (recorded, not changed). Fixed-step secants must take no step when f1 == f0.
- Keep the Fortran golden tests passing. Their JSON files are fixed reference data; never edit
  them to make a test pass.
- **Don't change the science while refactoring.** Keep refactors and science changes in
  separate commits and PRs.
- Every refactored function keeps a `Fortran reference:` line in its docstring (module and
  subroutine) so it can be traced back to CLM-ml v2.
- Record design decisions in `docs/design/` in the same change. Don't leave documentation for
  later.
- Commit and push when the required tests pass (see standing preferences). Work on a branch per
  phase (`phase0/safety-net`, ...).

## Modernization plan

The full plan is in **`docs/design/modernization_plan.md`**. The proposed API is in
`docs/design/api_sketch.py`, and example config files are in `docs/design/examples/`.

**Decisions:**
- **Built-in towers: CHATS7 only** (`src/clm_ml/data/towers.yaml`). Other towers go in a
  user `towers.yaml`, with explicit layer counts per patch (`n_layers_within`/
  `n_layers_above`) where default spacing gives too many layers.
- **Names.** Import as `import clm_ml`. The install name is `clm-ml-jax`. The command-line tool
  is `clm-ml run | check | convert`.
- **Configuration** is in three YAML files:
  - `run.yaml` is the experiment (towers, dates, forcing files, initial conditions, output). It
    replaces the namelist.
  - `model.yaml` holds scheme choices and numerics. These are static, so changing them
    recompiles.
  - `params.yaml` holds parameters **per PFT**. It uses `extends: clm5_defaults` with
    overrides, and each patch looks up its PFT's values.
- YAML loads into typed dataclass pytrees. Validation is plain Python, with errors that name the
  file and key. No Hydra, no Flax.
- **Readable option names** (`stomatal_conductance: wue`). Fortran integer codes are accepted
  only by `clm-ml convert`.
- **Vocabulary** (use these terms precisely; avoid "site" except for observation locations
  in the scientific sense):
  - **gridcell**: a location with its own atmospheric forcing.
  - **patch**: one PFT's canopy on a gridcell, with an area weight. **The unit of computation.**
  - **soil column**: the soil under a patch.
  - **tower**: a named flux-tower location in the registry (`towers.yaml`), which defines a
    gridcell and its patches.
- **Spatial scope:** from one tower up to global (many gridcells, no lateral exchange).
  - Patches are a flat list with `gridcell_index`/`column_index`.
  - The physics is written for one patch, and `vmap` runs it over the patch axis.
  - Forcing is gathered from gridcells and parameters from the PFT table. Gridcell output is the
    area-weighted `segment_sum` of patches.
- **Soil under patches.** Option A (default, `soil.columns: per_patch`): each patch has its own
  soil column, identical to today's p == c. Option B (`shared`) is a later, separately validated
  science change.
- Every patch gets a **fixed `max_layers` (default 50) with masks**. Padded layers must not
  change results, and must not give NaN gradients (safe-`where`).
- **Memory must not depend on run length** (target: global, 150 years of 6-hourly forcing).
  Forcing comes from **streams** (meteorology, vegetation, soil water, each with its own time
  axis) read in **chunks** with one halo record at each end. A run is a loop over chunks that
  carries the state, reads the next chunk in the background, and writes output per chunk. Never
  design something that needs the whole run's forcing or per-step output in memory (plan §6.9).
- **Chunks vs output intervals** (as jax-gcm): output is accumulated *inside* the compiled scan
  (nested scan: outer loop over output records, inner loop over steps, `jax.checkpoint` on the
  step), so memory scales with output records, not steps. The chunk length is host
  bookkeeping (forcing in memory, restarts with `.prev`, health checks, file writes). A
  compiled chunk is reused for all chunks of the same shape; data never trigger compilation.
- **Model time step is configurable** (`numerics.step_seconds`); forcing streams are
  interpolated to it (linear, `coszen` shortwave, constant precipitation, midpoint
  interpolation of daily/monthly LAI, SAI and soil water).
- **Acclimation temperature: 10-day running mean** carried in the state (science change,
  Phase 4b). The Fortran whole-run mean stays as an option for validation only.
- **LAI, SAI and soil water** come from user-supplied daily or monthly files (CLM history at the
  forcing step still supported for CHATS7).
- **Output is configurable and independent** of forcing and model step: output streams with
  their own variables, frequency, statistic, level and file split, accumulated during the run.
- **Gradients** with respect to parameters, meteorological forcing and initial state.
- Regression tolerance is `rtol = 1e-6`. The legacy `.out` output is only kept until validation
  moves to netCDF.

**Phases:**

| Phase | Work |
|---|---|
| 0 | Safety net: regression test, packaging, timing test (done) |
| 1 | YAML config and `clm_ml` API as a wrapper around the existing code (done) |
| 2 | Forcing and time become data (done) |
| 3 | Explicit state and a pure `step` (done) |
| 4 | Chunked `jit(scan)` runs, forcing streams, output streams, restarts (done, except GPU timing) |
| 4b | Science changes: 10-day acclimation, configurable model step with time interpolation, daily/monthly land data (done; plus forcing split in time and the §4.1 forcing layout) |
| 5a | Explicit configuration: `MLOptions`, cacheable compiled programs, several same-tower models per process (done) |
| 5b | Mask-aware layers **in place** (same files, 1-based): one compiled step for any layer count (done; gradient padding test and GPU timing after Phase 6) |
| 6 | Many patches and gridcells, **on Derecho (GPU)**: steps 0-6 (per-site data, batched patches, C3/C4 groups, devices, vectorized setup) done 2026-10-29; GPU optimization (step 1c) deferred; progress log `docs/design/phase6_progress.md` |
| 6b | Real global input before Phase 7 (decisions 26-32): steps 1-6 done (grid loader, gridded forcing, setup, regional test), done 2026-10-30 (step 7 closed; review fixes, calendar decision 34); fixes: per-layer minimum plant area (30), no 0.05 LAI cutoff (31), canopy-top latent heat per sub-step (32); open issues in `global_input.md` |
| 7 | Restructure (decision 35; `phase7_scope.md`): 7.0 preparation, 7.1 step state (0-based), 7.2 physics by process (`clm_ml/physics/`), 7.3 forcing in the chunk, 7.4 setup without globals, 7.5 netCDF restarts, 7.6 cleanup. Not started (2026-10-30) |
| 8 | Docs and release |

Decisions 18–21 (after Phase 4b): Phase 5 converts to masks in place; masks rather than grouping
patches by layer structure; configuration passed explicitly (5a); process interface in Phase 7.
Decision 22 (Phase 5b, measured): C3/C4 stays a static choice per group of patches; Phase 6
groups patches by pathway inside one compiled program (computing both pathways per patch
doubles photosynthesis work on GPU). Masks for layer counts (many values), groups for
few-valued code switches.

Every phase must give the same answer as the previous one. Use adapters first and rewrite
internals later.

**Layer counts.** With the default spacing, short canopies under tall towers get very many
layers: above-canopy spacing equals within-canopy spacing, so a 0.5 m grass under a 30 m tower
needs 300 layers. Bonan et al. (2021) found 5–10 within-canopy layers sufficient for fluxes;
use explicit `n_layers_within`/`n_layers_above` for such towers.

## Design guidance: jax-gcm

Use **[jax-gcm](https://github.com/climate-analytics-lab/jax-gcm)** (JCM; docs at
[jax-gcm.readthedocs.io](https://jax-gcm.readthedocs.io/en/latest/)) as the reference for software
design. Like this repo, it is a differentiable JAX Earth-system model that grew from a Fortran
translation (SPEEDY). Its design philosophy:

- **Pure functions and immutable pytree state.** No hidden globals. Everything a function
  depends on is an argument, and everything it changes is returned. No Python `if` on traced
  values: use `jnp.where` or `lax.cond`. Array shapes are known statically.
- **Separate static configuration from differentiable parameters.** Flags and enums that choose
  code paths are static. **Every numeric tunable must be differentiable**; never hide a
  parameter from `jax.grad` by making it static.
- **Composable physics.** Each parameterization is a `PhysicsTerm` with one uniform signature
  `(state, diagnostics, forcing, terrain) -> (tendency, diagnostics)`. Each term declares what it
  `requires` and `provides`, and the container validates the ordering (single writer, upstream
  requirements met) when the model is built. Schemes are swapped by name (`replace`, `remove`,
  `+`). Our plan uses a lighter registry with the same declare-reads/writes idea (plan §6.6).
- **Column physics is written once and runs anywhere.** jax-gcm writes columns
  "broadcasting-native" (vertical on axis 0, horizontal axes broadcast) or wraps them with
  `vmap` (`vectorize_columns`). The same code runs on one column or a whole grid.
- **Wrap translated Fortran, don't rewrite it.** The SPEEDY numerics were left untouched behind
  thin term wrappers that own unit conversions at the boundary, then modernized gradually.
- **Files are organized by physical process and scheme name**, not by the model they were ported
  from.
- **Configuration through files, one run command.** Validated configurations are named config
  files with comments explaining why each setting is what it is. There are no ad-hoc driver
  scripts. (jax-gcm uses Hydra; we use plain YAML.)
- **Labelled output.** Results go to xarray with coordinates and units. Select by coordinate
  value, never by an assumed index (for example, which end of the vertical axis is the ground).
- **Documentation lives with the change.** Design decisions go in `docs/.../design/*.md`, and a
  science choice is stated with its reference provenance (Fortran `file::routine` or paper).
- **Gradient hygiene.** Test gradients against finite differences. Watch for NaN gradients from
  `jnp.where`/`jnp.clip` guards (both branches are differentiated), `sqrt` at 0, and float32
  overflow in derivatives. See jax-gcm's `JAX_gotchas.md`.
- **Finish the job.** Deliver complete, faithful fixes that match the reference formulation, not
  partial workarounds presented as done.
