# Phase 6 progress log (Derecho)

Where Phase 6 stands, for picking the work up after a disconnect. The plan and its reasons are
in `modernization_plan.md`, Phase 6 (steps 0–6, agreed with the maintainer on 2026-10-28);
this file records what is done, what is running, and what comes next. Newest entries last.

The maintainer asked for Phase 6 to be worked through **without their input until it is
done**, committing regularly (branch `phase6/patches`, push after the required tests pass).

## Environment (Derecho)

- Checkout: `/glade/work/linnia/clm-ml-jax`; conda env `/glade/work/linnia/conda-envs/clm-ml-jax`
  (Python 3.11, JAX 0.9.2 with CUDA 12 plugin; `filelock` not installed, so the compilation
  cache has no size cap).
- Compilation caches on scratch: `/glade/derecho/scratch/linnia/jax_compile_cache_{cpu,gpu}`
  (set `JAX_COMPILATION_CACHE_DIR`). On the login node set `JAX_PLATFORMS=cpu`.
- PBS: GPU jobs `-A UCLB0069` (`-q develop`, routes to `gpudev`; or `-q main`), CPU jobs
  `-A P93300041` (`-q main`). `qstat <id>` does not find jobs routed from `main`: use
  `qstat -u linnia`.
- Job scripts and logs: `/glade/derecho/scratch/linnia/clm-ml-jax/jobs/`; nsys profiles:
  `.../clm-ml-jax/prof/`; probes: `.../clm-ml-jax/probe/`.
- Tests: fast tier and before-commit tests on the login node (`JAX_PLATFORMS=cpu`, pytest
  `-p no:cacheprovider -o addopts=""`); slow and end-of-phase tests as PBS jobs.

## Log

### 2026-10-28: deferred GPU work, step 1b, step 1c measurements

- Fast tier passes on Derecho (267 tests). Bit-for-bit regression fails on 65 fields by
  ~1e-16 (JAX 0.9.2, new CPU); noise-twin passes (87). → step 0a.
- XLA CPU-runtime crash re-test: the workaround flag is "no longer supported" in 0.9.2 and a
  2-step CPU gradient runs without crashing. → step 0c.
- GPU timing (A100 40 GB, one patch, `timing/results_model.jsonl`): forward 366–384 ms/step,
  gradient 2.0 s/step (8-step chunk); compile with an empty cache 100 s (forward) and 430 s
  (gradient), warm cache 13 s and 83 s. Legacy offline run on GPU 3.05 s/step
  (`timing/results.jsonl`), laptop CPU 0.17 s.
- Gradient padding test (46 vs 60 slots) **passes on GPU** (18 min).
- Found: float32 LAI/SAI/height/albedo/coszen containers (plan open issues). → step 0b.
- Step 1b (`timing/time_vmap.py`, `timing/results_vmap.jsonl`): forward throughput rises
  ~linearly to ~4 000 patches and levels at ~28 500 patch-steps/s (65 536 patches, 21.5 GB).
  Batched patches equal the single run exactly. Gradient: 1.3 s/step at N = 1 and 64.
- Step 1c baseline (job `gpu_1c`): nsys at N = 1: 31 600 kernels/step, GPU busy 16 %; N = 4 096
  profile in `prof/base_N4096*`. CUDA graphs over loops slower (see plan); optimization
  deferred by the maintainer.
- Plan revised and agreed (decisions 23–25; steps 0, 2–6).

- Step 1c CUDA-graph results (`timing/results_vmap.jsonl`, labels "1c ..."): loops captured
  (`CB`) slower (N = 1: 530 ms/step, N = 4 096: 632 ms; default 367 and 508 ms); with
  `--xla_gpu_command_buffer_unroll_loops=true` (`CBU`) N = 1 368 ms, N = 4 096 456 ms (10 %
  faster). nsys: with command buffers the kernels move into ~20 000 graph launches per step,
  and ~22 600 device-to-device copies per step remain (loop-carried values). Profiles:
  `prof/{base,CB}_N{1,4096}*` (`*_cuda_gpu_kern_sum.csv`, `*_cuda_api_sum.csv`).

- Step 1b gradients (A100, 8-step chunk, gradient of the summed chunk-mean latent heat with
  respect to the shared parameters): N = 1: 1.30 s/step, 0.10 GB; 64: 1.35 s, 0.14 GB; 512:
  1.48 s, 1.1 GB; 4 096: 1.95 s/step (2 100 patch-steps/s, 3.8x the forward), 9.0 GB (~2.2 MB
  per patch). Compile time grows with N: 391 s (N = 1) to 1 568 s (N = 4 096) for the
  gradient, 110-460 s forward. One-patch 47-step chunk: 383 ms/step (as 8 steps).

### 2026-10-28: step 0 done, step 2a

- 0a (`98921b6`): reference recorded on Derecho. 0b (`3248b2b`): float64 inputs, ≤ 7e-9.
  0c (`d9a2019`): CPU-runtime flag only for JAX < 0.7.
- Fortran comparison after 0b (job `fortran_compare`, May 2007, `acclimation = run_mean`,
  from a git worktree of `3248b2b`): max rel. error 0.068 %, mean 0.0078 % over 42 columns
  (flux, aux, soiltemp; `aux` col 2 excluded; fsun/profile/fluxprofile skipped as before);
  2 columns below r = 0.9999 (flux col 15 r = 0.9968, aux col 3 r = 0.99984), as at the end of
  Phase 5. Summary script: `probe/fortran_compare_summary.py`; job `jobs/fortran_compare.pbs`
  (env `REV`).
- 2a (`8c1606b`): `clm_src_main.SiteType.site_type` in `ClmInstances.site` (soil z/dz/zi,
  bedrock index, soil colour, lat/lon), passed to the time-step routines; bedrock as a mask.
  Round-off only (XLA no longer folds divisions by the constant `dz`; identical-input check
  1e-15; run ≤ 1.4e-11); reference regenerated. Full `tests/regression` job `reg_2a`
  (`jobs/run_tests.pbs`, env `REV TESTS MARK TAG`, runs on a worktree snapshot).

### 2026-10-28: step 2 done

- 2b (`96fbeac`): `MLOptions.step_options()`; per-patch `n_layers_within/above` in
  towers.yaml; models of any towers in one process; `test_sites_one_program.py` (site B with
  different lat/lon, bedrock, soil colour, layer counts through site A's program, no
  recompilation, equal to B's own run and B alone). Bit for bit.

### 2026-10-28: step 3 (3a-3d)

- 3a run path batched (`PatchBatch`, `make_batch_runner`, `read_batch_inputs`, `run_streams`
  on a batch, `Outputs.gridcell_data`); each patch set up and first-stepped before the next
  setup (module state); `GridInfo` per patch; mixed C3/C4 refused until step 4.
- 3b `Model` batched (PFT table, weights and gridcell map as inputs, `program=`).
- 3c decision 25 (`on_patch_error`, `max_failed_patch_fraction`).
- 3d padding (`execution.patch_count`, `gridcell_count`).
- Fixed: `pbeta_lai/sai` initial value `-spval` (Fortran) instead of `+spval` (PFT default
  leaf/stem profiles were never applied; setup stopped for patches without explicit profiles).
- `test_process_io`: `shair_profile` allowed 3e-10 (1.1e-10 with JAX 0.9.2 before any Phase 6
  change; solved temperatures agree < 1e-13).
- Login-node limit: running 5 model processes at once was killed (SIGKILL); tests run at most
  2 in parallel.
- New tests: `test_patches_batched`, `test_model_patches` (slow part: PFT gradient vs finite
  difference), `test_patch_exclusion`, `test_patch_padding`.

- Committed: `2e219dd` (pbeta fix), `d9e6470` (steps 3 and 4; full regression 110 passed on
  the snapshot, slow PFT-gradient test passed; reference regenerated).

### 2026-10-29: step 5 in progress

- CPU node (AMD EPYC 7763, job `cpu_node`, `timing/results_vmap.jsonl` "Derecho CPU node"):
  one JAX process levels at ~170-200 patch-steps/s (N = 1: 81; 16: 199; 128: 173; 1 024:
  171): XLA CPU runs these small operations essentially on one core. An A100 (~28 500) is
  ~140 such processes. (The job's 8-step `time_model` run used the working tree during the
  step 5 edits and recorded nothing; rerun later.)
- Step 5 code (uncommitted while debugging): `PatchBatch.blocked`, `layout`, `source`,
  `real_index`, `runner_gidx`; `_sharded` (`jax.shard_map`, `check_vma=False`);
  `execution.devices`; restart `layout`. `test_devices.py` (4 simulated CPU devices).
- **Bug found (open):** with C3 and C4 patches on one device, the grouped path (gather
  `[0, 2, 3]` / `[1]`) gives wrong C3 results from the second compiled step (leaf temperature
  state off by up to 0.4 K after one step while that step's fluxes agree); the 4-device run
  (contiguous groups per device) matches a C3-only domain to 1e-10. `test_c3_c4_groups`
  missed it (C3 group = patch 0 only). Diagnostics: `probe/devices_diag*.py`, job
  `jobs/diag.pbs`. A segfault on the login node running three models in one process (maybe
  memory) moved the diagnostics to a compute node.

### 2026-10-29: steps 5 and 6

- Step 5 committed (`832d311`) with the step 4 fix (contiguous C3/C4 slices; no scatter into
  the scan's carry: it gave wrong states and segfaults). 4-GPU timing (`jobs/gpu4.log`):
  16 384 patches 30 900 patch-steps/s, 65 536 patches 67 900 (3.8x one GPU at the same load
  per GPU); `time_vmap` with 262 144 copies ran out of memory (it builds all copies on GPU 0).
- Step 6a (batched compiled first step) and 6b (vectorized setup, bit for bit vs legacy).
  Bug found and fixed on the way: `vector_setup` read `nlevsoi` before the first setup set it.
- Running: full regression on the step 6 snapshot (`jobs/s6c.log`), global timing on 4 A100s
  (`jobs/global.log`: 65 536 and 262 144 patches).

- Full regression on the step 6 code (`s6c`): 198 passed, 2 failed (`test_c3_c4_groups`,
  `test_devices`): worker segfaults. `gdb` (`probe/segv_repro.py`, `probe/segv_2.log`): the
  crash is in LLVM's JIT linker during compilation (recursion overflowing a thread stack), not
  in the model. `--xla_cpu_parallel_codegen_split_count=1`, set in `clm_ml/__init__.py`, gave
  5/5 clean compilations (about 140-155 s each for the reproducer). Rerun of the failing tests
  and the before-commit tests: job `s6d`.

### 2026-10-29: end of Phase 6 checks; Phase 6b agreed

- Fortran comparison (job `fortran_compare`, commit `2205796`): max 0.068 %, mean 0.0078 %,
  the same 2 columns below r = 0.9999 as at the end of Phase 5.
- Global timing (`jobs/global.log`, 4 A100s): 65 536 patches: setup 53 s, first step 838 s,
  chunk compile 1 019 s, steady 51 000 patch-steps/s, all finite; 262 144 patches: setup
  111 s, first step 809 s, then out of GPU memory in the chunk. The compilation cache could
  not store these programs (`SerializeExecutable proto serialization failed`).
- Full regression + slow tests on the final code: job `endphase` (queued at the time of
  writing).
- Maintainer agreed Phase 6b (plan, decisions 26-29): real global input before Phase 7; the
  maintainer provides parameters, surface data, CRUJRA and soil moisture files.

### 2026-10-29: Phase 6b plan

- Inventory done (`docs/design/global_input.md`); input decisions recorded there and in the
  plan's Phase 6b section (1°, year 2000, CLM land mask: 14 346 gridcells, 72 575 patches).

### 2026-10-29: Phase 6b steps 2-5 (maintainer: work through 6b, copy CLM, document)

- Committed `b8d975d`: `domain.type: grid` (`clm_ml/grid.py`), `clm_parameters` in
  params.yaml (CLM6 file), gridded forcing as CDEPS/CTSM handle it (`clm_ml/grid_forcing.py`),
  grid setup and chunk reader (`clm_ml/legacy/grid_setup.py`). Every choice that came up
  copies CLM (CTSM 5.4 / CDEPS source in `/glade/work/linnia/ctsm5.4.004`); decisions G1-G19
  and open issues 1-11 in `global_input.md`.
- Setup fields that depend on the forcing and land data: found by perturbing each CHATS7
  input one at a time (probe `probe/6b/setup_deps.py`): the forcing containers, `t_a10`,
  `pref_forcing`, soil temperature and water, canopy-air/leaf profiles, first-step inputs.
- Tests (`tests/regression/test_grid_setup.py`): the grid setup equals the legacy setup of
  tower files holding each gridcell's gridded data, bit for bit; a July day: chunk inputs
  identical, outputs round-off through the night, then up to ~10 W m-2 apart after sunrise
  because these patches amplify round-off (1e-15 in LAI -> ~8 W m-2; open issue 10).
- First grid run (CPU, 12 gridcells, 61 patches, 1 July 2000): 140 s, all finite, gridcell
  latent heat 104 W m-2, GPP 12.8 umol m-2 s-1 (day means).
- Found: loading the cached batched first step crashes on CPU (open issue 11); now compiled
  without the cache on CPU.
- Before-commit tests, fast tier (267), coarse forcing, vector setup, land data: pass.
- Running: regional test (step 6) on one A100, job 8004270 (`jobs/regional.log`, outputs in
  `runs/regional/`).

### 2026-10-29: Phase 6b step 6 (regional) and the winter fixes

- `6c1b66f`: zero plant area keeps 0.05 of stems (G20); `numerics.ignore_checks` (CLM-ml's
  canopy check 3 fails below freezing: mixed latent heats, issue 12).
- Regional test done (results table in `global_input.md`): January and July 2000, 522
  gridcells, one A100, ~33-44 min per month; 5 shrub patches NaN in January (issue 14,
  reproduction recipe there).
- Recheck of the end-of-Phase-6 failures (job `recheck_dev`, commit `6c1b66f`):
  `test_c3_c4_groups` and `test_devices` pass now (they were the cache-load crash, issue 11);
  the CHATS7 50-step CPU gradient test still segfaults while running (issue 16, pre-existing).
- Running: global 1° run, year 2000, 4 A100s (job 8005591, `jobs/global6b.pbs`, log
  `jobs/global6b.log`, outputs `runs/global2000/`, a restart every month; to continue after
  the walltime: `qsub -v REV=...,RESTART=<runs/global2000/restart/domain_restart.pkl>`).

### 2026-10-29: params files are YAML only (maintainer)

- `params.yaml` no longer reads CLM netCDF files (`clm_parameters` removed). `clm-ml params
  SOURCE -o OUT.yaml` (`clm_ml/params_convert.py`) writes a complete table from a CLM
  parameter file (CLM-ml-specific values from `--base`, default `clm5_defaults`) or flattens
  a params file. Built-in `clm6` = the CTSM 6.0 file converted (identical values to before).
  The roughness ratios are ordinary optional parameters (`z0m_to_height_ratio`,
  `displacement_to_height_ratio`), needed only by grid runs.
- Open issues of Phase 6b to come back to (maintainer): `global_input.md` issues 1-16.

### 2026-10-29: issue 14 diagnosed and fixed (decision 30)

- Diagnosis (probes `probe/issue14/`): CLM-ml's 9-angle diffuse transmittance is 1.0051 at zero
  plant area, so layers thinner than ~0.005 m2 m-2 have td > 1; the Norman longwave then makes
  a layer's net longwave grow with its temperature and calm conditions let it run away. Same
  in the Fortran. Affects up to a third of the vegetated area in winter (most do not diverge
  but create energy).
- Fix (maintainer's choice among the options in `global_input.md` issue 14): every leafy layer
  keeps 0.01 m2 m-2 (CLM-ml v1's dpai_min) at every step, added as stems; replaces G20.
  CHATS7 bit for bit; reference regenerated to add the new field `sai_floor_canopy` (all
  existing fields identical). New test `test_thin_layers_keep_minimum_plant_area`.
- New open issue 17: LAI non-differentiability (CLM's 0.05 cutoff, the floor's kink); raise
  again later (maintainer).

### 2026-10-29: CLM's 0.05 LAI/SAI cutoff removed (decision 31)

- Grid runs use the interpolated monthly LAI/SAI as given (not below 0); the cutoff's job is
  done by the per-layer floor (decision 30), and it was a jump that blocked LAI gradients
  (issue 17a resolved; 17b, the floor's kink, remains). Towers are unaffected (CHATS7 reads
  CLM's history ELAI/ESAI).

### 2026-10-29: issue 12 diagnosed and fixed (decision 32)

- The canopy check 3 failures in winter come from the diagnostic canopy-top latent heat
  (averaged etair times the last sub-step's latent heat) on steps where the air crosses 0 C,
  not from the physics. The diagnostic is now accumulated per sub-step; check 3 closes on every
  step; `ignore_checks` removed from the grid example and tests. CHATS7 round-off only
  (`lhflx_canopy` <= 4.8e-16), reference regenerated.

### 2026-10-30: global run out of GPU memory; fixed-length chunks (decision 33)

- Global run (job 8008287, commit `aa43e33`): January to March done (setup 100 s, first step
  872 s, ~2300 s per month = ~1000 s compiling + ~1300 s running, 46 000 patch-steps/s
  including the compile; no patch excluded), then `RESOURCE_EXHAUSTED: [0] Failed to load
  in-memory CUBIN ... out of memory` loading April's newly compiled program on device 0.
  Restart on disk: 2000-04-01 (`runs/global2000/restart/`; it holds March's pending output:
  a month's file is written when the next chunk completes its last daily record).
- Diagnosis so far: (1) each month length (1487, 1392, 1488, 1440 steps) compiled a new
  program and the persistent cache cannot store them; (2) the grid reader put each month's
  forcing (~30 float64 fields per step and gridcell, ~5 GB a month) on device 0 unsplit, for
  the next month too while one ran (prefetch); the error names device 0. Probes in
  `probe/oom/` (scratch): a `sitecustomize.py` hook logging device memory per chunk, jobs on
  1, 2 and 4 GPUs (chunks of 480, 480 and 95 steps) to measure memory per chunk, whether old
  programs keep memory, and multi-GPU scaling (earlier: one A100 ~28 500 patch-steps/s at
  65 536 patches, four ~51 000).
- Maintainer: switch to 10-day chunks (jax-gcm's design: fixed `chunk_days`, calendar output
  assembled on the host). Decision 33: `execution.chunk: N days` (default 10), last chunk
  padded with inactive steps, fixed record slots, one program per run; restarts at the first
  chunk end on or after each boundary; forcing placed on the devices split by patch in the
  reader thread (`place_inputs`); the grid reader returns host arrays. Tests: fast tier,
  bit-for-bit, noise twin, restart (now with padded chunks), padded chunks vs one chunk (every
  step bit for bit, a record spanning chunks to 1e-15), devices, grid setup, stream runner,
  patch exclusion. Commit `607fe1f`.
- **Probe on 1 A100 (old code, global domain, chunks 480/480/95 steps; `probe/oom/live_g1.log`):**
  - each compiled chunk program has **2.32 GB of GPU code** (`generated_code_size`), loaded on
    every device and kept while the program is cached: GPU memory rose by exactly that when
    the 95-step program loaded. With JAX's pool capped at 31.8 of 40 GB, four month programs
    (9.3 GB of code) did not fit: **this is the April failure**. The code size is also over
    protobuf's 2 GB limit, so these programs can never enter the persistent cache
    (`proto serialization failed`) and every run compiles them (~840 s each). Why the code is
    so large (unrolled loops in the physics?) belongs with the deferred GPU optimization
    (step 1c: 31 600 kernels per step).
  - temporary memory grew by ~17 MB per step (12.0 GB at 480 steps, 5.5 GB at 95): the
    per-patch parameters, weights, alive flags and gridcell indices (~29 numbers per patch)
    were broadcast along the step axis of the scan. Now held once (round-off only: CHATS7
    differences at 1.18x the noise-twin spread at most, reference regenerated).
  - **one A100 runs the whole 1° globe**: 72 575 patches, 816 s per 10 days (42 700
    patch-steps/s), peak 21.2 GB; setup 98 s, first step 794 s. The 4-GPU global run gave
    ~83 000 patch-steps/s without compile (~21 000 per GPU), so one GPU is about twice as
    efficient per GPU-hour.
- **Scaling with GPUs** (same probe, old code; steady 480-step chunk, global domain):

  | A100s | patch-steps/s | per GPU | year, running only | GPU-hours per year | peak GB per GPU |
  |---|---|---|---|---|---|
  | 1 | 42 700 | 42 700 | ~8.3 h | 8.3 | 21.2 |
  | 2 | 60 200 | 30 100 | ~5.9 h | 11.8 | 16.2, 11.1 |
  | 4 | 76 400 | 19 100 | ~4.6 h | 18.6 | 11.2, 6.1, 6.1, 6.1 |

  Compiling (~840 s) and the first step (~800-900 s) do not shrink with more GPUs. Device 0's
  extra memory is the unsplit forcing (fixed in `607fe1f`). One GPU per domain of this size
  is the efficient choice; more GPUs only for wall time.

### 2026-10-30: adversarial review of Phase 6/6b (`d112478..HEAD`)

Four independent reviews (batch runner and run loop; setup; gridded input against CTSM 5.4 /
CDEPS; physics). Probes in `probe/review6/` (scratch). Nothing changed yet; fixes await the
maintainer.

**Confirmed defects**
- R1. *Multi-device tower runs read the wrong forcing*: `PatchBatch.gridcell_patch` takes the
  first slot of a gridcell, which in `blocked()` can be a padding copy of a patch from another
  gridcell, so that gridcell's meteorology comes from another tower's files (first step
  right, every later step wrong; `Model.forcing` too). Grid runs unaffected (own reader).
  `p1_blocked_gridcell_patch.py`. Fix: the first real slot.
- R2. *Calendar drift in multi-year runs*: `clm_time_manager.date_after` (Fortran
  `get_curr_date`) counts years with the start year's length: +37 days after 150 years from
  1901, -112 days from 2000; repeated or missing dates. Affects the solar geometry against
  the forcing (grid runs read forcing at true time), output record keys, and restart resume
  (wrong step or refused). Within one year from a 1 January start it is exact (CHATS7, the
  2000 run). `p2_date_after.py`, `p3_clock_drift.py`. CLM's ESMF clock is exact.
- R3. *A run started from a restart at its start time cannot be resumed* with fixed chunks
  (chunks then start at step 0, not 1; the resume check refuses). `p4_restart_from_start.py`.

**Likely / minor**
- R4. A patch failing a check with finite values still enters the gridcell sums (and
  parameter gradients) for the chunk in which it fails (now 10 days); only its patch output
  is filled.
- R5. Padding slots placed on real gridcells combine one gridcell's meteorology with another
  patch's site data; stop mode checks them as if real.
- R6. Restart files hold every collected record when `return_records` is on (the
  `Run.execute` default): they grow with run length.
- R7. Tower vectorized setup is not bit for bit when the porosity cap binds (no effect: the
  first step rewrites those values); docstring/test claim too strong.
- R8. `soil.layer_structure: clm4_5` with `domain.type: grid` not rejected at load.
- R9. LAI/SAI time weights one model step earlier than CLM (CLM: end of step); on 29
  February the weights overshoot (CLM's no-leap months).
- R10. Minimum plant area keeps diffuse transmittance below 1 only for `clumping_index`
  above ~0.55 (all built-in PFTs 1.0; a calibration could go below).
- R11. Docs: CO2 stamps are not mid-year (G8); plan decision 27 says CLM5 parameters, the
  runs use `clm6`. Smaller: per-variable exclusion in `gridcell_sum`, unmasked first-step
  gridcell sums, `Model.weights()` in slot order with devices > 1, a run starting in the first
  forcing year cannot start, 184 gridcells nearest-point where ESMF may interpolate.

**Checked and fine** (highlights): tridiagonal scans and their gradients against dense
solves; longwave rows and padding; the minimum plant area in every consumer and not applied
twice; decision 32 consistent with check 3; sharding specs; padded chunks; record merging;
grid setup bit for bit in the southern hemisphere, Siberia (snow, shallow bedrock), the Sahel
(C4, crops) and with other CO2; gridded derivations and units against CDEPS/CTSM; clm6.yaml
equal to the netCDF file.

**Not checkable on CPU**: end-to-end gradients (a 1-step `Model.run_chunk` gradient
segfaults, issue 16).

### 2026-10-30: review fixes (maintainer's choices)

- R1: not a use case (several towers on several devices); `load_run` refuses devices > 1
  with towers of different meteorology files.
- R2 fixed, decision 34: `time.calendar: gregorian | noleap` (noleap for grid runs), exact
  date arithmetic as CTSM's clock (`clm_time_manager.add_seconds`, `date_after`), used by
  run length, step times, output records, restarts and the gridded streams. Orbit of the
  start year (known departure, ~0.02° over 150 years).
- R3 fixed: restarts keep the chunk schedule's start (`chunk_anchor`).
- R6 fixed: restarts hold the state, the output record in progress and bookkeeping only;
  records not yet in a file are written at the restart and the resumed run appends.
- R9: timing fixed (vegetation weights at the end of the step, as CTSM's ordering); the 29
  February weights are CLM's own (28-day February in CTSM), kept and documented.
- Norman solar radiation refused at load; bare ground (issue 13) after the modernization;
  docs: G8 CO2 stamps, decision 27 (CLM6 parameters).
- Not changed (minor, recorded): R4, R5, R7, R8, R10, R11 smaller items.
- Tests: fast tier (289, new: calendars, anchor, writer across a restart, guards), all of
  `tests/regression` (206, bit for bit), Fortran comparison unchanged (max 0.068 %, the same
  2 columns below r = 0.9999). **GPU gradients (A100, `jobs/grad_gpu.pbs`, develop queue,
  ~45 min): all pass** — the differentiability tests, the per-patch weight and PFT gradients,
  and the chunk gradient against finite differences (its script had rotted since Phase 6:
  per-PFT parameters; fixed `35fbdb2`). Gradients are checked on the GPU while issue 16
  blocks them on CPU.

## Next (resume here in a new session) — updated 2026-10-30 (Phase 6b closed)

**State.** Branch `phase6/patches` (all pushed). Phase 6 and 6b done (step 7 closed as it
stands); adversarial review done and its chosen fixes in (`b45bd9a`, decision 34); GPU
gradients pass. End-of-phase CPU job (`jobs/endphase6b_dev.pbs`): slow tests 9 pass, 1 fails
(the CHATS7 chunk gradient test segfaults on CPU, issue 16; it passes on the A100); all of
`tests/regression` passed separately (206). Phase 7 scope: `docs/design/phase7_scope.md`. Decisions: plan §0 (23-34); Phase 6b
details, G1-G20 and open issues: `docs/design/global_input.md`.

**Working agreement with the maintainer**: copy CLM where a choice comes up; for a bug,
diagnose first with probes (scratch, no code changes) and bring the diagnosis before changing
code; no shortcuts; science changes are the maintainer's decision; document every decision
and issue.

**Before Phase 7 (standing rule: stop and discuss a new phase):**
1. Gradient testing tier for Phase 7: CPU gradients crash (issue 16) with LLVM optimization
   on; with `--xla_backend_optimization_level=0` all CPU gradient tests pass (8 tests, 9.7 min
   on a login node, 2026-10-30). Proposed: that CPU tier after process changes, the GPU job
   (`jobs/grad_gpu.pbs`, develop queue) at package ends (phase7_scope §7).
2. Phase 7 starting decisions (plan Phase 7): 0-based indexing or not; the §6.6 process
   interface; restarts as netCDF (include `chunk_anchor` and the writers' file in progress,
   no output history); meteorology interpolated inside the chunk (issue 7, assigned to
   Phase 7); orbital parameters per year (decision 34's known departure) could become time
   data; avoid unrolled Python loops (2.3 GB GPU programs, deferred GPU work).
3. Recorded, not blocking: review R4 (a failing patch counts in its chunk's gridcell sums),
   R5 (padding slots checked as real), R7, R8, R10 (minimum plant area vs clumping < 0.55),
   R11 smaller items; after Phase 8: issue 10, bare ground (13), GPU performance; open
   science issues 1, 3-6, 15, 17b with the maintainer.

**Probes and inputs (scratch, not in the repo)**: `/glade/derecho/scratch/linnia/clm-ml-jax/
probe/6b/` (Phase 6b), `probe/issue14/` (thin layers: `common.py` sets up the Wyoming gridcell
and the failing step, `wrap.py`/`track*.py` routine-by-routine tracking, `td_threshold.py`,
`how_common.py`, `layer_floor.py`), `probe/issue12/per_step.py`. CLM source:
`/glade/work/linnia/ctsm5.4.004` (CTSM 5.4, CDEPS); CLM-ml v2 Fortran:
`/glade/u/home/linnia/clm-ml-jax/run_model/CLM-ml_v2.CHATS`; CLM-ml v1:
`/glade/derecho/scratch/linnia/ext/CLM-ml_v1` (github.com/gbonan/CLM-ml_v1, `5c8d01d`).
Regional outputs: `runs/regional/` (Jan, Jul 2000).

**Jobs**: scripts in `/glade/derecho/scratch/linnia/clm-ml-jax/jobs/` (`regional_dev.pbs`,
`global6b.pbs`, `recheck_dev.pbs`, `run_grid.py`, plus the Phase 6 ones). GPU: `-A UCLB0069`
(`-q develop` starts quickly, 1.5 h max used; `-q main` waited ~3 h); CPU: `-A P93300041`.
`qstat -u linnia` lists jobs (develop jobs too). Pipe job output through
`grep --line-buffered` so lines are not lost, and **not through `cut`** (Derecho's `cut` buffers its output even under `stdbuf -oL`: nothing appears until the job ends); PBS writes `-o` logs only at the end, so tee live output to a scratch file.

**Other checkout**: `/glade/u/home/linnia/clm-ml-jax/run_model/clm-ml-jax` (home) is on
`phase6/patches` at an older commit with a git stash "profile_gpu WIP (1a/1b/10+2)" from the
old `profile_gpu` plan (superseded); pull before using it, or ignore it.
