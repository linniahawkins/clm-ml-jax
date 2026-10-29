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

## Next

Commit step 6 when `s6c` passes; record the global timing; end-of-phase tests (all of
`tests/regression`, the slow tests, Fortran comparison), CLAUDE.md update, final report.
