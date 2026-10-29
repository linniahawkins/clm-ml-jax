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

## Next

Step 0a (regenerate the reference on Derecho), then 0b, 0c, then step 2.
