# Phase 7 progress log

Branch `phase7/restructure` (from `phase6/patches` at the end of Phase 6b). Scope, work packages
and the maintainer's decisions (D1-D12): `docs/design/phase7_scope.md`. Decisions Claude makes
while working: `docs/design/phase7_decisions.md`. Fields removed from the state (D6):
`docs/design/phase7_removed_diagnostics.md` (created in 7.1).

One entry per commit: what changed, which tests ran and their result, any round-off change and
its reference update. Each session ends at a commit with "Next" updated.

## How to work (agreed with the maintainer, 2026-10-30)

- **One session per work package** (7.2 split into sessions of 2-3 processes). Start from
  "Next" below; end at a commit with this log and "Next" updated. Don't rely on compaction to
  carry a package across sessions.
- **Stop at the end of each package** and report to the maintainer (tests, timing, round-off
  changes, decisions logged) before the next one. Within a package, commit and push when the
  tests below pass; stop only for science questions or a slowdown above ~1.5x.
- **Independent review** (a reviewer agent that sees the diff, not Claude's reasoning) before
  the last commit of each package.
- **Log every decision** you make that the scope does not already settle (naming, layout,
  an adapter, a test skipped, a deviation) in `phase7_decisions.md`, in the same commit.

## Testing in Phase 7 (minimum, maintainer 2026-10-30)

Phase 7 changes structure, not arithmetic, and the bit-for-bit CHATS7 test compares every
remaining state and output field after every step. So the testing is cut to:

| When | Tests | Time |
|---|---|---|
| Each commit | `black`/`ruff`; bit-for-bit + noise twin (`tests/regression/test_end_to_end_regression.py -m regression`, one run); the Fortran golden tests of the routines moved in that commit; `tests/clm_ml` only if `src/clm_ml/` config/IO code changed | ~1-2 min |
| A commit that moves a routine with a hand-written gradient (implicit-function solvers: Obukhov length, stomatal conductance bisection, any `stop_gradient` construction) | plus `test_model_api.py::test_model_chunk_gradient_matches_finite_difference` on CPU with optimization off (command below) | ~5 min |
| End of a package | the CPU gradient tier (below), and the feature tests the package touched (7.3: `test_coarse_forcing`, `test_land_data_files`, `test_grid_setup`, `test_stream_runner`; 7.4: `test_vector_setup`, `test_grid_setup`, `test_patches_batched`, `test_c3_c4_groups`, `test_sites_one_program`; 7.5: `test_restart`) | ~10-20 min |
| End of Phase 7 | all of `tests/regression`, the slow tests, the Fortran comparison, and **one** GPU gradient job (`jobs/grad_gpu.pbs`, develop queue) | ~1.5 h |

Not run per commit any more: the fast tier after every edit, `test_clm_ml_equivalence`, the
other feature tests, GPU jobs. Accepted risk: a break in a feature not tested until the package
end is found later; commits stay small (one process each) so `git bisect` finds it quickly.

CPU gradient tier (issue 16: the default CPU build crashes running gradients; with LLVM
optimization off it runs; round-off differs, so never use this flag for the regression tests):

```bash
JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_parallel_codegen_split_count=1 --xla_backend_optimization_level=0" \
  python -m pytest tests/test_model_api.py::test_model_chunk_gradient_matches_finite_difference \
  tests/test_differentiability.py -m slow -p no:cacheprovider -o addopts="" -q   # 8 tests, ~10 min
```

## Log

(entries start with 7.0)

## Next (resume here)

**Start package 7.0 (preparation)**, scope §4:
1. Delete the dead modules listed in scope §2; remove `controlMod` writes and `clm_instRest`.
2. The legacy-view adapter (new state -> reference field names and 1-based shapes) and a test
   that it reproduces today's reference from today's containers exactly.
3. A pytest marker or small script for the CPU gradient tier (command above) and the GPU
   gradient job with a warm cache.
4. CLAUDE.md: remove gotchas as they stop holding.
Then stop and report to the maintainer.
