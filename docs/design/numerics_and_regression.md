# Iterative solvers, differentiability, and regression testing

Decisions made during Phase 3 (making the physics traceable for `jax.grad`, and merging the
"normal" and "differentiable" code paths). Each states the decision, why, and the evidence.
Fortran references are to CLM-ml v2 (`gbonan/CLM-ml_v2.CHATS`).

---

## 1. One implementation per process, using the Fortran algorithm

**Context.** Before Phase 3, several physics routines existed twice: a "normal" version
(Python floats and loops, used for runs) and a traceable `_jax` / `grid`-branch version (used
only by `make_clm_ml_forward` for gradients). The normal versions were validated against
Fortran; the traceable ones were not, and some had drifted.

**Decision.** Each process gets **one implementation**: the Fortran algorithm, written so JAX can
trace it. A traceable copy is accepted only when it agrees with the Fortran-matching version
**on identical input** to round-off (about 1e-15 relative). Then the duplicate is deleted.

**How it is checked.** `tests/regression/process_io.py --record` captures the real inputs and
outputs of each such routine at dusk, night and midday of the CHATS7 reference day, from the
code before the merge (`reference/process_io.pkl.gz`, 2.4 MB). `test_process_io.py` replays
the inputs through the current routines and requires every output field to match within 1e-12
relative. (The recording must be loaded *after* the model is initialized: unpickling imports
model modules, which copy run options such as `met_type` at import.) During the merge, a
scratch harness also called both versions on the same captured inputs. As of Phase 3c, every routine with a `grid` branch
(`GetAtmForcing`, `SolarRadiation`, `LongwaveRadiation`, `CanopyTurbulence`,
`LeafPhotosynthesis`, `FluxProfileSolution`, `RungeKuttaUpdate`) agrees to within 7e-16 on every
captured call of a 2-step run, after the two fixes in §2 and §3.

## 2. Root solvers: the forward value is the solver's answer; gradients use the implicit function theorem

**Context.** Three quantities are found by iterative root solvers: the Obukhov length
(`_GetObu`), leaf intercellular CO₂ (`_ci_solver_scan_ift`) and stomatal conductance under
water-use-efficiency optimization (`_bisect_gs_ift`). Differentiating *through* the iterations
gives wrong or exploding gradients (observed ~1e144 for the Obukhov solver), so the traceable
code uses the implicit function theorem (IFT). If the solver finds `x*` with `F(x*; θ) = 0`, then
`dx*/dθ = −(∂F/∂θ) / (∂F/∂x)`.

**Problem found.** The IFT was implemented as one Newton step,
`x = x0 − F(x0)/stop_gradient(F′(x0))`. That gives the right gradient, but **it also changes the
forward value** whenever the solver is not fully converged (`F(x0) ≠ 0`). The traceable path
therefore computed slightly different physics from the model that runs (Obukhov length, CO₂,
stomatal conductance), and gradients described that slightly different model.

**Decision.** Apply the correction to the gradient only:

```python
step = F(x0) / stop_gradient(F'(x0))          # zero where the IFT is ill-conditioned
x = x0 - (step - stop_gradient(step))         # forward: exactly x0; gradient: the IFT
```

The forward value is exactly the solver's root, as in the normal path and Fortran. The
gradient is unchanged.

## 3. Roughness length: the Fortran bisection stopping rule

**Problem found.** Fortran `RoughnessLength` finds `z0m` by bisection on `[0, h_top]`, stopping
when the bracket is narrower than `err = 0.001 m` (at most `nmax = 20` iterations), and returns
the last midpoint. The normal Python version matches this. The traceable version always ran 20
iterations and returned the midpoint of the final bracket. That is more refined, but it is not
what CLM-ml v2 computes: `z0m` differed by up to 4e-4 relative.

**Decision.** The traceable version uses the same stopping rule. `lax.while_loop` cannot be
reverse-differentiated, so it is written as a fixed 20-iteration `fori_loop` whose updates stop
once `|b − a| ≤ err`. The iterations and result are identical to Fortran's, and it remains
differentiable. (The derivative of a bisection result is zero almost everywhere. This is the
Fortran algorithm's property, not something introduced here.)

## 4. Known deviation from Fortran: Obukhov-length solver (not changed in Phase 3)

Fortran `GetObu` (MLCanopyTurbulenceMod.F90, lines 217–260) calls the hybrid solver
`MLMathToolsMod::hybrid`:
- secant iterations from two guesses, `obu0 = 100 m` and `obu1 = −100 m`;
- stops when the step is smaller than `tol = 0.1 m`;
- switches to Brent's method (`zbrent`) if a sign change brackets the root;
- at most 40 iterations.

The model state is left as computed by the *last* function evaluation (the returned root is a
dummy).

The JAX translation, in both its normal and traceable paths, instead runs **exactly 25 secant
steps** from a single start (`_obu_fixed_iter`) and then recomputes the derived quantities from
that value. This predates the modernization. It is within the validated agreement with Fortran
(CHATS7 May 2007: ≤ 2.5% per output column, r ≥ 0.9999), but it is not the Fortran algorithm.

**Status:** recorded, not changed. Matching Fortran here would change results, so it is a
deliberate fidelity change with its own validation (plan Phase 4b or later), not part of a
refactor.

**Bug fixed (Phase 3).** The fixed-step secant iteration took a huge step when two consecutive
residuals were exactly equal after convergence; it now takes no step (§5).

## 5. The apparent evening-transition sensitivity was a solver bug (fixed)

**What was seen.** Two runs differing only by `leaf_emissivity × (1 ± 1e-12)` (the noise twins)
differed by up to 4.5% in turbulence fields (`mflx_profile`, `gac_profile`, wind at the leaves,
u*), with jumps at single steps (largest at step 7, the evening transition). A refactor that
changed only round-off produced jumps of the same kind. This was first read as a property of
the model.

**Cause.** The secant iteration of the Obukhov-length solver (`_obu_body_pure`; the same code is
in the intercellular-CO₂ solver `_ci_solver_scan`) runs a fixed number of steps and does not stop
once converged. After about five steps the residual is at round-off level (±1e-14). When two
consecutive residuals come out *exactly equal*, the secant denominator `f1 − f0` is zero, and the
code divided by a floor of 1e-30 instead:

```
dx = −f1 · (x1 − x0) / max(|f1 − f0|, 1e-30)  ≈  (1e-14 · 1e-14) / 1e-30  ≈  100 m
```

The converged iterate was thrown far away (in one case from L = 73.6 m, stable, to −128 m,
unstable). The secant then pulled it back over the following steps, in a cycle, and the value
returned after step 25 depended on where in that cycle the iteration ended. Whether this
happened depended on round-off in the last digits, hence the "sensitivity".

**Fix.** When `f1 == f0` the secant step is undefined, so no step is taken
(`dx = where(f1 == f0, 0, …)`). Converged values are unchanged; Fortran's `hybrid`, which stops
when the step is below 0.1 m, never does this.

**Effect on the reference day** (CHATS7, 50 steps):
- The old reference itself was distorted by the bug at several steps: the fix changes it by up
  to 2.4% (relative to the field's magnitude) at step 32, 0.8% at step 7 and 0.4% at steps 5 and
  23, and by ≤ 1e-4 elsewhere.
- **The twins now differ from the reference by at most 4e-11** (any field, any step), against
  4.5e-2 before. The model is well conditioned: a perturbation in the 12th digit stays in the
  11th digit.

| Field | Twin difference before | after |
|---|---|---|
| momentum flux profile (`mflx_profile`) | 4.5e-2 | 7.9e-13 |
| aerodynamic conductance profile (`gac_profile`) | 4.4e-2 | 3.8e-12 |
| wind speed at shaded leaves | 4.4e-2 | 2.6e-12 |
| canopy air heat storage (`stflx_air_canopy`) | 3.2e-2 | 3.1e-12 |
| u* | 3.1e-2 | 7.7e-13 |
| sensible heat flux | 1.2e-2 | 3.0e-11 |

The bug may account for part of the 2.5% tolerance needed in the JAX–Fortran comparison of the
full month (`tests/test_fortran_python_comparison.py`, not yet re-run with the fix).

## 6. Regression testing policy

A change that alters only round-off (compiling code, a different order of operations, GPU)
cannot reproduce the reference bit for bit, so a tolerance is needed. The model's own
sensitivity to round-off, measured by noise twins, sets it.

The policy has two checks:

1. **Bit-for-bit** (`test_bit_identical_to_reference`). Refactors that should not change the
   arithmetic (moving code, explicit arguments instead of globals) must reproduce the reference
   exactly. All of Phases 0–3b met this.
2. **Noise twin** (`test_within_noise_of_reference`). For changes that legitimately alter
   round-off (a traceable implementation, whole-run compilation, GPU), the comparison is against
   the model's own sensitivity. The reference is stored with **twin runs** that differ only by
   a 1e-12 relative perturbation (`leaf_emissivity × (1 ± 1e-12)`). For each output field, the
   noise is the largest difference between the reference and either twin over all steps. A
   change passes if, for every field, its largest difference from the reference is at most
   **2 × the noise** (plus 1e-12 × the field's magnitude, for fields the twins leave unchanged).

**Workflow for a round-off-changing step:**
1. Run the regression tests. The noise-twin test must pass. The bit-for-bit test is expected to
   fail.
2. Regenerate the reference and twins with `python tests/regression/record_run.py
   --update-reference`. Say in the commit that the step changes round-off only, and why.
3. The bit-for-bit test then guards the next refactors again.

**How tight it is.** Since the solver fix (§5), the twin noise is at most about 4e-11 relative,
so the noise-twin test detects any change larger than about 1e-10. (Before the fix it allowed
up to ~9% at a single step.) The other checks remain:
- the bit-for-bit test, for refactors;
- the per-routine identical-input checks (§1), for changes that alter round-off;
- the Fortran golden tests (`rtol = 1e-9`).

A change that alters round-off is accepted only when it passes **both** the noise-twin test and
an identical-input check of every routine it touches.
