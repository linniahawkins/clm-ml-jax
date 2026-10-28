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

**How it is checked.** A harness captures the real inputs of each routine during a run and
calls both versions on them. As of Phase 3c, every routine with a `grid` branch
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
refactor. The sensitivity in §5 may be related.

## 5. The model is sensitive at the evening transition

**Evidence.** Two runs of the same code that differ only by `leaf_emissivity × (1 + 1e-12)`
agree to about 1e-13 for six half-hour steps. At step 7 (00:00–03:30 UTC, about 17:00–20:30 PST,
the evening transition to stable conditions) they differ by:

| Quantity | Difference |
|---|---|
| u* | 1.1e-3 m s⁻¹ |
| sensible heat flux | 0.12 W m⁻² |
| aerodynamic conductance | 2.5 |
| roughness-sublayer β | 2.2e-4 |

The difference then decays. A refactor that changes only round-off (moving to the traceable
branches) produced exactly the same jump at the same step.

**Over the full 50-step reference day** (twins `leaf_emissivity × (1 ± 1e-12)`), the largest
difference from the reference, relative to the field's magnitude, is:

| Field | Twin difference |
|---|---|
| momentum flux profile (`mflx_profile`) | 4.5e-2 |
| aerodynamic conductance profile (`gac_profile`) | 4.4e-2 |
| wind speed at shaded leaves / all leaves | 4.4e-2 / 3.9e-2 |
| canopy air heat storage (`stflx_air_canopy`) | 3.2e-2 |
| canopy evaporation (`evveg_canopy`) | 3.1e-2 |

35 of the 88 recorded fields do not change at all. Those are radiation, APAR, albedo and soil
water stress, which do not depend on the turbulence solution. A perturbation in the 12th
digit therefore moves turbulence-related fields by several percent at some steps. That is a
scientifically notable property of the model (and may account for part of the 2.5% tolerance
needed in the existing JAX–Fortran comparison).

**Interpretation.** At the transition, a discrete outcome flips on round-off-sized input
differences. The most likely candidates are:
- the stability bounds of the roughness-sublayer theory (−2 ≤ L_c/L ≤ 1, where the Obukhov length
  is clipped);
- the stopping tests of the Obukhov and roughness-length solvers.

The same sensitivity is expected in Fortran. It is a property of the model, not of the
translation or the refactor. Worth investigating as a science question (§4 may contribute);
not a refactoring bug.

## 6. Regression testing policy

Because of §5, **no tolerance on per-step values separates round-off changes from real
changes**. A correct change that alters round-off can move a single step by 1e-3 relative.

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

**What the noise twin cannot catch:** an error smaller than one regime flip. For turbulence
fields that means up to ~9% (2 × 4.5%) at a single step, so **the noise-twin test is a coarse
guard, not a precise one.** Precise checking comes from three other tests:
- the bit-for-bit test, for refactors;
- the per-routine identical-input checks (§1), for changes that alter round-off;
- the Fortran golden tests (`rtol = 1e-9`).

A change that alters round-off is accepted only when it passes **both** the noise-twin test and
an identical-input check of every routine it touches.
