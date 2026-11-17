# Gradient experiments at CHATS

When can the model's gradients be trusted, and when are they useful? A set of experiments at
the CHATS walnut orchard (branch `chats_demos`; open item O82, decision 126), after the
maintainer's note "Diagnosing gradient failure in a JAX land model" (2026-11-17). A gradient
can fail in four ways, each needing its own test:

1. **wrong**: a bug, a NaN;
2. **right but uninformative**: zero or piecewise constant, behind a threshold;
3. **right but ill-conditioned**: exploding or vanishing over long runs;
4. **right at a point but misleading at the scale of interest**: a rough loss surface, whose
   local slope is not the trend across the parameter's range.

Earlier work (`docs/design/gradient_progress.md`, rounds 2, 3 and 5) checked failure 1 at
CHATS7 for one clear day (1 May 2007). This work covers the month and failures 2 to 4.

## What is here

| | |
|---|---|
| `may2007/` | experiment folder: May 2007 (31 days), the model's defaults, one-day chunks. The case of every experiment unless said. |
| `aprmay2007/` | the same from 1 April (61 days): the long-run tests only. |
| `harness.py` | the one harness (step 1): a window of days in, a table of derivatives out. |
| `README.md` | this plan and its status. |

The two folders are ordinary experiment folders (`clm-ml check experiments/chats_gradients/may2007`;
written by `clm-ml new ... --from chats7_1day`, then the dates, the forcing files, the chunk
and the output directory edited). **Results are not in the repository**: the table, the
figures and any model output go to `$SCRATCH/clm-ml-jax/chats_gradients/`.

## Choices (maintainer, 2026-11-17; decision 126)

- **Loss: a synthetic twin.** The "observations" are the model itself run with other
  parameters (each within ±5 % of its default, a fixed random draw), so the right answer
  of a calibration is known. The loss is the mean squared misfit of latent heat, sensible
  heat and GPP, each divided by its variance over the window.
- **Parameter ranges: ±10 % of the default value**, for every parameter.
- **No emulator comparison.** In its place a one-at-a-time experiment: each parameter at
  the minimum and at the maximum of its range, compared with what the derivative predicts
  (derivative × range).
- **May 2007**; April and May together for the long-run tests.
- Not done, because the model at CHATS has nothing of the kind: snow, phenology and other
  discrete events, carbon pools, decades, spin-up (soil water is prescribed).

## The harness

```bash
JAX_PLATFORMS=cpu python experiments/chats_gradients/harness.py --case may2007 --first 0 --days 1 \
    --label "1 May, clear" [--set model.numerics.wetted_fraction=explicit]
```

(with the model's environment active; a tower belongs on the CPU). For a window of days it
appends to `table.csv`, one row per parameter and quantity (the loss and the window mean
of seven outputs):

- the derivative by automatic differentiation, in forward and in reverse mode;
- central finite differences at seven relative steps, 1e-2 to 1e-8 (the "V" of error
  against step size);
- the quantity with the parameter at its minimum and at its maximum (one at a time);
- NaN flags, the size of the gradient with respect to the state at the window's start,
  and the window, its regime label and the settings.

In a script: `Case("may2007", overrides).window(first, days)` has `quantities`,
`forward_mode`, `reverse_mode` (parameters and initial state), `finite_differences`,
`one_at_a_time` and `outputs` (every step). One day takes about 6 minutes the first time
(compilation and 27 parameters × 7 steps × 2 runs), less afterwards.

## Work plan

| Step | What | Status |
|---|---|---|
| 1 | The harness, the two experiment folders | **done** (smoke test below) |
| 2 | **Baseline, all of May.** Every day through the harness; the dot-product test on the one-step state map; NaN over every input. Every step labelled by regime: stable night, the two transitions, unstable afternoon, `L_c/L` at its limits, water stress active, stomata at `gs_min`, Rubisco- or light-limited, wet canopy (May has 20 half-hours of rain) | next |
| 3 | **Threshold sweeps.** One input pushed through a threshold, output and derivative plotted together: soil water from wilting to saturation, LAI towards 0, shortwave through sunrise, wind through the 0.5 m s⁻¹ minimum, air temperature so that L crosses its limits, intercepted water towards dry, leaf water potential through ψ₅₀. With it a convergence audit of the Obukhov and stomatal solvers over the month (their gradients come from the solved equation, right only where the solver converged) | |
| 4 | **Parameters.** One at a time, minimum and maximum, against derivative × range (1 day, 31 days). A dense sweep of each parameter across its range (~200 points): loss, its derivative, and the slope smoothed over 0.1, 1 and 10 % of the range (plateaus, roughness, staircases). The Hessian of the loss at the default and its eigenvalues | |
| 5 | **Time.** Horizon sweep from one step to 31 days (61 with `aprmay2007`), for parameters and the initial state: gradient size and agreement with finite differences against the horizon. The one-step Jacobian (396 × 396) through a day, with growth rates per group of state variables. The spread of the gradient over 31 one-day and 6 five-day windows. Checkpoints on and off give the same gradient | |
| 6 | **Remedies that are already settings** (no code change): steps 2 and 5 with `wetted_fraction: explicit` against `implicit`, and `water_stress_derivative` and `obukhov_derivative` `exact` against `stabilized`: what each costs in bias and gains in gradient quality (decisions 122, 124) | |
| 7 | Only after the results are discussed: a soft threshold as a scratch prototype for whichever threshold step 3 shows to matter; a twin calibration with full-horizon against windowed gradients | |

## Status and first results

**2026-11-17: step 1.** Smoke test, one day from 1 May (`smoke.csv` in the results
directory; the window is steps 1 to 48, the first step of a run being taken at setup):

- Forward and reverse mode agree to 8 digits, also chained over two days and over the
  short last chunk of the month.
- No NaN in any derivative. The four Medlyn and Ball-Berry parameters have zero derivatives,
  as they should (the run uses the water-use-efficiency optimization).
- Derivatives of the loss agree with the best finite difference within 0.4 % for most
  parameters. `psi50_gs` and `plant_conductance` are 2.3-2.5 % off finite differences that
  are steady across their steps: expected, the default `water_stress_derivative: stabilized`
  is by design not the exact derivative (decision 124); step 6 measures it against `exact`.
  Finite differences of several parameters are erratic at a relative step of 1e-4 while
  steady at 1e-6 and below: for step 2 to look at.
- **Found: `leaf_emissivity` = 1 makes every output of the day NaN** (the top of its range,
  capped at the physical limit; 0.999999 runs). A forward-model matter, not a gradient
  one: open item O83, not diagnosed, nothing changed.
