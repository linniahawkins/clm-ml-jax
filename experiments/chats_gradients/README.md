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

Steps 1 to 5 of the plan below were run on 2026-11-17 with the model as it is (no change
to `src/`). Of steps 6 and 7 (remedies), only the twin calibration of step 7 was run, at
the maintainer's request (2026-11-17).

## Resume here (state on 2026-11-17, afternoon)

- **Done and committed:** steps 1-5, the default-against-exact comparison (every day of May
  and the month run again with `water_stress_derivative: exact`), 45 figures in `figures/`,
  the results below.
- **Done: the twin calibration (step 7)**, job 8214770 (eight runs of `s7_calibrate.py`,
  all 80 iterations, finished 09:52): results in "Step 7" below and result 8, figures
  `calibration_progress.png` and `calibration_parameters.png`; the runs' files are
  `$SCRATCH/clm-ml-jax/chats_gradients/calibration/<params>_<gradient>_s<seed>.npz`.
- **Not run:** step 6 (the remedies that are settings: `wetted_fraction`,
  `obukhov_derivative`) and the soft-threshold prototype of step 7.
- **Open questions left by the results** (none diagnosed): O84 (round-off growth under
  water stress: a discrete defect or the scheme?), O85 (zero derivative at dry leaves),
  O83 (`leaf_emissivity` = 1), forward against reverse mode differing by a factor 1.44 in
  the month's exact gradient, and the default gradient's wrong trend for
  `plant_capacitance`.
- **Ideas discussed with the maintainer, not run:** the exact gradient drawn across the
  31-day sweeps (23 month-long sweeps, about 1.5 hours); the Hessian with the misfit on
  daily and on monthly means, to see how many parameter directions each constrains.
- **Elsewhere, same day:** the maintainer's four 1-year global runs with other Vcmax
  (`$SCRATCH/grad_tests`, cases `vcmax_default`, `vcmax_x1.2`, `vcmax_x1.01`,
  `vcmax_x0.99`) stopped at 2 March on an output-writer bug (O86), fixed on branch
  `bug_fix` (from `main`, commit be91435, pushed, no pull request); the checkout the runs
  use (`/glade/work/linnia/clm-ml-jax`, branch `grad_tests`) was fast-forwarded to it and
  the cases resubmitted from 1 January (default and x1.2: jobs 8215089, 8215090; x1.01 and
  x0.99 by a background task once their old jobs ended: check `qstat -u linnia` and each
  case's `run.log`). Each failed run's log is `run_failed_20261117.log` in its folder.
- **How to run anything here:** "How to run" below; jobs go through `run.pbs` (a GPU node
  of the develop queue used for its CPUs: the account has no CPU hours; `NPAR=32` and
  `CORES=n` give 32/n commands at a time).

## Results in short

1. **The machinery is right.** Over the 31 days of May: no NaN in any derivative (parameters,
   start state, forcing fields); forward and reverse mode agree; the dot-product test on the
   one-step map of the state passes (1e-8); the checkpoints change nothing (1e-10). With
   `numerics.water_stress_derivative: exact`, automatic differentiation (AD) equals finite
   differences to about 1e-6 on the four days tried, the water-stressed ones included.
2. **The default gradient is not the model's derivative on water-stressed afternoons, by
   design, and that is most of May.** The default setting `water_stress_derivative:
   stabilized` (decision 124) agrees with finite differences at night, around sunrise and
   sunset, with a wet canopy and in unstressed daylight (97-100 % of the derivatives within
   1 %), but only 40 % are within 1 % in the steps where the top leaves are below about
   -1.43 MPa. That happens on 28 of the 31 days (below ψ50 = -1.6 MPa on 24). The one-day loss gradient is within 1 % of
   finite differences for a quarter of the parameter-days (median difference 7.5 %).
3. **The exact derivative there is erratic, because the forward model is.** In stressed
   afternoons the exact derivative jumps from step to step by a factor of ten and changes
   sign (`figures/stabilized_vs_exact_day08.png`); the exact one-step Jacobian has growth
   factors of 12 (9 May) to 8700 (21 May) for single steps; two runs that start a rounding
   error apart (1e-12) drift to 6e-3 W m-2 in latent heat on 22 May and come back
   (`figures/twins.png`). The stabilized derivative is the smooth curve through this.
4. **Over the month the loss is rough at small scale.** Across ±10 % of a parameter the
   31-day loss is a clean bowl for the parameters that matter (ι, Vcmax, ψ50, plant
   conductance, near-infrared leaf optics) with ripples of about 1e-5 on top; for the weak
   parameters (root radius and density, SLA, stem transmittance, minimum conductance) the
   ripples are the whole signal, with 20-30 local minima. Finite differences with small
   steps measure the ripples, not the trend. The default AD gradient follows the trend
   (slope over a tenth of the range) for the strong parameters and points to the minimum
   over 92-99 % of the range; it has the wrong trend for `plant_capacitance` and is offset
   for `shape_gs` (`figures/sweep_slopes_31d.png`).
5. **Thresholds give zero derivatives where expected, and one wrong one.** Wind below the
   0.5 m s-1 minimum: exactly zero. Friction velocity on a stable night at the limit of
   L_c/L: flat, zero. Dry leaves: the derivative with respect to intercepted water is zero
   at exactly zero water although the slope just above is large (10⁴ W m-2 per kg m-2);
   so the start state's intercepted water looks like it lasts for ever on a dry day
   (gain 1 after a day) when in truth it evaporates within the hour.
6. **Time.** Nothing explodes or vanishes in the default gradient over 61 days. The
   canopy's state is forgotten within a day (factors 1e-5 to 1e-6); soil temperature is
   the memory (deep soil: 0.99999 per day; the response of latent heat to deep soil
   temperature still grows after two months). The gradient's sign is the same in all 31
   one-day windows for 17 of 23 parameters, but its size varies from day to day (the
   interquartile range is 1-2 times the median; dry against wet days); five-day windows are much steadier and close to
   the month.
7. **Conditioning.** The Hessian's eigenvalues fall by four orders of magnitude over the
   first ten directions and its condition number is 1e13 (Gauss-Newton, month). The stiffest direction mixes ψ50, ι and the near-infrared leaf
   optics; the sloppiest are stem transmittance and the root parameters (root radius and
   density are one direction). Differencing AD gradients gives an asymmetric Hessian
   with negative eigenvalues: the gradient is itself not smooth at the 0.1-1 % scale.

8. **A twin calibration works with the default gradient and fails with the exact one.**
   Fitting a month of daily means (twins with parameters moved by up to ±5 %), the default
   gradient lowers the loss 30-500 times and recovers the five strong parameters to
   0.4-0.7 % on average; with all 23 parameters the loss falls as far but the parameters
   barely get closer to the truth (the daily means do not constrain the weak ones). With
   the exact gradient the loss ends higher than it started or barely lower: its gradient
   reaches 1e4 to 1e39 on single iterations (step 7).

In the four failure classes: (1) none found; (2) at the wind minimum, the stable limit and
dry leaves (the last one misleading); (3) not in the default gradient, yes in the exact
one under water stress; (4) the main finding: under water stress the local derivative is
not the useful one, and the default trades exactness for a usable trend.

## What is here

| | |
|---|---|
| `may2007/` | experiment folder: May 2007 (31 days), the model's defaults, one-day chunks. The case of every experiment unless said. |
| `aprmay2007/` | the same from 1 April (61 days): the long-run tests only. The orchard is leafless until 10 April. |
| `harness.py` | step 1 and 2: a window of days in, a table of derivatives out (and every step's derivatives in a file). |
| `s3_thresholds.py` | step 3: threshold sweeps. |
| `s4_parameters.py` | step 4: parameter sweeps across the range; the Hessian. |
| `s5_time.py` | step 5: horizon sweep, one-step Jacobian, checkpoints, round-off twins. |
| `s7_calibrate.py` | step 7: a twin calibration against a month of daily means (Adam; default or exact gradient). |
| `launch.py`, `run.pbs` | run a file of commands on one node, several at a time. |
| `plots.py` | every figure in `figures/` and the numbers quoted here (`summary.txt` in the results directory). |
| `figures/` | the figures (committed). |

The two folders are ordinary experiment folders (`clm-ml check experiments/chats_gradients/may2007`).
**Data are not in the repository**: the table, the arrays and the logs are in
`$SCRATCH/clm-ml-jax/chats_gradients/` (`table.csv`, `baseline/`, `month/`, `window5/`,
`exact_water_stress/`, `thresholds/`, `sweeps/`, `hessian/`, `horizon/`, `jacobian/`,
`twin/`, `checkpoint.npz`, `logs/`, `tasks/`).

## Choices (maintainer, 2026-11-17; decision 126)

- **Loss: a synthetic twin.** The "observations" are the model itself run with other
  parameters (each within ±5 % of its default, a fixed random draw), so the right answer
  of a calibration is known. The loss is the mean squared misfit of latent heat, sensible
  heat and GPP, each divided by its variance over the window.
- **Parameter ranges: ±10 % of the default value**, for every parameter; `leaf_emissivity`
  stops at 0.999 (1 gives NaN, O83) and `clumping_index` at 1.
- **No emulator comparison.** In its place a one-at-a-time experiment: each parameter at
  the minimum and at the maximum of its range, compared with derivative × range.
- **May 2007**; April and May together for the long-run tests.
- **No remedies** (steps 6 and 7) and no change to the model's code.
- Not done, because the model at CHATS has nothing of the kind: snow, phenology and other
  discrete events, carbon pools, decades, spin-up (soil water is prescribed).

23 parameters are tested. The four Medlyn and Ball-Berry parameters have zero derivatives,
as they should (the run uses the water-use-efficiency optimization); `zeta_max_stable` is
"no limit" for trees.

## How to run

```bash
# one window through the harness (about 6 minutes for a day)
JAX_PLATFORMS=cpu python harness.py --case may2007 --first 0 --days 1 --label "1 May" \
    [--experiment NAME] [--set model.numerics.water_stress_derivative=exact]
JAX_PLATFORMS=cpu python s3_thresholds.py --day 0
JAX_PLATFORMS=cpu python s4_parameters.py sweep --param iota --days 31 --points 101
JAX_PLATFORMS=cpu python s4_parameters.py hessian --days 1
JAX_PLATFORMS=cpu python s5_time.py horizon | jacobian --day 20 | checkpoint | twin --day 20 --days 4
# everything: a file with one command per line, 32 at a time on one node (3 hours)
qsub -v TASKS=/path/tasks.txt,TAG=name run.pbs
python plots.py [--only baseline,thresholds,parameters,time]
```

The task files of the runs reported here are in the results directory (`tasks/all.txt`:
91 commands; `exact.txt`, `diag.txt`). Throughput: a one-day window with seven
finite-difference steps takes 5-7 minutes on one core, the 61-day horizon sweep 1.9 hours,
the 31-day Hessian 3 hours.

## Results by step

### Step 2. Baseline, every day of May

- 31 one-day windows, 23 parameters, the loss and the mean of seven outputs: no NaN in any
  row, in the gradient with respect to the start state, or in the gradient with respect to
  any forcing field.
- Forward against reverse mode: equal to 1e-7 (worst 3e-5 for a derivative that is nearly
  zero).
- AD against finite differences (FD), by regime of the time step (share of
  parameter-output pairs whose AD is within 1 % of the closest FD;
  `figures/baseline_regimes.png`):

  | regime | steps | within 0.1 % | within 1 % |
  |---|---|---|---|
  | night | 450 | 97 % | 99 % |
  | sunrise ±1 h | 91 | 98 % | 100 % |
  | day, no water stress | 281 | 91 % | 97 % |
  | day, water stress (top leaf below -1.43 MPa) | 412 | 18 % | 40 % |
  | sunset ±1 h | 122 | 92 % | 98 % |
  | wet canopy | 131 | 98 % | 99 % |

- The loss gradient per day and parameter (`figures/baseline_loss_agreement.png`): 1-4 May
  (rain on the 2nd and 4th, little stress) within 0.3 % (median); from 5 May daily medians
  of 0.3 to 150 %.
- The finite differences are not at fault: their error does not fall with the step
  (`figures/baseline_vcurve.png`), and with `water_stress_derivative: exact` the same days
  agree to 1e-6 (9, 10 and 31 May: default median 9 %, 44 % and 3 %, worst parameter over
  100 %; exact 2e-6, 5e-6, 4e-7). `figures/stabilized_vs_exact_day08.png` shows the three
  at every step of 9 May.
- One at a time (`figures/oat_1day.png`, `figures/oat_month.png`): the largest effects of
  ±10 % are 8 W m-2 in mean latent heat (ι, ψ50, Vcmax, plant conductance) and 1 µmol m-2
  s-1 in GPP (Vcmax). Derivative × range predicts the difference between the maximum and
  the minimum within 5 % for GPP; for the heat fluxes it is off by up to 24 % (month) for
  ψ50 (a curved response and the stabilized derivative together).
  The green diamonds are the same with `water_stress_derivative: exact` (every day and the
  month run again with that setting). Over one day the exact derivative is the better
  predictor (median of 31 days: off by 0.4 % of the largest effect in latent heat against
  0.9 % for the default; ψ50 and plant conductance sit on the model's difference). Over
  the month it is useless: thousands of W m-2 for an effect of 8, wrong signs (the numbers
  at the arrows), the growth without bound that decision 124 describes. There its forward
  and reverse mode also differ by a constant factor of 1.44 for every parameter (not
  diagnosed; one exploding direction whose size depends on round-off would do that).

### The default against the exact derivative (four figures)

Every day of May and the month were run a second time with
`numerics.water_stress_derivative: exact` (forward values are the same).

- `figures/compare_1_two_days.png`: every step of a day without stress (3 May) and of one
  with (9 May). Without stress the two derivatives and finite differences are one line.
  Under stress the exact one is the model's slope, with spikes of 10 times the level and
  changes of sign; the default runs through them.
- `figures/compare_2_regimes.png`: share of step derivatives within 1 % of finite
  differences. Exact: 99-100 % in every regime. Default: 97-100 %, but 40 % under stress.
- `figures/exact_vs_fd_scatter.png`: the exact derivative against finite differences at
  every step of May (31 one-day windows, 1487 steps), for latent heat, sensible heat, GPP
  and canopy-top air temperature against ι, Vcmax25, ψ50 and plant conductance, with one
  fixed relative step (1e-6) and a per-point test (within 1 % of the larger of the two,
  or both below a thousandth of the typical size). Without stress: 100 % agree for any
  step from 1e-5 to 1e-7. Under stress the agreement depends on the step: 41 % at 1e-2,
  78 % at 1e-4, 93 % at 1e-6, 98 % at 1e-8. The exact derivative is the model's slope,
  but in stressed afternoons that slope holds only for ever smaller changes; for a 1 %
  change the model responds like the default derivative (21 May, 17:30 UTC, latent heat
  against ι: exact 1030, FD 0.1 at 1e-2, -82 at 1e-5, 1030 at 1e-7; default -0.25). The
  disagreements are almost all on the afternoons of 21, 22, 29 and 30 May, where the
  exact derivative reaches 1e7-1e9 × range and the finite differences jump with the step
  even at 1e-8. (`compare_2_regimes.png` takes, for each derivative, the closest of the
  seven steps and measures against the day's largest derivative, which on those days is
  huge: it overstates the exact derivative's agreement under stress, 99 %; the scatter's
  numbers are the fair ones.)
- `figures/exact_vs_fd_two_days.png`: the same scatter for one day without stress (3 May;
  1 May has eight stressed steps, 2 May rain) and one with (9 May), the four parameters in
  one panel. With the step 1e-6 every point is on the line on both days (largest
  difference 3e-5 of 3.5 in canopy-top air temperature against ψ50 on 9 May); on 9 May the
  derivatives are 10-20 times those of 3 May, and ψ50 and plant conductance, near zero
  without stress, are as large as ι and Vcmax25.
- `figures/compare_3_daily_prediction.png`: for each one-day window, derivative × range
  against the model's change of mean latent heat over ±10 %. On the four unstressed days
  both predict it. On stressed days the default stays close (worst day 20-37 % of the
  largest effect, right sign on 31 of 31 days for the six parameters shown); the exact one
  is as good on most days and far off on some (13, 21, 22, 30 May: off the axis; 21 May:
  1e7 W m-2; wrong sign on 2 to 9 days of 31).
- `figures/compare_4_month.png`: from 1 May onwards. The exact derivative at a step grows
  from one stressed day to the next (up to 1e6 times the default's on 21-22 May); the
  derivative of the running mean follows the model until the first stressed days, steps
  away on 13 May and is lost from 21 May. The default follows the model's change through
  the month (ψ50: 30 % too large at the end).

So: exact is the right choice where there is no water stress and for checking the code
against finite differences; the default is the usable one for windows that contain
stressed afternoons, which in May is nearly all of them.

### Step 3. Threshold sweeps

`figures/threshold_<sweep>_d00.png` (1 May) and `_d20.png` (21 May): the outputs on top,
below them AD against the slope of the curve itself (401 points).

- **Soil water** (onset of water stress near 60-80 % of the prescribed water): a steep
  transition; the curve has small wiggles there that AD does not follow (the stabilized
  derivative again); below the transition the fluxes are nearly flat, small derivative.
- **Leaf water potential** through ψ50: AD has the right sign and about a third of the
  size of the curve's slope below -2.2 MPa.
- **LAI** towards zero, **shortwave** through zero, **air temperature** at noon: smooth,
  AD on the curve.
- **Wind**: exactly zero derivative below 0.5 m s-1 (the forcing minimum; 15 % of the
  sweep), a step at 0.5; small steps in the derivative on stable nights, AD on the curve.
- **Air temperature at night**: friction velocity is flat with a zero derivative once the
  air is more than 1.5 K warmer (the stable limit of L_c/L); a sharp peak in the derivative
  at -4.4 K where the surface layer turns unstable (largest value 170 × the median).
- **Intercepted water**: above about 0.004 kg m-2 per layer the water drips, zero
  derivative (right). At exactly zero the derivative is zero, while just above it is the
  largest of the sweep: the dry-leaf branch hides the response.

Not done: the convergence audit of the Obukhov and stomatal solvers. It needs iteration
counts out of the compiled code, which is a code change. In its place: AD against FD at
every step (above), which a non-converged solver would show.

### Step 4. Parameters across their range

- One-day loss (1 May, 401 points; `figures/sweep_loss_1d.png`, `sweep_slopes_1d.png`):
  smooth bowls, AD on the slope everywhere but isolated kinks (root radius and density,
  SLA, leaf dimension, each at one or a few points).
- 31-day loss (101 points; `figures/sweep_loss_31d.png`, `sweep_slopes_31d.png`): see
  result 4 above. Share of the range where the AD gradient points towards the minimum of
  the curve: 98 % (ι, Vcmax, near-infrared leaf optics), 94 % (ψ50), 92 % (plant
  conductance), 2 % (`plant_capacitance`), 59 % (visible stem transmittance).
- Hessian (`figures/hessian_1d.png`, `hessian_31d.png`): result 7 above. Month,
  Gauss-Newton: eigenvalues from 9e-2 to 9e-15 (per range squared).

### Step 5. Time

- Horizon from 1 May (`figures/horizon_may2007_00_31.png`): the loss gradient settles
  after a day or two and changes by less than a factor of 2 from 10 to 31 days. Median
  difference from FD: below 1 % up to two days, 10-40 % from 10 days (water stress).
- Horizon from 1 April (`figures/horizon_aprmay2007_61.png`, `horizon_state_aprmay2007_61.png`):
  nothing depends on the parameters before leaf-out (10 April); AD equals FD to 1e-5 until
  the first stressed days, then as in May. No NaN in forward mode over 2927 steps.
- One-step Jacobian (`figures/jacobian_day_XX.png`, and `_exact` for the exact setting):
  default setting, largest one-step amplification 4-5, largest eigenvalue 1.00-1.01; a
  perturbation of the canopy state is damped by 1e-5 or more over a day; 21 of 395
  directions last (soil temperature, CO2, the acclimation temperature, intercepted water).
- Spread over windows (`figures/windows_spread.png`): result 6 above.
- Checkpoints: step and sub-step checkpoints on or off change the gradient by less than 3e-10.

### Step 7. A twin calibration

The question: does a calibration that uses the gradient reach the right answer, with the
default (stabilized) and with the exact water-stress derivative? `s7_calibrate.py`: the
"observations" are the 31 daily means of May of latent heat, sensible heat and GPP (the
loss outputs) from a run whose parameters are moved by up to ±5 % (a random draw, two
draws = two twins); the calibration starts at the defaults and takes 80 Adam steps (first
step 3 % of the range, cosine decay to a tenth) inside the ±10 % ranges, with either the 5
strong parameters (ι, Vcmax25, ψ50, plant conductance, near-infrared leaf reflectance) or
all 23. Eight runs, about 93 s per iteration (the month's gradient), 2.2 hours each.

| Parameters | Gradient | Twin | Loss, start → end | Mean distance from the truth (% of the value) | Largest gradient norm |
|---|---|---|---|---|---|
| 5 strong | default | 0 | 7.3e-4 → 1.0e-5 | 0.96 → 0.65 | 0.03 |
| 5 strong | default | 1 | 2.1e-2 → 4.3e-5 | 3.38 → 0.44 | 0.2 |
| 5 strong | exact | 0 | 7.3e-4 → 6.1e-3 | 0.96 → 2.45 | 5e10 |
| 5 strong | exact | 1 | 2.1e-2 → 8.5e-3 | 3.38 → 2.52 | 2e17 |
| all 23 | default | 0 | 3.5e-4 → 1.3e-5 | 2.21 → 1.99 | 0.04 |
| all 23 | default | 1 | 6.2e-3 → 1.5e-4 | 3.17 → 2.63 | 0.05 |
| all 23 | exact | 0 | 3.5e-4 → 2.0e-2 | 2.21 → 3.28 | 2e4 |
| all 23 | exact | 1 | 6.2e-3 → 6.5e-3 | 3.17 → 3.15 | 1e39 |

(`figures/calibration_progress.png`: loss and distance from the truth by iteration;
`figures/calibration_parameters.png`: the path of each strong parameter.)

- **Default gradient, 5 parameters:** the loss falls 70-500 times; ι, Vcmax25, ψ50 and
  the near-infrared reflectance end on their true values; plant conductance gets the
  direction right but stops short (twin 0: -1 % instead of -3 %; twin 1: +2.9 % instead
  of +3.6 %), so it trades off against the others in the daily means. The loss is
  still falling slowly at iteration 80.
- **Default gradient, 23 parameters:** the loss falls as far (to about 1e-5 and 1e-4), but
  the mean distance from the truth hardly changes (2.2 → 2.0 %, 3.2 → 2.6 %): the twin
  can be matched with other combinations of the weak parameters, as the Hessian's 1e13
  condition number said (result 7). In twin 1, 11 of 23 parameters got closer by half
  or more, 4 moved away (`plant_capacitance`, `root_resistivity`, `clumping_index`,
  `stem_transmittance[1]`); in twin 0 only 4 got closer by half.
- **Exact gradient:** every strong parameter moves about 2 % in the first few iterations,
  in some direction regardless of its truth, and then stops (the flat blue lines). Its
  gradient norm reaches 5e10 to 1e39 on single iterations. Probably (not checked) Adam's
  scaling by the running size of the gradient: the first steps follow the signs of an
  erratic gradient, and after a spike the scale is so large that the steps vanish. The
  loss ends above its start in two runs out of four.
- In short: the stabilized derivative is the one a calibration can use; the exact one,
  correct at a point, is unusable over a month with water stress (results 2 and 3). The
  month of daily means constrains about the five strong parameters, not 23.
- **Not run:** a calibration with more days or half-hourly observations, other optimizers
  or settings (a gradient clip would hide, not cure, the exact gradient's spikes), the
  soft-threshold prototype.

## Issues met, and what was done

- **Account UCLB0069 has no CPU hours** (`qsub: Invalid account for CPU usage`). The jobs
  asked for a GPU node of the develop queue (32 cores, one GPU) and used only its CPUs.
- **Solver convergence audit** skipped (needs a code change), see step 3.
- **Regimes**: Rubisco- against light-limited and "stomata at `gs_min`" are not among the
  model's outputs; the regimes used are the six in the table of step 2.
- **Smoothing scales** of step 4 are the spacing of the sweep (0.25 % of the range in
  one-day sweeps, 1 % in 31-day ones), a tenth of the range and the whole range, instead
  of 0.1, 1 and 10 %.
- **The horizon sweep from April** mixes horizon and leaf-out; the horizon figure for the
  parameters was therefore also made from 1 May (from the month's record). The harness's
  loss gave NaN in reverse mode for windows without leaves (GPP constant, variance zero):
  fixed in the harness (such an output is left out); a probe showed the model's own
  reverse-mode gradient on a leafless day is finite. The four shortest horizons of the
  April start-state figure are missing for that reason.
- **Two diagnostics were added** to explain result 2: four days and two Jacobians with
  `water_stress_derivative: exact`, and the round-off twins. They are diagnoses, not the
  comparison of remedies of step 6.
- **Recorded as open items, nothing changed:** `leaf_emissivity` = 1 gives NaN (O83); the
  growth of round-off under water stress (O84); the zero derivative at dry leaves (O85).

## Work plan and status

| Step | What | Status |
|---|---|---|
| 1 | The harness, the two experiment folders | done |
| 2 | Baseline, all of May: every day through the harness, regimes, NaN over every input, dot-product test | done |
| 3 | Threshold sweeps | done (solver audit not possible without a code change) |
| 4 | Parameters: one at a time, dense sweeps, Hessian | done |
| 5 | Time: horizon, one-step Jacobian, spread over windows, checkpoints | done |
| 6 | Remedies that are already settings (`wetted_fraction`, `water_stress_derivative`, `obukhov_derivative`): cost in bias against gain in gradient quality | not run (maintainer) |
| 7 | A twin calibration (daily means of the month; default against exact gradient) | done 2026-11-17 ("Step 7", result 8) |
| 7 | A soft threshold as a prototype; full-horizon against windowed gradients | not run |
