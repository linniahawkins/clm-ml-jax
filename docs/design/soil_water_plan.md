# Phase 10 scope: predicted soil water

Status: **scoping, nothing implemented** (2026-11-24, branch `phase10/soil_water`). This
document is deleted when the phase closes; what is decided moves to `decisions.md`, what
stays open to `open_items.md` (this is open item O64).

Sources read for this assessment:

- this repository at `a77c4f6`;
- CTSM from GitHub, `ctsm5.4.070` (the default branch on 2026-11-24), compared with
  `ctsm5.4.004` (the version named in CLAUDE.md) for the files below;
- CLM-ml v2 Fortran (`gbonan/CLM-ml_v2.CHATS`);
- legoLAND (`climate-federation/legoESM`, `packages/land/legoesm/land/`: `richards.py`,
  `soil_hydraulics.py`, `multilayer_land.py`).

---

## 1. Summary

1. **The model already computes every flux a soil water balance needs, and discards them.**
   Throughfall and canopy drip, soil evaporation and dew, transpiration, and the share of
   transpiration taken from each soil level are all results of a step. What is missing is
   the soil side: the state (liquid water per level), the routines that move water, and the
   bookkeeping around them.
2. **CLM-ml v2 is no help as a reference.** Its `SoilWaterMovementMod.F90` is a 166-line
   stub that computes only hydraulic conductivity and matric potential from the prescribed
   water. The reference for the translation is CLM itself.
3. **In CLM, the path actually used is a small part of the files.** The eight files
   involved hold about 7,800 lines; for a snow-free vegetated soil column with the CLM5/CLM6
   default settings, my estimate is that well under 1,000 lines are science on the path we
   need. Section 3 lists, routine by routine, what is used, what is dead, and what is used
   by CLM but not applicable here.
4. **Testing is possible without a CLM-ml reference**, in three independent ways:
   (a) routine-level golden data produced by running CLM's own Fortran routines in a small
   harness; (b) a *replay* test, where the JAX hydrology is driven with CLM's own half-hourly
   fluxes from a single-point CLM run and must reproduce CLM's soil water step by step;
   (c) tests of properties that need no reference (water conservation, equilibrium,
   convergence against an independent integrator). Section 6.
5. **The main differentiability problem is CLM's adaptive sub-stepping**, a loop whose
   number of passes depends on the data. legoLAND's answer (a loop with a fixed number of
   passes in which finished columns are frozen) applies directly. Section 5.
6. **Twelve choices are yours** (section 8). The most consequential: which CTSM version is
   the reference; that the first version is unfrozen and snow-free; how ponded surface water
   is treated; and where to evaluate, because I believe the CHATS orchard is irrigated, which
   a predicted soil water without irrigation cannot reproduce.

---

## 2. What the model does now

Each step, `step.py` calls `soil.soil_water_from_history`: volumetric water of the 20
hydrologically active levels from the CLM history record (`H2OSOI`), times the CHATS
adjustment factor in the top levels, capped at porosity, zero below bedrock, **all liquid**.
That water is then used in five places:

| Use | Routine | Levels |
|---|---|---|
| Soil albedo | `soil.soil_albedo` | top |
| Soil evaporative resistance (dry surface layer) | `soil.soil_evaporative_resistance` | top |
| Thermal conductivity and heat capacity | `soil.soil_thermal_properties` | all |
| Conductivity and matric potential, then soil-to-root resistance, weighted soil water potential and the uptake fraction of each level | `soil.soil_hydraulic_properties`, `plant_hydraulics.soil_resistance` | above bedrock |
| Humidity of the soil pore air at the surface | `canopy.soil_surface_humidity` | top |

What exists and is not used:

| Quantity | Where | Today |
|---|---|---|
| Rain and snow reaching the ground (throughfall + drip) | `canopy_water` → `qflx_tflrain`, `qflx_tflsnow` | computed, discarded |
| Soil evaporation / dew | `lhsoi_soil` (÷ latent heat = `etsoi`) | output only |
| Transpiration | `trveg_canopy` | output only |
| Fraction of transpiration from each soil level | `SoilResistance.soil_et_loss` | computed, discarded |
| Porosity, saturated conductivity, saturated suction, Clapp-Hornberger *b*, bedrock index | `PatchConstants` | used for the properties above |

What does not exist: soil water as state (the `State` carries soil temperature only), the
water table, ponded surface water, runoff and drainage, a soil water conservation check,
soil water in restarts and outputs, the surface-data inputs runoff needs (maximum
saturated fraction, slope), and the hydrology parameters.

Related open items that this phase touches but does not close: O9 (all soil water liquid),
O10 (no snowpack), O12 (initial state from monthly means), O57 (shared soil columns).

---

## 3. What CLM does, and what we need of it

### 3.1 The settings that select the code path

CLM5 and CLM6 defaults (`bld/namelist_files/namelist_defaults_ctsm.xml`):

| Setting | Default | Consequence |
|---|---|---|
| `soilwater_movement_method` | 1 (adaptive moisture form) | `soilwater_moisture_form` is used; `soilwater_zengdecker2009` (570 lines, CLM4.5) is dead |
| `upper_boundary_condition` | 1 (flux) | infiltration is given; the head condition is dead |
| `lower_boundary_condition` | 2 (zero flux) | no flow through the bottom; free drainage, head and water-table conditions are dead |
| `use_bedrock` | true | the column is solved down to the bedrock level, not all 20 levels |
| aquifer layer | none (follows from the lower boundary) | `WaterTable`, `Drainage`, `compute_qcharge`, `BaseflowSink` and the aquifer water `wa` are dead; `ThetaBasedWaterTable` and `SubsurfaceLateralFlow` are used instead |
| retention curve | Clapp-Hornberger 1978 | van Genuchten file not needed |
| `use_hillslope`, `use_vichydro` | false | hillslope branches (most of `PerchedLateralFlow`, `SubsurfaceLateralFlow`) and all VIC code are dead |
| `use_hydrstress` | true | CLM takes root uptake from its own plant hydraulics; not applicable here (3.3) |
| `h2osfcflag` | 1 | ponded surface water is active |

### 3.2 Routine by routine

In CLM's calling order (`HydrologyNoDrainageMod`, then `HydrologyDrainageMod`).
**T** = translate in the first version; **L** = later, with frozen soil or snow;
**N** = not needed.

| # | Routine (file) | What it computes | |
|---|---|---|---|
| 1 | `SetSoilWaterFractions` (SoilHydrologyMod) | ice fraction and effective porosity of each level | T |
| 2 | `SaturatedExcessRunoff`, `ComputeFsatTopmodel` (SaturatedExcessRunoffMod) | saturated fraction of the surface from the water table depth, `fsat = wtfact·exp(−0.5·fff·zwt)`; runoff = `fsat` × water reaching the ground | T |
| 3 | `SetQflxInputs` (SoilHydrologyMod) | water offered to the soil: input minus saturation runoff minus soil evaporation, split between soil and pond | T |
| 4 | `InfiltrationExcessRunoff`, `ComputeQinmaxHksat` (InfiltrationExcessRunoffMod) | maximum infiltration rate (smallest saturated conductivity of the top three levels, reduced by ice); the excess | T |
| 5 | `RouteInfiltrationExcess` (SoilHydrologyMod) | the excess goes to the pond | T |
| 6 | `UpdateFracH2oSfc`/`BulkDiag_FracH2oSfc`, `UpdateH2osfc`, `QflxH2osfcSurf`, `QflxH2osfcDrain` (SurfaceWaterMod) | ponded water: its covered fraction (from microtopography), its runoff and its drainage into the soil | T (decision 3) |
| 7 | `Infiltration`, `TotalSurfaceRunoff` (SoilHydrologyMod) | sums | T |
| 8 | `Compute_EffecRootFrac_And_VertTranSink*` (SoilWaterPlantSinkMod) | root uptake per level | N: replaced by CLM-ml's own (3.3) |
| 9 | `SoilWater` → `soilwater_moisture_form` with `compute_hydraulic_properties`, `compute_moisture_fluxes_and_derivs`, `compute_RHS_moisture_form`, `compute_LHS_moisture_form`, `IceImpedance`; `soil_hk`, `soil_suction` (SoilWaterMovementMod, SoilWaterRetentionCurveClappHornberg1978Mod) | the Richards equation: vertical movement of water between levels, with root uptake as a sink | T: the core |
| 10 | `ThetaBasedWaterTable` (SoilHydrologyMod) | water table depth: the deepest level less than 90 % saturated, interpolated | T |
| 11 | `PerchedWaterTable` (SoilHydrologyMod) | frost table and a water table perched on frozen soil | L: with unfrozen soil it finds no perched table |
| 12 | `RenewCondensation` (SoilHydrologyMod) | dew added to the top level | T (liquid part) |
| 13 | `SubsurfaceLateralFlow`, non-hillslope branch (SoilHydrologyMod) | baseflow `∝ tan(slope)·(depth of bedrock − zwt)^n`, removed level by level using the specific yield; water above saturation passed upward (to the pond at the top); a minimum of 0.01 mm kept in every level | T |
| 14 | `PerchedLateralFlow`, non-hillslope branch | drainage of the perched zone | L: zero without a frost table |
| 15 | Water balance check (BalanceCheckMod) | storage change = inputs − outputs; CLM warns above 1e-9 mm and stops above 1e-5 mm | T, as a new entry of `checks.BALANCE_CHECKS` |
| – | Snow (SnowHydrologyMod, 4,067 lines; snow cover fraction; `HandleNewSnow`) | | L (O10) |
| – | Freeze and thaw of soil water (`SoilTemperatureMod::Phasechange`) | | L (O9): it is part of CLM's soil temperature routine, which CLM-ml replaces with its own |
| – | Irrigation (IrrigationMod, 1,786 lines), groundwater withdrawal | | N (decision 8) |
| – | Urban ponding, flood water from the river model, VIC, hillslope hydrology and streams, water tracers and isotopes, excess ice, lakes, wetlands, glaciers, FATES hydraulics, the `use_flexibleCN` negative-water patch, soil moisture streams | | N |

Inside `soilwater_moisture_form` itself, these are dead and are not translated: the
"expensive" flux calculation, the "iterate" option of the right-hand side (it stops the run),
the `clmOrig` tridiagonal solver, the head and water-table boundary conditions, `zdflag`.

### 3.3 Where CLM's code does not fit CLM-ml

- **Root uptake.** CLM (default) takes the uptake of each level from its plant hydraulic
  stress scheme. CLM-ml has its own plant hydraulics (Bonan et al. 2014), which already gives
  the fraction of transpiration taken from each level (`soil_et_loss`). The sink of the
  Richards equation is therefore `soil_et_loss × transpiration` (the step mean), not a CLM
  routine. This is what the name `soil_et_loss_soil` in CLM-ml v2 was prepared for; the
  Fortran never uses it.
- **Conductivity and matric potential for the plants.** CLM-ml v2's stub computes the
  conductivity at each level's own saturation, without ice impedance, and limits the matric
  potential at −1e8 mm. CLM's real routine computes the conductivity at the *interface*
  between two levels (mean saturation, with ice impedance), because that is what the flux
  between levels needs. Both are needed: the stub's form for the plant hydraulics (no change
  of today's science), CLM's form inside the Richards solver.
- **One ground, no pond energy balance.** In CLM the ponded fraction of the ground has its
  own temperature and evaporation. CLM-ml has one ground temperature and sets the pond to
  zero. See decision 3.
- **Soil evaporation and dew** come from the multilayer canopy's ground flux (one number,
  positive or negative), where CLM has four (liquid/solid × evaporation/dew).
- **The order within a step** is the same as CLM's: canopy fluxes, soil temperature, then
  hydrology with the step-mean fluxes. The canopy therefore sees the soil water of the end of
  the previous step. Hydrology runs once per 30-min model step (CLM's own step), with its own
  adaptive sub-steps; it does not enter the 5-min canopy sub-steps.

### 3.4 CTSM versions differ in these files

Between `ctsm5.4.004` and `ctsm5.4.070` the files above changed in ways that change results:

- `soilwater_moisture_form`: water above saturation is now passed to the level above after
  the solve (absent in 5.4.004);
- `QflxH2osfcDrain`: pond drainage is now a Darcy flux into the top level, limited by the
  maximum infiltration rate (5.4.004: the limit only);
- `baseflow_scalar` moved from the namelist (0.01) to the parameter file;
- the snow-fraction variables were renamed and the overlap of snow and pond is split
  differently (not relevant until snow).

The reference version must be pinned, and the CLM run for the replay test (6.3) must be made
with that same version (decision 1).

---

## 4. What has to be added to the model

**State** (`state.SoilState`): liquid water of each level (kg m⁻²), ponded water (mm),
water table depth (m). Ice is carried as an argument fixed at zero in the first version, so
that frozen soil later changes values, not function signatures.

**The step** (`step.py`), with `soil.water: predicted`:

1. properties from the state's water (as today from the prescribed water);
2. canopy sub-steps (unchanged), giving the step means of throughfall, soil evaporation,
   transpiration;
3. soil temperature (unchanged);
4. hydrology: rows 1-7, 9, 10, 12 of the table, then 13; the conservation check.

`soil.water: prescribed` stays, as the default, and must remain bit for bit: it is the only
mode that can be compared with CLM-ml v2 Fortran. The choice is static (it selects a code
path), in `model.yaml`.

**New modules** (by process, as the rest of `processes/`): `soil_water.py` (retention curve,
hydraulic properties, the Richards solver), `surface_runoff.py` (saturated fraction,
infiltration, pond), `drainage.py` (water table, baseflow). Pure functions of one column,
0-based, every level slot computed with masks for the levels below bedrock.

**Inputs at setup**

| Input | CLM source | Towers | Grid |
|---|---|---|---|
| maximum saturated fraction `wtfact` | surface dataset `FMAX` | `domain.yaml` key (decision 9) | surface dataset |
| slope (degrees, at least 0.2) | surface dataset `SLOPE` | `domain.yaml` key | surface dataset |
| microtopography `micro_sigma`, pond threshold | computed from the slope at setup | same | same |
| initial soil water | – | the history record at the start, or a restart | as today for TSOI, or a restart |

**Parameters**, all differentiable: `e_ice`, `fff`, `pc`, `mu`, `aq_sp_yield_min`,
`n_baseflow`, `baseflow_scalar`, `perched_baseflow_scalar` (later), `slopebeta`, `slopemax`,
from the CLM parameter file. They are not PFT parameters, so `params.yaml` needs a section
that is not per PFT (decision 11), and `clm-ml params` must read them.

**Around the physics:** restart fields; output variables (soil water per level, water table,
pond, infiltration, surface runoff, drainage, number of sub-steps, water balance error);
`clm-ml upgrade` adds the new keys to existing experiment folders; the `.out` writer is
unchanged (CLM-ml v2 has no such output); documentation of the new mode.

---

## 5. Making it differentiable

### 5.1 The adaptive sub-step loop

CLM solves the Richards equation with one linearized implicit solve per sub-step, estimates
the error, and then halves the sub-step and retries (error above 0.1 mm, sub-step above 60 s)
or accepts and possibly doubles it (error below 0.01 mm), until the 30 minutes are covered.
The number of passes depends on the data.

JAX can run such a loop (`lax.while_loop`) but cannot compute reverse-mode gradients through
it. legoLAND's solver shows the way around: a loop with a **fixed number of passes**, in
which a column that has finished is frozen (its state kept by `jnp.where`), with a count of
the passes actually used returned as a diagnostic.

For CLM's algorithm the number of passes is bounded (each accepted sub-step is at least 60 s
except the last; a rejection needs a preceding doubling or the initial large sub-step): under
70 trials for an 1800 s step. The exact bound will be established by enumeration and tested.
Finished trials change nothing, so **forward values are identical** to CLM's algorithm:
the solver's answer stays the solver's own answer, and no custom gradient rule is needed
(each sub-step is a direct solve, not an iteration to convergence). The gradient is the
exact derivative of the path the solver took; accept/reject decisions are discrete, so the
gradient is piecewise, as for any adaptive method.

Cost: each trial is a 20-level tridiagonal solve, small beside the canopy's work, but 70 of
them per step is not nothing. A timing test decides between always using the fixed-count
loop, or `while_loop` for forward-only runs and the fixed count for gradients, following the
existing `numerics.solver_loops` (decision 10). More than ~1.5× is discussed before going on.

### 5.2 Patterns to take from legoLAND

- The fixed-count loop with frozen columns (above).
- **Keep every division finite in a frozen or masked lane.** `jnp.where` hides a value, not
  a NaN gradient made on the hidden side; inputs of the hidden side must be safe (the
  "double where"). This is already this repository's rule for padded canopy layers; it applies
  to levels below bedrock, to finished sub-step trials, and to the dry pond.
- A floor on the dry side of the retention curve (CLM has one: relative saturation is kept
  in [0.01, 1]).
- The water budget residual and the pass count are outputs checked on the host, never a
  stop inside compiled code (this repository's `balance_error` already works so).

What not to take: legoLAND's solver is a different scheme (mixed-form Picard iteration after
Celia et al. 1990, with its own limiters, ponding cap and choice of retention curves). It is
not CLM's, so its science is not copied; only the JAX techniques are.

### 5.3 Places in CLM's formulas that need care

| Where | Hazard | Treatment |
|---|---|---|
| `s^(2b+3)`, `s^(−b)` with *b* a parameter | a power with a parameter as exponent | safe: `s` is bounded to [0.01, 1] before the power |
| `(bedrock depth − zwt)^n_baseflow` | base reaches 0; derivative with respect to the exponent is NaN there | safe base inside the `where` |
| `(frac_h2osfc − pc)^mu`, `mu` < 1 | infinite slope at base 0 | safe base inside the `where` |
| Newton iteration for the pond fraction (10 fixed passes, `erf`) | division by a slope near 0 when there is almost no pond | fixed count, so plain differentiation works; safe inputs when the pond is below its cutoff |
| Water table search (deepest level under 90 % saturation, then interpolation) | a loop with an exit; the result jumps when the level changes | masks instead of the exit; the jump is CLM's and stays (the gradient is valid between jumps) |
| Level-by-level removal of drainage, passing of excess water upward, minimum water | loops that carry a remainder, with exits | `lax.scan` over the 20 levels carrying the remainder; no index scatter into a carry |
| Levels below bedrock (`nbedrock` is data, differs per patch) | must not change results or give NaN gradients | every slot computed, identity rows in the matrix below bedrock, top-level inputs as padding |
| `min`/`max` limits (saturation bounds, flux limits, small-value cutoffs) | zero gradient on the limited side | kept as in CLM; documented |
| Tridiagonal solver: CLM uses LAPACK `dgtsv` (with pivoting), this repository's `tridiag` has none | results equal to round-off unless a pivot is needed | compared in the tests; if pivoting ever matters, add it |

CLM codes the derivatives of the fluxes by hand to build the matrix; these are part of the
forward algorithm and are translated as written (they are not replaced by automatic
differentiation, which would change forward values).

Every new derivative is checked against finite differences (6.5), including with respect to
the soil parameters and the initial soil water.

---

## 6. Testing and evaluation

There is no CLM-ml run with predicted soil water to compare with. The plan replaces that one
comparison with four independent lines of evidence.

### 6.1 Routine level, against CLM's Fortran (golden data)

A small Fortran harness that compiles **CLM's own source files for these routines,
unmodified**, against stub versions of CLM's data types, calls each routine on prepared
inputs and writes inputs and outputs as JSON, like the existing 160+ golden tests
(`tests/fortran_validation`, rtol 1e-9). CLM-ml v2 is itself such a stub environment
(`clm_src_main`, `clm_src_biogeophys` hold cut-down `ColumnType`, `SoilStateType`,
`WaterStateBulkType`, ...), so the harness extends something that exists rather than
starting from nothing. Built on Derecho (Fortran compiler and LAPACK there).

Routines: `soil_hk`, `soil_suction`; `compute_hydraulic_properties`;
`compute_moisture_fluxes_and_derivs`; the matrix and right-hand side; a whole
`soilwater_moisture_form` call including the number of sub-steps; `ComputeFsatTopmodel`;
`ComputeQinmaxHksat`; `SetQflxInputs`; the pond routines; `ThetaBasedWaterTable`;
`SubsurfaceLateralFlow`. Cases: dry, moist, a saturated level, a strong infiltration pulse
into dry soil (forces sub-step halving), shallow bedrock, strong root uptake from a dry level,
water table at several depths, plus random columns.

Risk: untangling CLM's modules for the harness may take longer than the translation of a
routine. If one routine resists, its golden data can come from the replay run (6.3) instead.

### 6.2 Properties that need no reference

- **Water conservation every step**, to round-off: change of (soil water + pond) =
  (water reaching the ground + dew − evaporation − root uptake − surface runoff − drainage)
  × step. Recorded as a balance check, CLM's thresholds.
- Hydrostatic equilibrium stays at rest; a closed column redistributes toward it and keeps
  its total.
- Steady infiltration into a uniform deep column approaches unit-gradient flow (flux =
  conductivity).
- **Convergence**: the same spatial equations integrated by an independent, tightly
  controlled integrator (SciPy, in the test only) agree with the solver, and the difference
  shrinks as the solver's tolerances are tightened.
- Bounds (water between the minimum and the effective porosity), no NaN.
- Levels below bedrock and padded patches do not change results (as `test_padding`).
- The fixed-count loop and the `while_loop` give identical results; the trial bound is
  never reached in a set of extreme cases.
- `soil.water: prescribed` passes the bit-for-bit and noise-twin tests unchanged.

### 6.3 The replay test: the whole hydrology against CLM, step by step

A single-point CLM run (the pinned CTSM version, satellite phenology) writing every 30-min
step in double precision: the fluxes that drive the soil (water reaching the ground, soil
evaporation and dew, root uptake of each level, soil temperature) and everything the
hydrology produces (liquid water and ice of each level, water table, pond and its fraction,
saturated fraction, infiltration, surface runoff, drainage, number of sub-steps, water
balance error), with a restart file for the exact starting state.

The test feeds CLM's driving fluxes to the JAX hydrology (without the canopy) and compares
its soil water with CLM's at every step. Since the inputs are CLM's own, the translated
hydrology has to follow CLM closely for the whole run, through rain events and dry-downs;
any difference is a translation difference and can be traced to a step and a routine. The
target tolerance is set when the first run exists (history output limits it; states near
1e-9 relative are plausible if all inputs are in double precision).

A few days of one site are committed as a regression reference. Sites to cover between
them: a rain event on dry soil, a water table within the column, shallow bedrock, sand and
clay. Unfrozen, snow-free periods only in the first version.

This needs you (or me on Derecho with your go-ahead) to run CLM and list the history fields;
the exact field names are to be confirmed against CLM's history field list for the pinned
version. Root uptake per level and the driving fluxes exist as history fields in CLM5; if one
is missing from the list, the harness of 6.1 is the fallback for that routine.

### 6.4 Coupled evaluation (science, not pass/fail)

- CHATS, May 2007: predicted against prescribed soil water, and the effect on latent heat,
  sensible heat, GPP and leaf water potential. **Caution:** I believe the orchard is
  irrigated, and the prescribed water carries an observation-based correction factor
  (`CHATS_soil_moisture_correction`). A predicted soil water without irrigation would then
  dry below what was observed, for a reason that is not a model error. Please confirm; see
  decision 8 and 12.
- A rain-fed forest tower with measured soil moisture (the sites of the 2021 paper;
  `origin/add-us-umb-site` has US-UMB): predicted soil water against measurements and against
  CLM's own, over a season with rain and dry-down.
- A run of a year or more: water balance closed over the run (precipitation =
  evapotranspiration + runoff + storage change), no drift, statistics of the sub-step count.
- Grid: a few days of the 1° globe for robustness across soils (unfrozen, snow-free patches
  only make sense scientifically; the rest runs but is not evaluated).

### 6.5 Gradients

Finite-difference checks, in the existing gradient tier, of soil water and latent heat with
respect to: saturated conductivity, *b*, saturated suction, porosity, `fff`,
`baseflow_scalar`, `n_baseflow`, the initial soil water, and a canopy parameter (through
transpiration). Cases chosen to sit on the hazards of 5.3: dry top level, saturated column,
rain and no rain, shallow bedrock, a step with halved sub-steps. Gradients over several days
(time gradients from the third call, as now).

### 6.6 Timing

Timing tests of a step and of its gradient, predicted against prescribed, on CPU (tower) and
GPU (grid). More than ~1.5× is brought to you before continuing.

---

## 7. Proposed order of work

Each sub-phase ends with its tests passing and a short report; the phase stops for discussion
where marked.

| | Content | Ends with |
|---|---|---|
| 10.0 | This scope; your decisions (section 8) | **discussion** |
| 10.1 | Reference data: the Fortran harness and its golden JSON; the CLM single-point run for the replay test | data committed (small) |
| 10.2 | `soil_water.py`: retention curve, hydraulic properties, fluxes, the adaptive solver; not yet in the step | golden tests, property tests 6.2 |
| 10.3 | `surface_runoff.py`, `drainage.py` | golden tests; conservation of the assembled hydrology |
| 10.4 | The replay test against CLM (hydrology alone) | **discussion** of the agreement reached |
| 10.5 | Into the model: state, step, setup inputs, parameters, restart, outputs, balance check, `upgrade` | prescribed mode bit for bit; predicted mode end-to-end test (one day) |
| 10.6 | Gradients and timing | gradient tier; **discussion** if slower than ~1.5× |
| 10.7 | Evaluation runs (6.4), documentation, close of the phase | all of `tests/regression`, slow tests, docs build |

Code-only and science changes stay in separate commits. No change to the existing science is
planned: with `prescribed` everything is as today.

---

## 8. Decisions for the maintainer

My recommendation first in each.

1. **Reference CTSM version.** Pin one tag for the translation and the replay run. Either
   `ctsm5.4.004` (what CLAUDE.md names, on Derecho) or a current tag (5.4.070 has the
   over-saturation fix and the Darcy pond drainage, 3.4). Recommendation: the version you
   will run for the replay test and compare science against; if you have no preference, the
   current tag.
2. **First version unfrozen and snow-free** (ice stays 0, as today; O9, O10 stay open), with
   ice already an argument of every routine. Freeze/thaw, the perched water table and snow are
   a later phase. Alternative: include freeze/thaw now, which means translating part of CLM's
   soil temperature routine into CLM-ml's own.
3. **Ponded surface water.** (a) Keep CLM's pond as a water store that affects hydrology
   only, the ground's energy balance unchanged (recommended: CLM's default hydrology, water
   conserved, no change to the canopy model); (b) no pond (`h2osfcflag = 0`: infiltration
   excess runs off at once); (c) the pond with its own evaporation and temperature, a change
   to CLM-ml's ground energy balance.
4. **Snowfall reaching the ground** (grid runs; discarded today). Until there is a snowpack:
   (a) added to the soil input as liquid water at once (recommended: conserves water; the
   timing of melt is wrong); (b) discarded as today; (c) predicted mode refuses patches with
   snowfall.
5. **Root uptake** = CLM-ml's `soil_et_loss × transpiration` (recommended; 3.3), not a CLM
   routine.
6. **Conductivity and matric potential for the plant hydraulics** stay in CLM-ml v2's form
   (recommended; no change of today's science), with CLM's interface form used only inside
   the Richards solver.
7. **Soil evaporation and dew** from the canopy model's single ground flux: evaporation
   removed from the infiltration input, dew added to the top level, as CLM does with its
   liquid terms (recommended).
8. **Irrigation.** Not translated (recommended for this phase). If CHATS is irrigated and you
   want predicted soil water there, a prescribed irrigation input (a forcing variable added
   to the water reaching the ground) is a small addition; CLM's irrigation scheme is not.
9. **Tower inputs for runoff** (`FMAX`, `SLOPE`): keys in `domain.yaml` per tower, which
   `clm-ml new` fills from the compset (recommended), with the values taken from the CLM
   surface dataset at the tower's gridcell.
10. **Form of the sub-step loop**: decided after the timing test of 10.6 (5.1).
11. **Where the hydrology parameters live**: a new section of `params.yaml` that is not per
    PFT (recommended), read from the CLM parameter file by `clm-ml params`.
12. **Evaluation sites** beyond CHATS (6.4), and whether measured soil moisture is available
    for them.

## 9. Risks

- The Fortran harness (6.1) is the least certain part of the effort.
- Agreement in the replay test may stop at the precision of CLM's history output rather than
  at round-off; the tolerance is fixed only after the first run.
- Predicted soil water makes the model's known biases interact: daytime latent heat is biased
  high (~70 W m⁻²), which would dry the soil too fast. This is a result to report, not to
  tune away.
- Spin-up: predicted soil water starts from CLM's state, which is in balance with CLM's
  fluxes, not CLM-ml's. Long runs will drift toward CLM-ml's own balance; O12 becomes more
  important.
- The water table jumps between levels (CLM's formulation), which makes saturated fraction
  and baseflow piecewise in the state: gradients are valid between jumps only.
