# Phase 10 scope: predicted soil water

Status: **scoped; decisions 131-134 taken; nothing implemented; next: 10.1, US-MMS set up on
Derecho (section 7.1)** (2026-11-24, branch `phase10/soil_water`). This
document is deleted when the phase closes; what is decided moves to `decisions.md`, what
stays open to `open_items.md` (this is open item O64).

Sources read for this assessment:

- this repository at `a77c4f6`;
- CTSM from GitHub: **`ctsm5.4.004`, the reference (decision 131)**, and `ctsm5.4.070` (the
  default branch on 2026-11-24) to see what changed since;
- CLM-ml v2 Fortran (`gbonan/CLM-ml_v2.CHATS`);
- legoLAND (`climate-federation/legoESM`, `packages/land/legoesm/land/`: `richards.py`,
  `soil_hydraulics.py`, `multilayer_land.py`);
- the earlier translation, `AyaLahlou/clm-ml-dev` (branches `main`, `differentiable-physics`,
  `eager_python`), and its fork of the Fortran, `AyaLahlou/CLM-ml_v2.CHATS` (`tests/`,
  `golden_IO/`, branch `claude/instrument-fortran-logging-SMZx4`), where this repository's
  golden data come from (section 6.1).

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
   (a) routine-level golden data recorded from CLM's own Fortran; (b) a *replay* test, where
   the JAX hydrology is driven with CLM's own fluxes from a single-point CLM run and must
   reproduce CLM's soil water step by step; (c) tests of properties that need no reference
   (water conservation, equilibrium, convergence against an independent integrator).
   Section 6.
5. **The main differentiability problem is CLM's adaptive sub-stepping**, a loop whose
   number of passes depends on the data. legoLAND's answer (a loop with a fixed number of
   passes in which finished columns are frozen) applies directly. Section 5.
6. **Decided:** a translation of CLM, reference `ctsm5.4.004` (131); a first version that
   is unfrozen and snow-free (132); CLM's ponded surface water as a store for hydrology only
   (133); evaluation at US-MMS, a rain-fed forest, because the CHATS orchard is irrigated
   (134). Nine smaller choices remain (section 8), each with a recommendation.
7. **The earlier translation (`clm-ml-dev`) has no CLM soil water to reuse, but its way of
   recording golden data does apply.** Its `SoilWaterMovementMod.py` translates CLM-ml v2's
   stub, as this repository does. What carries over is the method: one of its two methods,
   logging every input and output of a routine during a normal Fortran run, fits CTSM much
   better than the stand-alone harness first planned here. The golden data and the replay
   data then come from one traced US-MMS CLM run, in full double precision (6.1, 6.3).

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
| 6 | `UpdateFracH2oSfc`/`BulkDiag_FracH2oSfc`, `UpdateH2osfc`, `QflxH2osfcSurf`, `QflxH2osfcDrain` (SurfaceWaterMod) | ponded water: its covered fraction (from microtopography), its runoff and its drainage into the soil (at most the maximum infiltration rate over the ponded fraction) | T (decision 3) |
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

### 3.4 The reference is `ctsm5.4.004` (decision 131)

Later tags changed these files in ways that change results. What `ctsm5.4.004` does, and
what `ctsm5.4.070` does instead (not translated):

| | `ctsm5.4.004` (translated) | `ctsm5.4.070` |
|---|---|---|
| After the Richards solve | nothing: water above saturation stays until `SubsurfaceLateralFlow` passes it upward | water above the effective porosity passed to the level above at once |
| Pond drainage (`QflxH2osfcDrain`) | `min(frac_h2osfc·qinmax, h2osfc/dtime)` | a Darcy flux into the top level, with those two limits |
| `baseflow_scalar` | namelist; 0.001 with the zero-flux lower boundary | parameter file |
| Snow fraction and pond overlap | `frac_sno`, `frac_sno_eff` | renamed, overlap split differently (not relevant until snow) |

The traced CLM run for the golden data (6.1) and the replay test (6.3) uses `ctsm5.4.004`
(`/glade/work/linnia/ctsm5.4.004`).

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

**New modules** (by process, as the rest of `processes/`; CLM-ml's style and modularity, decision 138): `soil_water.py` (retention curve,
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
`n_baseflow`, `perched_baseflow_scalar` (later), `slopebeta`, `slopemax`, from the CLM
parameter file, and `baseflow_scalar` (a namelist value in `ctsm5.4.004`: 0.001). They are not PFT parameters, so `params.yaml` needs a section
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

From the earlier translation (`clm-ml-dev`), whose lessons this repository already follows:
every hydrology parameter and surface input (`FMAX`, slope, `baseflow_scalar`, ...) is an
input of the compiled chunk, never a Python constant or module value. There, a parameter read
from a module was captured as a compile-time constant: a change recompiled the program
(290 s against 0.3 s) and its gradient was zero.

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

**How this repository's golden data were made.** The 160+ golden tests in
`tests/fortran_validation` come from the earlier translation (`AyaLahlou/clm-ml-dev`) and its
fork of the Fortran (`AyaLahlou/CLM-ml_v2.CHATS`, `tests/` and `golden_IO/`). Two methods were
built there:

1. **Driver programs** (used for the committed data). One small Fortran program per routine,
   linked against the object files of the whole model, built normally. It reads its inputs as a
   namelist on standard input and prints `name=value` lines (`ES24.16`). A Python script
   (`generate_golden.py`) runs each driver on a list of hand-chosen cases and writes
   `{inputs, outputs}` JSON. The drivers were compiled with bounds checks and floating-point
   traps. Routines that could not be called alone (private ones, or ones needing the model's
   full data structures) were reached by adding public helper routines to the Fortran.
2. **Call tracing** (a branch, not used for the committed data). Each routine's inputs and
   outputs were logged during a normal run, by lines added behind a compile flag
   (`#ifdef IO_TRACE`, a 257-line `io_logger.F90`), for the first two calls of each routine.

**What changes for CLM's hydrology.** CLM-ml v2 is a small stand-alone program, so a driver
could build its inputs by hand. CTSM is different in two ways:

- The routines take CLM's data structures (`bounds`, filters, `col`, `soilstate_inst`,
  `waterstatebulk_inst`, `waterfluxbulk_inst`, `soilhydrology_inst`, ...), which exist only
  after CLM's initialization.
- Most of the solver is private to its module: `soilwater_moisture_form`,
  `compute_hydraulic_properties`, `compute_moisture_fluxes_and_derivs`, the matrix and
  right-hand side, `QflxH2osfcSurf`, `QflxH2osfcDrain` and `BulkDiag_FracH2oSfc`. The runoff
  routines are methods of objects.

The harness first planned here (CLM's files compiled against hand-written stand-ins for its
data structures) would rebuild much of that initialization by hand. Instead, call tracing
becomes the main method:

- **Tracing in the US-MMS CLM case (main source).** Copies of the hydrology files go in the
  case's `SourceMods/src.clm`, with only logging added, behind `IO_TRACE`. Two checks keep
  CLM's science unchanged: a `diff` against `ctsm5.4.004` may show logging lines only, and the
  traced run's history must equal the untraced run's bit for bit. The copies write each
  routine's inputs and outputs in full double precision. Private routines are traced from
  inside their own module, so their visibility does not change. The case is the one set up in
  10.1, so no separate build system is needed.
- **What gets recorded.** "The first two calls" is not enough. Each routine gets conditions
  that select its calls: a rain step, a step where the solver halves its sub-step, a water
  table within the column, a dry top level, water passed upward from a saturated level. A cap
  per condition keeps the files small. A second, short run with a DEBUG build (bounds checks,
  floating-point traps, as the CLM-ml drivers had) checks the same paths. The golden data come
  from the normal optimized build, so they match the replay run (6.3).
- **Driver programs for the small public pieces**, for cases the run never visits: the
  Clapp-Hornberger curve (`soil_hk` and `soil_suction`, through the retention-curve object),
  `ComputeFsatTopmodel`, `ComputeQinmaxHksat`, and the interpolation in
  `ThetaBasedWaterTable`. They are linked against the case's object files, as the CLM-ml
  drivers were linked against the model's. CLM's source is not changed to make a private
  routine callable; tracing covers those.
- **Same JSON layout as the existing golden files** (`generated_utc`, the routine,
  `cases: [{inputs, outputs}]`). The new files go in `tests/fortran_validation/golden_IO/`,
  and `test_golden_jax.py` gets adapters as for the canopy routines, with rtol 1e-9. Arrays
  keep CLM's 1-based level index in the JSON, with `nbedrock`; the adapter converts them, as
  the existing `tridiag` adapter does.

Routines: `soil_hk`, `soil_suction`; `compute_hydraulic_properties`;
`compute_moisture_fluxes_and_derivs`; the matrix and right-hand side; a whole
`soilwater_moisture_form` call including the number of sub-steps; `ComputeFsatTopmodel`;
`ComputeQinmaxHksat`; `SetQflxInputs`; the pond routines; `ThetaBasedWaterTable`;
`SubsurfaceLateralFlow`. Cases: dry, moist, a saturated level, a strong infiltration pulse
into dry soil (forces sub-step halving), shallow bedrock, strong root uptake from a dry level,
water table at several depths. Cases that US-MMS does not reach (shallow bedrock, sand)
come from drivers for the public pieces, or from a traced run of a second site, a choice made
in 10.2.

Risk: one run's states may not cover every case. The drivers fill the gaps for the public
pieces; a private routine that a run never reaches in a given case is tested through the
whole solver (`SoilWater`) in the replay test.

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

The traced single-point CLM run (6.1, `ctsm5.4.004`, satellite phenology) also records, at
every model step, the hydrology as a whole, at the call of `HydrologyNoDrainage` and
`HydrologyDrainage`:

- in: the fluxes that drive the soil (water reaching the ground, soil evaporation and dew,
  root uptake of each level, soil temperature) and the state before the step;
- out: everything the hydrology produces (liquid water and ice of each level, water table,
  pond and its fraction, saturated fraction, infiltration, surface runoff, drainage, number
  of sub-steps, water balance error).

All of it is in full double precision and taken at the call itself, so the test is not
limited by history-file precision, nor by fields that history files average or leave out. A
restart file at the start gives the exact initial state. History output (double precision,
`hist_ndens = 1`) is still written for the evaluation (6.4) and as a cross-check of the
trace.

The test feeds CLM's driving fluxes to the JAX hydrology (without the canopy) and compares
its soil water with CLM's at every step. Since the inputs are CLM's own, the translated
hydrology has to follow CLM closely for the whole run, through rain events and dry-downs;
any difference is a translation difference and can be traced to a step and a routine. The
target tolerance is set when the first run exists; with every input in double precision,
states agreeing to about 1e-9 relative are plausible. The step is CLM's (1 hour at US-MMS,
7.1).

A few days are committed as a regression reference. Cases to cover between them: a rain
event on dry soil, a water table within the column, shallow bedrock, sand and clay.
Unfrozen, snow-free periods only in the first version.

This needs you (or me on Derecho with your go-ahead) to build and run the traced CLM case.

### 6.4 Coupled evaluation (science, not pass/fail)

**Narrowed by decision 136:** this phase is a code translation; there is no evaluation
against measurements. What stays: the coupled CLM-ML runs at US-MMS beside the CLM run, with
CLM's CO₂ and soil (10.1b), the long run's water balance and the grid robustness run. The
multi-layer canopy's fluxes differ from CLM's by design, so the coupled runs are compared
with CLM for understanding, not for agreement; agreement is the business of 6.1 and 6.3.


- **US-MMS** (Morgan-Monroe State Forest, Indiana; rain-fed deciduous broadleaf forest;
  decision 134), a warm season with rain and dry-downs: predicted soil water against the
  tower's measured soil moisture and against CLM's own at the same point; latent heat,
  sensible heat and GPP against the tower and against a run with prescribed soil water.
  Winters there have frozen soil and snow, outside the first version (132): evaluation runs
  start after the last frost and snow and end before the first.
  Needed, none of it in the repository yet: the tower forcing; a single-point CLM run
  (`ctsm5.4.004`) for leaf and stem area, the initial soil state, and the replay data of 6.3;
  the tower's description (heights, soil texture, bedrock depth, `FMAX`, `SLOPE`); the
  measured soil moisture and fluxes.
- **CHATS is not used to evaluate predicted soil water:** the orchard is irrigated
  (maintainer, 2026-11-24), which a soil water balance without irrigation cannot reproduce.
  CHATS stays the test of the prescribed mode (bit for bit, and against Fortran) and can serve
  the replay test (6.3), where CLM's own fluxes drive the hydrology.
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
| 10.0 | This scope; the maintainer's decisions (section 8) | done: decisions 131-134 |
| 10.1 | **US-MMS set up** (7.1): the CLM single-point case, the tower as an experiment folder running with prescribed soil water, the measurements | a US-MMS run with prescribed soil water; **discussion** of the period chosen |
| 10.1b | CLM-ML beside CLM at US-MMS with CLM's inputs (decision 136): CO₂ from CLM's CO₂ file, as DATM (O87); soil properties at every level as `ctsm5.4.004` `SoilStateInitTimeConst` (O88), both as options with today's behaviour the default | done (decision 137): CHATS bit for bit; US-MMS CO₂ and soil properties equal CLM's to single precision |
| 10.2 | Reference data (6.1, 6.3): the traced copies of the hydrology files in the US-MMS case (logging only, checked against `ctsm5.4.004` and against the untraced run), drivers for the small public pieces, the golden JSON and the replay data | golden data committed (small); **discussion** if the run's states leave cases uncovered |
| 10.3 | `soil_water.py`: retention curve, hydraulic properties, fluxes, the adaptive solver; not yet in the step | golden tests, property tests 6.2 |
| 10.4 | `surface_runoff.py`, `drainage.py` | golden tests; conservation of the assembled hydrology |
| 10.5 | The replay test against CLM (hydrology alone), US-MMS and CHATS | **discussion** of the agreement reached |
| 10.6 | Into the model: state, step, setup inputs, parameters, restart, outputs, balance check, `upgrade` | prescribed mode bit for bit; predicted mode end-to-end test (one day) |
| 10.7 | Gradients and timing | gradient tier; **discussion** if slower than ~1.5× |
| 10.8 | Coupled runs at US-MMS beside CLM (6.4; no evaluation against measurements, decision 136), documentation, close of the phase | all of `tests/regression`, slow tests, docs build |

### 7.1 First step: US-MMS (10.1, on Derecho)

Nothing for US-MMS is in the repository; only CHATS is set up. This step comes first because
everything after the routine tests depends on it (the replay test, the end-to-end test of the
predicted mode, the evaluation), and because it needs CLM runs, which are on Derecho.

What CTSM `ctsm5.4.004` already knows about the tower
(`tools/site_and_regional/PLUMBER2_sites.csv`): latitude 39.3232, longitude −86.4131, PFT 7
(broadleaf deciduous temperate tree) on 100 %, canopy top 27.0 m and bottom 15.52 m, forcing
years 1999-2014, and **24 coupling steps per day: the PLUMBER2 forcing is hourly, so that CLM
case runs with a 1-hour step.**

1. **The CLM case.** A single-point case at US-MMS with `ctsm5.4.004`, satellite phenology,
   through CTSM's tower tooling in `tools/site_and_regional` (`run_tower`,
   `plumber2_usermods`, `plumber2_surf_wrapper`); the exact commands are to be worked out in
   the checkout. Spun up, then run over the chosen period.
2. **The period.** One warm season without frozen soil or snow (decision 132), with rain and
   at least one clear dry-down, in a year with measured soil moisture. The maintainer
   chooses the year.
3. **The model step.** With hourly forcing, CLM's hydrology step is 3600 s; CLM-ML's default
   is 1800 s. Two possibilities, to decide when the forcing is in hand: run CLM-ML with a
   1-hour model step at this tower (`numerics.step_seconds`), so that the replay and the
   coupled runs share CLM's step; or use 30-min forcing if the tower's own record has it.
   The replay test must use CLM's step whatever the choice, and the bound on the number of
   sub-step trials (5.1) is larger for 3600 s.
4. **History output for CLM-ML as it is now** (prescribed mode): `ELAI`, `ESAI`, `H2OSOI`,
   `TSOI` at every step, as the CHATS file.
5. **History output for the evaluation**, every step, double precision (`hist_ndens = 1`),
   with a restart file at the start of the period: the soil water, water table, runoff and
   drainage fields, checked against the history field list of the case. The replay data
   themselves come from the traced run of 10.2 (6.1, 6.3), not from history files; the case
   is set up so that the traced copies can be added to its `SourceMods` later without
   rebuilding anything else.
6. **The tower as an experiment folder**, `experiments/us_mms_2012/` (not a built-in
   compset: built-in towers are CHATS7 only, decision 13): location, forcing height, canopy height, soil texture, organic
   matter, colour and bedrock depth from the case's surface dataset; `FMAX` and `SLOPE` noted
   for later (decision 9 of section 8); PFT 7 with default parameters (the CHATS parameter
   set is the orchard's, not this forest's); the leaf and stem area profile parameters are a
   choice to bring to the maintainer (the 2021 paper's values for this tower, if available).
7. **The forcing** converted to the tower-forcing format of `data/input_files/tower-forcing`.
8. **A run of CLM-ML with prescribed soil water** over the period: it must pass the
   conservation checks, and its fluxes are the baseline that the predicted mode is later
   compared with.
9. **The measurements**: soil moisture (depths and units noted), latent and sensible heat,
   GPP, for the period. Large files stay outside the repository (a path on Derecho recorded
   in the experiment folder's README); only what a test needs is committed.

Settled: a 1-hour model step and May to September 2012 (decision 135). Area profiles: the PFT
values, as CLM-ml v2's `TowerDataMod` has for US-MMS. No measurements are needed (decision
136). CO₂ and soil: CLM's (decision 136; step 10.1b, O87, O88).

**Where 10.1 stands** (2026-11-24):

- Everything outside the repository is in `/glade/work/linnia/phase10_us_mms/` (inputs) and
  `/glade/work/linnia/cases/` (CLM cases; their runs and archives on scratch).
- CLM cases, `ctsm5.4.004`, compset `I1PtClm60SpRs` / `HIST_DATM%1PT_CLM60%SP_...`, resolution
  `CLM_USRDAT`, user mods `PLUMBER2/US-MMS`, CPU account P93300041:
  `US-MMS_ctsm54_SP.spinup` (cold start, 48 years cycling the 1999-2014 forcing) and
  `US-MMS_ctsm54_SP.transient` (1999-2014 from the spin-up; history `h2`: hourly, double
  precision, the CLM-ML inputs `ELAI`, `ESAI`, `H2OSOI`, `TSOI` and the soil water fields).
  The default physics are the ones of 3.1 (moisture-form solver, zero-flux lower boundary,
  `baseflow_scalar` 0.001, bedrock, hydraulic stress), checked in the case's `lnd_in`.
- Found on the way: in `ctsm5.4.004` the forcing stream is named `PLUMBER2.US-MMS`, while the
  PLUMBER2 user mods set `taxmode`, `dtlimit` and `datavars` for `CLM_USRDAT.PLUMBER2`,
  which no longer exists, so those lines are ignored; the cases set them for
  `PLUMBER2.US-MMS`. The forcing file is not in Derecho's shared input data; it is downloaded
  from the CESM input data server into `phase10_us_mms/inputdata/` and named in
  `user_nl_datm_streams`.
- The site: `FMAX` is 0, so CLM's saturated fraction is always 0 here (the saturation-excess
  runoff of 3.2 row 2 is not exercised at US-MMS: its golden data come from drivers, 6.1);
  bedrock at 20.3 m lies below the 8.5 m soil column (no shallow bedrock); slope 0.51°.
- Forcing for CLM-ML: `data/input_files/tower-forcing/US-MMS/` (May-September 2012, made by
  `convert_plumber2.py` in that folder).
- The experiment folder `experiments/us_mms_2012/` (its README records the tower, the inputs
  and the CLM runs) and the CLM history it reads, `data/input_files/ctsm5_4/US-MMS/`.
- The baseline (prescribed soil water) runs: 3672 hourly steps in 3 minutes on one CPU, no
  conservation warning, no NaN. Monthly means against CLM (latent heat, sensible heat in
  W m⁻²; GPP in µmol m⁻² s⁻¹): May 102/101, 41/42, 10.3/9.4; June 109/96, 50/61, 10.3/8.0;
  July 97/59, 47/80, 8.8/4.4; August 98/90, 33/39, 9.3/7.5; September 61/64, 31/27,
  7.4/6.9. In the July drought CLM's latent heat falls by 40 %, CLM-ML's by 10 %: CLM-ML
  feels the drought (soil water potential −0.46 MPa, midday leaf water potential −0.72 MPa,
  the top leaves at their minimum water potential in 20 % of midday hours) but its
  transpiration stays high. A science result for the discussion (with the known daytime
  latent heat bias, O62), not a defect; with predicted soil water it would dry the soil
  faster than CLM (risk in section 9).

Code-only and science changes stay in separate commits. No change to the existing science is
planned: with `prescribed` everything is as today.

---

## 8. Decisions for the maintainer

My recommendation first in each.

1. ~~Reference CTSM version.~~ **Decided 2026-11-24 (decision 131): a translation of CLM,
   reference `ctsm5.4.004`.**
2. ~~First version unfrozen and snow-free.~~ **Decided 2026-11-24 (decision 132):** yes; ice
   stays 0 and is already an argument of every routine. Freeze/thaw, the perched water table
   and snow are a later phase (O9, O10 stay open).
3. ~~Ponded surface water.~~ **Decided 2026-11-24 (decision 133):** CLM's pond as a water
   store that affects hydrology only; the ground's energy balance is unchanged.
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
8. **Irrigation.** Not translated (recommended). With evaluation moved to a rain-fed tower
   (134) nothing needs it now; a prescribed irrigation input would be a small later addition
   if predicted soil water is wanted at CHATS.
9. **Tower inputs for runoff** (`FMAX`, `SLOPE`): keys in `domain.yaml` per tower, which
   `clm-ml new` fills from the compset (recommended), with the values taken from the CLM
   surface dataset at the tower's gridcell.
10. **Form of the sub-step loop**: decided after the timing test of 10.7 (5.1).
11. **Where the hydrology parameters live**: a new section of `params.yaml` that is not per
    PFT (recommended), read from the CLM parameter file by `clm-ml params`.
12. ~~Evaluation sites.~~ **Decided 2026-11-24 (decision 134):** US-MMS; not CHATS, which is
    irrigated.
13. **How the golden and replay data are recorded** (6.1): (a) tracing of the hydrology
    routines in the US-MMS CLM case (copies in `SourceMods` with logging only, checked by
    `diff` and by a bit-for-bit history comparison) plus driver programs for the small
    public pieces (recommended: the method of the earlier translation, without stand-ins for
    CLM's data structures, and with the replay data in full double precision); (b) the
    stand-alone harness of the first version of this plan (CLM's files compiled against
    hand-written stand-ins); (c) history files only (no routine-level data; the replay test
    limited by what history files hold).

## 9. Risks

- Recording the golden data (6.1) is the least certain part of the effort, less so than
  with the stand-alone harness first planned: tracing needs no stand-ins for CLM's data
  structures, but one run's states may not reach every case, and the traced copies must be
  kept to logging only.
- Agreement in the replay test may stop above round-off (for example LAPACK's `dgtsv`
  against this repository's `tridiag`); the tolerance is fixed only after the first run.
- Predicted soil water makes the model's known biases interact: daytime latent heat is biased
  high (~70 W m⁻²), which would dry the soil too fast. This is a result to report, not to
  tune away.
- Spin-up: predicted soil water starts from CLM's state, which is in balance with CLM's
  fluxes, not CLM-ml's. Long runs will drift toward CLM-ml's own balance; O12 becomes more
  important.
- The water table jumps between levels (CLM's formulation), which makes saturated fraction
  and baseflow piecewise in the state: gradients are valid between jumps only.
