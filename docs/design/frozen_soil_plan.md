# Phase 12 plan: frozen soil water, freeze and thaw, the perched water table

Written 2026-11-26 on branch `phase12/frozen_soil` (from `phase11/drought_response`).
**Status: a proposal. Nothing is implemented and no decision of section 8 is taken.**
The plan follows Phase 10's method (decisions 136, 139-144): a code translation of CLM
(`ctsm5.4.004`), each routine equal to CLM's Fortran on idealized sweeps, then sequences of
the whole soil over time, with derivatives checked against finite differences at every step
of the work, not at the end. No snowpack (open item O10 stays open).

Contents: 1 summary · 2 what the model does now · 3 what CLM does and what is needed ·
4 what is added to the model · 5 derivatives · 6 testing · 7 order of work ·
8 decisions for the maintainer · 9 risks.

---

## 1. Summary

1. **Most of the hydrology already handles ice.** Phase 10 translated every hydrology
   routine with ice as an argument and tested it against CLM's Fortran with ice in the
   pores (decision 141): the reduction of conductivity by ice in the Richards solver, in
   the infiltration limit and in the baseflow; the water table; frost and sublimation at
   the top level; ice above the pore space leaving as runoff. The model only ever passes
   ice = 0.
2. **What is missing is small in lines and central in effect:**
   - *freeze and thaw*: the routine that turns liquid into ice and back and holds the
     temperature at freezing meanwhile (CLM `SoilTemperatureMod.F90::Phasechange_beta`,
     about 150 lines on our path), with the liquid water that stays unfrozen below 0 °C
     (supercooled water);
   - *the frost table and the perched water table* (`PerchedWaterTable`, 115 lines), their
     use in the saturated fraction (3 lines), and the drainage of the perched zone
     (`PerchedLateralFlow`, about 50 lines outside the hillslope branches);
   - ice as model state, at the start of a run, in restarts and outputs.
3. **The hard part is the fit, not the translation.** CLM's freeze and thaw is the last
   step of CLM's soil temperature routine, which CLM-ml replaces with its own (the ground
   heat flux is given by the canopy). The routine can be translated as written and called
   with the two inputs that differ set for CLM-ml (section 3.3). Other places where CLM's
   frozen-soil code meets the canopy model need a choice by the maintainer: snowfall on
   the ground, sublimation, roots in freezing levels (section 8).
4. **Without snow a frozen soil is not yet realistic in winter.** Snow insulates: CLM's
   soil at US-MMS freezes to about 0.7 m under snow; a bare soil freezes deeper and
   earlier. This phase is judged, as Phase 10 was, by agreement with CLM's Fortran given
   the same inputs, not by winter results.
5. **Fortran validation is possible with the existing harness**, with one unknown:
   `SoilTemperatureMod` is not in CTSM's unit-test build and `Phasechange_beta` is
   private. A feasibility probe comes first (12.1), as in Phase 10.
6. **Derivatives:** freeze and thaw is piecewise linear with switches (frozen or not,
   melting or freezing, ice left or not). The derivative is exact between switches. The
   places that would give NaN are listed in 5.2; each sub-phase ends with its
   finite-difference checks.

---

## 2. What the model does now

| | With `soil.water: prescribed` | With `soil.water: predicted` |
|---|---|---|
| Ice | 0 everywhere (as CLM-ml v2; O9) | 0 everywhere (decision 132) |
| Soil temperature | heat diffusion with the canopy's ground heat flux at the top (`soil.soil_temperature`, CLM-ml v2's own routine); no latent heat of freezing: the soil cools below 0 °C as wet unfrozen soil | the same |
| Thermal properties | `soil.soil_thermal_properties` has CLM's frozen forms (ice in the heat capacity, the frozen Kersten number) and receives ice = 0 | the same |
| Hydrology | none | CLM's, called with ice = 0; the frost table and the perched water table are not computed, the perched zone is not drained |
| Snowfall reaching the ground | discarded | enters the soil as liquid water at once (decision 145) |
| Ground evaporation | from CLM-ml's flux-profile solution, latent heat of vaporization | all taken as liquid from the top level and the pond |
| Roots | a level with any ice gives no water (CLM-ml v2's rule, `plant_hydraulics`), never triggered | the same |

State (`SoilState`): temperature, liquid water, pond, water table depth. No ice.

Inputs: towers read `H2OSOI` and `TSOI` of a CLM history file. The US-MMS CLM run
(`/glade/work/linnia/phase10_us_mms/clm_runs/`) also has `SOILLIQ`, `SOILICE`, `ZWT_PERCH`
and `QDRAI_PERCH`: in 1999-2014 its top eight levels freeze (ice up to 38 kg m⁻² in a
level) and a perched water table occurs.

---

## 3. What CLM does, and what is needed of it

### 3.1 Routine by routine

**T** translate in this phase · **D** done in Phase 10 with ice as an argument (now given
real ice) · **C** a choice of section 8 · **N** not needed.

| # | Routine (file) | What it computes | |
|---|---|---|---|
| 1 | `Phasechange_beta`, soil levels (SoilTemperatureMod) | after the temperature solve: a level with ice above 0 °C melts, a level below 0 °C with more liquid than the supercooled amount freezes; its temperature is set to 0 °C, the energy this frees or needs becomes a mass of water changing phase, limited by the ice or liquid there is; energy left over moves the temperature away from 0 °C again | T: the core |
| 2 | supercooled water, in the same routine (Niu and Yang 2006) | the liquid that stays unfrozen at a temperature below 0 °C: `watsat (smp/sucsat)^(-1/bsw)` with `smp = hfus (tfrz − T)/(grav T)` | T |
| 3 | `Phasechange_beta`, snow too thin for layers (`h2osno_no_layers`) | snow lying on the ground without snow layers is melted with the top level's energy and drains to the ground | C (choice 2) |
| 4 | `Phasechange_beta`, snow layers; excess ice | | N: no snow layers; excess ice is off in CLM's defaults |
| 5 | `PhaseChangeH2osfc` | the pond freezes and its ice joins the bottom snow layer | N: needs a snow layer; the pond stays liquid (decision 133) |
| 6 | `SoilThermProp` | conductivity and heat capacity with ice | D in `soil.soil_thermal_properties` (from CLM-ml v2's copy); to check in 12.1 that the copy equals `ctsm5.4.004` for soil levels and that its golden cases include ice |
| 7 | `SetSoilWaterFractions`, `IceImpedance` in the Richards solver, `ComputeQinmaxHksat`, baseflow impedance and ice runoff in `SubsurfaceLateralFlow`, `ThetaBasedWaterTable`, `RenewCondensation` (frost, sublimation) | | D |
| 8 | `PerchedWaterTable` (SoilHydrologyMod) | the frost table: the top of the first frozen level that has an unfrozen level above it (the top of the column if the surface is frozen, the bottom if nothing is). If the water table lies below the frost table: the perched water table, found upward from the frost table as the first level less than 90 % saturated, interpolated | T |
| 9 | `ComputeFsatTopmodel` (SaturatedExcessRunoffMod) | where a frost table lies between the perched and the main water table, the saturated fraction is computed from the perched water table | T: three lines added to `surface_runoff.saturated_fraction` |
| 10 | `PerchedLateralFlow`, non-hillslope branch (SoilHydrologyMod) | drainage of the perched zone: `perched_baseflow_scalar × sin(slope) × mean hksat × (frost table − perched table)`, removed level by level with the specific yield | T |
| 11 | `SubsurfaceLateralFlow`, rising water table | uses the frost table; needs inflow from a neighbouring column | N (hillslope only, decision 143) |
| 12 | Split of the ground's vapour flux (SoilFluxesMod) | evaporation is taken from liquid and ice of the top level in their proportion; dew is frost when the ground is below 0 °C | T, as a small function (the step already passes the four fluxes; today the solid ones are 0) |
| 13 | Latent heat of the ground flux (BiogeophysPreFluxCalcsMod) | sublimation heat when the top level has ice and no liquid | C (choice 4): it is inside the canopy's flux solution |
| 14 | Ground humidity (SurfaceHumidityMod) | CLM uses liquid + ice for the matric potential at the surface; CLM-ml v2 uses liquid only | C (choice 5) |
| 15 | Root uptake from freezing levels | CLM-ml v2: none from a level with any ice. CLM's roots (decision 151): to read in 12.1 | C (choice 6) |
| – | Snowpack (SnowHydrologyMod, snow cover, snow albedo, snow layers in the temperature solve) | | not in this phase (O10) |
| – | `ActiveLayerMod`, hillslope, VIC, urban, lakes, glaciers | | N |

### 3.2 CLM's order within a step, with the new calls

1. canopy fluxes (CLM-ml's own), giving the ground heat flux;
2. soil temperature: CLM-ml's heat diffusion, **then freeze and thaw** (new);
3. hydrology, as today, with: the saturated fraction using the frost and perched tables
   **of the step before** (CLM computes them later in the step); after the Richards
   solver **the frost table and the perched water table** (new; before the water table,
   whose value of the step before they use); water table, dew and frost as today;
   **drainage of the perched zone** (new) before the baseflow.

So the frost table and the perched water table depth are carried from step to step: they
are state, with the ice.

### 3.3 Where CLM's code does not fit CLM-ml

- **The temperature solve before the phase change is CLM-ml's.** `Phasechange_beta`
  takes two things from CLM's solve: `fact` (time step over heat capacity, with a
  geometric factor in CLM's top level) and `dhsdT` (how the surface heat flux changes
  with ground temperature, because CLM's top soil level is also its surface). In CLM-ml
  the ground heat flux is given to the soil and the surface temperature belongs to the
  canopy solution, so the consistent call is `fact = step / heat capacity` (CLM-ml's)
  and `dhsdT = 0`. The translated function keeps both as arguments, the golden data sweep
  CLM's values of both, and the model passes CLM-ml's. With these, energy is conserved:
  heat capacity × temperature change + heat of fusion × ice change = ground heat flux,
  which becomes the soil energy check (today it has no fusion term).
- **The canopy sees the ground through the top level's temperature and conductivity**
  within the 5-minute sub-steps, as now; freeze and thaw is applied once per model step,
  after them. A freezing top level therefore stays at 0 °C from one step to the next,
  which the canopy then sees.
- **No snow** (choice 2).
- **The pond does not freeze** (row 5): it has no temperature of its own (decision 133).
  Rain on frozen ground ponds and runs off, as in CLM without the freezing.

---

## 4. What has to be added to the model

**Option** (`model.yaml`, static): `soil.freeze_thaw: false | true`, default `false`,
allowed with `soil.water: predicted` (choice 1 for prescribed water). `false` is today's
model bit for bit, prescribed and predicted.

**State** (`SoilState`): `ice` (the 20 hydrologically active levels, kg m⁻²),
`frost_table_depth`, `perched_water_table_depth` (m). Restart format version 8.

**Start of a run:** liquid and ice from `SOILLIQ` and `SOILICE` of the CLM history where
it has them; otherwise the total water split by the supercooled-water curve at the
initial temperature (choice 3); or a restart.

**New code** (pure functions of one column, in CLM-ml's style, decision 138):
- `processes/soil.py`: `supercooled_water`, `phase_change` (rows 1-2, optionally 3); the
  soil energy check with the fusion term.
- `processes/drainage.py`: `perched_water_table` (frost table, perched table),
  `perched_drainage`.
- `processes/surface_runoff.py`: the perched water table in `saturated_fraction`; the
  split of the ground's vapour flux into liquid, ice, dew and frost.
- `processes/hydrology.py::soil_hydrology`: the new calls in CLM's order (3.2).

**Parameter:** `perched_baseflow_scalar` in the `hydrology` section of `params.yaml`
(CLM's parameter file; `clm-ml params` reads it, `clm-ml upgrade` adds it). `e_ice`
exists and starts to act.

**Outputs:** `soil_ice` per level, `frost_table_depth`, `perched_water_table_depth`,
`perched_drainage`, `ice_runoff`, the phase-change energy, and (choice 2) the snow store.

**Unchanged:** the canopy, unless choices 4 to 6 say otherwise; the `.out` writer; the
regression reference (new fields only, the existing ones bit for bit).

---

## 5. Making it differentiable

### 5.1 What kind of function freeze and thaw is

No iteration and no loop whose length depends on the data: for each level a few
comparisons and limits. The forward values are the routine's own. Between switches the
derivative is exact and needs no custom rule. What is particular:

- **A level that is changing phase has temperature 0 °C whatever its inputs**: the
  derivative of its temperature with respect to anything earlier is zero, and the
  sensitivity is carried by its ice instead. This is the physics (the "zero curtain"),
  not a defect; a gradient through a winter is checked for it in 12.6.
- **Switches** (starts to freeze, last ice melts, the frost table moves a level) are
  kinks or jumps of the forward model, as the water table's level changes already are. A
  finite difference across one is not a derivative; the tests take two step sizes and
  compare no closer than they agree (the method of decision 147).

### 5.2 Places that need care

| Where | Hazard | Treatment |
|---|---|---|
| supercooled water `(smp/sucsat)^(-1/bsw)`, `smp ∝ (tfrz − T)/T` | at and above 0 °C the base is 0 or negative with a negative, parameter-dependent exponent: infinite value, NaN derivative, also on the hidden side of a `where` | safe base inside the `where`; test at T = tfrz exactly and just below |
| melt and freeze tests (`T > tfrz` with ice, `T < tfrz` with liquid above supercooled) | switches | masks; both branches computed with safe inputs |
| ice and liquid after the change (`max(0, ice − melt)`, `min(water − supercooled, ...)`) | zero derivative on the limited side | as CLM; documented |
| frost table and perched table: searches with an exit; interpolation `1/(s2 − s1)` | exits; division by 0 when two levels are equally saturated | masks instead of exits (as `water_table_depth`); safe denominator in the unused branch |
| perched drainage level by level, carrying the remainder | loop with a remainder | `lax.scan` over the levels (as the baseflow); no index scatter into a carry |
| specific yield `(1 + 1000 zwt_perched/sucsat)^(-1/bsw)` | none: the base is ≥ 1 | – |
| evaporation split `liquid/(liquid + ice)` | 0/0 for a dry top level | safe denominator |
| `10^(−e_ice × ice fraction)` | none (smooth); the derivative with respect to `e_ice` is no longer zero | finite-difference check |
| thermal conductivity: frozen or thawed Kersten number chosen at `T ≥ tfrz` | a jump of conductivity at 0 °C, already in the code and in CLM | as CLM; documented |
| root uptake: none from a level with any ice (CLM-ml v2) | a jump of the forward model at the first trace of ice | choice 6 |
| levels below bedrock, padded patches | must not change results or give NaN | every slot computed with the top level's inputs, as now |

Every hydrology parameter, the soil properties, the initial temperature, liquid and ice
are inputs of the compiled program, as now.

---

## 6. Testing and evaluation

### 6.1 Routine level, against CLM's Fortran (golden data)

The Phase 10 harness (`tests/fortran_validation/ctsm_harness/`: CTSM's unit-test
framework, a checked copy of `src/`, drivers that read cases and write 17 digits; relative
tolerance 1e-9), with new drivers:

| Routine | Reached how | Sweeps |
|---|---|---|
| `Phasechange_beta` (private; `SoilTemperatureMod` is not in the test build) | **probe first (12.1):** add the module and what it needs to the copy's build, one visibility line. If it cannot be built there: a driver holding a copy of the routine's statements, compared with the source by the build (the way of the microtopography in decision 143) | temperature from −30 to +10 °C around 0 °C, exactly 0 °C; water from dry to saturated, all liquid, all ice, mixed; soils from sand to clay and organic; `fact` and `dhsdT` over CLM's ranges and CLM-ml's values (`dhsdT = 0`); pond fraction 0 and positive; with choice 2, thin snow from none to more than the energy can melt; both step lengths |
| `PerchedWaterTable` (public; already called by the sequence driver, its outputs not recorded) | driver | temperature profiles: unfrozen, frozen from the top, thawed over frozen, frozen at the bottom level only, several frozen bands; saturation above the frost table from dry to full; water table above and below the frost table; shallow bedrock |
| saturated fraction with a perched table | the existing surface-chain driver, two more inputs | frost table between, above and below the two water tables |
| `PerchedLateralFlow` (public) | driver | perched zone of none, part of a level, several levels; slope flat to steep; drainage larger than the zone holds |
| split of the ground's vapour flux | inside `SoilFluxes`, which cannot be called for one column: a copy of its statements in the driver, compared with the source by the build | liquid only, ice only, mixed, none; evaporation and dew; ground above and below 0 °C |
| `SoilThermProp` with ice | only if 12.1 finds the existing golden cases lack ice or CLM-ml v2's copy differs from `ctsm5.4.004` | |

As in Phase 10, each golden test also checks that its sweep reaches every branch of the
routine.

### 6.2 Properties that need no reference

- **Energy:** heat capacity × temperature change + heat of fusion × ice change equals the
  ground heat flux, to round-off, every step (the recorded soil energy check).
- **Water:** freeze and thaw changes no level's total water; the step's water balance
  closes as now (1e-12 mm), with ice in the storage and the perched drainage in the
  outputs.
- A level changing phase is at 0 °C; no ice above 0 °C and no liquid above the
  supercooled amount below it after the routine, except where the energy ran out.
- **The freezing front against the analytical solution** (Stefan problem: a wet column
  with a cold surface; the front's depth grows as the square root of time), within the
  error of the level spacing: a check of the temperature solve and the phase change
  together that depends on neither CLM nor this code.
- **A soil that never freezes gives the same results with `freeze_thaw: true` as with
  `false`**, bit for bit (CHATS in May; US-MMS May-September 2012).
- `freeze_thaw: false` passes the bit-for-bit and noise-twin tests unchanged, prescribed
  and predicted.
- Levels below bedrock and padded patches change nothing.

### 6.3 Sequences over time, against CLM

The sequence driver of decision 144 extended: each step it adds a given temperature
change to each level (standing for the temperature solve, which is CLM-ml's and tested in
6.2), calls `Phasechange_beta`, then CLM's hydrology in CLM's order with
`PerchedWaterTable` recorded and `PerchedLateralFlow` added. The JAX functions are given
the same series. Scenarios, 160 steps at each step length:

1. a wet soil freezing from the top, then thawing from the top (a perched table forms on
   the frozen levels, drains, and disappears);
2. rain on frozen ground: little infiltration, pond, runoff;
3. frozen at depth throughout, the levels above thawing and refreezing (the saturated
   fraction switches between the perched and the main water table);
4. a nearly dry soil cooled to −20 °C (all water stays supercooled);
5. a saturated soil freezing (ice at the pore space, ice runoff at the top level);
6. frost and sublimation at the surface over days;
7. daily cycles across 0 °C in the top levels;
8. shallow bedrock with a frozen level above it.

Compared as before: one step ahead from the Fortran's state (the acceptance test, 1e-9,
equal Richards sub-step counts) and free running (looser, set when the first sequences
exist). A level within round-off of 0 °C or of the supercooled amount can switch in one
code and not the other; such a step is examined, as a flipped sub-step decision was.

### 6.4 Derivatives, at each step of the work

- **With each routine (12.2-12.4):** forward and reverse derivatives against central
  finite differences in `tests/clm_ml/`, for every input and parameter, in every branch
  of the sweep and on the hazards of 5.2; finite in every lane, padding included.
- **With the assembled soil (12.5):** derivatives through 160-step sequences (ice,
  liquid, temperature and drainage at the end with respect to the soil properties,
  `e_ice`, `perched_baseflow_scalar`, the initial state and the temperature forcing).
- **In the model (12.6-12.7), in the gradient tier**
  (`tests/test_gradient_accuracy_soil_water.py`): a freezing night, a thawing day with a
  perched zone, an unfrozen case (the derivatives of today), and several days across a
  freeze and a thaw in reverse mode; latent heat, soil temperature, ice, runoff and
  drainage with respect to `e_ice`, `perched_baseflow_scalar`, `bsw`, `sucsat`,
  `watsat`, `hksat`, the initial temperature and water, and a canopy parameter. Forward
  mode (`jax.jvp`) on one case. The whole CPU gradient tier, and the GPU tier once.

### 6.5 Coupled runs (understanding, not pass or fail)

- **US-MMS, the winters of 2010-2014** beside CLM's `SOILICE`, `SOILLIQ`, `TSOI`,
  `ZWT_PERCH`, `QDRAI_PERCH` (the CLM run exists). CLM has snow and CLM-ML does not, so
  the frost goes deeper in CLM-ML: the comparison shows by how much, the water and energy
  balances over five years, and that summers are unchanged.
- **The 1° globe**, ten days of January and of July 2000: every patch runs and conserves;
  whether the start is nearer balance with CLM's ice (decision 148 found the July start
  out of balance where CLM's water is partly ice). Needs `SOILLIQ` and `SOILICE` in the
  monthly history files: to check in 12.1.
- CHATS and the comparison with Fortran CLM-ml: unchanged.

### 6.6 Timing

Timing tests of a step and its gradient with `freeze_thaw` on and off (CPU tower, A100
grid). Expected small: no new loop except a 20-level scan. More than ~1.5× is discussed
before going on.

---

## 7. Proposed order of work

Each step ends with its tests passing, its derivative checks, and a short report.

| | Content | Ends with |
|---|---|---|
| 12.0 | This plan; the maintainer's choices (section 8) | decisions recorded (152 onward) |
| 12.1 | **Probes, no model code:** (a) `SoilTemperatureMod` in the harness build and a first call of `Phasechange_beta`; (b) CLM-ml v2's `SoilThermProp` against `ctsm5.4.004`'s, and ice in its golden cases; (c) how CLM's root conductance treats ice; (d) `SOILLIQ`/`SOILICE` in the grid's history files; (e) how deep CLM-ML's soil freezes at US-MMS today, without latent heat | a report; **discussion** |
| 12.2 | **Freeze and thaw**: golden sweeps of `Phasechange_beta`, `soil.supercooled_water` and `soil.phase_change` (not yet in the step), the energy property, the Stefan test, derivatives | golden and property tests; derivative checks; **discussion** of the agreement and of the fit of 3.3 |
| 12.3 | **Frost table and perched water table**; the saturated fraction with it | golden tests; derivative checks |
| 12.4 | **Perched drainage**; the split of the ground's vapour flux; the assembled `soil_hydrology` with real ice; water conservation | golden tests; derivative checks |
| 12.5 | **Sequences** of freeze, thaw and hydrology against CLM (6.3), with derivatives through them | **discussion** of the agreement |
| 12.6 | **Into the model:** the option, state, start of a run, restart, outputs, parameter, energy check, `upgrade`; the choices of section 8 as decided | `freeze_thaw: false` bit for bit; an unfrozen run equal with `true`; a one-day end-to-end test with a frozen start; model derivatives |
| 12.7 | Gradient tier and timing | **discussion** if slower than ~1.5× |
| 12.8 | Coupled runs (6.5), documentation for users, close of the phase | all of `tests/regression`, the slow tests, the docs build |

---

## 8. Decisions for the maintainer

Each with a recommendation; none is taken.

1. **Where freeze and thaw applies.** (a) Only with predicted soil water [recommended:
   liquid and ice are then both the model's own]. (b) Also with prescribed soil water:
   the total water of a level is prescribed and the model splits it into liquid and ice
   with its own phase change. (c) Prescribed runs read `SOILICE` from the history: the
   ice would not match the model's own soil temperature. (b) or (c) would address O9 for
   prescribed runs and can be decided after 12.6.
2. **Snowfall reaching the ground, until there is a snowpack.** Today it enters the soil
   as liquid. With freeze and thaw that liquid freezes in the top level and releases
   heat of fusion that the snow never had: 5 mm of snowfall in a day is 19 W m⁻² of
   heating for that day. Options: (a) as today, with that error; (b) add it to the top
   level as ice, as CLM adds frost: no false heating, but ice beyond the pore space
   leaves as runoff (CLM's rule, already translated), so winter snowfall mostly does not
   recharge the soil; (c) hold it in CLM's store for snow too thin for layers
   (`h2osno_no_layers`), which `Phasechange_beta` melts with the top level's energy
   (row 3 of 3.1; the thermal properties already take it): water and energy are
   conserved and the meltwater arrives at the thaw, but the store has no albedo and does
   not insulate, and CLM uses it only for snow under 1 cm [recommended: it is CLM's code
   in the routine being translated, and the snowpack phase replaces it]. It is a first
   piece of snow, so it is yours to accept or leave out.
3. **Ice at the start of a run.** (a) `SOILLIQ`/`SOILICE` from the CLM history where
   present, otherwise the total water split by the supercooled-water curve at the initial
   temperature [recommended]; (b) always the split; (c) all liquid, the first steps
   freeze it (a burst of heat of fusion at the start).
4. **Latent heat of the ground's vapour flux.** CLM uses the heat of sublimation when
   the top level has ice and no liquid; CLM-ml's flux solution uses the heat of
   vaporization. (a) Leave the canopy as it is and record the difference [recommended
   for this phase: the soil is translated, the canopy is not changed]; (b) CLM's rule in
   the flux-profile solution, a canopy science change with its own tests.
5. **Humidity at the soil surface.** CLM-ml v2 takes it from the liquid water's matric
   potential; CLM from liquid plus ice. In a frozen top level CLM-ml's form gives very
   dry pore air and almost no evaporation. (a) Keep CLM-ml v2's [recommended for this
   phase, with the effect measured in 12.8]; (b) CLM's form when `freeze_thaw` is on.
6. **Roots and ice.** CLM-ml v2 takes no water from a level with any ice, a jump at the
   first trace of ice. (a) Keep it with CLM-ml's roots [recommended]; with CLM's roots
   (decision 151) follow what CLM does, reported in 12.1.
7. **Coupled runs.** (a) US-MMS winters and the globe only [recommended until there is
   snow]; (b) also a cold tower (which one, and a CLM case for it).
8. **Soil properties per level on grids (O88) and the grid's start (O12)** affect the
   globe's start as much as ice does. (a) Not in this phase [recommended]; (b) O88 here.
9. **The default.** `freeze_thaw` stays `false` by default through the phase; whether it
   becomes the default of predicted soil water is decided at the close, or with snow.

---

## 9. Risks

- **`SoilTemperatureMod` may not build in CTSM's unit-test build** (urban and snow
  dependencies). Then the driver holds a checked copy of the routine's statements; the
  golden data are still CLM's arithmetic, compiled by CLM's compiler, but the routine is
  not called in place. Known after 12.1(a).
- **Switches at 0 °C in one-step-ahead comparisons:** a temperature within round-off of
  freezing can melt in one code and not the other. Expected rare (the sweeps place
  temperatures exactly at 0 °C on purpose, where both codes must agree on the
  comparison itself).
- **The fit of 3.3** (`dhsdT = 0`, CLM-ml's `fact`) is this plan's reading of the two
  codes. It is checked by the energy property and the Stefan test, and discussed at 12.2
  before anything enters the model.
- **Winters without snow** will look wrong (deep frost, early spring runoff from frozen
  ground). The phase does not try to correct that; the snowpack does.
- **Oscillation across 0 °C between steps** (the canopy solved against a top level that
  freeze and thaw has reset): looked for in scenario 7 and in the US-MMS winters; if
  found, it is brought to you, not damped.
