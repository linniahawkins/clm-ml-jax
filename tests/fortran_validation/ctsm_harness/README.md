# CTSM test harness: golden data from CLM's own soil water Fortran

The golden files of CLM's soil routines (`../golden_IO/`: soil water, runoff, drainage, frozen
soil) are made here: CLM's Fortran (CTSM `ctsm5.4.004`) is run for one idealized soil column on
inputs that sweep the cases a routine can meet, and its outputs are recorded.
`../test_golden_jax.py` then requires the JAX functions to reproduce them (relative
tolerance 1e-9). Running the tests needs none of this; the harness is only needed to remake
or extend the golden data, on Derecho.

## How it works

The harness is a set of extra tests inside **CTSM's own unit-test framework** (pFUnit, built
by CIME's `run_tests.py`). That framework already has stand-ins for MPI, history and file
input, and helpers that set up a single vegetated column.

The CTSM checkout is never modified. `build.py` copies its `src/` to a work directory on
scratch and makes three kinds of change to the copy:

1. **Sources added to the test build** (CTSM's test build compiles only the files its own
   tests use): the Clapp-Hornberger retention curve, `SoilWaterMovementMod`,
   `SoilWaterPlantSinkMod`, `SurfaceWaterMod`, `TridiagonalMod`; for frozen soil,
   `SoilTemperatureMod` with `BandDiagonalMod` and `UrbBuildTempOleson2015Mod`.
2. **Visibility:** on a short list of lines (`VISIBILITY` in `build.py`: routines of
   `SoilWaterMovementMod.F90`, `Phasechange_beta` and `SoilThermProp` of
   `SoilTemperatureMod.F90`, and the parameter variables of the runoff, surface water and
   soil hydrology modules, which CLM reads from files), `private` becomes `public`, so that
   a driver can call a routine that CLM keeps internal or set a parameter. No science line
   changes.
3. **The harness's files** (`fortran/`): a six-line stand-in for `controlMod` (the name of
   the namelist file), a stand-in for `UrbanTimeVarType` (the type alone: CLM's module
   reads urban data streams) and the test directory `ctsm_harness_test/`.

Before building, `build.py` compares the copy with the checkout file by file and stops unless
the only differences are those three. It also stops if the checkout is not at `ctsm5.4.004`
or has local changes in `src/`.

| File | What it is |
|---|---|
| `build.py` | makes the copy, checks it, builds and runs the tests in a batch job (`qcmd`, account P93300041) |
| `generate_golden.py` | writes the cases, calls `build.py`, turns the drivers' outputs into `golden_IO/*.json` |
| `fortran/ctsm_harness_test/HarnessColumnMod.F90` | the column: CLM's 20 soil levels, the fields the routines read, the CLM6 default soil water namelist, parameters of `ctsm60_params.c251211.nc` |
| `fortran/ctsm_harness_test/HarnessIOMod.F90` | reading case files, writing outputs |
| `fortran/ctsm_harness_test/test_column.pf` | checks of the column itself: the water table routine and the Richards solver run and conserve water |
| `fortran/ctsm_harness_test/test_retention_curve.pf` | drivers of `soil_hk` and `soil_suction` |
| `fortran/ctsm_harness_test/test_soil_water.pf` | drivers of the Richards solver (`SoilWater`): single calls, and sequences of calls with the water carried from step to step |
| `fortran/ctsm_harness_test/test_hydrology_sequence.pf` | driver of the whole soil hydrology over many steps: CLM's routines in CLM's order, the state carried, forced by time series |
| `fortran/ctsm_harness_test/test_frozen_soil.pf` | drivers of the routines for frozen soil: the thermal properties (`SoilThermProp`), the phase change (`Phasechange_beta`), the frost table and perched water table (`PerchedWaterTable`), the saturated fraction with them, the perched zone's drainage (`PerchedLateralFlow`); and a copy of the statements of `SoilFluxes` that divide the ground's vapour flux between liquid water and ice |
| `fortran/ctsm_harness_test/test_frozen_sequence.pf` | driver of a column that freezes and thaws over many steps: CLM's routines in the order of `clm_driver` (new snow, ponded fraction, thermal properties, phase change, vapour flux, hydrology with the perched water table and its drainage), the temperature solution stood for by a relaxation toward a given temperature |
| `fortran/ctsm_harness_test/test_surface_and_drainage.pf` | drivers of the surface chain (saturated-area runoff, infiltration limit, pond, infiltration), the pond fraction, the water table, dew and drainage; and a copy of CLM's statements for the microtopography and the pond threshold, which CLM computes inside initialization routines that cannot be called for one column |

A **driver** reads `<work>/cases/<routine>.nml` (one namelist group `&case ... /` per case),
calls the CLM routine for each case, and writes `<work>/out/<routine>.out`: a line `case=i`,
then `name=value` lines with 17 significant digits (`name(j)=value` for arrays, `j` being
CLM's 1-based level). A sequence driver writes a line `step=k` before the outputs of each
model step.

Golden files of scalar cases are tested by `../test_golden_jax.py` (step 4 below). The
solver's cases are arrays over the soil levels and are tested together, in one vectorized
call, by `../test_golden_soil_water.py`; those of the surface, water table and drainage
routines by `../test_golden_soil_hydrology.py`, the sequences of the whole hydrology by
`../test_golden_hydrology_sequences.py`, the routines for frozen soil by
`../test_golden_frozen_soil.py` and the sequences of a freezing soil by
`../test_golden_frozen_sequences.py`.

## Remaking the golden data

    python tests/fortran_validation/ctsm_harness/generate_golden.py            # every routine
    python tests/fortran_validation/ctsm_harness/generate_golden.py soil_hk    # one

The first build takes a few minutes (all of CTSM's test build); later ones compile only
what changed. The cases are random draws with a fixed seed plus hand-picked edge values, so
a rerun gives the same cases and, with the same compiler, the same numbers. Work directory:
`$SCRATCH/ctsm_harness` (`--work-dir` to change); the log of the last build and run is
`last_run.log` there, and the tests' printed output is in
`build/__command_line_test__/__command_line_test__/Testing/Temporary/LastTest.log`.

`python tests/fortran_validation/ctsm_harness/build.py` alone builds and runs all of CTSM's
unit tests together with the harness's.

## Adding a routine

1. A driver in a `.pf` file of `fortran/ctsm_harness_test/` (listed in its `CMakeLists.txt`):
   read the case's namelist, fill the column's fields, call the routine, `put` the outputs.
2. If the routine is private in CLM, add its `private ::` line to `VISIBILITY` in `build.py`.
3. A case generator and an entry of `ROUTINES` in `generate_golden.py`.
4. An adapter in `../test_golden_jax.py`, keyed `ctsm_harness:<routine>` (or, for cases with
   arrays, a test of its own like `../test_golden_soil_water.py`).

## Practical notes

- **Debugging on a login node, without a batch job:** `source <work>/build/.env_mach_specific.sh`,
  then `make ctsm_harness` in `<work>/build/__command_line_test__/__command_line_test__` and
  run `clm_biogeophys_test/ctsm_harness_test/ctsm_harness`. `gdb` fails on this build (an
  internal error on Fortran types): trace with `write` statements.
- **A test file added to the harness** is compiled but not run unless pFUnit's main program
  is compiled again: `build.py` touches it before every build.
- **The column is unfrozen and snow-free by construction** (`HarnessColumnMod`): the frost
  table and the perched water table are as `PerchedWaterTable` leaves them above freezing.
- **CLM constants written without `_r8` are single precision** (`sat_lev=0.9`, `1.0e-8`):
  look for them in any routine translated later; the JAX code must use the same value.
- **Very dry clay beside wet levels:** levels at the dry bound of the sweeps (relative
  saturation 0.01) in clay have matric potentials of 1e20 mm and more; they make a
  sub-step's matrix ill-conditioned and the result sensitive to the last digit.
- **The golden tests' absolute guard (1e-15)** passes any value smaller than that, such as
  the conductivity of very dry clay: compare such outputs in relative terms when judging
  agreement.
- **Found in `ctsm5.4.004`:** the PLUMBER2 user mods' forcing-stream settings are ignored
  (the stream is named `PLUMBER2.US-MMS`); `QCHARGE` is not a history field in the
  configuration used at US-MMS.
