# CTSM test harness: golden data from CLM's own soil water Fortran

The golden files `../golden_IO/test_Soil*.json` (and later the other soil water routines')
are made here: CLM's Fortran (CTSM `ctsm5.4.004`) is run for one idealized soil column on
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
   `SoilWaterPlantSinkMod`, `SurfaceWaterMod`, `TridiagonalMod`.
2. **Visibility:** on a short list of lines of `SoilWaterMovementMod.F90`, `private` becomes
   `public`, so that a driver can call a routine that CLM keeps internal. No science line
   changes.
3. **The harness's files** (`fortran/`): a six-line stand-in for `controlMod` (the name of
   the namelist file) and the test directory `ctsm_harness_test/`.

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

A **driver** reads `<work>/cases/<routine>.nml` (one namelist group `&case ... /` per case),
calls the CLM routine for each case, and writes `<work>/out/<routine>.out`: a line `case=i`,
then `name=value` lines with 17 significant digits (`name(j)=value` for arrays, `j` being
CLM's 1-based level).

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
4. An adapter in `../test_golden_jax.py`, keyed `ctsm_harness:<routine>`.
