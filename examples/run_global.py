"""
Set up and run a CLM-ML-JAX case on the 1° CLM grid.

Edit the settings in step 1 and run the script:

    python examples/run_global.py

The case is a folder of four files (run.yaml, domain.yaml, model.yaml, params.yaml) made
from the compset `grid_1deg_2000` (CLM 5.4 surface data at 0.9x1.25°, CRUJRA meteorology,
CTSM 5.4 monthly soil water, CLM6 parameters; inputs on NCAR's Glade storage). The script
writes the settings below into those files, so the folder records the run, and
`clm-ml run CASE_DIR` repeats it (e.g. from a batch job). Anything else (output variables
and frequency, model options, parameters) is changed by editing the files.

The whole globe (about 72 600 patches on 14 300 gridcells) needs a GPU: a 10-day chunk
takes about 15 minutes on one A100. Run the script on a GPU node, or set RUN_NOW = False
and submit `clm-ml run CASE_DIR` as a batch job. A small region runs on a CPU (set
JAX_PLATFORMS=cpu on a node without a GPU).
"""

import os
from pathlib import Path

import clm_ml

# --- 1. Settings --------------------------------------------------------------------------

CASE_DIR = Path(os.environ["SCRATCH"]) / "global_1mon_test"  # the case folder
COMPSET = "grid_1deg_2000"  # what a new case starts from

START = "2000-01-01T00:00"  # first time step
LENGTH = "31 days"  # e.g. "1 day", "31 days", "until 2001-01-01T00:00"
CALENDAR = "noleap"  # noleap (as the CRUJRA forcing) or gregorian

REGION = "global"  # or a box: "{lat: [35, 40], lon: [-100, -95]}" (degrees N, degrees E)

DEVICES = 1  # GPUs the patches are spread over (4 on a full Derecho GPU node)
CHUNK = "10 days"  # forcing read and compiled per chunk: "N days" or "1 month"

RUN_NOW = True  # False: set the case up only, then `clm-ml run CASE_DIR`

# --- 2. Create the case from the compset (an existing case is kept, with its edits) --------

if not CASE_DIR.exists():
    clm_ml.new_experiment(CASE_DIR, COMPSET)
    print(f"created {CASE_DIR} from the compset {COMPSET}")

# --- 3. Write the settings into the case's files --------------------------------------------

settings = [
    f"run.time.start={START}",
    f"run.time.length={LENGTH}",
    f"run.time.calendar={CALENDAR}",
    f"domain.grid.region={REGION}",
    f"run.execution.devices={DEVICES}",
    f"run.execution.chunk={CHUNK}",
]
case = clm_ml.load_experiment(CASE_DIR, settings)  # reads and checks the four files
case.write_copy(CASE_DIR)

# --- 4. Check: what the run will use and where each value comes from ------------------------

print(case.summary())

# --- 5. Run ---------------------------------------------------------------------------------
# Output: netCDF files in CASE_DIR/output, per output stream and month (run.yaml output):
# *_patch.nc per patch, *_gridcell.nc on the latitude-longitude grid.

if RUN_NOW:
    case.execute(return_outputs=False)
    print(f"output: {case.output_directory}")
else:
    print(f"case ready: clm-ml run {CASE_DIR}")
