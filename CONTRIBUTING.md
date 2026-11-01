# Contributing to CLM-ML-JAX

Thank you for your interest in contributing! This guide is aimed at atmospheric and land-surface scientists who want to add physics, fix bugs, or improve documentation.

## Development Setup

```bash
# Clone with submodules (Fortran reference code lives in CLM-ml_v1/)
git clone --recurse-submodules https://github.com/AyaLahlou/clm-ml-jax.git
cd clm-ml-jax

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify the install
clm-ml --help
pytest tests/clm_ml tests/fortran_validation/ -m "not slow" -q
```

## Repository Layout

```
src/clm_ml/
  physics/      # the processes, pure functions on 0-based arrays  ← most contributions go here
  step.py       # one time step of one patch (State + inputs + constants -> StepState)
  state.py      # State, Diagnostics, PatchLayers
  setup/        # every patch's constants and initial state (pure functions)
  driver.py     # compiled chunks over all patches, output streams, checks, restarts
  io/           # tower and CLM history readers, restarts (netCDF), legacy .out writer
  run.py, model.py, config.py, params.py, outputs.py, cli.py
data/           # CHATS7 forcing and the CLM-ml namelists (test and example data)
tests/
  clm_ml/              # unit tests
  fortran_validation/  # 160+ tests vs. Fortran golden values (REL_TOL=1e-9)
  regression/          # end-to-end runs against committed references
```

## Adding a Physics Module

1. **Write it by process** in `src/clm_ml/processes/`: pure functions on 0-based arrays
   (layers from the ground up, leaves sunlit/shaded, bands visible/near-infrared); everything
   read is an argument and everything changed is returned.
2. **Name the Fortran source** in the docstring (`Fortran reference: MLFooMod.F90::Bar`).
3. **Masks, not loops:** compute on every layer slot and keep results with the masks of
   `clm_ml.processes.layers`; no Python loops over layers inside the step.
4. **64-bit precision**: `clm_ml` enables float64; constants are Python floats.

## Writing Tests

Test files mirror the source tree:
`src/clm_ml/processes/foo.py` → `tests/clm_ml/test_processes_foo.py`

For Fortran validation tests, capture golden I/O from the Fortran build and add a JSON file to `tests/fortran_validation/golden_IO/`, then add a parametrized case in `tests/fortran_validation/test_golden_jax.py`.

Tolerances for validation: `REL_TOL = 1e-9`, `ABS_TOL = 1e-15`.

Run only fast tests during development:
```bash
pytest tests/fortran_validation/ -m "not slow" -q
```

## Code Style

```bash
black --line-length 100 src/ tests/ timing/   # auto-format
ruff check src/ tests/ timing/                # linting
```

Both are enforced in CI on every pull request.

## Submitting a Pull Request

1. Fork the repository and create a feature branch (`git checkout -b feat/my-physics`).
2. Make your changes with tests.
3. Run the test suite locally (`pytest tests/fortran_validation/ -m "not slow" -q`).
4. Open a PR against `main` with a description of what changed and why.

CI will automatically run lint checks and the fortran validation suite. Please ensure both pass before requesting review.

## Reporting Bugs

Open an issue at https://github.com/AyaLahlou/clm-ml-jax/issues with:
- A minimal reproduction script
- The Python and JAX versions (`python --version`, `python -c "import jax; print(jax.__version__)"`)
- The expected vs. actual output
