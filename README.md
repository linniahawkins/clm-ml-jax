# CLM-ML-JAX: Community Land Model with Multi-Layer Canopy in JAX

[![CI](https://github.com/linniahawkins/clm-ml-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/linniahawkins/clm-ml-jax/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE)

> CLM-ml v2, the multilayer canopy model developed for the Community Land Model (Bonan et al.
> 2021, 2026), in JAX: the same results as the Fortran model, on CPUs or GPUs, with gradients.

**Reference model in Fortran**: [CLM-ml_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS)

### Features

- Reproduces the Fortran CLM-ml v2 results (CHATS walnut orchard test case; validated
  routine by routine and end to end)
- From one flux tower to global runs: patches on gridcells, prescribed meteorology, leaf area
  and soil water, configurable model step and output streams
- Experiment folders: every setting written out and explained in four YAML files
- GPU execution, ensembles (`vmap`) and gradients with respect to parameters, forcing and
  initial state (`jax.grad`) for calibration and sensitivity studies

---

## Installation

```bash
# 1. Clone and navigate to repository
git clone https://github.com/linniahawkins/clm-ml-jax.git
cd clm-ml-jax

# 2. Install package and dependencies
pip install -e .

# 3. Verify installation
clm-ml --help
```

This installs the `clm-ml` command and the `clm_ml` package (the only package in `src/`).

---

## Run a Simulation

### Experiment folders (`clm-ml`)

An experiment is a folder with four files, and every setting the model has is written out in
them with an explanation (keys marked REQUIRED have no default):

| File | What it sets |
|---|---|
| `run.yaml` | when and with what: dates, calendar, forcing and land-data files, initial conditions, output streams, chunks and restarts |
| `domain.yaml` | where: towers (location, forcing height, soil, patches) or a grid (CLM surface dataset, region) |
| `model.yaml` | how: physics schemes and numerics (a change recompiles the model) |
| `params.yaml` | which plants: every parameter of every PFT (the numbers gradients come back for) |

Start from a preset; ready-made folders are in `configs/` (`configs/README.md`):

```bash
clm-ml new my_site --from chats7_1day      # write the four complete files
clm-ml check my_site                        # validate everything and print a summary
clm-ml run my_site                          # run; one netCDF file per output stream
clm-ml run my_site --set 'run.time.length=7 days' --set model.processes.stomatal_conductance.scheme=medlyn
clm-ml upgrade my_site                      # after updating clm-ml: add new settings
```

Each run also writes the four files as it read them into `OUTPUT/config/`, a folder that
repeats the run. From Python:

```python
import clm_ml

exp = clm_ml.load_experiment("my_site")
out = exp.execute()["halfhourly"]   # one result per output stream (run.yaml output.streams)
ds = out.to_xarray()                # dims (time, patch, layer, ...), with units
ds["latent_heat_flux"].plot()
```

Output streams are accumulated during the run, so memory does not grow with the number of
model steps. The parameter presets are `clm5_defaults`, `chats` (the CHATS walnut-orchard
values of Bonan, Burns & Patton 2026, Table 2) and `clm6` (CTSM 6.0); a folder's
params.yaml is a complete copy of one. The design: `docs/design/experiment_folders.md`.

### From a CLM-ml namelist

The Fortran-style namelists (`data/namelists/nl.*`) are converted to an experiment folder
with `clm-ml convert data/namelists/nl.CHATS7.1day -o chats7/`; the former `clm-ml-offline` driver was retired in Phase 7. The
Fortran-format `.out` files are still written with `output.legacy_ascii: true` (one patch,
for the comparison with the Fortran model).

---

## Validation Tests

We compare every JAX routine against golden reference values of the Fortran build:

```bash
/path/to/conda/envs/clm-ml-jax/bin/python -m pytest tests/fortran_validation/ -q
```

Golden JSON files live in `tests/fortran_validation/golden_IO/`.

### End-to-end regression test

A 50-step CHATS7 run (one full diurnal cycle) is compared, field by field and at full
precision, against a committed reference. Two checks: refactors must reproduce it bit for bit,
and every change must stay within the model's own round-off sensitivity, measured by two
"noise twin" runs (see `docs/design/numerics_and_regression.md`):

```bash
python -m pytest tests/regression/ -m regression -q     # ~20 min on CPU
```

If results change **on purpose** (a science change or bug fix), or a change alters only
round-off and passes the noise-twin test, regenerate the reference and its twins and note it
in the PR:

```bash
python tests/regression/record_run.py --update-reference
```

### Timing test

```bash
python timing/time_offline.py     # wall-clock time of a 50-step `clm-ml run`; timing/results.jsonl
python timing/time_model.py       # the compiled chunk and its gradient; timing/results_model.jsonl
```

---

## Development

### Adding a New Process or Scheme

1. Write the process as pure functions on 0-based arrays in `src/clm_ml/processes/` (one file
   per process, e.g. `photosynthesis.py`): everything it reads is an argument, everything it
   changes is returned; compute on every layer slot and keep results with the layer masks
   (`clm_ml.processes.layers`).
2. Name the Fortran source in the docstring (`Fortran reference: MLFooMod.F90::Bar`).
3. Static choices (schemes) come from `model.yaml` through `clm_ml.config`; numeric
   parameters are per PFT in `params.yaml` (`clm_ml.params`) and stay differentiable.
4. Call it from the time step (`src/clm_ml/step.py`, `src/clm_ml/processes/canopy.py`) and
   add unit tests in `tests/clm_ml/`.
5. Register output variables in `src/clm_ml/outputs.py` (`VARIABLES`).

### Adding a New Tower

A tower is an entry of an experiment's `domain.yaml`: location, forcing height, soil (texture
as percentages or a USDA class, organic matter, colour, bedrock) and its patches (PFT,
weight, canopy height, root biomass, area profiles, layer counts). Start from a preset
(`clm-ml new my_site --from chats7_1day`) and edit its `domain.yaml`; the forcing files and
their interval go in `run.yaml`.

### Model options

How each process is represented is in `model.yaml` under `processes` (e.g.
`processes.stomatal_conductance.scheme: wue | medlyn | ball_berry`), with `numerics`,
`canopy_layers` and `soil`; any key can be changed on the command line with
`--set model.<section>.<key>=<value>`. Every setting is explained in each folder's
`model.yaml` (e.g. `configs/chats7_1day/model.yaml`).

---

## Contributing

See `CONTRIBUTING.md`: how the code is organized, how to add a process or scheme, and the
tests every change must pass. Bugs and questions: open an issue on GitHub.

---

## Citation

Please cite the software (`CITATION.cff`) and the model it translates:

```bibtex
@software{clm_ml_jax,
  title  = {CLM-ML-JAX: the CLM-ml v2 multilayer canopy model in JAX},
  author = {Hawkins, Linnia and Lahlou, Aya},
  year   = {2026},
  url    = {https://github.com/linniahawkins/clm-ml-jax},
  note   = {JAX translation of CLM-ml v2 (Bonan et al.)}
}
```

The multilayer canopy model: Bonan, G. B., Burns, S. P., and Patton, E. G. (2026), Beyond
surface fluxes: Observational and computational needs of multilayer canopy models - A walnut
orchard test case, *Agricultural and Forest Meteorology*, 378, 110960,
doi:10.1016/j.agrformet.2025.110960; and Bonan et al. (2021), *Agricultural and Forest
Meteorology*, 306, 108435, doi:10.1016/j.agrformet.2021.108435.

---

## License

This project is licensed under the **BSD-3-Clause License**. See [LICENSE](LICENSE) for details.

---

⭐ **If this helps your research, please star the repository!**

🐛 **Found a bug?** Please [open an issue](https://github.com/linniahawkins/clm-ml-jax/issues)

