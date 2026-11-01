# Configurations

Runnable experiments, one folder each. Each holds a `run.yaml` (the experiment: dates, domain,
forcing files, output) that names a `model.yaml` (physics schemes and numerics) and a params
file (parameters per PFT). Paths inside are relative to each `run.yaml`.

| Folder | What it runs |
|---|---|
| `chats7_1day/` | One day (1 May 2007) at the CHATS walnut orchard, the CLM-ml v2 test case (Bonan, Burns & Patton 2026). The quickest complete run. |
| `chats7_may2007_fortran/` | May 2007 at CHATS with the Fortran acclimation temperature and the legacy `.out` files: the run compared with Fortran CLM-ml v2 (`tests/test_fortran_python_comparison.py`). |
| `grid_1deg_2000/` | A 1° grid run of 2000 (CRUJRA forcing, CLM6 parameters, CTSM 5.4 soil moisture) over 30-50 N, 110-80 W; inputs on NCAR storage. |

```bash
clm-ml check configs/chats7_1day/run.yaml   # validate and summarize, no run
clm-ml run configs/chats7_1day/run.yaml     # output in configs/chats7_1day/output/
clm-ml run configs/chats7_1day/run.yaml --set time.days=3 --set model.physics.stomatal_conductance=medlyn
```

## The files

- **`model.yaml`** (this folder): every model setting at its default (CLM-ml v2 as run for
  CHATS, the configuration validated against Fortran), each explained. It is generated from
  `src/clm_ml/config.py` (`clm-ml model-yaml -o configs/model.yaml`); do not edit it by hand.
  An experiment that changes settings carries its own complete `model.yaml`
  (`clm-ml model-yaml --from its/model.yaml -o its/model.yaml` rewrites it with every key and
  its explanation). `tests/clm_ml/test_configs.py` checks that every `model.yaml` here is
  complete and current.
- **Params files** are complete tables: every parameter of every PFT 1-16, nothing inherited.
  The built-in ones are named in `run.yaml` by name (`clm5_defaults`, `chats`, `clm6`; in
  `src/clm_ml/data/params/`). For your own parameter set, copy one and edit it:
  `clm-ml params chats -o mine.yaml`, then `params: mine.yaml`.
- **`towers_example.yaml`**: how to describe your own towers (location, soil, patches);
  name it in `run.yaml` with `domain.towers_file`.
