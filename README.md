# CLM-ML-JAX: Community Land Model with Multi-Layer Canopy in JAX

[![CI](https://github.com/AyaLahlou/clm-ml-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/AyaLahlou/clm-ml-jax/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/AyaLahlou/clm-ml-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/AyaLahlou/clm-ml-jax)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE)

> A complete JAX/Python translation of the Community Land Model (CLM) Multi-Layer Canopy physics, enabling  GPU acceleration 

 **Reference CLM-ML in Fortran**: [CLM-ML_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS)


### Features

-  Full Fortran→Python/JAX translation maintaining scientific accuracy
-  Tower-site offline driver for point-scale simulations with prescribed meteorology
-  GPU-accelerated physics through JAX JIT compilation
-  Vectorization with vmap for ensemble and sensitivity analyses
-  Namelist-driven configuration (mirrors Fortran interface)
-  Modular architecture 

---

## Installation

```bash
# 1. Clone and navigate to repository
git clone https://github.com/AyaLahlou/clm-ml-jax.git
cd clm-ml-jax

# 2. Install package and dependencies
pip install -e .

# 3. Verify installation
clm-ml-offline --help
```

This installs the `clm-ml-offline` command and all physics modules (`clm_share`, `clm_src_*`, `multilayer_canopy`, etc.).

---

## Run a Simulation

Execute a 1-day tower-site simulation with the CHATS7 site:

```bash
# Using the installed command (recommended)
clm-ml-offline < nl.CHATS7.1day

# Or with explicit namelist argument  
clm-ml-offline input_files/nl.CHATS7.1day

# Or run directly with Python
python -m offline_executable.main input_files/nl.CHATS7.1day
```

Output files are written to the `output_files/` directory.

---

## Validation Tests

We compare every JAX routine against golden reference values of the Fortran build:

```bash
/path/to/conda/envs/clm-ml-jax/bin/python -m pytest tests/fortran_validation/ -q
```

Golden JSON files live in `tests/fortran_validation/golden_IO/`.

### End-to-end regression test

A 50-step CHATS7 run (one full diurnal cycle) is compared, field by field and at full
precision, against a committed reference (`rtol=1e-6`). Any change that alters model results
fails this test:

```bash
python -m pytest tests/regression/ -m regression -q     # ~5 min on CPU
```

If results change **on purpose** (a science change or bug fix), regenerate the reference
and note it in the PR:

```bash
python tests/regression/record_run.py --update-reference
```

### Timing test

```bash
python timing/time_offline.py     # wall-clock time of a 50-step run; appends to timing/results.jsonl
```

---

## Development

### Adding a New Physics Module

1. Create `multilayer_canopy/MyPhysicsMod.py` mirroring Fortran module structure
2. Include Fortran source reference in docstring:
   ```python
   """
   JAX translation of Fortran module MyPhysics.

   Mirrors Fortran subroutine ``my_physics_sub`` (lines 45-120).
   """
   ```
3. Use module-level globals from `MLclm_varctl.py` for configuration
4. Add tests in `tests/multilayer_canopy/test_my_physics.py`
5. Register output variables in `clm_src_main/histFileMod.py`

### Adding a New Tower Site

Edit `offline_driver/TowerDataMod.py`:

```python
ntower = 16  # Increment from 15

# Extend all arrays by one element
tower_id[16] = 'MYNEWSITE'
tower_lon[16] = -120.5
tower_lat[16] = 38.2
tower_elev[16] = 500.0
# ... (add other fields)
```

Then create a namelist file: `src/offline_executable/nl.MYNEWSITE.1day`

### Configuration & Switches

Global physics switches are in `multilayer_canopy/MLclm_varctl.py`—change directly, no config object needed:

```python
gs_type = 0              # Stomatal model: 0=Medlyn, 1=Ball-Berry, 2=WUE
flux_profile_type = 1    # Flux-profile: -1=dataset, 0=well-mixed, 1=implicit
runge_kutta_type = 41    # Time integration: 10=Euler, 21=2nd-order, 41=4th-order
dtime_ml = 60.0          # Sub-step interval (s); must divide CLM timestep
```

---

## Contributing

Contributions are welcome! Here's how you can help: 

1. **Report Bugs**: Open an issue with reproduction steps
2. **Suggest Features**: Open an issue with use case description
3. **Submit PRs**: Fork, create feature branch, submit PR
4. **Improve Docs**: Documentation improvements are always welcome

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on improving the project

---

## Citation


```bibtex
@software{clm_ml_jax,
  title={CLM-ML-JAX: Community Land Model with Multi-Layer Canopy in JAX},
  author={Lahlou, Aya},
  year={2024},
  url={https://github.com/AyaLahlou/clm-ml-jax},
  note={JAX translation of CTSM CLM-ML}
}
```

---

## License

This project is licensed under the **BSD-3-Clause License**. See [LICENSE](LICENSE) for details.

---

⭐ **If this helps your research, please star the repository!**

🐛 **Found a bug?** Please [open an issue](https://github.com/AyaLahlou/clm-ml-jax/issues)

