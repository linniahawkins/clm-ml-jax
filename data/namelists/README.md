# CLM-ML Offline Executable

Python/JAX equivalent of the Fortran `prgm.exe` offline executable.

## Installation

From the project root directory:

```bash
# Install in editable mode (replaces 'make' step from Fortran Makefile)
pip install -e .
```

This will install the `clm-ml-offline` console script.

## Usage

### Method 1: Console Script (Recommended)

After installation, run directly:

```bash
# With namelist from stdin (mirrors Fortran: ./prgm.exe < nl.CHATS7.05.2007)
clm-ml-offline < nl.CHATS7.05.2007

# With namelist as argument
clm-ml-offline nl.CHATS7.05.2007
```

### Method 2: Direct Python Execution

Without installation, from the project root:

```bash
# With stdin
python -m offline_executable.main < nl.CHATS7.05.2007

# With argument
python -m offline_executable.main nl.CHATS7.05.2007
```

### Method 3: From This Directory

```bash
# With stdin
cd src/offline_executable
python main.py < ../../input_files/nl.CHATS7.05.2007

# With argument
python main.py ../../input_files/nl.CHATS7.05.2007
```

## Namelist Format

The namelist file should contain a `&clm_inparm` section with parameters:

```fortran
&clm_inparm
  tower_site = "CHATS"
  iyear      = 2007
  imonth     = 5
  finidat    = "path/to/initial/conditions"
  fsurdat    = "path/to/surface/data"
  ntimes     = 2880    ! Number of timesteps
/
```

## Migration from Fortran Makefile

**Fortran Workflow (old):**
```bash
make           # Compile
./prgm.exe < nl.CHATS7.05.2007   # Run
make clean     # Clean up
```

**Python Workflow (new):**
```bash
pip install -e .                  # Install (once)
clm-ml-offline < nl.CHATS7.05.2007   # Run
```

No compilation or cleanup needed!

## What Replaced the Makefile

| Makefile Component | Python Equivalent |
|-------------------|-------------------|
| `FC = gfortran` | Python interpreter |
| `FFLAGS = ...` | N/A (interpreted) |
| `make` | `pip install -e .` |
| `./prgm.exe` | `clm-ml-offline` |
| `make clean` | N/A (no build artifacts) |
| Object files (`.o`) | Bytecode (`.pyc`, auto-managed) |

## Dependencies

Required packages (auto-installed via `pip install -e .`):
- `jax` - JAX array library
- `jaxlib` - JAX backend
- `numpy` - Numerical arrays
- `f90nml` - Fortran namelist parser

## Troubleshooting

**Import errors:**
```bash
# Make sure you're running from project root or have installed with pip
pip install -e .
```

**Module not found:**
```bash
# Add project root to PYTHONPATH if not using pip install
export PYTHONPATH=/path/to/clm-ml-jax:$PYTHONPATH
```

**f90nml not found:**
```bash
pip install f90nml
```
