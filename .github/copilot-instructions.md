# CLM-ML-JAX AI Coding Agent Instructions

## Project Overview

This project uses a multi-agent LLM system to translate the Community Land Model (CLM) from Fortran to Python/JAX, preserving exact scientific accuracy while enabling modern optimization. The translation system itself is AI-powered using Claude Sonnet 4.5.

**Key components:**
- **Translated JAX modules** ([src/](../src/)): 32,689 LOC of physics code organized by CLM subsystem
- **Translation system** ([jax-agents/](../jax-agents/)): Multi-agent workflow (TranslatorAgent, TestAgent, RepairAgent)
- **Test suite** ([tests/](../tests/)): 68 pytest files with fixtures for state management
- **Static analysis** ([jax-agents/static_analysis_output/](../jax-agents/static_analysis_output/)): Pre-analyzed Fortran module structure (311 translation units, 68 modules)

## Critical Architecture Patterns

### 1. Immutable State with NamedTuples

**Core principle**: All state objects use NamedTuples (not classes!) for JAX compatibility. Use `._replace()` to "mutate":

```python
from typing import NamedTuple

class FrictionVelType(NamedTuple):
    fv_patch: Array  # (n_patches,)
    u10_clm_patch: Array

# Update state immutably
new_state = old_state._replace(fv_patch=new_values)
```

See [FrictionVelocityMod.py](../src/clm_src_biogeophys/FrictionVelocityMod.py) for the canonical pattern.

### 2. Global State Singletons

Module-level state instances are stored in [clm_instMod.py](../src/clm_src_main/clm_instMod.py):

```python
from clm_src_main import clm_instMod
canopystate_inst = clm_instMod.canopystate_inst  # Access singleton
```

**Test setup caveat**: [conftest.py](../tests/conftest.py) wraps NamedTuples in `MutableWrapper` to support legacy driver code expecting mutability. Pure JAX functions should unwrap via `._data` if needed.

### 3. Parameter Initialization Pattern

Always call `clm_varpar_init()` before using layer parameters:

```python
from clm_src_main import clm_varpar
clm_varpar.clm_varpar_init()  # Sets nlevgrnd, nlevsno, nlevsoi based on CLM version
from clm_src_main.clm_varpar import nlevgrnd, nlevsno  # Import AFTER init
```

The model has the CLM soil layer structure only (`nlevsoi = 20`, `nlevgrnd = 25`: `clm_ml.setup.soil`; decision 37).

### 4. Vectorization with `jax.vmap`

Replace Fortran DO loops with vmap:

```python
# Fortran: DO j = 1, nlevgrnd
#            rootfr(j) = compute_layer(j)
#          END DO

# JAX:
rootfr = jax.vmap(compute_single_layer)(jnp.arange(1, nlevgrnd + 1))
```

Examples in [SoilStateInitTimeConstMod.py](../src/clm_src_biogeophys/SoilStateInitTimeConstMod.py#L164).

### 5. Array Indexing Convention

**Fortran vs JAX indexing**:
- Fortran uses 1-based indexing: arrays(1:nlevgrnd)
- Python/JAX uses 0-based indexing: arrays[0:nlevgrnd]
- When translating, adjust loop bounds and array access accordingly

**Snow layer indexing**:
- Snow layers use negative indices in Fortran: -nlevsno+1 to 0
- In JAX, store in array with nlevsno leading entries (index 0 to nlevsno-1)
- Access snow data: `array[0:nlevsno]`, soil data: `array[nlevsno:]`

See [ColumnType.py](../src/clm_src_main/ColumnType.py) for layer indexing patterns.

## Translation Workflow

### Using the Agent System

**Primary interface**: [run_translation_workflow.sh](../jax-agents/run_translation_workflow.sh)

```bash
cd jax-agents

# Complete workflow: translate → test → repair
./run_translation_workflow.sh --all --modules "SoilTemperatureMod,WaterFluxType"

# Just translate (iterative unit-by-unit approach)
./run_translation_workflow.sh --translate --modules "SoilTemperatureMod"

# Generate tests for existing translations
./run_translation_workflow.sh --test

# Auto-repair failed tests
./run_translation_workflow.sh --repair --max-iterations 5
```

**Key files**:
- [config.yaml](../jax-agents/config.yaml): LLM config, JAX patterns (use_jit, use_vmap), output settings
- [static_analysis_output/](../jax-agents/static_analysis_output/): JSON files with analyzed module structure (311 translation units)
- [translated_modules/](../jax-agents/translated_modules/): Agent output (legacy location; structured output goes to project src/ and tests/)

### Translation Units Approach

The system breaks large modules into **translation units** (subroutines, functions, types) and translates iteratively:

1. Each unit translated separately with previous context
2. All units assembled into final module
3. Unit-by-unit enables focused debugging and better LLM context

### Structured Output

New translations automatically route to correct directories:

```
src/clm_src_biogeophys/SoilTemperatureMod.py    # Physics code
tests/clm_src_biogeophys/test_SoilTemperatureMod.py  # Tests
docs/translation_notes/SoilTemperatureMod.md     # Translation decisions
```

Source directory mapping based on original Fortran location.

See [jax-agents/README.md](../jax-agents/README.md#translation-approach) for details.

## Testing Conventions

### Running Tests

```bash
# Unit tests
pytest                          # All tests
pytest tests/clm_src_main/      # Specific module
pytest -m "not slow"            # Skip performance tests
pytest --cov=src --cov-report=html  # With coverage

# Integration tests (Fortran-Python comparison)
cd tests/integration
source /path/to/venv/bin/activate
PYTHONPATH=/path/to/clm-ml-jax:/path/to/clm-ml-jax/src python comparison.py
# Generates: fortran_python_comparison.txt, fortran_python_summary.csv

# Integration test suite (pytest-based with parametrized tests)
cd tests/integration
PYTHONPATH=/path/to/clm-ml-jax:/path/to/clm-ml-jax/src pytest test_fortran_python_comparison.py -v
# Specific variable: pytest test_fortran_python_comparison.py -v -k "rnet"
# HTML report: pytest test_fortran_python_comparison.py --html=report.html --self-contained-html
```

**Key markers** ([pytest.ini](../pytest.ini)):
- `@pytest.mark.slow`: Performance/integration tests
- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Multi-module tests

### Test Structure

Tests mirror source structure exactly:

```
src/clm_src_biogeophys/SoilStateType.py
tests/clm_src_biogeophys/test_SoilStateType.py
```

**Shared fixtures** in [tests/conftest.py](../tests/conftest.py):
- `default_test_bounds`: Pre-initialized BoundsType(begc=0, endc=199, ...)
- Global state instances (canopystate_inst, waterstate_inst, etc.)
- JAX 64-bit precision enabled (`jax_enable_x64`)
- `MutableWrapper`: Wraps NamedTuples for legacy driver compatibility (unwrap via `._data` in pure JAX code)

### Writing New Tests

```python
import pytest
import jax.numpy as jnp

@pytest.fixture
def sample_state():
    """Create test state with realistic values"""
    return MyStateType(field1=jnp.array([1.0, 2.0]), ...)

@pytest.mark.parametrize("input,expected", [(1.0, 2.0), (3.0, 4.0)])
def test_function_properties(input, expected):
    """Test with multiple parameter sets"""
    result = my_function(input)
    assert jnp.allclose(result, expected)
```

## Documentation Standards

### Module Docstrings

Always reference original Fortran source:

```python
"""
Module description.

Translated from [OriginalMod.F90](path/to/OriginalMod.F90) (lines 1-330).

Key differences from Fortran:
- DO loops → jax.vmap
- TYPE(derived_type) → NamedTuple
- INTENT(INOUT) → Returns new state

Reference:
    Original Fortran: OriginalMod.F90
"""
```

### Function Docstrings (Google Style)

```python
def compute_soil_properties(bounds: BoundsType, state: SoilStateType) -> SoilStateType:
    """
    Compute soil thermal and hydraulic properties.
    
    Translated from Fortran lines 145-230 in SoilMod.F90.
    
    Args:
        bounds: Domain bounds with column indices (begc:endc)
        state: Current soil state with temperature and moisture
        
    Returns:
        Updated soil state with new thermal conductivity arrays
        
    Note:
        Original Fortran modified state in-place. JAX version returns new state.
        
    Reference:
        SoilMod.F90:145-230
    """
```

## Debugging & Common Issues

### JAX Precision Issues

**Symptom**: Test failures with small numerical differences

**Solution**: Ensure 64-bit precision is enabled:
```python
import jax
jax.config.update("jax_enable_x64", True)  # BEFORE any JAX operations
```

### MutableWrapper Confusion

**Symptom**: `AttributeError: 'MutableWrapper' object has no attribute '_replace'`

**Solution**: In tests, unwrap to get underlying NamedTuple:
```python
# In conftest.py, instances are wrapped
state = clm_instMod.waterstate_inst._data  # Get underlying NamedTuple
```

### Module Initialization Errors

**Symptom**: `nlevgrnd = -1` or uninitialized parameters

**Solution**: Initialize parameters before importing layer values:
```python
from clm_src_main import clm_varpar
clm_varpar.clm_varpar_init()  # Initialize FIRST
from clm_src_main.clm_varpar import nlevgrnd  # Import AFTER
```

### Integration Test Discrepancies

**Symptom**: Large errors in integration tests (e.g., shflx, lhflx)

**Likely causes**:
1. Multilayer canopy wrapper using simplified physics (see Known Limitations)
2. Boundary layer conductance differences
3. Radiation/turbulence module not fully connected

**Check**: [tests/integration/fortran_python_comparison.txt](../tests/integration/fortran_python_comparison.txt) for detailed metrics

### Physics Validation Strategy

**When adding new physics modules**:
1. Start with unit tests using synthetic data
2. Run integration tests to compare with Fortran outputs
3. Check correlation first (indicates pattern match), then absolute errors
4. Variables with high correlation (>0.9) but high relative errors often indicate scaling or unit conversion issues
5. Variables with low correlation (<0.5) indicate fundamental logic differences

**Common error patterns from integration testing**:
- **100% error with zero output**: Missing initialization or function not called (e.g., sunlit/shaded fluxes)
- **High correlation + high error**: Unit conversion or scaling factor missing (e.g., lhsoi: 0.74 corr, 162000% error)
- **Low correlation + high error**: Logic error or missing coupling (e.g., shflx: -0.18 corr, 4428% error)
- **Small error + high correlation**: Good match, may have minor numerical differences (e.g., rnet: 0.999 corr, 27.6% error)

## Development Workflow

### Adding a New Translated Module

1. **Translate**: `./run_translation_workflow.sh --translate --modules "NewMod"`
2. **Check output**: Verify files in `src/appropriate_subdir/NewMod.py` and `tests/appropriate_subdir/test_NewMod.py`
3. **Run tests**: `pytest tests/appropriate_subdir/test_NewMod.py`
4. **Repair if needed**: `./run_translation_workflow.sh --repair --modules "NewMod"`

### Python Environment Setup

**Required virtual environment with specific packages**:

```bash
# Create and activate virtual environment
python3 -m venv /path/to/venv-arm64
source /path/to/venv-arm64/bin/activate

# Install core dependencies
pip install jax jaxlib numpy pytest netCDF4 xarray

# Install jax-agents tools
cd jax-agents && pip install -e .

# Verify setup
python -c "import jax; import jax_agents; print('Setup OK')"
```

### CLUBB Integration (Advanced)

```bash
# Update CLUBB submodule
git submodule update --init clubb_ML

# Compile original Fortran (for reference/validation)
cd clubb_ML && ./compile/compile.bash
```

**Translation process**:
1. Static analysis already done → `CLUBB_static_analysis/`
2. Run translation: `./run_translation_workflow.sh --translate --modules "module_name"`
3. Output appears in `src/` and `tests/` directories
4. CLUBB modules use similar patterns to CLM (NamedTuples, vmap, etc.)

### Running Integration Tests

Integration tests compare Python JAX implementation against original Fortran:

```bash
cd tests/integration
source /path/to/venv/bin/activate  # Use venv with JAX, netCDF4, xarray

# Standalone comparison script (generates detailed reports)
PYTHONPATH=/path/to/clm-ml-jax:/path/to/clm-ml-jax/src python comparison.py
# Generates: fortran_python_comparison.txt, fortran_python_summary.csv

# Pytest test suite (automated validation)
PYTHONPATH=/path/to/clm-ml-jax:/path/to/clm-ml-jax/src pytest test_fortran_python_comparison.py -v
# Run specific categories: -k "flux" / -k "auxiliary" / -k "sunshade"
# Generate HTML report: --html=comparison_report.html --self-contained-html

# Integration test tolerance configuration in test_fortran_python_comparison.py
# Customize per-variable tolerances in TOLERANCES dict:
# TOLERANCES = {
#     'rnet': {'max_rel_error': 0.10, 'max_abs_error': 50.0, 'min_correlation': 0.95},
#     'gpp': {'max_rel_error': 0.50, 'max_abs_error': 15.0, 'min_correlation': 0.85},
# }
```

**Key metrics** (from [fortran_python_summary.csv](../tests/integration/fortran_python_summary.csv)):
- Net radiation (rnet): Mean rel error 27.6%, correlation 0.999 (good)
- Sensible heat flux (shflx): Mean rel error 4428%, correlation -0.18 (poor - canopy issues)
- Latent heat flux (lhflx): Mean rel error 34,747%, correlation 0.926 (poor magnitude, good pattern)
- GPP: Mean rel error 29.4%, correlation 0.920 (acceptable for simplified physics)
- Soil heat flux (gsoi): Mean rel error 1120%, correlation 0.853 (poor - missing surface coupling)
- ustar: Mean rel error 95.5%, correlation 0.778 (reasonable)

**Variables with zero output** (100% error in fsun category):
- All sunlit/shaded variables: `lh_sun`, `lh_shade`, `sh_sun`, `sh_shade`, `gpp_sun`, `gpp_shade`
- Reason: MLCanopyFluxes_wrapper only computes aggregated temperatures, not per-layer fluxes yet

See "Known Limitations" section for details on multilayer canopy simplifications.

## Project-Specific Commands

```bash
# Setup
pip install -e jax-agents/          # Install agent tools
pytest --version                     # Verify test environment

# Translation
cd jax-agents && ./run_translation_workflow.sh --all --auto-repair

# Testing
pytest -v                            # Verbose test output
pytest --lf                          # Re-run last failures
pytest --tb=short                    # Short traceback format

# Integration testing
cd tests/integration
PYTHONPATH=/path/to/clm-ml-jax:/path/to/clm-ml-jax/src python comparison.py

# Code quality
black src/ tests/                    # Format code (100 char line length)
ruff check src/                      # Lint code

# Git submodules
git submodule update --init clubb_ML  # Initialize CLUBB submodule
```

## Key Files Quick Reference

- [run_translation_workflow.sh](../jax-agents/run_translation_workflow.sh) - Main translation interface
- [config.yaml](../jax-agents/config.yaml) - LLM and translation settings
- [conftest.py](../tests/conftest.py) - Global test fixtures and state initialization
- [clm_varpar.py](../src/clm_src_main/clm_varpar.py) - Layer parameters (must init first!)
- [clm_instMod.py](../src/clm_src_main/clm_instMod.py) - Global state singleton registry
- [FrictionVelocityMod.py](../src/clm_src_biogeophys/FrictionVelocityMod.py) - Reference pattern for immutable state
- [pytest.ini](../pytest.ini) - Test configuration and markers

## Cost Management

Translation workflow uses Claude API with token tracking:
- **Per module cost**: $0.10-$3.00 depending on complexity
- **Track costs**: Check `logs/*.log` for token usage
- **Limit exposure**: Set `max_cost_per_module` in [config.yaml](../jax-agents/config.yaml)

## Known Limitations

### Multilayer Canopy Integration

**Status**: Critical physics modules connected (3/9), remaining modules as TODO

**Current state**: 
1. ✅ Sub-timestep integration loop structure (Fortran lines ~427-500)
2. ✅ Flux accumulator arrays initialized and updated
3. ✅ SubTimeStepFluxIntegration called correctly
4. ✅ **LeafBoundaryLayer connected** - Computes gbh, gbv, gbc conductances from wind/temperature
5. ✅ **LeafPhotosynthesis connected** - Computes agross, anet, gs from light/CO2/temperature
6. ✅ **LeafFluxes connected** - Solves leaf energy balance for tleaf and heat fluxes (shleaf, lhleaf, evleaf, trleaf, stleaf)
7. ⚠️ Physics module calls still as TODO:
   - CanopyInterception (LOW priority)
   - LongwaveRadiation (MEDIUM priority)
   - LeafHeatCapacity (MEDIUM priority)
   - CanopyTurbulence (HIGH priority - turbulent fluxes)
   - LeafWaterPotential (MEDIUM priority)
   - CanopyEvaporation (HIGH priority - soil fluxes)
8. ✅ Time averaging after loop (divide by num_sub_steps)
9. ⚠️ Post-loop diagnostics marked as TODO

**Major Progress**: The three critical physics modules forming the core energy balance are now connected:
- LeafBoundaryLayer → computes conductances
- LeafPhotosynthesis → uses conductances to compute stomatal conductance
- LeafFluxes → uses all conductances to solve energy balance and compute leaf temperatures

This means leaf temperatures are now computed correctly (not zero), and sun/shade heat fluxes (shleaf, lhleaf) should now have realistic values instead of zeros.

**Expected Impact on Integration Tests**:
- sun/shade flux variables (lh_sun, lh_shade, sh_sun, sh_shade, gpp_sun, gpp_shade): Should improve from 100% error to <50%
- Overall heat fluxes (shflx, lhflx): Should improve significantly from current 4400%+ errors
- GPP: Should improve from current errors to more realistic values

**Next steps to complete**:
1. **HIGH PRIORITY**: Connect CanopyTurbulence module for turbulent transfer (shair, etair, ustar)
2. **HIGH PRIORITY**: Connect CanopyEvaporation module for soil fluxes (gsoi, shsoi, lhsoi)
3. **MEDIUM PRIORITY**: Connect LongwaveRadiation module for lwleaf, lwsoi, lwup
4. Implement full CanopyFluxesDiagnostics (688 lines from Fortran)
5. Add energy/water balance checks
6. Run integration tests to validate improvements

**Comparison with Fortran**: 
- Fortran MLCanopyFluxesMod.F90 lines 400-500: Full physics workflow with sub-stepping
- Python MLCanopyFluxesMod.py ml_canopy_fluxes: **Core energy balance physics now connected** (3/9 modules)
- Python conftest.py MLCanopyFluxes_wrapper: Fallback for when PHYSICS_MODULES_AVAILABLE=False
- Individual physics modules are fully translated and core ones are now integrated
- **Status**: Core physics operational (energy balance solved), auxiliary modules pending

**Translation status**:
- ✅ MLLeafFluxesMod.py - Leaf energy balance solver (complete and **connected**)
- ✅ MLLeafPhotosynthesisMod.py - C3/C4 photosynthesis (complete and **connected**)
- ✅ MLLeafBoundaryLayerMod.py - Aerodynamic conductance (complete and **connected**)
- ✅ MLCanopyTurbulenceMod.py - Within-canopy transport (complete, **not yet connected**)
- 🔄 MLCanopyFluxesMod.py ml_canopy_fluxes - Main driver with core physics connected, auxiliary modules pending


## Further Reading

- [README.md](../README.md): Full project overview and installation
- [jax-agents/GETTING_STARTED.md](../jax-agents/GETTING_STARTED.md): 5-minute agent quickstart
- [jax-agents/WORKFLOW_SCRIPT_GUIDE.md](../jax-agents/WORKFLOW_SCRIPT_GUIDE.md): Detailed workflow options
- [tests/README.md](../tests/README.md): Testing guide and structure
- [docs/code_review_january_2026.md](../docs/code_review_january_2026.md): Code quality review
