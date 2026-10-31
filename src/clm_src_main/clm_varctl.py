"""
JAX translation of clm_varctl Fortran module.

Run control variables for the CLM land surface model.
Contains the log file unit number.

Original Fortran module: clm_varctl
Fortran lines 1-17
"""

# "stdout" log file unit number — Fortran line 13
# Used throughout the model as the target for diagnostic output.
iulog: int = 6

# RSL psihat look-up table file (Fortran rslfile): the package's clm_ml/data/psihat.nc,
# clm_ml.io.rsl_tables.rsl_table_file() (Phase 7 decision C15).
