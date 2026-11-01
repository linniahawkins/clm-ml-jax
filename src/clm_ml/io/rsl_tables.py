"""The roughness sublayer psihat lookup tables (``clm_ml.processes.turbulence.RSLTables``).

The tables ship with the package (``clm_ml/data/psihat.nc``, from CLM-ml v2's
``rsl_lookup_tables/psihat.nc``): variables ``zdtgridM``/``zdtgridH`` (nZ),
``dtLgridM``/``dtLgridH`` (nL) and ``psigridM``/``psigridH`` (nZ, nL) for momentum and
scalars.

Fortran reference: MLCanopyTurbulenceMod.F90::LookupPsihatINI.
"""

from __future__ import annotations

import functools
from importlib import resources
from pathlib import Path

import numpy as np

from clm_ml.processes.turbulence import PsihatTable, RSLTables


def rsl_table_file() -> Path:
    """The package's psihat lookup table file."""
    return Path(str(resources.files("clm_ml") / "data" / "psihat.nc"))


def read_rsl_tables(path=None) -> RSLTables:
    """The psihat tables in ``path`` (default: :func:`rsl_table_file`) as float64 numpy
    arrays. Fortran checks fixed sizes (nZ = 276, nL = 41); here each table's grids must
    match its psihat array."""
    import netCDF4

    path = rsl_table_file() if path is None else Path(path)
    tables = {}
    with netCDF4.Dataset(path, "r") as ds:
        for kind in ("M", "H"):

            def read(name, kind=kind):
                if name + kind not in ds.variables:
                    raise ValueError(f"{path}: variable {name + kind} missing")
                return np.asarray(ds.variables[name + kind][:], dtype=np.float64)

            table = PsihatTable(zdt=read("zdtgrid"), dtl=read("dtLgrid"), psi=read("psigrid"))
            if table.psi.shape != (table.zdt.size, table.dtl.size):
                raise ValueError(
                    f"{path}: psigrid{kind} has shape {table.psi.shape}, the grids "
                    f"({table.zdt.size}, {table.dtl.size})"
                )
            tables[kind] = table
    return RSLTables(momentum=tables["M"], scalar=tables["H"])


@functools.lru_cache(maxsize=None)
def default_rsl_tables() -> RSLTables:
    """The package's tables, read once per process (immutable)."""
    return read_rsl_tables()
