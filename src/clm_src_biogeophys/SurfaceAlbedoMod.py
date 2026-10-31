"""
JAX translation of SurfaceAlbedoMod Fortran module.

Calculates ground surface (soil) albedo for the CLM land surface model.
Provides time-constant initialization of soil color lookup tables (the per-step soil
albedo is ``clm_ml.physics.soil.soil_albedo``).

Original Fortran module: SurfaceAlbedoMod
Fortran lines 1-130
"""

import jax.numpy as jnp
from jax import Array

from clm_src_biogeophys.SurfaceAlbedoType import surfalb_type  # noqa: F401
from clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_varpar import inir, ivis, numrad  # noqa: F401
from clm_src_main.decompMod import bounds_type  # noqa: F401
from offline_driver import TowerDataMod  # tower table, read when called (not copied at import)

# ---------------------------------------------------------------------------
# Module-level private data (Fortran lines 22-25)
# Wet and dry soil albedo lookup tables, initialised by
# SurfaceAlbedoInitTimeConst and consumed by SoilAlbedo.
# ---------------------------------------------------------------------------

# albsat(mxsoil_color, numrad): Wet soil albedo by color class and waveband
# albdry(mxsoil_color, numrad): Dry soil albedo by color class and waveband
# isoicol(begc:endc):           Column soil color class
albsat: Array = jnp.empty((0, 0), dtype=jnp.float64)
albdry: Array = jnp.empty((0, 0), dtype=jnp.float64)
isoicol: Array = jnp.empty((0,), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Public: time-constant initialization
# ---------------------------------------------------------------------------


def SurfaceAlbedoInitTimeConst(bounds: bounds_type) -> None:
    """
    Initialize module time-constant variables for soil albedo.

    Populates the module-level ``albsat``, ``albdry``, and ``isoicol``
    arrays. Only ``mxsoil_color == 20`` (the CLM5 default) and
    ``mxsoil_color == 8`` (legacy) are supported; any other value
    triggers an abort.

    Mirrors Fortran subroutine ``SurfaceAlbedoInitTimeConst``
    (lines 35-90).

    Args:
        bounds: CLM column decomposition bounds, supplying ``begc``
            and ``endc``.
    """
    global albsat, albdry, isoicol

    begc = bounds.begc
    endc = bounds.endc

    # Assign soil color class for every column — Fortran lines 45-48
    n_col = endc - begc + 1
    soil_color = int(TowerDataMod.tower_isoicol[TowerDataMod.tower_num])
    isoicol = jnp.full((n_col,), soil_color, dtype=jnp.int32)

    # Saturated and dry soil albedo tables — Fortran lines 51-88
    mxsoil_color = 20

    if mxsoil_color == 8:
        # 8-class tables — Fortran lines 56-60
        # Size numrad+1: slot 0 unused; ivis=1 and inir=2 are valid 1-based indices
        _albsat = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)
        _albdry = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)

        _albsat = _albsat.at[:, ivis].set(
            jnp.array([0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05], dtype=jnp.float64)
        )
        _albsat = _albsat.at[:, inir].set(
            jnp.array([0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10], dtype=jnp.float64)
        )
        _albdry = _albdry.at[:, ivis].set(
            jnp.array([0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10], dtype=jnp.float64)
        )
        _albdry = _albdry.at[:, inir].set(
            jnp.array([0.48, 0.44, 0.40, 0.36, 0.32, 0.28, 0.24, 0.20], dtype=jnp.float64)
        )

    elif mxsoil_color == 20:
        # 20-class tables (CLM5 default) — Fortran lines 61-77
        # Size numrad+1: slot 0 unused; ivis=1 and inir=2 are valid 1-based indices
        _albsat = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)
        _albdry = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)

        _albsat = _albsat.at[:, ivis].set(
            jnp.array(
                [
                    0.25,
                    0.23,
                    0.21,
                    0.20,
                    0.19,
                    0.18,
                    0.17,
                    0.16,
                    0.15,
                    0.14,
                    0.13,
                    0.12,
                    0.11,
                    0.10,
                    0.09,
                    0.08,
                    0.07,
                    0.06,
                    0.05,
                    0.04,
                ],
                dtype=jnp.float64,
            )
        )
        _albsat = _albsat.at[:, inir].set(
            jnp.array(
                [
                    0.50,
                    0.46,
                    0.42,
                    0.40,
                    0.38,
                    0.36,
                    0.34,
                    0.32,
                    0.30,
                    0.28,
                    0.26,
                    0.24,
                    0.22,
                    0.20,
                    0.18,
                    0.16,
                    0.14,
                    0.12,
                    0.10,
                    0.08,
                ],
                dtype=jnp.float64,
            )
        )
        _albdry = _albdry.at[:, ivis].set(
            jnp.array(
                [
                    0.36,
                    0.34,
                    0.32,
                    0.31,
                    0.30,
                    0.29,
                    0.28,
                    0.27,
                    0.26,
                    0.25,
                    0.24,
                    0.23,
                    0.22,
                    0.20,
                    0.18,
                    0.16,
                    0.14,
                    0.12,
                    0.10,
                    0.08,
                ],
                dtype=jnp.float64,
            )
        )
        _albdry = _albdry.at[:, inir].set(
            jnp.array(
                [
                    0.61,
                    0.57,
                    0.53,
                    0.51,
                    0.49,
                    0.48,
                    0.45,
                    0.43,
                    0.41,
                    0.39,
                    0.37,
                    0.35,
                    0.33,
                    0.31,
                    0.29,
                    0.27,
                    0.25,
                    0.23,
                    0.21,
                    0.16,
                ],
                dtype=jnp.float64,
            )
        )

    else:
        endrun(msg=" ERROR: SurfaceAlbedoInitTimeConst: maximum color class is not supported")
        # Unreachable; satisfies static type checkers
        _albsat = jnp.empty((0, 0), dtype=jnp.float64)
        _albdry = jnp.empty((0, 0), dtype=jnp.float64)

    albsat = _albsat
    albdry = _albdry
