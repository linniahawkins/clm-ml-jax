"""Saturation vapor pressure and latent heat of vaporization.

Elementwise functions of temperature (any shape). Polynomial fits of Flatau et al. (1992)
J. Appl. Meteor. 31, 1507-1513, for water (0 to 100 C) and ice (-75 to 0 C); the molar
latent heat of CLM5 (vaporization above freezing, sublimation at or below).

Fortran reference: CLM-ml v2 ``MLWaterVaporMod.F90``.
"""

from __future__ import annotations

import jax.numpy as jnp

from clm_ml.processes.constants import hsub, hvap, mmh2o, tfrz

# Water (0 to 100 C): es and d(es)/dT, mb
_A = (6.11213476, 0.444007856, 0.143064234e-1, 0.264461437e-3, 0.305903558e-5,
      0.196237241e-7, 0.892344772e-10, -0.373208410e-12, 0.209339997e-15)  # fmt: skip
_B = (0.444017302, 0.286064092e-1, 0.794683137e-3, 0.121211669e-4, 0.103354611e-6,
      0.404125005e-9, -0.788037859e-12, -0.114596802e-13, 0.381294516e-16)  # fmt: skip
# Ice (-75 to 0 C)
_C = (6.11123516, 0.503109514, 0.188369801e-1, 0.420547422e-3, 0.614396778e-5,
      0.602780717e-7, 0.387940929e-9, 0.149436277e-11, 0.262655803e-14)  # fmt: skip
_D = (0.503277922, 0.377289173e-1, 0.126801703e-2, 0.249468427e-4, 0.313703411e-6,
      0.257180651e-8, 0.133268878e-10, 0.394116744e-13, 0.498070196e-16)  # fmt: skip


def _horner(c, tc):
    """``c[0] + tc (c[1] + tc (c[2] + ... + tc c[8]))`` (the Fortran nesting)."""
    r = c[8]
    for k in range(7, -1, -1):
        r = c[k] + tc * r
    return r


def sat_vap(t):
    """Saturation vapor pressure ``es`` (Pa) and its temperature derivative ``desdt`` (Pa/K)
    at temperature ``t`` (K): over water for ``t >= tfrz``, over ice below; the temperature
    is clamped to -75..100 C.

    Fortran reference: MLWaterVaporMod.F90::SatVap.
    """
    tc = jnp.clip(jnp.asarray(t) - tfrz, -75.0, 100.0)
    above_zero = tc >= 0.0
    es = jnp.where(above_zero, _horner(_A, tc), _horner(_C, tc)) * 100.0  # mb -> Pa
    desdt = jnp.where(above_zero, _horner(_B, tc), _horner(_D, tc)) * 100.0
    return es, desdt


def lat_vap(t):
    """Molar latent heat (J/mol) at temperature ``t`` (K): of vaporization above ``tfrz``,
    of sublimation at or below.

    Fortran reference: MLWaterVaporMod.F90::LatVap.
    """
    return jnp.where(jnp.asarray(t) > tfrz, hvap, hsub) * mmh2o
