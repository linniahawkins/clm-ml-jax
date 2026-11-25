"""Soil water: the Clapp and Hornberger (1978) retention curve of CLM's soil water movement.

The functions work on scalars or on arrays of soil levels (elementwise). The relative
saturation ``s`` is the caller's, already limited to CLM's bounds [0.01, 1]: with ``s``
bounded away from 0 the powers and the divisions by ``s`` are finite, and so are their
gradients, including the gradient with respect to the exponent ``bsw``.

Not in CLM-ml v2: these are translated from CLM itself (CTSM ``ctsm5.4.004``).

Fortran reference: CTSM ``SoilWaterRetentionCurveClappHornberg1978Mod.F90``.
"""

from __future__ import annotations

from jax import Array


def soil_hk(s: Array, imped: Array, hksat: Array, bsw: Array) -> tuple[Array, Array]:
    """Hydraulic conductivity (mm s⁻¹) and its derivative with respect to the relative
    saturation (mm s⁻¹).

    ``s``: relative saturation, in [0.01, 1]; ``imped``: ice impedance factor (1 without ice);
    ``hksat``: conductivity at saturation (mm s⁻¹); ``bsw``: Clapp-Hornberger *b*.

    The derivative is CLM's hand-coded one, which the Richards solver uses to build its
    matrix; it is part of the forward calculation.

    Fortran reference: SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_hk.
    """
    hk = imped * hksat * s ** (2.0 * bsw + 3.0)
    dhkds = (2.0 * bsw + 3.0) * hk / s
    return hk, dhkds


def soil_suction(s: Array, sucsat: Array, bsw: Array) -> tuple[Array, Array]:
    """Soil matric potential (mm, negative) and its derivative with respect to the relative
    saturation (mm).

    ``s``: relative saturation, in [0.01, 1]; ``sucsat``: suction at saturation (mm,
    positive); ``bsw``: Clapp-Hornberger *b*.

    Fortran reference: SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_suction.
    """
    smp = -sucsat * s ** (-bsw)
    dsmpds = -bsw * smp / s
    return smp, dsmpds
