"""Leaf boundary layer conductance for heat, water vapor and CO2, from the Nusselt number of
laminar or turbulent forced convection and free convection.

One patch. Layers are 0-based slots on axis 0; leaves (0 sunlit, 1 shaded) on axis 1 of the
leaf temperature and of the results. Layers without plant area get zero conductance.

Fortran reference: CLM-ml v2 ``MLLeafBoundaryLayerMod.F90``. Bonan (2019) Climate Change and
Terrestrial Ecosystem Modeling, chapter 10.

Reads and writes are the arguments and results; ``Reads``/``Writes`` in the docstrings name
the legacy container fields they come from and go to (through the adapter in
``clm_ml.legacy.leaf_boundary_layer`` until the containers leave the step, Phase 7.2
process 10).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from clm_ml.physics.constants import dc0, dh0, dv0, gb_factor, gbh_min, grav, tfrz, visc0

#: Flow regimes by name (model.yaml ``physics.leaf_boundary_layer``; Fortran ``gb_type``
#: 1, 2, 3): laminar forced convection; the larger of laminar and turbulent forced
#: convection; that plus free convection.
REGIMES = ("laminar", "turbulent", "free_convection")


class BoundaryLayerConductance(NamedTuple):
    gbh: Array  # (layer, leaf) heat, mol/m2 leaf/s
    gbv: Array  # (layer, leaf) water vapor, mol H2O/m2 leaf/s
    gbc: Array  # (layer, leaf) CO2, mol CO2/m2 leaf/s


def leaf_boundary_layer(
    dpai: Array,
    wind: Array,
    tair: Array,
    tleaf: Array,
    pref,
    tref,
    rhomol,
    leaf_dimension,
    regime: str = "free_convection",
) -> BoundaryLayerConductance:
    """Leaf boundary layer conductances (layer, leaf) of each layer's sunlit and shaded leaves.

    The molecular diffusivities and viscosity are corrected to the reference temperature and
    pressure (``fac = 101325 / pref * (tref / tfrz)^1.81``). From the Reynolds number
    ``Re = wind dleaf / visc``, the Prandtl number ``Pr = visc / dh`` and the Grashof number
    ``Gr = grav dleaf^3 max(tleaf - tair, 0) / (tair visc^2)``, the Nusselt numbers of laminar
    (``1.5 * 0.66 Pr^0.33 Re^0.5``) and turbulent (``1.5 * 0.036 Pr^0.33 Re^0.8``) forced
    convection and of free convection (``0.54 Pr^0.25 Gr^0.25``) give
    ``gbh = dh Nu / dleaf * rhomol`` (forced convection at least ``gbh_min``), and
    ``gbv = gbh (dv/dh)^n``, ``gbc = gbh (dc/dh)^n`` (``n`` = 0.67 forced, 0.75 free).
    ``regime`` (one of :data:`REGIMES`) chooses how they combine. Inputs per layer:
    ``dpai`` (m2/m2), ``wind`` (m/s), ``tair`` (K); ``tleaf`` (layer, leaf), K.

    Fortran reference: MLLeafBoundaryLayerMod.F90::LeafBoundaryLayer (called once per leaf).
    Reads: ``dpai_profile``, ``wind_profile``, ``tair_profile``, ``tleaf_leaf``,
    ``pref_forcing``, ``tref_forcing``, ``rhomol_forcing``, the PFT's ``dleaf``.
    Writes: ``gbh_leaf``, ``gbv_leaf``, ``gbc_leaf``.
    """
    if regime not in REGIMES:
        raise ValueError(f"leaf_boundary_layer: regime {regime!r} not one of {REGIMES}")
    # Diffusivity correction for temperature and pressure
    fac = 101325.0 / pref * (tref / tfrz) ** 1.81
    visc = visc0 * fac
    dh = dh0 * fac
    dv = dv0 * fac
    dc = dc0 * fac
    dl = leaf_dimension
    dv_dh = dv / dh
    dc_dh = dc / dh

    wind = wind[:, None]
    tair = tair[:, None]
    re = wind * dl / visc
    pr = visc / dh
    # Free convection only when the leaf is warmer than the air
    gr = grav * dl**3 * jnp.maximum(tleaf - tair, 0.0) / (tair * visc * visc)

    # Bases of fractional powers kept above zero: the gradient of x^n at 0 is infinite, and
    # an infinite gradient times the zero mask below gives NaN
    _eps_pow = jnp.asarray(1.0e-30)
    re_safe = jnp.maximum(re, _eps_pow)
    gr_safe = jnp.maximum(gr, _eps_pow)

    # Laminar forced convection
    nu_lam = gb_factor * 0.66 * pr**0.33 * re_safe**0.5
    gbh_lam = jnp.maximum((dh * nu_lam / dl) * rhomol, gbh_min)
    gbv_lam = gbh_lam * dv_dh**0.67
    gbc_lam = gbh_lam * dc_dh**0.67

    # Turbulent forced convection
    nu_turb = gb_factor * 0.036 * pr**0.33 * re_safe**0.8
    gbh_turb = jnp.maximum((dh * nu_turb / dl) * rhomol, gbh_min)
    gbv_turb = gbh_turb * dv_dh**0.67
    gbc_turb = gbh_turb * dc_dh**0.67

    # Free convection
    nu_free = 0.54 * pr**0.25 * gr_safe**0.25
    gbh_free = (dh * nu_free / dl) * rhomol
    gbv_free = gbh_free * dv_dh**0.75
    gbc_free = gbh_free * dc_dh**0.75

    if regime == "laminar":
        gbh, gbv, gbc = gbh_lam, gbv_lam, gbc_lam
    elif regime == "turbulent":
        gbh = jnp.maximum(gbh_lam, gbh_turb)
        gbv = jnp.maximum(gbv_lam, gbv_turb)
        gbc = jnp.maximum(gbc_lam, gbc_turb)
    else:
        gbh = jnp.maximum(gbh_lam, gbh_turb) + gbh_free
        gbv = jnp.maximum(gbv_lam, gbv_turb) + gbv_free
        gbc = jnp.maximum(gbc_lam, gbc_turb) + gbc_free
    # One value per leaf, as Fortran (called once per leaf): without free convection the
    # conductances do not depend on the leaf, so they are the same for sunlit and shaded
    gbh, gbv, gbc = (jnp.broadcast_to(g, jnp.shape(tleaf)) for g in (gbh, gbv, gbc))

    has_pai = (dpai > 0.0)[:, None]
    return BoundaryLayerConductance(
        gbh=jnp.where(has_pai, gbh, 0.0),
        gbv=jnp.where(has_pai, gbv, 0.0),
        gbc=jnp.where(has_pai, gbc, 0.0),
    )
