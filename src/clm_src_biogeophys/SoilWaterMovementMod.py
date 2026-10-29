"""
JAX translation of SoilWaterMovementMod Fortran module.

Couples soil and root water interactions. Computes liquid volumetric
water content and derives hydraulic conductivity and matric potential
for each soil layer using the Clapp-Hornberger (1978) relationships.

Original Fortran module: SoilWaterMovementMod
Fortran lines 1-160
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from clm_src_biogeophys.SoilStateType import soilstate_type  # noqa: F401
from clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from clm_src_main.clm_varcon import denh2o  # noqa: F401
from clm_src_main.clm_varpar import nlevsoi  # noqa: F401
from clm_src_main.decompMod import bounds_type  # noqa: F401
from clm_src_main.SiteType import site_or_modules

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def SoilWater(
    bounds: bounds_type,
    num_hydrologyc: int,
    filter_hydrologyc: np.ndarray,
    soilstate_inst: soilstate_type,
    waterstatebulk_inst: waterstatebulk_type,
    site=None,
) -> soilstate_type:
    """
    Couple soil and root water interactions.

    Mirrors Fortran subroutine ``SoilWater`` (lines 26-50).
    Delegates entirely to ``soilwater_moisture_form``.

    Args:
        bounds: CLM column decomposition bounds.
        num_hydrologyc: Number of columns in the CLM hydrology filter.
        filter_hydrologyc: 1-D array of column indices in the hydrology
            filter. Fortran: ``integer, intent(in) :: filter_hydrologyc(:)``.
        soilstate_inst: Soil state container; hydraulic properties are
            written and returned in a new instance.
        waterstatebulk_inst: Bulk water state container (read-only).
        site: Per-site data (``SiteType.site_type``: ``dz``, ``nbedrock``); default: the
            subgrid modules' values.

    Returns:
        Updated :class:`soilstate_type` with hydraulic conductivity and
        matric potential populated.
    """
    return soilwater_moisture_form(
        bounds,
        num_hydrologyc,
        filter_hydrologyc,
        soilstate_inst,
        waterstatebulk_inst,
        site=site,
    )


# ---------------------------------------------------------------------------
# Private: moisture-form soil hydrology
# ---------------------------------------------------------------------------


def soilwater_moisture_form(
    bounds: bounds_type,
    num_hydrologyc: int,
    filter_hydrologyc: np.ndarray,
    soilstate_inst: soilstate_type,
    waterstatebulk_inst: waterstatebulk_type,
    site=None,
) -> soilstate_type:
    """
    Compute liquid volumetric water content and call hydraulic property
    solver for each column in the hydrology filter.

    Mirrors Fortran subroutine ``soilwater_moisture_form`` (lines 52-105).

    For each active column the number of active layers is set to
    ``nbedrock(c)``. Liquid volumetric water content is computed as:

    .. code-block:: none

        vwc_liq(c,j) = max(h2osoi_liq(c,j), 1e-6) / (dz(c,j) * denh2o)

    Hydraulic conductivity and matric potential are then derived by
    :func:`compute_hydraulic_properties`.

    Args:
        bounds: CLM column decomposition bounds.
        num_hydrologyc: Number of columns in the CLM hydrology filter.
        filter_hydrologyc: 1-D array of column indices in the hydrology
            filter.
        soilstate_inst: Soil state container; updated fields are returned
            in a new instance.
        waterstatebulk_inst: Bulk water state container (read-only).

    Returns:
        Updated :class:`soilstate_type`.
    """
    # Unpack inputs (Fortran associate block, lines 83-87)
    site = site_or_modules(site)
    nbedrock = site.nbedrock_col  # Depth to bedrock index (data: a soil-layer mask)
    dz = site.dz_col  # Soil layer thickness (m)
    h2osoi_liq = waterstatebulk_inst.h2osoi_liq_col  # Soil layer liquid water (kg H2O/m2)

    n_col = bounds.endc - bounds.begc + 1

    # Local work arrays — Fortran lines 73-75
    # Shape (n_col, nlevsoi); indexed relative to begc
    vwc_liq = jnp.zeros((n_col, nlevsoi), dtype=jnp.float64)

    for fc in range(num_hydrologyc):  # Fortran: do fc = 1, num_hydrologyc
        c = int(np.asarray(filter_hydrologyc)[fc])  # structure: static
        ci = c - bounds.begc  # 0-based offset into local arrays

        # Number of active layers — Fortran line 92: nlayers = nbedrock(c). The bedrock
        # depth is data, so all nlevsoi layers are computed and those below bedrock keep
        # their values (Fortran leaves them untouched).
        active = jnp.arange(1, nlevsoi + 1) <= nbedrock[c]

        # Liquid volumetric water content — Fortran lines 95-97
        for j in range(1, nlevsoi + 1):  # Fortran: do j = 1, nlayers
            vwc_liq = vwc_liq.at[ci, j - 1].set(
                jnp.where(
                    active[j - 1],
                    jnp.maximum(h2osoi_liq[ci, j - 1], 1.0e-6) / (dz[c, j] * denh2o),
                    vwc_liq[ci, j - 1],
                )
            )

        # Hydraulic conductivity and matric potential — Fortran lines 100-102
        soilstate_inst = compute_hydraulic_properties(
            c,
            nlevsoi,
            soilstate_inst,
            vwc_liq[ci, :],
            active=active,
        )

    return soilstate_inst


# ---------------------------------------------------------------------------
# Private: hydraulic conductivity and matric potential
# ---------------------------------------------------------------------------


def compute_hydraulic_properties(
    c: int,
    nlayers: int,
    soilstate_inst: soilstate_type,
    vwc_liq: Array,
    active: Array | None = None,
) -> soilstate_type:
    """
    Compute hydraulic conductivity and matric potential for each soil
    layer using the Clapp-Hornberger (1978) relationships.

    Mirrors Fortran subroutine ``compute_hydraulic_properties``
    (lines 107-155).

    The Clapp-Hornberger (1978) Water Resources Research 14:601-604
    equations are applied per layer:

    .. code-block:: none

        s   = min(max(vwc_liq(j) / watsat(c,j), 0.01), 1.0)
        hk  = hksat(c,j) * s ** (2*bsw(c,j) + 3)
        smp = max(-sucsat(c,j) * s ** (-bsw(c,j)), -1e8)

    Args:
        c: Column index for the CLM g/l/c/p hierarchy.
        nlayers: Number of active soil layers (``nbedrock(c)``).
        soilstate_inst: Soil state container; ``hk_l_col`` and
            ``smp_l_col`` are updated and returned in a new instance.
        vwc_liq: Liquid volumetric water content per layer (m3/m3),
            shape ``(nlayers,)``.
            Fortran: ``real(r8), intent(in) :: vwc_liq(1:nlayers)``.
        active: Optional mask, shape ``(nlayers,)``: layers where it is False keep their
            ``hk_l``/``smp_l`` values (soil layers below bedrock, when the bedrock depth is
            data). Their inputs are replaced by the first layer's, so they stay finite.

    Returns:
        Updated :class:`soilstate_type` with ``hk_l_col`` and
        ``smp_l_col`` populated for column ``c``.
    """
    # Unpack soil properties (Fortran associate block, lines 130-139)
    watsat = soilstate_inst.watsat_col  # Porosity
    hksat = soilstate_inst.hksat_col  # Saturated hydraulic conductivity (mm H2O/s)
    sucsat = soilstate_inst.sucsat_col  # Saturated suction (mm)
    bsw = soilstate_inst.bsw_col  # Clapp-Hornberger b parameter

    hk_l = soilstate_inst.hk_l_col  # Hydraulic conductivity output (mm H2O/s)
    smp_l = soilstate_inst.smp_l_col  # Matric potential output (mm)

    for j in range(1, nlayers + 1):  # Fortran: do j = 1, nlayers

        on = True if active is None else active[j - 1]
        # Inputs of inactive layers: the first layer's (finite values, finite gradients)
        vwc_j = jnp.where(on, vwc_liq[j - 1], vwc_liq[0])
        watsat_j = jnp.where(on, watsat[c, j], watsat[c, 1])
        hksat_j = jnp.where(on, hksat[c, j], hksat[c, 1])
        sucsat_j = jnp.where(on, sucsat[c, j], sucsat[c, 1])
        bsw_j = jnp.where(on, bsw[c, j], bsw[c, 1])

        # Relative saturation, clamped to [0.01, 1] — Fortran lines 147-150
        s = vwc_j / watsat_j
        s = jnp.minimum(s, 1.0)
        s = jnp.maximum(s, 0.01)

        # Hydraulic conductivity (mm H2O/s) — Fortran line 151
        hk_j = hksat_j * s ** (2.0 * bsw_j + 3.0)

        # Matric potential (mm), bounded below at -1e8 — Fortran lines 152-153
        smp_j = -sucsat_j * s ** (-bsw_j)
        smp_j = jnp.maximum(smp_j, -1.0e8)

        # Write to output arrays — Fortran lines 155-156
        hk_l = hk_l.at[c, j].set(jnp.where(on, hk_j, hk_l[c, j]))
        smp_l = smp_l.at[c, j].set(jnp.where(on, smp_j, smp_l[c, j]))

    return soilstate_inst._replace(
        hk_l_col=hk_l,
        smp_l_col=smp_l,
    )
