"""
JAX translation of clm_driver Fortran module.

Main CLM model driver to calculate surface fluxes. Orchestrates the
full physics sequence for a single driver time step, including soil
albedo, evaporative resistance, thermal conductivity, soil water
movement, multilayer canopy fluxes, and soil temperature updates.

Original Fortran module: clm_driver
Fortran lines 1-110
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from clm_src_biogeophys.SoilWaterMovementMod import SoilWater  # noqa: F401
from clm_src_biogeophys.SurfaceAlbedoMod import SoilAlbedo  # noqa: F401
from clm_src_biogeophys.SurfaceResistanceMod import calc_soilevap_resis  # noqa: F401
from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_instMod import ClmInstances
from clm_src_main.clm_varpar import nlevgrnd, nlevsno  # noqa: F401
from clm_src_main.ColumnType import col  # noqa: F401
from clm_src_main.decompMod import bounds_type  # noqa: F401
from clm_src_main.filterMod import clumpfilter, setExposedvegpFilter  # noqa: F401
from clm_src_utils.clm_time_manager import StepTime
from multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes  # noqa: F401
from multilayer_canopy.MLGetAtmForcingMod import AtmForcingRecord
from multilayer_canopy.MLSoilTemperatureMod import SoilTemperature, SoilThermProp  # noqa: F401
from offline_driver.clmDataMod import ClmHistoryRecord, clmData  # noqa: F401

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clm_drv(
    bounds: bounds_type,
    insts: ClmInstances,
    filter: clumpfilter,
    clm_record: ClmHistoryRecord,
    step_time: StepTime,
    forcing_bef: AtmForcingRecord,
    forcing_next: AtmForcingRecord,
    initialize: bool,
    params=None,
    *,
    opts,
    grid=None,
) -> tuple[ClmInstances, clumpfilter]:
    """
    Main CLM model driver to calculate surface fluxes.

    Mirrors Fortran subroutine ``clm_drv`` (lines 22-105).

    Executes the following physics sequence for a single driver
    time step:

    1. Read CLM data for the current time slice (``clmData``).
    2. Set fractional vegetation and its filter
       (``setExposedvegpFilter``).
    3. Compute soil albedo (``SoilAlbedo``).
    4. Compute soil evaporative resistance
       (``calc_soilevap_resis``).
    5. Zero out snow and surface water for non-lake columns.
    6. Compute soil thermal conductivity and heat capacity
       (``SoilThermProp``).
    7. Compute soil hydraulic conductivity and matric potential
       (``SoilWater``).
    8. Compute multilayer canopy and soil fluxes
       (``MLCanopyFluxes``).
    9. Update soil temperatures (``SoilTemperature``).

    .. warning::
        The soil temperature routine (``SoilTemperature``) is specific
        to the multilayer canopy and is **not** the same as in
        standard CLM. Fortran lines 96-100.

    Args:
        bounds: CLM decomposition bounds for the local MPI task,
            supplying ``begp``, ``endp``, ``begc``, and ``endc``.
        insts: The model state containers (``clm_instMod.ClmInstances``).
            Fortran updated module-level instances in place; here the
            updated containers are returned.
        filter: Patch/column filters; the exposed-vegetation filter is
            rebuilt each step and returned.
        clm_record: CLM history values for this time step (see
            ``clmDataMod.read_clm_history_series``). Replaces the Fortran
            arguments ``time_indx``, ``fin1`` and ``fin2``: the history
            file is read once before the time loop. Fortran ``time_indx``
            was the time index from the reference date (0Z January 1
            of the current year, where ``calday == 1.000``).
        step_time: Time information for this step (replaces the Fortran
            calls to the time manager inside the physics).
        forcing_bef, forcing_next: Multilayer canopy forcing at the previous
            and next CLM steps, for time interpolation (see ``MLCanopyFluxes``).
        initialize: True on the run's first time step (multilayer canopy
            vertical-structure setup; see ``MLCanopyFluxes``).
    """
    # ------------------------------------------------------------------
    # Unpack the model state (passed in explicitly; no module globals)
    # ------------------------------------------------------------------

    canopystate_inst = insts.canopystate
    waterdiagnosticbulk_inst = insts.waterdiagnosticbulk
    water_inst = insts.water
    waterstatebulk_inst = insts.waterstatebulk
    soilstate_inst = insts.soilstate
    temperature_inst = insts.temperature
    atm2lnd_inst = insts.atm2lnd
    waterfluxbulk_inst = insts.waterfluxbulk
    energyflux_inst = insts.energyflux
    frictionvel_inst = insts.frictionvel
    surfalb_inst = insts.surfalb
    solarabs_inst = insts.solarabs
    mlcanopy_inst = insts.mlcanopy
    wateratm2lndbulk_inst = insts.wateratm2lndbulk

    # ------------------------------------------------------------------
    # Unpack associate aliases (Fortran lines 47-54)
    # ------------------------------------------------------------------
    snl: Array = col.snl  # Number of snow layers
    frac_veg_nosno: Array = (
        canopystate_inst.frac_veg_nosno_patch
    )  # Fraction of veg not covered by snow (0 or 1)
    frac_sno_eff: Array = (
        waterdiagnosticbulk_inst.frac_sno_eff_col
    )  # Effective fraction of ground covered by snow
    h2osno: Array = water_inst.h2osno_col  # Total snow water (kg H2O/m2)
    h2osfc: Array = waterstatebulk_inst.h2osfc_col  # Surface water (kg H2O/m2)

    # Local work arrays for soil thermal properties — Fortran lines 41-43
    # Shape: (n_col, nlevsno + nlevgrnd + 1) — the +1 is needed because
    # SoilThermProp uses jpy = j + nlevsno mapping j from -nlevsno+1..nlevgrnd
    # giving jpy from 1..nlevsno+nlevgrnd (0-based needs size nlevsno+nlevgrnd+1)
    n_col = bounds.endc - bounds.begc + 1
    n_lev = (
        nlevsno + nlevgrnd + 1
    )  # Fortran: -nlevsno+1:nlevgrnd → size nlevsno+nlevgrnd, offset by 1
    cv = jnp.zeros((n_col, n_lev), dtype=jnp.float64)  # Soil heat capacity (J/m2/K)
    tk = jnp.zeros((n_col, n_lev), dtype=jnp.float64)  # Soil thermal conductivity (W/m/K)
    tk_h2osfc = jnp.zeros((n_col,), dtype=jnp.float64)  # Thermal conductivity of h2osfc (W/m/K)

    # ------------------------------------------------------------------
    # 1. CLM data for current time slice — Fortran lines 57-59
    # (read before the time loop; installed here)
    # ------------------------------------------------------------------
    waterstatebulk_inst, canopystate_inst, surfalb_inst = clmData(
        clm_record,
        bounds.begp,
        bounds.endp,
        bounds.begc,
        bounds.endc,
        soilstate_inst,
        waterstatebulk_inst,
        canopystate_inst,
        surfalb_inst,
    )

    # ------------------------------------------------------------------
    # 2. Set frac_veg_nosno and exposedvegp filter — Fortran lines 61-64
    # ------------------------------------------------------------------
    # Local and constant (1): a NumPy array of the same shape, so the filter stays static when
    # the step is traced (the container's values are not needed; they may be traced)
    frac_veg_nosno = np.zeros(np.shape(frac_veg_nosno), dtype=int)
    for p in range(bounds.begp, bounds.endp + 1):  # Fortran: do p = bounds%begp, bounds%endp
        frac_veg_nosno[p] = 1

    # setExposedvegpFilter returns a new filter; it is returned to the caller
    # with the updated state.
    filter = setExposedvegpFilter(filter, frac_veg_nosno)

    # ------------------------------------------------------------------
    # 3. Soil albedo — Fortran lines 66-67
    # ------------------------------------------------------------------
    assert filter.nourbanc is not None, "filter.nourbanc must be initialized"
    surfalb_inst = SoilAlbedo(
        bounds,
        filter.num_nourbanc,
        filter.nourbanc,
        waterstatebulk_inst,
        surfalb_inst,
    )

    # ------------------------------------------------------------------
    # 4. Soil evaporative resistance — Fortran lines 69-71
    # ------------------------------------------------------------------
    assert filter.nolakec is not None, "filter.nolakec must be initialized"
    soilstate_inst = calc_soilevap_resis(
        bounds,
        filter.num_nolakec,
        filter.nolakec,
        soilstate_inst,
        waterstatebulk_inst,
        temperature_inst,
    )

    # ------------------------------------------------------------------
    # 5. Zero out snow and surface water — Fortran lines 73-79
    # ------------------------------------------------------------------
    for f in range(filter.num_nolakec):  # Fortran: do f = 1, filter%num_nolakec
        c = int(filter.nolakec[f])
        snl = snl.at[c].set(0)
        frac_sno_eff = frac_sno_eff.at[c - bounds.begc].set(0.0)
        h2osno = h2osno.at[c - bounds.begc].set(0.0)
        h2osfc = h2osfc.at[c - bounds.begc].set(0.0)

    # Write zeroed arrays back into their containers
    waterdiagnosticbulk_inst = waterdiagnosticbulk_inst._replace(frac_sno_eff_col=frac_sno_eff)
    water_inst = water_inst._replace(h2osno_col=h2osno)
    waterstatebulk_inst = waterstatebulk_inst._replace(h2osfc_col=h2osfc)

    # ------------------------------------------------------------------
    # 6. Soil thermal conductivity and heat capacity — Fortran lines 83-87
    # ------------------------------------------------------------------
    assert filter.nolakec is not None, "filter.nolakec must be initialized"
    soilstate_inst, temperature_inst, tk, cv, tk_h2osfc = SoilThermProp(
        bounds,
        filter.num_nolakec,
        filter.nolakec,
        tk,
        cv,
        tk_h2osfc,
        temperature_inst,
        waterdiagnosticbulk_inst,
        waterstatebulk_inst,
        water_inst,
        soilstate_inst,
    )

    # ------------------------------------------------------------------
    # 7. Soil hydraulic conductivity and matric potential — Fortran lines 89-91
    # ------------------------------------------------------------------
    assert filter.hydrologyc is not None, "filter.hydrologyc must be initialized"
    soilstate_inst = SoilWater(
        bounds,
        filter.num_hydrologyc,
        filter.hydrologyc,
        soilstate_inst,
        waterstatebulk_inst,
    )

    # ------------------------------------------------------------------
    # 8. Multilayer canopy and soil fluxes — Fortran lines 93-99
    # ------------------------------------------------------------------
    assert filter.exposedvegp is not None, "filter.exposedvegp must be initialized"
    # Convert JAX array to list for MLCanopyFluxes signature
    exposedvegp_list = [int(p) for p in filter.exposedvegp[: filter.num_exposedvegp]]
    mlcanopy_inst = MLCanopyFluxes(
        bounds,
        filter.num_exposedvegp,
        exposedvegp_list,
        atm2lnd_inst,
        canopystate_inst,
        soilstate_inst,
        temperature_inst,
        waterstatebulk_inst,
        waterfluxbulk_inst,
        energyflux_inst,
        frictionvel_inst,
        surfalb_inst,
        solarabs_inst,
        mlcanopy_inst,
        wateratm2lndbulk_inst,
        waterdiagnosticbulk_inst,
        step_time,
        forcing_bef,
        forcing_next,
        initialize,
        params=params,
        opts=opts,
        grid=grid,
    )

    # ------------------------------------------------------------------
    # 9. Update soil temperatures — Fortran lines 101-104
    # WARNING: specific to multilayer canopy, not standard CLM
    # ------------------------------------------------------------------
    assert filter.nolakec is not None, "filter.nolakec must be initialized"
    temperature_inst, soilstate_inst = SoilTemperature(
        bounds,
        filter.num_nolakec,
        filter.nolakec,
        soilstate_inst,
        temperature_inst,
        waterdiagnosticbulk_inst,
        waterstatebulk_inst,
        water_inst,
        mlcanopy_inst,
        step_time.dtime,
    )

    # ------------------------------------------------------------------
    # Return the updated state (Fortran updated module-level instances)
    # ------------------------------------------------------------------
    return (
        insts._replace(
            canopystate=canopystate_inst,
            waterdiagnosticbulk=waterdiagnosticbulk_inst,
            water=water_inst,
            waterstatebulk=waterstatebulk_inst,
            soilstate=soilstate_inst,
            temperature=temperature_inst,
            waterfluxbulk=waterfluxbulk_inst,
            energyflux=energyflux_inst,
            frictionvel=frictionvel_inst,
            surfalb=surfalb_inst,
            solarabs=solarabs_inst,
            mlcanopy=mlcanopy_inst,
        ),
        filter,
    )
