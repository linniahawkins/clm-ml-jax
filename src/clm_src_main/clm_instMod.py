"""
JAX translation of clm_instMod Fortran module.

Instances and definitions of all CLM component data types.
Provides module-level singletons for every state container used
throughout the CLM and CLMml physics, together with initialization.
The Fortran ``clm_instRest`` (restart I/O, all calls commented out) is not
translated; restarts are written by the run driver.

Original Fortran module: clm_instMod
Fortran lines 1-105
"""

from typing import NamedTuple, Optional

from clm_src_biogeophys.CanopyStateType import (
    canopystate_type,
)  # noqa: F401
from clm_src_biogeophys.CanopyStateType import (
    init_allocate_from_bounds as _canopystate_Init,
)
from clm_src_biogeophys.EnergyFluxType import (
    energyflux_type,
)  # noqa: F401
from clm_src_biogeophys.EnergyFluxType import (
    init as _energyflux_Init,
)
from clm_src_biogeophys.FrictionVelocityMod import (
    frictionvel_type,
)  # noqa: F401
from clm_src_biogeophys.FrictionVelocityMod import (
    init_friction_velocity as _frictionvel_Init,
)
from clm_src_biogeophys.SoilStateInitTimeConstMod import SoilStateInitTimeConst  # noqa: F401
from clm_src_biogeophys.SoilStateType import (
    create_soilstate as _soilstate_Init,
)
from clm_src_biogeophys.SoilStateType import (
    soilstate_type,
)  # noqa: F401
from clm_src_biogeophys.SolarAbsorbedType import init as _solarabs_Init
from clm_src_biogeophys.SolarAbsorbedType import solarabs_type  # noqa: F401
from clm_src_biogeophys.SurfaceAlbedoMod import SurfaceAlbedoInitTimeConst  # noqa: F401
from clm_src_biogeophys.SurfaceAlbedoType import (
    init_allocate as _surfalb_Init,
)
from clm_src_biogeophys.SurfaceAlbedoType import (
    surfalb_type,
)  # noqa: F401
from clm_src_biogeophys.TemperatureType import (
    Init as _temperature_Init,
)
from clm_src_biogeophys.TemperatureType import (
    temperature_type,
)  # noqa: F401
from clm_src_biogeophys.WaterDiagnosticBulkType import (
    Init as _waterdiagnosticbulk_Init,
)
from clm_src_biogeophys.WaterDiagnosticBulkType import (
    waterdiagnosticbulk_type,
)  # noqa: F401
from clm_src_biogeophys.WaterFluxBulkType import (
    Init as _waterfluxbulk_Init,
)
from clm_src_biogeophys.WaterFluxBulkType import (
    waterfluxbulk_type,
)  # noqa: F401
from clm_src_biogeophys.WaterStateBulkType import (
    Init as _waterstatebulk_Init,
)
from clm_src_biogeophys.WaterStateBulkType import (
    waterstatebulk_type,
)  # noqa: F401
from clm_src_biogeophys.WaterType import Init as _water_Init
from clm_src_biogeophys.WaterType import water_type  # noqa: F401
from clm_src_main.atm2lndType import Init as _atm2lnd_Init
from clm_src_main.atm2lndType import atm2lnd_type  # noqa: F401
from clm_src_main.decompMod import bounds_type  # noqa: F401
from clm_src_main.initVerticalMod import initVertical  # noqa: F401
from clm_src_main.SiteType import site_type  # noqa: F401
from clm_src_main.wateratm2lndBulkType import (
    Init as _wateratm2lndbulk_Init,
)
from clm_src_main.wateratm2lndBulkType import (
    wateratm2lndbulk_type,
)  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import (
    create_mlcanopy as _mlcanopy_Init,
)
from multilayer_canopy.MLCanopyFluxesType import (
    mlcanopy_type,
)  # noqa: F401  # CLMml

# ---------------------------------------------------------------------------
# Module-level singletons (Fortran lines 40-55)
# These are initialised to None and populated by clm_instInit.
# Mirrors Fortran public module-level instances.
# ---------------------------------------------------------------------------

atm2lnd_inst: atm2lnd_type = None  # type: ignore[assignment]
wateratm2lndbulk_inst: wateratm2lndbulk_type = None  # type: ignore[assignment]
soilstate_inst: soilstate_type = None  # type: ignore[assignment]
water_inst: water_type = None  # type: ignore[assignment]
waterstatebulk_inst: waterstatebulk_type = None  # type: ignore[assignment]
waterfluxbulk_inst: waterfluxbulk_type = None  # type: ignore[assignment]
waterdiagnosticbulk_inst: waterdiagnosticbulk_type = None  # type: ignore[assignment]
canopystate_inst: canopystate_type = None  # type: ignore[assignment]
temperature_inst: temperature_type = None  # type: ignore[assignment]
energyflux_inst: energyflux_type = None  # type: ignore[assignment]
frictionvel_inst: frictionvel_type = None  # type: ignore[assignment]
surfalb_inst: surfalb_type = None  # type: ignore[assignment]
solarabs_inst: solarabs_type = None  # type: ignore[assignment]
mlcanopy_inst: mlcanopy_type = None  # type: ignore[assignment]  # CLMml


class ClmInstances(NamedTuple):
    """All model state containers, passed explicitly through the time step.

    The module-level ``*_inst`` singletons above are filled once by
    :func:`clm_instInit`; after initialization the driver takes them with
    :func:`get_instances` and carries them as this one value, which
    ``ModelAdvance``/``clm_drv`` take and return. No routine in the time loop
    reads or writes the singletons.
    """

    atm2lnd: atm2lnd_type
    wateratm2lndbulk: wateratm2lndbulk_type
    soilstate: soilstate_type
    water: water_type
    waterstatebulk: waterstatebulk_type
    waterfluxbulk: waterfluxbulk_type
    waterdiagnosticbulk: waterdiagnosticbulk_type
    canopystate: canopystate_type
    temperature: temperature_type
    energyflux: energyflux_type
    frictionvel: frictionvel_type
    surfalb: surfalb_type
    solarabs: solarabs_type
    mlcanopy: mlcanopy_type
    # Per-site data the time step reads (soil geometry, soil colour, location; Phase 6 step 2),
    # set at the end of setup (offline_executable.main.setup_run); None before.
    site: Optional[site_type] = None


def get_instances() -> ClmInstances:
    """The state containers created by :func:`clm_instInit`, as one value."""
    return ClmInstances(
        atm2lnd=atm2lnd_inst,
        wateratm2lndbulk=wateratm2lndbulk_inst,
        soilstate=soilstate_inst,
        water=water_inst,
        waterstatebulk=waterstatebulk_inst,
        waterfluxbulk=waterfluxbulk_inst,
        waterdiagnosticbulk=waterdiagnosticbulk_inst,
        canopystate=canopystate_inst,
        temperature=temperature_inst,
        energyflux=energyflux_inst,
        frictionvel=frictionvel_inst,
        surfalb=surfalb_inst,
        solarabs=solarabs_inst,
        mlcanopy=mlcanopy_inst,
    )


# ---------------------------------------------------------------------------
# Public: initialize all component instances
# ---------------------------------------------------------------------------


def clm_instInit(bounds: bounds_type, max_layers: int) -> None:
    """
    Initialize all public CLM component-type instances.

    Mirrors Fortran subroutine ``clm_instInit`` (lines 62-83).

    Initialization order matches the Fortran source exactly:

    1. ``initVertical`` — vertical grid setup.
    2. ``atm2lnd_inst.Init`` — atmosphere-to-land forcing.
    3. ``wateratm2lndbulk_inst.Init`` — bulk atm-to-land water fluxes.
    4. ``soilstate_inst.Init`` — soil state allocation.
    5. ``SoilStateInitTimeConst`` — soil hydraulic/thermal constants.
    6. ``water_inst.Init`` — snow water.
    7. ``waterstatebulk_inst.Init`` — bulk water state.
    8. ``waterfluxbulk_inst.Init`` — bulk water fluxes.
    9. ``waterdiagnosticbulk_inst.Init`` — bulk water diagnostics.
    10. ``canopystate_inst.Init`` — canopy state.
    11. ``temperature_inst.Init`` — temperature state.
    12. ``energyflux_inst.Init`` — energy fluxes.
    13. ``frictionvel_inst.Init`` — friction velocity.
    14. ``surfalb_inst.Init`` — surface albedo.
    15. ``solarabs_inst.Init`` — solar absorption.
    16. ``SurfaceAlbedoInitTimeConst`` — albedo lookup tables.
    17. ``mlcanopy_inst.Init`` — multilayer canopy (CLMml).

    Args:
        bounds: Decomposition bounds for the local MPI task, supplying
            ``begg``, ``endg``, ``begc``, ``endc``, ``begp``, and
            ``endp``.
    """
    global atm2lnd_inst, wateratm2lndbulk_inst, soilstate_inst
    global water_inst, waterstatebulk_inst, waterfluxbulk_inst
    global waterdiagnosticbulk_inst, canopystate_inst, temperature_inst
    global energyflux_inst, frictionvel_inst, surfalb_inst, solarabs_inst
    global mlcanopy_inst

    # Fortran line 65
    initVertical(bounds)

    # Fortran lines 66-82
    atm2lnd_inst = _atm2lnd_Init(bounds)
    wateratm2lndbulk_inst = _wateratm2lndbulk_Init(bounds)
    soilstate_inst = _soilstate_Init(bounds)
    soilstate_inst = SoilStateInitTimeConst(bounds, soilstate_inst)
    water_inst = _water_Init(bounds)
    waterstatebulk_inst = _waterstatebulk_Init(bounds)
    waterfluxbulk_inst = _waterfluxbulk_Init(bounds)
    waterdiagnosticbulk_inst = _waterdiagnosticbulk_Init(bounds)
    canopystate_inst = _canopystate_Init(bounds)  # type: ignore[arg-type]
    temperature_inst = _temperature_Init(bounds)  # type: ignore[arg-type]
    energyflux_inst = _energyflux_Init(bounds)  # type: ignore[arg-type]
    frictionvel_inst = _frictionvel_Init(bounds)  # type: ignore[arg-type]
    surfalb_inst = _surfalb_Init(bounds)  # type: ignore[arg-type]
    solarabs_inst = _solarabs_Init(bounds)  # type: ignore[arg-type]
    SurfaceAlbedoInitTimeConst(bounds)
    mlcanopy_inst = _mlcanopy_Init(bounds.begp, bounds.endp, max_layers)  # CLMml
