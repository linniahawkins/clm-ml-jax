"""
JAX translation of lnd_comp_nuopc Fortran module.

Interface of the active land model component of CESM (CLM, Community Land Model)
with the main CESM driver. Provides initialization and time-stepping entry points
for the land surface model.

Original Fortran module: lnd_comp_nuopc
Fortran lines 1-72
"""

from clm_src_main.decompMod import bounds_type

# ---------------------------------------------------------------------------
# Public interface functions
# ---------------------------------------------------------------------------


def InitializeRealize(bounds: bounds_type) -> None:
    """
    Initialize the land surface model (CLM).

    Mirrors Fortran subroutine ``InitializeRealize`` (lines 30-44).
    Calls the two-phase CLM initialization sequence: ``initialize1``
    performs domain/grid setup independent of decomposition bounds, while
    ``initialize2`` completes state-variable allocation using the provided
    decomposition bounds.

    Args:
        bounds: Decomposition bounds for the local MPI task, describing the
            index ranges of gridcells, landunits, columns, and patches owned
            by this task.
    """
    from clm_src_main.clm_initializeMod import initialize1, initialize2  # noqa: F401

    initialize1()
    initialize2(bounds)


def ModelAdvance(
    bounds: bounds_type,
    insts,
    filter,
    clm_record,
    step_time,
    forcing_bef,
    forcing_next,
    initialize,
):
    """
    Advance the CLM model by one time step.

    Mirrors Fortran subroutine ``ModelAdvance`` (lines 46-68).
    Delegates entirely to ``clm_drv``, which orchestrates the full
    physics sequence for a single driver time step. The model state is
    passed in and the updated state returned (Fortran modified module-level
    instances in place).

    Args:
        bounds: Decomposition bounds for the local MPI task.
        insts: Model state containers (``clm_instMod.ClmInstances``).
        filter: Patch/column filters (``filterMod.clumpfilter``).
        clm_record: CLM history values for this time step
            (``clmDataMod.ClmHistoryRecord``). Replaces the Fortran
            arguments ``time_indx``, ``fin1`` and ``fin2``: input files are
            read once before the time loop.
        step_time: Time information for this step
            (``clm_time_manager.StepTime``).
        forcing_bef, forcing_next: Multilayer canopy forcing at the previous
            and next CLM steps (``MLGetAtmForcingMod.AtmForcingRecord``).
        initialize: True on the run's first time step.

    Returns:
        ``(insts, filter)``: the updated state containers and filters.
    """
    from clm_src_main.clm_driver import clm_drv  # noqa: F401

    return clm_drv(
        bounds, insts, filter, clm_record, step_time, forcing_bef, forcing_next, initialize
    )
