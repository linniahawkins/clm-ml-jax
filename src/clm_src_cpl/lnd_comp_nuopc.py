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
    clm_record,
    step_time,
    forcing_bef,
    forcing_next,
) -> None:
    """
    Advance the CLM model by one time step.

    Mirrors Fortran subroutine ``ModelAdvance`` (lines 46-68).
    Delegates entirely to ``clm_drv``, which orchestrates the full
    physics sequence for a single driver time step. Model state is
    modified through global module-level instances in ``clm_instMod``.

    Args:
        bounds: Decomposition bounds for the local MPI task.
        clm_record: CLM history values for this time step
            (``clmDataMod.ClmHistoryRecord``). Replaces the Fortran
            arguments ``time_indx``, ``fin1`` and ``fin2``: input files are
            read once before the time loop.
        step_time: Time information for this step
            (``clm_time_manager.StepTime``).
        forcing_bef, forcing_next: Multilayer canopy forcing at the previous
            and next CLM steps (``MLGetAtmForcingMod.AtmForcingRecord``).
    """
    from clm_src_main.clm_driver import clm_drv  # noqa: F401

    clm_drv(bounds, clm_record, step_time, forcing_bef, forcing_next)
