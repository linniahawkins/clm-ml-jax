"""The atmospheric forcing routine of the legacy time step as an adapter to
``clm_ml.physics.atm_forcing`` (Phase 7.2, process 3; transition only: it goes when the
containers leave the step, process 10), and the legacy forcing window (until the forcing is
built inside the chunk, Phase 7.3).

Legacy layout: a patch axis, shortwave bands at ``ivis = 1``, ``inir = 2`` (column 0 unused).
``DECLARED`` lists the container fields the adapter reads and writes (checked by
``tests/clm_ml/test_physics_atm_forcing.py``).
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from clm_ml.physics import atm_forcing as physics

#: adapter -> (fields read, fields written), as ``container.field``; the forcing window is
#: an argument, not a container field
_WRITES = ("uref", "tref", "qref", "pref", "co2ref", "swskyb", "swskyd", "lwsky", "eref",
           "rhomol", "rhoair", "mmair", "cpair", "thref", "thvref")  # fmt: skip
DECLARED = {
    "GetAtmForcing": (
        ("mlcanopy.zref_forcing", *(f"mlcanopy.{f}_forcing" for f in _WRITES)),
        tuple(f"mlcanopy.{f}_forcing" for f in _WRITES),
    )
}

#: Fortran ``met_type`` -> ``physics.TIME_INTERPOLATION`` name (2, two-point, stops the
#: model in CLM-ml v2)
MET_TYPE = {0: "none", 3: "three_point"}


class AtmForcingRecord(NamedTuple):
    """Atmospheric forcing at one CLM time step, per patch (legacy layout).

    Same arrays (and shapes) as the Fortran ``*_bef/_cur/_next_forcing`` fields:
    ``(np,)``, or ``(np, numrad+1)`` for shortwave (columns ``ivis``, ``inir``).
    """

    uref: Array  # Wind speed (m/s)
    tref: Array  # Air temperature (K)
    qref: Array  # Specific humidity (kg/kg)
    pref: Array  # Air pressure (Pa)
    co2ref: Array  # Atmospheric CO2 (umol/mol)
    swskyb: Array  # Direct beam solar radiation (W/m2)
    swskyd: Array  # Diffuse solar radiation (W/m2)
    lwsky: Array  # Atmospheric longwave radiation (W/m2)


class AtmForcingWindow(NamedTuple):
    """Forcing at the previous, current and next CLM time steps, for time interpolation.

    In Fortran these are stored in ``mlcanopy_type`` as ``*_bef/_cur/_next_forcing`` and
    carried from step to step; here they are an argument built from the run's forcing data.
    """

    bef: AtmForcingRecord
    cur: AtmForcingRecord
    next: AtmForcingRecord


def _patch_record(rec: AtmForcingRecord, p: int) -> physics.ForcingRecord:
    """Patch ``p`` of a legacy record, bands 0-based."""
    from clm_src_main.clm_varpar import numrad

    bands = slice(1, numrad + 1)
    return physics.ForcingRecord(
        rec.uref[p], rec.tref[p], rec.qref[p], rec.pref[p], rec.co2ref[p],
        rec.swskyb[p, bands], rec.swskyd[p, bands], rec.lwsky[p],
    )  # fmt: skip


def GetAtmForcing(
    time_bef,
    time_cur,
    time_next,
    time_ml,
    num_filter: int,
    filter,
    mlcanopy_inst,
    window: AtmForcingWindow,
    grid,
    *,
    opts,
):
    """Forcing at the reference height for the current sub-step (``physics.atm_forcing``).

    Fortran reference: MLGetAtmForcingMod.F90::GetAtmForcing.
    """
    from clm_src_main.clm_varpar import numrad

    if opts.met_type not in MET_TYPE:
        raise ValueError(f"met_type {opts.met_type} is not supported (0 or 3)")
    interpolation = MET_TYPE[opts.met_type]
    bands = slice(1, numrad + 1)
    inst = mlcanopy_inst
    for _ in range(num_filter):
        p = grid.p
        f = physics.atm_forcing(
            _patch_record(window.bef, p),
            _patch_record(window.cur, p),
            _patch_record(window.next, p),
            inst.zref_forcing[p],
            time_bef,
            time_cur,
            time_next,
            time_ml,
            interpolation=interpolation,
        )
        new = {}
        for name, value in f._asdict().items():
            field = f"{name}_forcing"
            old = getattr(inst, field)
            index = (p, bands) if name in ("swskyb", "swskyd") else p
            new[field] = old.at[index].set(value)
        inst = inst._replace(**new)
    return inst
