"""
Build :class:`clm_ml.forcing.Forcing` with the legacy readers, and convert records back.

Must be imported after ``InitializeRealize`` (the legacy modules it uses import
``clm_instMod``; see ``offline_executable/main.py``).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from clm_ml.forcing import ClmHistoryForcing, Forcing, TowerForcing, chunk_run_steps
from clm_src_main.clm_varpar import inir, ivis, numrad
from multilayer_canopy.MLGetAtmForcingMod import AtmForcingRecord, clm_forcing_to_ml
from offline_driver.clmDataMod import ClmHistoryRecord, read_clm_history_series
from offline_driver.TowerMetMod import TowerMet, derive_tower_met, read_tower_met_series


def load_forcing(
    tower_file: str,
    clm_history_file: str,
    soil_adjust_file: str,
    *,
    first_record: int,
    run_n_steps: int,
    first_step: int,
    n_steps: int,
    history_index: Callable[[int], int],
    tower_height: float,
) -> Forcing:
    """Read and derive one chunk of forcing for one tower (one gridcell).

    Args:
        tower_file, clm_history_file, soil_adjust_file: Input files.
        first_record: 0-based record of ``tower_file`` at the run's first step
            (``TowerMetMod.tower_first_record``).
        run_n_steps: Steps in the whole run (to know where the run ends).
        first_step, n_steps: The chunk, in 0-based run steps.
        history_index: 1-based CLM history time slice for a 0-based run step.
        tower_height: Forcing height from the tower table (m).
    """
    run_steps = chunk_run_steps(run_n_steps, first_step, n_steps)
    raws = read_tower_met_series(tower_file, [first_record + k for k in run_steps])
    mets = [derive_tower_met(raw, tower_height) for raw in raws]

    def col(name):
        return np.array([[getattr(m, name)] for m in mets], dtype=float)

    tower = TowerForcing(
        wind_speed=col("uref"),
        air_temperature=col("tref"),
        specific_humidity=col("qref"),
        air_pressure=col("pref"),
        longwave_down=col("lwsky"),
        rainfall=col("rain"),
        forcing_height=col("hgt"),
        co2=col("co2_ppm"),
        co2_partial_pressure=col("pco2"),
        o2_partial_pressure=col("po2"),
        shortwave_direct=np.array([[m.solad] for m in mets], dtype=float),
        shortwave_diffuse=np.array([[m.solai] for m in mets], dtype=float),
    )

    slices = [history_index(k) for k in run_steps]
    records = read_clm_history_series(clm_history_file, soil_adjust_file, slices)
    steps = [records[i] for i in slices]
    factors = [r.h2osoi_factor for r in steps]
    clm = ClmHistoryForcing(
        history_index=np.asarray(slices, dtype=int),
        leaf_area_index=np.array([[r.elai] for r in steps], dtype=float),
        stem_area_index=np.array([[r.esai] for r in steps], dtype=float),
        soil_water_clm45=np.array([[np.asarray(r.h2osoi_clm45)] for r in steps], dtype=float),
        soil_water_clm50=np.array([[np.asarray(r.h2osoi_clm50)] for r in steps], dtype=float),
        soil_water_factor=(
            None if factors[0] is None else np.array([[f] for f in factors], dtype=float)
        ),
    )
    return Forcing(tower=tower, clm_history=clm, first_step=first_step)


@dataclasses.dataclass(frozen=True)
class TowerForcingSource:
    """:class:`clm_ml.forcing.ForcingSource` for one tower: the tower meteorology file and
    the CLM history file (leaf/stem area, soil water), read chunk by chunk with
    :func:`load_forcing`."""

    tower_file: str
    clm_history_file: str
    soil_adjust_file: str
    first_record: int  # 0-based tower-file record at the run's first step
    n_steps: int  # steps in the whole run
    history_index: Callable[[int], int]
    tower_height: float

    def chunk(self, first_step: int, n_steps: int) -> Forcing:
        return load_forcing(
            self.tower_file,
            self.clm_history_file,
            self.soil_adjust_file,
            first_record=self.first_record,
            run_n_steps=self.n_steps,
            first_step=first_step,
            n_steps=n_steps,
            history_index=self.history_index,
            tower_height=self.tower_height,
        )


def tower_met_at(forcing: Forcing, t: int, g: int = 0) -> TowerMet:
    """The legacy ``TowerMet`` record at array index ``t`` of a chunk, gridcell ``g``."""
    f = forcing.tower
    return TowerMet(
        uref=float(f.wind_speed[t, g]),
        solad=(float(f.shortwave_direct[t, g, 0]), float(f.shortwave_direct[t, g, 1])),
        solai=(float(f.shortwave_diffuse[t, g, 0]), float(f.shortwave_diffuse[t, g, 1])),
        tref=float(f.air_temperature[t, g]),
        qref=float(f.specific_humidity[t, g]),
        pref=float(f.air_pressure[t, g]),
        lwsky=float(f.longwave_down[t, g]),
        rain=float(f.rainfall[t, g]),
        hgt=float(f.forcing_height[t, g]),
        co2_ppm=float(f.co2[t, g]),
        pco2=float(f.co2_partial_pressure[t, g]),
        po2=float(f.o2_partial_pressure[t, g]),
    )


def clm_record_at(forcing: Forcing, t: int, c: int = 0) -> ClmHistoryRecord:
    """The legacy ``ClmHistoryRecord`` at array index ``t`` of a chunk, soil column ``c``."""
    import jax.numpy as jnp

    h = forcing.clm_history
    return ClmHistoryRecord(
        elai=float(h.leaf_area_index[t, c]),
        esai=float(h.stem_area_index[t, c]),
        h2osoi_clm45=jnp.asarray(h.soil_water_clm45[t, c], dtype=jnp.float64),
        h2osoi_clm50=jnp.asarray(h.soil_water_clm50[t, c], dtype=jnp.float64),
        h2osoi_factor=None if h.soil_water_factor is None else float(h.soil_water_factor[t, c]),
    )


def _record(values, n_patch: int) -> AtmForcingRecord:
    """Per-patch forcing arrays (all patches of one gridcell share ``values``)."""
    import jax.numpy as jnp

    u, t, q, p, co2, solad, solai, lw = values
    one = jnp.zeros((n_patch,), dtype=jnp.float64)
    bands = jnp.zeros((n_patch, numrad + 1), dtype=jnp.float64)
    sel = slice(0, n_patch)
    return AtmForcingRecord(
        uref=one.at[sel].set(u),
        tref=one.at[sel].set(t),
        qref=one.at[sel].set(q),
        pref=one.at[sel].set(p),
        co2ref=one.at[sel].set(co2),
        swskyb=bands.at[sel, ivis].set(solad[0]).at[sel, inir].set(solad[1]),
        swskyd=bands.at[sel, ivis].set(solai[0]).at[sel, inir].set(solai[1]),
        lwsky=one.at[sel].set(lw),
    )


def atm_record_current(forcing: Forcing, t: int, n_patch: int, g: int = 0) -> AtmForcingRecord:
    """Multilayer canopy forcing at chunk array index ``t`` as ``_GetCLMVar`` derives it from the CLM
    forcing (wind resultant, CO2 from partial pressure): used as the previous-step
    record of the forcing window."""
    import jax.numpy as jnp

    m = tower_met_at(forcing, t, g)

    def f(x):
        return jnp.asarray(x, dtype=jnp.float64)

    values = clm_forcing_to_ml(
        f(m.uref), f(0.0), f(m.tref), f(m.qref), f(m.pref), f(m.pco2),
        (f(m.solad[0]), f(m.solad[1])), (f(m.solai[0]), f(m.solai[1])), f(m.lwsky),
    )  # fmt: skip
    return _record(values, n_patch)


def atm_record_next(forcing: Forcing, t: int, n_patch: int, g: int = 0) -> AtmForcingRecord:
    """Multilayer canopy forcing at chunk array index ``t`` as Fortran ``TowerMetNext`` stored it (wind
    speed and CO2 mole fraction taken directly): the next-step record of the window."""
    m = tower_met_at(forcing, t, g)
    values = (m.uref, m.tref, m.qref, m.pref, m.co2_ppm, m.solad, m.solai, m.lwsky)
    return _record(values, n_patch)
