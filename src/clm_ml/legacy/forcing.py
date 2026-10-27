"""
Build :class:`clm_ml.forcing.Forcing` with the legacy readers, and convert records back.

Must be imported after ``InitializeRealize`` (the legacy modules it uses import
``clm_instMod``; see ``offline_executable/main.py``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from clm_ml.forcing import ClmHistoryForcing, Forcing, TowerForcing
from offline_driver.clmDataMod import ClmHistoryRecord, read_clm_history_series
from offline_driver.TowerMetMod import TowerMet, derive_tower_met, read_tower_met_series


def load_forcing(
    tower_file: str,
    clm_history_file: str,
    soil_adjust_file: str,
    n_steps: int,
    history_index: Sequence[int],
    tower_height: float,
) -> Forcing:
    """Read and derive all forcing for ``n_steps`` steps of one tower (one gridcell)."""
    mets = [
        derive_tower_met(raw, tower_height) for raw in read_tower_met_series(tower_file, n_steps)
    ]

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

    records = read_clm_history_series(clm_history_file, soil_adjust_file, history_index)
    steps = [records[i] for i in history_index]
    factors = [r.h2osoi_factor for r in steps]
    clm = ClmHistoryForcing(
        history_index=np.asarray(history_index, dtype=int),
        leaf_area_index=np.array([[r.elai] for r in steps], dtype=float),
        stem_area_index=np.array([[r.esai] for r in steps], dtype=float),
        soil_water_clm45=np.array([[np.asarray(r.h2osoi_clm45)] for r in steps], dtype=float),
        soil_water_clm50=np.array([[np.asarray(r.h2osoi_clm50)] for r in steps], dtype=float),
        soil_water_factor=(
            None if factors[0] is None else np.array([[f] for f in factors], dtype=float)
        ),
    )
    return Forcing(tower=tower, clm_history=clm)


def tower_met_at(forcing: Forcing, t: int, g: int = 0) -> TowerMet:
    """The legacy ``TowerMet`` record for 0-based step ``t`` of gridcell ``g``."""
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
    """The legacy ``ClmHistoryRecord`` for 0-based step ``t`` of soil column ``c``."""
    import jax.numpy as jnp

    h = forcing.clm_history
    return ClmHistoryRecord(
        elai=float(h.leaf_area_index[t, c]),
        esai=float(h.stem_area_index[t, c]),
        h2osoi_clm45=jnp.asarray(h.soil_water_clm45[t, c], dtype=jnp.float64),
        h2osoi_clm50=jnp.asarray(h.soil_water_clm50[t, c], dtype=jnp.float64),
        h2osoi_factor=None if h.soil_water_factor is None else float(h.soil_water_factor[t, c]),
    )
