"""A chunk's inputs as 0-based arrays (Phase 7.3): the step's time and meteorology per
gridcell (:class:`clm_ml.step.MetInputs`) and its land data per patch
(:class:`clm_ml.step.LandInputs`), stacked along a leading step axis.

The forcing is prepared on the host a chunk ahead (plan decision 36): the meteorology at the
model step (already interpolated from the data records), the previous and next steps'
forcing records of the canopy's 3-point time interpolation (``*_bef``/``*_next`` in
Fortran), the leaf and stem area and the soil water. Grid runs form the land data inside the
compiled chunk from monthly records (``PatchBatch.land_fn``).

- :func:`met_inputs`: meteorology dicts of the previous, current and next steps (the derived
  fields of Fortran ``TowerMetMod``, ``TowerMet`` names) as :class:`MetInputs`.
- :func:`forcing_inputs`: a tower chunk (:class:`clm_ml.forcing.Forcing`) as
  ``(MetInputs, LandInputs)``.

Fortran reference: CLM-ml v2 ``TowerMetMod.F90::TowerMetCurr``/``TowerMetNext`` (the
current and next records), ``MLCanopyFluxesMod.F90::GetCLMVar`` (the previous record from the
CLM forcing) and ``clmDataMod.F90::clmData`` (the history record).
"""

from __future__ import annotations

import numpy as np

from clm_ml.forcing import Forcing
from clm_ml.physics.atm_forcing import ForcingRecord, clm_forcing_to_ml
from clm_ml.step import LandInputs, MetInputs

#: ``TowerMet`` field -> ``clm_ml.forcing.TowerForcing`` field
TOWER_FIELDS = {
    "uref": "wind_speed",
    "tref": "air_temperature",
    "qref": "specific_humidity",
    "pref": "air_pressure",
    "lwsky": "longwave_down",
    "rain": "rainfall",
    "co2_ppm": "co2",
    "pco2": "co2_partial_pressure",
    "po2": "o2_partial_pressure",
    "solad": "shortwave_direct",
    "solai": "shortwave_diffuse",
}


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _bands(x) -> np.ndarray:
    """Shortwave as ``(..., band)``: a ``(vis, nir)`` pair of arrays, or already stacked."""
    if isinstance(x, (tuple, list)):
        return np.stack([_f(x[0]), _f(x[1])], axis=-1)
    return _f(x)


def previous_record(met: dict) -> ForcingRecord:
    """The previous step's forcing record as ``GetCLMVar`` derives it from the CLM forcing
    (wind resultant with no north component, CO2 mole fraction from the partial pressure),
    with the same arithmetic as the current step's record inside the step."""
    import jax.numpy as jnp

    a = lambda x: jnp.asarray(_f(x))  # noqa: E731
    solad, solai = _bands(met["solad"]), _bands(met["solai"])
    u, t, q, p, co2, *_ = clm_forcing_to_ml(
        a(met["uref"]), jnp.zeros(()), a(met["tref"]), a(met["qref"]), a(met["pref"]),
        a(met["pco2"]), None, None, a(met["lwsky"]),
    )  # fmt: skip
    return ForcingRecord(
        uref=_f(u), tref=_f(t), qref=_f(q), pref=_f(p), co2ref=_f(co2),
        swskyb=solad, swskyd=solai, lwsky=_f(met["lwsky"]),
    )  # fmt: skip


def next_record(met: dict) -> ForcingRecord:
    """The next step's forcing record as Fortran ``TowerMetNext`` stores it (wind speed and
    CO2 mole fraction taken directly)."""
    return ForcingRecord(
        uref=_f(met["uref"]), tref=_f(met["tref"]), qref=_f(met["qref"]),
        pref=_f(met["pref"]), co2ref=_f(met["co2_ppm"]), swskyb=_bands(met["solad"]),
        swskyd=_bands(met["solai"]), lwsky=_f(met["lwsky"]),
    )  # fmt: skip


def met_inputs(bef: dict, cur: dict, nxt: dict, calday, calday_beg) -> MetInputs:
    """:class:`MetInputs` from the derived meteorology of the previous, current and next
    steps (``TowerMet`` field names; the same leading axes in all three, e.g. (step,
    gridcell)). ``snow`` is 0 when absent (tower files have no snowfall)."""
    rain = _f(cur["rain"])
    return MetInputs(
        calday=_f(calday),
        calday_beg=_f(calday_beg),
        uref=_f(cur["uref"]),
        tref=_f(cur["tref"]),
        qref=_f(cur["qref"]),
        pref=_f(cur["pref"]),
        lwsky=_f(cur["lwsky"]),
        rain=rain,
        snow=_f(cur["snow"]) if "snow" in cur else np.zeros_like(rain),
        solad=_bands(cur["solad"]),
        solai=_bands(cur["solai"]),
        pco2=_f(cur["pco2"]),
        po2=_f(cur["po2"]),
        forcing_bef=previous_record(bef),
        forcing_next=next_record(nxt),
    )


def forcing_inputs(forcing: Forcing, calday, calday_beg) -> tuple:
    """A tower chunk's ``(MetInputs, LandInputs)``: axes (step, gridcell, ...) and (step,
    patch, ...) for the chunk's steps (the halo records give the previous and next steps).
    ``calday``, ``calday_beg``: (step,) calendar days at the end and the beginning of each
    step."""
    tower, h = forcing.tower, forcing.clm_history
    n = forcing.n_steps

    def met(rows: slice) -> dict:
        return {k: getattr(tower, v)[rows] for k, v in TOWER_FIELDS.items()}

    g = tower.air_temperature.shape[1]
    day = np.broadcast_to(_f(calday)[:, None], (n, g))
    day_beg = np.broadcast_to(_f(calday_beg)[:, None], (n, g))
    met_in = met_inputs(met(slice(0, n)), met(slice(1, n + 1)), met(slice(2, n + 2)), day, day_beg)
    cur = slice(1, n + 1)
    elai = _f(h.leaf_area_index[cur])
    factor = h.soil_water_factor
    land = LandInputs(
        elai=elai,
        esai=_f(h.stem_area_index[cur]),
        h2osoi_clm45=_f(h.soil_water_clm45[cur]),
        h2osoi_clm50=_f(h.soil_water_clm50[cur]),
        h2osoi_factor=np.ones_like(elai) if factor is None else _f(factor[cur]),
    )
    return met_in, land
