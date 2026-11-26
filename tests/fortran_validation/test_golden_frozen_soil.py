"""Frozen soil against CLM's Fortran (CTSM ``ctsm5.4.004``): the thermal properties of soil
with ice, the freezing and thawing of soil water, the frost table and the perched water table, the saturated fraction with them,
the drainage of the perched zone, and the division of the ground's water vapour flux.

Golden data from the harness in ``ctsm_harness/`` (idealized sweeps, both model steps). All
the cases of a routine run in one vectorized call, and every output must equal the
Fortran's to ``REL_TOL`` of its value (``ABS_TOL`` near zero, as in ``test_golden_jax.py``).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import drainage, soil, surface_runoff  # noqa: E402
from clm_ml.setup.soil import soil_layers  # noqa: E402

from .test_golden_soil_hydrology import (  # noqa: E402
    AQ_SP_YIELD_MIN,
    DZ,
    FFF,
    NLEVSOI,
    ZI,
    Z,
    _assert_equal,
    _load,
)

pytestmark = pytest.mark.fortran_validation

TFRZ = 273.15
PERCHED_BASEFLOW_SCALAR = 1.0e-5  # ctsm60_params.c251211.nc


def test_soil_thermal_properties_with_ice_equal_the_fortran():
    """CLM-ml v2's copy of the routine against CTSM's, over frozen and thawed soil."""
    inputs, golden = _load("test_SoilTemperatureMod_SoilThermProp.json")
    z, dz, zi = (jnp.asarray(x) for x in soil_layers())

    def column(i):
        out = soil.soil_thermal_properties(
            i["t_soisno"], i["h2osoi_liq"], i["h2osoi_ice"], i["h2osfc"],
            i["h2osno_no_layers"], dz, z, zi[1:], i["watsat"], i["tkmg"], i["tkdry"],
            i["csol"], i["nbedrock"],
        )  # fmt: skip
        return out._asdict()

    _assert_equal(jax.jit(jax.vmap(column))(inputs), golden, "soil thermal properties")
    frozen = np.asarray(inputs["t_soisno"]) < TFRZ
    ice = np.asarray(inputs["h2osoi_ice"]) > 0
    assert (
        np.sum(frozen & ice) > 500 and np.sum(~frozen & ice) > 500 and np.sum(frozen & ~ice) > 500
    )


def test_phase_change_equals_the_fortran():
    inputs, golden = _load("test_SoilTemperatureMod_Phasechange_beta.json")

    def column(i):
        out = soil.phase_change(
            i["t_soisno"], i["h2osoi_liq"], i["h2osoi_ice"], i["h2osno_no_layers"],
            i["snow_depth"], i["fact"], i["dhsdT"], i["frac_h2osfc"], i["watsat"],
            i["sucsat"], i["bsw"], DZ, i["dtime"],
        )  # fmt: skip
        return {
            **out._asdict(),
            "h2osno_no_layers": out.h2osno,
            "qflx_snow_drain": out.qflx_snomelt,
        }

    got = jax.jit(jax.vmap(column))(inputs)
    _assert_equal(got, golden, "phase change")

    # the sweep reaches melting and freezing, complete and partial, the supercooled limit,
    # snow melted in part and wholly, and levels left alone
    t0, liq0, ice0 = (np.asarray(inputs[k]) for k in ("t_soisno", "h2osoi_liq", "h2osoi_ice"))
    imelt, ice, liq, t = (golden[k] for k in ("imelt", "h2osoi_ice", "h2osoi_liq", "t_soisno"))
    assert np.sum(imelt == 1) > 500 and np.sum(imelt == 2) > 500 and np.sum(imelt == 0) > 500
    assert np.sum((imelt == 1) & (ice > 0)) > 100  # ice left: the level stays at freezing
    assert np.sum((imelt == 1) & (ice == 0) & (t > TFRZ)) > 100  # all melted, warming again
    assert np.sum((imelt == 2) & (t == TFRZ)) > 100  # freezing in part
    assert np.sum((imelt == 2) & (t < TFRZ) & (liq > 0)) > 100  # down to the supercooled water
    assert np.sum((t0 < TFRZ) & (liq0 > 0) & (imelt == 0)) > 100  # all of it supercooled
    snow0, snow = np.asarray(inputs["h2osno_no_layers"]), golden["h2osno_no_layers"]
    assert np.sum((snow0 > 0) & (snow == 0)) > 10 and np.sum((snow > 0) & (snow < snow0)) > 10
    assert np.sum((snow0 > 0) & (snow == snow0)) > 10
    # with a heat flux given to the soil the routine conserves energy
    given = np.asarray(inputs["dhsdT"]) == 0.0
    fact = np.asarray(inputs["fact"])
    heat = np.sum((t - t0) / fact, axis=1) + golden["xmf"]
    assert np.max(np.abs(heat[given])) < 1e-6


def test_perched_water_table_equals_the_fortran():
    inputs, golden = _load("test_SoilHydrologyMod_PerchedWaterTable.json")

    def column(i):
        frost_table, zwt_perched = drainage.perched_water_table(
            i["t_soisno"], i["h2osoi_liq"], i["h2osoi_ice"], i["zwt"], i["watsat"], DZ, Z, ZI
        )
        return {"frost_table": frost_table, "zwt_perched": zwt_perched}

    _assert_equal(jax.jit(jax.vmap(column))(inputs), golden, "perched water table")
    frost, perched = golden["frost_table"], golden["zwt_perched"]
    interfaces = np.asarray(ZI)
    assert np.sum(frost == 0.0) > 30  # frozen from the surface
    assert np.sum(frost == interfaces[NLEVSOI - 1]) > 30  # nothing frozen under thawed soil
    assert np.sum((frost > 0.0) & (frost < interfaces[NLEVSOI - 1])) > 100
    assert np.sum(perched < frost) > 60  # a perched zone
    assert np.sum((perched < frost) & np.isin(perched, interfaces)) > 10  # at a level's top
    assert np.sum((perched < frost) & ~np.isin(perched, interfaces)) > 30  # interpolated


def test_saturated_fraction_with_a_frost_table_equals_the_fortran():
    inputs, golden = _load("test_SaturatedExcessRunoffMod_SaturatedExcessRunoff.json")
    fsat = surface_runoff.saturated_fraction(
        inputs["zwt"], inputs["wtfact"], FFF, inputs["frost_table"], inputs["zwt_perched"]
    )
    got = {"fsat": fsat, "qflx_sat_excess_surf": fsat * inputs["qflx_rain_plus_snomelt"]}
    _assert_equal(got, golden, "saturated fraction")
    unfrozen = np.asarray(surface_runoff.saturated_fraction(inputs["zwt"], inputs["wtfact"], FFF))
    assert np.sum(golden["fsat"] != unfrozen) > 15  # the perched water table is used


def test_perched_drainage_equals_the_fortran():
    inputs, golden = _load("test_SoilHydrologyMod_PerchedLateralFlow.json")

    def column(i):
        liq, qflx = drainage.perched_drainage(
            i["h2osoi_liq"], i["frost_table"], i["zwt_perched"], i["watsat"], i["hksat"],
            i["sucsat"], i["bsw"], DZ, ZI, i["nbedrock"], i["topo_slope"], i["dtime"],
            PERCHED_BASEFLOW_SCALAR, AQ_SP_YIELD_MIN,
        )  # fmt: skip
        return {"h2osoi_liq": liq, "qflx_drain_perched": qflx}

    _assert_equal(jax.jit(jax.vmap(column))(inputs), golden, "perched drainage")
    drained = golden["qflx_drain_perched"]
    taken = np.sum(np.asarray(inputs["h2osoi_liq"]) != golden["h2osoi_liq"], axis=1)
    assert np.sum(drained > 0) > 100 and np.sum(drained == 0) > 40
    assert np.sum(taken > 1) > 15  # from several levels
    # the zone gives less than was asked of it (a sliver, or no level between the tables)
    asked = (
        PERCHED_BASEFLOW_SCALAR
        * np.sin(np.asarray(inputs["topo_slope"]) * np.pi / 180.0)
        * np.max(np.asarray(inputs["hksat"]), axis=1)
        * np.maximum(np.asarray(inputs["frost_table"] - inputs["zwt_perched"]), 0.0)
    )
    assert np.sum((asked > 0) & (drained == 0)) > 5


def test_ground_vapor_fluxes_equal_the_fortran():
    inputs, golden = _load("test_SoilFluxesMod_evaporation_split.json")
    got = surface_runoff.ground_vapor_fluxes(
        inputs["qflx_ev_snow"], inputs["h2osoi_liq_top"], inputs["h2osoi_ice_top"],
        inputs["t_grnd"], inputs["frac_h2osfc"], inputs["dtime"],
    )._asdict()  # fmt: skip
    _assert_equal(got, golden, "ground vapour fluxes")
    for name in got:
        assert np.sum(golden[name] > 0) > 15, name
    # sublimation limited to the ice of the level
    limit = np.asarray(inputs["h2osoi_ice_top"]) / (
        np.asarray(inputs["dtime"]) * np.maximum(1.0 - np.asarray(inputs["frac_h2osfc"]), 1e-30)
    )
    assert np.sum((golden["qflx_solidevap_from_top_layer"] == limit) & (limit > 0)) > 5
