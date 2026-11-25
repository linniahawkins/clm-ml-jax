"""The soil hydrology around the Richards solver against CLM's Fortran (CTSM
``ctsm5.4.004``): surface runoff, infiltration and the pond; the pond's covered fraction and
its constants; the water table; dew; drainage.

Golden data from the harness in ``ctsm_harness/`` (idealized sweeps, both model steps). All
the cases of a routine run in one vectorized call, and every output must equal the
Fortran's to ``REL_TOL`` of its value (``ABS_TOL`` near zero, as in ``test_golden_jax.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import drainage, soil_water, surface_runoff  # noqa: E402
from clm_ml.setup.soil import soil_layers  # noqa: E402

pytestmark = pytest.mark.fortran_validation

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_IO"
REL_TOL = 1e-9
ABS_TOL = 1e-15
NLEVSOI = 20

# The harness's parameters (ctsm60_params.c251211.nc; baseflow_scalar of the CLM6 namelist)
E_ICE, FFF, PC, MU = 6.0, 0.5, 0.4, 0.13889
BASEFLOW_SCALAR, N_BASEFLOW, AQ_SP_YIELD_MIN = 0.001, 1.0, 0.02
SLOPEBETA, SLOPEMAX = -3.0, 0.4

_z, _dz, _zi = soil_layers()
Z, DZ, ZI = (jnp.asarray(x) for x in (_z[:NLEVSOI], _dz[:NLEVSOI], _zi[: NLEVSOI + 1]))


def _load(name: str) -> tuple[dict, dict]:
    """The inputs (as arrays over the cases; ``dtime`` as a float) and the outputs."""
    with open(GOLDEN_DIR / name) as f:
        cases = json.load(f)["cases"]
    inputs = {k: jnp.asarray(np.array([c["inputs"][k] for c in cases])) for k in cases[0]["inputs"]}
    if "dtime" in inputs:
        inputs["dtime"] = inputs["dtime"].astype(jnp.float64)
    outputs = {k: np.array([c["outputs"][k] for c in cases]) for k in cases[0]["outputs"]}
    return inputs, outputs


def _assert_equal(got: dict, golden: dict, what: str) -> None:
    for key, expected in golden.items():
        value = np.asarray(got[key])
        excess = np.abs(value - expected) - (REL_TOL * np.abs(expected) + ABS_TOL)
        worst = np.unravel_index(np.argmax(excess), excess.shape)
        assert excess[worst] <= 0.0, (
            f"{what}: {key} of case {worst[0]}: got {value[worst]!r}, Fortran "
            f"{expected[worst]!r} ({int((excess > 0).sum())} values differ)"
        )


def test_surface_fluxes_equal_the_fortran():
    inputs, golden = _load("test_HydrologyNoDrainageMod_surface_fluxes.json")

    def column(i):
        icefrac, eff_porosity = soil_water.soil_water_fractions(i["h2osoi_ice"], i["watsat"], DZ)
        fluxes = surface_runoff.surface_fluxes(
            i["qflx_rain_plus_snomelt"], i["qflx_liqevap_from_top_layer"], i["qflx_ev_h2osfc"],
            i["h2osfc"], i["frac_h2osfc"], i["zwt"], icefrac, i["hksat"], i["wtfact"],
            i["h2osfc_thresh"], i["topo_slope"], i["dtime"], FFF, E_ICE, PC, MU,
        )  # fmt: skip
        return {"icefrac": icefrac, "eff_porosity": eff_porosity, **fluxes._asdict()}

    _assert_equal(jax.jit(jax.vmap(column))(inputs), golden, "surface fluxes")
    # the sweep reaches the pond's runoff, the infiltration limit, a refilled negative pond
    # and evaporation larger than the input
    assert np.sum(golden["qflx_h2osfc_surf"] > 0) > 30
    assert np.sum(golden["qflx_infl_excess"] > 0) > 30
    assert np.sum(golden["qflx_h2osfc_drain"] < 0) > 10
    assert np.sum(golden["qflx_infl"] < 0) > 30


def test_pond_fraction_equals_the_fortran():
    inputs, golden = _load("test_SurfaceWaterMod_UpdateFracH2oSfc.json")
    got = jax.jit(
        jax.vmap(
            lambda i: surface_runoff.pond_fraction(
                i["h2osfc"], i["h2osoi_liq_top"], i["micro_sigma"], i["dtime"]
            )._asdict()
        )
    )(inputs)
    _assert_equal(got, golden, "pond fraction")
    assert np.sum(golden["qflx_too_small_h2osfc_to_soil"] > 0) > 10


def test_pond_constants_equal_the_fortran():
    inputs, golden = _load("test_SoilHydrologyInitTimeConstMod_pond_constants.json")
    micro_sigma = surface_runoff.microtopography(inputs["topo_slope"], SLOPEBETA, SLOPEMAX)
    got = {
        "micro_sigma": micro_sigma,
        "h2osfc_thresh": surface_runoff.pond_threshold(micro_sigma, PC),
    }
    _assert_equal(got, golden, "pond constants")


def test_water_table_equals_the_fortran():
    inputs, golden = _load("test_SoilHydrologyMod_ThetaBasedWaterTable.json")
    zwt = jax.jit(
        jax.vmap(
            lambda i: drainage.water_table_depth(
                i["h2osoi_liq"], i["h2osoi_ice"], i["watsat"], DZ, Z, ZI, i["nbedrock"]
            )
        )
    )(inputs)
    _assert_equal({"zwt": zwt}, golden, "water table")
    # at the bottom of the top level, at the bedrock, and interpolated in between
    at_top = golden["zwt"] == float(ZI[1])
    at_bedrock = golden["zwt"] == np.asarray(ZI)[np.asarray(inputs["nbedrock"])]
    assert at_top.sum() > 10 and at_bedrock.sum() > 10
    assert np.sum(~at_top & ~at_bedrock) > 100


def test_renew_condensation_equals_the_fortran():
    inputs, golden = _load("test_SoilHydrologyMod_RenewCondensation.json")
    liq, ice = surface_runoff.renew_condensation(
        inputs["h2osoi_liq_top"], inputs["h2osoi_ice_top"], inputs["frac_h2osfc"],
        inputs["qflx_liqdew_to_top_layer"], inputs["qflx_soliddew_to_top_layer"],
        inputs["qflx_solidevap_from_top_layer"], inputs["dtime"],
    )  # fmt: skip
    _assert_equal({"h2osoi_liq_top": liq, "h2osoi_ice_top": ice}, golden, "condensation")


def test_subsurface_drainage_equals_the_fortran():
    inputs, golden = _load("test_SoilHydrologyMod_SubsurfaceLateralFlow.json")

    def column(i):
        _, eff_porosity = soil_water.soil_water_fractions(i["h2osoi_ice"], i["watsat"], DZ)
        out = drainage.subsurface_drainage(
            i["h2osoi_liq"], i["h2osoi_ice"], i["zwt"], i["h2osfc"], eff_porosity, i["watsat"],
            i["sucsat"], i["bsw"], DZ, ZI, i["nbedrock"], i["topo_slope"], i["dtime"],
            BASEFLOW_SCALAR, N_BASEFLOW, AQ_SP_YIELD_MIN, E_ICE,
        )  # fmt: skip
        return {**out._asdict(), "h2osoi_ice_top": out.h2osoi_ice[0]}

    _assert_equal(jax.jit(jax.vmap(column))(inputs), golden, "subsurface drainage")
    # the sweep reaches: drainage; less drained than asked (the column runs dry); a water
    # table falling through several levels; water passed up to the pond; water created to
    # keep the minimum; ice removed from the top level
    asked, drained = golden["qflx_latflow_out"], golden["qflx_drain"]
    zwt_in = np.asarray(inputs["zwt"])
    levels_crossed = np.sum(
        (np.asarray(ZI)[None, 1:] > zwt_in[:, None])
        & (np.asarray(ZI)[None, 1:] <= golden["zwt"][:, None]),
        axis=1,
    )
    ran_dry = (drained < asked * (1.0 - 1e-12)) & (golden["qflx_rsub_sat"] == 0)
    assert np.sum(asked > 0) > 100
    assert np.sum(ran_dry) >= 3
    assert np.sum(levels_crossed >= 2) > 10
    assert np.sum(golden["h2osfc"] > np.asarray(inputs["h2osfc"])) > 30
    assert np.sum(golden["qflx_rsub_sat"] < 0) >= 5
    assert np.sum(golden["qflx_ice_runoff_xs"] > 0) > 10
