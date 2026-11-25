"""
Predicted soil water end to end (model.yaml ``soil.water: predicted``) at CHATS.

- The run starts from the soil water a prescribed run has at its first step, so the first
  step's fluxes are those of the regression reference (to round-off).
- The soil water balance closes at every step, and the change of the stored water from one
  step to the next is the step's infiltration less root uptake and drainage.
- A two-day run with rain on the second day (3 mm; the canopy has unused layer slots)
  keeps the water balance closed, and resumed from the restart written after its first
  day it gives the same results, bit for bit: the soil water, the pond and the water table
  are in the restart.
- Without the tower's largest saturated fraction and slope the run refuses to start.
- A grid run (two 1° gridcells in Illinois, 11 patches of 6 PFTs, two days of July 2000
  with rain on the second) takes the largest saturated fraction and the slope from the
  surface data and keeps the water balance of every patch closed.

(CHATS is an irrigated orchard: the runs test the code, not the site's water.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests" / "regression" / "reference" / "CHATS7_50steps.npz"
SCALARS = [
    "latent_heat_flux", "ground_water_input", "ground_evaporation", "root_water_uptake",
    "infiltration", "surface_runoff", "drainage", "saturated_fraction", "surface_water",
    "surface_water_fraction", "water_table_depth", "soil_water_substeps",
    "soil_water_balance_error",
]  # fmt: skip
VARIABLES = SCALARS + ["soil_liquid_water", "soil_water_content"]
SURFACE = {"max_saturated_fraction": 0.4, "slope": 1.0}


def _experiment(
    workdir: Path, length: str, restart: Path | None = None, surface=SURFACE, soil=None
):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from tests.experiment_tools import make_experiment

    run = {
        "time": {"length": length},
        "execution": {"chunk": "1 day"},  # a restart continues at a chunk's start
        "output": {
            "directory": str(workdir / "out"),
            "streams": [{"name": "step", "frequency": "step", "statistic": "instantaneous",
                         "variables": VARIABLES}],
        },
    }  # fmt: skip
    if restart is not None:
        run["initial_conditions"] = {"source": "restart", "restart_file": str(restart)}
    make_experiment(
        workdir,
        namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day",
        run=run,
        model={"soil": {"water": "predicted", **(soil or {})}},
        domain=lambda data: data["towers"]["CHATS7"]["soil"].update(surface),
    )
    return load_experiment(workdir)


def test_predicted_soil_water_one_day(tmp_path):
    full = _experiment(tmp_path / "full", "1 day").execute(quiet=True)["step"].data
    assert full["latent_heat_flux"].shape[0] == 48
    for name in VARIABLES:
        assert np.all(np.isfinite(full[name])), name

    # the first step has the prescribed run's soil water (to round-off: the state holds it
    # in kg m-2, the forcing gives it by volume). Its fluxes are close, not equal: with
    # predicted soil water the canopy sees the soil levels weighted by their conductance
    with np.load(REFERENCE) as reference:
        np.testing.assert_allclose(
            full["soil_water_content"][0, 0],
            reference["diagnostics.h2osoi_vol_soil"][0, 0],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            full["latent_heat_flux"][0, 0], reference["diagnostics.lhflx_canopy"][0, 0], rtol=0.01
        )
        assert full["latent_heat_flux"][0, 0] != reference["diagnostics.lhflx_canopy"][0, 0]

    # the water balance of every step, and of the stored water between steps (the soil
    # water written at a step is that of its beginning; no rain and no pond on this day)
    assert np.max(np.abs(full["soil_water_balance_error"])) < 1e-9
    stored = full["soil_liquid_water"][:, 0].sum(axis=1) + full["surface_water"][:, 0]
    net = full["infiltration"] - full["root_water_uptake"] - full["drainage"]
    np.testing.assert_allclose(np.diff(stored), net[:-1, 0] * 1800.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        full["infiltration"], -full["ground_evaporation"], rtol=1e-12, atol=1e-18
    )
    # the trees dry the soil: several mm over the day, taken through the roots
    assert -8.0 < stored[-1] - stored[0] < -2.0
    assert np.all(full["root_water_uptake"] > 0.0)
    assert np.all((full["soil_water_substeps"] >= 1) & (full["soil_water_substeps"] <= 64))


def test_clm_roots_and_no_redistribution(tmp_path):
    """One day with CLM's root conductances, CLM's fine root biomass from the leaf area and
    no redistribution, beside the default (CLM-ml's root resistances, the prescribed
    biomass, redistribution): the water balance closes either way and the roots take the
    transpiration."""
    default = _experiment(tmp_path / "default", "1 day").execute(quiet=True)["step"].data
    soil = {"root_conductance": "clm", "root_biomass": "from_leaf_area",
            "root_redistribution": False}  # fmt: skip
    clm = _experiment(tmp_path / "clm", "1 day", soil=soil).execute(quiet=True)["step"].data
    for data in (default, clm):
        assert np.max(np.abs(data["soil_water_balance_error"])) < 1e-9
        assert np.all(data["root_water_uptake"] > 0.0)
    # the same soil water at the first step, a different canopy response to it
    np.testing.assert_allclose(
        clm["soil_water_content"][0, 0], default["soil_water_content"][0, 0], rtol=1e-14
    )
    daily = [d["latent_heat_flux"][:, 0].mean() for d in (default, clm)]
    assert daily[0] != daily[1] and abs(daily[1] - daily[0]) < 0.2 * daily[0]
    # the dry top level: wetted by the roots with redistribution (the default, CLM-ml's
    # conductances); CLM's top level has no exchange with the roots and dries
    top = [
        d["soil_liquid_water"][-1, 0, 0] - d["soil_liquid_water"][0, 0, 0] for d in (default, clm)
    ]
    assert top[0] > 0.0 > top[1]


def test_resumed_from_a_restart_equals_the_uninterrupted_run(tmp_path, monkeypatch):
    from clm_ml import driver

    # a restart after every chunk: the one after the first day is kept as ``.prev``
    uninterrupted = _experiment(tmp_path / "full", "2 days")
    with monkeypatch.context() as patch:
        patch.setattr(driver, "restart_boundary", lambda start, end, every: True)
        full = uninterrupted.execute(quiet=True)["step"].data
    after_first_day = Path(str(uninterrupted.restart_path) + ".prev")
    assert after_first_day.exists()

    # rain reaches the ground on the second day, and the balance stays closed
    assert np.all(full["ground_water_input"][:48] == 0.0)
    rain = full["ground_water_input"][48:].sum() * 1800.0
    assert 2.0 < rain < 4.0  # mm
    assert np.max(np.abs(full["soil_water_balance_error"])) < 1e-9
    for name in VARIABLES:
        assert np.all(np.isfinite(full[name])), name

    # the resumed run completes the record of the first day's last step (still pending
    # when the restart was written), then gives the second day
    resumed = _experiment(tmp_path / "resumed", "2 days", restart=after_first_day)
    second = resumed.execute(quiet=True)["step"].data
    for name in VARIABLES:
        assert second[name].shape[0] == 49
        np.testing.assert_array_equal(second[name], full[name][-49:], err_msg=name)


def test_predicted_soil_water_needs_the_runoff_inputs(tmp_path):
    from clm_ml.io.yaml_schema import ConfigError

    experiment = _experiment(tmp_path, "1 day", surface={"slope": 1.0})
    with pytest.raises(ConfigError, match="needs max_saturated_fraction.*CHATS7"):
        experiment.execute(quiet=True, write=False)


def test_grid_run_with_predicted_soil_water(tmp_path):
    sys.path.insert(0, str(ROOT))
    from tests.experiment_tools import make_experiment
    from tests.regression.test_grid_setup import HIST, SURFACE, TRENDY

    if not (
        Path(SURFACE).exists() and Path(TRENDY).exists() and Path(f"{HIST}.2000-07.nc").exists()
    ):
        pytest.skip("1° global input files (NCAR GLADE) not available")
    from clm_ml.experiment import load_experiment

    def run(d):
        d["time"].update(start="2000-07-01T00:00", length="2 days", calendar="gregorian")
        d["forcing"]["gridded"]["co2"] = {"source": "constant", "ppm": 383}
        d["output"] = {
            "directory": str(tmp_path / "out"),
            "streams": [{"name": "step", "frequency": "step", "statistic": "instantaneous",
                         "level": ["patch"], "variables": VARIABLES}],
        }  # fmt: skip

    def model(d):
        d["numerics"].update(on_patch_error="stop", max_failed_patch_fraction=0.01)
        d["soil"]["water"] = "predicted"

    folder = make_experiment(
        tmp_path / "grid", compset="grid_1deg_2000", run=run, model=model,
        domain={"grid": {"surface_data": SURFACE, "land_mask": f"{HIST}.2000-01.nc",
                         "region": {"lat": [39.5, 40.5], "lon": [-90.0, -88.0]}}},
    )  # fmt: skip
    experiment = load_experiment(folder)
    columns = experiment.domain.columns
    assert experiment.domain.n_patches == 11
    assert np.all((columns.max_saturated_fraction > 0.3) & (columns.max_saturated_fraction < 0.5))
    assert np.all((columns.slope > 0.1) & (columns.slope < 0.3))  # degrees: flat country

    out = experiment.execute(write=False, quiet=True)["step"].data
    for name in VARIABLES:
        assert out[name].shape[:2] == (96, 11), name
        assert np.all(np.isfinite(out[name])), name
    assert np.max(np.abs(out["soil_water_balance_error"])) < 1e-9
    stored = out["soil_liquid_water"].sum(axis=2) + out["surface_water"]
    net = out["infiltration"] - out["root_water_uptake"] - out["drainage"]
    np.testing.assert_allclose(np.diff(stored, axis=0), net[:-1] * 1800.0, rtol=0, atol=1e-9)

    # rain on the second day: less reaches the ground under denser canopies, part of it
    # runs off the saturated fraction, the rest infiltrates
    rain = out["ground_water_input"].sum(axis=0) * 1800.0  # mm
    runoff = out["surface_runoff"].sum(axis=0) * 1800.0
    assert np.all((rain > 0.3) & (rain < 4.0)) and np.ptp(rain[:6]) > 0.1
    assert np.all((runoff > 0.05 * rain) & (runoff < 0.15 * rain))
    assert np.all((out["saturated_fraction"] > 0.05) & (out["saturated_fraction"] < 0.15))
    assert np.all((out["water_table_depth"] > 4.0) & (out["water_table_depth"] < 8.0))
    assert np.all(out["drainage"] > 0.0)
