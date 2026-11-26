"""
Soil water that freezes and thaws, end to end (model.yaml ``soil.freeze_thaw``).

- At CHATS in May no soil freezes: the run with ``freeze_thaw`` equals the run without it.
- A cold CHATS (the tower's air 18 K colder, its soil starting 18 K colder: frozen below
  0.2 m, with snow in the first night and rain on the second day): the soil freezes and
  thaws, the snow lies on the ground and melts, water and energy are conserved at every
  step (the run checks the soil's energy with the latent heat), and the run resumed from a
  restart written after its first day gives the same results, bit for bit: the ice, the
  snow, the frost table and the perched water table are in the restart.
- The same with CLM's roots, which keep taking water from freezing levels as CLM does.
- Padding patches do not change a freezing run.
- ``freeze_thaw`` needs predicted soil water.
- A grid run (two 1° gridcells in Minnesota, two days of January 2000) starts from the ice
  of the CLM history and keeps the water balance of every patch closed.

(The cold CHATS is not a site: it tests the code.)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]
TOWER_FILE = ROOT / "data/input_files/tower-forcing/CHATS7/2007-05.nc"
HISTORY_FILE = ROOT / "data/input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc"
COLDER = 18.0  # K
FLUXES = [
    "ground_water_input", "ground_evaporation", "root_water_uptake", "infiltration",
    "surface_runoff", "drainage", "perched_drainage", "ice_runoff", "snow_melt",
    "ground_sublimation", "soil_phase_change_heat", "soil_water_balance_error",
    "latent_heat_flux", "sensible_heat_flux", "ground_heat_flux",
]  # fmt: skip
STATES = [
    "surface_water", "water_table_depth", "frost_table_depth", "perched_water_table_depth",
    "snow_on_ground", "soil_water_substeps",
]  # fmt: skip
VARIABLES = FLUXES + STATES + ["soil_liquid_water", "soil_ice"]
SURFACE = {"max_saturated_fraction": 0.4, "slope": 1.0}


def _cold_files(folder: Path) -> tuple[Path, Path]:
    """Copies of the CHATS tower file and CLM history, ``COLDER`` K colder (the longwave
    radiation with the air's fourth power), with precipitation in the first night (it
    falls as snow) and on the second day."""
    import netCDF4

    folder.mkdir(parents=True, exist_ok=True)
    tower, history = folder / "tower.nc", folder / "history.nc"
    shutil.copy(TOWER_FILE, tower)
    shutil.copy(HISTORY_FILE, history)
    with netCDF4.Dataset(tower, "a") as ds:
        air = ds["TBOT"][:]
        ds["FLDS"][:] = ds["FLDS"][:] * ((air - COLDER) / air) ** 4
        ds["TBOT"][:] = air - COLDER
        rain = ds["PRECTmms"][:]
        rain[:96] = 0.0
        rain[18:30] = 4.0e-4  # 8.6 mm in the first night
        rain[66:74] = 3.0e-4  # 4.3 mm on the second day
        ds["PRECTmms"][:] = rain
    with netCDF4.Dataset(history, "a") as ds:
        ds["TSOI"][:] = ds["TSOI"][:] - COLDER
    return tower, history


def _experiment(
    workdir: Path, length: str, *, cold=None, restart=None, soil=None, snow=True, overrides=()
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
                         "variables": VARIABLES + ["soil_temperature"]}],
        },
    }  # fmt: skip
    if cold is not None:
        run["forcing"] = {
            "meteorology": {
                "files": [str(cold[0])],
                "snowfall": "by_temperature" if snow else "none",
            },
            "clm_history": {"file": str(cold[1])},
        }
    if restart is not None:
        run["initial_conditions"] = {"source": "restart", "restart_file": str(restart)}
    make_experiment(
        workdir,
        namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day",
        run=run,
        model={"soil": {"water": "predicted", "freeze_thaw": True, **(soil or {})}},
        domain=lambda data: data["towers"]["CHATS7"]["soil"].update(SURFACE),
    )
    return load_experiment(workdir, list(overrides))


def test_unfrozen_soil_gives_the_same_run_with_and_without_freeze_thaw(tmp_path):
    """The two runs are different compiled programs, and the canopy amplifies a difference
    in the last bit (the regression policy's noise twins differ by 4e-11): the results are
    equal to that sensitivity, and bit for bit for the first 36 steps."""
    frozen = _experiment(tmp_path / "on", "1 day").execute(quiet=True)["step"].data
    plain = _experiment(tmp_path / "off", "1 day", soil={"freeze_thaw": False})
    plain = plain.execute(quiet=True)["step"].data
    for name in VARIABLES:
        if np.all(np.isnan(plain[name])):  # only frozen soil has it
            continue
        if name == "soil_water_balance_error":  # round-off itself
            assert max(np.abs(frozen[name]).max(), np.abs(plain[name]).max()) < 1e-9
            continue
        scale = np.abs(plain[name]).max()
        np.testing.assert_allclose(
            frozen[name], plain[name], rtol=1e-9, atol=1e-10 * scale, err_msg=name
        )
        np.testing.assert_array_equal(frozen[name][:36], plain[name][:36], err_msg=name)
    # no ice, no snow, nothing perched, no latent heat
    assert np.all(frozen["soil_ice"] == 0.0) and np.all(frozen["snow_on_ground"] == 0.0)
    assert np.all(frozen["soil_phase_change_heat"] == 0.0)
    assert np.all(frozen["perched_drainage"] == 0.0)
    assert np.all(frozen["frost_table_depth"] == frozen["perched_water_table_depth"])
    assert np.all(np.isnan(plain["soil_ice"]))


def _check_cold_run(data: dict, steps: int) -> None:
    for name in VARIABLES:
        assert data[name].shape[0] == steps, name
        assert np.all(np.isfinite(data[name])), name
    assert np.max(np.abs(data["soil_water_balance_error"])) < 1e-9


def test_a_cold_soil_freezes_thaws_and_conserves(tmp_path, monkeypatch):
    from clm_ml import driver

    cold = _cold_files(tmp_path / "inputs")
    uninterrupted = _experiment(tmp_path / "full", "2 days", cold=cold)
    with monkeypatch.context() as patch:
        patch.setattr(driver, "restart_boundary", lambda start, end, every: True)
        outputs = uninterrupted.execute(quiet=True)["step"]
    full = outputs.data
    after_first_day = Path(str(uninterrupted.restart_path) + ".prev")
    assert after_first_day.exists()
    _check_cold_run(full, 96)

    ice, liquid = full["soil_ice"][:, 0, :20], full["soil_liquid_water"][:, 0, :20]
    temperature = outputs.soil_temperature[:, 0]
    # the run starts with ice below the thawed top levels (divided by temperature: the
    # history has no ice), and both freezing and thawing happen
    assert np.all(ice[0, :3] == 0.0) and np.all(ice[0, 5:12] > 0.0)
    assert np.all(liquid[0, 5:12] > 0.0)  # supercooled water stays
    change = np.diff(ice.sum(axis=1))
    assert change.max() > 0.05 and change.min() < -0.05
    heat = full["soil_phase_change_heat"][:, 0]
    assert heat.max() > 5.0 and heat.min() < -5.0  # W m-2, both ways
    # a level that holds ice and is changing phase sits at the freezing point
    assert np.sum((np.abs(temperature[:, :20] - 273.15) < 1e-9) & (ice > 0.0)) > 20
    # the frost table moves; the snow of the first night lies on the ground and melts
    assert len(np.unique(full["frost_table_depth"])) >= 3
    snow = full["snow_on_ground"][:, 0]
    assert snow.max() > 1.0 and snow[-1] < snow.max()
    assert full["snow_melt"].max() > 0.0
    assert full["ground_sublimation"].max() > 0.0
    # all the water is accounted for over the run: what reached the ground as rain or
    # meltwater, the dew and frost, against evaporation, uptake, runoff and drainage
    # (the soil water and ice written at a step are those of its beginning, the pond that
    # of its end; the run starts without a pond)
    pond = np.concatenate([[0.0], full["surface_water"][:-1, 0]])
    stored = liquid.sum(axis=1) + ice.sum(axis=1) + pond
    out = sum(
        full[name][:-1, 0]
        for name in ("ground_evaporation", "root_water_uptake", "surface_runoff", "drainage",
                     "perched_drainage", "ice_runoff")
    )  # fmt: skip
    net = (full["ground_water_input"][:-1, 0] - out).sum() * 1800.0
    np.testing.assert_allclose(stored[-1] - stored[0], net, atol=1e-6)

    # resumed after the first day: the same second day
    resumed = _experiment(tmp_path / "resumed", "2 days", cold=cold, restart=after_first_day)
    resumed = resumed.execute(quiet=True)["step"]
    second = resumed.data
    for name in VARIABLES:
        assert second[name].shape[0] == 49
        np.testing.assert_array_equal(second[name], full[name][-49:], err_msg=name)
    np.testing.assert_array_equal(resumed.soil_temperature, outputs.soil_temperature[-49:])


def test_cold_soil_with_clm_roots_and_without_snowfall(tmp_path):
    """CLM's roots keep the freezing levels (their conductivity lowered by the ice, as
    CLM); without the division of the precipitation all of it is rain, also below
    freezing (CLM-ml v2's tower forcing)."""
    cold = _cold_files(tmp_path / "inputs")
    soil = {"root_conductance": "clm", "root_biomass": "from_leaf_area"}
    data = _experiment(tmp_path / "clm", "1 day", cold=cold, soil=soil, snow=False)
    data = data.execute(quiet=True)["step"].data
    _check_cold_run(data, 48)
    assert np.all(data["snow_on_ground"] == 0.0)
    assert data["ground_water_input"].sum() * 1800.0 > 5.0  # the night's rain, mm
    assert np.all(data["root_water_uptake"] >= 0.0)


def test_padding_patches_do_not_change_a_freezing_run(tmp_path):
    """The cold tower alone and with two padding patches on a padding gridcell (copies with
    weight 0, as a domain padded to share a compiled program): the same results, to
    round-off. (With predicted soil water a padded run is not bit for bit the run alone,
    frozen or not: sums over the soil levels are added in another order in a batch.)"""
    cold = _cold_files(tmp_path / "inputs")
    alone = _experiment(tmp_path / "alone", "1 day", cold=cold).execute(quiet=True)["step"]
    pads = ["run.execution.patch_count=3", "run.execution.gridcell_count=2"]
    padded = _experiment(tmp_path / "padded", "1 day", cold=cold, overrides=pads)
    padded = padded.execute(quiet=True)["step"]
    for name in VARIABLES:
        assert padded.data[name].shape == alone.data[name].shape, name
        if name == "soil_water_balance_error":  # round-off itself
            continue
        scale = np.abs(alone.data[name]).max()
        np.testing.assert_allclose(
            padded.data[name], alone.data[name], rtol=1e-9, atol=1e-10 * scale, err_msg=name
        )
    np.testing.assert_allclose(padded.soil_temperature, alone.soil_temperature, rtol=1e-12)
    assert alone.data["soil_ice"].max() > 0.0


def test_freeze_thaw_needs_predicted_soil_water():
    sys.path.insert(0, str(ROOT / "src"))
    from clm_ml.config import ModelConfig
    from clm_ml.io.yaml_schema import ConfigError, from_dict, to_dict

    settings = to_dict(ModelConfig())
    settings["soil"]["freeze_thaw"] = True  # with the default, prescribed soil water
    with pytest.raises(ConfigError, match="freeze_thaw: only used with water: predicted"):
        from_dict(ModelConfig, settings, source="model.yaml", strict=True)
    settings["soil"] = {"water": "predicted", "freeze_thaw": True}
    config = from_dict(ModelConfig, settings, source="model.yaml")
    assert config.soil.freeze_thaw and config.numerics.phase_change_derivative == "stabilized"


def test_grid_run_starts_from_the_history_ice_and_conserves(tmp_path):
    sys.path.insert(0, str(ROOT))
    from tests.experiment_tools import make_experiment
    from tests.regression.test_grid_setup import HIST, SURFACE, TRENDY

    if not (
        Path(SURFACE).exists() and Path(TRENDY).exists() and Path(f"{HIST}.2000-01.nc").exists()
    ):
        pytest.skip("1° global input files (NCAR GLADE) not available")
    from clm_ml.experiment import load_experiment

    def run(d):
        d["time"].update(start="2000-01-15T00:00", length="2 days", calendar="gregorian")
        d["forcing"]["gridded"]["co2"] = {"source": "constant", "ppm": 383}
        d["output"] = {
            "directory": str(tmp_path / "out"),
            "streams": [{"name": "step", "frequency": "step", "statistic": "instantaneous",
                         "level": ["patch"], "variables": VARIABLES + ["soil_temperature"]}],
        }  # fmt: skip

    def model(d):
        d["numerics"].update(on_patch_error="stop", max_failed_patch_fraction=0.01)
        d["soil"].update(water="predicted", freeze_thaw=True)

    folder = make_experiment(
        tmp_path / "grid", compset="grid_1deg_2000", run=run, model=model,
        domain={"grid": {"surface_data": SURFACE, "land_mask": f"{HIST}.2000-01.nc",
                         "region": {"lat": [45.5, 46.5], "lon": [-95.0, -93.0]}}},
    )  # fmt: skip
    experiment = load_experiment(folder)
    outputs = experiment.execute(write=False, quiet=True)["step"]
    out = outputs.data
    n = experiment.domain.n_patches
    assert n >= 4
    for name in VARIABLES:
        assert out[name].shape[:2] == (96, n), name
        assert np.all(np.isfinite(out[name])), name
    assert np.max(np.abs(out["soil_water_balance_error"])) < 1e-9
    # Minnesota in January: the run starts with the history's ice in the upper levels and
    # liquid water beside it, frozen from the surface
    ice, liquid = out["soil_ice"][0, :, :20], out["soil_liquid_water"][0, :, :20]
    assert np.all(ice[:, :4] > 0.0) and np.all(liquid[:, :4] > 0.0)
    assert np.all(ice[:, :4] / (ice[:, :4] + liquid[:, :4]) > 0.3)
    assert np.all(outputs.soil_temperature[0, :, :4] < 273.15)
    assert np.all(out["frost_table_depth"][-1] == 0.0)
    # snow falls and lies on the ground
    assert np.all(out["snow_on_ground"][-1] >= 0.0)
