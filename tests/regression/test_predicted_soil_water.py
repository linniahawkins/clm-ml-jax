"""
Predicted soil water end to end (model.yaml ``soil.water: predicted``) at CHATS.

- The run starts from the soil water a prescribed run has at its first step, so the first
  step's fluxes are those of the regression reference (to round-off).
- The soil water balance closes at every step, and the change of the stored water from one
  step to the next is the step's infiltration less root uptake and drainage.
- A run resumed from a restart written after the first day gives the same results, bit for
  bit, as the uninterrupted run: the soil water, the pond and the water table are in the
  restart. (Expected to fail for now: it rains on the second day, and the canopy's
  interception gives a wrong throughfall when the canopy has unused layer slots; the test
  passes when that is corrected.)
- Without the tower's largest saturated fraction and slope the run refuses to start.

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


def _experiment(workdir: Path, length: str, restart: Path | None = None, surface=SURFACE):
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
        model={"soil": {"water": "predicted"}},
        domain=lambda data: data["towers"]["CHATS7"]["soil"].update(surface),
    )
    return load_experiment(workdir)


def test_predicted_soil_water_one_day(tmp_path):
    full = _experiment(tmp_path / "full", "1 day").execute(quiet=True)["step"].data
    assert full["latent_heat_flux"].shape[0] == 48
    for name in VARIABLES:
        assert np.all(np.isfinite(full[name])), name

    # the first step is the prescribed run's: the same soil water (to round-off: the state
    # holds it in kg m-2, the forcing gives it by volume) and so the same fluxes
    with np.load(REFERENCE) as reference:
        np.testing.assert_allclose(
            full["soil_water_content"][0, 0],
            reference["diagnostics.h2osoi_vol_soil"][0, 0],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            full["latent_heat_flux"][0, 0], reference["diagnostics.lhflx_canopy"][0, 0], rtol=1e-9
        )

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


@pytest.mark.xfail(
    strict=True,
    reason="rain on the second day: canopy interception counts unused layer slots (O89)",
)
def test_resumed_from_a_restart_equals_the_uninterrupted_run(tmp_path):
    full = _experiment(tmp_path / "full", "2 days").execute(quiet=True)["step"].data
    first = _experiment(tmp_path / "first_day", "1 day")
    first.execute(quiet=True)
    resumed = _experiment(tmp_path / "resumed", "2 days", restart=first.restart_path)
    second = resumed.execute(quiet=True)["step"].data
    assert np.max(np.abs(full["soil_water_balance_error"])) < 1e-9
    for name in VARIABLES:
        np.testing.assert_array_equal(second[name], full[name][-48:], err_msg=name)


def test_predicted_soil_water_needs_the_runoff_inputs(tmp_path):
    from clm_ml.io.yaml_schema import ConfigError

    experiment = _experiment(tmp_path, "1 day", surface={"slope": 1.0})
    with pytest.raises(ConfigError, match="needs max_saturated_fraction.*CHATS7"):
        experiment.execute(quiet=True, write=False)
