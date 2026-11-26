"""The inputs of a run with soil that freezes: precipitation divided into rain and snow at
a tower, and the share of ice in the soil water of a CLM history file."""

import numpy as np

from clm_ml.chunk_inputs import rain_and_snow
from clm_ml.io import netcdf
from clm_ml.setup.soil import ice_volume_fraction, initial_ice_fraction


def test_precipitation_is_snow_below_freezing_and_rain_above_two_degrees():
    air = np.array([250.0, 273.15, 273.65, 274.15, 275.15, 290.0])
    rain, snow = rain_and_snow(np.full(6, 2.0e-4), air)
    np.testing.assert_allclose(rain / 2.0e-4, [0.0, 0.0, 0.25, 0.5, 1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(rain + snow, 2.0e-4, rtol=1e-15)
    assert np.all(snow >= 0.0)


def test_ice_share_of_the_water_volume():
    # equal masses of liquid water and ice: the ice takes more room
    share = ice_volume_fraction([10.0, 10.0, 0.0, 5.0], [10.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(share, [(10 / 917) / (10 / 1000 + 10 / 917), 0.0, 0.0, 0.0])


def test_initial_ice_fraction_of_a_history_file(tmp_path):
    def history(path, with_ice):
        with netcdf.dataset(path, "w") as ds:
            ds.createDimension("time", 3)
            ds.createDimension("levsoi", 20)
            ds.createDimension("lndgrid", 1)
            if with_ice:
                liq = ds.createVariable("SOILLIQ", "f8", ("time", "levsoi", "lndgrid"))
                ice = ds.createVariable("SOILICE", "f8", ("time", "levsoi", "lndgrid"))
                liq[:] = 20.0
                ice[:] = 0.0
                ice[1, :5, 0] = 20.0  # the second record: ice in the upper five levels
        return path

    share = initial_ice_fraction(history(tmp_path / "ice.nc", True), 2, 20)
    np.testing.assert_allclose(share[:5], (20 / 917) / (20 / 1000 + 20 / 917))
    assert np.all(share[5:] == 0.0)
    assert np.all(initial_ice_fraction(tmp_path / "ice.nc", 1, 20) == 0.0)
    # a history without them: not known (the run divides the water by temperature)
    assert np.all(np.isnan(initial_ice_fraction(history(tmp_path / "none.nc", False), 2, 20)))
