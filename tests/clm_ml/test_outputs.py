"""Named outputs and netCDF writing, built from the Phase 0 reference fields (no model run)."""

import numpy as np
import pytest

from clm_ml.outputs import VARIABLES, Outputs
from clm_ml.run import load_run

from .conftest import PROJECT_ROOT

REFERENCE = PROJECT_ROOT / "tests" / "regression" / "reference" / "CHATS7_50steps.npz"


@pytest.fixture
def outputs(chats_run_dir):
    with np.load(REFERENCE) as data:
        raw = {k: data[k] for k in data.files if not k.startswith("meta.")}
    n_steps = raw["mlcanopy.ncan_canopy"].shape[0]
    # Fields the worker adds that the Phase 0 reference does not record:
    raw["derived.latvap_tref"] = np.full((n_steps, 1), 44000.0)  # J/mol, roughly
    raw["static.nlevgrnd"] = np.asarray(25)
    raw["static.soil_depth"] = np.linspace(0, 10, 37)[None, :]
    return Outputs.from_legacy(raw, load_run(chats_run_dir / "run.yaml"))


def test_every_variable_has_expected_shape(outputs):
    sizes = {"layer": 50, "leaf_class": 2, "leaf_group": 3, "band": 2}
    for spec in VARIABLES:
        expected = (50, 1) + tuple(sizes[d] for d in spec.dims)
        assert outputs.data[spec.name].shape == expected, spec.name


def test_profiles_are_padded_with_nan(outputs):
    t = outputs.air_temperature[10, 0]
    assert np.isfinite(t[:46]).all() and np.isnan(t[46:]).all()
    leaf_t = outputs.leaf_temperature[10, 0]  # leaves only within the canopy (20 layers)
    assert np.isnan(leaf_t[20:]).all()


def test_gridcell_mean_with_one_patch_equals_patch(outputs):
    means = outputs.gridcell_mean()
    np.testing.assert_array_equal(means["latent_heat_flux"], outputs.latent_heat_flux)
    assert "air_temperature" not in means  # profiles are per patch only


def test_netcdf_round_trip(outputs, tmp_path):
    import xarray as xr

    written = outputs.to_netcdf(tmp_path / "run.nc")
    assert [p.name for p in written] == ["run_patch.nc", "run_gridcell.nc"]
    with xr.open_dataset(written[0]) as ds:
        assert ds["gpp"].attrs["units"] == "umol CO2 m-2 s-1"
        assert ds["pft"].values.tolist() == ["broadleaf_deciduous_temperate_tree"]
        np.testing.assert_array_equal(ds["latent_heat_flux"].values, outputs.latent_heat_flux)
        assert str(ds["time"].values[0]).startswith("2007-05-01T00:30")


def test_select(outputs):
    small = outputs.select(["gpp", "latent_heat_flux"])
    assert set(small.data) == {"gpp", "latent_heat_flux"}
    with pytest.raises(KeyError, match="unknown output variable"):
        outputs.select(["nope"])
