"""
Phase 1 acceptance test: the YAML/``clm_ml`` route gives exactly the same results as
``clm-ml-offline``.

``nl.CHATS7.50steps`` is converted to run.yaml + model.yaml with ``clm-ml convert`` and run
with ``Run.execute()``. Every field recorded from the legacy driver must be bit-for-bit
identical to the Phase 0 reference (which was recorded from ``clm-ml-offline``), and the
named outputs must agree with the legacy fields they are built from.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tests.regression.record_run import REFERENCE_FILE, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE_FILE) as data:
        return {key: data[key] for key in data.files if not key.startswith("meta.")}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    import yaml

    from clm_ml.legacy.namelist import convert_namelist
    from clm_ml.run import load_run

    workdir = tmp_path_factory.mktemp("clm_ml_run")
    convert_namelist(SRC_DIR / "offline_executable" / "nl.CHATS7.50steps", workdir)
    run_yaml = workdir / "run.yaml"
    data = yaml.safe_load(run_yaml.read_text())
    data["output"] = {"directory": str(workdir / "out"), "format": ["netcdf", "legacy_ascii"]}
    run_yaml.write_text(yaml.safe_dump(data))

    run = load_run(run_yaml)
    return run, run.execute(), workdir / "out"


def test_recorded_fields_identical_to_clm_ml_offline(reference, result):
    _, outputs, _ = result
    assert set(reference) <= set(outputs.raw)
    different = [k for k in reference if not np.array_equal(outputs.raw[k], reference[k])]
    assert not different, f"fields differ from the clm-ml-offline reference: {different}"


def test_named_outputs_match_legacy_fields(reference, result):
    _, outputs, _ = result
    np.testing.assert_array_equal(
        outputs.latent_heat_flux, reference["mlcanopy.lhflx_canopy"].astype(float)
    )
    np.testing.assert_array_equal(outputs.gpp, reference["mlcanopy.gppveg_canopy"].astype(float))
    ncan = int(reference["mlcanopy.ncan_canopy"][0, 0])
    profile = outputs.air_temperature[0, 0]
    np.testing.assert_array_equal(
        profile[:ncan], reference["mlcanopy.tair_profile"][0, 0, 1 : ncan + 1]
    )
    assert np.isnan(profile[ncan:]).all()  # padding above the 46 real layers


def test_output_files_written(result):
    import xarray as xr

    run, outputs, out_dir = result
    legacy = sorted(p.name for p in out_dir.glob("*.out"))
    assert len(legacy) == 6, legacy
    with xr.open_dataset(out_dir / "CHATS7_2007-05-01_patch.nc") as ds:
        assert ds["latent_heat_flux"].attrs["units"] == "W m-2"
        assert ds.sizes == {
            "time": 50, "patch": 1, "layer": 50, "leaf_class": 2, "leaf_group": 3, "band": 2,
            "soil_layer": outputs.soil_temperature.shape[-1],
        }  # fmt: skip
        np.testing.assert_array_equal(ds["latent_heat_flux"].values, outputs.latent_heat_flux)
    with xr.open_dataset(out_dir / "CHATS7_2007-05-01_gridcell.nc") as ds:
        # One patch with weight 1: the gridcell mean equals the patch value.
        np.testing.assert_array_equal(ds["latent_heat_flux"].values, outputs.latent_heat_flux)
