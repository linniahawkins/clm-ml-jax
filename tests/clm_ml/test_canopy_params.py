"""CanopyParams (the parameters passed to the physics) matches clm_ml's parameter list."""

from clm_ml.params import HYDROLOGY_NAMES, PARAM_SPECS, CanopyParams, Params

# Static (selects a code path; in GridInfo) or used only at initialization (the roughness
# ratios: the forcing height of grid runs).
NOT_PASSED = {"c3_photosynthesis", "root_profile_beta", "z0m_to_height_ratio",
              "displacement_to_height_ratio"}  # fmt: skip


def test_fields_are_the_params_yaml_names():
    expected = [s.name for s in PARAM_SPECS if s.name not in NOT_PASSED] + ["hydrology"]
    assert sorted(CanopyParams._fields) == sorted(expected)


def test_every_pft_row_carries_the_soil_hydrology_parameters():
    """In the order of the processes' ``HydrologyParameters``, with params.yaml's values."""
    params = Params.from_yaml("chats").set_hydrology(fff=0.7)
    hydrology = params.canopy_params("c4_grass").hydrology
    assert hydrology._fields == HYDROLOGY_NAMES
    assert float(hydrology.fff) == 0.7 and float(hydrology.e_ice) == 6.0
