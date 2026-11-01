"""CanopyParams (the parameters passed to the physics) matches clm_ml's parameter list."""

from clm_ml.params import PARAM_SPECS, CanopyParams

# Static (selects a code path; in GridInfo) or used only at initialization (the roughness
# ratios: the forcing height of grid runs).
NOT_PASSED = {"c3_photosynthesis", "root_profile_beta", "z0m_to_height_ratio",
              "displacement_to_height_ratio"}  # fmt: skip


def test_fields_are_the_params_yaml_names():
    expected = [s.name for s in PARAM_SPECS if s.name not in NOT_PASSED]
    assert sorted(CanopyParams._fields) == sorted(expected)
