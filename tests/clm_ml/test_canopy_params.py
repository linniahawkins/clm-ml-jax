"""CanopyParams (the parameters passed to the physics) matches clm_ml's parameter list."""

from clm_ml.params import PARAM_SPECS
from multilayer_canopy.MLCanopyParamsMod import LEGACY_FIELDS, CanopyParams

# Static (selects a code path; in GridInfo) or used only at initialization.
NOT_PASSED = {"c3_photosynthesis", "root_profile_beta"}


def test_fields_are_the_params_yaml_names():
    expected = [s.name for s in PARAM_SPECS if s.name not in NOT_PASSED]
    assert sorted(CanopyParams._fields) == sorted(expected)


def test_legacy_mapping_matches_param_specs():
    specs = {s.name: (s.legacy_table, s.legacy_field) for s in PARAM_SPECS}
    assert set(LEGACY_FIELDS) == set(CanopyParams._fields)
    for name, legacy in LEGACY_FIELDS.items():
        assert legacy == specs[name], name
