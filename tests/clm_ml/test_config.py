"""model.yaml loading, validation, and mapping to the legacy option codes."""

import pytest

from clm_ml.config import ModelConfig
from clm_ml.io.yaml_schema import ConfigError

# Legacy MLclm_varctl defaults (CLM-ml v2 / CHATS), with met_type set by the forcing.
LEGACY_DEFAULTS = {
    "gs_type": 2, "gspot_type": 1, "gs_solver": 2, "colim_type": 1, "acclim_type": 1,
    "kn_val": -999.0, "turb_type": 1, "sparse_canopy_type": 1, "HF_extension_type": 2,
    "flux_profile_type": 1, "gb_type": 3, "light_type": 2, "leaf_optics_type": 0,
    "longwave_type": 1, "dtime_ml": 300.0, "runge_kutta_type": 41, "nrk": 4, "dz_tall": 0.5,
    "dz_short": 0.1, "dz_param": 2.0, "nlayer_above": 0, "nlayer_within": 0, "dpai_min": 0.01,
}  # fmt: skip


def test_defaults_match_legacy_module_defaults():
    codes = ModelConfig().to_fortran_codes("none")
    assert {k: codes[k] for k in LEGACY_DEFAULTS} == LEGACY_DEFAULTS
    assert codes["met_type"] == 0
    assert ModelConfig().to_fortran_codes("three_point")["met_type"] == 3


def test_empty_file_gives_defaults(tmp_path):
    (tmp_path / "model.yaml").write_text("")
    assert ModelConfig.from_yaml(tmp_path / "model.yaml") == ModelConfig()


def test_readable_options_map_to_codes():
    config = ModelConfig.from_dict(
        {"physics": {"stomatal_conductance": "medlyn", "solar_radiation": "norman"},
         "numerics": {"integrator": "euler"}}
    )  # fmt: skip
    codes = config.to_fortran_codes("three_point")
    assert (codes["gs_type"], codes["light_type"], codes["runge_kutta_type"], codes["nrk"]) == (
        0, 1, 10, 1,
    )  # fmt: skip
    assert config.non_default_options() == [
        "physics.stomatal_conductance", "physics.solar_radiation", "numerics.integrator",
    ]  # fmt: skip


@pytest.mark.parametrize(
    "data, message",
    [
        (
            {"physics": {"stomatal_conductance": "jarvis"}},
            "is not one of [medlyn, ball_berry, wue]",
        ),
        ({"physics": {"stomatal_conductanse": "wue"}}, "unknown key(s) ['stomatal_conductanse']"),
        ({"numerics": {"substep_seconds": "fast"}}, "expected a number"),
        ({"physics": {"sparse_canopy": 1}}, "expected true or false"),
        ({"soil": {"columns": "shared"}}, "option B"),
        ({"canopy_layers": {"n_layers_within": 10}}, "must be set together"),
    ],
)
def test_invalid_settings_name_the_key(data, message):
    with pytest.raises(ConfigError, match=None) as exc:
        ModelConfig.from_dict(data)
    assert message in str(exc.value)


def test_replace():
    config = ModelConfig().replace(**{"physics.stomatal_conductance": "ball_berry"})
    assert config.physics.stomatal_conductance == "ball_berry"
    with pytest.raises(ConfigError):
        ModelConfig().replace(**{"physics.stomatal_conductance": "nope"})


def test_yaml_round_trip(tmp_path):
    config = ModelConfig().replace(**{"numerics.integrator": "rk2_midpoint"})
    config.to_yaml(tmp_path / "model.yaml")
    assert ModelConfig.from_yaml(tmp_path / "model.yaml") == config
