"""params.yaml loading, and exact agreement with the legacy PFT tables."""

import subprocess
import sys

import numpy as np
import pytest

from clm_ml.io.yaml_schema import ConfigError
from clm_ml.legacy.tables import fresh_legacy_pft_params, params_to_legacy
from clm_ml.params import PFT_NAMES, Params

from .conftest import SRC_DIR

WALNUT = "broadleaf_deciduous_temperate_tree"


def test_builtin_defaults_equal_legacy_tables():
    """clm5_defaults.yaml reproduces pftconMod/MLpftconMod with pftcon_val = 0."""
    legacy = fresh_legacy_pft_params(0)
    params = Params.from_yaml("clm5_defaults")
    for pft, entries in legacy.items():
        for name, value in entries.items():
            np.testing.assert_array_equal(params.get(pft, name), np.asarray(value, dtype=float))


def test_chats_preset_equals_legacy_pftcon_val_1():
    """chats.yaml reproduces the legacy tables with pftcon_val = 1."""
    legacy = fresh_legacy_pft_params(1)
    params = Params.from_yaml("chats")
    for pft, entries in legacy.items():
        for name, value in entries.items():
            np.testing.assert_array_equal(params.get(pft, name), np.asarray(value, dtype=float))


def test_chats_values_match_paper_table_2():
    """Bonan, Burns & Patton (2026), Table 2, CHATS column."""
    p = Params.from_yaml("chats")
    expected = {
        "vcmax25": 125.0, "iota": 375.0, "psi50_gs": -1.6, "shape_gs": 40.0,
        "plant_conductance": 7.0, "root_resistivity": 14.0, "leaf_angle_index": 0.53,
    }  # fmt: skip
    for name, value in expected.items():
        assert p.get(WALNUT, name) == value, name
    np.testing.assert_array_equal(p.get(WALNUT, "leaf_reflectance"), [0.06, 0.42])
    np.testing.assert_array_equal(p.get(WALNUT, "leaf_transmittance"), [0.04, 0.43])
    np.testing.assert_array_equal(p.get(WALNUT, "stem_reflectance"), [0.21, 0.49])


_DUMP = """
import sys, numpy as np
from clm_ml.legacy.tables import dump_legacy_pft_tables
np.savez(sys.argv[2], **dump_legacy_pft_tables(int(sys.argv[1])))
"""


def _fresh_full_tables(pftcon_val, path):
    subprocess.run([sys.executable, "-c", _DUMP, str(pftcon_val), str(path)],
                   cwd=SRC_DIR, check=True)  # fmt: skip
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def test_params_to_legacy_reproduces_full_legacy_tables(tmp_path):
    """Injecting chats.yaml into the pftcon_val=0 tables gives the pftcon_val=1 tables,
    including the untouched rows and unused columns (what the legacy adapter relies on)."""
    base = _fresh_full_tables(0, tmp_path / "t0.npz")
    target = _fresh_full_tables(1, tmp_path / "t1.npz")
    injected = params_to_legacy(Params.from_yaml("chats"), base)
    assert injected.keys() == target.keys()
    for key in target:
        np.testing.assert_array_equal(injected[key], target[key], err_msg=key)


def test_extends_and_override(tmp_path):
    (tmp_path / "mine.yaml").write_text(f"extends: chats\npft:\n  {WALNUT}:\n    vcmax25: 100.0\n")
    p = Params.from_yaml(tmp_path / "mine.yaml")
    assert p.get(WALNUT, "vcmax25") == 100.0
    assert p.get(WALNUT, "iota") == 375.0  # from chats
    assert p.get("c4_grass", "g0_ball_berry") == 0.04  # from clm5_defaults
    assert np.isnan(p.values["vcmax25"][PFT_NAMES.index("not_vegetated")])


@pytest.mark.parametrize(
    "body, message",
    [
        (f"pft:\n  {WALNUT}:\n    vcmax: 1.0\n", "unknown parameter"),
        ("pft:\n  walnut:\n    vcmax25: 1.0\n", "unknown PFT 'walnut'"),
        (f"pft:\n  {WALNUT}:\n    leaf_reflectance: 0.1\n", "expected [vis, nir]"),
        ("extends: mine.yaml\n", "circular"),
        ("pfts: {}\n", "unknown key(s) ['pfts']"),
    ],
)
def test_invalid_params(tmp_path, body, message):
    (tmp_path / "mine.yaml").write_text(body)
    with pytest.raises(ConfigError) as exc:
        Params.from_yaml(tmp_path / "mine.yaml")
    assert message in str(exc.value)


def test_set_returns_copy():
    p = Params.from_yaml("chats")
    q = p.set(WALNUT, vcmax25=90.0)
    assert (p.get(WALNUT, "vcmax25"), q.get(WALNUT, "vcmax25")) == (125.0, 90.0)
