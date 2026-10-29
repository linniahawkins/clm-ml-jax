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


# --- clm-ml params: CLM parameter files and flattened params files (Phase 6b) -------------


def _fake_clm_parameter_file(path, n_pft=20):
    """A small CLM-style parameter file: every converted variable, value = 0.01 * row + k."""
    import netCDF4 as nc

    from clm_ml.params_convert import CLM_PARAMETER_FILE_FIELDS

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("pft", n_pft)
        ds.createDimension("string_length", 40)
        ds.createDimension("variants", 2)
        names = list(PFT_NAMES) + [f"crop_{i}" for i in range(n_pft - len(PFT_NAMES))]
        v = ds.createVariable("pftname", "S1", ("pft", "string_length"))
        v[:] = np.array([[c.encode() for c in n.ljust(40)] for n in names], dtype="S1")
        variables = sorted({x for vs, _ in CLM_PARAMETER_FILE_FIELDS.values() for x in vs})
        for k, name in enumerate(variables):
            row = 0.01 * np.arange(n_pft) + k
            if name == "rootprof_beta":
                ds.createVariable(name, "f8", ("variants", "pft"))[:] = np.stack([row, row + 0.5])
            elif name == "c3psn":
                ds.createVariable(name, "f8", ("pft",))[:] = (np.arange(n_pft) != 14).astype(float)
            else:
                ds.createVariable(name, "f8", ("pft",))[:] = row
    return variables


def test_params_from_a_clm_parameter_file(tmp_path):
    """The converted table: CLM-file values for what CLM defines (units converted, the water
    root-profile variant), the base's for the CLM-ml-specific parameters; complete."""
    from clm_ml.params import PARAM_SPECS
    from clm_ml.params_convert import write_params_yaml

    variables = _fake_clm_parameter_file(tmp_path / "clm.nc")
    out = write_params_yaml(str(tmp_path / "clm.nc"), tmp_path / "p.yaml")
    p, base = Params.from_yaml(out), Params.from_yaml("clm5_defaults")
    k = {name: i for i, name in enumerate(variables)}
    row = PFT_NAMES.index(WALNUT)
    assert p.get(WALNUT, "leaf_dimension") == 0.01 * row + k["dleaf"]
    assert p.get(WALNUT, "g0_medlyn") == (0.01 * row + k["medlynintercept"]) * 1e-6
    assert p.get(WALNUT, "root_profile_beta") == 0.01 * row + k["rootprof_beta"]  # variant 1
    np.testing.assert_array_equal(
        p.get(WALNUT, "leaf_reflectance"), [0.01 * row + k["rholvis"], 0.01 * row + k["rholnir"]]
    )
    assert p.get(WALNUT, "z0m_to_height_ratio") == 0.01 * row + k["z0mr"]
    assert p.get("c4_grass", "c3_photosynthesis") == 0.0
    for name in ("vcmax25", "iota", "plant_conductance", "gs_min"):  # CLM-ml specific
        assert p.get(WALNUT, name) == base.get(WALNUT, name)
    every = [s.name for s in PARAM_SPECS]
    assert not p.missing_for(list(range(1, len(PFT_NAMES))), every)
    text = out.read_text()
    assert "extends" not in text and "clm.nc" in text  # standalone, source recorded


def test_flattened_params_file_equals_its_extends_chain(tmp_path):
    from clm_ml.params_convert import write_params_yaml

    out = write_params_yaml("chats", tmp_path / "chats_full.yaml")
    a, b = Params.from_yaml("chats"), Params.from_yaml(out)
    for name in a.values:
        np.testing.assert_array_equal(a.values[name], b.values[name])


def test_clm_parameter_files_are_not_read_by_params_yaml(tmp_path):
    (tmp_path / "p.yaml").write_text("extends: clm5_defaults\nclm_parameters: x.nc\n")
    with pytest.raises(ConfigError, match="clm-ml params"):
        Params.from_yaml(tmp_path / "p.yaml")


def test_builtin_clm6_is_complete():
    """data/params/clm6.yaml (the CTSM 6.0 file converted): every parameter of every PFT,
    including the roughness ratios grid runs need."""
    from clm_ml.params import PARAM_SPECS

    p = Params.from_yaml("clm6")
    assert not p.missing_for(list(range(1, len(PFT_NAMES))), [s.name for s in PARAM_SPECS])
    assert p.get("needleleaf_evergreen_temperate_tree", "z0m_to_height_ratio") == 0.055
    assert p.get("needleleaf_evergreen_temperate_tree", "leaf_dimension") == 0.000648
