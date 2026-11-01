"""Experiment folders (decision 91): loading and validation, strictness, overrides, presets
(`clm-ml new`), `clm-ml upgrade`, namelist conversion."""

import datetime as dt

import numpy as np
import pytest
import yaml

from clm_ml.experiment import load_experiment, new_experiment, upgrade_experiment
from clm_ml.io.namelist import convert_namelist
from clm_ml.io.yaml_schema import ConfigError

from .conftest import CHATS_NAMELISTS


def _edit(folder, name, change):
    path = folder / name
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return folder


def test_converted_50_steps(chats_run_dir):
    exp = load_experiment(chats_run_dir)
    assert exp.start == dt.datetime(2007, 5, 1)
    assert exp.n_steps == 50
    assert exp.domain.gridcells.names == ("CHATS7",)
    assert exp.run is exp.run_file and exp.model is exp.config
    assert "canopy layers 46" in exp.summary()


def test_converted_folder_has_the_four_complete_files(chats_run_dir):
    for name in ("run.yaml", "domain.yaml", "model.yaml", "params.yaml"):
        header = " ".join(line[2:] for line in (chats_run_dir / name).read_text().splitlines()
                          if line.startswith("# "))  # fmt: skip
        assert "Converted from nl.CHATS7.50steps" in header
    assert "REQUIRED" in (chats_run_dir / "run.yaml").read_text()


def test_run_reads_the_files_and_times_of_the_original_namelist(chats_run_dir):
    import f90nml

    from clm_ml.setup.towers import gridcell_sources

    exp = load_experiment(chats_run_dir)
    original = f90nml.read(str(CHATS_NAMELISTS / "nl.CHATS7.50steps"))["clmml_inparm"]
    assert int(f"{exp.start:%Y%m%d}") == original["start_ymd"]
    assert exp.n_steps == original["stop_n"]
    assert original["clm_phys"] == "CLM5_0"  # the only soil layer structure (C35)
    (source,) = gridcell_sources(exp)
    assert source.tower_file.paths[0].endswith(original["fin_tower"].replace("../", ""))
    assert source.clm_history_file.endswith(original["fin_clm"].replace("../", ""))
    assert source.soil_adjust_file.endswith(original["fin_soil_adjust"].replace("../", ""))
    assert source.history_index(0) == 1441  # 1 May 00:30 in the April-May history file


def test_convert_refuses_clm4_5_soil(tmp_path):
    text = (CHATS_NAMELISTS / "nl.CHATS7.1day").read_text()
    assert "'CLM5_0'" in text
    namelist = tmp_path / "nl.clm45"
    namelist.write_text(text.replace("'CLM5_0'", "'CLM4_5'"))
    with pytest.raises(ConfigError, match="clm_phys 'CLM4_5' is not supported"):
        convert_namelist(namelist, tmp_path / "out")


@pytest.mark.parametrize("name", ["nl.CHATS7.1day", "nl.CHATS7.05.2007", "nl.CHATS7.50steps"])
def test_all_chats_namelists_convert_and_load(tmp_path, name):
    load_experiment(convert_namelist(CHATS_NAMELISTS / name, tmp_path))


def test_days_and_end_are_equivalent(chats_run_dir):
    def by_days(d):
        d["time"].update(steps=None, days=1)

    def by_end(d):
        d["time"].update(steps=None, days=None, end="2007-05-02T00:00")

    assert load_experiment(_edit(chats_run_dir, "run.yaml", by_days)).n_steps == 48
    assert load_experiment(_edit(chats_run_dir, "run.yaml", by_end)).n_steps == 48


def test_overrides_name_the_file(chats_run_dir):
    walnut = "broadleaf_deciduous_temperate_tree"
    exp = load_experiment(chats_run_dir, ["run.time.steps=10",
                                          "model.physics.stomatal_conductance=medlyn",
                                          f"params.{walnut}.vcmax25=100"])  # fmt: skip
    assert exp.n_steps == 10
    assert exp.config.physics.stomatal_conductance == "medlyn"
    assert exp.params.get(walnut, "vcmax25") == 100.0
    assert "not been validated" in exp.summary()
    with pytest.raises(ConfigError, match="name the file first"):
        load_experiment(chats_run_dir, ["time.steps=10"])


@pytest.mark.parametrize(
    "name, change, message",
    [  # fmt: skip
        ("run.yaml", lambda d: d["time"].update(days=1), "exactly one of end, days, steps"),
        ("run.yaml", lambda d: d["forcing"]["meteorology"].update(interval_seconds=1000),
         "must divide the forcing interval"),
        ("run.yaml", lambda d: d["initial_conditions"].update(source="restart"),
         "restart_file: required"),
        ("run.yaml", lambda d: d["execution"].update(chunk="2 months"), "expected e.g. '10 days'"),
        ("run.yaml", lambda d: d["execution"].update(chunk="5 hours"), "expected e.g. '10 days'"),
        ("run.yaml", lambda d: d["execution"].update(restart_every="weekly"), "expected e.g."),
        ("run.yaml", lambda d: d.update(colour="blue"), "unknown key(s) ['colour']"),
        ("run.yaml", lambda d: d["time"].update(start="May 1"), "is not a date-time"),
        ("run.yaml", lambda d: d["forcing"]["meteorology"].update(files="nope.nc"),
         "file not found"),
        ("run.yaml", lambda d: d["forcing"]["meteorology"].update(files="met_{year}-{month:02d}.nc"),
         "file not found for 2007-05"),
        ("run.yaml", lambda d: d["forcing"]["meteorology"].update(shortwave="cloudy"), "cloudy"),
        ("run.yaml", lambda d: d["forcing"]["vegetation"].update(source="file"),
         "forcing.vegetation.files: required"),
        ("run.yaml", lambda d: d["forcing"]["soil_water"].update(source="file", files="nope.nc"),
         "file not found"),
        ("run.yaml", lambda d: d["forcing"]["vegetation"].update(source="monthly"), "monthly"),
        ("run.yaml", lambda d: d["time"].update(calendar="noleap"), "only for grid runs"),
        ("run.yaml", lambda d: d["time"].update(calendar="julian"), "calendar"),
        # strict: complete files, keys in their own file, sections for this kind of run
        ("run.yaml", lambda d: d["time"].pop("calendar"), "missing key(s) time.calendar"),
        ("run.yaml", lambda d: d["time"].pop("calendar"), "clm-ml upgrade"),
        ("model.yaml", lambda d: d["numerics"].pop("integrator"),
         "missing key(s) numerics.integrator"),
        ("run.yaml", lambda d: d.update(params="chats"), "unknown key(s) ['params']"),
        ("run.yaml", lambda d: d["forcing"].update(gridded={}), "only for grid runs"),
        ("domain.yaml", lambda d: d.update(type="grid"), "only for towers runs"),
        ("domain.yaml", lambda d: d.update(type="island"), "not one of [towers, grid]"),
        ("domain.yaml", lambda d: d["towers"]["CHATS7"]["patches"][0].update(weight=0.5),
         "weights sum to 0.5"),
        ("params.yaml", lambda d: d["pft"].pop("c4_grass"), "missing: c4_grass"),
        ("params.yaml", lambda d: d.update(extends="chats"), "'extends' is not supported"),
    ]
    ,
)
def test_invalid_folders(chats_run_dir, name, change, message):
    with pytest.raises(ConfigError) as exc:
        load_experiment(_edit(chats_run_dir, name, change))
    assert message in str(exc.value)


def test_a_missing_file_or_a_file_given_is_explained(chats_run_dir, tmp_path):
    with pytest.raises(ConfigError, match="give the experiment folder"):
        load_experiment(chats_run_dir / "run.yaml")
    (chats_run_dir / "domain.yaml").unlink()
    with pytest.raises(ConfigError, match="missing domain.yaml.*clm-ml upgrade"):
        load_experiment(chats_run_dir)
    with pytest.raises(ConfigError, match="clm-ml new"):
        load_experiment(tmp_path / "empty")


def test_cli_check_and_errors(chats_run_dir, capsys):
    from clm_ml.cli import main

    assert main(["check", str(chats_run_dir)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["check", str(chats_run_dir), "--set", "run.time.steps=0"]) == 2
    assert "at least one step" in capsys.readouterr().err


@pytest.mark.parametrize(
    "setting, message",
    [
        ("model.physics.solar_radiation=norman", "norman is not supported"),
        ("model.physics.wue_solver=brent", "brent is not supported"),
        ("model.physics.flux_profile_solution=well_mixed", "well_mixed is not supported"),
    ],
)
def test_unsupported_schemes_are_refused(chats_run_dir, setting, message):
    with pytest.raises(ConfigError, match=message):
        load_experiment(chats_run_dir, [setting])


def test_new_from_a_preset(tmp_path, capsys):
    from clm_ml.cli import main

    folder = tmp_path / "my_site"
    assert main(["new", str(folder), "--from", "chats7_1day"]) == 0
    exp = load_experiment(folder)
    assert (exp.n_steps, exp.domain.gridcells.names) == (48, ("CHATS7",))
    assert "--from chats7_1day" in (folder / "params.yaml").read_text()
    with pytest.raises(ConfigError, match="already holds"):
        new_experiment(folder, "chats7_1day")
    with pytest.raises(ConfigError, match="unknown preset 'nope'; presets: chats7_1day"):
        new_experiment(tmp_path / "x", "nope")


def test_upgrade_adds_missing_keys_and_keeps_values(chats_run_dir):
    def change(d):
        d["numerics"].pop("integrator")
        d["physics"]["stomatal_conductance"] = "medlyn"

    _edit(chats_run_dir, "model.yaml", change)
    report = upgrade_experiment(chats_run_dir)
    assert any("numerics.integrator" in line for line in report)
    exp = load_experiment(chats_run_dir)
    assert exp.config.numerics.integrator == "rk4"
    assert exp.config.physics.stomatal_conductance == "medlyn"
    assert (chats_run_dir / ".previous" / "model.yaml").exists()


def test_upgrade_converts_the_layout_before_experiment_folders(chats_run_dir, tmp_path):
    """A folder as `clm-ml convert` wrote it before decision 91 becomes the same experiment
    as the current conversion."""
    old = tmp_path / "old"
    old.mkdir()
    (old / "run.yaml").write_text(yaml.safe_dump({
        "model": "model.yaml",
        "params": "chats",
        "time": {"start": "2007-05-01T00:00", "steps": 50, "step_seconds": 1800},
        "domain": {"type": "towers", "towers": ["CHATS7"]},
        "forcing": yaml.safe_load((chats_run_dir / "run.yaml").read_text())["forcing"],
        "initial_conditions": {"source": "clm_history"},
        "output": {"directory": "output",
                   "streams": [{"name": "step", "frequency": "step", "variables": ["all"]}]},
    }, sort_keys=False))  # fmt: skip
    (old / "model.yaml").write_text("canopy_layers:\n  min_plant_area_per_layer: 0.01\n")
    report = upgrade_experiment(old)
    assert any("layout before experiment folders" in line for line in report)
    a, b = load_experiment(old), load_experiment(chats_run_dir)
    assert (a.domain_file, a.config, a.n_steps) == (b.domain_file, b.config, b.n_steps)
    assert a.run_file.forcing.meteorology.interval_seconds == 1800
    for k in a.params.values:
        np.testing.assert_array_equal(a.params.values[k], b.params.values[k])
