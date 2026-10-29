"""run.yaml loading and validation, and namelist conversion."""

import datetime as dt

import pytest
import yaml

from clm_ml.io.yaml_schema import ConfigError
from clm_ml.legacy.namelist import convert_namelist
from clm_ml.run import load_run

from .conftest import CHATS_NAMELISTS


def _edit(run_dir, change):
    data = yaml.safe_load((run_dir / "run.yaml").read_text())
    change(data)
    (run_dir / "run.yaml").write_text(yaml.safe_dump(data))
    return run_dir / "run.yaml"


def test_converted_50_steps(chats_run_dir):
    run = load_run(chats_run_dir / "run.yaml")
    assert run.start == dt.datetime(2007, 5, 1)
    assert run.n_steps == 50
    assert run.params.source == "chats.yaml"
    assert run.domain.gridcells.names == ("CHATS7",)
    assert "canopy layers 46" in run.summary()


def test_legacy_spec_matches_the_original_namelist(chats_run_dir):
    import f90nml

    spec = load_run(chats_run_dir / "run.yaml").legacy_spec(None)["namelist"]
    original = f90nml.read(str(CHATS_NAMELISTS / "nl.CHATS7.50steps"))["clmml_inparm"]
    for key in ("tower_name", "start_ymd", "start_tod", "stop_n", "clm_start_ymd",
                "clm_start_tod", "clm_phys", "nlev_soil_adjust", "met_type", "dpai_min"):  # fmt: skip
        assert spec[key] == original[key], key
    for key in ("fin_tower", "fin_clm", "fin_soil_adjust"):
        assert spec[key].endswith(original[key].replace("../", "")), key


@pytest.mark.parametrize("name", ["nl.CHATS7.1day", "nl.CHATS7.05.2007", "nl.CHATS7.50steps"])
def test_all_chats_namelists_convert_and_load(tmp_path, name):
    convert_namelist(CHATS_NAMELISTS / name, tmp_path)
    load_run(tmp_path / "run.yaml")


def test_days_and_end_are_equivalent(chats_run_dir):
    def by_days(d):
        d["time"].pop("steps")
        d["time"]["days"] = 1

    def by_end(d):
        d["time"].pop("days")
        d["time"]["end"] = "2007-05-02T00:00"

    assert load_run(_edit(chats_run_dir, by_days)).n_steps == 48
    assert load_run(_edit(chats_run_dir, by_end)).n_steps == 48


def test_overrides(chats_run_dir):
    run = load_run(chats_run_dir / "run.yaml",
                   ["time.steps=10", "model.physics.stomatal_conductance=medlyn"])  # fmt: skip
    assert run.n_steps == 10
    assert run.config.physics.stomatal_conductance == "medlyn"
    assert "not been validated" in run.summary()


@pytest.mark.parametrize(
    "change, message",
    [  # fmt: skip
        (lambda d: d["time"].update(days=1), "exactly one of end, days, steps"),
        (lambda d: d["time"].update(step_seconds=3600), "forcing_step_minutes 30"),
        (lambda d: d["domain"].update(towers=["CHATS7", "CHATS7"]), "listed twice"),
        (lambda d: d["domain"].update(type="grid"), "not yet supported"),
        (lambda d: d["initial_conditions"].update(source="restart"), "restart_file: required"),
        (lambda d: d.update(execution={"chunk": "5 days"}), "only '1 month'"),
        (lambda d: d.update(execution={"restart_every": "weekly"}), "expected e.g."),
        (lambda d: d.update(colour="blue"), "unknown key(s) ['colour']"),
        (lambda d: d["time"].update(start="May 1"), "is not a date-time"),
        # forcing (plan §4.1 layout)
        (lambda d: d["forcing"]["meteorology"].update(files="nope.nc"), "file not found"),
        (lambda d: d["forcing"].update(meteorology="met.nc"), "meteorology is now a section"),
        (lambda d: d["forcing"].update(interpolation="none"), "forcing.meteorology.interpolation"),
        (lambda d: d["forcing"]["meteorology"].update(files="met_{year}-{month:02d}.nc"),
         "file not found for 2007-05"),
        (lambda d: d["forcing"]["meteorology"].update(shortwave="cloudy"), "cloudy"),
        (lambda d: d["forcing"].update(vegetation={"source": "file"}),
         "forcing.vegetation.files: required"),
        (lambda d: d["forcing"].update(soil_water={"source": "file", "files": "nope.nc"}),
         "file not found"),
        (lambda d: d["forcing"].update(vegetation={"source": "file", "file": "x.nc"}),
         "replaced by files"),
        (lambda d: d["forcing"].update(vegetation={"source": "monthly"}), "monthly"),
    ]
    ,
)
def test_invalid_run_files(chats_run_dir, change, message):
    with pytest.raises(ConfigError) as exc:
        load_run(_edit(chats_run_dir, change))
    assert message in str(exc.value)


def test_cli_check_and_errors(chats_run_dir, capsys):
    from clm_ml.cli import main

    assert main(["check", str(chats_run_dir / "run.yaml")]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["check", str(chats_run_dir / "run.yaml"), "--set", "time.steps=0"]) == 2
    assert "at least one step" in capsys.readouterr().err
