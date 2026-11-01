"""run.yaml loading and validation, and namelist conversion."""

import datetime as dt

import pytest
import yaml

from clm_ml.io.namelist import convert_namelist
from clm_ml.io.yaml_schema import ConfigError
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


def test_run_reads_the_files_and_times_of_the_original_namelist(chats_run_dir):
    import f90nml

    from clm_ml.setup.towers import gridcell_sources

    run = load_run(chats_run_dir / "run.yaml")
    original = f90nml.read(str(CHATS_NAMELISTS / "nl.CHATS7.50steps"))["clmml_inparm"]
    assert int(f"{run.start:%Y%m%d}") == original["start_ymd"]
    assert run.n_steps == original["stop_n"]
    assert original["clm_phys"] == "CLM5_0"  # the only soil layer structure (C35)
    (source,) = gridcell_sources(run)
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
        (lambda d: d["domain"].update(type="grid"), "domain.grid: required"),
        (lambda d: d["initial_conditions"].update(source="restart"), "restart_file: required"),
        (lambda d: d.update(execution={"chunk": "2 months"}), "expected e.g. '10 days'"),
        (lambda d: d.update(execution={"chunk": "5 hours"}), "expected e.g. '10 days'"),
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
        (lambda d: d["time"].update(calendar="noleap"), "only for grid runs"),
        (lambda d: d["time"].update(calendar="julian"), "calendar"),
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


def test_norman_solar_radiation_is_not_supported(chats_run_dir):
    with pytest.raises(ConfigError, match="norman is not supported"):
        load_run(chats_run_dir / "run.yaml", ["model.physics.solar_radiation=norman"])


def test_brent_wue_solver_is_not_supported(chats_run_dir):
    with pytest.raises(ConfigError, match="brent is not supported"):
        load_run(chats_run_dir / "run.yaml", ["model.physics.wue_solver=brent"])


def test_well_mixed_flux_profile_is_not_supported(chats_run_dir):
    with pytest.raises(ConfigError, match="well_mixed is not supported"):
        load_run(chats_run_dir / "run.yaml", ["model.physics.flux_profile_solution=well_mixed"])
