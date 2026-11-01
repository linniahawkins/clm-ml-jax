"""The configurations in ``configs/``: every model.yaml complete and current (generated from
``clm_ml.config``), every run.yaml valid, every built-in params file complete."""

from pathlib import Path

import pytest

from clm_ml.config import ModelConfig, model_yaml_text
from clm_ml.params import Params
from clm_ml.run import load_run

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MODEL_FILES = sorted(CONFIGS.rglob("model.yaml"))
RUN_FILES = sorted(CONFIGS.glob("*/run.yaml"))


def test_default_model_yaml_is_current():
    """configs/model.yaml is exactly what `clm-ml model-yaml` writes for the defaults."""
    assert (CONFIGS / "model.yaml").read_text() == model_yaml_text(
        ModelConfig()
    ), "configs/model.yaml is out of date: run `clm-ml model-yaml -o configs/model.yaml`"


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_every_model_yaml_is_complete_and_current(path):
    """Each model.yaml lists every setting with its explanation (as `clm-ml model-yaml
    --from` writes it), so nothing is left to hidden defaults."""
    assert path.read_text() == model_yaml_text(ModelConfig.from_yaml(path)), (
        f"{path} is not complete or not current: run "
        f"`clm-ml model-yaml --from {path} -o {path}`"
    )


@pytest.mark.parametrize("path", RUN_FILES, ids=lambda p: p.parent.name)
def test_every_run_yaml_loads(path):
    if path.parent.name.startswith("grid") and not Path("/glade/campaign").exists():
        pytest.skip("grid inputs are on NCAR storage")
    run = load_run(path)
    assert run.n_steps > 0


@pytest.mark.parametrize("name", ["clm5_defaults", "chats", "clm6"])
def test_builtin_params_files_are_complete(name):
    Params.from_yaml(name)  # refuses a file missing any parameter of any PFT 1-16
