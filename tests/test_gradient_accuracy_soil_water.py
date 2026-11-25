"""Accuracy of the model's derivatives with predicted soil water (model.yaml
``soil.water: predicted``): forward-mode derivatives of a CHATS7 chunk against central
finite differences, and a reverse-mode gradient over two days.

- *A wet column in rain* (the night of 2 May 2007, the soil water raised to near
  saturation): a water table in the column, drainage, runoff from the saturated fraction,
  and the Richards solver taking 6 to 18 sub-steps through its fixed-count loop.
  Derivatives with respect to the runoff and drainage parameters and the soil's hydraulic
  properties.
- *A dry day* (the morning and midday of 1 May): the trees take the water. Derivatives
  with respect to the soil's hydraulic properties and porosity, the soil water at the
  start, and three plant parameters (through transpiration and root uptake).
- *Two days*: the reverse-mode gradient of the mean latent heat and of the soil water
  left at the end.

The exact derivatives are compared (``numerics.water_stress_derivative: exact``; the
default rule replaces the water stress's derivative by a bounded approximation). The
model is piecewise smooth (limits, the sub-step count, leaves reaching the minimum
conductance): where the two finite differences (steps h and 2h) disagree, a kink lies
within the step, and the derivative is compared no closer than they agree.

Gradient tier (``tests/gradient_tier.py``): compiles three chunks, two forward
derivatives and a gradient.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytestmark = [pytest.mark.gradient, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]
NAMELIST = ROOT / "data" / "namelists" / "nl.CHATS7.1day"
STEPS = 8
HYDROLOGY = ("fff", "baseflow_scalar", "n_baseflow")
CONSTANTS = ("hksat", "bsw", "sucsat", "watsat", "wtfact", "topo_slope")
PLANT = ("vcmax25", "psi50_gs", "plant_conductance")
INPUTS = HYDROLOGY + CONSTANTS + PLANT + ("soil_water",)
#: per window: the first run step, the factor on the soil water at its start (capped just
#: below saturation), the inputs differentiated and the step outputs compared
WINDOWS = {
    "wet column in rain": (
        60,
        1.4,
        ("fff", "baseflow_scalar", "n_baseflow", "hksat", "bsw", "sucsat", "wtfact", "topo_slope"),
        ("latent_heat_flux", "surface_runoff", "drainage", "water_table_depth", "infiltration"),
    ),
    "dry day": (
        30,
        1.0,
        ("hksat", "bsw", "sucsat", "watsat", "soil_water") + PLANT,
        ("latent_heat_flux", "root_water_uptake", "ground_evaporation"),
    ),
}


@pytest.fixture(scope="module")
def model():
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from clm_ml.model import Model
    from tests.experiment_tools import make_experiment

    folder = make_experiment(
        Path(tempfile.mkdtemp()),
        namelist=NAMELIST,
        run={"time": {"length": "2 days"}},
        model={"soil": {"water": "predicted"}},
        domain=lambda d: d["towers"]["CHATS7"]["soil"].update(
            {"max_saturated_fraction": 0.4, "slope": 1.0}
        ),
    )
    return Model.from_experiment(
        load_experiment(folder, ["model.numerics.water_stress_derivative=exact"])
    )


def _scaled(model, state, scale, names):
    """The parameters, constants and state with the inputs ``names`` multiplied by
    ``1 + scale[k]`` (the soil water: of the levels above the bedrock)."""
    factor = {name: 1.0 + scale[k] for k, name in enumerate(names)}
    params, constants = model.default_params(), jax.tree.map(jnp.asarray, model.constants())
    hydrology = params.hydrology._replace(
        **{n: getattr(params.hydrology, n) * factor[n] for n in HYDROLOGY if n in factor}
    )
    params = params._replace(
        hydrology=hydrology, **{n: getattr(params, n) * factor[n] for n in PLANT if n in factor}
    )
    constants = constants._replace(
        **{n: getattr(constants, n) * factor[n] for n in CONSTANTS if n in factor}
    )
    if "soil_water" in factor:
        water = state.soil.liquid_water
        above_bedrock = jnp.arange(water.shape[1])[None, :] < constants.nbedrock[:, None]
        water = water * jnp.where(above_bedrock, factor["soil_water"], 1.0)
        state = state._replace(soil=state.soil._replace(liquid_water=water))
    return params, constants, state


def _stored(state):
    return state.soil.liquid_water[0].sum() + state.soil.surface_water[0]


@pytest.mark.parametrize("window", list(WINDOWS))
def test_derivatives_match_finite_differences(model, window):
    from clm_ml.output_streams import StreamSpec

    first, wet, names, variables = WINDOWS[window]
    state, _ = model.run_chunk(model.initial_state(), model.forcing(1, first - 1), 1)
    constants = model.constants()
    capacity = jnp.asarray(constants.watsat[:, :20] * constants.dz[:, :20] * 1000.0)
    water = jnp.minimum(state.soil.liquid_water * wet, 0.999 * capacity)
    state = state._replace(soil=state.soil._replace(liquid_water=water))
    forcing = model.forcing(first, STEPS)
    stream = StreamSpec("s", "step", "mean", variables + ("soil_water_substeps",))

    def values(scale):  # (output, step): the step outputs, then the water stored at the end
        params, constants, start = _scaled(model, state, scale, names)
        end, out = model.run_chunk(
            start, forcing, first, params, streams=[stream], constants=constants
        )
        rows = [out["s"][v][:, 0] for v in variables]
        return jnp.stack(rows + [jnp.broadcast_to(_stored(end), (STEPS,))]), out["s"]

    zero = jnp.zeros(len(names))
    _, base = values(zero)  # fills the host caches outside a trace
    substeps = np.asarray(base["soil_water_substeps"][:, 0])
    run = jax.jit(lambda scale: values(scale)[0])
    eye = jnp.eye(len(names))
    ad = np.asarray(jax.jit(jax.vmap(lambda d: jax.jvp(run, (zero,), (d,))[1]))(eye))

    def central(h):
        return np.array([(np.asarray(run(h * d)) - np.asarray(run(-h * d))) / (2 * h) for d in eye])

    fd, fd_wide = central(1e-5), central(2e-5)
    assert np.isfinite(ad).all()
    if wet > 1.0:  # the fixed-count loop with several sub-steps, drainage and runoff
        assert substeps.max() > 10 and substeps.min() > 1
        assert np.all(np.asarray(base["drainage"]) > 0.0)
        assert np.max(np.asarray(base["surface_runoff"])) > 0.0  # in the steps with rain
    else:
        assert np.all(np.asarray(base["root_water_uptake"]) > 0.0)

    outputs = variables + ("stored water at the end",)
    close = []
    for k, name in enumerate(names):
        assert np.abs(fd[k]).max() > 0.0, name  # every input matters in its window
        for v, output in enumerate(outputs):
            scale = np.abs(fd[k, v]).max()
            if scale == 0.0:
                assert np.abs(ad[k, v]).max() < 1e-20, (name, output)
                continue
            error = np.abs(ad[k, v] - fd[k, v])
            kink = np.abs(fd[k, v] - fd_wide[k, v])  # the finite differences disagree
            assert np.all(error <= 1e-3 * scale + 3.0 * kink), (name, output, error / scale)
            close.append(error <= 1e-3 * scale)
    close = np.concatenate(close)
    assert close.mean() > 0.9, close.mean()  # kinks within a step of 1e-5 are rare


def test_two_day_gradient_matches_finite_differences(model):
    """Reverse mode over two days (95 steps, rain on the second): the mean latent heat and
    the soil water left at the end, along a 0.1 % change of each input."""
    from clm_ml.output_streams import StreamSpec

    names = ("fff", "hksat", "bsw", "wtfact", "vcmax25", "psi50_gs", "soil_water")
    state, first, steps = model.initial_state(), 1, 95
    forcing = model.forcing(first, steps)
    stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

    def values(scale):
        params, constants, start = _scaled(model, state, scale, names)
        end, out = model.run_chunk(
            start, forcing, first, params, streams=[stream], constants=constants
        )
        return jnp.stack([out["s"]["latent_heat_flux"][:, 0].mean(), _stored(end)])

    zero = jnp.zeros(len(names))
    values(zero)
    gradient = np.asarray(jax.jit(jax.jacrev(values))(zero))  # (output, input)
    run, h = jax.jit(values), 1e-3
    assert np.isfinite(gradient).all()
    for k, name in enumerate(names):
        d = jnp.zeros(len(names)).at[k].set(h)
        fd = (np.asarray(run(d)) - np.asarray(run(-d))) / (2 * h)
        assert np.all(fd != 0.0), name
        np.testing.assert_allclose(gradient[:, k], fd, rtol=0.02, err_msg=name)
