"""Accuracy of the model's derivatives with soil that freezes and thaws (model.yaml
``soil.freeze_thaw``), on a cold CHATS (the tower's air and its initial soil 18 K colder,
snow in the first night, rain on the second day: ``tests/regression/test_frozen_soil.py``).

- *A thawing morning* (snow melting on the ground, the top level at the freezing point)
  and *a freezing night*: forward-mode derivatives of ten steps against central finite
  differences, with respect to the ice impedance, the soil's hydraulic properties and
  porosity, the liquid water and the ice at the start, and a canopy parameter. In these
  windows no level freezes down to its supercooled water, and the stabilized rule
  (``numerics.phase_change_derivative``) gives the same derivatives as the exact one.
- *Three days* in reverse mode. Over a freeze the exact derivative of CLM's phase change
  grows without bound (a level that freezes down to its supercooled water overshoots):
  here to 1e12 and more. The stabilized rule is bounded and follows finite changes of 1 %
  for the fluxes. For the soil's ice with respect to the retention curve and the water it
  has the right sign and is several times too small: a level freezes in pulses (it must
  first cool below the freezing point of its water), and how many steps freeze is a count,
  which no derivative sees.

The derivative with respect to the soil temperature at the start is not compared: a level
that is changing phase sits exactly at the freezing point, where CLM's thermal conductivity
jumps from its thawed to its frozen form, so that a finite difference there is the jump.

Gradient tier (``tests/gradient_tier.py``): compiles two models, four forward derivatives
and two gradients.
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
INPUTS = ("e_ice", "hksat", "bsw", "sucsat", "watsat", "liquid", "ice", "vcmax25")
FLUXES = ("latent_heat_flux", "ground_heat_flux", "soil_phase_change_heat", "snow_on_ground")
OUTPUTS = FLUXES + ("ice at the end", "liquid water at the end", "temperature at the end")
WINDOWS = {"thawing morning": 30, "freezing night": 56}  # the first run step of ten


@pytest.fixture(scope="module")
def models():
    sys.path.insert(0, str(ROOT))
    from clm_ml.experiment import load_experiment
    from clm_ml.model import Model
    from tests.experiment_tools import make_experiment
    from tests.regression.test_frozen_soil import SURFACE, _cold_files

    work = Path(tempfile.mkdtemp())
    tower, history = _cold_files(work / "inputs")

    def model(rule):
        folder = make_experiment(
            work / rule,
            namelist=ROOT / "data" / "namelists" / "nl.CHATS7.1day",
            run={
                "time": {"length": "3 days"},
                "forcing": {
                    "meteorology": {"files": [str(tower)], "snowfall": "by_temperature"},
                    "clm_history": {"file": str(history)},
                },
            },
            model={"soil": {"water": "predicted", "freeze_thaw": True}},
            domain=lambda d: d["towers"]["CHATS7"]["soil"].update(SURFACE),
        )
        overrides = [
            "model.numerics.water_stress_derivative=exact",
            f"model.numerics.phase_change_derivative={rule}",
        ]
        return Model.from_experiment(load_experiment(folder, overrides))

    return {rule: model(rule) for rule in ("exact", "stabilized")}


def _values(model, first: int, steps: int):
    """``scale -> outputs``: the means of ``FLUXES`` over ``steps`` steps from run step
    ``first`` and the soil's total ice, liquid water and mean temperature of the upper ten
    levels at the end, with each of ``INPUTS`` multiplied by ``1 + scale[k]``."""
    from clm_ml.output_streams import StreamSpec

    state = model.initial_state()
    if first > 1:
        state, _ = model.run_chunk(state, model.forcing(1, first - 1), 1)
    forcing = model.forcing(first, steps)
    stream = StreamSpec("s", "step", "mean", FLUXES)

    def values(scale):
        factor = {name: 1.0 + scale[k] for k, name in enumerate(INPUTS)}
        params, constants = model.default_params(), jax.tree.map(jnp.asarray, model.constants())
        hydrology = params.hydrology._replace(e_ice=params.hydrology.e_ice * factor["e_ice"])
        params = params._replace(hydrology=hydrology, vcmax25=params.vcmax25 * factor["vcmax25"])
        constants = constants._replace(
            **{n: getattr(constants, n) * factor[n] for n in ("hksat", "bsw", "sucsat", "watsat")}
        )
        soil = state.soil._replace(
            liquid_water=state.soil.liquid_water * factor["liquid"],
            ice=state.soil.ice * factor["ice"],
        )
        end, out = model.run_chunk(
            state._replace(soil=soil), forcing, first, params, streams=[stream],
            constants=constants,
        )  # fmt: skip
        means = [out["s"][name][:, 0].mean() for name in FLUXES]
        return jnp.stack(
            means
            + [
                end.soil.ice[0].sum(),
                end.soil.liquid_water[0].sum(),
                end.soil.temperature[0, :10].mean(),
            ]
        )

    values(jnp.zeros(len(INPUTS)))  # fills the host caches outside a trace
    return values


@pytest.mark.parametrize("window", list(WINDOWS))
def test_derivatives_match_finite_differences(models, window):
    zero, eye = jnp.zeros(len(INPUTS)), jnp.eye(len(INPUTS))
    derivatives = {}
    for rule, model in models.items():
        values = _values(model, WINDOWS[window], 10)
        jvp = jax.jit(jax.vmap(lambda d, f=values: jax.jvp(f, (zero,), (d,))[1]))
        derivatives[rule] = np.asarray(jvp(eye))  # (input, output)
        run = jax.jit(values)
    ad = derivatives["exact"]
    assert np.isfinite(ad).all()
    np.testing.assert_allclose(derivatives["stabilized"], ad, rtol=1e-9, atol=1e-12)

    def central(h):
        return np.array([(np.asarray(run(h * d)) - np.asarray(run(-h * d))) / (2 * h) for d in eye])

    fd, fd_wide = central(1e-5), central(2e-5)
    scale = 1e-3 * np.abs(fd) + 1e-7 * (1.0 + np.abs(np.asarray(run(zero))))[None, :]
    kink = np.abs(fd - fd_wide)  # the two finite differences disagree: a switch within reach
    assert np.all(np.abs(ad - fd) <= scale + 3.0 * kink), np.max(np.abs(ad - fd) / scale)
    assert np.mean(np.abs(ad - fd) <= scale) > 0.9
    # the phase change responds: to the ice, the water, the retention curve
    heat = OUTPUTS.index("soil_phase_change_heat")
    for name in ("ice", "liquid", "bsw", "watsat"):
        assert abs(ad[INPUTS.index(name), heat]) > 1e-3, name


def test_three_day_gradient_is_bounded_with_the_stabilized_rule(models):
    zero, eye = jnp.zeros(len(INPUTS)), jnp.eye(len(INPUTS))
    gradient = {}
    for rule, model in models.items():
        values = _values(model, 1, 143)
        gradient[rule] = np.asarray(jax.jit(jax.jacrev(values))(zero)).T  # (input, output)
        run = jax.jit(values)
    stabilized, exact = gradient["stabilized"], gradient["exact"]
    assert np.isfinite(stabilized).all()
    value = np.abs(np.asarray(run(zero)))
    # bounded: the derivative with respect to a relative change of an input stays within
    # two orders of magnitude of the value it changes (the exact one exceeds it by eight
    # and more)
    assert np.all(np.abs(stabilized) < 100.0 * (1.0 + value)[None, :])
    assert np.max(np.abs(exact)) > 1e8 * np.max(np.abs(stabilized))

    def change(name, h=1e-2):
        d = eye[INPUTS.index(name)]
        return (np.asarray(run(h * d)) - np.asarray(run(-h * d))) / (2 * h)

    def ruled(name):
        return stabilized[INPUTS.index(name)]

    latent, ground = OUTPUTS.index("latent_heat_flux"), OUTPUTS.index("ground_heat_flux")
    np.testing.assert_allclose(ruled("vcmax25")[latent], change("vcmax25")[latent], rtol=0.1)
    np.testing.assert_allclose(ruled("e_ice")[latent], change("e_ice")[latent], rtol=0.15)
    np.testing.assert_allclose(ruled("e_ice")[ground], change("e_ice")[ground], rtol=0.15)
    # the ice left after three days with respect to the retention curve, the porosity and
    # the water: the right sign, too small (see the module docstring)
    ice = OUTPUTS.index("ice at the end")
    for name in ("bsw", "watsat", "liquid"):
        ratio = ruled(name)[ice] / change(name)[ice]
        assert 0.1 < ratio < 1.5, (name, ratio)
