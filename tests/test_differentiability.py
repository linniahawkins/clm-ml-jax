"""
Integration tests for differentiability of the CLM-ML-JAX model.

Tests:
  - jax.grad(forward_fn) completes without error
  - Gradients are finite and non-zero for key state variables
  - Gradient magnitudes pass a finite-difference sanity check (loose tolerance)

The forward pass is one time step (``clm_ml.step.step``) of the CHATS7 patch after its first
step (``clm_ml.model.Model``); the loss is the sum of the sensible heat and water vapor fluxes
of the layers in use. Mark them slow so the default test run skips them:

    pytest -m "not slow"   # skip differentiability tests
    pytest -m slow         # run only differentiability tests
    pytest                 # run everything

"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------
# JAX must be configured before any import that touches jax arrays
# -----------------------------------------------------------------------
import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytestmark = pytest.mark.gradient  # the gradient tier (tests/gradient_tier.py)

# Project root on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

NAMELIST = PROJECT_ROOT / "data" / "namelists" / "nl.CHATS7.1day"


# -----------------------------------------------------------------------
# Shared model initialization (done once per pytest session)
# -----------------------------------------------------------------------


def _build_initialized_model():
    """``(forward, state, info)``: ``forward(state, params, inputs)`` -> scalar loss of one
    step (run step 1) of the CHATS7 patch from ``state`` (after the first step); ``info``
    holds the step's default ``params`` and ``inputs`` and the patch's layer grid."""
    import tempfile

    from clm_ml import driver as D
    from clm_ml import step as S
    from clm_ml.experiment import load_experiment
    from clm_ml.io.namelist import convert_namelist
    from clm_ml.model import Model

    tmp = Path(tempfile.mkdtemp())
    convert_namelist(NAMELIST, tmp)
    model = Model.from_experiment(load_experiment(tmp))
    sim = model._sim
    one = lambda t: jax.tree.map(lambda a: jnp.asarray(a)[0], t)  # noqa: E731
    xg, xp = model.forcing(1, 1)
    inputs = S.step_inputs(jax.tree.map(lambda a: a[0, 0], xg), jax.tree.map(lambda a: a[0, 0], xp))
    constants, params = one(jax.tree.map(jnp.asarray, sim.constants)), one(sim.patch_params())
    use = constants.layers.in_use

    def forward(state, params=params, inputs=inputs):
        state = D.advance_acclimation(sim, state, inputs.tref)
        d = S.step(
            state, inputs, constants, params, sim.tables, options=sim.options, c3=bool(sim.c3[0])
        ).diagnostics
        return jnp.sum(jnp.where(use, d.shair_profile, 0.0)) + jnp.sum(
            jnp.where(use, d.etair_profile, 0.0)
        )

    info = {"params": params, "inputs": inputs, "layers": constants.layers}
    return forward, one(model.initial_state()), info


@pytest.fixture(scope="session")
def initialized_model():
    if not NAMELIST.exists():
        pytest.skip(f"Namelist not found: {NAMELIST}")
    return _build_initialized_model()


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


@pytest.mark.slow
def test_forward_pass_returns_finite_scalar(initialized_model):
    """Differentiable forward pass returns a finite scalar loss."""
    forward, state, _ = initialized_model
    loss = forward(state)
    assert jnp.isfinite(loss), f"Forward loss is not finite: {loss}"
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


@pytest.mark.slow
def test_jax_grad_completes(initialized_model):
    """jax.grad(forward) with respect to the state completes without raising."""
    forward, state, _ = initialized_model
    assert jax.grad(forward)(state) is not None


@pytest.mark.slow
def test_gradients_are_finite(initialized_model):
    """The gradients with respect to the prognostic state are finite in the layers in use."""
    forward, state, info = initialized_model
    g = jax.grad(forward)(state).canopy
    use = info["layers"].in_use
    for name in ("air_temperature", "vapor_pressure", "leaf_temperature"):
        values = getattr(g, name)
        mask = use.reshape((-1,) + (1,) * (values.ndim - 1))
        assert jnp.all(jnp.isfinite(jnp.where(mask, values, 0.0))), name
    assert jnp.isfinite(g.ground_temperature)


@pytest.mark.slow
def test_gradients_are_nonzero(initialized_model):
    """The loss depends on the prognostic state."""
    forward, state, _ = initialized_model
    g = jax.grad(forward)(state).canopy
    assert any(
        float(jnp.max(jnp.abs(getattr(g, name)))) > 0.0
        for name in ("leaf_temperature", "air_temperature", "ground_temperature")
    ), "all primary gradients are zero"


@pytest.mark.slow
def test_finite_difference_gradient_check(initialized_model):
    """The gradient with respect to the sunlit leaf temperature of a leafy layer agrees with a
    centred finite difference (generous tolerance: the physics is not everywhere smooth)."""
    forward, state, info = initialized_model
    lay = info["layers"]
    ic = (int(lay.first_leafy) + int(lay.n_within) - 1) // 2  # a layer with leaves
    g_analytic = float(jax.grad(forward)(state).canopy.leaf_temperature[ic, 0])
    if g_analytic == 0.0:
        pytest.skip("Analytic gradient is zero at test point")
    eps = 1e-3  # K
    t = state.canopy.leaf_temperature

    def loss_at(dt):
        return float(forward(state._replace(canopy=state.canopy._replace(
            leaf_temperature=t.at[ic, 0].add(dt)))))  # fmt: skip

    g_fd = (loss_at(eps) - loss_at(-eps)) / (2.0 * eps)
    denom = abs(g_fd) if abs(g_fd) > 1e-30 else abs(g_analytic)
    rel_err = abs(g_analytic - g_fd) / denom if denom > 0 else 0.0
    assert rel_err < 0.10, f"analytic={g_analytic:.6e}, fd={g_fd:.6e}, rel_err={rel_err:.3%}"


@pytest.mark.slow
def test_parameter_gradient_matches_finite_difference(initialized_model):
    """d(loss)/d(vcmax25) from jax.grad agrees with a centred finite difference."""
    forward, state, info = initialized_model
    params = info["params"]
    g_analytic = float(jax.grad(forward, argnums=1)(state, params).vcmax25)
    assert g_analytic != 0.0, "loss does not depend on vcmax25"
    eps = 1e-3 * float(params.vcmax25)
    f_plus = float(forward(state, params._replace(vcmax25=params.vcmax25 + eps)))
    f_minus = float(forward(state, params._replace(vcmax25=params.vcmax25 - eps)))
    g_fd = (f_plus - f_minus) / (2.0 * eps)
    rel_err = abs(g_analytic - g_fd) / abs(g_fd)
    assert rel_err < 0.05, f"analytic={g_analytic:.6e}, fd={g_fd:.6e}, rel_err={rel_err:.3%}"


@pytest.mark.slow
def test_forcing_gradient_matches_finite_difference(initialized_model):
    """d(loss)/d(air temperature of the next forcing record) from jax.grad agrees with a
    centred finite difference (the forcing is time-interpolated within the step)."""
    forward, state, info = initialized_model
    params, inputs = info["params"], info["inputs"]
    grads = jax.grad(forward, argnums=2)(state, params, inputs)
    g_analytic = float(grads.forcing_next.tref)
    assert g_analytic != 0.0, "loss does not depend on the next record's air temperature"
    eps = 0.01  # K

    def loss_at(dt):
        nxt = inputs.forcing_next._replace(tref=inputs.forcing_next.tref + dt)
        return float(forward(state, params, inputs._replace(forcing_next=nxt)))

    g_fd = (loss_at(eps) - loss_at(-eps)) / (2.0 * eps)
    rel_err = abs(g_analytic - g_fd) / abs(g_fd)
    assert rel_err < 0.05, f"analytic={g_analytic:.6e}, fd={g_fd:.6e}, rel_err={rel_err:.3%}"
