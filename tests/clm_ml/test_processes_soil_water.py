"""clm_ml.processes.soil_water."""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from clm_ml.processes import soil_water  # noqa: E402

# relative saturation from CLM's dry bound to saturated; a loam
S = jnp.array([0.01, 0.05, 0.3, 0.75, 1.0])
HKSAT, SUCSAT, BSW = 0.0035, 230.0, 6.9


def test_conductivity_and_suction_at_saturation_are_the_saturated_values():
    hk, _ = soil_water.soil_hk(1.0, 1.0, HKSAT, BSW)
    smp, _ = soil_water.soil_suction(1.0, SUCSAT, BSW)
    assert float(hk) == HKSAT
    assert float(smp) == -SUCSAT


def test_hand_coded_derivatives_are_the_derivatives():
    """CLM's hand-coded slopes with respect to saturation equal automatic differentiation."""
    _, dhkds = soil_water.soil_hk(S, 0.5, HKSAT, BSW)
    _, dsmpds = soil_water.soil_suction(S, SUCSAT, BSW)
    auto_hk = jax.vmap(jax.grad(lambda s: soil_water.soil_hk(s, 0.5, HKSAT, BSW)[0]))(S)
    auto_smp = jax.vmap(jax.grad(lambda s: soil_water.soil_suction(s, SUCSAT, BSW)[0]))(S)
    np.testing.assert_allclose(np.asarray(dhkds), np.asarray(auto_hk), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(dsmpds), np.asarray(auto_smp), rtol=1e-12)


def test_gradients_with_respect_to_soil_parameters_match_finite_differences():
    """Including the exponent ``bsw``, at the dry bound and at saturation."""

    def outputs(p):
        hk, dhkds = soil_water.soil_hk(S, 0.5, p[0], p[2])
        smp, dsmpds = soil_water.soil_suction(S, p[1], p[2])
        # logarithms: the values span many orders of magnitude over the saturations
        return jnp.log(jnp.concatenate([hk, dhkds, -smp, dsmpds]))

    p0 = jnp.array([HKSAT, SUCSAT, BSW])
    jac = np.asarray(jax.jacfwd(outputs)(p0))
    assert np.all(np.isfinite(jac))
    for k in range(3):
        h = 1e-6 * float(p0[k])
        fd = (outputs(p0.at[k].add(h)) - outputs(p0.at[k].add(-h))) / (2 * h)
        np.testing.assert_allclose(jac[:, k], np.asarray(fd), rtol=1e-6, atol=1e-9)
