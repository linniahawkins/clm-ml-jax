"""The output variables give the same values computed with jax.numpy (as inside a compiled
chunk, for output streams) as with numpy (after a recorded run)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clm_ml.outputs import VARIABLES, _Raw

jax.config.update("jax_enable_x64", True)

REFERENCE = Path(__file__).parents[1] / "regression" / "reference" / "CHATS7_50steps.npz"


@pytest.mark.skipif(not REFERENCE.exists(), reason="regression reference not present")
def test_variables_numpy_equals_jax():
    with np.load(REFERENCE) as data:
        raw = {k: data[k] for k in data.files if k.startswith("mlcanopy.")}
    r_np = _Raw(raw, max_layers=50, xp=np)
    r_jx = _Raw({k: jnp.asarray(v) for k, v in raw.items()}, max_layers=50, xp=jnp)
    for spec in VARIABLES:
        a = np.asarray(spec.compute(r_np), dtype=float)
        b = np.asarray(spec.compute(r_jx), dtype=float)
        assert a.shape == b.shape, spec.name
        np.testing.assert_allclose(b, a, rtol=1e-14, atol=0, equal_nan=True, err_msg=spec.name)
