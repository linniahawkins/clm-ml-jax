"""clm_ml.physics.math: the tridiagonal solver as a scan (Phase 7.2, process 1).

The Fortran golden tests check every routine against Fortran (unguarded pivots); these check
the guarded variant used by the solar radiation and soil temperature solves.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from clm_ml.physics.math import tridiag  # noqa: E402


def _system(n, seed):
    rng = np.random.default_rng(seed)
    a, c, r = rng.uniform(-1, 0, n), rng.uniform(-1, 0, n), rng.normal(size=n)
    b = 2.5 + rng.uniform(size=n)  # diagonally dominant: every pivot > 1
    a[0], c[-1] = 0.0, 0.0
    return a, b, c, r


def _dense(a, b, c):
    return np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)


def test_tridiag_solves_the_system():
    for n in (1, 2, 7, 103):
        a, b, c, r = _system(n, n)
        expected = np.linalg.solve(_dense(a, b, c), r)
        for min_pivot in (None, 1.0e-10):
            u = np.asarray(tridiag(a, b, c, r, min_pivot=min_pivot))
            np.testing.assert_allclose(u, expected, rtol=1e-12, atol=1e-14)


def test_guard_changes_nothing_when_pivots_exceed_it():
    a, b, c, r = _system(50, 3)
    a[0] = 123.0  # a[0] is not used
    guarded = np.asarray(tridiag(a, b, c, r, min_pivot=1.0e-10))
    a[0] = 0.0
    np.testing.assert_array_equal(guarded, np.asarray(tridiag(a, b, c, r)))


def test_guard_keeps_a_zero_pivot_finite():
    a, b, c, r = (np.zeros(3), np.array([0.0, 1.0, 1.0]), np.zeros(3), np.ones(3))
    assert np.all(np.isfinite(np.asarray(tridiag(a, b, c, r, min_pivot=1.0e-10))))
