"""
JAX translation of MLRungeKuttaMod Fortran module.

Runge-Kutta method for state updates in the multilayer canopy model.
Provides Butcher tableau initialisation and per-step state updates for
explicit Runge-Kutta time integration of canopy and soil variables.

Original Fortran module: MLRungeKuttaMod
Fortran lines 1-240
"""

from typing import Tuple

import jax.numpy as jnp
from jax import Array

from clm_src_main.abortutils import endrun  # noqa: F401
from clm_src_main.clm_varcon import spval  # noqa: F401
from clm_src_main.clm_varctl import iulog  # noqa: F401
from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from multilayer_canopy.MLclm_varctl import MLOptions
from multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401

# ---------------------------------------------------------------------------
# Public: Runge-Kutta state update
# ---------------------------------------------------------------------------


def RungeKuttaUpdate(
    irk: int,
    a: Array,
    b: Array,
    c: Array,
    num_filter: int,
    filter: Array,
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo",  # noqa: F821
) -> mlcanopy_type:
    """
    Runge-Kutta state update: update states for the next Runge-Kutta step.

    Implements the general p-order explicit Runge-Kutta scheme:

    .. code-block:: none

        k(1) = h * f[t_n, y_n]
        k(j) = h * f[t_n + c(j)*h, y_n + sum_{i<j} a(j,i) * k(i)]
        y_{n+1} = y_n + sum_j b(j) * k(j)

    At each step ``irk < nrk`` the states are set to an intermediate
    value for the next stage evaluation. At ``irk == nrk`` the states
    are set to the final weighted-average update.

    Mirrors Fortran subroutine ``RungeKuttaUpdate`` (lines 27-140).

    Args:
        irk: Current Runge-Kutta step index (1-based, matching Fortran).
        a: Butcher tableau lower-triangular matrix, shape ``(nrk, nrk)``.
            Fortran: ``real(r8), intent(in) :: a(nrk,nrk)``.
        b: Butcher tableau weight vector, shape ``(nrk,)``.
            Fortran: ``real(r8), intent(in) :: b(nrk)``.
        c: Butcher tableau node vector, shape ``(nrk,)`` (unused in
            state update but kept for interface consistency).
            Fortran: ``real(r8), intent(in) :: c(nrk)``.
        num_filter: Number of patches in the filter.
        filter: 1-D array of patch indices to process.
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Updated :class:`mlcanopy_type` with state variables and
        accumulated Runge-Kutta increments written back.
    """
    # irk is 1-based (matching Fortran). It may be a Python int or a traced integer
    # (the Runge-Kutta stages run as a lax.scan when the model is traced), so the
    # stage-dependent sums use masks instead of slices of variable length:
    #   stage irk < nrk:  x = x_bef + sum_{j<irk} a(irk+1, j) dx_j   (next stage's state)
    #   stage irk = nrk:  x = x_bef + sum_{j<=nrk} b(j) dx_j        (final update)
    #   irk > nrk:        unchanged (physics-only pass after the last stage)
    nrk = b.shape[0]  # number of stages, from the tableau (RungeKuttaIni)
    irk0 = jnp.asarray(irk) - 1
    do_update = irk0 < nrk
    k = jnp.minimum(irk0, nrk - 1)  # column written this stage (clamped when no update)
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    weights = jnp.where(irk0 < nrk - 1, a[jnp.minimum(irk0 + 1, nrk - 1)], b)
    w_curr = weights[k]

    def _advance(x_bef, x_cur, dx_prev):
        """New state and this stage's increment; dx_prev has the stage axis last."""
        dx_new = x_cur - x_bef
        m = dx_prev.shape[-1]  # stage axis (may be longer than nrk)
        j = jnp.arange(m)
        # earlier stages j < k only (a's upper triangle is spval)
        w_prev = jnp.zeros(m).at[:nrk].set(jnp.where(j[:nrk] < k, weights, 0.0))
        x_new = x_bef + jnp.where(j < k, dx_prev * w_prev, 0.0).sum(axis=-1) + w_curr * dx_new
        return jnp.where(do_update, x_new, x_cur), dx_new

    # Canopy layers 1..ncan (Fortran: do ic = 1, ncan(p)); the arrays are 1-based.
    p, n = grid.p, grid.ncan
    updates = {}
    for state, delta, bef, sl in (
        ("tair_profile", "dtair_profile", "tair_bef_profile", (p, slice(1, n + 1))),
        ("eair_profile", "deair_profile", "eair_bef_profile", (p, slice(1, n + 1))),
        ("h2ocan_profile", "dh2ocan_profile", "h2ocan_bef_profile", (p, slice(1, n + 1))),
        ("tleaf_leaf", "dtleaf_leaf", "tleaf_bef_leaf", (p, slice(1, n + 1), slice(None))),
        ("lwp_leaf", "dlwp_leaf", "lwp_bef_leaf", (p, slice(1, n + 1), slice(None))),
        ("tg_soil", "dtg_soil", "tg_bef_soil", (p,)),
    ):
        x_arr = getattr(mlcanopy_inst, state)
        dx_arr = getattr(mlcanopy_inst, delta)
        x_new, dx_new = _advance(getattr(mlcanopy_inst, bef)[sl], x_arr[sl], dx_arr[sl])
        dx_col = dx_arr[sl + (k,)]
        updates[state] = x_arr.at[sl].set(x_new)
        updates[delta] = dx_arr.at[sl + (k,)].set(jnp.where(do_update, dx_new, dx_col))

    # ------------------------------------------------------------------
    # Write all updated arrays back into the immutable state container
    # ------------------------------------------------------------------
    mlcanopy_inst = mlcanopy_inst._replace(**updates)

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Public: Butcher tableau initialisation
# ---------------------------------------------------------------------------


def RungeKuttaIni(opts: MLOptions) -> Tuple[Array, Array, Array]:
    """
    Initialise the Butcher tableau for the selected Runge-Kutta method.

    The method is chosen by ``opts.runge_kutta_type``. Unused tableau entries (upper-triangular part
    of ``a``) are set to ``spval`` matching the Fortran convention.

    Mirrors Fortran subroutine ``RungeKuttaIni`` (lines 142-240).

    References:
        Kutta, W. (1901). Beitrag zur näherungsweisen Integration totaler
        Differentialgleichungen. Z. Math. Phys., 46, 435-453.

        Butcher, J. C. (1996). A history of Runge-Kutta methods. Applied
        Numerical Mathematics, 20, 247-260.

        Ralston, A. (1962). Runge-Kutta methods with minimum error bounds.
        Mathematics of Computation, 16, 431-437.

    Returns:
        Tuple ``(a, b, c)`` where:

        - ``a``: Butcher tableau lower-triangular matrix,
          shape ``(nrk, nrk)``, dtype ``float64``.
          Fortran: ``real(r8), intent(out) :: a(nrk,nrk)``.
        - ``b``: Weight vector, shape ``(nrk,)``, dtype ``float64``.
          Fortran: ``real(r8), intent(out) :: b(nrk)``.
        - ``c``: Node vector, shape ``(nrk,)``, dtype ``float64``.
          Fortran: ``real(r8), intent(out) :: c(nrk)``.
    """
    nrk, runge_kutta_type = opts.nrk, opts.runge_kutta_type
    # Initialise with spval; specific entries overwritten below
    a = jnp.full((nrk, nrk), spval, dtype=jnp.float64)
    b = jnp.zeros(nrk, dtype=jnp.float64)
    c = jnp.zeros(nrk, dtype=jnp.float64)

    if runge_kutta_type == 21:
        # 2nd-order trapezoidal rule (Butcher 1996) — Fortran lines 176-184
        a = a.at[1, 0].set(1.0)

        b = b.at[0].set(1.0 / 2.0)
        b = b.at[1].set(1.0 / 2.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0)

    elif runge_kutta_type == 22:
        # 2nd-order midpoint rule (Butcher 1996) — Fortran lines 185-193
        a = a.at[1, 0].set(1.0 / 2.0)

        b = b.at[0].set(0.0)
        b = b.at[1].set(1.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)

    elif runge_kutta_type == 23:
        # 2nd-order Ralston (1962) — Fortran lines 194-202
        a = a.at[1, 0].set(2.0 / 3.0)

        b = b.at[0].set(1.0 / 4.0)
        b = b.at[1].set(3.0 / 4.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(2.0 / 3.0)

    elif runge_kutta_type == 31:
        # 3rd-order Heun's method (Butcher 1996) — Fortran lines 203-215
        a = a.at[1, 0].set(1.0 / 3.0)
        a = a.at[2, 0].set(0.0)
        a = a.at[2, 1].set(2.0 / 3.0)

        b = b.at[0].set(1.0 / 4.0)
        b = b.at[1].set(0.0)
        b = b.at[2].set(3.0 / 4.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 3.0)
        c = c.at[2].set(2.0 / 3.0)

    elif runge_kutta_type == 32:
        # 3rd-order Ralston (1962) — Fortran lines 216-228
        a = a.at[1, 0].set(1.0 / 2.0)
        a = a.at[2, 0].set(0.0)
        a = a.at[2, 1].set(3.0 / 4.0)

        b = b.at[0].set(2.0 / 9.0)
        b = b.at[1].set(3.0 / 9.0)
        b = b.at[2].set(4.0 / 9.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)
        c = c.at[2].set(3.0 / 4.0)

    elif runge_kutta_type == 33:
        # 3rd-order Kutta's method (Kutta 1901) — Fortran lines 229-241
        a = a.at[1, 0].set(1.0 / 2.0)
        a = a.at[2, 0].set(-1.0)
        a = a.at[2, 1].set(2.0)

        b = b.at[0].set(1.0 / 6.0)
        b = b.at[1].set(4.0 / 6.0)
        b = b.at[2].set(1.0 / 6.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)
        c = c.at[2].set(1.0)

    elif runge_kutta_type == 41:
        # 4th-order classical Kutta's method (Butcher 1996, Kutta 1901)
        # Fortran lines 242-260
        a = a.at[1, 0].set(1.0 / 2.0)
        a = a.at[2, 0].set(0.0)
        a = a.at[2, 1].set(1.0 / 2.0)
        a = a.at[3, 0].set(0.0)
        a = a.at[3, 1].set(0.0)
        a = a.at[3, 2].set(1.0)

        b = b.at[0].set(1.0 / 6.0)
        b = b.at[1].set(2.0 / 6.0)
        b = b.at[2].set(2.0 / 6.0)
        b = b.at[3].set(1.0 / 6.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)
        c = c.at[2].set(1.0 / 2.0)
        c = c.at[3].set(1.0)

    return a, b, c
