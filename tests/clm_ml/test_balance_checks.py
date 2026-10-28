"""Fortran's conservation checks (plan §6.8): recorded in the step instead of a host callback
inside compiled code, and checked on the host; a run stops when a tolerance is exceeded."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)  # before any model module is imported

from multilayer_canopy.MLCanopyFluxesType import (  # noqa: E402
    BALANCE_CHECKS,
    LONGWAVE,
    NITROGEN,
    check_balance,
    record_balance_error,
)


class _Inst:
    """The one field the helpers use (a full container needs the CLM initialization)."""

    def __init__(self, balance_error):
        self.balance_error = balance_error

    def _replace(self, **kw):
        return _Inst(kw["balance_error"])


def test_records_the_largest_error_per_check_and_patch():
    inst = _Inst(jnp.zeros((2, len(BALANCE_CHECKS))))
    inst = record_balance_error(inst, 1, LONGWAVE, jnp.array([1e-5, -3e-4]))
    inst = record_balance_error(inst, 1, LONGWAVE, 2e-4)  # smaller: maximum kept
    assert float(inst.balance_error[1, LONGWAVE]) == pytest.approx(3e-4)
    assert float(inst.balance_error.sum()) == pytest.approx(3e-4)


def test_old_containers_without_the_field_are_left_alone():
    inst = _Inst(None)
    assert record_balance_error(inst, 1, LONGWAVE, 1.0) is inst


def test_check_stops_above_the_tolerance_only():
    errors = jnp.zeros((5, 2, len(BALANCE_CHECKS)))  # e.g. (step, patch, check)
    check_balance(errors.at[3, 1, LONGWAVE].set(9e-4), "test")  # below 1e-3
    with pytest.raises(SystemExit):
        check_balance(errors.at[3, 1, NITROGEN].set(2e-6), "test")  # above 1e-6
