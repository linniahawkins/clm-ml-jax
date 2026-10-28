"""
Model state: the prognostic variables carried from one time step to the next.

Plan Phase 3e. A time step is ``state -> state`` given forcing and parameters; everything else
the physics uses is either fixed structure (the layer grid, set once at initialization),
forcing for the step, or a diagnostic recomputed every step. The fields below were found by
setting each field of the legacy containers to NaN before a step and seeing whether the NaN
reaches the results (docs/design/numerics_and_regression.md, §4.4 for one finding).

The arrays keep the legacy layout for now: patch first, then the 1-based level axis padded to
``nlevmlcan + 1`` (index 0 unused for canopy layers), then sunlit/shaded leaves (index 0
unused). Phase 5 moves to ``(patch, max_layers)`` with masks.

All fields are JAX arrays (a pytree), so ``jax.grad`` can differentiate with respect to them.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class CanopyState(NamedTuple):
    """Per patch: canopy air, leaves and ground surface."""

    air_temperature: Array  # (np, nlev) K
    vapor_pressure: Array  # (np, nlev) Pa
    co2: Array  # (np, nlev) umol mol-1
    leaf_temperature: Array  # (np, nlev, nleaf + 1) K
    leaf_water_potential: Array  # (np, nlev, nleaf + 1) MPa
    intercepted_water: Array  # (np, nlev) kg m-2
    ground_temperature: Array  # (np,) K


class SoilState(NamedTuple):
    """Per soil column."""

    temperature: Array  # (nc, nlev_soil) K


class State(NamedTuple):
    canopy: CanopyState
    soil: SoilState
