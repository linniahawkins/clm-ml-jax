"""
Canopy layer masks (plan Phase 5b): layer counts as data instead of Python loop bounds.

Every profile array has ``max_layers`` slots (indices 1..max_layers; index 0 is the ground
or unused, as in Fortran). A patch uses layers 1..``ncan`` (``ncan_canopy``): 1..``ntop``
within the canopy, ``ntop+1``..``ncan`` above it, and ``nbot``..``ntop`` hold leaves (some may
still have no plant area). Slots ``ncan+1``..``max_layers`` are padding.

Routines compute on every slot and keep results only where a mask is true, so one compiled
program serves patches with different layer counts. Inputs of padding slots are first
replaced by those of layer ``ncan`` (:func:`pad_with_top`): the top layer is always above the
canopy and leafless, so the padding computes a harmless leafless layer instead of arithmetic
on fill values (1e36), which would give infinities and NaN gradients even though the results
are discarded (a masked ``jnp.where`` differentiates both branches).

Not Fortran: Fortran loops ``do ic = 1, ncan(p)``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class Layers(NamedTuple):
    """Layer counts and masks of one patch; masks have one entry per slot 1..max_layers."""

    ncan: jnp.ndarray  # number of layers in use (int scalar)
    ntop: jnp.ndarray  # top canopy layer
    nbot: jnp.ndarray  # bottom canopy layer with leaves
    index: jnp.ndarray  # slot numbers 1..max_layers
    in_use: jnp.ndarray  # 1..ncan
    in_canopy: jnp.ndarray  # 1..ntop
    above_canopy: jnp.ndarray  # ntop+1..ncan
    leafy: jnp.ndarray  # nbot..ntop


def layers(mlcanopy_inst, p: int) -> Layers:
    """The :class:`Layers` of patch ``p`` (traced when the container is)."""
    n_slots = mlcanopy_inst.dpai_profile.shape[1] - 1
    index = jnp.arange(1, n_slots + 1)
    ncan = mlcanopy_inst.ncan_canopy[p]
    ntop = mlcanopy_inst.ntop_canopy[p]
    nbot = mlcanopy_inst.nbot_canopy[p]
    return Layers(
        ncan=ncan,
        ntop=ntop,
        nbot=nbot,
        index=index,
        in_use=index <= ncan,
        in_canopy=index <= ntop,
        above_canopy=(index > ntop) & (index <= ncan),
        leafy=(index >= nbot) & (index <= ntop),
    )


def expand(mask: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """``mask`` (layers,) broadcast against ``x`` (layers, ...)."""
    return mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))


def pad_with_top(lay: Layers, x: jnp.ndarray) -> jnp.ndarray:
    """``x`` (slots 1..max_layers on axis 0) with padding slots set to layer ``ncan``'s value."""
    top = jnp.take(x, lay.ncan - 1, axis=0)
    return jnp.where(expand(lay.in_use, x), x, top)


def keep(lay_mask: jnp.ndarray, new: jnp.ndarray, old: jnp.ndarray) -> jnp.ndarray:
    """``new`` where ``lay_mask`` is true, ``old`` elsewhere (layers on axis 0)."""
    return jnp.where(expand(lay_mask, new), new, old)


def restore_padding(lay: Layers, p: int, old, new, il=None):
    """``new`` (a container) with the padding slots of every layer field it changed taken
    back from ``old``: a routine that computed on all slots leaves padding untouched.
    ``il``: only that leaf index (sunlit or shaded) of ``(patch, layer, leaf)`` fields."""
    n_slots = lay.index.shape[0]
    changes = {}
    for name in new._fields:
        a, b = getattr(new, name), getattr(old, name)
        if a is b or a is None or a.ndim < 2 or a.shape[1] != n_slots + 1:
            continue
        sl = (p, slice(1, None)) + ((il,) if il is not None and a.ndim >= 3 else ())
        changes[name] = a.at[sl].set(keep(lay.in_use, a[sl], b[sl]))
    return new._replace(**changes)
