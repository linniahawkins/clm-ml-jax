"""Soil water: the vertical movement of water between soil levels (the Richards equation in
its moisture form, with CLM's adaptive sub-steps) and the Clapp and Hornberger (1978)
retention curve it uses.

Each function computes one soil column. Soil levels are 0-based from the top down (level
``k`` is Fortran's layer ``j = k + 1``); the column is solved down to the bedrock: the first
``nbedrock`` levels. Every level slot is computed and the levels below the bedrock are kept
out with masks, so ``nbedrock`` is data, not a shape. Lengths are in mm and fluxes in
mm s⁻¹ (kg m⁻² s⁻¹), except the levels' thickness and depth (m).

The settings are CLM's defaults: the infiltration is given at the surface (a flux
condition) and no water leaves through the bottom of the column (zero flux).

The retention curve works on scalars or on arrays of soil levels (elementwise). Its
relative saturation ``s`` is the caller's, already limited to CLM's bounds [0.01, 1]: with
``s`` bounded away from 0 the powers and the divisions by ``s`` are finite, and so are their
gradients, including the gradient with respect to the exponent ``bsw``.

Not in CLM-ml v2: these are translated from CLM itself (CTSM ``ctsm5.4.004``).

Fortran reference: CTSM ``SoilWaterMovementMod.F90``,
``SoilWaterRetentionCurveClappHornberg1978Mod.F90``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from clm_ml.processes.constants import denh2o
from clm_ml.processes.math import tridiag_pivoting

M_TO_MM = 1.0e3  # millimetres in a metre


def soil_hk(s: Array, imped: Array, hksat: Array, bsw: Array) -> tuple[Array, Array]:
    """Hydraulic conductivity (mm s⁻¹) and its derivative with respect to the relative
    saturation (mm s⁻¹).

    ``s``: relative saturation, in [0.01, 1]; ``imped``: ice impedance factor (1 without ice);
    ``hksat``: conductivity at saturation (mm s⁻¹); ``bsw``: Clapp-Hornberger *b*.

    The derivative is CLM's hand-coded one, which the Richards solver uses to build its
    matrix; it is part of the forward calculation.

    Fortran reference: SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_hk.
    """
    hk = imped * hksat * s ** (2.0 * bsw + 3.0)
    dhkds = (2.0 * bsw + 3.0) * hk / s
    return hk, dhkds


def soil_suction(s: Array, sucsat: Array, bsw: Array) -> tuple[Array, Array]:
    """Soil matric potential (mm, negative) and its derivative with respect to the relative
    saturation (mm).

    ``s``: relative saturation, in [0.01, 1]; ``sucsat``: suction at saturation (mm,
    positive); ``bsw``: Clapp-Hornberger *b*.

    Fortran reference: SoilWaterRetentionCurveClappHornberg1978Mod.F90::soil_suction.
    """
    smp = -sucsat * s ** (-bsw)
    dsmpds = -bsw * smp / s
    return smp, dsmpds


# ---------------------------------------------------------------------------
# The Richards equation: properties, fluxes, the linear system of a sub-step
# ---------------------------------------------------------------------------


def _level_below(x: Array) -> Array:
    """``x`` of the level below each level (the last level: its own)."""
    return jnp.concatenate([x[1:], x[-1:]])


def ice_impedance(icefrac: Array, e_ice) -> Array:
    """Factor (0 to 1) by which ice in the pores reduces the hydraulic conductivity.

    ``icefrac``: fraction of the pore space filled with ice; ``e_ice``: CLM's ice impedance
    parameter.

    Fortran reference: SoilWaterMovementMod.F90::IceImpedance.
    """
    return 10.0 ** (-e_ice * icefrac)


class HydraulicProperties(NamedTuple):
    hk: Array  # (n,) conductivity at the interface below each level (mm/s)
    smp: Array  # (n,) matric potential of each level (mm)
    dhkds: Array  # (n,) d(hk)/d(relative saturation at the interface) (mm/s)
    dsmpdw: Array  # (n,) d(smp)/d(volumetric liquid water of the level) (mm)


def hydraulic_properties(
    vwc_liq: Array,
    icefrac: Array,
    watsat: Array,
    hksat: Array,
    sucsat: Array,
    bsw: Array,
    e_ice,
    nbedrock,
) -> HydraulicProperties:
    """Hydraulic conductivity and matric potential of a column, with the derivatives the
    Richards solver builds its matrix from.

    The matric potential is that of each level. The conductivity is that of the interface
    between a level and the one below: computed at the mean relative saturation of the two,
    with the upper level's soil properties and the mean ice fraction (the lowest level above
    the bedrock: its own saturation and ice). Relative saturations are limited to
    [0.01, 1]. Levels below the bedrock hold 0.

    Args:
        vwc_liq: (n,) volumetric liquid water (m³ m⁻³).
        icefrac: (n,) fraction of the pore space filled with ice.
        watsat, hksat, sucsat, bsw: (n,) porosity, conductivity at saturation (mm s⁻¹),
            suction at saturation (mm), Clapp-Hornberger *b*.
        e_ice: ice impedance parameter.
        nbedrock: number of levels above the bedrock.

    Fortran reference: SoilWaterMovementMod.F90::compute_hydraulic_properties.
    """
    level = jnp.arange(vwc_liq.shape[0])
    active, bottom = level < nbedrock, level == nbedrock - 1

    s2 = jnp.maximum(0.01, jnp.minimum(vwc_liq / watsat, 1.0))
    s1 = jnp.where(bottom, s2, 0.5 * (s2 + _level_below(s2)))
    s1 = jnp.maximum(0.01, jnp.minimum(s1, 1.0))
    imped = ice_impedance(
        jnp.where(bottom, icefrac, 0.5 * (icefrac + _level_below(icefrac))), e_ice
    )
    hk, dhkds = soil_hk(s1, imped, hksat, bsw)
    smp, dsmpds = soil_suction(s2, sucsat, bsw)
    dsmpdw = dsmpds / watsat
    return HydraulicProperties(*(jnp.where(active, x, 0.0) for x in (hk, smp, dhkds, dsmpdw)))


class MoistureFluxes(NamedTuple):
    qin: Array  # (n,) flux into each level through its top (mm/s, positive downward)
    qout: Array  # (n,) flux out of each level through its bottom (mm/s)
    dqidw0: Array  # (n,) d(qin)/d(volumetric liquid water of the level above)
    dqidw1: Array  # (n,) d(qin)/d(volumetric liquid water of the level)
    dqodw1: Array  # (n,) d(qout)/d(volumetric liquid water of the level)
    dqodw2: Array  # (n,) d(qout)/d(volumetric liquid water of the level below)


def moisture_fluxes(
    properties: HydraulicProperties, qflx_infl, watsat: Array, z: Array, nbedrock
) -> MoistureFluxes:
    """Water fluxes through the top and bottom of each level (Darcy's law: the gradient of
    the matric potential plus gravity) and their derivatives with respect to the water of
    the levels on either side.

    The flux into the top level is the infiltration ``qflx_infl`` (mm s⁻¹; negative when
    evaporation exceeds the water reaching the soil); nothing leaves through the bottom of
    the lowest level above the bedrock. The derivatives are CLM's hand-coded ones: they
    form the matrix of the sub-step and are part of the forward calculation.

    ``z``: (n,) depth of each level's node (m).

    Fortran reference: SoilWaterMovementMod.F90::compute_moisture_fluxes_and_derivs.
    """
    hk, smp, dhkds, dsmpdw = properties
    interior = jnp.arange(z.shape[0]) < nbedrock - 1  # levels with a level below them

    # derivatives of the interface conductivity with respect to the water above and below
    dhkds1 = 0.5 * dhkds / watsat
    dhkds2 = 0.5 * dhkds / _level_below(watsat)
    num = _level_below(smp) - smp
    den = jnp.where(interior, M_TO_MM * (_level_below(z) - z), 1.0)
    qout = -hk * num / den + hk
    dqodw1 = (hk * dsmpdw - dhkds1 * num) / den + dhkds1
    dqodw2 = (-hk * _level_below(dsmpdw) - dhkds2 * num) / den + dhkds2
    qout, dqodw1, dqodw2 = (jnp.where(interior, x, 0.0) for x in (qout, dqodw1, dqodw2))

    def from_above(top, x):
        return jnp.concatenate([jnp.asarray(top, dtype=x.dtype)[None], x[:-1]])

    return MoistureFluxes(
        qin=from_above(qflx_infl, qout),
        qout=qout,
        dqidw0=from_above(0.0, dqodw1),
        dqidw1=from_above(0.0, dqodw2),
        dqodw1=dqodw1,
        dqodw2=dqodw2,
    )


def richards_system(
    fluxes: MoistureFluxes, sink: Array, dt_dz: Array
) -> tuple[Array, Array, Array, Array]:
    """The tridiagonal system of a sub-step, ``a, b, c, r``: its solution is the change of
    each level's volumetric liquid water, the fluxes being linearized about the water at the
    start of the sub-step.

    ``sink``: (n,) water taken from each level by roots (mm s⁻¹; 0 below the bedrock);
    ``dt_dz``: (n,) sub-step length over level thickness (s mm⁻¹). The rows of levels below
    the bedrock are the identity with a zero right-hand side.

    Fortran reference: SoilWaterMovementMod.F90::compute_RHS_moisture_form,
    compute_LHS_moisture_form.
    """
    qin, qout, dqidw0, dqidw1, dqodw1, dqodw2 = fluxes
    rmx = -(qin - qout - sink) * dt_dz
    amx = dqidw0 * dt_dz
    bmx = -1.0 - (-dqidw1 + dqodw1) * dt_dz
    cmx = -dqodw2 * dt_dz
    return amx, bmx, cmx, rmx


# ---------------------------------------------------------------------------
# The adaptive solver
# ---------------------------------------------------------------------------


def max_richards_trials(dtime: float, dtmin: float = 60.0) -> int:
    """The largest number of trial sub-steps the adaptive solver can take in a model step
    of ``dtime`` seconds (64 for 1800 s, 125 for 3600 s, with ``dtmin`` = 60 s).

    Every accepted sub-step but the last is at least ``dtmin`` long, so at most
    ``n = ceil(dtime/dtmin)`` are accepted. A rejection halves the sub-step and needs one
    longer than ``dtmin``: there can be ``ceil(log2(dtime/dtmin))`` from the start, plus one
    for each doubling, and a doubling follows an accepted sub-step that is not the last.
    """
    ratio = max(float(dtime) / float(dtmin), 1.0)
    accepted = math.ceil(ratio)
    return 2 * accepted - 1 + math.ceil(math.log2(ratio))


class SoilWaterMovement(NamedTuple):
    h2osoi_liq: Array  # (n,) liquid water at the end of the step (kg/m2)
    num_substeps: Array  # trial sub-steps taken, the rejected ones included
    hk: Array  # (n,) conductivity at the interface below each level (mm/s) *
    smp: Array  # (n,) matric potential (mm) *
    qin: Array  # (n,) flux into each level through its top (mm/s) *
    qout: Array  # (n,) flux out of each level through its bottom (mm/s) *
    # * at the start of the last sub-step, as CLM records them; 0 below the bedrock


def soil_water_movement(
    h2osoi_liq: Array,
    qflx_infl,
    qflx_rootsoi: Array,
    icefrac: Array,
    watsat: Array,
    hksat: Array,
    sucsat: Array,
    bsw: Array,
    dz: Array,
    z: Array,
    nbedrock,
    dtime,
    e_ice,
    *,
    fixed_trials: int | None = None,
    dtmin=60.0,
    xtol_upper=1.0e-1,
    xtol_lower=1.0e-2,
    verysmall=1.0e-8,
) -> SoilWaterMovement:
    """Move the soil water of one column over a model step: the Richards equation with
    infiltration at the surface, root uptake as a sink and no flow through the bottom.

    A sub-step is one linear solve (the fluxes linearized about the water at its start).
    Its error is estimated from the difference between the linearized net flux of each
    level and the net flux at the start. Above ``xtol_upper`` (mm) the sub-step is rejected
    and tried again at half the length (not below ``dtmin``); otherwise it is accepted, and
    the next one is twice as long if the error was below ``xtol_lower``. The first trial is
    the whole step. Water above saturation or below zero is not corrected here.

    Args:
        h2osoi_liq: (n,) liquid water of each level (kg m⁻²).
        qflx_infl: infiltration (mm s⁻¹; may be negative).
        qflx_rootsoi: (n,) root uptake of each level (mm s⁻¹).
        icefrac: (n,) fraction of the pore space filled with ice (it reduces the
            conductivity; the ice itself does not move).
        watsat, hksat, sucsat, bsw: (n,) porosity, conductivity at saturation (mm s⁻¹),
            suction at saturation (mm), Clapp-Hornberger *b*.
        dz, z: (n,) thickness and node depth of each level (m).
        nbedrock: number of levels above the bedrock, at least 2 (CLM's routine does not
            take a single level). The levels below keep their water.
        dtime: length of the model step (s).
        e_ice: ice impedance parameter.
        fixed_trials: ``None``: the loop ends when the step is covered (``lax.while_loop``;
            no reverse-mode gradient). An integer, at least
            :func:`max_richards_trials`: that many passes, of which those after the step is
            covered change nothing; the values are the same and reverse-mode gradients
            work. The gradient is that of the sequence of sub-steps taken: the decisions
            to accept, halve or double are not differentiated.
        dtmin: shortest sub-step that can still be halved (s).
        xtol_upper, xtol_lower: error above which a sub-step is halved, and below which
            the next is doubled (mm).
        verysmall: the step is covered when less than this remains (s).

    Not in CLM: ``fixed_trials``. Differs from CLM only in form: one loop with masks
    instead of a loop with exits.

    Fortran reference: SoilWaterMovementMod.F90::soilwater_moisture_form.
    """
    n = h2osoi_liq.shape[0]
    dtype = h2osoi_liq.dtype
    active = jnp.arange(n) < nbedrock
    dtime = jnp.asarray(dtime, dtype=dtype)
    sink = jnp.where(active, qflx_rootsoi, 0.0)
    zeros = jnp.zeros(n, dtype=dtype)

    def trial(state):
        liq, dtsub, dtdone, ntrial, _, _, _, _, _ = state
        vwc_liq = jnp.maximum(liq, 1.0e-6) / (dz * denh2o)
        dt_dz = dtsub / (M_TO_MM * dz)
        properties = hydraulic_properties(
            vwc_liq, icefrac, watsat, hksat, sucsat, bsw, e_ice, nbedrock
        )
        fluxes = moisture_fluxes(properties, qflx_infl, watsat, z, nbedrock)
        dwat = tridiag_pivoting(*richards_system(fluxes, sink, dt_dz))

        # error: the linearized net flux of each level against the net flux at the start
        flux_net0 = dwat / dt_dz
        flux_net1 = fluxes.qin - fluxes.qout - sink
        error = jnp.max(jnp.where(active, jnp.abs(flux_net1 - flux_net0) * dtsub * 0.5, 0.0))
        reject = (error > xtol_upper) & (dtsub > dtmin)

        # accepted: renew the water, then choose the next sub-step
        liq_new = liq + dwat * (M_TO_MM * dz)
        dtdone_new = dtdone + dtsub
        done = ~reject & (jnp.abs(dtime - dtdone_new) < verysmall)
        dtsub_next = jnp.where(error < xtol_lower, dtsub * 2.0, dtsub)
        dtsub_next = jnp.minimum(dtsub_next, dtime - dtdone_new)
        # (the last sub-step's length is kept once the step is covered, so that passes
        # after it divide by a positive number)
        dtsub_next = jnp.where(done, dtsub, dtsub_next)
        return (
            jnp.where(reject, liq, liq_new),
            jnp.where(reject, jnp.maximum(dtsub / 2.0, dtmin), dtsub_next),
            jnp.where(reject, dtdone, dtdone_new),
            ntrial + 1,
            done,
            properties.hk,
            properties.smp,
            fluxes.qin,
            fluxes.qout,
        )

    start = (
        h2osoi_liq,
        dtime,
        jnp.zeros((), dtype=dtype),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=bool),
        zeros,
        zeros,
        zeros,
        zeros,
    )
    if fixed_trials is None:
        end = jax.lax.while_loop(lambda state: ~state[4], trial, start)
    else:

        def one_pass(state, _):
            new = trial(state)
            return jax.tree.map(lambda old, x: jnp.where(state[4], old, x), state, new), None

        end, _ = jax.lax.scan(one_pass, start, None, length=fixed_trials)
    liq, _, _, ntrial, _, hk, smp, qin, qout = end
    return SoilWaterMovement(liq, ntrial, hk, smp, qin, qout)
