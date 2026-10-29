"""
JAX translation of MLLongwaveRadiationMod Fortran module.

Longwave radiation transfer through the multilayer canopy.
Provides one public driver (:func:`LongwaveRadiation`) and one
private solver (:func:`_Norman`).

Original Fortran module: MLLongwaveRadiationMod
Fortran lines 1-195

Differentiability notes
-----------------------
* All ``np.asarray()`` calls removed — JAX arrays used directly.
* All ``np.`` array operations replaced by ``jnp.``.
* ``float()`` wrappers on JAX scalars removed.
* The tridiagonal system is assembled as arrays over every layer slot and solved with
  ``MLMathToolsMod.tridiag_scan`` (``lax.scan`` loops), so the compiled program does not
  grow with ``max_layers`` (plan Phase 6).
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from clm_src_main.abortutils import endrun  # noqa: E402, F401
from clm_src_main.clm_varcon import sb  # noqa: E402, F401
from clm_src_main.clm_varctl import iulog  # noqa: E402, F401
from clm_src_main.decompMod import bounds_type  # noqa: E402, F401
from clm_src_main.PatchType import patch  # noqa: E402, F401
from multilayer_canopy.MLCanopyFluxesType import (  # noqa: E402
    LONGWAVE,
    mlcanopy_type,  # noqa: E402, F401
    record_balance_error,
)
from multilayer_canopy.MLCanopyParamsMod import CanopyParams  # noqa: E402
from multilayer_canopy.MLclm_varcon import emg  # noqa: E402, F401
from multilayer_canopy.MLclm_varctl import GridInfo, MLOptions  # noqa: E402
from multilayer_canopy.MLclm_varpar import isha, isun  # noqa: E402, F401
from multilayer_canopy.MLLayersMod import layers  # noqa: E402
from multilayer_canopy.MLMathToolsMod import tridiag, tridiag_scan  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def LongwaveRadiation(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    params: CanopyParams,
    grid: GridInfo,
    *,
    opts: MLOptions,
) -> mlcanopy_type:
    """
    Longwave radiation transfer through the multilayer canopy.

    Mirrors Fortran subroutine ``LongwaveRadiation`` (lines 30-45).

    Dispatches to :func:`_Norman` when ``longwave_type == 1``; any
    other value is fatal.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Multilayer canopy container; longwave flux
            fields are updated by the solver.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    if opts.longwave_type == 1:
        return _Norman(bounds, num_filter, filter_patch, mlcanopy_inst, params, grid=grid)
    else:
        endrun(msg=" ERROR: LongwaveRadiation: longwave_type not valid")
        return mlcanopy_inst  # Unreachable; satisfies type checker


# ---------------------------------------------------------------------------
# Private: Norman (1979) longwave radiative transfer
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0,))
def _Norman_patch(
    p: int,
    mlcanopy_inst: mlcanopy_type,
    params: CanopyParams,
) -> mlcanopy_type:
    """Norman longwave solver for one patch, for any layer counts (plan Phase 5b).

    The leafy layers ``nbot..ntop`` are data (``MLLayersMod``). The tridiagonal system always
    has ``2 * max_layers + 2`` rows: the real rows first, in Fortran order (soil, then two
    rows per leafy layer), the sky row right after them, then identity rows (x = 0) for the
    unused leafy-layer slots. The sky row has no upper coupling, so the identity rows do not
    change the real solution (each real row is computed as before).
    """
    nlevmlcan = mlcanopy_inst.dpai_profile.shape[1] - 1  # layer slots (model.yaml max_layers)
    lay = layers(mlcanopy_inst, p)
    ntop, nbot = lay.ntop, lay.nbot
    n_leafy = ntop - nbot + 1  # leafy layers (data)
    em_leaf = params.leaf_emissivity
    rho = 1.0 - em_leaf
    tau = 0.0

    _tleaf_sun = mlcanopy_inst.tleaf_leaf[p, :, isun]
    _tleaf_sha = mlcanopy_inst.tleaf_leaf[p, :, isha]
    _fracsun = mlcanopy_inst.fracsun_profile[p]
    _td = mlcanopy_inst.td_profile[p]
    _dpai = mlcanopy_inst.dpai_profile[p]
    _lwsky_p = mlcanopy_inst.lwsky_forcing[p]
    _tg_p = mlcanopy_inst.tg_soil[p]

    # Leafy-layer slot j = 0..max_layers-1 is layer ic = nbot + j (valid for j < n_leafy);
    # unused slots repeat layer ntop's values (finite), and their results are discarded.
    j = jnp.arange(nlevmlcan)
    valid = j < n_leafy
    ics = jnp.minimum(nbot + j, ntop)

    # Layer emission — JAX vectorised
    lw_sun = em_leaf * sb * _tleaf_sun[ics] ** 4
    lw_sha = em_leaf * sb * _tleaf_sha[ics] ** 4
    fs_arr = _fracsun[ics]
    src = (lw_sun * fs_arr + lw_sha * (1.0 - fs_arr)) * (1.0 - _td[ics])  # lw_source(ic)

    # Row coefficients per leafy layer: "a" rows (-aic, 1, -bic) and "f" rows (-eic, 1, -fic)
    td = _td[ics]
    refld = (1.0 - td) * rho
    trand = (1.0 - td) * tau + td
    aic = refld - trand * trand / refld
    bic = trand / refld
    fic = refld - trand * trand / refld
    eic = trand / refld

    m_max = 2 * nlevmlcan + 2
    sky_row = 2 * n_leafy + 2  # 1-based
    # Rows as arrays (0-based index = row - 1): soil row, then layer j's "a" row (row 2 + 2j)
    # and "f" row (row 3 + 2j); unused slots are identity rows (a = c = d = 0, b = 1)
    z = jnp.zeros(nlevmlcan)
    atri = jnp.zeros(m_max).at[1 : 2 * nlevmlcan + 1 : 2].set(jnp.where(valid, -aic, z))
    atri = atri.at[2 : 2 * nlevmlcan + 2 : 2].set(jnp.where(valid, -eic, z))
    ctri = jnp.zeros(m_max).at[0].set(-(1.0 - emg))
    ctri = ctri.at[1 : 2 * nlevmlcan + 1 : 2].set(jnp.where(valid, -bic, z))
    ctri = ctri.at[2 : 2 * nlevmlcan + 2 : 2].set(jnp.where(valid, -fic, z))
    dtri = jnp.zeros(m_max).at[0].set(emg * sb * _tg_p**4)
    dtri = dtri.at[1 : 2 * nlevmlcan + 1 : 2].set(jnp.where(valid, (1.0 - bic) * src, z))
    dtri = dtri.at[2 : 2 * nlevmlcan + 2 : 2].set(jnp.where(valid, (1.0 - eic) * src, z))
    btri = jnp.ones(m_max)
    # Sky row right after the last real row (overrides that slot's identity row)
    is_sky = jnp.arange(1, m_max + 1) == sky_row
    atri = jnp.where(is_sky, 0.0, atri)
    ctri = jnp.where(is_sky, 0.0, ctri)
    dtri = jnp.where(is_sky, _lwsky_p, dtri)

    u = tridiag_scan(atri, btri, ctri, dtri)  # rows 1..m_max

    # Unknowns: up(0), dn(0), then up(ic_j), dn(ic_j) for each leafy layer
    ics_out = jnp.where(valid, ics, nlevmlcan + 1)  # unused slots write a dropped entry
    lwupw_new = jnp.zeros(nlevmlcan + 2).at[0].set(u[0]).at[ics_out].set(u[2::2][:nlevmlcan])
    lwdwn_new = jnp.zeros(nlevmlcan + 2).at[0].set(u[1]).at[ics_out].set(u[3::2][:nlevmlcan])

    _lwsoi_p = lwdwn_new[0] - lwupw_new[0]

    icm1s = jnp.where(ics == nbot, 0, ics - 1)
    lwabs_arr = em_leaf * (lwdwn_new[ics] + lwupw_new[icm1s]) * (1.0 - _td[ics]) - 2.0 * src
    lwabs_arr = jnp.where(valid, lwabs_arr, 0.0)
    dpai_ics = jnp.where(valid, _dpai[ics], 1.0)
    lwleaf_arr = lwabs_arr / dpai_ics
    _lwveg_p = jnp.sum(lwabs_arr)

    lwleaf_new = jnp.zeros(nlevmlcan + 2).at[ics_out].set(lwleaf_arr)

    _lwup_p = lwupw_new[ntop]

    # Conservation check — Fortran stops here; recorded and checked on the host
    err = (_lwsky_p - _lwup_p) - (_lwveg_p + _lwsoi_p)
    mlcanopy_inst = record_balance_error(mlcanopy_inst, p, LONGWAVE, err)

    # Levels 0..ntop are written (Fortran: lwupw(0:ntop) etc.); the rest is kept
    upto_top = jnp.arange(nlevmlcan + 1) <= ntop

    def _set(arr, new):
        return arr.at[p].set(jnp.where(upto_top, new[: nlevmlcan + 1], arr[p]))

    lwleaf = mlcanopy_inst.lwleaf_leaf
    return mlcanopy_inst._replace(
        lwup_canopy=mlcanopy_inst.lwup_canopy.at[p].set(_lwup_p),
        lwveg_canopy=mlcanopy_inst.lwveg_canopy.at[p].set(_lwveg_p),
        lwsoi_soil=mlcanopy_inst.lwsoi_soil.at[p].set(_lwsoi_p),
        lwleaf_leaf=lwleaf.at[p, :, isun]
        .set(jnp.where(upto_top, lwleaf_new[: nlevmlcan + 1], lwleaf[p, :, isun]))
        .at[p, :, isha]
        .set(jnp.where(upto_top, lwleaf_new[: nlevmlcan + 1], lwleaf[p, :, isha])),
        lwupw_profile=_set(mlcanopy_inst.lwupw_profile, lwupw_new),
        lwdwn_profile=_set(mlcanopy_inst.lwdwn_profile, lwdwn_new),
    )


def _Norman(
    bounds: bounds_type,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    params: CanopyParams,
    grid: GridInfo,
) -> mlcanopy_type:
    """
    Norman (1979) tridiagonal longwave radiative transfer.

    Mirrors Fortran subroutine ``Norman`` (private, lines 47-195).

    Reference: Bonan (2019) *Climate Change and Terrestrial Ecosystem
    Modeling*, Chapter 14.

    The algorithm mirrors the solar :func:`MLSolarRadiationMod._Norman`
    structure but operates on a single spectral band (no waveband loop)
    and includes leaf thermal emission as an additional source term.

    **Optical properties** (Fortran lines 97-101):

    .. code-block:: none

        omega = 1 - emleaf(pft)    [scattering coefficient]
        rho   = omega              [intercepted radiation fully reflected]
        tau   = 0                  [no transmission of intercepted radiation]

    **Layer emission** (Fortran lines 103-107):

    .. code-block:: none

        lw_source(ic) = [emleaf*sb*T_sun^4 * fracsun
                         + emleaf*sb*T_sha^4 * (1-fracsun)]
                        * (1 - td(ic))

    **Tridiagonal system** (Fortran lines 109-158): same equation
    ordering as the solar Norman solver — soil upward (m=1), soil
    downward (m=2), then for ``ic = nbot, ..., ntop-1`` upward (odd)
    and downward (even), finally top-layer upward and downward — with
    leaf emission entering the right-hand-side ``dtri`` terms.

    **Leaf absorption** (Fortran lines 170-181): absorbed flux per
    unit leaf area is the same for sunlit and shaded fractions:

    .. code-block:: none

        lwabs = emleaf * (lwdwn(ic) + lwupw(icm1)) * (1-td(ic))
                - 2 * lw_source(ic)
        lwleaf(ic, isun) = lwleaf(ic, isha) = lwabs / dpai(ic)

    Energy conservation is checked to within 1e-3 W/m2.

    Args:
        bounds: Decomposition bounds.
        num_filter: Number of patches.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; the following output fields
            are updated: ``lwup_canopy``, ``lwveg_canopy``,
            ``lwsoi_soil``, ``lwleaf_leaf``, ``lwupw_profile``,
            ``lwdwn_profile``.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    """Dispatch to JIT'd per-patch Norman kernel.

    ntop and nbot are extracted concretely here (outside JIT) and passed
    as static arguments to ``_Norman_patch``, enabling the Python loop
    bounds and ``jnp.arange(nbot, ntop+1)`` inside the kernel.
    """
    for fp in range(num_filter):
        mlcanopy_inst = _Norman_patch(grid.p, mlcanopy_inst, params)
    return mlcanopy_inst
