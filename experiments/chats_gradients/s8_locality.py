"""
Step 8: the gradient is local. How far from the default does the derivative predict the
model's response, and where it does not, why (README, step 8).

    oat    each parameter on its own at a ladder of relative changes (LADDER: +-1, 3, 10,
           25, 50 % of its value, capped at the physical limits LIMITS), next to the
           derivative at the default. Results: ``locality/oat_<days>d_<first>.npz``.
    sweep  one parameter across its physical range (PHYSICAL): the window quantities, their
           derivative and the regime variables (LABELS) at every point, the default among
           them. Results: ``locality/sweep_<param>_<days>d_<first>.npz``.
    grid   two parameters on a grid over their physical ranges: the window quantities and
           the regime variables at every point; at the default the derivative and the second
           derivatives (central differences of forward-mode derivatives, 1 % of the range).
           Results: ``locality/grid_<a>__<b>_<days>d_<first>.npz``.

Derivatives are the default (stabilized) water-stress derivative (maintainer, 2026-11-17).

    JAX_PLATFORMS=cpu python s8_locality.py oat --days 1 --first 8
    JAX_PLATFORMS=cpu python s8_locality.py sweep --param psi50_gs --days 31 --points 101
    JAX_PLATFORMS=cpu python s8_locality.py grid --pair iota,psi50_gs --days 31 --points 21
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from harness import _STREAM, N_LOSS, RESULTS, VARS, Case
from jax.flatten_util import ravel_pytree

from clm_ml.output_streams import StreamSpec

LADDER = (0.01, 0.03, 0.10, 0.25, 0.50)  # relative changes of the one-at-a-time ladder
# physical ranges (maintainer, 2026-11-17; README step 8): the CLM-ml PFT tables
# (compsets/params) and the CHATS values; shape_gs has no source: +-50 %
PHYSICAL = {
    "vcmax25": (40.0, 150.0),  # PFT tables 39-101; CHATS 125 (measured)
    "iota": (250.0, 1500.0),  # PFT tables 500-1500; CHATS 375
    "psi50_gs": (-2.0, -1.0),  # CHATS -1.6; to -2 only (maintainer; CLM-ml default -2.3)
    "plant_conductance": (2.0, 14.0),  # CLM-ml default 4; CHATS 7
    "leaf_reflectance[1]": (0.28, 0.55),  # PFT tables 0.28-0.46; CHATS 0.42 (+0.43 < 1)
    "shape_gs": (20.0, 60.0),  # 40 everywhere
}
# what the regimes are read from, at every step
LABELS = ("solar_zenith_angle", "top_leaf_water_potential", "mid_leaf_water_potential",
          "fraction_at_minimum_leaf_water_potential", "soil_water_stress",
          "canopy_stomatal_conductance", "canopy_evaporation_latent_heat_flux", "rsl_beta")  # fmt: skip
ALBEDO_MAX = 0.99  # reflectance + transmittance of a band stays below this


def limits(c: Case):
    """Physical limits of every parameter (maintainer, 2026-11-17: cap where implausible):
    emissivity 0.999 (1 gives NaN, O83), clumping 1, the leaf angle index within the
    model's own clip (-0.4, 0.6; beyond it the derivative is zero by construction), and
    reflectance + transmittance of a band below ALBEDO_MAX, the partner at its default."""
    lo, hi = np.full(c.x0.size, -np.inf), np.full(c.x0.size, np.inf)
    name = {n: i for i, n in enumerate(c.names)}
    hi[name["leaf_emissivity"]] = 0.999
    hi[name["clumping_index"]] = 1.0
    lo[name["leaf_angle_index"]], hi[name["leaf_angle_index"]] = -0.4, 0.6
    for organ in ("leaf", "stem"):
        for b in (0, 1):
            r, t = name[f"{organ}_reflectance[{b}]"], name[f"{organ}_transmittance[{b}]"]
            hi[r] = ALBEDO_MAX - c.x0[t]
            hi[t] = ALBEDO_MAX - c.x0[r]
            lo[r] = lo[t] = 0.0
    return lo, hi


def dquantities_at(w, xs, dxs):
    """The window quantities and their derivatives along ``dxs``, each at its own parameter
    vector ``xs`` (the loss's chain rule needs the outputs at that point)."""
    y = w.outputs(xs)
    dy = w.forward_mode(dxs=dxs, xs=xs)
    weight = 2 * (y[:, :N_LOSS] - w.obs[:N_LOSS]) * w.inv_var[:N_LOSS, None]
    dloss = (dy[:, :N_LOSS] * weight).sum((-1, -2)) / N_LOSS / w.steps
    return w.quantities_of(y), np.concatenate([dloss[:, None], dy.mean(-1)], axis=1)


def labelled(c: Case, w, xs):
    """The window quantities and the regime variables at every step, (point, label, step),
    for a batch of parameter vectors."""
    stream = StreamSpec("l", "step", "mean", LABELS)

    def f(s, x, xg, xp):
        new, out = c.model.run_chunk(
            c.unravel_state(s), (xg, xp), 1, c.unravel_params(x), streams=[_STREAM, stream]
        )
        y = jnp.stack([out["s"][v][:, 0] for v in VARS])
        # canopy_stomatal_conductance is (canopy, sunlit, shaded): the canopy
        lab = [out["l"][v][:, 0].reshape(y.shape[1], -1)[:, 0] for v in LABELS]
        return ravel_pytree(new)[0], y, jnp.stack(lab)

    # one plain call first: it fills the model's host caches for these streams (as Case does)
    c.model.run_chunk(c.model.initial_state(), c.forcing(0), 1, streams=[_STREAM, stream])
    progs, ys, labs = {}, [], []
    for a in range(0, len(xs), 64):
        x = jnp.asarray(xs[a : a + 64])
        s = jnp.broadcast_to(w.s0, (x.shape[0],) + w.s0.shape)
        yk, lk = [], []
        for k in w.ks:
            n = c.chunks[k][1]
            if n not in progs:
                progs[n] = jax.jit(jax.vmap(f, in_axes=(0, 0, None, None)))
            s, y, lab = progs[n](s, x, *c.forcing(k))
            yk.append(np.asarray(y))
            lk.append(np.asarray(lab, dtype=np.float32))
        ys.append(np.concatenate(yk, axis=2))
        labs.append(np.concatenate(lk, axis=2))
    return w.quantities_of(np.concatenate(ys)), np.concatenate(labs)


def physical(c: Case, p: str):
    lo, hi = PHYSICAL[p]
    i = c.names.index(p)
    if not lo < c.x0[i] < hi:
        raise ValueError(f"{p}: the default {c.x0[i]} is outside its range {lo, hi}")
    return i, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oat", "sweep", "grid"])
    ap.add_argument("--case", default="may2007")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--param", default="psi50_gs")
    ap.add_argument("--pair", default="iota,psi50_gs")
    ap.add_argument("--points", type=int, default=101, help="per axis")
    args = ap.parse_args()

    c = Case(args.case)
    w = c.window(args.first, args.days)
    idx = np.flatnonzero(c.active)
    tag = f"{args.days}d_{args.first:02d}"
    out = RESULTS / "locality"
    out.mkdir(parents=True, exist_ok=True)
    common = dict(names=np.array(c.names), x0=c.x0, x_true=c.x_true, vars=np.array(VARS),
                  labels=np.array(LABELS), days=args.days, first=args.first)  # fmt: skip

    if args.mode == "oat":
        lo, hi = limits(c)
        ladder = np.array(LADDER)
        # (side, step, parameter): the parameter vector with one element changed
        xv = np.empty((2, len(ladder), len(idx)))
        for s, sign in enumerate((-1, 1)):
            xv[s] = np.clip(
                c.x0[idx] + sign * ladder[:, None] * np.abs(c.x0[idx]), lo[idx], hi[idx]
            )
        xs = np.tile(c.x0, (xv.size, 1))
        xs[np.arange(xv.size), np.tile(idx, 2 * len(ladder))] = xv.ravel()
        q = w.quantities(xs).reshape(xv.shape + (-1,))
        dq = w.dquantities_of(w.forward_mode(np.eye(c.x0.size)[idx]))  # (parameter, quantity)
        f = out / f"oat_{tag}.npz"
        np.savez(f, **common, idx=idx, ladder=ladder, x=xv, q=q, q0=w.quantities_of(w.y0),
                 dq=dq)  # fmt: skip
        print("saved", f, "NaN:", int(np.isnan(q).sum()), int(np.isnan(dq).sum()))
    elif args.mode == "sweep":
        i, lo, hi = physical(c, args.param)
        values = np.unique(np.append(np.linspace(lo, hi, args.points), c.x0[i]))
        xs = np.tile(c.x0, (len(values), 1))
        xs[:, i] = values
        q, lab = labelled(c, w, xs)
        _, dq = dquantities_at(w, xs, np.tile(np.eye(c.x0.size)[i], (len(values), 1)))
        f = out / f"sweep_{args.param}_{tag}.npz"
        np.savez(f, **common, param=args.param, i=i, values=values, q=q, dq=dq, lab=lab)
        print("saved", f, "NaN:", int(np.isnan(q).sum()), int(np.isnan(dq).sum()))
    else:
        pa, pb = args.pair.split(",")
        (ia, la, ha), (ib, lb, hb) = physical(c, pa), physical(c, pb)
        va, vb = np.linspace(la, ha, args.points), np.linspace(lb, hb, args.points)
        A, B = np.meshgrid(va, vb, indexing="ij")
        xs = np.tile(c.x0, (A.size, 1))
        xs[:, ia], xs[:, ib] = A.ravel(), B.ravel()
        q, lab = labelled(c, w, xs)
        # at the default: derivatives along a and b, and at the default +- h along each
        e = np.eye(c.x0.size)[[ia, ib]]
        h = 0.01 * np.array([ha - la, hb - lb])
        pts = [c.x0] + [c.x0 + sg * h[j] * e[j] for j in (0, 1) for sg in (1, -1)]
        xs2 = np.repeat(np.array(pts), 2, axis=0)
        _, dq2 = dquantities_at(w, xs2, np.tile(e, (len(pts), 1)))
        dq2 = dq2.reshape(len(pts), 2, -1)  # (point, direction, quantity)
        grad = dq2[0]
        hess = np.stack([(dq2[1 + 2 * j] - dq2[2 + 2 * j]) / (2 * h[j]) for j in (0, 1)])
        f = out / f"grid_{pa}__{pb}_{tag}.npz"
        np.savez(f, **common, pair=np.array([pa, pb]), ia=ia, ib=ib, va=va, vb=vb,
                 q=q.reshape(A.shape + (-1,)), lab=lab.reshape(A.shape + lab.shape[1:]),
                 q0=w.quantities_of(w.y0), grad=grad, hess=hess, h=h)  # fmt: skip
        print("saved", f, "NaN:", int(np.isnan(q).sum()), "hessian asymmetry",
              np.abs(hess[0, 1] - hess[1, 0]).max() / np.abs(hess).max())  # fmt: skip


if __name__ == "__main__":
    main()
