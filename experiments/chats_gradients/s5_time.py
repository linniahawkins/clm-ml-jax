"""
Step 5: time.

    horizon     one long window (default: April and May, 61 days). Forward mode carries the
                derivative of every output at every step with respect to each parameter and
                to one direction per group of state variables at the start; finite
                differences of the same; reverse mode gives the loss's gradient with respect
                to every start-state value for a set of horizons. Everything for "how does
                the gradient change with the horizon" is then arithmetic on these arrays.
                Results: ``horizon/<case>.npz``.
    jacobian    the one-step map of the state, state(t) -> state(t+1), at every step of a
                day: its Jacobian (forward mode), and the dot-product test
                <w, J v> = <J^T w, v> between forward and reverse mode with random v and w.
                Results: ``jacobian/day_<k>.npz``.
    checkpoint  the reverse-mode gradient of a two-day loss with the checkpoints of the
                step and of the sub-step on and off: they must agree.
                Results: ``checkpoint.npz``.
    twin        the model's own sensitivity to round-off: the same days run from start states
                whose leaf temperatures differ by 1e-12, 1e-9 and 1e-6 (relative); how far
                the outputs drift apart, step by step. Results: ``twin/<case>_<first>_<n>.npz``.

    JAX_PLATFORMS=cpu python s5_time.py horizon [--case aprmay2007]
    JAX_PLATFORMS=cpu python s5_time.py jacobian --day 0
    JAX_PLATFORMS=cpu python s5_time.py checkpoint
    JAX_PLATFORMS=cpu python s5_time.py twin --day 4 --days 6

``--set FILE.KEY=VALUE`` (jacobian, twin) changes a setting; ``--tag`` is added to the
result's file name.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from harness import RESULTS, VARS, Case, state_groups

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["horizon", "jacobian", "checkpoint", "twin"])
ap.add_argument("--case", default="")
ap.add_argument("--day", type=int, default=0)
ap.add_argument("--days", type=int, default=0, help="horizon: window length (default: the run)")
ap.add_argument("--set", action="append", default=[], metavar="FILE.KEY=VALUE")
ap.add_argument("--tag", default="")
args = ap.parse_args()
TAG = f"_{args.tag}" if args.tag else ""

# a perturbation of sensible size per group of state variables (for finite differences)
STATE_SCALE = {"air_temperature": 1.0, "vapor_pressure": 100.0, "co2": 10.0,
               "leaf_temperature": 1.0, "leaf_water_potential": 0.1, "ground_temperature": 1.0,
               "acclimation_temperature": 1.0, "soil_temperature top (0-4)": 1.0,
               "soil_temperature mid (5-11)": 1.0, "soil_temperature deep (12-24)": 1.0,
               "intercepted_water": 1e-3}  # fmt: skip

if args.mode == "horizon":
    c = Case(args.case or "aprmay2007")
    n = args.days or len(c.chunks)
    w = c.window(0, n)
    idx = np.flatnonzero(c.active)
    groups = state_groups(c)
    gnames = [g for g in STATE_SCALE if g in groups]
    m = len(idx) + len(gnames)
    dxs, dss = np.zeros((m, c.x0.size)), np.zeros((m, w.s0.size))
    dxs[np.arange(len(idx)), idx] = 1.0
    psi = np.asarray(w.s0)
    for a, g in enumerate(gnames):
        sel = groups[g]
        if g == "leaf_water_potential":  # only the layers with leaves (the others hold 0)
            sel = sel[psi[sel] != 0]
        dss[len(idx) + a, sel] = 1.0
    ad = w.forward_mode(dxs, dss)  # (direction, variable, step)
    print("forward mode done", ad.shape, flush=True)
    hs_p, hs_s = (1e-3, 1e-5, 1e-7), (1e-1, 1e-3)
    fd_p = w.finite_differences(hs_p)[:, idx]  # (h, parameter, variable, step)
    print("parameter finite differences done", flush=True)
    fd_s = np.full((len(hs_s), len(gnames)) + w.y0.shape, np.nan)
    for j, h in enumerate(hs_s):
        up = np.tile(np.asarray(w.s0), (len(gnames), 1))
        dn = up.copy()
        for a, g in enumerate(gnames):
            e = h * STATE_SCALE[g]
            up[a] += e * dss[len(idx) + a]
            dn[a] -= e * dss[len(idx) + a]
            if g == "intercepted_water":  # cannot be negative: a one-sided difference
                dn[a] = np.asarray(w.s0)
        xs = np.tile(c.x0, (len(gnames), 1))
        y_up, y_dn = w.outputs(xs, up), w.outputs(xs, dn)
        for a, g in enumerate(gnames):
            e = h * STATE_SCALE[g] * (1 if g == "intercepted_water" else 2)
            fd_s[j, a] = (y_up[a] - y_dn[a]) / e
    print("state finite differences done", flush=True)
    # reverse mode: the loss of the first T days differentiated by every start-state value
    horizons = [t for t in (1, 2, 3, 5, 10, 20, 31, 45, 61) if t <= n]
    d_state, d_params = [], []
    for t in horizons:
        gx, gs = c.window(0, t).reverse_mode(0)
        d_params.append(gx[idx])
        d_state.append(gs)
    out = RESULTS / "horizon" / f"{c.name}_{n}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(c.names)[idx], groups=np.array(gnames), vars=np.array(VARS),
             y0=w.y0, obs=w.obs, ad=ad, fd_params=fd_p, hs_params=np.array(hs_p), fd_state=fd_s,
             hs_state=np.array(hs_s), horizons=np.array(horizons), d_state=np.array(d_state),
             d_params=np.array(d_params), state_names=np.array(c.state_names),
             state_in_use=c.state_in_use(), x0=c.x0[idx], lo=c.lo[idx], hi=c.hi[idx],
             chunk_steps=np.array([s for _, s in c.chunks[:n]]), start=str(c.date(0)))  # fmt: skip
    print("saved", out)

elif args.mode == "jacobian":
    c = Case(args.case or "may2007", tuple(args.set))
    k, x0 = args.day, jnp.asarray(c.x0)
    forcing, n = c.forcing(k), c.chunks[k][1]
    use = np.flatnonzero(c.state_in_use())
    basis = np.eye(c.state_in_use().size)[use]

    def step(s, xg, xp):
        return c.traj(s, x0, xg, xp)[0]

    one = jax.jit(step)
    jac_part = jax.jit(lambda s, xg, xp, vs: jax.vmap(
        lambda v: jax.jvp(lambda a: step(a, xg, xp), (s,), (v,))[1])(vs))  # fmt: skip

    def jac(s, xg, xp, part=66):  # in parts: the tangents of a whole step take memory
        rows = [jac_part(s, xg, xp, jnp.asarray(basis[a : a + part]))
                for a in range(0, len(basis), part)]  # fmt: skip
        return np.concatenate([np.asarray(r) for r in rows])

    vjp = jax.jit(lambda s, xg, xp, wv: jax.vjp(lambda a: step(a, xg, xp), s)[1](wv)[0])
    rng = np.random.default_rng(0)
    s, jacs, dots = c.state(k), [], []
    for t in range(n):
        f = jax.tree.map(lambda a: a[t : t + 1], forcing)
        J = np.asarray(jac(s, *f))[:, use].T  # [output, input]
        v, wv = rng.standard_normal(len(use)), rng.standard_normal(len(use))
        w_full = np.zeros(c.state_in_use().size)
        w_full[use] = wv
        lhs = wv @ (J @ v)
        rhs = np.asarray(vjp(s, *f, jnp.asarray(w_full)))[use] @ v
        dots.append((lhs, rhs))
        jacs.append(J)
        s = one(s, *f)
    end = np.asarray(c.state(k + 1))[use] if k + 1 < len(c.chunks) else None
    diff = None if end is None else np.abs(np.asarray(s)[use] - end)
    gap = np.nan if end is None else float(diff.max())
    if end is not None:
        print("largest difference in", np.array(c.state_names)[use][diff.argmax()])
    dots = np.array(dots)
    print("dot-product test, worst relative difference:",
          np.abs(dots[:, 0] - dots[:, 1]).max() / np.abs(dots).max(),
          "; single steps against the chunk, largest state difference:", gap)  # fmt: skip
    out = RESULTS / "jacobian" / f"day_{k:02d}{TAG}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    lab = c.labels(k)
    np.savez(
        out,
        jac=np.array(jacs),
        dots=dots,
        state_names=np.array(c.state_names)[use],
        chunk_gap=gap,
        start=str(c.date(k)),
        **{f"label_{a}": b for a, b in lab.items()},
    )
    print("saved", out)

elif args.mode == "twin":
    c = Case(args.case or "may2007", tuple(args.set))
    n = args.days or 5
    w = c.window(args.day, n)
    sizes = np.array([0.0, 1e-12, 1e-9, 1e-6])
    s0 = np.tile(np.asarray(w.s0), (len(sizes), 1))
    sel = state_groups(c)["leaf_temperature"]
    s0[:, sel] *= 1 + sizes[:, None]
    y = w.outputs(np.tile(c.x0, (len(sizes), 1)), s0)  # (twin, variable, step)
    gap = np.abs(y[1:] - y[:1])
    for a, e in enumerate(sizes[1:]):
        print(f"twin {e:g}: largest difference by output", dict(zip(VARS, gap[a].max(axis=1))))
    out = RESULTS / "twin" / f"{c.name}_{args.day:02d}_{n:02d}{TAG}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, sizes=sizes, y=y, vars=np.array(VARS), start=str(c.date(args.day)),
             chunk_steps=np.array([s for _, s in c.chunks[args.day : args.day + n]]))  # fmt: skip
    print("saved", out)

else:
    variants = {
        "default": (),
        "no step checkpoint": ("model.numerics.checkpoint_steps=false",),
        "sub-step checkpoint": ("model.numerics.checkpoint_substeps=always",),
        "no checkpoint at all": ("model.numerics.checkpoint_steps=false",
                                 "model.numerics.checkpoint_substeps=never"),
    }  # fmt: skip
    res = {}
    for name, overrides in variants.items():
        c = Case(args.case or "may2007", overrides)
        gx, gs = c.window(0, 2).reverse_mode(0)
        res[name] = np.concatenate([gx[c.active], gs[c.state_in_use()]])
        ref = res["default"]
        print(name, "largest relative difference from the default:",
              float(np.abs(res[name] - ref).max() / np.abs(ref).max()), flush=True)  # fmt: skip
    np.savez(RESULTS / "checkpoint.npz", **{k.replace(" ", "_"): v for k, v in res.items()})
