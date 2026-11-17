"""
Step 4: parameters across their range.

    sweep    one parameter from the minimum to the maximum of its range, the others at their
             defaults: the window quantities and their derivative (forward mode) at every
             point. Results: ``sweeps/<parameter>_<days>d_<first>.npz``.
    hessian  the second derivatives of the loss at the default parameters, in two ways: the
             Gauss-Newton form (from the first derivatives of the outputs, exact forward
             mode) and central differences of reverse-mode gradients. Results:
             ``hessian/<days>d_<first>.npz``.

The one-at-a-time experiment (each parameter at its minimum and maximum) is part of every
window the harness records.

    JAX_PLATFORMS=cpu python s4_parameters.py sweep --param iota --days 1 [--first 0] [--points 401]
    JAX_PLATFORMS=cpu python s4_parameters.py hessian --days 1 [--first 0]
"""

from __future__ import annotations

import argparse

import numpy as np
from harness import N_LOSS, RESULTS, VARS, Case

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["sweep", "hessian"])
ap.add_argument("--case", default="may2007")
ap.add_argument("--param", default="iota")
ap.add_argument("--days", type=int, default=1)
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--points", type=int, default=401)
args = ap.parse_args()

c = Case(args.case)
w = c.window(args.first, args.days)
idx = np.flatnonzero(c.active)
tag = f"{args.days}d_{args.first:02d}"

if args.mode == "sweep":
    i = c.names.index(args.param)
    values = np.linspace(c.lo[i], c.hi[i], args.points)
    xs = np.tile(c.x0, (args.points, 1))
    xs[:, i] = values
    y = w.outputs(xs)  # (point, variable, step)
    dy = w.forward_mode(dxs=np.tile(np.eye(c.x0.size)[i], (args.points, 1)), xs=xs)
    q = w.quantities_of(y)
    weight = 2 * (y[:, :N_LOSS] - w.obs[:N_LOSS]) * w.inv_var[:N_LOSS, None]
    dloss = (dy[:, :N_LOSS] * weight).sum((-1, -2)) / N_LOSS / w.steps
    dq = np.concatenate([dloss[:, None], dy.mean(-1)], axis=1)
    out = RESULTS / "sweeps" / f"{args.param}_{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, param=args.param, values=values, x0=c.x0[i], x_true=c.x_true[i], q=q, dq=dq,
             vars=np.array(VARS), days=args.days, first=args.first)  # fmt: skip
    print("saved", out, "NaN:", int(np.isnan(q).sum()), int(np.isnan(dq).sum()))
else:
    scale = c.hi - c.lo  # second derivatives per (fraction of the range)^2
    # Gauss-Newton: the loss is a sum of squares, so 2 J^T W J needs first derivatives only
    dy = w.forward_mode(np.eye(c.x0.size)[idx])[:, :N_LOSS]  # (parameter, variable, step)
    wt = w.inv_var[:N_LOSS, None] / N_LOSS / w.steps
    gn = 2 * np.einsum("ivt,jvt->ij", dy * wt, dy)
    grad0 = w.reverse_mode(0)[0][idx]
    fd = {}
    for h in (1e-2, 1e-3):  # relative to the parameter's value
        hess = np.zeros((len(idx), len(idx)))
        for a, i in enumerate(idx):
            e = h * (abs(c.x0[i]) if c.x0[i] != 0 else 1.0)
            up, dn = c.x0.copy(), c.x0.copy()
            up[i] += e
            dn[i] -= e
            hess[a] = (w.reverse_mode(0, up)[0][idx] - w.reverse_mode(0, dn)[0][idx]) / (2 * e)
        fd[h] = hess
        print("h", h, "asymmetry", np.abs(hess - hess.T).max() / np.abs(hess).max(), flush=True)
    s = scale[idx]
    out = RESULTS / "hessian" / f"{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(c.names)[idx], scale=s, grad=grad0, gauss_newton=gn,
             fd_1e2=fd[1e-2], fd_1e3=fd[1e-3], days=args.days, first=args.first)  # fmt: skip
    print("saved", out)
