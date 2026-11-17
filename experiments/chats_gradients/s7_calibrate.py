"""
Step 7: a twin calibration against daily means, with the default (stabilized) and with the
exact water-stress derivative.

The "observations" are the daily means of latent heat, sensible heat and GPP of the 31
days of May from a run whose calibrated parameters are moved by up to ±5 % (a random draw);
the other parameters stay at their defaults, so the right answer is known and the loss can
reach zero. The loss is the mean squared misfit of the daily means, each variable divided
by the variance of its observed daily means. The calibration starts at the defaults and
takes Adam steps (in units of each parameter's range, so only the gradient's direction and
relative sizes matter, not its overall size), inside the ±10 % ranges. The gradient is the
reverse-mode gradient of the loss, chained day by day.

    JAX_PLATFORMS=cpu python s7_calibrate.py --gradient default --params strong --seed 0

Results: ``calibration/<params>_<gradient>_s<seed>.npz`` in the results directory, written
after every iteration.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from harness import N_LOSS, RESULTS, VARS, Case

STRONG = ["iota", "vcmax25", "psi50_gs", "plant_conductance", "leaf_reflectance[1]"]

ap = argparse.ArgumentParser()
ap.add_argument("--gradient", choices=["default", "exact"], default="default")
ap.add_argument("--params", choices=["strong", "all"], default="strong")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--iters", type=int, default=80)
ap.add_argument("--lr", type=float, default=0.03, help="first Adam step, in parameter ranges")
ap.add_argument("--days", type=int, default=31)
args = ap.parse_args()

overrides = () if args.gradient == "default" else ("model.numerics.water_stress_derivative=exact",)
c = Case("may2007", overrides)
ks = list(range(args.days))
if args.params == "strong":
    idx = np.array([c.names.index(p) for p in STRONG])
else:  # every parameter the run depends on
    idx = np.array([i for i, n in enumerate(c.names)
                    if c.active[i] and not n.startswith(("g0_", "g1_"))])  # fmt: skip
rng = np.random.default_rng(1000 + args.seed)
x_true = c.x0.copy()
x_true[idx] += 0.05 * rng.uniform(-1, 1, len(idx)) * np.abs(c.x0[idx])
x_true = np.clip(x_true, c.lo, c.hi)
width = (c.hi - c.lo)[idx]
s0 = c.state(0)


def chunk(s, x, xg, xp, obs, w):
    """One day: the new state, the day's part of the loss, the day's mean outputs."""
    new, y = c.traj(s, x, xg, xp)
    means = y.mean(axis=1)
    return new, jnp.sum((means[:N_LOSS] - obs) ** 2 * w), means


_fwd, _bwd = {}, {}


def programs(k):
    n = c.chunks[k][1]
    if n not in _fwd:
        _fwd[n] = jax.jit(chunk)
        _bwd[n] = jax.jit(lambda s, x, xg, xp, obs, w, cs: jax.vjp(
            lambda a, b: chunk(a, b, xg, xp, obs, w)[:2], s, x)[1]((cs, jnp.ones(()))))  # fmt: skip
    return _fwd[n], _bwd[n]


def run(x, obs=None, w=None, gradient=False):
    """The daily means (day, variable); with observations the loss and, if asked, its
    gradient with respect to the parameter vector."""
    x = jnp.asarray(x)
    zero = jnp.zeros(N_LOSS)
    states, means, loss = [s0], [], 0.0
    for j, k in enumerate(ks):
        s, part, m = programs(k)[0](states[-1], x, *c.forcing(k),
                                    zero if obs is None else obs[j], zero if w is None else w)  # fmt: skip
        states.append(s)
        means.append(np.asarray(m))
        loss += float(part)
    means = np.array(means)
    if obs is None:
        return means
    scale = 1.0 / (len(ks) * N_LOSS)
    if not gradient:
        return means, loss * scale
    cs, grad = jnp.zeros_like(s0), 0.0
    for j, k in reversed(list(enumerate(ks))):
        cs, gx = programs(k)[1](states[j], x, *c.forcing(k), obs[j], w, cs)
        grad = grad + np.asarray(gx)
    return means, loss * scale, grad * scale


truth = run(x_true)
obs = jnp.asarray(truth[:, :N_LOSS])
w = jnp.asarray(1.0 / np.var(truth[:, :N_LOSS], axis=0))
out = RESULTS / "calibration" / f"{args.params}_{args.gradient}_s{args.seed}.npz"
out.parent.mkdir(parents=True, exist_ok=True)

# Adam on u = (x - default) / range, kept inside the range
u = np.zeros(len(idx))
u_lo, u_hi = (c.lo[idx] - c.x0[idx]) / width, (c.hi[idx] - c.x0[idx]) / width
m1, m2 = np.zeros_like(u), np.zeros_like(u)
hist = {"u": [], "loss": [], "grad": [], "means": [], "seconds": []}
for it in range(args.iters + 1):
    t0 = time.time()
    x = c.x0.copy()
    x[idx] += u * width
    means, loss, grad = run(x, obs, w, gradient=True)
    g = grad[idx] * width
    hist["u"].append(u.copy())
    hist["loss"].append(loss)
    hist["grad"].append(g)
    hist["means"].append(means)
    hist["seconds"].append(time.time() - t0)
    np.savez(out, names=np.array(c.names)[idx], u_true=(x_true[idx] - c.x0[idx]) / width,
             x0=c.x0[idx], width=width, truth=truth, vars=np.array(VARS), lr=args.lr,
             **{k: np.array(v) for k, v in hist.items()})  # fmt: skip
    err = np.abs(u - (x_true[idx] - c.x0[idx]) / width)
    print(f"{it:3d} loss {loss:.3e} |grad| {np.linalg.norm(g):.2e} parameter error (ranges): "
          f"mean {err.mean():.3f} worst {err.max():.3f}  {time.time() - t0:.0f} s", flush=True)  # fmt: skip
    if not np.isfinite(g).all():
        print("stopped: the gradient is not finite")
        break
    lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters)))  # down to a tenth
    m1 = 0.9 * m1 + 0.1 * g
    m2 = 0.999 * m2 + 0.001 * g**2
    step = (m1 / (1 - 0.9 ** (it + 1))) / (np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-30)
    u = np.clip(u - lr * step, u_lo, u_hi)
print("saved", out)
