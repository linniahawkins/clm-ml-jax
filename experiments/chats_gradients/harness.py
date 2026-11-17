"""
One harness for the gradient experiments at CHATS (README.md in this folder; open item O82).

A *case* is one of the experiment folders next to this file (``may2007``, ``aprmay2007``),
with any settings changed on top (``model.numerics.wetted_fraction=explicit``). A *window*
is a number of whole chunks of the run (the folders use one-day chunks) starting at a chosen
chunk; the state at its start is that of the default run. For a window the harness gives

- the **window quantities**: the synthetic-twin loss and the window mean of each output
  in ``VARS`` (so both "how good is the fit" and "what is the flux" are differentiated);
- their derivatives with respect to every parameter, by automatic differentiation in
  forward mode (all quantities) and in reverse mode (one quantity, default the loss; the
  two must agree), and by central finite differences at several relative steps;
- the one-at-a-time experiment: the quantities with each parameter at the minimum and the
  maximum of its range (default +-10 %), next to what the derivative predicts for it;
- from the reverse pass, the derivative with respect to the state at the window's start.

``record`` appends all of this to one table (CSV, one row per input and output) in scratch:
the experiments are slices of that table by window, regime label and settings.

The derivatives are chained chunk by chunk on the host (the tangent of the state is carried
forward, its adjoint backward), so one compiled chunk serves every window length and memory
does not grow with the horizon.

    JAX_PLATFORMS=cpu python harness.py --case may2007 --first 0 --days 1 --label "1 May"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import clm_ml
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SCRATCH", "/glade/derecho/scratch/linnia")) / "clm-ml-jax" / "chats_gradients"  # fmt: skip

# outputs whose window means are differentiated; the first N_LOSS of them make the loss
VARS = ("latent_heat_flux", "sensible_heat_flux", "gpp", "friction_velocity", "net_radiation",
        "ground_heat_flux", "canopy_top_air_temperature")  # fmt: skip
N_LOSS = 3
QUANTITIES = ("loss",) + tuple(f"mean:{v}" for v in VARS)
FD_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)  # relative to the parameter's value
RANGE = 0.10  # the parameter range: +-10 % of the value (maintainer, 2026-11-17)
UPPER_BOUNDS = {"leaf_emissivity": 1.0, "clumping_index": 1.0}  # physical limits inside +10 %
TWIN_SHIFT, TWIN_SEED = 0.05, 0  # the twin's "true" parameters: within +-5 % of the default
_STREAM = StreamSpec("s", "step", "mean", VARS)


class Case:
    """A model, its default run's states at the start of every chunk (computed when first
    asked for), and its parameters as one vector."""

    def __init__(self, case: str = "may2007", overrides=()):
        self.name, self.overrides = case, tuple(overrides)
        self.model = Model.from_experiment(clm_ml.load_experiment(HERE / case, list(overrides)))
        self.chunks = self.model.chunks()
        params = self.model.default_params()
        x0, self._unravel_params = ravel_pytree(params)
        self.x0 = np.asarray(x0)
        self.names = _names(params)
        s0, self._unravel_state = ravel_pytree(self.model.initial_state())
        self._states = [s0]
        self.state_names = _names(self.model.initial_state())
        # parameters that are switches in disguise (zeta_max_stable = 1e30: no limit) stay out
        self.active = np.abs(self.x0) < 1e29
        hi = self.x0 + RANGE * np.abs(self.x0)
        for key, bound in UPPER_BOUNDS.items():
            hi = np.where([n.split("[")[0] == key for n in self.names], np.minimum(hi, bound), hi)
        self.lo, self.hi = self.x0 - RANGE * np.abs(self.x0), hi
        rng = np.random.default_rng(TWIN_SEED)
        shift = TWIN_SHIFT * rng.uniform(-1, 1, self.x0.size) * np.abs(self.x0)
        self.x_true = np.where(self.active, np.clip(self.x0 + shift, self.lo, self.hi), self.x0)
        self._forcing, self._programs = {}, {}
        # one plain call before any compiled one: it fills the model's host caches, which
        # cannot be filled while a program is being traced (gradient log, "easy to trip over")
        self.model.run_chunk(self.model.initial_state(), self.forcing(0), 1, streams=[_STREAM])

    # -- pieces ---------------------------------------------------------------------

    def forcing(self, k: int):
        if k not in self._forcing:
            self._forcing[k] = self.model.forcing(*self.chunks[k])
        return self._forcing[k]

    def date(self, k: int) -> dt.datetime:
        """The start of chunk ``k`` (chunks start one step after midnight: the first step of
        the run is taken at setup)."""
        sim = self.model._sim
        return sim.step_starts(self.chunks[k][0], 1)[0]

    def state(self, k: int):
        """The default run's state vector at the start of chunk ``k``."""
        while len(self._states) <= k:
            j = len(self._states) - 1
            s, _ = self._program("traj", j)(self._states[j], jnp.asarray(self.x0), *self.forcing(j))
            self._states.append(s)
        return self._states[k]

    def _traj(self, s, x, xg, xp):
        """One chunk: the new state vector and the outputs, (variable, step)."""
        new, out = self.model.run_chunk(
            self._unravel_state(s), (xg, xp), 1, self._unravel_params(x), streams=[_STREAM]
        )
        return ravel_pytree(new)[0], jnp.stack([out["s"][v][:, 0] for v in VARS])

    def _sums(self, s, x, xg, xp, obs, inv_var):
        """One chunk's contribution to the window quantities, as sums over its steps."""
        new, y = self._traj(s, x, xg, xp)
        misfit = jnp.sum((y[:N_LOSS] - obs[:N_LOSS]) ** 2 * inv_var[:N_LOSS, None]) / N_LOSS
        return new, jnp.concatenate([misfit[None], jnp.sum(y, axis=1)])

    def _program(self, kind: str, k: int):
        """The compiled chunk programs, one per kind and chunk length (the run step a chunk
        starts at only labels its output records, so every chunk of a length shares them)."""
        key = (kind, self.chunks[k][1])
        if key not in self._programs:
            f = self._sums
            if kind == "traj":
                p = jax.jit(self._traj)
            elif kind == "sums":  # a batch of states and parameter vectors
                p = jax.jit(jax.vmap(f, in_axes=(0, 0, None, None, None, None)))
            elif kind == "jvp":  # a batch of tangents (state, parameters)

                def one(s, x, xg, xp, obs, iv, ds, dx):
                    return jax.jvp(lambda a, b: f(a, b, xg, xp, obs, iv), (s, x), (ds, dx))[1]

                p = jax.jit(jax.vmap(one, in_axes=(None,) * 6 + (0, 0)))
            else:  # "vjp": the adjoint of (new state, sums) back to (state, parameters)

                def p_(s, x, xg, xp, obs, iv, cs, cq):
                    return jax.vjp(lambda a, b: f(a, b, xg, xp, obs, iv), s, x)[1]((cs, cq))

                p = jax.jit(p_)
            self._programs[key] = p
        return self._programs[key]

    # -- a window -------------------------------------------------------------------

    def window(self, first: int = 0, n: int = 1) -> "Window":
        return Window(self, first, n)


class Window:
    """Chunks ``first ... first + n - 1`` of a case, with the twin's observations."""

    def __init__(self, case: Case, first: int, n: int):
        if first + n > len(case.chunks):
            raise ValueError(f"the run has {len(case.chunks)} chunks")
        self.case, self.first, self.n = case, first, n
        self.ks = list(range(first, first + n))
        self.steps = sum(case.chunks[k][1] for k in self.ks)
        self.s0 = case.state(first)
        self.obs = self.outputs(case.x_true)  # the twin: the model with its "true" parameters
        whole = np.concatenate(self.obs, axis=1)
        self.inv_var = jnp.asarray(1.0 / np.var(whole, axis=1))

    def outputs(self, x) -> list:
        """The outputs of every chunk, each (variable, step)."""
        s, ys = self.s0, []
        for k in self.ks:
            s, y = self.case._program("traj", k)(s, jnp.asarray(x), *self.case.forcing(k))
            ys.append(np.asarray(y))
        return ys

    def quantities(self, xs) -> np.ndarray:
        """The window quantities for a batch of parameter vectors: (batch, quantity)."""
        xs = jnp.atleast_2d(jnp.asarray(xs))
        s = jnp.broadcast_to(self.s0, (xs.shape[0],) + self.s0.shape)
        total = 0.0
        for k, obs in zip(self.ks, self.obs):
            c = self.case
            s, q = c._program("sums", k)(s, xs, *c.forcing(k), jnp.asarray(obs), self.inv_var)
            total = total + np.asarray(q)
        return total / self.steps

    def forward_mode(self, dirs) -> np.ndarray:
        """d(quantity)/d(direction) for parameter directions ``dirs``: (direction, quantity)."""
        c, x, s = self.case, jnp.asarray(self.case.x0), self.s0
        dx, ds = jnp.asarray(dirs), jnp.zeros((len(dirs),) + self.s0.shape)
        total = 0.0
        for k, obs in zip(self.ks, self.obs):
            args = (s, x, *c.forcing(k), jnp.asarray(obs), self.inv_var)
            ds, dq = c._program("jvp", k)(*args, ds, dx)
            s, _ = c._program("traj", k)(s, x, *c.forcing(k))
            total = total + np.asarray(dq)
        return total / self.steps

    def reverse_mode(self, quantity: int = 0):
        """The gradient of one window quantity with respect to the parameters and to the
        state at the window's start: ``(d/dparams, d/dstate)``."""
        c, x = self.case, jnp.asarray(self.case.x0)
        states = [self.s0]
        for k in self.ks[:-1]:
            states.append(c._program("traj", k)(states[-1], x, *c.forcing(k))[0])
        cq = jnp.zeros(len(QUANTITIES)).at[quantity].set(1.0 / self.steps)
        cs, grad = jnp.zeros_like(self.s0), 0.0
        for k, s, obs in reversed(list(zip(self.ks, states, self.obs))):
            args = (s, x, *c.forcing(k), jnp.asarray(obs), self.inv_var)
            cs, gx = c._program("vjp", k)(*args, cs, cq)
            grad = grad + np.asarray(gx)
        return grad, np.asarray(cs)

    def finite_differences(self, steps=FD_STEPS) -> np.ndarray:
        """Central differences for every active parameter: (step, parameter, quantity)."""
        c = self.case
        idx = np.flatnonzero(c.active)
        fd = np.full((len(steps), c.x0.size, len(QUANTITIES)), np.nan)
        for j, h in enumerate(steps):
            e = h * np.where(c.x0[idx] != 0, np.abs(c.x0[idx]), 1.0)
            up, dn = np.tile(c.x0, (len(idx), 1)), np.tile(c.x0, (len(idx), 1))
            up[np.arange(len(idx)), idx] += e
            dn[np.arange(len(idx)), idx] -= e
            q = self.quantities(np.concatenate([up, dn]))
            fd[j, idx] = (q[: len(idx)] - q[len(idx) :]) / (2 * e[:, None])
        return fd

    def one_at_a_time(self):
        """The quantities with each active parameter at the minimum and at the maximum of
        its range, the others at their defaults: two arrays (parameter, quantity)."""
        c = self.case
        idx = np.flatnonzero(c.active)
        lo, hi = np.tile(c.x0, (len(idx), 1)), np.tile(c.x0, (len(idx), 1))
        lo[np.arange(len(idx)), idx] = c.lo[idx]
        hi[np.arange(len(idx)), idx] = c.hi[idx]
        q = self.quantities(np.concatenate([lo, hi]))
        out = np.full((2, c.x0.size, len(QUANTITIES)), np.nan)
        out[0, idx], out[1, idx] = q[: len(idx)], q[len(idx) :]
        return out[0], out[1]


def _names(tree) -> list:
    """A name for every element of a pytree's vector (``ravel_pytree`` order)."""
    names = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        key, shape = jax.tree_util.keystr(path).lstrip("."), np.shape(leaf)[1:]  # no PFT/patch axis
        names += [key + (str(list(i)) if shape else "") for i in np.ndindex(*shape)] or [key]
    return names


def record(window: Window, label: str = "", experiment: str = "baseline", table=None,
           fd_steps=FD_STEPS) -> Path:  # fmt: skip
    """Everything the harness measures for one window, appended to the table: one row per
    (parameter, quantity)."""
    c, t0 = window.case, time.perf_counter()
    table = Path(table) if table else RESULTS / "table.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    q0 = window.quantities(c.x0)[0]
    idx = np.flatnonzero(c.active)
    ad = np.full((c.x0.size, len(QUANTITIES)), np.nan)
    ad[idx] = window.forward_mode(np.eye(c.x0.size)[idx])
    rev, d_state = window.reverse_mode(0)
    fd = window.finite_differences(fd_steps)
    q_lo, q_hi = window.one_at_a_time()
    in_use = np.abs(np.asarray(window.s0)) < 1e30  # not the layer slots above the canopy air
    common = {
        "experiment": experiment,
        "case": c.name,
        "settings": " ".join(c.overrides),
        "label": label,
        "start": c.date(window.first).isoformat(timespec="minutes"),
        "first_chunk": window.first,
        "chunks": window.n,
        "steps": window.steps,
        "wrt": "params",
        "state_grad_norm": float(np.linalg.norm(d_state[in_use])),
        "state_grad_nan": bool(np.isnan(d_state[in_use]).any()),
    }
    rows = []
    for i in idx:
        for m, quantity in enumerate(QUANTITIES):
            row = dict(common, input=c.names[i], x0=c.x0[i], x_min=c.lo[i], x_max=c.hi[i],
                       x_true=c.x_true[i], output=quantity, y0=q0[m], ad_forward=ad[i, m],
                       ad_reverse=rev[i] if m == 0 else np.nan)  # fmt: skip
            row.update({f"fd_{h:.0e}": fd[j, i, m] for j, h in enumerate(fd_steps)})
            row.update(y_at_min=q_lo[i, m], y_at_max=q_hi[i, m],
                       nan=bool(np.isnan([ad[i, m], *fd[:, i, m], q_lo[i, m], q_hi[i, m]]).any()))  # fmt: skip
            rows.append(row)
    new = not table.exists()
    with table.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"{label or experiment}: {len(rows)} rows to {table} ({time.perf_counter() - t0:.0f} s)")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Record one window in the table.")
    ap.add_argument("--case", default="may2007", help="experiment folder next to this file")
    ap.add_argument("--first", type=int, default=0, help="first chunk (day) of the window")
    ap.add_argument("--days", type=int, default=1, help="chunks (days) in the window")
    ap.add_argument("--label", default="", help="regime label written to the table")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--table", default="", help="the CSV table (default: in scratch)")
    ap.add_argument("--set", action="append", default=[], metavar="FILE.KEY=VALUE",
                    help="a setting changed on top of the folder, as `clm-ml run --set`")  # fmt: skip
    args = ap.parse_args()
    window = Case(args.case, args.set).window(args.first, args.days)
    record(window, args.label, args.experiment, args.table or None)


if __name__ == "__main__":
    main()
