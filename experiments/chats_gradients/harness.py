"""
One harness for the gradient experiments at CHATS (README.md in this folder; open item O82).

A *case* is one of the experiment folders next to this file (``may2007``, ``aprmay2007``),
with any settings changed on top (``model.numerics.checkpoint_steps=false``). A *window*
is a number of whole chunks of the run (the folders use one-day chunks) starting at a chosen
chunk; the state at its start is that of the default run. For a window the harness gives

- the outputs ``VARS`` at every step, and the **window quantities** made from them: the
  synthetic-twin loss and the window mean of each output;
- their derivatives with respect to every parameter (and to chosen directions of the
  state at the window's start), by automatic differentiation in forward mode (every step
  of every output) and in reverse mode (one quantity, default the loss; the two must
  agree), and by central finite differences at several relative steps;
- the one-at-a-time experiment: the quantities with each parameter at the minimum and the
  maximum of its range (default +-10 %), next to what the derivative predicts for it;
- from the reverse pass, the derivative with respect to the state at the window's start.

``record`` appends the window quantities to one table (CSV, one row per parameter and
quantity) and saves the per-step arrays next to it (npz), both in scratch: the experiments
are slices of these by window, regime label and settings.

The derivatives are chained chunk by chunk on the host (the tangent of the state is carried
forward, its adjoint backward), so one compiled chunk serves every window length and memory
does not grow with the horizon.

    JAX_PLATFORMS=cpu python harness.py --case may2007 --first 0 --days 1 --label "1 May"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
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

# outputs that are differentiated; the first N_LOSS of them make the loss
VARS = ("latent_heat_flux", "sensible_heat_flux", "gpp", "friction_velocity", "net_radiation",
        "ground_heat_flux", "canopy_top_air_temperature")  # fmt: skip
N_LOSS = 3
QUANTITIES = ("loss",) + tuple(f"mean:{v}" for v in VARS)
FD_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)  # relative to the parameter's value
RANGE = 0.10  # the parameter range: +-10 % of the value (maintainer, 2026-11-17)
# physical limits inside +10 %; an emissivity of exactly 1 gives NaN (O83): 0.999 (maintainer)
UPPER_BOUNDS = {"leaf_emissivity": 0.999, "clumping_index": 1.0}
TWIN_SHIFT, TWIN_SEED = 0.05, 0  # the twin's "true" parameters: within +-5 % of the default
BATCH = 64  # parameter vectors run together
_STREAM = StreamSpec("s", "step", "mean", VARS)
# what the regimes of a step are read from
LABEL_VARS = ("solar_zenith_angle", "rsl_beta", "top_leaf_water_potential",
              "fraction_at_minimum_leaf_water_potential", "canopy_evaporation_latent_heat_flux",
              "sensible_heat_flux", "soil_water_stress", "incident_visible_radiation")  # fmt: skip


class Case:
    """A model, its default run's states at the start of every chunk (computed when first
    asked for), and its parameters as one vector."""

    def __init__(self, case: str = "may2007", overrides=()):
        self.name, self.overrides = case, tuple(overrides)
        self.model = Model.from_experiment(clm_ml.load_experiment(HERE / case, list(overrides)))
        self.chunks = self.model.chunks()
        params = self.model.default_params()
        x0, self.unravel_params = ravel_pytree(params)
        self.x0 = np.asarray(x0)
        self.names = _names(params)
        s0, self.unravel_state = ravel_pytree(self.model.initial_state())
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
        return self.model._sim.step_starts(self.chunks[k][0], 1)[0]

    def state(self, k: int):
        """The default run's state vector at the start of chunk ``k``."""
        while len(self._states) <= k:
            j = len(self._states) - 1
            s, _ = self.program("traj", j)(self._states[j], jnp.asarray(self.x0), *self.forcing(j))
            self._states.append(s)
        return self._states[k]

    def state_in_use(self) -> np.ndarray:
        """Which elements of the state vector are real (not the slots above the canopy air)."""
        return np.abs(np.asarray(self._states[0])) < 1e30

    def labels(self, k: int) -> dict:
        """The default run's regime variables at every step of chunk ``k`` (and the rain)."""
        stream = StreamSpec("l", "step", "mean", LABEL_VARS)
        _, out = self.model.run_chunk(
            self.unravel_state(self.state(k)), self.forcing(k), 1, streams=[stream]
        )
        lab = {v: np.asarray(out["l"][v][:, 0]) for v in LABEL_VARS}
        lab["rain"] = np.asarray(self.forcing(k)[0].rain[:, 0])
        return lab

    def traj(self, s, x, xg, xp):
        """One chunk (or any number of steps of forcing): the new state vector and the
        outputs, (variable, step)."""
        new, out = self.model.run_chunk(
            self.unravel_state(s), (xg, xp), 1, self.unravel_params(x), streams=[_STREAM]
        )
        return ravel_pytree(new)[0], jnp.stack([out["s"][v][:, 0] for v in VARS])

    def _sums(self, s, x, xg, xp, obs, inv_var):
        """One chunk's contribution to the window quantities, as sums over its steps."""
        new, y = self.traj(s, x, xg, xp)
        misfit = jnp.sum((y[:N_LOSS] - obs[:N_LOSS]) ** 2 * inv_var[:N_LOSS, None]) / N_LOSS
        return new, jnp.concatenate([misfit[None], jnp.sum(y, axis=1)])

    def program(self, kind: str, k: int):
        """The compiled chunk programs, one per kind and chunk length (the run step a chunk
        starts at only labels its output records, so every chunk of a length shares them)."""
        key = (kind, self.chunks[k][1])
        if key not in self._programs:
            f, g = self._sums, self.traj
            if kind == "traj":
                p = jax.jit(g)
            elif kind == "batch":  # a batch of states and parameter vectors
                p = jax.jit(jax.vmap(g, in_axes=(0, 0, None, None)))
            elif kind == "jvp":  # a batch of points and tangents (state, parameters)

                def one(s, x, ds, dx, xg, xp):
                    return jax.jvp(lambda a, b: g(a, b, xg, xp), (s, x), (ds, dx))

                p = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, None, None)))
            elif kind == "vjp":  # the adjoint of (new state, sums) back to (state, parameters)

                def p_(s, x, xg, xp, obs, iv, cs, cq):
                    return jax.vjp(lambda a, b: f(a, b, xg, xp, obs, iv), s, x)[1]((cs, cq))

                p = jax.jit(p_)
            else:  # "grad_all": the chunk's misfit sum differentiated by every input

                def p_(s, x, xg, xp, obs, iv):
                    return jax.grad(lambda *a: f(*a, obs, iv)[1][0], argnums=(0, 1, 2, 3))(
                        s, x, xg, xp
                    )

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
        self.fd_loss = {}
        self.y0 = self.outputs(case.x0[None])[0]  # (variable, step)
        self.obs = self.outputs(case.x_true[None])[0]  # the twin: its "true" parameters
        var = np.var(self.obs, axis=1)  # an output that never varies (GPP without leaves) is
        self.inv_var = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 0.0)  # left out

    # -- forward ----------------------------------------------------------------------

    def outputs(self, xs, s0s=None) -> np.ndarray:
        """The outputs at every step for a batch of parameter vectors (and, if given, of
        start states): (batch, variable, step)."""
        xs = np.atleast_2d(np.asarray(xs))
        out = []
        for a in range(0, len(xs), BATCH):
            x = jnp.asarray(xs[a : a + BATCH])
            if s0s is None:
                s = jnp.broadcast_to(self.s0, (x.shape[0],) + self.s0.shape)
            else:
                s = jnp.asarray(s0s[a : a + BATCH])
            ys = []
            for k in self.ks:
                s, y = self.case.program("batch", k)(s, x, *self.case.forcing(k))
                ys.append(np.asarray(y))
            out.append(np.concatenate(ys, axis=2))
        return np.concatenate(out)

    def _scale(self, steps):
        n = self.steps if steps is None else steps
        if steps is None:
            return n, self.inv_var
        return n, 1.0 / np.maximum(np.var(self.obs[:, :n], axis=1), 1e-300)

    def quantities_of(self, y, steps=None) -> np.ndarray:
        """The window quantities of outputs (..., variable, step): (..., quantity). ``steps``:
        only the first so many steps of the window (a shorter horizon; the loss then scales
        each misfit with the variance of the observations over those steps)."""
        n, iv = self._scale(steps)
        y, obs = np.asarray(y)[..., :n], self.obs[:, :n]
        misfit = ((y[..., :N_LOSS, :] - obs[:N_LOSS]) ** 2 * iv[:N_LOSS, None]).sum((-1, -2))
        return np.concatenate([misfit[..., None] / N_LOSS / n, y.mean(-1)], axis=-1)

    def dquantities_of(self, dy, steps=None) -> np.ndarray:
        """The derivative of the window quantities given that of the outputs at the default
        parameters, (..., variable, step): the chain rule on the host."""
        n, iv = self._scale(steps)
        dy, y, obs = np.asarray(dy)[..., :n], self.y0[:, :n], self.obs[:, :n]
        w = 2 * (y[:N_LOSS] - obs[:N_LOSS]) * iv[:N_LOSS, None]
        dloss = (dy[..., :N_LOSS, :] * w).sum((-1, -2)) / N_LOSS / n
        return np.concatenate([dloss[..., None], dy.mean(-1)], axis=-1)

    def quantities(self, xs) -> np.ndarray:
        return self.quantities_of(self.outputs(xs))

    # -- derivatives --------------------------------------------------------------------

    def forward_mode(self, dxs=None, ds0s=None, xs=None) -> np.ndarray:
        """Forward-mode derivatives of the outputs at every step, (direction, variable,
        step), for directions of the parameters (``dxs``) and of the state at the window's
        start (``ds0s``; zero if not given), at the default parameters or, with ``xs``, each
        direction at its own parameter vector."""
        c = self.case
        m = len(dxs) if dxs is not None else len(ds0s)
        dxs = np.zeros((m, c.x0.size)) if dxs is None else np.asarray(dxs)
        ds0s = np.zeros((m,) + self.s0.shape) if ds0s is None else np.asarray(ds0s)
        xs = np.tile(c.x0, (m, 1)) if xs is None else np.asarray(xs)
        out = []
        for a in range(0, m, BATCH):
            x, dx = jnp.asarray(xs[a : a + BATCH]), jnp.asarray(dxs[a : a + BATCH])
            ds = jnp.asarray(ds0s[a : a + BATCH])
            s = jnp.broadcast_to(self.s0, ds.shape)
            dys = []
            for k in self.ks:
                (s, _), (ds, dy) = c.program("jvp", k)(s, x, ds, dx, *c.forcing(k))
                dys.append(np.asarray(dy))
            out.append(np.concatenate(dys, axis=2))
        return np.concatenate(out)

    def _states(self):
        c, x, states = self.case, jnp.asarray(self.case.x0), [self.s0]
        for k in self.ks[:-1]:
            states.append(c.program("traj", k)(states[-1], x, *c.forcing(k))[0])
        return states

    def _obs_chunks(self):
        edges = np.cumsum([0] + [self.case.chunks[k][1] for k in self.ks])
        return [jnp.asarray(self.obs[:, a:b]) for a, b in zip(edges[:-1], edges[1:])]

    def reverse_mode(self, quantity: int = 0, x=None):
        """The gradient of one window quantity with respect to the parameters and to the
        state at the window's start: ``(d/dparams, d/dstate)``; at the default parameters,
        or at ``x`` (the states of the window are then those of ``x``)."""
        c, iv = self.case, jnp.asarray(self.inv_var)
        x = jnp.asarray(c.x0 if x is None else x)
        states = [self.s0]
        for k in self.ks[:-1]:
            states.append(c.program("traj", k)(states[-1], x, *c.forcing(k))[0])
        cq = jnp.zeros(len(QUANTITIES)).at[quantity].set(1.0 / self.steps)
        cs, grad = jnp.zeros_like(self.s0), 0.0
        for k, s, obs in reversed(list(zip(self.ks, states, self._obs_chunks()))):
            cs, gx = c.program("vjp", k)(s, x, *c.forcing(k), obs, iv, cs, cq)
            grad = grad + np.asarray(gx)
        return grad, np.asarray(cs)

    def gradient_of_every_input(self) -> dict:
        """The loss differentiated by every input of each chunk on its own (state at its
        start, parameters, every forcing field): the largest absolute value and a NaN flag
        per input, over the window. A search for NaN, not a sensitivity."""
        c, x, iv = self.case, jnp.asarray(self.case.x0), jnp.asarray(self.inv_var)
        out = {}
        for k, s, obs in zip(self.ks, self._states(), self._obs_chunks()):
            gs, gx, gg, gp = c.program("grad_all", k)(s, x, *c.forcing(k), obs, iv)
            parts = {"state": np.asarray(gs)[c.state_in_use()], "params": np.asarray(gx)[c.active]}
            for tree in (gg, gp):
                for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
                    parts["forcing" + jax.tree_util.keystr(path)] = np.asarray(leaf)
            for name, g in parts.items():
                big, bad = out.get(name, (0.0, False))
                finite = np.abs(g[np.isfinite(g)])
                out[name] = (max(big, float(finite.max()) if finite.size else 0.0),
                             bad or not bool(np.isfinite(g).all()))  # fmt: skip
        return out

    def finite_differences(self, steps=FD_STEPS) -> np.ndarray:
        """Central differences of the outputs at every step, for every active parameter:
        (step size, parameter, variable, step). The differences of the loss (not linear in
        the outputs) are kept in ``fd_loss[step size]``."""
        c = self.case
        idx = np.flatnonzero(c.active)
        fd = np.full((len(steps), c.x0.size) + self.y0.shape, np.nan)
        for j, h in enumerate(steps):
            e = h * np.where(c.x0[idx] != 0, np.abs(c.x0[idx]), 1.0)
            up, dn = np.tile(c.x0, (len(idx), 1)), np.tile(c.x0, (len(idx), 1))
            up[np.arange(len(idx)), idx] += e
            dn[np.arange(len(idx)), idx] -= e
            y = self.outputs(np.concatenate([up, dn]))
            fd[j, idx] = (y[: len(idx)] - y[len(idx) :]) / (2 * e[:, None, None])
            q = self.quantities_of(y)
            self.fd_loss[h] = (q[: len(idx), 0] - q[len(idx) :, 0]) / (2 * e)
        return fd

    def one_at_a_time(self):
        """The outputs with each active parameter at the minimum and at the maximum of
        its range, the others at their defaults: two arrays (parameter, variable, step)."""
        c = self.case
        idx = np.flatnonzero(c.active)
        lo, hi = np.tile(c.x0, (len(idx), 1)), np.tile(c.x0, (len(idx), 1))
        lo[np.arange(len(idx)), idx] = c.lo[idx]
        hi[np.arange(len(idx)), idx] = c.hi[idx]
        y = self.outputs(np.concatenate([lo, hi]))
        out = np.full((2, c.x0.size) + self.y0.shape, np.nan)
        out[0, idx], out[1, idx] = y[: len(idx)], y[len(idx) :]
        return out[0], out[1]


def _names(tree) -> list:
    """A name for every element of a pytree's vector (``ravel_pytree`` order)."""
    names = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        key, shape = jax.tree_util.keystr(path).lstrip("."), np.shape(leaf)[1:]  # no PFT/patch axis
        names += [key + (str(list(i)) if shape else "") for i in np.ndindex(*shape)] or [key]
    return names


def state_groups(case: Case) -> dict:
    """The state vector's elements in use, by variable (soil temperature in three depths)."""
    use, groups = case.state_in_use(), {}
    for i, name in enumerate(case.state_names):
        if not use[i]:
            continue
        key = name.split("[")[0].replace("canopy.", "")
        if key == "soil.temperature":
            level = int(name.split("[")[1].rstrip("]"))
            depth = "top (0-4)" if level < 5 else "mid (5-11)" if level < 12 else "deep (12-24)"
            key = "soil_temperature " + depth
        groups.setdefault(key, []).append(i)
    return {k: np.array(v) for k, v in groups.items()}


def record(window: Window, label: str = "", experiment: str = "baseline", table=None,
           fd_steps=FD_STEPS, tag: str = "", every_input: bool = True) -> dict:  # fmt: skip
    """Everything the harness measures for one window: the window quantities appended to
    the table, one row per (parameter, quantity); the per-step arrays saved as
    ``<experiment>/<tag>.npz`` next to it, and returned."""
    c, t0 = window.case, time.perf_counter()
    table = Path(table) if table else RESULTS / "table.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    idx = np.flatnonzero(c.active)
    ad_step = np.full((c.x0.size,) + window.y0.shape, np.nan)
    ad_step[idx] = window.forward_mode(np.eye(c.x0.size)[idx])
    fd_step = window.finite_differences(fd_steps)
    lo_step, hi_step = window.one_at_a_time()
    rev, d_state = window.reverse_mode(0)
    q0, ad = window.quantities_of(window.y0), window.dquantities_of(ad_step)
    fd = np.stack([window.dquantities_of(f) for f in fd_step])  # right for the means
    for j, h in enumerate(fd_steps):
        fd[j, idx, 0] = window.fd_loss[h]
    q_lo, q_hi = window.quantities_of(lo_step), window.quantities_of(hi_step)
    in_use = c.state_in_use()
    every = window.gradient_of_every_input() if every_input else {}
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
        "nan_inputs": " ".join(k for k, (_, bad) in every.items() if bad),
    }
    column = {h: j for j, h in enumerate(fd_steps)}
    rows = []
    for i in idx:
        for m, quantity in enumerate(QUANTITIES):
            row = dict(common, input=c.names[i], x0=c.x0[i], x_min=c.lo[i], x_max=c.hi[i],
                       x_true=c.x_true[i], output=quantity, y0=q0[m], ad_forward=ad[i, m],
                       ad_reverse=rev[i] if m == 0 else np.nan)  # fmt: skip
            for h in FD_STEPS:
                row[f"fd_{h:.0e}"] = fd[column[h], i, m] if h in column else np.nan
            row.update(y_at_min=q_lo[i, m], y_at_max=q_hi[i, m],
                       nan=bool(np.isnan([ad[i, m], *fd[:, i, m], q_lo[i, m], q_hi[i, m]]).any()))  # fmt: skip
            rows.append(row)
    with table.open("a", newline="") as f:  # several processes may write: one at a time
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)
    arrays = dict(
        names=np.array(c.names), vars=np.array(VARS), active=c.active, x0=c.x0, lo=c.lo, hi=c.hi,
        x_true=c.x_true, y0=window.y0, obs=window.obs, ad_step=ad_step, fd_step=fd_step,
        fd_steps=np.array(fd_steps), lo_step=lo_step, hi_step=hi_step, rev=rev, d_state=d_state,
        state_names=np.array(c.state_names), state_in_use=in_use, q0=q0, ad=ad, fd=fd, q_lo=q_lo,
        q_hi=q_hi, every_names=np.array(list(every)),
        every_max=np.array([v[0] for v in every.values()]),
        every_nan=np.array([v[1] for v in every.values()]),
    )  # fmt: skip
    if window.n == 1:
        arrays.update({f"label_{k}": v for k, v in c.labels(window.first).items()})
    out = table.parent / experiment / f"{tag or f'{c.name}_{window.first:02d}_{window.n:02d}'}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"{label or experiment}: {len(rows)} rows to {table}, arrays to {out} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)  # fmt: skip
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser(description="Record one window in the table.")
    ap.add_argument("--case", default="may2007", help="experiment folder next to this file")
    ap.add_argument("--first", type=int, default=0, help="first chunk (day) of the window")
    ap.add_argument("--days", type=int, default=1, help="chunks (days) in the window")
    ap.add_argument("--label", default="", help="label written to the table")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--table", default="", help="the CSV table (default: in scratch)")
    ap.add_argument("--fd", default="", help="finite-difference steps, e.g. 1e-3,1e-5,1e-7")
    ap.add_argument("--set", action="append", default=[], metavar="FILE.KEY=VALUE",
                    help="a setting changed on top of the folder, as `clm-ml run --set`")  # fmt: skip
    args = ap.parse_args()
    window = Case(args.case, args.set).window(args.first, args.days)
    fd = tuple(float(h) for h in args.fd.split(",")) if args.fd else FD_STEPS
    record(window, args.label, args.experiment, args.table or None, fd)


if __name__ == "__main__":
    main()
