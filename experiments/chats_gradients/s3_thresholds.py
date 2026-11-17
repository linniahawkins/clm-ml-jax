"""
Step 3: threshold sweeps. One input (a forcing field or a state variable) is pushed
through a threshold of the model; the outputs and their derivatives with respect to that
input are computed together at every point (forward mode), so jumps, kinks and stretches
of exactly zero derivative show.

Each sweep starts from the default run's state at a chosen step of a day (noon or the
middle of the night), changes the input for the next few steps, and averages the outputs
over them. Results: ``thresholds/<sweep>_d<day>.npz`` in the results directory.

    JAX_PLATFORMS=cpu python s3_thresholds.py [--day 0] [--points 401] [--only NAME]
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from harness import _STREAM, RESULTS, VARS, Case

ap = argparse.ArgumentParser()
ap.add_argument("--day", type=int, default=0)
ap.add_argument("--points", type=int, default=401)
ap.add_argument("--only", default="")
args = ap.parse_args()

c = Case("may2007")
k = args.day
x0 = jnp.asarray(c.x0)
forcing = c.forcing(k)
n_steps = c.chunks[k][1]
lab = c.labels(k)
zen = lab["solar_zenith_angle"]
NOON = int(np.argmin(zen))
dark = np.flatnonzero(zen >= zen.max() - 1e-9)
NIGHT = int(dark[len(dark) // 2])
print("day", k, c.date(k), "noon step", NOON, "night step", NIGHT, flush=True)


def cut(t, n):
    return jax.tree.map(lambda a: a[t : t + n], forcing)


# the default run's state at every step of the day (one compiled single step)
one_step = jax.jit(lambda s, xg, xp: c.traj(s, x0, xg, xp)[0])
states = [c.state(k)]
for t in range(n_steps - 1):
    states.append(one_step(states[-1], *cut(t, 1)))


def scale_shortwave(v, xg):
    b, n = xg.forcing_bef, xg.forcing_next
    return xg._replace(
        solad=xg.solad * v,
        solai=xg.solai * v,
        forcing_bef=b._replace(swskyb=b.swskyb * v, swskyd=b.swskyd * v),
        forcing_next=n._replace(swskyb=n.swskyb * v, swskyd=n.swskyd * v),
    )


def set_wind(v, xg):
    b, n = xg.forcing_bef, xg.forcing_next
    return xg._replace(
        uref=v + 0 * xg.uref,
        forcing_bef=b._replace(uref=v + 0 * b.uref),
        forcing_next=n._replace(uref=v + 0 * n.uref),
    )


def shift_air_temperature(v, xg):
    b, n = xg.forcing_bef, xg.forcing_next
    return xg._replace(
        tref=xg.tref + v,
        forcing_bef=b._replace(tref=b.tref + v),
        forcing_next=n._replace(tref=n.tref + v),
    )


def leafy(st):  # the layers that carry leaves: their leaf water potential is in use
    psi = st.canopy.leaf_water_potential
    return (psi != 0) & (jnp.abs(psi) < 1e30)


def set_leaf_water(v, st):
    can = st.canopy
    wet = jnp.where(leafy(st)[..., 0], v, can.intercepted_water)
    return st._replace(canopy=can._replace(intercepted_water=wet))


def set_leaf_water_potential(v, st):
    can = st.canopy
    psi = jnp.where(leafy(st), v, can.leaf_water_potential)
    return st._replace(canopy=can._replace(leaf_water_potential=psi))


F = lambda f: (lambda v, st, xg, xp: (st, f(v, xg), xp))  # noqa: E731  a change of the met
S = lambda f: (lambda v, st, xg, xp: (f(v, st), xg, xp))  # noqa: E731  a change of the state
soil = lambda v, st, xg, xp: (st, xg, xp._replace(h2osoi=xp.h2osoi * v))  # noqa: E731
lai = lambda v, st, xg, xp: (st, xg, xp._replace(elai=xp.elai * v))  # noqa: E731

# name: (first step, steps, values, change, x label, the threshold it crosses)
N = args.points
SWEEPS = {
    "soil_water_noon": (NOON - 3, 6, np.linspace(0.3, 1.2, N), soil,
                        "soil water, fraction of the prescribed", "onset of water stress"),
    "lai_noon": (NOON, 2, np.linspace(0.0, 1.5, N), lai, "LAI, fraction of the prescribed",
                 "sparse canopy; minimum plant area of a layer"),
    "shortwave_noon": (NOON, 2, np.linspace(0.0, 1.2, N), F(scale_shortwave),
                       "shortwave, fraction of the observed", "night to day; light limitation"),
    "wind_noon": (NOON, 2, np.linspace(0.05, 3.0, N), F(set_wind), "wind speed (m s-1)",
                  "minimum wind 0.5 m s-1; limits of L"),
    "wind_night": (NIGHT, 2, np.linspace(0.05, 3.0, N), F(set_wind), "wind speed (m s-1)",
                   "minimum wind 0.5 m s-1; limits of L"),
    "air_temperature_noon": (NOON, 2, np.linspace(-8.0, 8.0, N), F(shift_air_temperature),
                             "forcing air temperature shift (K)", "stability; limits of L"),
    "air_temperature_night": (NIGHT, 2, np.linspace(-8.0, 8.0, N), F(shift_air_temperature),
                              "forcing air temperature shift (K)", "stability; limits of L"),
    "leaf_water_noon": (NOON, 1, np.linspace(0.0, 0.02, N) ** 1.0, S(set_leaf_water),
                        "intercepted water per leafy layer (kg m-2)", "dry leaves; drip"),
    "leaf_water_potential_noon": (NOON, 1, np.linspace(-2.6, -0.2, N),
                                  S(set_leaf_water_potential), "leaf water potential (MPa)",
                                  "psi50 = -1.6 MPa"),
}  # fmt: skip

out_dir = RESULTS / "thresholds"
out_dir.mkdir(parents=True, exist_ok=True)
for name, (t, n, values, change, xlabel, what) in SWEEPS.items():
    if args.only and name != args.only:
        continue
    st0, (xg, xp) = c.unravel_state(states[t]), cut(t, n)

    def outputs(v):
        st, g, p = change(v, st0, xg, xp)
        _, out = c.model.run_chunk(st, (g, p), 1, c.unravel_params(x0), streams=[_STREAM])
        return jnp.stack([out["s"][var][:, 0] for var in VARS]).mean(axis=1)

    both = jax.jit(jax.vmap(lambda v: jax.jvp(outputs, (v,), (jnp.ones_like(v),))))
    y, dy = (np.asarray(a) for a in both(jnp.asarray(values)))  # (point, variable)
    np.savez(out_dir / f"{name}_d{k:02d}.npz", values=values, y=y, dy=dy, vars=np.array(VARS),
             xlabel=xlabel, what=what, first_step=t, steps=n, day=k,
             start=str(c.date(k)))  # fmt: skip
    slope = np.diff(y, axis=0) / np.diff(values)[:, None]  # the curve's own slope
    mid = 0.5 * (dy[1:] + dy[:-1])
    off = np.abs(slope - mid) / (np.abs(mid).max(axis=0) + 1e-300)
    print(f"{name}: NaN {int(np.isnan(y).sum())} / {int(np.isnan(dy).sum())}; "
          f"exactly zero derivative at {(dy == 0).mean(axis=0).round(2)} of the points; "
          f"curve slope against AD, worst {off.max(axis=0).round(3)}", flush=True)  # fmt: skip
