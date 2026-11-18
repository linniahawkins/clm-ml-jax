"""
The figures and summary numbers of the gradient experiments at CHATS, from the arrays the
other scripts leave in the results directory. Each section is skipped if its results are
not there yet.

    python plots.py [--out figures] [--only baseline,thresholds,parameters,time,locality]

Figures go to ``figures/`` next to this file; the numbers quoted in RESULTS.md are printed
(and saved as ``summary.txt`` in the results directory).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SCRATCH", "/glade/derecho/scratch/linnia")) / "clm-ml-jax" / "chats_gradients"  # fmt: skip
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "figures"))
ap.add_argument("--only", default="")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
ONLY = set(args.only.split(",")) if args.only else None

# one look for every figure: categorical colours in a fixed order, quiet axes
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, INK, INK2, SURFACE = "#b5b4ae", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GREY, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 1.8, "legend.frameon": False, "figure.dpi": 130,
})  # fmt: skip
UNITS = {"latent_heat_flux": "W m$^{-2}$", "sensible_heat_flux": "W m$^{-2}$",
         "gpp": "µmol m$^{-2}$ s$^{-1}$", "friction_velocity": "m s$^{-1}$",
         "net_radiation": "W m$^{-2}$", "ground_heat_flux": "W m$^{-2}$",
         "canopy_top_air_temperature": "K"}  # fmt: skip
SHORT = {"latent_heat_flux": "latent heat", "sensible_heat_flux": "sensible heat", "gpp": "GPP",
         "friction_velocity": "friction velocity", "net_radiation": "net radiation",
         "ground_heat_flux": "ground heat", "canopy_top_air_temperature": "canopy-top air T"}  # fmt: skip
STATE_SCALE = {"air_temperature": 1.0, "vapor_pressure": 100.0, "co2": 10.0,
               "leaf_temperature": 1.0, "leaf_water_potential": 0.1, "ground_temperature": 1.0,
               "acclimation_temperature": 1.0, "acclimation_steps": 1.0, "soil_temperature": 1.0,
               "intercepted_water": 1e-3}  # fmt: skip
_lines = []


def say(*a):
    text = " ".join(str(x) for x in a)
    print(text, flush=True)
    _lines.append(text)


def want(section):
    return ONLY is None or section in ONLY


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", OUT / name, flush=True)


def group_of(name):
    key = name.split("[")[0].replace("canopy.", "")
    if key == "soil.temperature":
        level = int(name.split("[")[1].rstrip("]"))
        return "soil T " + ("top" if level < 5 else "mid" if level < 12 else "deep")
    return key.replace("_", " ").replace("temperature", "T")


def best_fd(ad, fd):
    """The finite difference closest to AD among the step sizes (first axis of ``fd``)."""
    j = np.nanargmin(np.abs(fd - ad[None]), axis=0)
    return np.take_along_axis(fd, j[None], axis=0)[0]


def regimes(z):
    """A regime for every step of a day, from the label variables of a baseline file."""
    zen = z["label_solar_zenith_angle"]
    n = len(zen)
    night = zen >= zen.max() - 1e-6
    reg = np.where(night, "night", "day").astype(object)
    edges = np.flatnonzero(night[1:] != night[:-1])
    for e in edges:
        kind = "sunrise ±1 h" if night[e] else "sunset ±1 h"
        reg[max(0, e - 1) : min(n, e + 3)] = kind
    stressed = z["label_top_leaf_water_potential"] < -1.43  # stress factor below 0.99
    reg[(reg == "day") & stressed] = "day, water stress"
    reg[(reg == "day")] = "day, no stress"
    wet = (z["label_rain"] > 0) | (np.abs(z["label_canopy_evaporation_latent_heat_flux"]) > 0.5)
    reg[wet] = "wet canopy"
    return reg


# =========================================================================== step 2
if want("baseline") and (RESULTS / "table.csv").exists():
    say("\n== Step 2: baseline, every day of May")
    t = pd.read_csv(RESULTS / "table.csv")
    b = t[(t.experiment == "baseline") & (t.chunks == 1)].drop_duplicates(
        ["first_chunk", "input", "output"], keep="last"
    )
    fdc = [c for c in b.columns if c.startswith("fd_")]
    hs = np.array([float(c[3:]) for c in fdc])
    say("days recorded:", b.first_chunk.nunique(), "; rows:", len(b))
    say("rows with NaN:", int(b.nan.sum()), "; days with a NaN gradient in the start state:",
        int(b.groupby("first_chunk").state_grad_nan.first().sum()), "; inputs with NaN:",
        sorted(set(" ".join(b.nan_inputs.dropna().astype(str)).split())))  # fmt: skip
    loss = b[b.output == "loss"]
    live = loss[loss.ad_forward != 0]
    say("forward against reverse mode (loss), worst relative difference:",
        f"{(np.abs(live.ad_forward - live.ad_reverse) / np.abs(live.ad_forward)).max():.1e}")  # fmt: skip
    zero = sorted(set(b[(b.ad_forward == 0)].input))
    say("parameters with a derivative of exactly zero:", zero)
    b = b[~b.input.isin(zero)]

    # V curves: the error of the finite difference against its step
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, q in zip(axes, ["loss", "mean:latent_heat_flux", "mean:gpp"]):
        d = b[b.output == q]
        err = np.abs(d[fdc].values - d.ad_forward.values[:, None]) / np.abs(d.ad_forward.values[:, None])  # fmt: skip
        for p in d.input.unique():
            ax.plot(hs, np.nanmedian(err[(d.input == p).values], axis=0), color=GREY, lw=0.7)
        ax.plot(hs, np.nanmedian(err, axis=0), color=C[0], lw=2.2)
        ax.set(xscale="log", yscale="log", xlabel="finite-difference step (relative)",
               title=q.replace("mean:", "mean ").replace("_", " "))  # fmt: skip
        say(f"V curve, {q}: median relative error by step size",
            dict(zip(fdc, np.nanmedian(err, axis=0).round(5))))  # fmt: skip
    axes[0].set_ylabel("|FD − AD| / |AD|")
    axes[0].text(0.03, 0.05, "grey: each parameter (median of 31 days)\nblue: all",
                 transform=axes[0].transAxes, color=INK2, fontsize=8)  # fmt: skip
    fig.suptitle("Finite differences against AD, one-day windows of May: error against step size",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    save(fig, "baseline_vcurve.png")

    # the loss gradient: best agreement per day and parameter
    d = b[b.output == "loss"].copy()
    d["err"] = np.nanmin(np.abs(d[fdc].values - d.ad_forward.values[:, None]), axis=1) / np.abs(d.ad_forward)  # fmt: skip
    pv = d.pivot(index="input", columns="first_chunk", values="err")
    pv = pv.loc[pv.median(axis=1).sort_values().index].reindex(columns=np.arange(31))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.pcolormesh(pv.columns + 1, np.arange(len(pv)), np.ma.masked_invalid(np.log10(np.maximum(pv.values, 1e-6))),
                       cmap="Blues", vmin=-5, vmax=0, edgecolors=SURFACE, linewidth=0.8)  # fmt: skip
    ax.set_yticks(np.arange(len(pv)), pv.index)
    ax.set(xlabel="day of May 2007 (one-day window)", title="Loss gradient: relative difference between AD and the closest finite difference")  # fmt: skip
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_ticks([-5, -4, -3, -2, -1, 0], labels=["≤1e-5", "1e-4", "0.1 %", "1 %", "10 %", "100 %"])
    save(fig, "baseline_loss_agreement.png")
    say("loss gradient, AD against the closest FD: median", f"{d.err.median():.2e}",
        "; share within 1 %:", f"{(d.err < 1e-2).mean():.3f}", "; within 5 %:",
        f"{(d.err < 5e-2).mean():.3f}")  # fmt: skip
    say("  per parameter (median, worst day):")
    for p, r in d.groupby("input").err.agg(["median", "max"]).sort_values("median").iterrows():
        say(f"    {p:24s} {r['median']:.2e}  {r['max']:.2e}")

    # by regime: every step of every day
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "baseline" / "may2007_*_01.npz"))):
        z = np.load(f, allow_pickle=True)
        keep = z["active"] & ~np.isin(z["names"], zero)
        ad, fd = z["ad_step"][keep], z["fd_step"][:, keep]
        bf = best_fd(ad, fd)
        scale = np.abs(ad).max(axis=2, keepdims=True) + 1e-300  # the pair's largest of the day
        err = np.abs(bf - ad) / scale  # (parameter, variable, step)
        reg = regimes(z)
        day = int(Path(f).stem.split("_")[1])
        for s in range(err.shape[2]):
            rows.append((day, s, reg[s], np.mean(err[:, :, s] < 1e-3), np.mean(err[:, :, s] < 1e-2),
                         np.median(err[:, :, s]), err[:, :, s].max(),
                         np.isnan(ad[:, :, s]).any()))  # fmt: skip
    if rows:
        r = pd.DataFrame(rows, columns=["day", "step", "regime", "in01", "in1", "median", "worst", "nan"])  # fmt: skip
        order = ["night", "sunrise ±1 h", "day, no stress", "day, water stress", "sunset ±1 h", "wet canopy"]  # fmt: skip
        g = r.groupby("regime").agg(steps=("in1", "size"), in01=("in01", "mean"), in1=("in1", "mean"),
                                    worst=("worst", "median")).reindex(order).dropna()  # fmt: skip
        say("per-step derivatives of the seven outputs by regime (share of parameter-output pairs",
            "whose AD is within 0.1 % / 1 % of the closest FD, relative to the pair's largest value of the day):")  # fmt: skip
        say(g.round(4).to_string())
        say("steps with a NaN derivative:", int(r.nan.sum()))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
        y = np.arange(len(g))[::-1]
        axes[0].barh(y + 0.19, g.in1, height=0.34, color=C[0], label="within 1 %")
        axes[0].barh(y - 0.19, g.in01, height=0.34, color=C[1], label="within 0.1 %")
        for yy, a, bb in zip(y, g.in1, g.in01):
            axes[0].text(a + 0.01, yy + 0.19, f"{a:.0%}", va="center", fontsize=8, color=INK2)
            axes[0].text(bb + 0.01, yy - 0.19, f"{bb:.0%}", va="center", fontsize=8, color=INK2)
        axes[0].set_yticks(y, [f"{k} ({int(n)} steps)" for k, n in zip(g.index, g.steps)])
        axes[0].set(xlim=(0, 1.12), xlabel="share of parameter–output pairs",
                    title="AD agrees with finite differences, by regime")  # fmt: skip
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
        axes[0].grid(axis="y", visible=False)
        hour = r.groupby("step")
        axes[1].plot(hour.in1.mean().index / 2, 1 - hour.in1.mean(), color=C[0], label="more than 1 % off")  # fmt: skip
        axes[1].plot(hour.in01.mean().index / 2, 1 - hour.in01.mean(), color=C[1], label="more than 0.1 % off")  # fmt: skip
        axes[1].set(yscale="log", xlabel="hours into the window (UTC; local noon near 20 h)",
                    ylabel="share of pairs", title="Where in the day AD and FD differ (mean of 31 days)")  # fmt: skip
        axes[1].legend(fontsize=8)
        save(fig, "baseline_regimes.png")

    # the default (stabilized) water-stress derivative against the exact one
    ex = t[(t.experiment == "exact_water_stress") & (t.output == "loss") & (t.ad_forward != 0)]
    ex = ex.drop_duplicates(["first_chunk", "input"], keep="last")
    if len(ex):
        e = np.nanmin(np.abs(ex[fdc].values - ex.ad_forward.values[:, None]), axis=1) / np.abs(ex.ad_forward.values)  # fmt: skip
        say("numerics.water_stress_derivative = exact, loss gradient against the closest FD, by day",
            "(median, worst parameter):")  # fmt: skip
        for k, v in pd.Series(e, index=ex.first_chunk.values).groupby(level=0):
            dflt = d[d.first_chunk == k].err
            say(f"    day {k:2d}: exact {v.median():.1e} {v.max():.1e}; default (stabilized)",
                f"{dflt.median():.1e} {dflt.max():.1e}")  # fmt: skip
    fe, fb = (
        RESULTS / "exact_water_stress" / "may2007_08_01.npz",
        RESULTS / "baseline" / "may2007_08_01.npz",
    )
    if fe.exists() and fb.exists():
        ze, zb = np.load(fe, allow_pickle=True), np.load(fb, allow_pickle=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
        hrs = (np.arange(zb["ad_step"].shape[2]) + 1) / 2
        for col, p in enumerate(["vcmax25", "psi50_gs"]):
            i = list(zb["names"]).index(p)
            fdv = best_fd(ze["ad_step"][i], ze["fd_step"][:, i])
            for row, (m, var) in enumerate([(0, "latent heat"), (2, "GPP")]):
                ax = axes[row, col]
                ax.plot(hrs, fdv[m], color=GREY, lw=4, label="finite difference")
                ax.plot(hrs, ze["ad_step"][i, m], color=C[0], lw=1.3, label="AD, exact setting")
                ax.plot(
                    hrs, zb["ad_step"][i, m], color=C[1], lw=1.8, label="AD, default (stabilized)"
                )
                ax.set_title(f"d {var} / d {p}")
                if row == 1:
                    ax.set_xlabel("hours into the day (UTC; local noon near 20 h)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("9 May (strong water stress): the derivative at every step, three ways",
                     x=0.01, ha="left", fontweight="bold")  # fmt: skip
        save(fig, "stabilized_vs_exact_day08.png")

    # the one-at-a-time experiment
    for exp, exp_x, title, name in [
        ("baseline", "exact_water_stress", "one-day windows (median of 31 days)", "oat_1day.png"),
        (
            "month",
            None,
            "the 31 days of May",
            "oat_month.png",
        ),  # default derivative only (maintainer)
    ]:
        d = t[t.experiment == exp].drop_duplicates(["first_chunk", "input", "output"], keep="last")
        d = d[~d.input.isin(zero)]
        if d.empty:
            continue
        d = d.assign(actual=d.y_at_max - d.y_at_min, linear=d.ad_forward * (d.x_max - d.x_min))
        outs = ["mean:latent_heat_flux", "mean:sensible_heat_flux", "mean:gpp", "loss"]
        fig, axes = plt.subplots(1, 4, figsize=(13, 5.2), sharey=True)
        m = d.groupby(["input", "output"])[["actual", "linear"]].median().reset_index()
        order_p = m[m.output == outs[0]].set_index("input").actual.abs().sort_values().index
        # the same with the exact water-stress derivative (a setting; forward values the same)
        dx = t[t.experiment == exp_x].drop_duplicates(["first_chunk", "input", "output"], keep="last")  # fmt: skip
        dx = dx.assign(exact=dx.ad_forward * (dx.x_max - dx.x_min))
        mx = dx.groupby(["input", "output"]).exact.median().reset_index()
        if len(dx):
            say(f"one at a time, {exp}: exact derivative from {dx.first_chunk.nunique()} window(s)")
        for ax, q in zip(axes, outs):
            mm = m[m.output == q].set_index("input").loc[order_p]
            yy = np.arange(len(mm))
            ax.hlines(yy, mm.actual, mm.linear, color=GREY, lw=1.2)
            ax.scatter(mm.linear, yy, s=34, color=C[1], label="default derivative × range", zorder=3,
                       edgecolors=SURFACE, linewidths=1)  # fmt: skip
            ax.scatter(mm.actual, yy, s=34, color=C[0], label="model at max − at min", zorder=3,
                       edgecolors=SURFACE, linewidths=1)  # fmt: skip
            lim = 1.15 * max(mm.actual.abs().max(), mm.linear.abs().max())
            if len(mx):
                ex_ = mx[mx.output == q].set_index("input").exact.reindex(order_p)
                inside = ex_.abs() <= lim
                ax.scatter(ex_[inside], yy[inside.values], s=30, color=C[2], marker="D", zorder=4,
                           label="exact derivative × range", edgecolors=SURFACE, linewidths=1)  # fmt: skip
                for y_, v in zip(yy[~inside.values], ex_[~inside]):  # off the axis: its value
                    if np.isfinite(v):
                        ax.annotate(f"{v:.3g}", (np.sign(v) * lim, y_), color=C[2], fontsize=7,
                                    ha="right" if v > 0 else "left", va="center",
                                    xytext=(-3 if v > 0 else 3, 0), textcoords="offset points",
                                    arrowprops={"arrowstyle": "-|>", "color": C[2], "lw": 0.8})  # fmt: skip
                relx = (ex_ - mm.actual).abs() / mm.actual.abs().max()
                say(f"   exact derivative × range, {q}: off by at most {relx.max():.1%} of the",
                    f"largest effect ({relx.idxmax()}); median {relx.median():.2%};",
                    f"default: median {((mm.linear - mm.actual).abs() / mm.actual.abs().max()).median():.2%}")  # fmt: skip
            ax.set_xlim(-lim, lim)
            ax.axvline(0, color=GREY, lw=0.8)
            unit = UNITS.get(q.replace("mean:", ""), "")
            ax.set(title=q.replace("mean:", "mean ").replace("_", " "), xlabel=unit or "loss")
            ax.set_yticks(yy, mm.index)
            if q != "loss":
                rel = (mm.linear - mm.actual).abs() / mm.actual.abs().max()
                say(f"one at a time, {exp}, {q}: largest effect {mm.actual.abs().max():.3g}",
                    f"({mm.actual.abs().idxmax()}); derivative × range off by at most",
                    f"{rel.max():.1%} of that ({rel.idxmax()})")  # fmt: skip
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8)
        fig.suptitle(f"One at a time: each parameter from −10 % to +10 %, {title}", x=0.01,
                     ha="left", fontweight="bold")  # fmt: skip
        save(fig, name)
        if exp == "month":
            dd = d[d.output == "loss"]
            say("month, loss: AD", dict(zip(dd.input, dd.ad_forward.round(6))))
            e = np.nanmin(np.abs(dd[fdc].values - dd.ad_forward.values[:, None]), axis=1) / np.abs(dd.ad_forward.values)  # fmt: skip
            say("month, loss gradient against the closest FD:", dict(zip(dd.input, e.round(4))))
            dm = d[d.output == "mean:latent_heat_flux"]
            e = np.nanmin(np.abs(dm[fdc].values - dm.ad_forward.values[:, None]), axis=1) / np.abs(dm.ad_forward.values)  # fmt: skip
            say("month, mean latent heat against the closest FD:", dict(zip(dm.input, e.round(4))))

    # the spread of the gradient over windows
    d1 = t[(t.experiment == "baseline") & (t.output == "loss")].drop_duplicates(["first_chunk", "input"], keep="last")  # fmt: skip
    d5 = t[(t.experiment == "window5") & (t.output == "loss")].drop_duplicates(["first_chunk", "input"], keep="last")  # fmt: skip
    dm = t[(t.experiment == "month") & (t.output == "loss")].drop_duplicates(["input"], keep="last")
    if len(d1):
        for d in (d1, d5, dm):
            d["g"] = d.ad_forward * (d.x_max - d.x_min)  # per range of the parameter
        d1 = d1[~d1.input.isin(zero)]
        order_p = d1.groupby("input").g.apply(lambda v: v.abs().median()).sort_values().index
        fig, ax = plt.subplots(figsize=(9, 5.6))
        rng = np.random.default_rng(0)
        for yy, p in enumerate(order_p):
            v = d1[d1.input == p].g.values
            ax.scatter(v, yy + rng.uniform(-0.18, 0.18, len(v)), s=9, color=C[0], alpha=0.55,
                       label="one-day windows" if yy == 0 else None, linewidths=0)  # fmt: skip
            if len(d5):
                v5 = d5[d5.input == p].g.values
                ax.scatter(v5, np.full(len(v5), yy), s=42, color=C[1], marker="D", zorder=3,
                           edgecolors=SURFACE, linewidths=0.8, label="five-day windows" if yy == 0 else None)  # fmt: skip
            if len(dm):
                ax.scatter(dm[dm.input == p].g.values, [yy], s=70, color=INK, marker="|", zorder=4,
                           label="the month" if yy == 0 else None)  # fmt: skip
        ax.set_yticks(np.arange(len(order_p)), order_p)
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set(xlabel="d loss / d parameter × range of the parameter (symmetric log axis)",
               title="The loss gradient over different windows of May")  # fmt: skip
        ax.legend(fontsize=8, loc="lower right")
        save(fig, "windows_spread.png")
        say("\n== Step 5: the spread of the loss gradient over 31 one-day windows")
        for p in order_p[::-1]:
            v = d1[d1.input == p].g.values
            same = max((v > 0).mean(), (v < 0).mean())
            line = f"    {p:24s} median {np.median(v):+.2e}  spread (IQR/|median|) {np.subtract(*np.percentile(v, [75, 25])) / abs(np.median(v)):.2f}  same sign {same:.0%}"  # fmt: skip
            if len(dm):
                line += f"  month {dm[dm.input == p].g.values[0]:+.2e}"
            say(line)

# =========================================================================== step 3
if want("thresholds"):
    files = sorted(glob.glob(str(RESULTS / "thresholds" / "*.npz")))
    if files:
        say("\n== Step 3: threshold sweeps")
    for f in files:
        z = np.load(f, allow_pickle=True)
        v, y, dy = z["values"], z["y"], z["dy"]
        show = [0, 1, 2, 3]
        fig, axes = plt.subplots(2, 4, figsize=(13, 5), sharex=True)
        slope = np.diff(y, axis=0) / np.diff(v)[:, None]
        vm = 0.5 * (v[1:] + v[:-1])
        for a, m in enumerate(show):
            var = str(z["vars"][m])
            axes[0, a].plot(v, y[:, m], color=C[0])
            axes[0, a].set(title=SHORT[var], ylabel=UNITS[var])
            axes[1, a].plot(vm, slope[:, m], color=GREY, lw=3, label="slope of the curve above")
            axes[1, a].plot(v, dy[:, m], color=C[1], lw=1.1, label="AD")
            lim = np.nanpercentile(np.abs(slope[:, m]), 98) * 2.5 + 1e-12
            axes[1, a].set(ylim=(-lim, lim), xlabel=str(z["xlabel"]), ylabel="derivative")
        axes[1, 0].legend(fontsize=8)
        tag = Path(f).stem
        fig.suptitle(f"{tag}: {z['what']} (day starting {str(z['start'])[:10]}, steps {int(z['first_step'])}"
                     f"–{int(z['first_step']) + int(z['steps']) - 1})", x=0.01, ha="left", fontweight="bold")  # fmt: skip
        fig.tight_layout()
        save(fig, f"threshold_{tag}.png")
        mid = 0.5 * (dy[1:] + dy[:-1])
        typical = np.nanmedian(np.abs(slope), axis=0) + 1e-300
        jump = np.abs(np.diff(y, axis=0)).max(axis=0) / (typical * np.diff(v).mean())
        off = (np.abs(slope - mid) > 0.05 * (np.abs(mid) + np.abs(slope)).max(axis=0)).mean(axis=0)
        say(f"{tag}: NaN values {int(np.isnan(y).sum())}, NaN derivatives {int(np.isnan(dy).sum())};",
            f"share of points with exactly zero derivative {np.round((dy[:, show] == 0).mean(axis=0), 2)};",
            f"largest step between neighbours / typical step {np.round(jump[show], 1)};",
            f"share of intervals where AD and the curve's slope differ by > 5 % of the largest {np.round(off[show], 3)};",
            f"largest |AD| / median |AD| {np.round(np.nanmax(np.abs(dy[:, show]), axis=0) / (np.nanmedian(np.abs(dy[:, show]), axis=0) + 1e-300), 1)}")  # fmt: skip

# =========================================================================== step 4
if want("parameters"):
    for days in (1, 31):
        files = sorted(glob.glob(str(RESULTS / "sweeps" / f"*_{days}d_00.npz")))
        if not files:
            continue
        say(f"\n== Step 4: parameter sweeps, {days}-day loss")
        zs = [np.load(f, allow_pickle=True) for f in files]
        zs = [z for z in zs if np.ptp(z["q"][:, 0]) > 0]
        ncol = 6
        nrow = int(np.ceil(len(zs) / ncol))
        fig1, ax1 = plt.subplots(nrow, ncol, figsize=(14, 2.3 * nrow))
        fig2, ax2 = plt.subplots(nrow, ncol, figsize=(14, 2.3 * nrow))
        for ax in list(ax1.ravel()) + list(ax2.ravel()):
            ax.set_visible(False)
        stats = []
        for a, z in enumerate(zs):
            v, L, dL = z["values"], z["q"][:, 0], z["dq"][:, 0]
            rel = (v - z["x0"]) / abs(float(z["x0"])) * 100
            ax = ax1.ravel()[a]
            ax.set_visible(True)
            ax.plot(rel, L, color=C[0])
            ax.axvline(0, color=GREY, lw=0.8)
            ax.axvline(
                (z["x_true"] - z["x0"]) / abs(float(z["x0"])) * 100, color=C[1], lw=0.9, ls="--"
            )
            ax.set_title(str(z["param"]), fontsize=8.5)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            # slopes at three scales: neighbours, a tenth of the range, the whole range
            n = len(v)
            fine = np.diff(L) / np.diff(v)
            k = max(1, n // 20)  # half of a tenth of the range
            wide = np.array([np.polyfit(v[max(0, i - k) : i + k + 1], L[max(0, i - k) : i + k + 1], 1)[0]
                             for i in range(n)])  # fmt: skip
            secant = (L[-1] - L[0]) / (v[-1] - v[0])
            ax = ax2.ravel()[a]
            ax.set_visible(True)
            ax.plot(
                0.5 * (rel[1:] + rel[:-1]),
                fine,
                color=GREY,
                lw=2.6,
                label="slope between neighbours",
            )
            ax.plot(rel, wide, color=C[2], lw=1.2, label="slope over a tenth of the range")
            ax.plot(rel, dL, color=C[1], lw=1.0, label="AD")
            ax.axhline(secant, color=C[0], lw=0.9, ls="--", label="slope over the whole range")
            ax.axvline(0, color=GREY, lw=0.8)
            lim = np.percentile(np.abs(wide), 100) * 1.6 + 1e-300
            ax.set_ylim(-lim, lim)
            ax.set_title(str(z["param"]), fontsize=8.5)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            big = np.abs(wide).max() + 1e-300
            mid = 0.5 * (dL[1:] + dL[:-1])
            stats.append(dict(
                param=str(z["param"]), loss_min=L.min(), loss_max=L.max(),
                rough=np.sqrt(np.mean((dL - wide) ** 2)) / big,
                ad_vs_fine=np.median(np.abs(mid - fine)) / big,
                ad_vs_fine_worst=np.max(np.abs(mid - fine)) / big,
                sign_right=np.mean(np.sign(dL) == np.sign(v - v[np.argmin(L)])),
                minima=int(np.sum((fine[1:] > 0) & (fine[:-1] < 0))),
                nan=int(np.isnan(L).sum() + np.isnan(dL).sum()),
            ))  # fmt: skip
        for fig, ax, ttl, name in [
            (fig1, ax1, "loss across each parameter's range (grey line: default; dashed: the twin's true value)", "loss"),
            (fig2, ax2, "slope of the loss at three scales, and AD", "slopes"),
        ]:  # fmt: skip
            fig.suptitle(
                f"{days}-day window from 1 May: {ttl}", x=0.01, ha="left", fontweight="bold"
            )
            fig.supxlabel("parameter, % from its default", fontsize=9, color=INK2)
            fig.tight_layout()
        h, lb = ax2.ravel()[0].get_legend_handles_labels()
        fig2.legend(h, lb, loc="upper right", ncol=4, fontsize=8)
        save(fig1, f"sweep_loss_{days}d.png")
        save(fig2, f"sweep_slopes_{days}d.png")
        s = pd.DataFrame(stats).set_index("param")
        say("columns: rough = rms(AD − slope over a tenth of the range) / largest such slope;",
            "ad_vs_fine = median |AD − slope between neighbours| / the same; sign_right = share of the",
            "range where AD points to the minimum of the curve; minima = interior local minima")  # fmt: skip
        say(s.round(4).to_string())

    for days in (1, 31):
        f = RESULTS / "hessian" / f"{days}d_00.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        names, s = z["names"], z["scale"]
        live = np.abs(z["grad"]) + np.abs(np.diag(z["gauss_newton"])) > 0
        names, s = names[live], s[live]
        say(f"\n== Step 4: Hessian of the {days}-day loss (per fraction of each parameter's range)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
        eig = {}
        for key, lab, col in [("gauss_newton", "Gauss–Newton (forward mode)", C[0]),
                              ("fd_1e3", "differences of AD gradients, step 0.1 %", C[1]),
                              ("fd_1e2", "differences of AD gradients, step 1 %", C[2])]:  # fmt: skip
            H = z[key][np.ix_(live, live)] * np.outer(s, s)
            H = 0.5 * (H + H.T)
            w, V = np.linalg.eigh(H)
            eig[key] = (w[::-1], V[:, ::-1])
            pos = w[::-1]
            axes[0].plot(
                np.arange(1, len(pos) + 1), np.abs(pos), marker="o", ms=4, color=col, label=lab
            )
            say(f"  {lab}: largest {pos[0]:.3e}, smallest {pos[-1]:.3e}, negative eigenvalues {int((pos < 0).sum())},",
                f"condition (largest / smallest positive) {pos[0] / pos[pos > 0].min():.2e}")  # fmt: skip
        axes[0].set(yscale="log", xlabel="eigenvalue number", ylabel="|eigenvalue|",
                    title=f"Curvature of the {days}-day loss")  # fmt: skip
        axes[0].legend(fontsize=8)
        w, V = eig["gauss_newton"]
        order = np.argsort(-np.abs(V[:, 0]))
        x = np.arange(len(names))
        axes[1].bar(
            x - 0.2, np.abs(V[order, 0]), width=0.38, color=C[0], label="stiffest direction"
        )
        axes[1].bar(
            x + 0.2, np.abs(V[order, -1]), width=0.38, color=C[1], label="sloppiest direction"
        )
        axes[1].set_xticks(x, names[order], rotation=60, ha="right", fontsize=7.5)
        axes[1].set(ylabel="|component|", title="Which parameters the extreme directions are made of (Gauss–Newton)")  # fmt: skip
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="x", visible=False)
        save(fig, f"hessian_{days}d.png")
        for j, lab in [(0, "stiffest"), (1, "second"), (-1, "sloppiest")]:
            top = np.argsort(-np.abs(V[:, j]))[:4]
            say(f"  {lab} direction (eigenvalue {w[j]:.2e}):",
                ", ".join(f"{names[i]} {V[i, j]:+.2f}" for i in top))  # fmt: skip
        g = z["grad"][live] * s
        say("  gradient per range:", dict(zip(names, np.round(g, 6))))
        a, bq = z["fd_1e3"][np.ix_(live, live)], z["fd_1e2"][np.ix_(live, live)]
        say("  the two difference steps disagree by (largest element difference / largest element):",
            f"{np.abs((a - bq) * np.outer(s, s)).max() / np.abs(a * np.outer(s, s)).max():.3f};",
            "asymmetry of the 0.1 % one:", f"{np.abs((a - a.T) * np.outer(s, s)).max() / np.abs(a * np.outer(s, s)).max():.3f}")  # fmt: skip

# =========================================================================== step 5
if want("time"):
    sources = sorted(glob.glob(str(RESULTS / "horizon" / "*.npz")))
    sources += [str(p) for p in [RESULTS / "month" / "may2007_00_31.npz"] if p.exists()]
    for f in sources:
        z = np.load(f, allow_pickle=True)
        full = "groups" in z.files  # a horizon file: the start state too
        if full:
            names, groups, ad, y0, obs = z["names"], z["groups"], z["ad"], z["y0"], z["obs"]
            rng_, fdp, start = z["hi"] - z["lo"], z["fd_params"], str(z["start"])
        else:  # the month recorded by the harness: parameters only, more difference steps
            act = z["active"]
            names, groups, ad, y0, obs = z["names"][act], [], z["ad_step"][act], z["y0"], z["obs"]
            rng_, fdp, start = (z["hi"] - z["lo"])[act], z["fd_step"][:, act], "2007-05-01"
        P, n = len(names), y0.shape[1]
        say(f"\n== Step 5: horizon sweep, {Path(f).stem}: {n} steps from {start}")
        say("NaN in forward-mode derivatives:", int(np.isnan(ad).sum()))
        Ts = np.array(
            [t for t in (1, 2, 6, 12, 24, 48, 96, 240, 480, 960, 1488, 2160, n) if t <= n]
        )
        Ts = np.unique(Ts)

        def loss_grad(d, T):  # d: (direction, variable, step)
            var = np.var(obs[:3, :T], axis=1)  # a variable that does not vary yet is left out
            iv = np.where(var > 1e-12, 1.0 / np.maximum(var, 1e-12), 0.0)
            w = 2 * (y0[:3, :T] - obs[:3, :T]) * iv[:, None]
            return (d[:, :3, :T] * w).sum((1, 2)) / 3 / T

        gl = np.array([loss_grad(ad[:P], T) for T in Ts]) * rng_  # (T, parameter)
        live = np.abs(ad[:P]).max((1, 2)) > 0
        errs = {}
        for m, var in [(0, "latent_heat_flux"), (2, "gpp")]:
            a = np.array([ad[:P, m, :T].mean(1) for T in Ts])
            fds = np.array([[fdp[j, :, m, :T].mean(1) for T in Ts] for j in range(len(fdp))])
            e = np.nanmin(np.abs(fds - a[None]), axis=0) / (np.abs(a) + 1e-300)
            errs[var] = e[:, live]
        fl = np.array([[loss_grad(fdp[j], T) for T in Ts] for j in range(len(fdp))]) * rng_
        errs["loss"] = (np.nanmin(np.abs(fl - gl[None]), axis=0) / (np.abs(gl) + 1e-300))[:, live]
        fig, axes = plt.subplots(1, 3 if full else 2, figsize=(16 if full else 10.5, 3.9),
                                 gridspec_kw={"wspace": 0.32})  # fmt: skip
        days_ = Ts / 48
        top = np.argsort(-np.abs(gl[-1]))[:6]
        for i in range(P):
            if live[i] and i not in top:
                axes[0].plot(
                    days_, np.where(gl[:, i] == 0, np.nan, np.abs(gl[:, i])), color=GREY, lw=0.7
                )
        for cidx, i in enumerate(top):
            axes[0].plot(
                days_,
                np.where(gl[:, i] == 0, np.nan, np.abs(gl[:, i])),
                color=C[cidx],
                label=str(names[i]),
            )
        axes[0].set(xscale="log", yscale="log", xlabel="horizon (days)",
                    ylabel="|d loss / d parameter| × range",
                    title=f"Loss gradient against the horizon (from {start[:10]})")  # fmt: skip
        big = np.nanmax(np.abs(gl))
        axes[0].set_ylim(big * 1e-6, big * 3)
        axes[0].legend(fontsize=7.5, ncol=2, loc="lower right")
        for cidx, (k, e) in enumerate(errs.items()):
            e = np.where(np.isfinite(e) & (e > 0), e, np.nan)
            axes[1].plot(days_, np.nanmedian(e, axis=1), color=C[cidx], label=f"{k.replace('_', ' ')}: median")  # fmt: skip
            axes[1].plot(days_, np.nanmax(e, axis=1), color=C[cidx], ls=":", lw=1.2, label="worst parameter")  # fmt: skip
        axes[1].set(xscale="log", yscale="log", xlabel="horizon (days)", ylabel="|FD − AD| / |AD|",
                    title="AD against finite differences")  # fmt: skip
        axes[1].set_ylim(1e-8, 1e3)
        axes[1].legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        if full:
            # memory of the start state: latent heat's response to each group, daily largest
            edges = np.concatenate([[0], np.cumsum(z["chunk_steps"])])
            fds_ = z["fd_state"]
            mem = {}
            for a, g in enumerate(groups):
                d = np.abs(ad[P + a, 0]) * STATE_SCALE.get(str(g).split(" ")[0], 1.0)
                mem[str(g)] = np.array([d[i:j].max() for i, j in zip(edges[:-1], edges[1:])])
            order_g = sorted(mem, key=lambda g: -mem[g][-1])
            for cidx, g in enumerate(order_g):
                axes[2].plot(np.arange(1, len(mem[g]) + 1), np.maximum(mem[g], 1e-20),
                             color=C[cidx % 8], ls="-" if cidx < 8 else "--", label=g.replace("_", " "))  # fmt: skip
            axes[2].set(yscale="log", xlabel="days after the start", ylabel="W m$^{-2}$ per unit*",
                        title="Memory: latent heat's response to the start state")  # fmt: skip
            axes[2].legend(fontsize=7, ncol=2)
            axes[2].set_ylim(bottom=1e-14)
            fig.text(0.99, -0.02, "*unit: 1 K, 100 Pa vapour pressure, 10 ppm CO$_2$, 0.1 MPa, 0.001 kg m$^{-2}$ leaf water",
                     ha="right", fontsize=7.5, color=INK2)  # fmt: skip
        save(fig, f"horizon_{Path(f).stem}.png")
        say("horizons (days):", np.round(days_, 3))
        for i in top:
            say(f"  |d loss/d {names[i]}| × range:", " ".join(f"{v:.2e}" for v in np.abs(gl[:, i])))
        for k, e in errs.items():
            say(f"  AD against FD, {k}: median", np.array2string(np.nanmedian(e, axis=1), precision=1),
                "worst", np.array2string(np.nanmax(e, axis=1), precision=1),
                "(worst parameter at the longest horizon:", names[live][np.nanargmax(e[-1])] + ")")  # fmt: skip
        if not full:
            continue
        say(
            "  memory of latent heat (day's largest |d LH / d start state|, scaled), day 1, 2, 5, 10, 30, last:"
        )
        for g in order_g:
            v = mem[g]
            pick = [v[i] for i in (0, 1, 4, 9, 29) if i < len(v)] + [v[-1]]
            say(f"    {g:32s}", " ".join(f"{x:9.2e}" for x in pick))
        # the state directions against finite differences, first day and whole run
        for a, g in enumerate(groups):
            d, fdg = ad[P + a, 0], fds_[:, a, 0]
            e1 = np.nanmin(np.abs(fdg[:, :48] - d[:48]).max(1)) / (np.abs(d[:48]).max() + 1e-300)
            say(
                f"    AD against FD for the start state, {g}: first day, worst step / largest: {e1:.2e}"
            )
        # reverse mode: gradient by state group for each horizon
        use, sn = z["state_in_use"], z["state_names"]
        gr = sorted({group_of(str(nm)) for nm, u in zip(sn, use) if u})
        tab = {}
        for g in gr:
            sel = np.array([u and group_of(str(nm)) == g for nm, u in zip(sn, use)])
            sc = STATE_SCALE.get(str(sn[sel][0]).split("[")[0].replace("canopy.", "").replace("soil.", "soil_"), 1.0)  # fmt: skip
            tab[g] = np.linalg.norm(z["d_state"][:, sel], axis=1) * sc
        tb = pd.DataFrame(tab, index=[f"{h} d" for h in z["horizons"]]).T
        say(
            "  reverse mode: size of d loss / d start state by group and horizon (scaled as above):"
        )
        say(tb.to_string(float_format=lambda v: f"{v:.2e}"))
        say(
            "  NaN in the reverse-mode start-state gradients:",
            int(np.isnan(z["d_state"][:, use]).sum()),
        )
        fig, ax = plt.subplots(figsize=(7, 3.8))
        tb2 = tb.loc[tb.iloc[:, -1].sort_values(ascending=False).index]
        for cidx, (g, row) in enumerate(tb2.iterrows()):
            ax.plot(z["horizons"], np.where(row.values > 0, row.values, np.nan), color=C[cidx % 8],
                    ls="-" if cidx < 8 else "--", marker="o", ms=3, label=g)  # fmt: skip
        ax.set(xscale="log", yscale="log", xlabel="horizon (days)", ylabel="|d loss / d start state| (scaled)",
               title="How much the loss over a horizon depends on the state at its start")  # fmt: skip
        ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
        save(fig, f"horizon_state_{Path(f).stem}.png")

    for f in sorted(glob.glob(str(RESULTS / "jacobian" / "day_*.npz"))):
        z = np.load(f, allow_pickle=True)
        J, sn = z["jac"], [str(s) for s in z["state_names"]]
        keep = np.array(["acclimation_steps" not in s for s in sn])  # a counter, not a state
        J, sn = J[:, keep][:, :, keep], [s for s, k in zip(sn, keep) if k]
        sc = np.array([STATE_SCALE.get(s.split("[")[0].replace("canopy.", "").replace("soil.", "soil_"), 1.0) for s in sn])  # fmt: skip
        Js = J * sc[None, None, :] / sc[None, :, None]  # in units of each variable's scale
        n = len(Js)
        say(f"\n== Step 5: one-step Jacobian, {Path(f).stem} (day starting {z['start']})")
        d = z["dots"]
        say("dot-product test <w, J v> = <J^T w, v>, worst relative difference over the day:",
            f"{(np.abs(d[:, 0] - d[:, 1]) / np.abs(d).max(axis=1)).max():.1e};",
            "single steps against the one-day chunk, largest state difference:", z["chunk_gap"])  # fmt: skip
        say("NaN in the Jacobians:", int(np.isnan(J).sum()))
        sv = np.array([np.linalg.svd(j, compute_uv=False)[0] for j in Js])
        rho = np.array([np.abs(np.linalg.eigvals(j)).max() for j in Js])
        Q, logs = np.eye(Js.shape[1]), np.zeros(Js.shape[1])
        for j in Js:
            Q, R = np.linalg.qr(j @ Q)
            logs += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
        lyap = np.sort(logs / n)[::-1]
        prop = np.linalg.multi_dot(list(Js[::-1]))
        gs = [group_of(s) for s in sn]
        gl_ = list(dict.fromkeys(gs))
        gain = np.array([[np.linalg.norm(prop[np.ix_([g == a for g in gs], [g == b for g in gs])], 2)
                          for b in gl_] for a in gl_])  # fmt: skip
        fig, axes = plt.subplots(
            1, 3, figsize=(17, 4), gridspec_kw={"width_ratios": [1.1, 1, 1.25], "wspace": 0.45}
        )
        hrs = (np.arange(n) + 1) / 2
        axes[0].plot(hrs, sv, color=C[0], label="largest singular value (one-step amplification)")
        axes[0].plot(hrs, rho, color=C[1], label="spectral radius (eigenvalue of largest size)")
        axes[0].axhline(1, color=GREY, lw=0.9)
        axes[0].set(
            yscale="log",
            xlabel="hours into the day (UTC)",
            title="One step of the state: growth factors",
        )
        axes[0].title.set_fontsize(9.5)
        axes[1].title.set_fontsize(9.5)
        axes[2].title.set_fontsize(9.5)
        axes[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        axes[1].plot(np.arange(1, len(lyap) + 1), np.exp(lyap * 48), color=C[0])
        axes[1].axhline(1, color=GREY, lw=0.9)
        axes[1].set(yscale="log", xlabel="direction number (sorted)", ylabel="growth factor over the day",
                    title="Growth or decay of perturbations over the day")  # fmt: skip
        axes[1].set_ylim(1e-12, max(10.0, 3 * np.exp(lyap[0] * 48)))
        im = axes[2].pcolormesh(np.log10(np.maximum(gain, 1e-12)), cmap="Blues", vmin=-8, vmax=1,
                                edgecolors=SURFACE, linewidth=0.8)  # fmt: skip
        axes[2].set_xticks(np.arange(len(gl_)) + 0.5, gl_, rotation=60, ha="right", fontsize=7.5)
        axes[2].set_yticks(np.arange(len(gl_)) + 0.5, gl_, fontsize=7.5)
        axes[2].set(xlabel="perturbed at the start of the day", ylabel="felt at its end",
                    title="One day later: gain, group to group (log10)")  # fmt: skip
        axes[2].grid(False)
        fig.colorbar(im, ax=axes[2], pad=0.01)
        save(fig, f"jacobian_{Path(f).stem}.png")
        say(f"largest singular value: median {np.median(sv):.2f}, largest {sv.max():.2f} at step {sv.argmax()};",
            f"spectral radius: median {np.median(rho):.3f}, largest {rho.max():.3f} at step {rho.argmax()}")  # fmt: skip
        say("growth factor over the day of the 5 slowest-decaying directions:", np.round(np.exp(lyap[:5] * 48), 4),
            "; directions decaying by less than 1e-3 over the day:", int((np.exp(lyap * 48) > 1e-3).sum()), "of", len(lyap))  # fmt: skip
        say("gain over the day, group to itself:", dict(zip(gl_, np.diag(gain).round(6))))

    tw = sorted(glob.glob(str(RESULTS / "twin" / "*.npz")))
    if tw:
        say(
            "\n== Step 5: round-off twins (start leaf temperatures changed by a relative 1e-12, 1e-9, 1e-6)"
        )
        fig, axes = plt.subplots(1, len(tw), figsize=(4.6 * len(tw), 3.6), sharey=True)
        for ax, f in zip(np.atleast_1d(axes), tw):
            z = np.load(f, allow_pickle=True)
            gap = np.abs(z["y"][1:, 0] - z["y"][:1, 0])  # latent heat
            dd = (np.arange(gap.shape[1]) + 1) / 48
            for a_, e in enumerate(z["sizes"][1:]):
                ax.plot(dd, np.maximum(gap[a_], 1e-16), color=C[a_], lw=1.2, label=f"{e:g}")
                say(f"  {Path(f).stem}, twin {e:g}: largest latent heat difference {gap[a_].max():.1e} W m-2",
                    f"at day {dd[gap[a_].argmax()]:.2f}")  # fmt: skip
            ax.set(
                yscale="log", xlabel="days after the start", title=f"from {str(z['start'])[:10]}"
            )
        np.atleast_1d(axes)[0].set_ylabel("|latent heat difference| (W m$^{-2}$)")
        np.atleast_1d(axes)[0].legend(title="start difference", fontsize=8, title_fontsize=8)
        fig.suptitle("Two runs that start a rounding error apart: how far latent heat drifts",
                     x=0.01, ha="left", fontweight="bold")  # fmt: skip
        save(fig, "twins.png")

    f = RESULTS / "checkpoint.npz"
    if f.exists():
        z = np.load(f)
        say(
            "\n== Step 5: checkpoints (reverse-mode gradient of a two-day loss, parameters and start state)"
        )
        for k in z.files:
            say(f"  {k.replace('_', ' ')}: largest relative difference from the default",
                f"{np.abs(z[k] - z['default']).max() / np.abs(z['default']).max():.1e}")  # fmt: skip

# ============================================== the default against the exact derivative
EXD, BAS = RESULTS / "exact_water_stress", RESULTS / "baseline"
DEF_C, EX_C, REF_C = C[1], C[0], GREY  # default, exact, the reference (model or FD)
if want("compare") and len(glob.glob(str(EXD / "may2007_*_01.npz"))) >= 31:
    say("\n== The default (stabilized) against the exact water-stress derivative")
    days = range(31)
    zb = [np.load(BAS / f"may2007_{k:02d}_01.npz", allow_pickle=True) for k in days]
    ze = [np.load(EXD / f"may2007_{k:02d}_01.npz", allow_pickle=True) for k in days]
    names = list(zb[0]["names"])
    psi_min = np.array([z["label_top_leaf_water_potential"].min() for z in zb])

    # 1. one day without stress and one with: every step
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [0.7, 1, 1]})  # fmt: skip
    for col, k in enumerate([2, 8]):
        hrs = (np.arange(zb[k]["ad_step"].shape[2]) + 1) / 2
        psi = zb[k]["label_top_leaf_water_potential"]
        ax = axes[0, col]
        ax.plot(hrs, psi, color=INK2, lw=1.5)
        ax.axhline(-1.6, color=C[7], lw=1, ls="--")
        ax.text(0.3, -1.57, "ψ50", color=C[7], fontsize=8, va="bottom")
        ax.set_ylim(-2.05, 0.05)
        ax.set_title(f"{k + 1} May: " + ("no water stress" if col == 0 else "water stress in the afternoon"))  # fmt: skip
        for row, (p, m, lab) in enumerate([("vcmax25", 0, "d latent heat / d Vcmax"),
                                           ("psi50_gs", 0, "d latent heat / d ψ50")], 1):  # fmt: skip
            i, ax = names.index(p), axes[row, col]
            fdv = best_fd(ze[k]["ad_step"][i], ze[k]["fd_step"][:, i])
            ax.plot(
                hrs, fdv[m], color=REF_C, lw=4.5, label="finite difference (the model's own slope)"
            )
            ax.plot(hrs, ze[k]["ad_step"][i, m], color=EX_C, lw=1.3, label="exact derivative")
            ax.plot(hrs, zb[k]["ad_step"][i, m], color=DEF_C, lw=1.8, label="default derivative")
            ax.axvspan(hrs[psi < -1.43].min() if (psi < -1.43).any() else 0,
                       hrs[psi < -1.43].max() if (psi < -1.43).any() else 0, color="#f3e9e2", lw=0, zorder=0)  # fmt: skip
            ax.set_title(lab, fontweight="normal")
    axes[0, 0].set_ylabel("top-leaf water\npotential (MPa)")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[2]:
        ax.set_xlabel("hours into the day (UTC; local noon near 20 h)")
    fig.suptitle("Step by step: both derivatives are right without stress; under stress the exact one "
                 "is the model's slope, and erratic", x=0.01, ha="left", fontweight="bold")  # fmt: skip
    save(fig, "compare_1_two_days.png")

    # 2. by regime: share of derivatives within 1 % of finite differences
    order = ["night", "sunrise ±1 h", "day, no stress", "day, water stress", "sunset ±1 h", "wet canopy"]  # fmt: skip
    share = {}
    for key, zs in [("default", zb), ("exact", ze)]:
        acc = {r: [] for r in order}
        for k in days:
            keep = zb[k]["active"] & (np.abs(zb[k]["ad_step"]).max((1, 2)) > 0)
            ad = zs[k]["ad_step"][keep]
            bf = best_fd(ad, zs[k]["fd_step"][:, keep])
            err = np.abs(bf - ad) / (np.abs(ad).max(axis=2, keepdims=True) + 1e-300)
            reg = regimes(zb[k])
            for r in order:
                if (reg == r).any():
                    acc[r].append((err[:, :, reg == r] < 1e-2).ravel())
        share[key] = np.array([np.concatenate(acc[r]).mean() for r in order])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.19, share["exact"], height=0.34, color=EX_C, label="exact derivative")
    ax.barh(y - 0.19, share["default"], height=0.34, color=DEF_C, label="default derivative")
    for yy, a, b_ in zip(y, share["exact"], share["default"]):
        ax.text(a + 0.01, yy + 0.19, f"{a:.0%}", va="center", fontsize=8, color=INK2)
        ax.text(b_ + 0.01, yy - 0.19, f"{b_:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, order)
    ax.set(xlim=(0, 1.12), xlabel="share of step derivatives within 1 % of finite differences (31 days)",
           title="Is it the model's own slope? By regime of the time step")  # fmt: skip
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "compare_2_regimes.png")
    say("within 1 % of FD by regime:", {r: (round(a, 3), round(b_, 3)) for r, a, b_ in zip(order, share["exact"], share["default"])}, "(exact, default)")  # fmt: skip

    # 3. one-day windows: does derivative x range predict the model's change over the range?
    t = pd.read_csv(RESULTS / "table.csv")
    pars = ["iota", "vcmax25", "psi50_gs", "plant_conductance", "leaf_reflectance[1]", "root_resistivity"]  # fmt: skip
    q = "mean:latent_heat_flux"
    tabs = {}
    for key, exp in [("default", "baseline"), ("exact", "exact_water_stress")]:
        d = t[(t.experiment == exp) & (t.chunks == 1) & (t.output == q) & t.input.isin(pars)]
        d = d.drop_duplicates(["first_chunk", "input"], keep="last")
        d = d.assign(pred=d.ad_forward * (d.x_max - d.x_min), actual=d.y_at_max - d.y_at_min)
        tabs[key] = (d.pivot(index="input", columns="first_chunk", values="pred").loc[pars],
                     d.pivot(index="input", columns="first_chunk", values="actual").loc[pars])  # fmt: skip
    fig, axes = plt.subplots(len(pars) + 1, 1, figsize=(10.5, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [0.8] + [1] * len(pars)})  # fmt: skip
    dd = np.arange(1, 32)
    axes[0].bar(dd, psi_min, color=np.where(psi_min < -1.6, C[7], GREY), width=0.7)
    axes[0].axhline(-1.6, color=C[7], lw=1, ls="--")
    axes[0].set(ylabel="lowest top-leaf\npotential (MPa)", ylim=(-2.1, -1.0),
                title="How stressed each day is (red: below ψ50)")  # fmt: skip
    for ax, p in zip(axes[1:], pars):
        actual = tabs["default"][1].loc[p].values
        ax.plot(
            dd, actual, color=REF_C, lw=5, label="model at max − at min", solid_capstyle="round"
        )
        lim = 2.2 * np.abs(actual).max()
        ex_ = tabs["exact"][0].loc[p].values
        off = np.abs(ex_) > lim
        ax.plot(dd, np.clip(ex_, -lim, lim), color=EX_C, lw=1.1)
        ax.plot(dd[~off], ex_[~off], color=EX_C, lw=0, marker="D", ms=3.5, label="exact derivative × range")  # fmt: skip
        ax.plot(dd[off & (ex_ > 0)], np.full((off & (ex_ > 0)).sum(), 0.93 * lim), color=EX_C, lw=0,
                marker="^", ms=7, label="exact, off the axis" if p == pars[-1] else None)  # fmt: skip
        ax.plot(dd[off & (ex_ < 0)], np.full((off & (ex_ < 0)).sum(), -0.93 * lim), color=EX_C, lw=0, marker="v", ms=7)  # fmt: skip
        ax.plot(dd, tabs["default"][0].loc[p].values, color=DEF_C, lw=1.3, marker="o", ms=3.5, label="default derivative × range")  # fmt: skip
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color=GREY, lw=0.7)
        ax.set_title(p, fontweight="normal")
        ax.set_ylabel("W m$^{-2}$")
        for key in ("exact", "default"):
            e = np.abs(tabs[key][0].loc[p].values - actual) / np.abs(actual).max()
            say(f"  one-day windows, {p}, {key}: derivative × range off by median {np.median(e):.1%},",
                f"worst day {e.max():.0%} of the largest effect; right sign on {(np.sign(tabs[key][0].loc[p].values) == np.sign(actual)).sum()} of 31 days")  # fmt: skip
    axes[-1].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.38))
    axes[-1].set_xlabel("day of May 2007 (one-day windows)")
    fig.suptitle("One day at a time: does the derivative predict the change of mean latent heat over ±10 %?",
                 x=0.01, ha="left", fontweight="bold", y=0.925)  # fmt: skip
    save(fig, "compare_3_daily_prediction.png")

    # 4. the month: the derivative of the running mean as the window grows
    fm, fx = RESULTS / "month" / "may2007_00_31.npz", RESULTS / "month_exact" / "may2007_00_31.npz"
    if fm.exists() and fx.exists():
        zm, zx = np.load(fm, allow_pickle=True), np.load(fx, allow_pickle=True)
        nm = list(zm["names"])
        rng_ = zm["hi"] - zm["lo"]
        nst = zm["y0"].shape[1]
        tt = (np.arange(nst) + 1) / 48
        run = lambda a: np.cumsum(a) / np.arange(1, nst + 1)  # noqa: E731  mean from the start
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
        for col, p in enumerate(["vcmax25", "iota", "psi50_gs"]):
            i = nm.index(p)
            ax = axes[0, col]
            ax.plot(
                tt, np.maximum(np.abs(zx["ad_step"][i, 0]), 1e-6), color=EX_C, lw=0.9, label="exact"
            )
            ax.plot(
                tt,
                np.maximum(np.abs(zm["ad_step"][i, 0]), 1e-6),
                color=DEF_C,
                lw=0.9,
                label="default",
            )
            ax.set(yscale="log", title=f"|d latent heat / d {p}| at every step")
            ax.set_ylim(bottom=1e-4 * np.abs(zm["ad_step"][i, 0]).max())
            ax = axes[1, col]
            actual = run(zm["hi_step"][i, 0] - zm["lo_step"][i, 0])
            lim = 3 * np.abs(actual[48:]).max()
            ax.plot(tt, actual, color=REF_C, lw=5, label="model at max − at min")
            ax.plot(tt, np.clip(run(zx["ad_step"][i, 0]) * rng_[i], -lim, lim), color=EX_C, lw=1.4, label="exact derivative × range")  # fmt: skip
            ax.plot(tt, run(zm["ad_step"][i, 0]) * rng_[i], color=DEF_C, lw=1.6, label="default derivative × range")  # fmt: skip
            ax.set_ylim(-lim, lim)
            ax.axhline(0, color=GREY, lw=0.7)
            ax.set(title=f"effect of ±10 % of {p}",
                   xlabel="days from 1 May")  # fmt: skip
            ax.title.set_fontweight("normal")
            ex_run = run(zx["ad_step"][i, 0]) * rng_[i]
            bad = np.flatnonzero(np.abs(ex_run - actual) > np.abs(actual[-1]))
            say(f"  month, {p}: the exact derivative × range of the running mean leaves the model's",
                f"change by more than the month's effect from day {tt[bad[bad > 96][0]]:.1f}" if (bad > 96).any() else "never")  # fmt: skip
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8, loc="lower left")
        axes[0, 0].set_ylabel("W m$^{-2}$ per unit")
        axes[1, 0].set_ylabel(
            "mean latent heat since 1 May, W m$^{-2}$\n(exact: clipped at the axis)"
        )
        fig.suptitle("Longer windows: the exact derivative grows from one stressed day to the next; "
                     "the default stays with the model", x=0.01, ha="left", fontweight="bold")  # fmt: skip
        save(fig, "compare_4_month.png")

# ===================== the exact derivative against finite differences: every step of May
if want("scatter") and len(glob.glob(str(EXD / "may2007_*_01.npz"))) >= 31:
    say("\n== The exact derivative against finite differences, every step of May (one-day windows)")
    zb = [np.load(BAS / f"may2007_{k:02d}_01.npz", allow_pickle=True) for k in range(31)]
    ze = [np.load(EXD / f"may2007_{k:02d}_01.npz", allow_pickle=True) for k in range(31)]
    names, outs, hs = list(ze[0]["names"]), list(ze[0]["vars"]), list(ze[0]["fd_steps"])
    H = 1e-6  # one relative FD step for every point of the scatter
    pars = [("iota", "ι"), ("vcmax25", "Vcmax25"), ("psi50_gs", "ψ50"),
            ("plant_conductance", "plant conductance")]  # fmt: skip
    rows = ["latent_heat_flux", "sensible_heat_flux", "gpp", "canopy_top_air_temperature"]
    st = np.concatenate([z["label_top_leaf_water_potential"] < -1.43 for z in zb])

    def series(zs, key, i, m, j=None):
        return np.concatenate([z[key][i, m] if j is None else z[key][j, i, m] for z in zs])

    def agree(ad, fd, typ):
        """Within 1 % of the larger of the two, or both below a thousandth of the typical size."""
        return np.abs(ad - fd) <= 0.01 * np.maximum(np.abs(ad), np.abs(fd)) + 1e-3 * typ

    def sym_ticks(typ, lim):
        """0 and ±powers of ten every three decades from the typical size up to the limit."""
        e = np.arange(np.ceil(np.log10(typ)), np.log10(lim) + 1e-9, 3)
        return np.concatenate([-(10.0 ** e[::-1]), [0.0], 10.0**e])

    def tick_label(x):
        return "0" if x == 0 else f"{'-' if x < 0 else ''}1e{int(round(np.log10(abs(x))))}"

    fig = plt.figure(figsize=(12, 14.5))
    gs = fig.add_gridspec(len(rows) + 1, len(pars), height_ratios=[1] * len(rows) + [0.8],
                          hspace=0.5, wspace=0.38, top=0.95)  # fmt: skip
    share = {"other": np.zeros(len(hs)), "stressed": np.zeros(len(hs))}
    for c_, (p, plab) in enumerate(pars):
        i = names.index(p)
        width = ze[0]["hi"][i] - ze[0]["lo"][i]  # the ±10 % range: derivative × range
        for r_, v in enumerate(rows):
            m, ax = outs.index(v), fig.add_subplot(gs[r_, c_])
            ad = series(ze, "ad_step", i, m) * width
            typ = np.quantile(np.abs(series(zb, "ad_step", i, m)), 0.95) * width
            for j, h in enumerate(hs):
                ok = agree(ad, series(ze, "fd_step", i, m, j) * width, typ)
                share["other"][j] += ok[~st].mean() / (len(rows) * len(pars))
                share["stressed"][j] += ok[st].mean() / (len(rows) * len(pars))
            fd = series(ze, "fd_step", i, m, hs.index(H)) * width
            ok = agree(ad, fd, typ)
            ax.scatter(fd[~st], ad[~st], s=7, color=C[0], alpha=0.6, lw=0, label="other steps")
            ax.scatter(fd[st], ad[st], s=7, color=C[7], alpha=0.6, lw=0,
                       label="top leaves below -1.43 MPa (water stress)")  # fmt: skip
            lim = max(np.abs(fd).max(), np.abs(ad).max()) * 2
            ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1, zorder=0)
            ax.set_xscale("symlog", linthresh=typ)
            ax.set_yscale("symlog", linthresh=typ)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
            tk = sym_ticks(typ, lim)
            ax.set_xticks(tk, [tick_label(x) for x in tk], rotation=90)
            ax.set_yticks(tk, [tick_label(x) for x in tk])
            ax.minorticks_off()
            ax.text(0.03, 0.97, f"agree: {ok[~st].mean():.0%} other\n{ok[st].mean():.0%} stressed",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)  # fmt: skip
            ax.tick_params(labelsize=7)
            if r_ == 0:
                ax.set_title(f"d / d {plab}", fontweight="bold")
            if c_ == 0:
                ax.set_ylabel(f"{SHORT[v]} ({UNITS[v]})\nexact derivative × range")
            if r_ == len(rows) - 1:
                ax.set_xlabel(f"finite difference × range (step {H:g})")
            say(f"  {SHORT[v]:>16} / {p:<18} agree: other {ok[~st].mean():6.1%}, "
                f"stressed {ok[st].mean():6.1%}; largest |exact| {np.abs(ad).max():.1e}, "
                f"largest |FD| {np.abs(fd).max():.1e}")  # fmt: skip
    ax = fig.add_subplot(gs[len(rows), :2])
    ax.plot(hs, share["other"], "o-", color=C[0], label="other steps")
    ax.plot(hs, share["stressed"], "o-", color=C[7], label="water-stressed steps")
    ax.set(xscale="log", ylim=(0, 1.05), xlabel="finite-difference step (relative to the parameter)",
           ylabel="share that agrees", title="Agreement against the size of the nudge (all 16 panels)")  # fmt: skip
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="lower left")
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower left",
               bbox_to_anchor=(0.52, 0.2), fontsize=8, markerscale=2.5)  # fmt: skip
    for h, a_, b_ in zip(hs, share["other"], share["stressed"]):
        say(f"  FD step {h:g}: agree other {a_:.1%}, stressed {b_:.1%}")
    fig.suptitle("The exact derivative against finite differences: every 30-min step of May "
                 "(31 one-day windows, 1487 steps per panel)", x=0.01, y=0.975, ha="left",
                 fontweight="bold")  # fmt: skip
    fig.text(0.52, 0.06, "Axes: the derivative of the step's output times the parameter's ±10 %\n"
             "range (the linearized change across the range), linear near zero, logarithmic\n"
             "beyond the typical size. Finite differences: central. 'Agree': within 1 % of the\n"
             "larger of the two, or both below a thousandth of the typical size (95th percentile\n"
             "of the default derivative). Setting water_stress_derivative: exact. Without stress\n"
             "any nudge from 1e-5 to 1e-7 agrees; under stress the model's slope holds only for\n"
             "ever smaller nudges. The points off the line are almost all from the afternoons of\n"
             "21, 22, 29 and 30 May.", fontsize=8, color=INK2,
             va="bottom")  # fmt: skip
    save(fig, "exact_vs_fd_scatter.png")

# ============ the same for two days, the four parameters in one panel: unstressed against stressed
if want("scatter_days") and len(glob.glob(str(EXD / "may2007_*_01.npz"))) >= 31:
    say("\n== The exact derivative against finite differences on 3 May (no stress) and 9 May")
    H = 1e-6
    days = [(3, "3 May: no water stress"), (9, "9 May: water stress in the afternoon")]
    pars = [("iota", "ι"), ("vcmax25", "Vcmax25"), ("psi50_gs", "ψ50"),
            ("plant_conductance", "plant conductance")]  # fmt: skip
    rows = ["latent_heat_flux", "sensible_heat_flux", "gpp", "canopy_top_air_temperature"]
    fig, axes = plt.subplots(len(rows), len(days), figsize=(8.6, 14))
    for c_, (d, dlab) in enumerate(days):
        z = np.load(EXD / f"may2007_{d - 1:02d}_01.npz", allow_pickle=True)
        names, outs = list(z["names"]), list(z["vars"])
        j = list(z["fd_steps"]).index(H)
        for r_, v in enumerate(rows):
            m, ax = outs.index(v), axes[r_, c_]
            lim = 0.0
            for n_, (p, plab) in enumerate(pars):
                i = names.index(p)
                width = z["hi"][i] - z["lo"][i]  # the ±10 % range: derivative × range
                ad, fd = z["ad_step"][i, m] * width, z["fd_step"][j, i, m] * width
                ax.scatter(fd, ad, s=16, color=C[n_], alpha=0.75, lw=0, label=plab)
                lim = max(lim, np.abs(ad).max(), np.abs(fd).max())
                say(f"  {d} May {SHORT[v]:>16} / {p:<18} largest |exact - FD| "
                    f"{np.abs(ad - fd).max():.1e} of {np.abs(ad).max():.1e}")  # fmt: skip
            lim *= 1.08
            ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1, zorder=0)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
            if r_ == 0:
                ax.set_title(dlab)
            if c_ == 0:
                ax.set_ylabel(f"{SHORT[v]} ({UNITS[v]})\nexact derivative × range")
            if r_ == len(rows) - 1:
                ax.set_xlabel(f"finite difference × range (step {H:g})")
    axes[0, 0].legend(fontsize=8, loc="upper left", title="parameter", title_fontsize=8,
                      markerscale=1.5)  # fmt: skip
    fig.suptitle("The exact derivative against finite differences: every 30-min step of one day",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.text(0.01, -0.01, "Both axes: the derivative of the step's output times the parameter's "
             "±10 % range (the linearized\nchange across the range). Finite differences: central, "
             f"relative step {H:g}. One-day windows from midnight UTC.\nSetting "
             "water_stress_derivative: exact.", fontsize=7.5, color=INK2, va="top")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "exact_vs_fd_two_days.png")

# ================================================================== step 7: calibration
cal = sorted(glob.glob(str(RESULTS / "calibration" / "*.npz")))
if want("calibration") and cal:
    say("\n== Step 7: twin calibration against daily means (Adam, from the defaults)")
    runs = {}
    for f in cal:
        pset, grad_, seed = Path(f).stem.split("_")
        runs[(pset, grad_, int(seed[1:]))] = np.load(f, allow_pickle=True)
    col = {"default": DEF_C, "exact": EX_C}
    psets = [p for p in ("strong", "all") if any(k[0] == p for k in runs)]
    titles = {"strong": "5 strong parameters", "all": "all 23 parameters"}
    fig, axes = plt.subplots(
        2, len(psets), figsize=(5.6 * len(psets), 6.6), sharex=True, squeeze=False
    )
    for j, pset in enumerate(psets):
        for (p_, g_, sd), z in sorted(runs.items()):
            if p_ != pset:
                continue
            err = np.abs(z["u"] - z["u_true"]) * 20  # in % of the parameter's value
            kw = {"color": col[g_], "ls": "-" if sd == 0 else "--", "lw": 1.6,
                  "label": f"{g_} derivative" if sd == 0 else None}  # fmt: skip
            axes[0, j].plot(z["loss"], **kw)
            axes[1, j].plot(err.mean(axis=1), **kw)
            best = int(np.argmin(z["loss"]))
            say(f"  {pset:6s} {g_:7s} twin {sd}: {len(z['loss']) - 1} iterations; loss {z['loss'][0]:.2e} ->",
                f"{z['loss'][-1]:.2e} (lowest {z['loss'][best]:.2e} at {best}); mean parameter error",
                f"{err[0].mean():.2f} % -> {err[-1].mean():.2f} % of the value; largest gradient norm {np.nanmax(np.linalg.norm(z['grad'], axis=1)):.1e}")  # fmt: skip
        axes[0, j].set(yscale="log", title=titles[pset] + ": the loss")
        axes[1, j].set(title="mean distance of the parameters from the truth", xlabel="iteration")
        axes[1, j].title.set_fontweight("normal")
        axes[1, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("loss (misfit of daily means)")
    axes[1, 0].set_ylabel("% of the parameter's value")
    axes[0, 0].legend(fontsize=8)
    fig.text(0.01, 0.005, "solid and dashed: two twins (two draws of the true parameters)", fontsize=8, color=INK2)  # fmt: skip
    fig.suptitle("Twin calibration on a month of daily means: the default against the exact derivative",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    save(fig, "calibration_progress.png")

    # where each strong parameter went
    keys = [k for k in sorted(runs) if k[0] == "strong"]
    if keys:
        names_ = [str(n) for n in runs[keys[0]]["names"]]
        seeds = sorted({k[2] for k in keys})
        fig, axes = plt.subplots(len(seeds), len(names_), figsize=(2.7 * len(names_), 2.7 * len(seeds)),
                                 sharex=True, squeeze=False)  # fmt: skip
        for r, sd in enumerate(seeds):
            for i, nme in enumerate(names_):
                ax = axes[r, i]
                for g_ in ("exact", "default"):
                    if ("strong", g_, sd) in runs:
                        z = runs[("strong", g_, sd)]
                        ax.plot(z["u"][:, i] * 20, color=col[g_], lw=1.5, label=f"{g_} derivative")
                        ax.axhline(z["u_true"][i] * 20, color=INK2, lw=1, ls="--")
                ax.axhline(0, color=GREY, lw=0.7)
                ax.set_ylim(-10.5, 10.5)
                if r == 0:
                    ax.set_title(nme)
                if i == 0:
                    ax.set_ylabel(f"twin {sd}\n% from the default")
                if r == len(seeds) - 1:
                    ax.set_xlabel("iteration")
        axes[0, 0].legend(fontsize=7.5, loc="lower left")
        fig.suptitle("The path of each parameter (dashed: the true value; the axis is the allowed range)",
                     x=0.01, ha="left", fontweight="bold")  # fmt: skip
        save(fig, "calibration_parameters.png")

# ================================================== step 8: how far the derivative holds
LOC = RESULTS / "locality"
SWEPT = ["iota", "vcmax25", "psi50_gs", "plant_conductance", "leaf_reflectance[1]", "shape_gs"]
PRETTY = {"iota": "ι", "vcmax25": "Vcmax25", "psi50_gs": "ψ50", "plant_conductance": "plant conductance",
          "leaf_reflectance[1]": "leaf reflectance, NIR", "shape_gs": "shape_gs",
          "clumping_index": "clumping index"}  # fmt: skip


def stress_share(z, values=None):
    """Share of daytime steps whose top leaves have a water-stress factor below 0.99, from
    a sweep or grid file's regime variables (the point's own ψ50 and shape)."""
    labels, lab = list(z["labels"]), z["lab"]
    names, x0 = list(z["names"]), z["x0"]
    zen = lab[..., labels.index("solar_zenith_angle"), :]
    psi = lab[..., labels.index("top_leaf_water_potential"), :]
    psi50 = np.full(lab.shape[:-2], x0[names.index("psi50_gs")])
    shape = np.full(lab.shape[:-2], x0[names.index("shape_gs")])
    if values is not None:
        for p, v in values.items():
            if p == "psi50_gs":
                psi50 = v
            if p == "shape_gs":
                shape = v
    day = zen < zen.max() - 1e-3
    f = 1 / (1 + np.clip(psi / psi50[..., None], 0, None) ** shape[..., None])
    return ((f < 0.99) & day).sum(-1) / day.sum(-1)


def one_sided(z, m):
    """(model change) / (derivative × change) on each side of the default, (side, step,
    parameter), and the model change."""
    x0 = z["x0"][z["idx"]]
    d = z["q"][..., m] - z["q0"][m]
    pred = z["dq"][None, None, :, m] * (z["x"] - x0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / np.where(pred == 0, np.nan, pred), d


if want("locality") and (LOC / "oat_31d_00.npz").exists():
    say("\n== Step 8: how far the derivative holds")
    mon = np.load(LOC / "oat_31d_00.npz")
    days = [np.load(f) for f in sorted(glob.glob(str(LOC / "oat_1d_*.npz")))]
    idx, lad = mon["idx"], mon["ladder"]
    names = [str(n) for n in mon["names"][idx]]
    for m, v in [(1, "latent heat"), (3, "GPP")]:
        r, d = one_sided(mon, m)
        say(
            f"month, mean {v}: (model change)/(derivative × change) at -50 -25 -10 -3 -1 | +1 +3 +10 +25 +50 %"
        )
        for j, n in enumerate(names):
            if np.nanmax(np.abs(d[:, -1, j])) < (0.5 if m == 1 else 0.05):
                continue
            say(f"  {n:22s}", " ".join(f"{x:5.2f}" for x in r[0, ::-1, j]), "|",
                " ".join(f"{x:5.2f}" for x in r[1, :, j]),
                f"| change at -50/+50 %: {d[0, -1, j]:6.1f} {d[1, -1, j]:6.1f}")  # fmt: skip

    # 1. the ladder: the change of May's latent heat, the model against the derivative
    show = SWEPT + ["clumping_index"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True)
    for ax, p in zip(axes.flat, show):
        j = names.index(p)
        _, d = one_sided(mon, 1)  # model change (side, step, parameter)
        dx = mon["x"][:, :, j] - mon["x0"][mon["idx"][j]]
        moved = np.abs(dx) > 0  # clumping: only smaller values (its default is its cap)
        keep = np.concatenate([moved[0, ::-1], [True], moved[1]])
        md = np.concatenate([d[0, ::-1, j], [0], d[1, :, j]])
        pr = mon["dq"][j, 1] * np.concatenate([dx[0, ::-1], [0], dx[1]])
        # the change actually made (a cap can shorten the last steps), in % of the default
        xp = 100 * np.concatenate([dx[0, ::-1], [0], dx[1]]) / abs(mon["x0"][mon["idx"][j]])
        ax.plot(xp[keep], md[keep], color=C[0], marker="o", ms=4, label="model")
        ax.plot(xp[keep], pr[keep], color=C[1], ls="--", lw=1.4, label="derivative × change")
        ax.axhline(0, color=GREY, lw=0.7)
        ax.axvline(0, color=GREY, lw=0.7)
        ax.set_title(PRETTY[p])
        ax.set_xticks([-50, -25, -10, 0, 10, 25, 50])
    axes.flat[-1].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("change of the parameter (% of its value)")
    for ax in axes[:, 0]:
        ax.set_ylabel("change of May's mean\nlatent heat (W m$^{-2}$)")
    h, lb = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, lb, loc="center")
    fig.suptitle("How far the derivative at the default holds: the model's change of May's latent heat "
                 "against derivative × change", x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "locality_ladder.png")

if want("locality") and len(glob.glob(str(LOC / "sweep_*.npz"))) >= 18:
    # 2. the sweeps: response, tangent, regime, loss
    fig, axes = plt.subplots(4, 6, figsize=(17, 11), sharex="col")
    for c_, p in enumerate(SWEPT):
        zs = {t: np.load(LOC / f"sweep_{p}_{t}.npz") for t in ("31d_00", "1d_02", "1d_08")}
        z = zs["31d_00"]
        i, v = int(z["i"]), z["values"]
        x0, xt = z["x0"][i], z["x_true"][i]
        for r_, (t, title) in enumerate([("31d_00", "May"), ("1d_02", "3 May (no stress)")]):
            ax, zz = axes[r_, c_], zs[t]
            k = int(np.argmin(np.abs(zz["values"] - x0)))
            lh, vv = zz["q"][:, 1], zz["values"]
            ax.plot(vv, lh, color=C[0], label="model")
            ax.plot(vv, lh[k] + zz["dq"][k, 1] * (vv - vv[k]), color=C[1], lw=1.4, ls="--",
                    label="derivative at the default")  # fmt: skip
            ax.plot([x0], [lh[k]], "o", color=INK, ms=5)
            if c_ == 0:
                ax.set_ylabel(f"{title}\nmean latent heat (W m$^{{-2}}$)")
            lo_, hi_ = lh.min(), lh.max()
            pad = 0.15 * (hi_ - lo_ + 1)
            ax.set_ylim(lo_ - pad, hi_ + pad)
        ax = axes[2, c_]
        for t, col, lab_ in [
            ("31d_00", C[0], "May"),
            ("1d_02", C[2], "3 May"),
            ("1d_08", C[3], "9 May"),
        ]:
            zz = zs[t]
            ax.plot(zz["values"], stress_share(zz, {p: zz["values"]}), color=col, label=lab_)
        ax.axvline(x0, color=INK2, lw=0.8)
        ax.set_ylim(-0.03, 1.03)
        if c_ == 0:
            ax.set_ylabel("share of daytime steps\nwith water stress")
        ax = axes[3, c_]
        loss = z["q"][:, 0]
        ax.plot(v, loss, color=C[0])
        ax.set_yscale("log")
        ax.axvline(x0, color=INK2, lw=0.8)
        ax.axvline(xt, color=INK2, lw=1, ls=":")
        imin = int(np.argmin(loss))
        towards = np.mean(np.sign(-z["dq"][:, 0]) == np.sign(v[imin] - v))
        ax.set_title(f"gradient points to the minimum: {100 * towards:.0f} %", fontsize=8,
                     fontweight="normal", color=INK2)  # fmt: skip
        ax.set_xlabel(PRETTY[p])
        if c_ == 0:
            ax.set_ylabel("May loss (twin)\nsolid: default, dotted: truth")
        axes[0, c_].set_title(PRETTY[p])
        say(f"sweep {p}: May loss minimum at {v[imin]:.3g} (truth {xt:.3g}); gradient towards it "
            f"{100 * towards:.0f} % of the range; stressed share {stress_share(z, {p: v})[[0, -1]]}")  # fmt: skip
    axes[0, 0].legend(fontsize=7.5, loc="lower left")
    axes[2, 0].legend(fontsize=7.5)
    fig.suptitle("Across the physical range: the model (blue) against the derivative at the default "
                 "(dashed), the share of stressed daytime steps, and the loss (dotted: the twin's truth)",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "locality_sweeps.png")

if want("locality") and len(glob.glob(str(LOC / "grid_*_31d_00.npz"))) >= 3:
    # 3. the grids: model, three predictions, loss
    pairs = ["iota__psi50_gs", "psi50_gs__plant_conductance", "vcmax25__iota"]
    fig, axes = plt.subplots(5, 3, figsize=(12, 17))
    for c_, pr in enumerate(pairs):
        z = np.load(LOC / f"grid_{pr}_31d_00.npz")
        va, vb, q, x0 = z["va"], z["vb"], z["q"], z["x0"]
        ia, ib = int(z["ia"]), int(z["ib"])
        pa, pb = [str(s) for s in z["pair"]]
        A, B = np.meshgrid(va - x0[ia], vb - x0[ib], indexing="ij")
        m, q0 = 1, z["q0"][1]
        g, H = z["grad"][:, m], z["hess"][:, :, m]
        H = 0.5 * (H + H.T)
        lin = q0 + g[0] * A + g[1] * B
        quad = lin + 0.5 * (H[0, 0] * A**2 + 2 * H[0, 1] * A * B + H[1, 1] * B**2)
        fa = np.array([np.interp(x0[ib], vb, q[i, :, m]) for i in range(len(va))])
        fb = np.array([np.interp(x0[ia], va, q[:, j, m]) for j in range(len(vb))])
        add = fa[:, None] + fb[None, :] - q0
        y = q[:, :, m]
        panels = [("model: mean latent heat (W m$^{-2}$)", y, "Blues", None),
                  ("model − sum of one-at-a-time changes", y - add, "RdBu_r", 60),
                  ("model − derivative (plane)", y - lin, "RdBu_r", 60),
                  ("model − derivative + second derivatives", y - quad, "RdBu_r", 60),
                  ("May loss (log10)", np.log10(q[:, :, 0]), "Blues_r", None)]  # fmt: skip
        for r_, (title, val, cmap, lim) in enumerate(panels):
            ax = axes[r_, c_]
            kw = dict(vmin=-lim, vmax=lim) if lim else {}
            im = ax.pcolormesh(vb, va, val, cmap=cmap, shading="nearest", **kw)
            if r_ == 0:
                ax.contour(vb, va, val, colors=SURFACE, linewidths=0.6, levels=8)
            ax.plot(x0[ib], x0[ia], "o", color=INK, ms=5)
            ax.plot(z["x_true"][ib], z["x_true"][ia], "x", color=INK, ms=6)
            ax.set_xlabel(PRETTY[pb])
            ax.set_ylabel(PRETTY[pa])
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.85)
            if r_ > 0 and r_ < 4:
                rms = np.sqrt(np.mean(val**2))
                ax.set_title(
                    f"{title}\nrms {rms:.1f}, largest {np.abs(val).max():.0f} W m$^{{-2}}$",
                    fontsize=8.5,
                )
                say(f"grid {pr}: {title}: rms {rms:.1f}, largest {np.abs(val).max():.0f} W m-2")
            else:
                ax.set_title(title, fontsize=8.5)
    fig.suptitle("Two parameters at once, May (dot: default; ×: the twin's truth). Rows 2-4: what each "
                 "prediction misses", x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "locality_grids.png")

(RESULTS / "summary.txt").write_text("\n".join(_lines) + "\n") if ONLY is None else None
sys.exit(0)
