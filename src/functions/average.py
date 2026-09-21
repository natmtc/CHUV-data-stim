"""Left/Right-averaged views: per muscle, mean of the L and R channels at each
intensity, with a shaded +/- standard deviation band. For latency and peak-to-peak."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plots import _base_side
from .quantify import peak_to_peak


def _lr_groups(muscles):
    """{'Biceps': ['L_BB', 'R_BB'], ...} grouping channels by muscle (ignoring side)."""
    groups = {}
    for m in muscles:
        b, _ = _base_side(m)
        groups.setdefault(b, []).append(m)
    return groups


def _plot_mean_sd(groups, amps, values, ylabel, ylim=None, save=None):
    """values[channel] = array over intensities. Plots mean +/- SD across L/R per muscle."""
    bases = list(groups)
    cmap = plt.cm.tab10
    colour = {b: cmap(i % 10) for i, b in enumerate(bases)}

    fig, ax = plt.subplots(figsize=(11, 7))
    for b, chs in groups.items():
        M = np.vstack([values[c] for c in chs])          # [n_side, n_amps]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns -> NaN
            mean = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0)
        c = colour[b]
        ax.fill_between(amps, mean - sd, mean + sd, color=c, alpha=0.20, lw=0)
        ax.plot(amps, mean, marker="o", color=c, lw=2, ms=9, mew=0)

    ax.set_xlabel("Stim amplitude (mA)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], linestyle="", marker="", label=b) for b in bases]
    leg = ax.legend(handles=handles, loc="upper left", ncol=2, frameon=True,
                    framealpha=0.9, fontsize=13, handlelength=0.0,
                    handletextpad=0.0, columnspacing=1.4)
    for txt, b in zip(leg.get_texts(), bases):
        txt.set_color(colour[b]); txt.set_fontweight("bold")
    fig.tight_layout()
    if save:
        import os; os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def latency_lr_mean(meta, t, sig, muscles, picks, ylim=(0, 30), save=None):
    """Latency vs intensity, Left/Right averaged per muscle, shaded +/- SD."""
    amps = np.array([m["amp_ma"] for m in meta])
    values = {m: np.array([r["t_peak"] for r in picks[m]], float) for m in muscles}
    _plot_mean_sd(_lr_groups(muscles), amps, values, "Latency (ms)", ylim=ylim, save=save)


def p2p_lr_mean(meta, t, sig, muscles, picks, window_ms=20.0, offset_ms=0.0, save=None):
    """Peak-to-peak vs intensity, Left/Right averaged per muscle, shaded +/- SD."""
    amps = np.array([m["amp_ma"] for m in meta])
    res = peak_to_peak(meta, t, sig, muscles, picks, window_ms, offset_ms)
    values = {m: np.array([r["p2p"] for r in res[m]], float) for m in muscles}
    _plot_mean_sd(_lr_groups(muscles), amps, values, "Peak-to-peak (mV)", save=save)


# ---------------------------------------------------------------------------
# before vs with lidocaine: Left arm | Right arm | L/R mean, one line per muscle
# ---------------------------------------------------------------------------
def compare_lr_curves(csv_before, csv_after, metric="p2p", window_ms=20.0, offset_ms=0.0,
                      results_dir="results", ylim=None,
                      labels=("before lidocaine", "with lidocaine"), save=None):
    """Three panels side by side - Left arm, Right arm, Left/Right mean - with one line
    per muscle (colour = muscle). Before lidocaine = solid line, filled markers; with
    lidocaine = dashed line, open markers. Legend on the right.

    metric: "p2p" (peak-to-peak, mV; needs window_ms / offset_ms) or "latency" (ms).
    Latencies come from the saved results/latency_<file>.csv of each recording.
    """
    import os
    from .io import load_run
    from .latency import load_latency_csv

    runs = []
    for path in (csv_before, csv_after):
        meta, t, sig = load_run(path)
        muscles = [c for c in sig if c != "Trigger A"]
        stem = os.path.splitext(os.path.basename(path))[0]
        latfile = os.path.join(results_dir, f"latency_{stem}.csv")
        if not os.path.exists(latfile):
            raise FileNotFoundError(f"No saved latencies for {stem} - pick and save them first.")
        picks = load_latency_csv(latfile)
        amps = np.array([m["amp_ma"] for m in meta])
        if metric == "latency":
            values = {m: np.array([r["t_peak"] for r in picks[m]], float) for m in muscles}
        else:
            res = peak_to_peak(meta, t, sig, muscles, picks, window_ms, offset_ms)
            values = {m: np.array([r["p2p"] for r in res[m]], float) for m in muscles}
        runs.append(dict(amps=amps, muscles=muscles, values=values))

    muscles = [m for m in runs[0]["muscles"] if m in runs[1]["muscles"]]
    groups = _lr_groups(muscles)
    bases = list(groups)
    cmap = plt.cm.tab10
    colour = {b: cmap(i % 10) for i, b in enumerate(bases)}
    style = [dict(ls="-", marker="o", mfc=None), dict(ls="--", marker="o", mfc="white")]
    ylabel = "Latency (ms)" if metric == "latency" else "Peak-to-peak (mV)"

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=True)
    panels = [("Left arm", "L"), ("Right arm", "R"), ("Left / Right mean", None)]
    for ax, (ttl, side) in zip(axes, panels):
        for b, chs in groups.items():
            for r, st in zip(runs, style):
                if side is None:                          # mean over L and R of that muscle
                    M = np.vstack([r["values"][c] for c in chs])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        y = np.nanmean(M, axis=0)
                else:
                    sel = [c for c in chs if _base_side(c)[1] == side]
                    if not sel:
                        continue
                    y = r["values"][sel[0]]
                ax.plot(r["amps"], y, color=colour[b], lw=2, ms=8,
                        mec=colour[b], mew=1.8, **st)
        ax.set_title(ttl, fontweight="bold", fontsize=15)
        ax.set_xlabel("Stim amplitude (mA)", fontsize=14)
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=14)
    if ylim:
        axes[0].set_ylim(*ylim)

    # legend on the right: muscle names in their colour, then the condition key
    handles = [Line2D([], [], color=colour[b], lw=3, label=b) for b in bases]
    handles += [Line2D([], [], color="0.3", lw=2, ls="-", marker="o", ms=8, mew=0, label=labels[0]),
                Line2D([], [], color="0.3", lw=2, ls="--", marker="o", ms=8, mfc="white", mec="0.3",
                       mew=1.8, label=labels[1])]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.995, 0.5),
                     frameon=False, fontsize=13, handlelength=2.2)
    for txt, b in zip(leg.get_texts()[:len(bases)], bases):
        txt.set_color(colour[b]); txt.set_fontweight("bold")
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
    return runs


def compare_per_muscle(csv_before, csv_after, metric="p2p", window_ms=20.0, offset_ms=0.0,
                       results_dir="results", ylim=None,
                       labels=("before lidocaine", "with lidocaine"), colours=None, save=None):
    """One panel per muscle, x = intensity. Gray = first condition, orange = second, blue = third.
    csv_before may be a list of files (then csv_after=None) for more than two conditions.
    Circle + solid = LEFT arm, triangle + dashed = RIGHT arm. At most 4 lines per panel.

    metric: "p2p" (mV, needs window_ms / offset_ms) or "latency" (ms).
    Latencies come from the saved results/latency_<file>.csv of each recording."""
    import os
    from .io import load_run
    from .latency import load_latency_csv
    from .burst import _conditions

    paths, labels, colours = _conditions(csv_before, csv_after, labels, colours)
    runs = []
    for path in paths:
        meta, t, sig = load_run(path)
        muscles = [c for c in sig if c != "Trigger A"]
        latfile = os.path.join(results_dir,
                               f"latency_{os.path.splitext(os.path.basename(path))[0]}.csv")
        if not os.path.exists(latfile):
            raise FileNotFoundError(f"No saved latencies for {path} - pick and save them first.")
        picks = load_latency_csv(latfile)
        amps = np.array([m["amp_ma"] for m in meta])
        if metric == "latency":
            values = {m: np.array([r["t_peak"] for r in picks[m]], float) for m in muscles}
        else:
            res = peak_to_peak(meta, t, sig, muscles, picks, window_ms, offset_ms)
            values = {m: np.array([r["p2p"] for r in res[m]], float) for m in muscles}
        runs.append(dict(amps=amps, muscles=muscles, values=values))

    muscles = [m for m in runs[0]["muscles"] if all(m in r["muscles"] for r in runs)]
    groups = _lr_groups(muscles)
    side_style = {"L": dict(marker="o", ls="-"), "R": dict(marker="^", ls="--"),
                  "": dict(marker="D", ls="-")}
    ylabel = "Latency (ms)" if metric == "latency" else "Peak-to-peak (mV)"

    n = len(groups); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.6 * nrow), squeeze=False,
                             sharex=True, sharey=(metric == "latency"))
    axes = axes.ravel()
    for i, (b, chs) in enumerate(groups.items()):
        ax = axes[i]
        for r, col in zip(runs, colours):
            for c in chs:
                side = _base_side(c)[1]
                ax.plot(r["amps"], r["values"][c], color=col, lw=2, ms=7, mew=0,
                        **side_style[side])
        ax.set_title(b, fontweight="bold")
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
        if i % ncol == 0:
            ax.set_ylabel(ylabel)
        if i >= n - ncol:
            ax.set_xlabel("Stim amplitude (mA)")
    for k in range(n, len(axes)):
        axes[k].axis("off")
        if k - ncol >= 0:
            axes[k - ncol].tick_params(labelbottom=True)
            axes[k - ncol].set_xlabel("Stim amplitude (mA)")

    handles = []
    for l, c in zip(labels, colours):
        handles += [Line2D([], [], color=c, lw=2, marker="o", ms=7, mew=0, label=f"{l} - left"),
                    Line2D([], [], color=c, lw=2, ls="--", marker="^", ms=7, mew=0, label=f"{l} - right")]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
    return runs
