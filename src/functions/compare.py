"""Compare peak-to-peak between two recordings (e.g. before vs with lidocaine),
averaged across all muscles (mean +/- SD), one curve per file."""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from .io import load_run, result_path
from .latency import load_latency_csv
from .quantify import peak_to_peak


def _load_res(csv_path, window_ms, offset_ms, results_dir, picks):
    """(amps, muscles, peak_to_peak-result) for one file, loading saved picks if needed."""
    meta, t, sig = load_run(csv_path)
    muscles = [c for c in sig if c != "Trigger A"]
    if picks is None:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        latfile = result_path(csv_path, "latency", results_dir)
        if not os.path.exists(latfile):
            raise FileNotFoundError(
                f"No saved latencies for this file:\n  {latfile}\n"
                f"Load this file as CSV in the notebook, pick its latencies, and save first.")
        picks = load_latency_csv(latfile)
    res = peak_to_peak(meta, t, sig, muscles, picks, window_ms, offset_ms)
    amps = [m["amp_ma"] for m in meta]
    return amps, muscles, res


SHORT = {"Deltoid med.": "Del", "Biceps": "BB", "Triceps long": "TB", "Trapezius": "Trap",
         "Thenar": "The", "Ext. digitorum": "ED", "Flex. digitorum": "FD", "Flex. carpi rad.": "FCR"}


def _short(channel):
    """'R_BB' -> 'BB R', 'L_The_ED_FD_FCR D' -> 'FCR L' (from the pretty label)."""
    from .labels import pretty
    lab = pretty(channel)
    side = "R" if lab.endswith("(R)") else "L" if lab.endswith("(L)") else ""
    base = lab[:-4].strip() if side else lab
    return f"{SHORT.get(base, base)} {side}".strip()


def _outlier_mask(v, k=1.5):
    """Tukey rule: outside [Q1 - k*IQR, Q3 + k*IQR]. With < 4 points nothing is an outlier."""
    v = np.asarray(v, float)
    if len(v) < 4:
        return np.zeros(len(v), bool)
    q1, q3 = np.nanpercentile(v, [25, 75]); iqr = q3 - q1
    return (v < q1 - k * iqr) | (v > q3 + k * iqr)


def _annotate(ax, xs, v, names, mask, colour, fontsize=8):
    for x, y, n, m in zip(xs, v, names, mask):
        if m:
            ax.annotate(n, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=fontsize,
                        color=colour, va="center", ha="left", fontweight="bold", zorder=6)


def _p2p_by_amp(amps, muscles, res):
    """{amp: array of p2p across muscles with NaNs dropped}."""
    out = {}
    for j, a in enumerate(amps):
        v = np.array([res[m][j]["p2p"] for m in muscles], float)
        out[a] = v[~np.isnan(v)]
    return out


def _p2p_by_amp_named(amps, muscles, res):
    """{amp: (values, short muscle names)} with NaNs dropped."""
    out = {}
    for j, a in enumerate(amps):
        v = np.array([res[m][j]["p2p"] for m in muscles], float)
        keep = ~np.isnan(v)
        out[a] = (v[keep], [_short(m) for m, k in zip(muscles, keep) if k])
    return out


def _p2p_across_muscles(csv_path, window_ms, offset_ms, results_dir="results", picks=None):
    """Return (amps, mean, sd) of peak-to-peak averaged across all muscles for one file.
    Uses `picks` if given, else loads results/latency_<stem>.csv."""
    meta, t, sig = load_run(csv_path)
    muscles = [c for c in sig if c != "Trigger A"]
    if picks is None:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        latfile = result_path(csv_path, "latency", results_dir)
        if not os.path.exists(latfile):
            raise FileNotFoundError(
                f"No saved latencies for this file:\n  {latfile}\n"
                f"Load this file as CSV in the notebook, pick its latencies, and save first.")
        picks = load_latency_csv(latfile)
    res = peak_to_peak(meta, t, sig, muscles, picks, window_ms, offset_ms)
    amps = np.array([m["amp_ma"] for m in meta])
    M = np.array([[r["p2p"] for r in res[m]] for m in muscles], float)   # [n_muscle, n_amp]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # all-NaN columns
        mean = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0)
    return amps, mean, sd


def compare_p2p_lidocaine(before_csv, after_csv, window_ms=20.0, offset_ms=2.0,
                          labels=("before lidocaine", "with lidocaine"),
                          colours=("0.45", "#f39c12"),
                          before_picks=None, after_picks=None,
                          results_dir="results", save=None):
    """Peak-to-peak averaged across all muscles (mean +/- SD) vs intensity, for two files."""
    a0, m0, s0 = _p2p_across_muscles(before_csv, window_ms, offset_ms, results_dir, before_picks)
    a1, m1, s1 = _p2p_across_muscles(after_csv, window_ms, offset_ms, results_dir, after_picks)

    fig, ax = plt.subplots(figsize=(11, 7))
    for a, mean, sd, lab, c in [(a0, m0, s0, labels[0], colours[0]),
                                (a1, m1, s1, labels[1], colours[1])]:
        ax.fill_between(a, mean - sd, mean + sd, color=c, alpha=0.22, lw=0)
        ax.plot(a, mean, marker="o", color=c, lw=2.4, ms=10, mew=0, label=lab)
    ax.set_xlabel("Stim amplitude (mA)", fontsize=18)
    ax.set_ylabel("Peak-to-peak (mV)  —  mean ± SD across muscles", fontsize=15)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=15, frameon=True)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def compare_p2p_violin(before_csv, after_csv, window_ms=20.0, offset_ms=2.0,
                       labels=("before lidocaine", "with lidocaine"),
                       colours=("0.62", "#f39c12"), edges=("0.30", "#b5651d"),
                       before_picks=None, after_picks=None,
                       results_dir="results", ylim=None, annotate="outliers", save=None):
    """Violin plots of peak-to-peak (all muscles pooled) at each intensity: two violins
    per intensity, before (gray) vs with lidocaine (orange), side by side.

    annotate: "outliers" -> label the dots outside Q1-1.5*IQR .. Q3+1.5*IQR with the muscle
              (BB R = right biceps, FCR L = left flexor carpi radialis, ...);
              "all" -> label every dot; None -> no labels."""
    from matplotlib.patches import Patch

    a0, mus0, res0 = _load_res(before_csv, window_ms, offset_ms, results_dir, before_picks)
    a1, mus1, res1 = _load_res(after_csv, window_ms, offset_ms, results_dir, after_picks)
    before = _p2p_by_amp_named(a0, mus0, res0)
    after = _p2p_by_amp_named(a1, mus1, res1)
    amps = sorted(set(a0) | set(a1))
    rng = np.random.default_rng(0)                       # reproducible jitter

    fig, ax = plt.subplots(figsize=(13, 7))
    dx, w = 0.22, 0.38

    def draw(by_amp, shift, colour, edge):
        data, pos = [], []
        for i, a in enumerate(amps):
            v, names = by_amp.get(a, (np.array([]), []))
            if len(v) >= 2:
                data.append(v); pos.append(i + shift)
            if len(v):                                   # individual muscle points
                xs = i + shift + rng.uniform(-0.055, 0.055, len(v))
                ax.scatter(xs, v, s=11, color=colour, edgecolor=edge,
                           linewidth=0.5, zorder=3, alpha=0.9)
                if annotate == "all":
                    _annotate(ax, xs, v, names, np.ones(len(v), bool), edge)
                elif annotate == "outliers":
                    _annotate(ax, xs, v, names, _outlier_mask(v), edge)
        if data:
            vp = ax.violinplot(data, positions=pos, widths=w,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(colour); body.set_edgecolor(edge)
                body.set_linewidth(1.8); body.set_alpha(0.55)
            if "cmedians" in vp:
                vp["cmedians"].set_color(edge); vp["cmedians"].set_linewidth(2.0)

    draw(before, -dx, colours[0], edges[0])
    draw(after, +dx, colours[1], edges[1])

    ax.set_xticks(range(len(amps))); ax.set_xticklabels(amps)
    ax.set_xlabel("Stim amplitude (mA)", fontsize=20)
    ax.set_ylabel("Peak-to-peak (mV)  —  all muscles", fontsize=18)
    ax.tick_params(labelsize=18)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=colours[0], alpha=0.6, label=labels[0]),
                       Patch(facecolor=colours[1], alpha=0.6, label=labels[1])],
              fontsize=15, frameon=True)
    if annotate:
        ax.text(0.01, 0.99, "labelled dots = outliers (Tukey 1.5 IQR): BB biceps, TB triceps long, Del deltoid, "
                "Trap trapezius, The thenar, ED ext. dig., FD flex. dig., FCR flex. carpi rad.  L/R = side",
                transform=ax.transAxes, fontsize=9, color="0.35", va="top")
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def compare_p2p_box(before_csv, after_csv, window_ms=20.0, offset_ms=2.0,
                    labels=("before lidocaine", "with lidocaine"),
                    colours=("0.6", "#f39c12"),
                    before_picks=None, after_picks=None,
                    results_dir="results", save=None):
    """Boxes = mean ± SD (across all muscles) with a mean line and SD whisker, two per
    intensity: before (gray) vs with lidocaine (orange), side by side."""
    from matplotlib.patches import Patch

    a0, mus0, res0 = _load_res(before_csv, window_ms, offset_ms, results_dir, before_picks)
    a1, mus1, res1 = _load_res(after_csv, window_ms, offset_ms, results_dir, after_picks)
    before = _p2p_by_amp_named(a0, mus0, res0)
    after = _p2p_by_amp_named(a1, mus1, res1)
    amps = sorted(set(a0) | set(a1))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(13, 7))
    dx, w = 0.22, 0.36

    def draw(by_amp, shift, colour):
        for i, a in enumerate(amps):
            v = by_amp.get(a, np.array([]))
            if len(v) == 0:
                continue
            pos = i + shift
            m, sd = float(np.mean(v)), float(np.std(v))
            # box = mean ± SD
            ax.bar(pos, 2 * sd, bottom=m - sd, width=w, color=colour, alpha=0.40,
                   edgecolor=colour, linewidth=1.3, zorder=2)
            # SD whisker (vertical line) + mean line
            ax.plot([pos, pos], [m - sd, m + sd], color=colour, lw=1.3, zorder=3)
            ax.hlines(m, pos - w / 2, pos + w / 2, color="black", lw=1.6, zorder=4)
            # individual muscle points
            xs = pos + rng.uniform(-0.06, 0.06, len(v))
            ax.scatter(xs, v, s=12, color=colour, edgecolor="white",
                       linewidth=0.3, zorder=5, alpha=0.85)

    draw(before, -dx, colours[0])
    draw(after, +dx, colours[1])

    ax.set_xticks(range(len(amps))); ax.set_xticklabels(amps)
    ax.set_xlabel("Stim amplitude (mA)", fontsize=18)
    ax.set_ylabel("Peak-to-peak (mV)  —  all muscles (mean ± SD)", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.grid(True, axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=colours[0], alpha=0.6, label=labels[0]),
                       Patch(facecolor=colours[1], alpha=0.6, label=labels[1])],
              fontsize=14, frameon=True)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def _load_picks(csv_path, results_dir, picks):
    """(muscles, picks) for one file, loading saved latency picks if none given."""
    meta, t, sig = load_run(csv_path)
    muscles = [c for c in sig if c != "Trigger A"]
    if picks is None:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        latfile = result_path(csv_path, "latency", results_dir)
        if not os.path.exists(latfile):
            raise FileNotFoundError(
                f"No saved latencies for this file:\n  {latfile}\n"
                f"Load this file as CSV in the notebook, pick its latencies, and save first.")
        picks = load_latency_csv(latfile)
    return muscles, picks


def _latency_by_amp_named(muscles, picks):
    """{amp: (latencies, short muscle names)} with NaNs dropped."""
    out = {}
    amps = sorted({r["amp"] for m in muscles for r in picks[m]})
    for a in amps:
        v, names = [], []
        for m in muscles:
            for r in picks[m]:
                if r["amp"] == a and r["t_peak"] == r["t_peak"]:
                    v.append(r["t_peak"]); names.append(_short(m))
        out[a] = (np.array(v, float), names)
    return out


def _latency_by_amp(muscles, picks):
    """{amp: array of picked latencies across muscles, NaNs dropped}."""
    amps = sorted({r["amp"] for m in muscles for r in picks[m]})
    out = {a: [] for a in amps}
    for m in muscles:
        for r in picks[m]:
            if r["t_peak"] == r["t_peak"]:
                out[r["amp"]].append(r["t_peak"])
    return {a: np.array(v, float) for a, v in out.items()}


def compare_latency_violin(before_csv, after_csv,
                           labels=("before lidocaine", "with lidocaine"),
                           colours=("0.62", "#f39c12"), edges=("0.30", "#b5651d"),
                           before_picks=None, after_picks=None,
                           results_dir="results", ylim=None, annotate="outliers", save=None):
    """Violin plots of latency (all muscles pooled) at each intensity: two violins per
    intensity, before (gray) vs with lidocaine (orange), side by side.
    annotate: "outliers" (Tukey 1.5*IQR, labelled with the muscle), "all", or None."""
    from matplotlib.patches import Patch

    mus0, p0 = _load_picks(before_csv, results_dir, before_picks)
    mus1, p1 = _load_picks(after_csv, results_dir, after_picks)
    before = _latency_by_amp_named(mus0, p0)
    after = _latency_by_amp_named(mus1, p1)
    amps = sorted(set(before) | set(after))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(13, 7))
    dx, w = 0.22, 0.38

    def draw(by_amp, shift, colour, edge):
        data, pos = [], []
        for i, a in enumerate(amps):
            v, names = by_amp.get(a, (np.array([]), []))
            if len(v) >= 2:
                data.append(v); pos.append(i + shift)
            if len(v):                                   # individual muscle points
                xs = i + shift + rng.uniform(-0.055, 0.055, len(v))
                ax.scatter(xs, v, s=11, color=colour, edgecolor=edge,
                           linewidth=0.5, zorder=3, alpha=0.9)
                if annotate == "all":
                    _annotate(ax, xs, v, names, np.ones(len(v), bool), edge)
                elif annotate == "outliers":
                    _annotate(ax, xs, v, names, _outlier_mask(v), edge)
        if data:
            vp = ax.violinplot(data, positions=pos, widths=w,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(colour); body.set_edgecolor(edge)
                body.set_linewidth(1.8); body.set_alpha(0.55)
            if "cmedians" in vp:
                vp["cmedians"].set_color(edge); vp["cmedians"].set_linewidth(2.0)

    draw(before, -dx, colours[0], edges[0])
    draw(after, +dx, colours[1], edges[1])

    ax.set_xticks(range(len(amps))); ax.set_xticklabels(amps)
    ax.set_xlabel("Stim amplitude (mA)", fontsize=20)
    ax.set_ylabel("Latency (ms)  —  all muscles", fontsize=18)
    ax.tick_params(labelsize=18)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=colours[0], alpha=0.6, label=labels[0]),
                       Patch(facecolor=colours[1], alpha=0.6, label=labels[1])],
              fontsize=15, frameon=True)
    if annotate:
        ax.text(0.01, 0.99, "labelled dots = outliers (Tukey 1.5 IQR): BB biceps, TB triceps long, Del deltoid, "
                "Trap trapezius, The thenar, ED ext. dig., FD flex. dig., FCR flex. carpi rad.  L/R = side",
                transform=ax.transAxes, fontsize=9, color="0.35", va="top")
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
