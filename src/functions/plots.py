"""Figure functions: waterfall (per-muscle recruitment) and latency-vs-intensity."""
import numpy as np
import matplotlib.pyplot as plt

from .labels import pretty
from .io import detect_stim, detect_pulses


def waterfall(meta, t, sig, muscles, xlim=(-20, 80), gain_frac=0.9, gains=None, title=None,
              highlight=None, highlight_label="analysed", save=None):
    """Per-muscle waterfall: one trace per intensity, stacked at its amplitude (mA).

    highlight : the intensity the ANALYSIS uses - one mA for every muscle, or a {muscle: mA}
                dict (e.g. each muscle's motor threshold). That trace is drawn in orange and
                labelled, so it is obvious which sweep the numbers come from.

    Returns the {muscle: gain} actually used. Pass that dict back in as `gains=` for a
    second recording so BOTH figures are drawn at the same scale - otherwise each file
    is auto-scaled to itself and the two are not comparable (e.g. before vs lidocaine).
    """
    def _hl(m):                      # the highlighted mA of one muscle, if any
        if not isinstance(highlight, dict):
            return highlight
        for k in (m, pretty(m)):
            if k in highlight:
                return highlight[k]
        return highlight.get("default")
    amps = np.array([m["amp_ma"] for m in meta])
    step = np.median(np.diff(np.unique(amps))) if len(np.unique(amps)) > 1 else 10
    tmask = (t >= xlim[0]) & (t <= xlim[1])
    pulses = detect_pulses(t, sig["Trigger A"])
    t0, t1 = pulses[0] if pulses else (0.0, 0.0)
    # exclude only the FIRST pulse's artifact when scaling (trains keep their response)
    respmask = tmask & (t > t1)
    if not respmask.any():
        respmask = tmask

    n = len(muscles)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    panel_h = max(3.2, 0.17 * len(np.unique(amps)) + 1.4)      # room for every intensity row
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, panel_h * nrow),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    used = {}
    for i, m in enumerate(muscles):
        ax = axes[i]
        data = sig[m][:, tmask]
        # one gain per muscle: scale so the largest evoked deflection ~ gain_frac*step
        if gains and m in gains:
            gain = gains[m]
        else:
            peak = np.percentile(np.abs(sig[m][:, respmask]), 99.5)
            gain = (gain_frac * step) / peak if peak > 0 else 1.0
        used[m] = gain
        hl = _hl(m)
        for w in range(len(amps)):
            on = hl is not None and amps[w] == hl
            ax.plot(t[tmask], data[w] * gain + amps[w], color="#e6550d" if on else "#1f3b73",
                    lw=2.0 if on else 0.9, alpha=1.0 if on else 0.7, zorder=5 if on else 2)
        if hl is not None and hl in amps:
            ax.annotate(f"{highlight_label} \u2014 {hl} mA", (xlim[1], hl), xytext=(-4, 7),
                        textcoords="offset points", ha="right", va="bottom", fontsize=9,
                        color="#e6550d", fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        for a0, a1 in pulses:                       # one band per pulse of the train
            if a1 >= xlim[0] and a0 <= xlim[1]:
                ax.axvspan(a0, a1, color="red", alpha=0.10, zorder=0)
                ax.axvline(a0, color="red", lw=1.2, alpha=0.85, zorder=1)
        _u = np.unique(amps)
        _tk = _u if len(_u) <= 14 else _u[::int(np.ceil(len(_u) / 14))]
        ax.set_yticks(_tk)
        ax.set_ylim(amps.min() - step, amps.max() + step)
        ax.set_xlim(*xlim)   # keep XLIM authoritative (the artifact band must not widen it)
        ax.set_title(pretty(m), fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25)
        if i % ncol == 0:
            ax.set_ylabel("Stim amplitude (mA)")
        if i >= n - ncol:
            ax.set_xlabel("Time (ms)")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    if title:                       # reserve the strip the title needs, then lay out into it
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 / (panel_h * nrow)) if title else 1))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print("saved", save)
    plt.show()
    return used


def plot_latency(meta, t, sig, muscles, picks, resp_end=50.0, save=None):
    t0, t1 = detect_stim(t, sig["Trigger A"])
    n = len(muscles); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for i, m in enumerate(muscles):
        ax = axes[i]
        a = [r["amp"] for r in picks[m]]
        lat = [r["t_peak"] for r in picks[m]]
        ax.plot(a, lat, "-o", color="#08519c", ms=6, lw=1.5)
        ax.axhspan(0, t1, color="red", alpha=0.10)   # artifact zone (no valid latency)
        ax.set_ylim(0, resp_end); ax.set_title(pretty(m), fontweight="bold")
        ax.grid(True, alpha=0.25)
        if i % ncol == 0: ax.set_ylabel("Latency (ms)")
        if i >= n - ncol: ax.set_xlabel("Stim amplitude (mA)")
    for k in range(n, len(axes)): axes[k].axis("off")
    fig.tight_layout()
    if save:
        import os; os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def _base_side(m):
    """('Biceps', 'L') from the pretty label 'Biceps (L)'."""
    p = pretty(m)
    if p.endswith("(L)"):
        return p[:-4].strip(), "L"
    if p.endswith("(R)"):
        return p[:-4].strip(), "R"
    return p, ""


def plot_latency_combined(meta, t, sig, muscles, picks, resp_end=50.0, save=None):
    """All muscles on ONE axes: colour = muscle, marker = side (L/R), with a legend."""
    from matplotlib.lines import Line2D
    t0, t1 = detect_stim(t, sig["Trigger A"])

    bases = []
    for m in muscles:
        b, _ = _base_side(m)
        if b not in bases:
            bases.append(b)
    cmap = plt.cm.tab10
    colour = {b: cmap(i % 10) for i, b in enumerate(bases)}
    marker = {"L": "o", "R": "^", "": "D"}

    fig, ax = plt.subplots(figsize=(11, 7))
    for m in muscles:
        b, side = _base_side(m)
        a = [r["amp"] for r in picks[m]]
        lat = [r["t_peak"] for r in picks[m]]
        ax.plot(a, lat, marker=marker.get(side, "o"), color=colour[b],
                lw=1.7, ms=12, mew=0)
    ax.axhspan(0, t1, color="red", alpha=0.08)   # artifact zone (no valid latency)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Stim amplitude (mA)", fontsize=18)
    ax.set_ylabel("Latency (ms)", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):          # keep only bottom & left spines
        ax.spines[s].set_visible(False)

    # one legend: muscle names shown in their own colour (no line), plus L/R markers
    muscle_handles = [Line2D([], [], linestyle="", marker="", label=b) for b in bases]
    side_handles = [Line2D([], [], linestyle="", marker=marker[s], color="0.25",
                           ms=13, mew=0, label=lab) for s, lab in [("L", "Left"), ("R", "Right")]]
    leg = ax.legend(handles=muscle_handles + side_handles, loc="upper left", ncol=2,
                    frameon=True, framealpha=0.9, fontsize=13,
                    handlelength=1.2, handletextpad=0.6, columnspacing=1.4)
    for txt, b in zip(leg.get_texts()[:len(bases)], bases):
        txt.set_color(colour[b]); txt.set_fontweight("bold")
    fig.tight_layout()
    if save:
        import os; os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def waterfall_overlay(meta_b, t_b=None, sig_b=None, meta_a=None, t_a=None, sig_a=None, muscles=None,
                      xlim=(-20, 80), gains=None, gain_frac=0.9,
                      labels=("before lidocaine", "with lidocaine"), colours=None, linestyles=None,
                      title=None, save=None):
    """Per-muscle waterfall with several recordings on the same panel, each trace stacked
    at its own intensity (mA). Gray = first condition, orange = second, blue = third...

    Two ways to call it:
      waterfall_overlay(meta_b, t_b, sig_b, meta_a, t_a, sig_a, muscles, ...)   # two files
      waterfall_overlay([(meta, t, sig), (meta, t, sig), ...], muscles=..., labels=...)

    Pass the `gains` returned by `waterfall(...)` on the BEFORE file so the three figures
    (before, after, overlay) share one scale per muscle; otherwise a shared gain is
    computed here from both files together."""
    from matplotlib.lines import Line2D
    from .io import detect_pulses
    from .burst import PALETTE

    if isinstance(meta_b, (list, tuple)) and meta_b and isinstance(meta_b[0], (list, tuple)):
        recs = list(meta_b)                       # [(meta, t, sig), ...]
        if muscles is None:
            muscles = [c for c in recs[0][2] if c != "Trigger A" and all(c in r[2] for r in recs)]
    else:
        recs = [(meta_b, t_b, sig_b), (meta_a, t_a, sig_a)]
    colours = list(colours) if colours else PALETTE[:len(recs)]
    linestyles = list(linestyles) if linestyles else ["-"] * len(recs)
    labels = list(labels)[:len(recs)]
    runs = [(m_, t_, s_, c_, l_) for (m_, t_, s_), c_, l_ in zip(recs, colours, linestyles)]
    meta_b, t_b, sig_b = recs[0]
    amps_all = np.array(sorted({m["amp_ma"] for mm, _, _ in recs for m in mm}))
    step = np.median(np.diff(amps_all)) if len(amps_all) > 1 else 10
    used = dict(gains or {})
    if not gains:                       # one gain per muscle from both files together
        for m in muscles:
            peaks = []
            for meta, t, sig, _, _ in runs:
                tmask = (t >= xlim[0]) & (t <= xlim[1])
                _, t1 = detect_pulses(t, sig["Trigger A"])[0]
                rm = tmask & (t > t1)
                peaks.append(np.percentile(np.abs(sig[m][:, rm if rm.any() else tmask]), 99.5))
            pk = max(peaks)
            used[m] = (gain_frac * step) / pk if pk > 0 else 1.0

    n = len(muscles); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    panel_h = max(3.2, 0.17 * len(amps_all) + 1.4)             # room for every intensity row
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, panel_h * nrow), sharex=True,
                             squeeze=False)
    axes = axes.ravel()
    for i, m in enumerate(muscles):
        ax = axes[i]
        for meta, t, sig, col, ls_ in runs:
            amps = np.array([x["amp_ma"] for x in meta])
            tmask = (t >= xlim[0]) & (t <= xlim[1])
            for w in range(len(amps)):
                ax.plot(t[tmask], sig[m][w, tmask] * used[m] + amps[w], color=col, lw=0.9, ls=ls_,
                        alpha=(0.8 if col == colours[0] else 0.95))
        for a0, a1 in detect_pulses(t_b, sig_b["Trigger A"]):
            if a1 >= xlim[0] and a0 <= xlim[1]:
                ax.axvspan(a0, a1, color="red", alpha=0.10, zorder=0)
                ax.axvline(a0, color="red", lw=1.2, alpha=0.85, zorder=1)
        _tk = amps_all if len(amps_all) <= 14 else amps_all[::int(np.ceil(len(amps_all) / 14))]
        ax.set_yticks(_tk)
        ax.set_ylim(amps_all.min() - step, amps_all.max() + step)
        ax.set_xlim(*xlim)
        ax.set_title(pretty(m), fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25)
        if i % ncol == 0:
            ax.set_ylabel("Stim amplitude (mA)")
        if i >= n - ncol:
            ax.set_xlabel("Time (ms)")
    for k in range(n, len(axes)):
        axes[k].axis("off")
        if k - ncol >= 0:
            axes[k - ncol].tick_params(labelbottom=True); axes[k - ncol].set_xlabel("Time (ms)")
    fig.legend(handles=[Line2D([], [], color=c, lw=2.5, ls=ls_, label=l) for l, c, ls_ in zip(labels, colours, linestyles)],
               loc="upper center", ncol=len(labels), fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, (0.97 if title else 1.0)))
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, (0.94 if title else 0.97)))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
    return used
