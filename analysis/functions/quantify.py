"""Peak-to-peak quantification from the manually-picked latencies.

For each muscle x intensity: starting at the picked latency (where the response
starts), look in the window [latency, latency + window_ms], take the max and the
min of the EMG, and report peak-to-peak = max - min (absolute magnitude).
"""
import numpy as np
import matplotlib.pyplot as plt

from .labels import pretty
from .io import detect_stim
from .plots import _base_side


def peak_to_peak(meta, t, sig, muscles, picks, window_ms=5.0, offset_ms=0.0):
    """Return res[muscle] = list (one per window) of
    dict(amp, lat, t_max, y_max, t_min, y_min, p2p). NaN where no latency was picked.

    The search window is [latency + offset_ms, latency + offset_ms + window_ms]."""
    res = {}
    for m in muscles:
        rows = []
        for w, r in enumerate(picks[m]):
            lat, amp = r["t_peak"], r["amp"]
            blank = dict(amp=amp, lat=np.nan, t_max=np.nan, y_max=np.nan,
                         t_min=np.nan, y_min=np.nan, p2p=np.nan)
            if lat != lat:                       # no latency picked -> NaN
                rows.append(blank); continue
            start = lat + offset_ms
            mask = (t >= start) & (t <= start + window_ms)
            if not mask.any():
                rows.append({**blank, "lat": float(lat)}); continue
            seg, tt = sig[m][w][mask], t[mask]
            imax, imin = int(np.argmax(seg)), int(np.argmin(seg))
            ymax, ymin = float(seg[imax]), float(seg[imin])
            rows.append(dict(amp=amp, lat=float(lat),
                             t_max=float(tt[imax]), y_max=ymax,
                             t_min=float(tt[imin]), y_min=ymin,
                             p2p=abs(ymax - ymin)))
        res[m] = rows
    return res


def plot_p2p_markers(meta, t, sig, muscles, picks, window_ms=5.0, offset_ms=0.0,
                     xlim=(-20, 80), gain_frac=0.9, save=None):
    """Per-muscle waterfall with the max (red ^) and min (blue v) used for peak-to-peak."""
    from matplotlib.lines import Line2D
    p2p = peak_to_peak(meta, t, sig, muscles, picks, window_ms, offset_ms)
    amps = np.array([m["amp_ma"] for m in meta])
    step = np.median(np.diff(np.unique(amps))) if len(np.unique(amps)) > 1 else 10
    tmask = (t >= xlim[0]) & (t <= xlim[1])
    t0, t1 = detect_stim(t, sig["Trigger A"])
    respmask = tmask & (t > t1)

    n = len(muscles); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    for i, m in enumerate(muscles):
        ax = axes[i]
        peak = np.percentile(np.abs(sig[m][:, respmask]), 99.5)
        gain = (gain_frac * step) / peak if peak > 0 else 1.0
        for w in range(len(amps)):
            ax.plot(t[tmask], sig[m][w, tmask] * gain + amps[w], color="#1f3b73", lw=0.8)
            r = p2p[m][w]
            if r["y_max"] == r["y_max"]:
                ax.plot(r["t_max"], r["y_max"] * gain + amps[w], "^", color="#d62728", ms=6, zorder=5)
                ax.plot(r["t_min"], r["y_min"] * gain + amps[w], "v", color="#1f77b4", ms=6, zorder=5)
        ax.axvspan(t0, t1, color="red", alpha=0.08)
        ax.axvline(0, color="red", lw=1.0, alpha=0.7)
        ax.set_yticks(np.unique(amps)); ax.set_ylim(amps.min() - step, amps.max() + step)
        ax.set_title(pretty(m), fontweight="bold"); ax.grid(True, axis="x", alpha=0.25)
        if i % ncol == 0: ax.set_ylabel("Stim amplitude (mA)")
        if i >= n - ncol: ax.set_xlabel("Time (ms)")
    for k in range(n, len(axes)): axes[k].axis("off")
    handles = [Line2D([], [], marker="^", color="#d62728", lw=0, ms=10, label="max"),
               Line2D([], [], marker="v", color="#1f77b4", lw=0, ms=10, label="min")]
    fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=True)
    fig.tight_layout()
    if save:
        import os; os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def plot_p2p_combined(meta, t, sig, muscles, picks, window_ms=5.0, offset_ms=0.0, save=None):
    """All muscles on ONE axes: peak-to-peak vs intensity (colour = muscle, marker = side)."""
    from matplotlib.lines import Line2D
    p2p = peak_to_peak(meta, t, sig, muscles, picks, window_ms, offset_ms)

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
        a = [r["amp"] for r in p2p[m]]
        y = [r["p2p"] for r in p2p[m]]
        ax.plot(a, y, marker=marker.get(side, "o"), color=colour[b], lw=1.7, ms=12, mew=0)
    ax.set_xlabel("Stim amplitude (mA)", fontsize=18)
    ax.set_ylabel("Peak-to-peak (mV)", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

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
