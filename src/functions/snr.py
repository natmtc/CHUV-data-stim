"""Signal-to-noise ratio per muscle: RMS of the evoked-response window divided by
RMS of the pre-stimulus baseline (noise). Independent of the manual latency picks."""
import os
import numpy as np
import matplotlib.pyplot as plt

from .labels import pretty
from .io import result_path
from .plots import _base_side


def snr(meta, t, sig, muscles, resp=(5.0, 45.0), pre=(-95.0, -5.0)):
    """res[muscle] = list (one per window) of
    dict(amp, noise_rms, signal_rms, snr, snr_db).

    signal = RMS of (response window - baseline mean); noise = SD of the pre-stim window.
    """
    pre_mask = (t >= pre[0]) & (t <= pre[1])
    resp_mask = (t >= resp[0]) & (t <= resp[1])
    res = {}
    for m in muscles:
        rows = []
        for w, mw in enumerate(meta):
            y = sig[m][w]
            base = y[pre_mask].mean()
            noise = y[pre_mask].std()
            signal = np.sqrt(np.mean((y[resp_mask] - base) ** 2))
            s = signal / noise if noise > 0 else np.nan
            db = 20 * np.log10(s) if (s == s and s > 0) else np.nan
            rows.append(dict(amp=mw["amp_ma"], noise_rms=float(noise),
                             signal_rms=float(signal), snr=float(s), snr_db=float(db)))
        res[m] = rows
    return res


def plot_snr(res, muscles, save=None):
    """SNR vs intensity, all muscles on ONE axes (colour = muscle, marker = side)."""
    from matplotlib.lines import Line2D
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
        a = [r["amp"] for r in res[m]]
        y = [r["snr"] for r in res[m]]
        ax.plot(a, y, marker=marker.get(side, "o"), color=colour[b], lw=1.7, ms=12, mew=0)
    ax.axhline(1, color="0.5", ls="--", lw=1)          # SNR = 1 reference
    ax.set_xlabel("Stim amplitude (mA)", fontsize=18)
    ax.set_ylabel("SNR (signal RMS / noise RMS)", fontsize=18)
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
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def save_snr_csv(res, muscles, csv_path, meta=None, out_dir="results"):
    """Write SNR to results/SNR_<source-filename>.csv (unique per recording). Returns path."""
    import pandas as pd
    src = os.path.splitext(os.path.basename(csv_path))[0]
    rows = []
    for m in muscles:
        for i, r in enumerate(res[m]):
            row = dict(source_file=src, muscle=pretty(m), channel=m, amp_ma=r["amp"],
                       noise_rms=r["noise_rms"], signal_rms=r["signal_rms"],
                       snr=r["snr"], snr_db=r["snr_db"])
            if meta is not None:
                row["electrode"] = meta[i]["electrode"]
                row["mode"] = meta[i]["mode"]
            rows.append(row)
    out_csv = result_path(csv_path, "SNR", out_dir)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
