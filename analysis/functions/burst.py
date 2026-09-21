"""Per-pulse response along a Burst / Modulated train — FULLY AUTOMATIC.

A burst file is one train per stimulation window (30 Hz x 1000 ms = 30 pulses).
Every pulse evokes its own response, so there is nothing to pick by hand: the
response to pulse k is simply whatever happens between the end of pulse k's own
artifact and the onset of pulse k+1.

    window_k = [pulse_k_start + resp_start_ms, pulse_(k+1)_start - guard_ms]
    p2p_k    = max(EMG) - min(EMG) inside that window

`resp_start_ms` matters: the stimulus artifact outlasts the trigger pulse by a lot and
by a different amount on every channel (2 ms on Deltoid, >11 ms on Thenar in the P04
30 Hz file). `artifact_extent` measures it per channel; `plot_burst_windows` prints the
table so you can set `resp_start_ms` above it and not measure artifact decay as signal.

`plot_burst_windows` draws those windows with the max/min actually used, so you can
check the detection before trusting the numbers.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from .labels import pretty
from .io import detect_pulses

BEFORE, AFTER = "0.45", "#f39c12"      # same gray/orange as functions/compare.py
PALETTE = [BEFORE, AFTER, "#1f77b4", "#2ca02c", "#9467bd"]   # condition 1, 2, 3, ...


def _conditions(csv_before, csv_after, labels, colours=None):
    """Accept either (csv_before, csv_after) or csv_before = [csv, csv, ...] with csv_after=None.
    Returns (paths, labels, colours) lists of equal length."""
    paths = list(csv_before) if isinstance(csv_before, (list, tuple)) else [csv_before, csv_after]
    paths = [p for p in paths if p is not None]
    labels = list(labels)[:len(paths)]
    if len(labels) < len(paths):
        labels += [f"condition {i + 1}" for i in range(len(labels), len(paths))]
    colours = list(colours) if colours else PALETTE[:len(paths)]
    return paths, labels, colours


def artifact_extent(t, sig, muscles, pulses, w=-1, frac=0.2, look_ms=14.0):
    """Width of the stimulus artifact SPIKE itself, per channel, in ms after pulse onset.

    Returns {channel: ms}. Measured as the CONTIGUOUS excursion starting at the pulse:
    walk forward from the onset until the signal first drops back under `frac` x the
    spike peak. Later deflections (the evoked response) are deliberately not counted -
    an earlier version took the last excursion and wrongly flagged real responses.
    """
    out = {}
    for m in muscles:
        y = sig[m][w]
        med = np.median(y)
        ext = []
        for p0, _ in pulses:
            k = np.where((t >= p0) & (t <= p0 + look_ms))[0]
            if not len(k):
                continue
            seg = np.abs(y[k] - med)
            thr = frac * seg.max()
            if seg[0] <= thr:                       # already back at baseline
                ext.append(0.0); continue
            below = np.where(seg <= thr)[0]
            end = k[below[0]] if len(below) else k[-1]
            ext.append(float(t[end] - p0))
        out[m] = float(np.mean(ext)) if ext else np.nan
    return out


def clipped_channels(sig, muscles, pct=0.5):
    """Channels whose signal sits at its own rail for more than `pct` % of samples
    (i.e. the amplifier saturated and peak-to-peak is meaningless there)."""
    bad = {}
    for m in muscles:
        y = np.abs(sig[m]); lim = y.max()
        frac = float((y > 0.999 * lim).mean() * 100)
        if frac > pct:
            bad[m] = (frac, float(lim))
    return bad


def noise_p2p(t, sig, muscles, win_len_ms, pre=(-95.0, -5.0)):
    """Peak-to-peak of the PRE-stimulus baseline, per muscle x intensity, measured over
    windows of the same length as the response window (so it is directly comparable).
    Returns {muscle: array [n_intensities]} = mean p2p over the non-overlapping windows."""
    out = {}
    starts = np.arange(pre[0], pre[1] - win_len_ms + 1e-9, win_len_ms)
    for m in muscles:
        vals = []
        for a in starts:
            k = (t >= a) & (t <= a + win_len_ms)
            seg = sig[m][:, k]
            vals.append(seg.max(axis=1) - seg.min(axis=1))
        out[m] = np.mean(vals, axis=0) if vals else np.full(sig[m].shape[0], np.nan)
    return out


def _has_dropout(seg, min_run=3):
    """True if the segment contains >= min_run consecutive EXACT zeros - the wireless
    EMG sensors write 0.0 when a packet is lost, never a real 0.0 otherwise."""
    z = (seg == 0.0).astype(int)
    if z.sum() < min_run:
        return False
    d = np.diff(np.r_[0, z, 0])
    return bool((np.where(d == -1)[0] - np.where(d == 1)[0]).max() >= min_run)


def burst_p2p(meta, t, sig, muscles, n_pulses=10, resp_start_ms=8.0, resp_end_ms=None,
              guard_ms=1.0, min_snr=None, max_edge_frac=0.5, edge_ms=1.0):
    """Automatic per-pulse peak-to-peak. No manual latencies involved.

    n_pulses      : how many pulses of the train to analyse (first N)
    resp_start_ms : start of the response window, measured from each pulse's ONSET.
                    Must clear that channel's artifact - see `artifact_extent`.
                    A number for all muscles, or {channel: number} to override some.
    resp_end_ms   : end of the window, also from pulse onset; None = run to the next
                    pulse minus `guard_ms`. Same number-or-dict rule.
    guard_ms      : stop this much before the next pulse starts
    min_snr       : keep only trains with a MOTOR RESPONSE - pulse 1 p2p must exceed
                    min_snr x the baseline p2p (same window length, pre-stimulus).
                    Trains that fail are set to NaN everywhere so every plot/average
                    downstream only sees real responses. None = keep everything.
    max_edge_frac : ARTIFACT rejection. If more than this fraction of a train's pulses
                    have their max or min sitting within `edge_ms` of a window border,
                    the trace between pulses is a monotone artifact-recovery curve, not
                    an EMG response (Deltoid / Biceps / Triceps here) -> the train is
                    rejected like a non-responding one. None = off.

    Returns dict with
      p2p[muscle]   array [n_intensities, n_pulses]  peak-to-peak
      tmax/tmin[m]  array [n_intensities, n_pulses]  time of the max / min used
      ymax/ymin[m]  same, the values
      win           list of (start_ms, stop_ms) per pulse
    """
    pulses = detect_pulses(t, sig["Trigger A"])
    if not pulses:
        raise ValueError("no stimulus pulses found in Trigger A")
    ipi = float(np.median(np.diff([p[0] for p in pulses]))) if len(pulses) > 1 else np.inf
    use = pulses[:n_pulses]

    def _per(m, v, default):
        if isinstance(v, dict):
            return v.get(m, default)
        return v

    def _windows(rs, re_):
        w_ = []
        for k, (p0, p1) in enumerate(use):
            nxt = pulses[k + 1][0] if k + 1 < len(pulses) else p0 + ipi
            start = p0 + rs
            stop = (p0 + re_) if re_ else (nxt - guard_ms)
            w_.append((float(start), float(min(stop, nxt - guard_ms))))
        if w_[0][1] <= w_[0][0]:
            raise ValueError(f"empty response window: resp_start_ms={rs} leaves nothing "
                             f"before the next pulse (IPI {ipi:.1f} ms)")
        return w_

    rs0 = resp_start_ms if not isinstance(resp_start_ms, dict) else 8.0
    re0 = resp_end_ms if not isinstance(resp_end_ms, dict) else None
    win = _windows(rs0, re0)                                   # default window
    wins = {m: _windows(_per(m, resp_start_ms, rs0), _per(m, resp_end_ms, re0))
            for m in muscles}                                  # per-muscle windows

    nW, nP = len(meta), len(use)
    out = {k: {m: np.full((nW, nP), np.nan) for m in muscles}
           for k in ("p2p", "tmax", "tmin", "ymax", "ymin")}
    dropouts = {m: np.zeros((nW, nP), bool) for m in muscles}
    for m in muscles:
        for w in range(nW):
            y = sig[m][w]
            for k, (a, b) in enumerate(wins[m]):
                mask = (t >= a) & (t <= b)
                if not mask.any():
                    continue
                seg, tt = y[mask], t[mask]
                if _has_dropout(seg):            # sensor packet loss -> exact zeros; skip pulse
                    dropouts[m][w, k] = True
                    continue
                i_hi, i_lo = int(np.argmax(seg)), int(np.argmin(seg))
                out["p2p"][m][w, k] = float(seg[i_hi] - seg[i_lo])
                out["tmax"][m][w, k], out["ymax"][m][w, k] = float(tt[i_hi]), float(seg[i_hi])
                out["tmin"][m][w, k], out["ymin"][m][w, k] = float(tt[i_lo]), float(seg[i_lo])
    # response criterion: pulse 1 vs pre-stimulus noise, same window length
    win_len = win[0][1] - win[0][0]
    noise = noise_p2p(t, sig, muscles, win_len)
    snr = {m: out["p2p"][m][:, 0] / noise[m] for m in muscles}
    responding = {m: (snr[m] >= min_snr) if min_snr else np.ones(nW, bool) for m in muscles}
    reason = {m: np.where(responding[m], "", "below criterion") for m in muscles}
    # artifact rejection: fraction of pulses whose max/min sit on a window border
    edge_frac = {}
    for m in muscles:
        w_ = np.array(wins[m])                                    # [nP, 2]
        d = np.minimum.reduce([out["tmax"][m] - w_[None, :, 0], w_[None, :, 1] - out["tmax"][m],
                               out["tmin"][m] - w_[None, :, 0], w_[None, :, 1] - out["tmin"][m]])
        with np.errstate(invalid="ignore"):
            edge_frac[m] = np.nanmean(d <= edge_ms, axis=1)
        if max_edge_frac is not None:
            art = responding[m] & (edge_frac[m] > max_edge_frac)
            responding[m] = responding[m] & ~art
            reason[m] = np.where(art, "artifact", reason[m])
    if min_snr or max_edge_frac is not None:
        for m in muscles:
            for key in ("p2p", "tmax", "tmin", "ymax", "ymin"):
                out[key][m][~responding[m], :] = np.nan
    out.update(win=win, wins=wins, ipi_ms=ipi, pulse_ms=np.array([p[0] for p in use]),
               freq_hz=(1000.0 / ipi if np.isfinite(ipi) else np.nan),
               noise_p2p=noise, snr_pulse1=snr, responding=responding, reason=reason,
               edge_frac=edge_frac, min_snr=min_snr, max_edge_frac=max_edge_frac,
               dropouts=dropouts,
               amps=np.array([m["amp_ma"] for m in meta]))
    return out


def motor_threshold(res, muscles=None):
    """Data-derived motor threshold per muscle: the LOWEST intensity (mA) whose train
    passes the response criterion (`min_snr`). None where no train passes.
    Note: this is per muscle; the recording-level MT the experimenters wrote in the
    log is one number (the first muscle to respond, judged live)."""
    muscles = muscles or list(res["responding"])
    return {m: (int(res["amps"][res["responding"][m]].min())
                if res["responding"][m].any() else None) for m in muscles}


def responding_table(res, muscles):
    """One line per muscle: how many trains pass the response criterion and from
    which intensity onward."""
    lines = []
    for m in muscles:
        r, a = res["responding"][m], res["amps"]
        n_art = int((res["reason"][m] == "artifact").sum())
        tail = f"  [{n_art} rejected as ARTIFACT]" if n_art else ""
        if r.all():
            lines.append(f"{pretty(m):22s} {r.sum()}/{len(r)}  (all)")
        elif r.any():
            lines.append(f"{pretty(m):22s} {r.sum()}/{len(r)}  from {a[r].min()} mA "
                         f"(SNR at max: {res['snr_pulse1'][m][-1]:.1f}){tail}")
        else:
            lines.append(f"{pretty(m):22s} 0/{len(r)}  NO RESPONSE "
                         f"(SNR at max: {res['snr_pulse1'][m][-1]:.1f}){tail}")
    return "\n".join(lines)


def normalize_burst(res, mode="max"):
    """Per train (muscle x intensity): "max" -> % of the biggest of the N pulses,
    "first" -> % of pulse 1, "none" -> raw."""
    import warnings
    if mode == "none":
        return res["p2p"]
    out = {}
    for m, a in res["p2p"].items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmax(a, axis=1) if mode == "max" else a[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m] = 100.0 * a / ref[:, None]
    return out


def _grid(n, ncol=4, w=5.0, h=3.2, **kw):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False, **kw)
    return fig, axes.ravel(), ncol


def _finish(fig, axes, n, ncol, save):
    for k in range(n, len(axes)):
        axes[k].axis("off")
        if k - ncol >= 0:
            axes[k - ncol].tick_params(labelbottom=True)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()


def diagnostics(meta, t, sig, muscles, n_pulses=10, resp_start_ms=8.0, resp_end_ms=None,
                guard_ms=1.0, min_snr=None, max_edge_frac=0.5, amp=None, _res=None):
    """Print-only health check of one recording: window used, artifact width per channel
    (does resp_start_ms clear it?), which trains pass the response criterion and from
    which intensity, and any clipped channel. Run it once per file before the figures."""
    res = _res or burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms,
                            guard_ms, min_snr, max_edge_frac)
    amps = np.array([m["amp_ma"] for m in meta])
    w = int(np.argmax(amps)) if amp is None else int(np.where(amps == amp)[0][0])
    pulses = detect_pulses(t, sig["Trigger A"])[:n_pulses]
    print(f"{meta[0]['mode']} | electrode {meta[0]['electrode']} | {res['freq_hz']:.0f} Hz | "
          f"{len(res['win'])} pulses analysed | intensities {list(amps)} mA")
    print(f"artifact table measured at {amps[w]} mA | "
          f"window = onset +{resp_start_ms} ms -> next pulse -{guard_ms} ms "
          f"(pulse 1: {res['win'][0][0]:.1f}-{res['win'][0][1]:.1f} ms)")
    ext = artifact_extent(t, sig, muscles, pulses, w)
    late = {m: v for m, v in ext.items() if v >= resp_start_ms}
    print("  artifact still dominant at (ms after onset): "
          + ", ".join(f"{pretty(m)} {v:.1f}" for m, v in sorted(ext.items(), key=lambda kv: -kv[1])[:5]))
    if late:
        print(f"  !! resp_start_ms={resp_start_ms} does NOT clear the artifact on: "
              + ", ".join(f"{pretty(m)} ({v:.1f} ms)" for m, v in late.items()))
    if min_snr:
        print(f"  response criterion: pulse-1 p2p >= {min_snr} x baseline p2p  ->  trains kept:")
        for line in responding_table(res, muscles).splitlines():
            print("    " + line)
        # (no single recording-level MT is printed on purpose: proximal channels pass on
        #  artifact recovery alone and would give a meaningless 10 mA - read "from X mA")
    n_drop = {m: int(res["dropouts"][m].sum()) for m in muscles if res["dropouts"][m].any()}
    if n_drop:
        print("  !! sensor DROPOUTS (exact-zero stretches) - those pulses are skipped: "
              + ", ".join(f"{pretty(m)} {n}" for m, n in n_drop.items()))
    bad = clipped_channels(sig, muscles)
    for m, (frac, lim) in bad.items():
        print(f"  !! {pretty(m)} is CLIPPED ({frac:.1f} % of samples at +-{lim:.2f} mV) "
              f"- peak-to-peak is not meaningful there")
    return res


def plot_burst_windows(meta, t, sig, muscles, amp=None, n_pulses=10, resp_start_ms=8.0,
                       resp_end_ms=None, guard_ms=1.0, min_snr=None, max_edge_frac=0.5, zoom=True, save=None):
    """CHECK PLOT — the trace for one intensity with, per pulse: the red artifact band,
    the shaded response window, and the max (red v) / min (blue ^) used for p2p."""
    res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
    raw = res if not min_snr else burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms,
                                            resp_end_ms, guard_ms, None, None)
    amps = np.array([m["amp_ma"] for m in meta])
    w = int(np.argmax(amps)) if amp is None else int(np.where(amps == amp)[0][0])
    pulses = detect_pulses(t, sig["Trigger A"])[:n_pulses]
    xlim = (-10, res["win"][-1][1] + 5)

    n = len(muscles)
    fig, axes, ncol = _grid(n, sharex=True)
    for i, m in enumerate(muscles):
        ax = axes[i]
        mask = (t >= xlim[0]) & (t <= xlim[1])
        ax.plot(t[mask], sig[m][w, mask], color="#1f3b73", lw=1.0)
        for (a, b), (p0, p1) in zip(res["win"], pulses):
            ax.axvspan(p0, p1, color="red", alpha=0.12, zorder=0)      # artifact
            ax.axvspan(a, b, color="#2ca25f", alpha=0.10, zorder=0)    # response window
        ax.plot(raw["tmax"][m][w], raw["ymax"][m][w], "v", color="#d62728", ms=7, zorder=5)
        ax.plot(raw["tmin"][m][w], raw["ymin"][m][w], "^", color="#1f77b4", ms=7, zorder=5)
        ax.set_xlim(*xlim)
        if min_snr and not res["responding"][m][w]:
            ax.text(0.5, 0.5, "no response\n(below criterion)", transform=ax.transAxes,
                    ha="center", va="center", color="0.4", fontsize=12, fontweight="bold")
        if zoom:      # scale y to the RESPONSE windows only - the artifact spike is
                      # 10-50x bigger and would flatten everything we want to look at
            inwin = np.zeros_like(t, dtype=bool)
            for a, b in res["win"]:
                inwin |= (t >= a) & (t <= b)
            v = sig[m][w][inwin]
            if len(v):
                lo, hi = float(np.min(v)), float(np.max(v))
                pad = 0.25 * (hi - lo) or 0.01
                ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(pretty(m), fontweight="bold")
        ax.grid(alpha=0.25)
        if i % ncol == 0: ax.set_ylabel("EMG (mV)")
        if i >= n - ncol: ax.set_xlabel("Time (ms)")
    diagnostics(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms,
                min_snr, max_edge_frac, amp=amps[w], _res=res)
    _finish(fig, axes, n, ncol, save)
    return res


def plot_burst_p2p(meta, t, sig, muscles, n_pulses=10, resp_start_ms=8.0, resp_end_ms=None,
                   guard_ms=1.0, min_snr=None, max_edge_frac=0.5, normalize="max", amp=None, save=None):
    """Bar plot per muscle: x = pulse # in the train, y = peak-to-peak.
    amp=None -> mean across all intensities (error bar = SD); amp=<mA> -> one intensity."""
    import warnings
    res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
    vals = normalize_burst(res, normalize)
    amps = np.array([m["amp_ma"] for m in meta])
    sel = None if amp is None else np.where(amps == amp)[0]
    if sel is not None and not len(sel):
        raise ValueError(f"{amp} mA not in this recording: {sorted(set(amps))}")
    x = np.arange(1, len(res["win"]) + 1)
    ylab = {"max": "Peak-to-peak (% of max)", "first": "Peak-to-peak (% of pulse 1)",
            "none": "Peak-to-peak (mV)"}[normalize]

    n = len(muscles)
    fig, axes, ncol = _grid(n, sharex=True, sharey=True)
    for i, m in enumerate(muscles):
        ax = axes[i]
        a = vals[m][sel] if sel is not None else vals[m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
            mean_all = np.nanmean(y)
        ax.bar(x, y, yerr=(sd if sel is None else None), color="#1f3b73", width=0.75,
               error_kw=dict(ecolor="0.35", lw=1.2, capsize=3))
        if mean_all == mean_all:
            ax.axhline(mean_all, color="#e6550d", ls="--", lw=2, zorder=3)
            ratio = y[-1] / y[0] * 100 if y[0] else np.nan
            ntr = int((~np.isnan(a[:, 0])).sum())
            ax.text(0.98, 0.95, f"mean {mean_all:.0f}%\nlast/first {ratio:.0f}%\nn = {ntr}",
                    transform=ax.transAxes, ha="right", va="top", color="#e6550d",
                    fontweight="bold", fontsize=11, linespacing=1.3)
        ax.set_xticks(x); ax.set_axisbelow(True)
        ax.set_title(pretty(m), fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if i % ncol == 0: ax.set_ylabel(ylab)
        if i >= n - ncol: ax.set_xlabel("Pulse # in the train")
    if normalize != "none":
        axes[0].set_ylim(0, 140)          # headroom for the corner text
    _finish(fig, axes, n, ncol, save)
    return res


def compare_burst_p2p(csv_before, csv_after, n_pulses=10, resp_start_ms=8.0,
                      resp_end_ms=None, guard_ms=1.0, min_snr=None, max_edge_frac=0.5, normalize="max", amp=None,
                      labels=("before lidocaine", "with lidocaine"), save=None):
    """Pre vs post lidocaine, per muscle: paired bars per pulse (gray vs orange),
    dashed line = each condition's mean over the N pulses."""
    import warnings
    from .io import load_run
    from matplotlib.patches import Patch

    amp_pair = amp if isinstance(amp, (tuple, list)) else (amp, amp)   # per-file MT allowed
    runs = []
    for path, a_sel in zip((csv_before, csv_after), amp_pair):
        meta, t, sig = load_run(path)
        muscles = [c for c in sig if c != "Trigger A"]
        res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
        runs.append((meta, muscles, res, normalize_burst(res, normalize), a_sel))
    muscles = [m for m in runs[0][1] if m in runs[1][1]]
    x = np.arange(1, min(len(runs[0][2]["win"]), len(runs[1][2]["win"])) + 1)
    ylab = {"max": "Peak-to-peak (% of max)", "first": "Peak-to-peak (% of pulse 1)",
            "none": "Peak-to-peak (mV)"}[normalize]

    n = len(muscles)
    fig, axes, ncol = _grid(n, sharex=True, sharey=True)
    for i, m in enumerate(muscles):
        ax = axes[i]
        for j, ((meta, _, res, vals, a_sel), col) in enumerate(zip(runs, (BEFORE, AFTER))):
            amps = np.array([d["amp_ma"] for d in meta])
            sel = None if a_sel is None else np.where(amps == a_sel)[0]
            if sel is not None and not len(sel):
                raise ValueError(f"{a_sel} mA not in {'before' if j == 0 else 'after'} file: "
                                 f"{sorted(set(amps))}")
            a = (vals[m][sel] if sel is not None else vals[m])[:, :len(x)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
                mean_all = np.nanmean(y)
            ax.bar(x + (j - 0.5) * 0.4, y, width=0.38, color=col,
                   yerr=(sd if sel is None else None),
                   error_kw=dict(ecolor="0.35", lw=1.0, capsize=2))
            if mean_all == mean_all:
                ax.axhline(mean_all, color=col, ls="--", lw=2, zorder=3)
        ax.set_xticks(x); ax.set_axisbelow(True)
        ax.set_title(pretty(m), fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if i % ncol == 0: ax.set_ylabel(ylab)
        if i >= n - ncol: ax.set_xlabel("Pulse # in the train")
    axes[0].legend(handles=[Patch(facecolor=BEFORE, label=labels[0]),
                            Patch(facecolor=AFTER, label=labels[1])],
                   loc="lower left", fontsize=11, frameon=True, framealpha=0.9)
    if normalize != "none":
        axes[0].set_ylim(0, 140)
    _finish(fig, axes, n, ncol, save)
    return runs


def save_burst_csv(res, muscles, csv_path, meta=None, normalize="max", out_dir="results"):
    """results/burstp2p_<source-filename>.csv — one row per muscle x intensity x pulse."""
    import pandas as pd
    src = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"burstp2p_{src}.csv")
    norm = normalize_burst(res, normalize)
    rows = []
    for m in muscles:
        nW, nP = res["p2p"][m].shape
        for w in range(nW):
            for k in range(nP):
                row = dict(source_file=src, muscle=pretty(m), channel=m, pulse=k + 1,
                           pulse_t_ms=res["pulse_ms"][k],
                           win_start_ms=res["win"][k][0], win_stop_ms=res["win"][k][1],
                           p2p=res["p2p"][m][w, k], p2p_norm_pct=norm[m][w, k],
                           norm_mode=normalize, freq_hz=round(res["freq_hz"], 1))
                if meta is not None:
                    row.update(amp_ma=meta[w]["amp_ma"], electrode=meta[w]["electrode"],
                               mode=meta[w]["mode"])
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


# ---------------------------------------------------------------------------
# pulse 1 vs the mean of the following pulses
# ---------------------------------------------------------------------------
def first_vs_rest(res, normalize="max", sel=None):
    """Per muscle: (pulse-1, rest) as mean +- SD across intensities, where `rest`
    is first averaged over pulses 2..N WITHIN each train, then across trains.

    Returns {muscle: dict(first, first_sd, rest, rest_sd, n)}."""
    import warnings
    vals = normalize_burst(res, normalize)
    out = {}
    for m, a in vals.items():
        a = a if sel is None else a[sel]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p1 = a[:, 0]
            rest = np.nanmean(a[:, 1:], axis=1)          # one number per train
            ok = ~np.isnan(p1) & ~np.isnan(rest)
            out[m] = dict(first=float(np.nanmean(p1)), first_sd=float(np.nanstd(p1)),
                          rest=float(np.nanmean(rest)), rest_sd=float(np.nanstd(rest)),
                          n=int(ok.sum()))
    return out


def _bar_first_rest(ax, stats, x0, colour, width=0.38, show_err=True):
    y = [stats["first"], stats["rest"]]
    e = [stats["first_sd"], stats["rest_sd"]] if show_err else None
    ax.bar([x0, x0 + 1], y, width=width, color=colour, yerr=e,
           error_kw=dict(ecolor="0.35", lw=1.2, capsize=3))


def plot_first_vs_rest(meta, t, sig, muscles, n_pulses=10, resp_start_ms=8.0,
                       resp_end_ms=None, guard_ms=1.0, min_snr=None, max_edge_frac=0.5, normalize="max", amp=None, save=None):
    """Two bars per muscle: pulse 1, and the mean of pulses 2..N.
    Error bar = SD across intensities (none when a single `amp` is given)."""
    res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
    amps = np.array([m["amp_ma"] for m in meta])
    sel = None if amp is None else np.where(amps == amp)[0]
    st = first_vs_rest(res, normalize, sel)
    ylab = {"max": "Peak-to-peak (% of max)", "first": "Peak-to-peak (% of pulse 1)",
            "none": "Peak-to-peak (mV)"}[normalize]
    n = len(muscles)
    fig, axes, ncol = _grid(n, sharex=True, sharey=True)
    for i, m in enumerate(muscles):
        ax = axes[i]
        _bar_first_rest(ax, st[m], 0, "#1f3b73", width=0.6, show_err=sel is None)
        ratio = st[m]["rest"] / st[m]["first"] * 100 if st[m]["first"] else np.nan
        ax.text(0.98, 0.95, f"rest/first {ratio:.0f}%\nn = {st[m]['n']}", transform=ax.transAxes,
                ha="right", va="top", color="#e6550d", fontweight="bold", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pulse 1", f"pulses 2-{n_pulses}"])
        ax.set_axisbelow(True); ax.set_title(pretty(m), fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if i % ncol == 0: ax.set_ylabel(ylab)
    if normalize != "none":
        axes[0].set_ylim(0, 140)
    _finish(fig, axes, n, ncol, save)
    return st


def compare_first_vs_rest(csv_before, csv_after, n_pulses=10, resp_start_ms=8.0,
                          resp_end_ms=None, guard_ms=1.0, min_snr=None, max_edge_frac=0.5, normalize="max", amp=None,
                          labels=("before lidocaine", "with lidocaine"), save=None):
    """Same as `plot_first_vs_rest`, before (gray) vs with lidocaine (orange) side by side."""
    from .io import load_run
    from matplotlib.patches import Patch
    amp_pair = amp if isinstance(amp, (tuple, list)) else (amp, amp)   # per-file MT allowed
    stats, common = [], None
    for path, a_sel in zip((csv_before, csv_after), amp_pair):
        meta, t, sig = load_run(path)
        muscles = [c for c in sig if c != "Trigger A"]
        res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
        amps = np.array([m["amp_ma"] for m in meta])
        sel = None if a_sel is None else np.where(amps == a_sel)[0]
        if sel is not None and not len(sel):
            raise ValueError(f"{a_sel} mA not in this file: {sorted(set(amps))}")
        stats.append(first_vs_rest(res, normalize, sel))
        common = muscles if common is None else [m for m in common if m in muscles]
    ylab = {"max": "Peak-to-peak (% of max)", "first": "Peak-to-peak (% of pulse 1)",
            "none": "Peak-to-peak (mV)"}[normalize]
    n = len(common)
    fig, axes, ncol = _grid(n, sharex=True, sharey=True)
    for i, m in enumerate(common):
        ax = axes[i]
        for j, (st, col) in enumerate(zip(stats, (BEFORE, AFTER))):
            _bar_first_rest(ax, st[m], (j - 0.5) * 0.4, col, show_err=amp_pair[j] is None)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pulse 1", f"pulses 2-{n_pulses}"])
        ax.set_axisbelow(True); ax.set_title(pretty(m), fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if i % ncol == 0: ax.set_ylabel(ylab)
    axes[0].legend(handles=[Patch(facecolor=BEFORE, label=labels[0]),
                            Patch(facecolor=AFTER, label=labels[1])],
                   loc="lower left", fontsize=11, frameon=True, framealpha=0.9)
    if normalize != "none":
        axes[0].set_ylim(0, 140)
    _finish(fig, axes, n, ncol, save)
    return stats


def resolve_muscles(available, wanted):
    """Map names given as channel ('R_BB') or label ('Biceps (R)') to channel names."""
    if wanted is None:
        return list(available)
    if isinstance(wanted, str):
        wanted = [wanted]
    out = []
    for name in wanted:
        hit = [m for m in available if m == name or pretty(m) == name]
        if not hit:
            raise ValueError(f"muscle {name!r} not found; choose from "
                             f"{[pretty(m) for m in available]}")
        out.append(hit[0])
    return out


# ---------------------------------------------------------------------------
# ONE intensity (e.g. motor threshold): EMG traces on top, per-pulse bars below
# ---------------------------------------------------------------------------
def compare_at_intensity(csv_before, csv_after, amp, n_pulses=10, resp_start_ms=8.0,
                         resp_end_ms=None, guard_ms=1.0, min_snr=None, max_edge_frac=0.5, normalize="none",
                         labels=("before lidocaine", "with lidocaine"), markers=True,
                         title=None, muscles=None, ncol=3, xlim=None, colours=None, save=None):
    """Per muscle, for ONE intensity: top = the two EMG traces overlaid (gray = before,
    orange = with lidocaine) with the artifact / response windows; bottom = per-pulse
    peak-to-peak of those two traces, side by side.

    amp can be one value (same mA in both files) or a (before, after) pair.
    normalize: "none"         -> mV (bars match the traces above)
               "before_first" -> % of the BEFORE-lidocaine pulse 1 of that muscle at that
                                 intensity, used as the reference for BOTH conditions. So the
                                 gray pulse-1 bar is 100 % by definition and every other bar
                                 (later pulses, and all the orange ones) reads relative to it.
               "max"/"first"  -> % of each train's own max / own pulse 1.
    """
    from .io import load_run
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    paths, labels, colours = _conditions(csv_before, csv_after, labels, colours)
    amp_pair = tuple(amp) if isinstance(amp, (tuple, list)) else (amp,) * len(paths)
    runs = []
    for path, a_sel in zip(paths, amp_pair):
        meta, t, sig = load_run(path)
        chans = [c for c in sig if c != "Trigger A"]
        amps = np.array([d["amp_ma"] for d in meta])
        w = np.where(amps == a_sel)[0]
        if not len(w):
            raise ValueError(f"{a_sel} mA not in {path.split('/')[-1]}: {sorted(set(amps))}")
        res = burst_p2p(meta, t, sig, chans, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
        raw = res if not min_snr else burst_p2p(meta, t, sig, chans, n_pulses, resp_start_ms,
                                                resp_end_ms, guard_ms, None, None)
        runs.append(dict(t=t, sig=sig, muscles=chans, w=int(w[0]), res=res, raw=raw,
                         vals=normalize_burst(res, normalize), amp=a_sel,
                         pulses=detect_pulses(t, sig["Trigger A"])[:n_pulses]))
    common = [m for m in runs[0]["muscles"] if all(m in r["muscles"] for r in runs)]
    muscles = resolve_muscles(common, muscles)
    npul = min(len(r["res"]["win"]) for r in runs)
    nc = len(runs); bw = 0.8 / nc                        # bar width per condition
    offs = [(j - (nc - 1) / 2) * bw for j in range(nc)]
    x = np.arange(1, npul + 1)
    if xlim is None:
        xlim = (-10, max(r["res"]["win"][npul - 1][1] for r in runs) + 5)
    if normalize == "before_first":
        # one reference per muscle: pulse 1 of the FIRST condition's train at this intensity
        r0 = runs[0]
        for r in runs:
            r["vals"] = {}
            for m in muscles:
                ref = r0["raw"]["p2p"][m][r0["w"], 0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    r["vals"][m] = 100.0 * r["res"]["p2p"][m] / ref
    ylab = {"max": "% of max", "first": "% of p1", "none": "p2p (mV)",
            "before_first": f"% of p1 ({labels[0]})" if len(labels[0]) <= 10 else "% of p1 (cond. 1)"}[normalize]

    n = len(muscles); ncol = min(ncol, n); nrow = int(np.ceil(n / ncol))
    single = (n == 1)
    pw = 6.2 if ncol > 1 else 12.0                       # a single muscle gets a wide panel
    fig = plt.figure(figsize=(pw * ncol, (7.5 if single else 6.4) * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.45, wspace=0.28,
                             top=(0.88 if single else 0.965), bottom=(0.07 if single else 0.03))
    bad = [clipped_channels(r["sig"], muscles) for r in runs]
    for i, m in enumerate(muscles):
        inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[i // ncol, i % ncol],
                                        height_ratios=[1.25, 1, 0.85],
                                        hspace=(0.45 if single else 0.16))
        ax_t, ax_b, ax_s = (fig.add_subplot(inner[0]), fig.add_subplot(inner[1]),
                            fig.add_subplot(inner[2]))

        # ---- top: the two traces at this intensity -------------------------
        lo, hi = np.inf, -np.inf
        for r, col in zip(runs, colours):
            t, y = r["t"], r["sig"][m][r["w"]]
            mask = (t >= xlim[0]) & (t <= xlim[1])
            ax_t.plot(t[mask], y[mask], color=col, lw=1.1, alpha=0.95)
            if markers:   # the exact max (v) and min (^) each bar below is made of
                rr, ww = r["raw"], r["w"]
                ax_t.plot(rr["tmax"][m][ww, :npul], rr["ymax"][m][ww, :npul], "v", ms=5,
                          color=col, mec="black", mew=0.6, zorder=6)
                ax_t.plot(rr["tmin"][m][ww, :npul], rr["ymin"][m][ww, :npul], "^", ms=5,
                          color=col, mec="black", mew=0.6, zorder=6)
            inwin = np.zeros_like(t, bool)
            for a, b in r["res"]["wins"][m][:npul]:
                inwin |= (t >= a) & (t <= b)
            if inwin.any():
                lo, hi = min(lo, y[inwin].min()), max(hi, y[inwin].max())
        r0 = runs[0]
        for (a, b), (p0, p1) in zip(r0["res"]["wins"][m][:npul], r0["pulses"]):
            ax_t.axvspan(p0, p1, color="red", alpha=0.12, zorder=0)
            ax_t.axvspan(a, b, color="#2ca25f", alpha=0.08, zorder=0)
        if np.isfinite(lo):
            pad = 0.25 * (hi - lo) or 0.01
            ax_t.set_ylim(lo - pad, hi + pad)
        ax_t.set_xlim(*xlim)
        ax_t.set_title("EMG traces  (v = max, ^ = min used for each pulse)" if single else pretty(m),
                       fontweight="bold", fontsize=(12 if single else None))
        for j, (b, col) in enumerate(zip(bad, colours)):
            if m in b:
                ax_t.text(0.02, 0.92 - 0.14 * j, f"CLIPPED ({labels[j]})",
                          transform=ax_t.transAxes, ha="left", va="top", color="#d62728",
                          fontsize=10, fontweight="bold")
        ax_t.tick_params(labelbottom=single, labelsize=10)
        if single:
            ax_t.set_xlabel("Time (ms)", fontsize=10, labelpad=1)
        ax_t.grid(alpha=0.25)
        if i % ncol == 0:
            ax_t.set_ylabel("EMG (mV)", fontsize=11)

        # ---- bottom: per-pulse p2p of exactly those two traces --------------
        for j, (r, col) in enumerate(zip(runs, colours)):
            y = r["vals"][m][r["w"], :npul]
            ax_b.bar(x + offs[j], y, width=bw * 0.95, color=col)
            if np.isfinite(y).any():
                ax_b.axhline(np.nanmean(y), color=col, ls="--", lw=1.6, zorder=3)
        if normalize == "before_first":
            ax_b.axhline(100, color="0.3", lw=1, ls=":", zorder=2)
        if single:
            ax_b.set_title("peak-to-peak per pulse  (dashed = mean over the train)",
                           fontweight="bold", fontsize=12)
            ax_s.set_title("pulse 1 vs mean of the rest  (error bar = SD over pulses 2-N)",
                           fontweight="bold", fontsize=12)
        ax_b.set_xticks(x); ax_b.tick_params(labelsize=10)
        ax_b.set_axisbelow(True); ax_b.grid(True, axis="y", alpha=0.25)
        for s in ("top", "right"):
            ax_b.spines[s].set_visible(False)
        if i % ncol == 0:
            ax_b.set_ylabel(ylab, fontsize=11)
        ax_b.set_xlabel("Pulse #", fontsize=10, labelpad=1)

        # ---- third row: pulse 1 vs the mean of pulses 2..N, same two trains -----
        for j, (r, col) in enumerate(zip(runs, colours)):
            y = r["vals"][m][r["w"], :npul]
            tail = y[1:][np.isfinite(y[1:])]
            rest, rest_sd = (tail.mean(), tail.std()) if len(tail) else (np.nan, np.nan)
            # pulse 1 is a single value -> no error bar; the mean bar gets +- SD over pulses 2..N
            ax_s.bar(0 + offs[j], y[0], width=bw * 0.95, color=col)
            ax_s.bar(1 + offs[j], rest, width=bw * 0.95, color=col,
                     yerr=(rest_sd if np.isfinite(rest_sd) else None),
                     error_kw=dict(ecolor="0.3", lw=1.2, capsize=3))
        if normalize == "before_first":
            ax_s.axhline(100, color="0.3", lw=1, ls=":", zorder=2)
        ax_s.set_xticks([0, 1]); ax_s.set_xticklabels(["pulse 1", f"mean 2-{npul}"], fontsize=10)
        ax_s.tick_params(labelsize=10); ax_s.set_axisbelow(True)
        ax_s.grid(True, axis="y", alpha=0.25)
        ax_s.margins(y=0.3)
        for sp in ("top", "right"):
            ax_s.spines[sp].set_visible(False)
        if i % ncol == 0:
            ax_s.set_ylabel(ylab, fontsize=11)
        for j, (r, col) in enumerate(zip(runs, colours)):          # say WHY a train is out
            if not r["res"]["responding"][m][r["w"]]:
                why = r["res"]["reason"][m][r["w"]]
                dy = 0.92 - 0.12 * j
                ax_b.text(0.98, dy, ("ARTIFACT - rejected" if why == "artifact" else "below criterion"),
                          transform=ax_b.transAxes, ha="right", va="top", color=col,
                          fontsize=9, fontweight="bold")

    handles = [Patch(facecolor=c, label=f"{l} ({a} mA)") for l, a, c in zip(labels, amp_pair, colours)]
    if single:                      # title + legend in their own band above the panels
        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=12,
                   frameon=False, bbox_to_anchor=(0.5, 0.955))
    else:
        if title:
            fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=15, fontweight="bold")
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=12,
                   frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 0.995))
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
    return runs


# ---------------------------------------------------------------------------
# summary over ALL intensities without averaging: muscle x intensity heatmap
# ---------------------------------------------------------------------------
def train_metric(res, normalize="none"):
    """Per train (muscle x intensity) numbers that describe the response along the train:
    p1 (pulse-1 p2p), rest (mean p2p of pulses 2..N), rest_first (rest / p1, in %),
    last_first (pulse N / pulse 1, in %). Arrays [n_intensities] per muscle."""
    import warnings
    vals = normalize_burst(res, normalize)
    out = {}
    for m, a in vals.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p1, rest, last = a[:, 0], np.nanmean(a[:, 1:], axis=1), a[:, -1]
            out[m] = dict(p1=p1, rest=rest,
                          rest_first=100 * rest / p1, last_first=100 * last / p1)
    return out


def summary_heatmap(csv_before, csv_after, metric="rest_first", n_pulses=10,
                    resp_start_ms=8.0, resp_end_ms=None, guard_ms=1.0, min_snr=None, max_edge_frac=0.5,
                    labels=("before lidocaine", "with lidocaine"), save=None):
    """Two heatmaps side by side (before | with lidocaine): rows = muscles,
    columns = intensities, colour = `metric` of THAT train. Nothing is averaged.

    metric: "rest_first" (mean of pulses 2..N as % of pulse 1 - <100 = depression),
            "last_first" (pulse N as % of pulse 1), "p1" (pulse-1 p2p, mV),
            "rest" (mean p2p of pulses 2..N, mV).
    Trains below the response criterion are hatched grey."""
    from .io import load_run
    from matplotlib.colors import Normalize

    runs = []
    for path in (csv_before, csv_after):
        meta, t, sig = load_run(path)
        muscles = [c for c in sig if c != "Trigger A"]
        res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
        runs.append((res, train_metric(res), clipped_channels(sig, muscles)))
    muscles = [m for m in runs[0][0]["p2p"] if m in runs[1][0]["p2p"]]
    amps_all = sorted(set(runs[0][0]["amps"]) | set(runs[1][0]["amps"]))

    grids, clipped = [], []
    for res, tm, bad in runs:
        g = np.full((len(muscles), len(amps_all)), np.nan)
        clipped.append([m for m in muscles if m in bad])
        for i, m in enumerate(muscles):
            if m in bad:                          # saturated channel -> whole row masked
                continue
            for j, a in enumerate(amps_all):
                k = np.where(res["amps"] == a)[0]
                if len(k):
                    g[i, j] = tm[m][metric][k[0]]
        grids.append(g)

    pct = metric in ("rest_first", "last_first")
    if pct:
        cmap, norm = plt.get_cmap("RdBu_r"), Normalize(0, 200)      # 100 % = white
    else:
        vmax = np.nanpercentile(np.concatenate([g.ravel() for g in grids]), 98)
        cmap, norm = plt.get_cmap("viridis"), Normalize(0, vmax)
    cmap = cmap.copy(); cmap.set_bad("0.85")
    unit = "% of pulse 1" if pct else "mV"
    ttl = {"rest_first": f"mean of pulses 2-{n_pulses}  ({unit})",
           "last_first": f"pulse {n_pulses}  ({unit})",
           "p1": "pulse 1 peak-to-peak (mV)", "rest": f"mean p2p of pulses 2-{n_pulses} (mV)"}[metric]

    fig, axes = plt.subplots(1, 2, figsize=(1.05 * len(amps_all) + 5, 0.42 * len(muscles) + 1.8),
                             sharey=True)
    for ax, g, lab, cl in zip(axes, grids, labels, clipped):
        im = ax.imshow(np.ma.masked_invalid(g), cmap=cmap, norm=norm, aspect="auto")
        for m in cl:
            ax.text(len(amps_all) / 2 - 0.5, muscles.index(m), "CLIPPED - not analysed",
                    ha="center", va="center", color="#d62728", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(amps_all))); ax.set_xticklabels(amps_all)
        ax.set_yticks(range(len(muscles))); ax.set_yticklabels([pretty(m) for m in muscles])
        ax.set_xlabel("Stim amplitude (mA)"); ax.set_title(lab, fontweight="bold")
        for i in range(len(muscles)):
            for j in range(len(amps_all)):
                v = g[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                            color=("black" if pct and abs(v - 100) < 60 else "white"))
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(ttl)
    fig.suptitle("grey = no motor response (below criterion)", fontsize=10, color="0.4", y=0.99)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
    return grids


# ---------------------------------------------------------------------------
# summary as RECRUITMENT CURVES: x = intensity, y = response - nothing averaged
# ---------------------------------------------------------------------------
def summary_curves(csv_before, csv_after, n_pulses=10, resp_start_ms=8.0, resp_end_ms=None,
                   guard_ms=1.0, min_snr=None, max_edge_frac=0.5, labels=("before lidocaine", "with lidocaine"),
                   colours=None, save=None):
    """Per muscle, two rows, x = stimulation intensity (mA):

    top    - RECRUITMENT: pulse-1 peak-to-peak (solid line, filled circles) and the mean
             of pulses 2..N (dashed line, triangles), in mV. Gray = before, orange = with
             lidocaine. The gap between solid and dashed IS the depression along the train.
    bottom - DEPRESSION RATIO: mean of pulses 2..N as % of pulse 1. 100 % = no change,
             below = depresses, above = facilitates. Same colours.

    A point is only drawn where the train passes the motor-response criterion; below
    that it is shown as a small hollow marker on the top row (so you see where the
    threshold is) and left out of the ratio row.
    """
    from .io import load_run
    from matplotlib.lines import Line2D

    paths, labels, colours = _conditions(csv_before, csv_after, labels, colours)
    runs = []
    for path in paths:
        meta, t, sig = load_run(path)
        muscles = [c for c in sig if c != "Trigger A"]
        res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
        raw = res if not min_snr else burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms,
                                                resp_end_ms, guard_ms, None, None)
        runs.append(dict(res=res, tm=train_metric(res), raw_tm=train_metric(raw),
                         amps=res["amps"], bad=clipped_channels(sig, muscles)))
    muscles = [m for m in runs[0]["res"]["p2p"] if all(m in r["res"]["p2p"] for r in runs)]
    all_amps = np.concatenate([r["amps"] for r in runs])
    xlim = (all_amps.min() - 2.5, all_amps.max() + 2.5)      # same x on every panel

    n = len(muscles); ncol = 4; nrow = int(np.ceil(n / ncol))
    # every muscle = a pair of rows; a spacer row between muscle pairs keeps titles clear
    hr = []
    for _ in range(nrow):
        hr += [1.3, 1, 0.28]
    fig, axes_all = plt.subplots(3 * nrow, ncol, figsize=(4.8 * ncol, 5.6 * nrow), squeeze=False,
                                 gridspec_kw=dict(height_ratios=hr, hspace=0.12, wspace=0.3))
    for r_ in range(nrow):
        for c_ in range(ncol):
            axes_all[3 * r_ + 2, c_].axis("off")
    axes = np.array([[axes_all[3 * r_, c_], axes_all[3 * r_ + 1, c_]]
                     for r_ in range(nrow) for c_ in range(ncol)])
    for i, m in enumerate(muscles):
        r_, c_ = divmod(i, ncol)
        ax_t, ax_r = axes[i]
        clipped = any(m in r["bad"] for r in runs)
        if clipped:
            ax_t.set_title(pretty(m), fontweight="bold")
            ax_t.text(0.5, 0.5, "CLIPPED\nnot analysed", transform=ax_t.transAxes, ha="center",
                      va="center", color="#d62728", fontweight="bold", fontsize=12)
            for ax in (ax_t, ax_r):
                ax.set_xlim(*xlim); ax.set_yticks([]); ax.grid(False)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            ax_t.tick_params(labelbottom=False)
            ax_r.set_xlabel("Stim amplitude (mA)")
            continue
        for r, col in zip(runs, colours):
            a = r["amps"]; ok = r["res"]["responding"][m]
            p1, rest = r["raw_tm"][m]["p1"], r["raw_tm"][m]["rest"]
            # top: recruitment. Filled where responding, hollow where not.
            ax_t.plot(a[ok], p1[ok], "-o", color=col, lw=2, ms=7, mew=0)
            ax_t.plot(a[ok], rest[ok], "--^", color=col, lw=1.6, ms=7, mew=0, alpha=0.9)
            ax_t.plot(a[~ok], p1[~ok], "o", mfc="white", mec=col, ms=5, mew=1.2, alpha=0.7)
            # bottom: ratio, responding trains only
            ratio = r["tm"][m]["rest_first"]
            ax_r.plot(a[ok], ratio[ok], "-o", color=col, lw=2, ms=7, mew=0)
        ax_r.axhline(100, color="0.3", lw=1, ls=":")
        ax_t.set_title(pretty(m), fontweight="bold")
        for ax in (ax_t, ax_r):
            ax.set_xlim(*xlim); ax.grid(alpha=0.25); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        ax_t.tick_params(labelbottom=False)
        ax_r.set_ylim(0, 160); ax_r.set_yticks([0, 50, 100, 150])
        ax_r.set_xlabel("Stim amplitude (mA)")
        if c_ == 0:
            ax_t.set_ylabel("p2p (mV)")
            ax_r.set_ylabel("rest / p1 (%)")
    for k in range(n, nrow * ncol):
        axes[k][0].axis("off"); axes[k][1].axis("off")

    handles = []
    for l, c in zip(labels, colours):
        handles += [Line2D([], [], color=c, lw=2, marker="o", ms=7, mew=0, label=f"{l} - pulse 1"),
                    Line2D([], [], color=c, lw=1.6, ls="--", marker="^", ms=7, mew=0, label=f"{l} - rest (mean 2-{n_pulses})")]
    handles.append(Line2D([], [], ls="", marker="o", mfc="white", mec="0.4", ms=5, mew=1.2,
                          label="rejected: below criterion / artifact"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), fontsize=11,
               frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.95, bottom=0.04)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
    return runs


# ---------------------------------------------------------------------------
# RELIABILITY of the automatic peak-to-peak
# ---------------------------------------------------------------------------
def detection_flags(res, muscles, edge_ms=1.0, jitter_ms=3.0):
    """Automatic sanity checks on every detected max / min. Returns
    flags[muscle] = bool array [n_intensities, n_pulses] (True = suspicious) and a
    dict of reasons. A pulse is flagged when, in a RESPONDING train,

      edge   - its max or min sits within `edge_ms` of a window edge: the peak is
               probably cut off, or the window is catching the artifact tail / the
               next pulse's onset
      jitter - the time of its max (or min) relative to its own pulse onset differs
               by more than `jitter_ms` from the train's median: it is not the same
               deflection as the other pulses (noise spike, movement, a different wave)
    """
    onset = res["pulse_ms"]
    flags, why = {}, {}
    for m in muscles:
        tmax, tmin = res["tmax"][m], res["tmin"][m]
        wins = np.array(res["wins"][m])            # [n_pulses, 2]
        f = np.zeros_like(tmax, bool); reasons = {}
        for w in range(tmax.shape[0]):
            if not res["responding"][m][w] or np.isnan(tmax[w]).all():
                continue
            lat_max, lat_min = tmax[w] - onset, tmin[w] - onset
            med_max, med_min = np.nanmedian(lat_max), np.nanmedian(lat_min)
            for k in range(tmax.shape[1]):
                if np.isnan(tmax[w, k]):
                    continue
                r = []
                a, b = wins[k]
                if (min(tmax[w, k] - a, b - tmax[w, k]) <= edge_ms
                        or min(tmin[w, k] - a, b - tmin[w, k]) <= edge_ms):
                    r.append("edge")
                if (abs(lat_max[k] - med_max) > jitter_ms
                        or abs(lat_min[k] - med_min) > jitter_ms):
                    r.append("jitter")
                if r:
                    f[w, k] = True; reasons[(w, k)] = "+".join(r)
        flags[m] = f; why[m] = reasons
    return flags, why


def detection_report(res, muscles, meta=None, edge_ms=1.0, jitter_ms=3.0):
    """Print, per muscle, how many detected pulses look suspicious (see
    `detection_flags`) and at which intensities. Returns the flags."""
    flags, why = detection_flags(res, muscles, edge_ms, jitter_ms)
    amps = res["amps"]
    print(f"{'muscle':22s} {'flagged/analysed':>16s}  {'reason':<12s} where (mA: pulses)")
    for m in muscles:
        f = flags[m]; ok = res["responding"][m]
        n_tot = int((~np.isnan(res["p2p"][m][ok])).sum())
        n_bad = int(f.sum())
        if n_tot == 0:
            print(f"{pretty(m):22s} {'-':>16s}  {'':<12s} no responding train"); continue
        reasons = [r for r in why[m].values()]
        n_edge = sum("edge" in r for r in reasons); n_jit = sum("jitter" in r for r in reasons)
        reason = ("edge" if n_edge > 2 * n_jit else "jitter" if n_jit > 2 * n_edge
                  else "edge+jitter") if n_bad else ""
        where = []
        for w in np.where(f.any(axis=1))[0]:
            ks = np.where(f[w])[0] + 1
            where.append(f"{amps[w]}: all" if len(ks) == f.shape[1]
                         else f"{amps[w]}: {','.join(map(str, ks))}")
        verdict = ("  <-- artifact / window problem" if reason == "edge" and n_bad / n_tot > 0.5
                   else "  <-- check" if n_bad / n_tot > 0.2 else "")
        print(f"{pretty(m):22s} {n_bad:>7d} / {n_tot:<6d}  {reason:<12s} {'; '.join(where)}{verdict}")
    print("\n  edge   = max/min sits on the window border -> the 'peak' is the artifact tail or a cut-off wave")
    print("  jitter = that pulse's peak is at a different time than the others -> not the same deflection")
    return flags


def plot_pulse_overlay(meta, t, sig, muscles, amp, n_pulses=10, resp_start_ms=8.0,
                       resp_end_ms=None, guard_ms=1.0, min_snr=None, max_edge_frac=0.5, edge_ms=1.0,
                       jitter_ms=3.0, title=None, ncol=4, save=None):
    """THE reliability check. Per muscle, for ONE intensity: the N pulse segments cut
    out of the trace and RE-ALIGNED to their own pulse onset (t = 0), overlaid, colour
    = pulse number (dark = pulse 1, light = pulse N). Response window shaded. Each
    segment carries its detected max (v) and min (^).

    If the automatic detection is picking the same deflection every time, the markers
    stack on top of each other. A marker off on its own is a misdetection - those are
    drawn with a red ring and listed by `detection_report`.
    """
    all_ch = [c for c in sig if c != "Trigger A"]
    muscles = resolve_muscles(all_ch, muscles)
    res = burst_p2p(meta, t, sig, muscles, n_pulses, resp_start_ms, resp_end_ms, guard_ms, min_snr, max_edge_frac)
    flags, _ = detection_flags(res, muscles, edge_ms, jitter_ms)
    amps = res["amps"]; w = int(np.where(amps == amp)[0][0])
    onset = res["pulse_ms"]; ipi = res["ipi_ms"]
    cmap = plt.get_cmap("viridis")
    n = len(muscles); ncol = min(ncol, n); nrow = int(np.ceil(n / ncol))
    pw = 4.8 if ncol > 1 else 7.0
    fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, 3.4 * nrow + (0.8 if n == 1 else 0)),
                             squeeze=False)
    axes = axes.ravel()
    for i, m in enumerate(muscles):
        ax = axes[i]
        a_rel, b_rel = np.array(res["wins"][m][0]) - onset[0]
        ax.axvspan(a_rel, b_rel, color="#2ca25f", alpha=0.10, zorder=0)
        for k in range(len(onset)):
            seg = (t >= onset[k] - 2) & (t <= onset[k] + ipi)
            col = cmap(k / max(len(onset) - 1, 1))
            ax.plot(t[seg] - onset[k], sig[m][w, seg], color=col, lw=1.0, alpha=0.85)
            tx, ty = res["tmax"][m][w, k] - onset[k], res["ymax"][m][w, k]
            tn, yn = res["tmin"][m][w, k] - onset[k], res["ymin"][m][w, k]
            if np.isfinite(tx):
                ax.plot(tx, ty, "v", color=col, mec="black", mew=0.5, ms=6, zorder=5)
                ax.plot(tn, yn, "^", color=col, mec="black", mew=0.5, ms=6, zorder=5)
                if flags[m][w, k]:
                    ax.plot([tx, tn], [ty, yn], "o", mfc="none", mec="#d62728", ms=13, mew=1.8,
                            zorder=6)
        # zoom to the response window
        inwin = np.zeros_like(t, bool)
        for a, b in res["wins"][m]:
            inwin |= (t >= a) & (t <= b)
        if inwin.any():
            v = sig[m][w][inwin]; pad = 0.25 * (v.max() - v.min()) or 0.01
            ax.set_ylim(v.min() - pad, v.max() + pad)
        ax.set_xlim(-2, ipi)
        ax.axvline(0, color="red", lw=1)
        ttl = pretty(m)
        if not res["responding"][m][w]:
            ttl += "  (ARTIFACT)" if res["reason"][m][w] == "artifact" else "  (below criterion)"
        nb = int(flags[m][w].sum())
        if nb:
            ttl += f"  - {nb} flagged"
        ax.set_title(ttl, fontweight="bold", fontsize=12,
                     color=("#d62728" if nb else "black"))
        ax.grid(alpha=0.25)
        if i % ncol == 0: ax.set_ylabel("EMG (mV)")
        if i >= n - ncol: ax.set_xlabel("Time from pulse onset (ms)")
    for k in range(n, len(axes)):
        axes[k].axis("off")
        if k - ncol >= 0: axes[k - ncol].tick_params(labelbottom=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, len(onset)))
    cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.01)
    cb.set_label("pulse #")
    if title:
        axes[0].annotate(title, (0, 1.25), xycoords="axes fraction", fontweight="bold",
                         fontsize=14, ha="left", va="bottom", annotation_clip=False)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight"); print("saved", save)
    plt.show()
    return res
