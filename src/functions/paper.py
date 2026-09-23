"""Paper-style summary figures (clean, no grids, scale bars, one colour per condition).

fig_train_modes(specs, muscles): any number of conditions (stimulation mode and/or
experimental state) overlaid. Each condition may use a different intensity per muscle.
    left  - raw traces of 2-3 muscles, first N pulses, all conditions overlaid, mV scale bar
    right - peak-to-peak of pulse 1 vs mean of pulses 2..N, each condition normalised to its
            own pulse 1 (100 %): bar = mean over the muscles shown, whisker = SD, one dot per
            muscle, thin lines join the same muscle across conditions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .io import load_run, detect_pulses
from .labels import pretty
from .burst import burst_p2p, resolve_muscles


def _amp_for(spec, m):
    """A spec's `amp` is a number, or {muscle: mA} keyed by pretty label or channel name."""
    a = spec["amp"]
    if isinstance(a, dict):
        if m in a:
            return a[m]
        if pretty(m) in a:
            return a[pretty(m)]
        if "default" in a:
            return a["default"]
        raise KeyError(f"no intensity given for {pretty(m)} in {spec['label']!r} "
                       f"(add it to the dict, or a 'default' entry)")
    return a


def _load(spec, muscles, kw):
    meta, t, sig = load_run(spec["csv"])
    chans = [c for c in sig if c != "Trigger A"]
    amps = np.array([m["amp_ma"] for m in meta])
    res = burst_p2p(meta, t, sig, chans, **kw)
    ms = resolve_muscles(chans, muscles)
    w, used = {}, {}
    for m in ms:
        a = _amp_for(spec, m)
        idx = np.where(amps == a)[0]
        if not len(idx):
            raise ValueError(f"{a} mA not in {spec['csv'].split('/')[-1]} "
                             f"({spec['label']}): {sorted(set(amps))}")
        w[m], used[m] = int(idx[0]), a
    return dict(meta=meta, t=t, sig=sig, w=w, amp_used=used, res=res,
                pulses=detect_pulses(t, sig["Trigger A"]), muscles=ms,
                label=spec["label"], colour=spec.get("colour", "black"),
                ls=spec.get("linestyle", "-"), hatch=spec.get("hatch", ""))


def _nice_scale(rng):
    """A round mV value for the scale bar, ~1/3 of the panel range."""
    for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5):
        if v >= rng / 3:
            return v
    return 5


def fig_train_modes(specs, muscles, n_pulses=10, resp_start_ms=8.0, guard_ms=1.0, min_snr=None,
                    max_edge_frac=None, xlim=None, title=None, save=None):
    """specs: list of dict(label=, csv=, amp=, colour=, linestyle=, hatch=).
    `amp` is one mA for every muscle, or {muscle: mA} (pretty label or channel; 'default' allowed).
    muscles: 2-3 labels.

    left  - the N pulses of each condition overlaid, one row per muscle, mV scale bar, the mA used
            written in each panel
    right - pulse 1 (reference, outlined) vs mean of pulses 2..N (filled), each condition as % of
            its own pulse 1; bar = mean over the muscles, whisker = SD, dots = muscles
    """
    kw = dict(n_pulses=n_pulses, resp_start_ms=resp_start_ms, guard_ms=guard_ms,
              min_snr=min_snr, max_edge_frac=max_edge_frac)
    runs = [_load(s, muscles, kw) for s in specs]
    muscles = runs[0]["muscles"]
    npul = min(len(r["res"]["win"]) for r in runs)
    if xlim is None:
        xlim = (-8, max(r["res"]["win"][npul - 1][1] for r in runs) + 4)

    nm, nc = len(muscles), len(runs)
    fig = plt.figure(figsize=(13, 2.35 * nm + 1.5))
    gs = fig.add_gridspec(nm, 2, width_ratios=[2.45, 1], wspace=0.24, hspace=0.42,
                          left=0.03, right=0.985, top=0.83, bottom=0.10)

    # ---------- left: traces ----------
    for i, m in enumerate(muscles):
        ax = fig.add_subplot(gs[i, 0])
        lo, hi = np.inf, -np.inf
        for r in runs:
            t, y = r["t"], r["sig"][m][r["w"][m]]
            mask = (t >= xlim[0]) & (t <= xlim[1])
            ax.plot(t[mask], y[mask], color=r["colour"], lw=1.15, alpha=0.92, ls=r["ls"],
                    solid_joinstyle="round")
            inwin = np.zeros_like(t, bool)
            for a, b in r["res"]["wins"][m][:npul]:
                inwin |= (t >= a) & (t <= b)
            if inwin.any():
                lo, hi = min(lo, y[inwin].min()), max(hi, y[inwin].max())
        if not np.isfinite(lo):
            lo, hi = -0.05, 0.05
        pad = 0.28 * (hi - lo) or 0.01
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(*xlim)
        y_top = hi + pad
        for p0, _ in runs[0]["pulses"][:npul]:               # pulse onsets as ticks on the top edge
            ax.plot([p0, p0], [y_top - 0.07 * (hi - lo + 2 * pad), y_top], color="0.5", lw=1.1,
                    solid_capstyle="butt", clip_on=False)
        sb = _nice_scale(hi - lo)                            # mV scale bar outside the right edge
        x0 = xlim[1] + 0.012 * (xlim[1] - xlim[0])
        ax.plot([x0, x0], [lo, lo + sb], color="black", lw=2, solid_capstyle="butt", clip_on=False)
        ax.text(x0 + 0.01 * (xlim[1] - xlim[0]), lo + sb / 2, f"{sb:g} mV", ha="left", va="center",
                fontsize=9.5, clip_on=False)
        ax.text(0.0, 1.0, pretty(m), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, fontweight="bold")
        # the intensity each condition uses for THIS muscle, in its own colour (one per colour)
        seen, x_t = [], 1.0
        for r in reversed(runs):
            key = (r["colour"], r["amp_used"][m])
            if key in seen:
                continue
            seen.append(key)
            ax.annotate(f"{r['amp_used'][m]} mA", (x_t, 1.0), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=10.5, color=r["colour"], fontweight="bold")
            x_t -= 0.085
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("0.3")
        ax.tick_params(labelsize=10, colors="0.3")
        if i < nm - 1:
            ax.spines["bottom"].set_visible(False); ax.set_xticks([])
        else:
            ax.set_xlabel("Time from first pulse (ms)", fontsize=11, color="0.2")
    fig.text(0.03, 0.855, "ticks = pulse onsets · mA per condition written in each panel",
             fontsize=9, color="0.45", ha="left", va="bottom")

    # ---------- right: pulse 1 vs mean of the rest ----------
    ax = fig.add_subplot(gs[:, 1])
    bw = 0.74 / nc
    for j, r in enumerate(runs):
        rest = []
        for m in muscles:
            y = r["res"]["p2p"][m][r["w"][m], :npul]
            rest.append(100.0 * np.nanmean(y[1:]) / y[0] if np.isfinite(y[0]) and y[0] > 0 else np.nan)
        rest = np.array(rest)
        xoff = (j - (nc - 1) / 2) * bw
        ax.bar(0 + xoff, 100, width=bw * 0.88, facecolor="white", edgecolor=r["colour"], lw=1.6,
               hatch=r["hatch"])
        ax.bar(1 + xoff, np.nanmean(rest), width=bw * 0.88, facecolor=r["colour"], alpha=0.85,
               edgecolor=("white" if r["hatch"] else "none"), hatch=r["hatch"], lw=0,
               yerr=np.nanstd(rest), error_kw=dict(ecolor="0.15", lw=1.3, capsize=5))
        rng = np.random.default_rng(j)
        xs = 1 + xoff + rng.uniform(-bw * 0.2, bw * 0.2, len(rest))
        ax.scatter(xs, rest, s=32, facecolor="white", edgecolor=r["colour"], linewidth=1.4, zorder=5)
        r["_xs"], r["_rest"] = xs, rest
    if nc >= 2:
        for k in range(nm):
            xs = [r["_xs"][k] for r in runs]; ys = [r["_rest"][k] for r in runs]
            if np.all(np.isfinite(ys)):
                ax.plot(xs, ys, color="0.72", lw=0.8, zorder=4)
    ax.axhline(100, color="0.4", lw=0.9, ls=(0, (2, 3)))
    top = np.nanmax([np.nanmax(r["_rest"]) for r in runs] + [100])
    ax.set_ylim(0, top * 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1st pulse\n(reference)", f"pulses 2-{npul}\n(mean)"], fontsize=10.5)
    ax.set_ylabel("Peak-to-peak (% of 1st pulse)", fontsize=11, color="0.2")
    ax.tick_params(labelsize=10, colors="0.3")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.3")
    ax.text(0.5, -0.13, f"bar = mean of the {nm} muscles, whisker = SD, dots = muscles",
            transform=ax.transAxes, ha="center", fontsize=9, color="0.45")

    handles = [Patch(facecolor=r["colour"], alpha=0.85, hatch=r["hatch"],
                     edgecolor=("white" if r["hatch"] else "none"), label=r["label"]) for r in runs]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, (0.965 if title else 0.995)),
               columnspacing=2.2, handlelength=1.5)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()
    return {r["label"]: r["_rest"] for r in runs}


# ---------------------------------------------------------------------------
# depression along the train: 1st vs 2nd pulse, and 1st vs the N-1 following
# ---------------------------------------------------------------------------
def depression_stats(specs, muscles=None, n_pulses=10, resp_start_ms=8.0, guard_ms=1.0,
                     min_snr=2.0, max_edge_frac=0.5):
    """Per condition and muscle: pulse-1 p2p, pulse-2 p2p, mean of pulses 2..N, and the two
    ratios (as % of pulse 1). Only muscles with a motor response in EVERY condition are kept,
    so the conditions are compared on the same set (paired).

    Returns (table, kept, dropped) - table is a list of dicts, one row per condition x muscle."""
    import warnings
    kw = dict(n_pulses=n_pulses, resp_start_ms=resp_start_ms, guard_ms=guard_ms,
              min_snr=min_snr, max_edge_frac=max_edge_frac)
    runs = [_load(s, muscles, kw) for s in specs]
    ms = runs[0]["muscles"]
    npul = min(len(r["res"]["win"]) for r in runs)

    vals = {}
    for r in runs:
        for m in ms:
            y = r["res"]["p2p"][m][r["w"][m], :npul]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ok = np.isfinite(y[0]) and y[0] > 0 and np.isfinite(y[1:]).any()
                vals[(r["label"], m)] = dict(
                    p1=float(y[0]) if ok else np.nan,
                    p2=float(y[1]) if ok else np.nan,
                    rest=float(np.nanmean(y[1:])) if ok else np.nan,
                    amp=r["amp_used"][m])
    kept = [m for m in ms if all(np.isfinite(vals[(r["label"], m)]["p1"]) for r in runs)]
    dropped = [m for m in ms if m not in kept]

    table = []
    for r in runs:
        for m in kept:
            v = vals[(r["label"], m)]
            table.append(dict(condition=r["label"], muscle=pretty(m), channel=m, amp_ma=v["amp"],
                              p1_mV=v["p1"], p2_mV=v["p2"], rest_mV=v["rest"],
                              p2_minus_p1_mV=v["p2"] - v["p1"], rest_minus_p1_mV=v["rest"] - v["p1"],
                              p2_pct=100 * v["p2"] / v["p1"], rest_pct=100 * v["rest"] / v["p1"]))
    return table, kept, dropped


def fig_depression(specs, muscles=None, n_pulses=10, resp_start_ms=8.0, guard_ms=1.0,
                   min_snr=2.0, max_edge_frac=0.5, metric="ratio", title=None, csv_out=None,
                   save=None):
    """Paper figure: how much the response drops after the first pulse, per condition.

    Left panel  - 2nd pulse vs the 1st.
    Right panel - mean of pulses 2..N vs the 1st.
    metric="ratio" -> % of pulse 1 (100 % = no change) · "diff" -> difference in mV.
    Bar = mean over the muscles, whisker = SD, dots = the individual muscles, grey lines join the
    same muscle across conditions. Only muscles responding in every condition are used.
    """
    table, kept, dropped = depression_stats(specs, muscles, n_pulses, resp_start_ms, guard_ms,
                                            min_snr, max_edge_frac)
    labels = [s["label"] for s in specs]
    colours = [s.get("colour", "black") for s in specs]
    hatches = [s.get("hatch", "") for s in specs]
    keyA, keyB = ("p2_pct", "rest_pct") if metric == "ratio" else ("p2_minus_p1_mV", "rest_minus_p1_mV")
    ylab = "Peak-to-peak (% of 1st pulse)" if metric == "ratio" else "Peak-to-peak − 1st pulse (mV)"
    by = {(row["condition"], row["muscle"]): row for row in table}
    names = [pretty(m) for m in kept]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, key, ttl in zip(axes, (keyA, keyB),
                            ("2nd pulse", f"mean of pulses 2–{n_pulses}")):
        pts = []
        for j, (lab, col, hat) in enumerate(zip(labels, colours, hatches)):
            v = np.array([by[(lab, n)][key] for n in names], float)
            ax.bar(j, np.nanmean(v), width=0.6, facecolor=col, alpha=0.85, hatch=hat,
                   edgecolor=("white" if hat else "none"), lw=0,
                   yerr=np.nanstd(v), error_kw=dict(ecolor="0.15", lw=1.3, capsize=5), zorder=2)
            xs = j + rng.uniform(-0.13, 0.13, len(v))
            ax.scatter(xs, v, s=34, facecolor="white", edgecolor=col, linewidth=1.4, zorder=5)
            pts.append((xs, v))
        for k in range(len(names)):                      # join the same muscle across conditions
            xs = [p[0][k] for p in pts]; ys = [p[1][k] for p in pts]
            if np.all(np.isfinite(ys)):
                ax.plot(xs, ys, color="0.72", lw=0.8, zorder=4)
        ax.axhline(100 if metric == "ratio" else 0, color="0.4", lw=0.9, ls=(0, (2, 3)))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(ttl, fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=10, colors="0.3")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("0.3")
    axes[0].set_ylabel(ylab, fontsize=12, color="0.2")
    fig.text(0.5, -0.02, f"bar = mean of {len(names)} muscles, whisker = SD, dots = muscles "
             f"(grey lines join the same muscle)", ha="center", fontsize=9.5, color="0.45")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()

    print(f"muscles used ({len(names)}): " + ", ".join(names))
    if dropped:
        print("dropped (no response in at least one condition): " + ", ".join(pretty(m) for m in dropped))
    for lab in labels:
        p2 = np.array([by[(lab, n)]["p2_pct"] for n in names], float)
        rest = np.array([by[(lab, n)]["rest_pct"] for n in names], float)
        print(f"{lab:28s} 2nd pulse {np.nanmean(p2):5.0f} +- {np.nanstd(p2):4.0f} %   "
              f"pulses 2-{n_pulses} {np.nanmean(rest):5.0f} +- {np.nanstd(rest):4.0f} % of pulse 1")
    if csv_out:
        import os, pandas as pd
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        pd.DataFrame(table).to_csv(csv_out, index=False); print("saved", csv_out)
    return table


# ---------------------------------------------------------------------------
# motor threshold per muscle: the intensity where the response first appears
# ---------------------------------------------------------------------------
def motor_thresholds(specs, muscles=None, n_pulses=10, resp_start_ms=8.0, guard_ms=1.0,
                     min_snr=2.0, max_edge_frac=0.5, consecutive=2, verbose=True):
    """Lowest intensity at which a motor response appears, per condition and muscle.

    A train counts as a response when its pulse-1 peak-to-peak clears `min_snr` x the
    pre-stimulus baseline and it is not rejected as artifact (see `burst_p2p`). The threshold is
    the lowest intensity that passes **and stays passing for `consecutive` tested intensities** -
    one isolated pass is a noise spike, not a threshold.

    Returns {condition label: {muscle label: mA or None}}; prints the table unless verbose=False.
    Use it as the per-muscle intensity of the figures:  amp=MT["cathodic"].
    """
    kw = dict(n_pulses=n_pulses, resp_start_ms=resp_start_ms, guard_ms=guard_ms,
              min_snr=min_snr, max_edge_frac=max_edge_frac)
    out, snr_at = {}, {}
    for spec in specs:
        meta, t, sig = load_run(spec["csv"])
        chans = resolve_muscles([c for c in sig if c != "Trigger A"], muscles)
        res = burst_p2p(meta, t, sig, chans, **kw)
        amps = res["amps"]
        th, sn = {}, {}
        for m in chans:
            ok = res["responding"][m]
            hit = None
            for i in range(len(amps)):
                if all(ok[i:i + consecutive]) and len(ok[i:i + consecutive]) == consecutive:
                    hit = int(amps[i]); break
            if hit is None and ok.any():          # only a single passing intensity
                hit = int(amps[ok].min())
            th[pretty(m)] = hit
            sn[pretty(m)] = float(res["snr_pulse1"][m][list(amps).index(hit)]) if hit else np.nan
        out[spec["label"]] = th; snr_at[spec["label"]] = sn
    if verbose:
        labels = [s["label"] for s in specs]
        names = sorted({m for d in out.values() for m in d})
        print(f"motor threshold (mA) - first intensity with a response, held for {consecutive} steps")
        print(f"{'muscle':22s}" + "".join(f"{l:>22s}" for l in labels))
        for m in names:
            line = f"{m:22s}"
            for l in labels:
                v = out[l].get(m)
                line += f"{(f'{v} mA  (SNR {snr_at[l][m]:.1f})' if v else '-'):>22s}"
            print(line)
    return out


def muscles_with_threshold(MT, labels=None):
    """Muscles that have a motor threshold in every condition - the ones that can be compared."""
    labels = labels or list(MT)
    names = sorted({m for l in labels for m in MT[l]})
    return [m for m in names if all(MT[l].get(m) for l in labels)]


def fig_thresholds(specs, muscles=None, n_pulses=10, resp_start_ms=8.0, guard_ms=1.0,
                   min_snr=2.0, max_edge_frac=0.5, consecutive=2, MT=None, title=None, save=None):
    """Paper figure: the recruitment curve of each muscle with its motor threshold marked.

    One panel per muscle, x = intensity, y = pulse-1 peak-to-peak (mV). Filled marker = the train
    passes the response criterion, hollow = it does not. The vertical dashed line and the label
    are the motor threshold used by the analyses (from `MT` if given, else computed here).
    """
    kw = dict(n_pulses=n_pulses, resp_start_ms=resp_start_ms, guard_ms=guard_ms,
              min_snr=min_snr, max_edge_frac=max_edge_frac)
    MT = MT or motor_thresholds(specs, muscles, n_pulses, resp_start_ms, guard_ms, min_snr,
                                max_edge_frac, consecutive, verbose=False)
    runs = []
    for spec in specs:
        meta, t, sig = load_run(spec["csv"])
        chans = resolve_muscles([c for c in sig if c != "Trigger A"], muscles)
        res = burst_p2p(meta, t, sig, chans, **kw)
        raw = burst_p2p(meta, t, sig, chans, n_pulses=n_pulses, resp_start_ms=resp_start_ms,
                        guard_ms=guard_ms, min_snr=None, max_edge_frac=None)
        runs.append(dict(spec=spec, res=res, raw=raw, chans=chans))
    chans = runs[0]["chans"]

    n = len(chans); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow), squeeze=False, sharex=True)
    axes = axes.ravel()
    for i, m in enumerate(chans):
        ax = axes[i]
        for r in runs:
            col = r["spec"].get("colour", "black"); lab = r["spec"]["label"]
            a = r["res"]["amps"]; ok = r["res"]["responding"][m]
            p1 = np.array([r["raw"]["p2p"][m][k, 0] for k in range(len(a))], float)
            ax.plot(a, p1, "-", color=col, lw=1.6, alpha=0.9, zorder=2)
            ax.plot(a[ok], p1[ok], "o", color=col, ms=6, mew=0, zorder=3)
            ax.plot(a[~ok], p1[~ok], "o", mfc="white", mec=col, ms=5, mew=1.2, alpha=0.8, zorder=3)
            th = MT.get(lab, {}).get(pretty(m))
            if th:
                ax.axvline(th, color=col, ls="--", lw=1.4, alpha=0.9, zorder=1)
                ax.annotate(f"{th}", (th, 1.0), xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", fontsize=10, color=col, fontweight="bold")
        ax.set_title(pretty(m), fontsize=12, fontweight="bold", pad=16)
        ax.tick_params(labelsize=10, colors="0.3")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("0.3")
        if i % ncol == 0:
            ax.set_ylabel("1st pulse p2p (mV)", fontsize=11, color="0.2")
        if i >= n - ncol:
            ax.set_xlabel("Stim amplitude (mA)", fontsize=11, color="0.2")
    for k in range(n, len(axes)):
        axes[k].axis("off")
        if k - ncol >= 0:
            axes[k - ncol].tick_params(labelbottom=True)
            axes[k - ncol].set_xlabel("Stim amplitude (mA)", fontsize=11, color="0.2")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=s.get("colour", "black"), lw=2, marker="o", ms=6, mew=0,
                      label=s["label"]) for s in specs]
    handles += [Line2D([], [], ls="", marker="o", mfc="white", mec="0.4", ms=6, mew=1.2,
                       label="below criterion"),
                Line2D([], [], color="0.4", ls="--", lw=1.4, label="motor threshold")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.045)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()
    return MT
