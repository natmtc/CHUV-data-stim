"""Comparing several participants.

Nothing in mA or mV can be compared between people - different anatomy, electrode placement,
impedance and EMG gain. Every figure here therefore normalises WITHIN a participant first:

    intensity  ->  multiples of that muscle's own motor threshold  (x MT)
    amplitude  ->  % of that muscle's own biggest response in that recording

What is then compared across participants is a ratio or a shape, never a raw number.
"""
import numpy as np
import matplotlib.pyplot as plt

from .io import load_run
from .labels import pretty
from .burst import burst_p2p, resolve_muscles


def recruitment(csv, muscles, mt, **kw):
    """One recording, normalised: for every muscle, its intensities as multiples of its own
    motor threshold and its pulse-1 p2p as % of its own biggest response.

    mt : {muscle: mA} for this recording. Muscles with no threshold are skipped.
    Returns {muscle: (x, y)} with x in x MT and y in %.
    """
    meta, t, sig = load_run(csv)
    chans = resolve_muscles([c for c in sig if c != "Trigger A"], muscles)
    res = burst_p2p(meta, t, sig, chans, min_snr=None, max_edge_frac=None,
                    **{k: v for k, v in kw.items() if k not in ("min_snr", "max_edge_frac")})
    amps = np.asarray(res["amps"], float)
    out = {}
    for m in chans:
        th = mt.get(pretty(m))
        if not th:
            continue
        p1 = np.array([res["p2p"][m][k, 0] for k in range(len(amps))], float)
        top = np.nanmax(p1) if np.isfinite(p1).any() else np.nan
        if not np.isfinite(top) or top <= 0:
            continue
        out[pretty(m)] = (amps / th, 100.0 * p1 / top)
    return out


def _grid(n, ncol=4, w=4.2, h=3.0):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    return fig, axes.ravel(), nrow


def _finish(fig, axes, n, ncol, title, save):
    for k in range(n, len(axes)):
        axes[k].axis("off")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()


# ---------------------------------------------------------------------------
# 1. how much more current one protocol needs than the other
# ---------------------------------------------------------------------------
def fig_threshold_ratio(MT, subjects, muscles, protocols=("burst", "arcex"), colours=None,
                        names=None, title=None, save=None):
    """Per muscle and participant: threshold(protocol B) / threshold(protocol A).

    A ratio cancels everything participant-specific, so this is the one threshold figure that
    can be read across people. 1.0 = the two protocols recruit that muscle at the same current.

    MT : {(subject, protocol): {muscle: mA}}
    Returns the ratios as {subject: {muscle: ratio}}.
    """
    names = names or {"burst": "30 Hz burst", "arcex": "ARC-EX"}
    colours = colours or dict(zip(subjects, ["#1f3b73", "#e6550d", "#2ca25f", "#9467bd"]))
    a, b = protocols
    ratio = {s: {} for s in subjects}
    for s in subjects:
        for m in muscles:
            va, vb = MT.get((s, a), {}).get(m), MT.get((s, b), {}).get(m)
            if va and vb:
                ratio[s][m] = vb / va

    y = {m: len(muscles) - 1 - i for i, m in enumerate(muscles)}
    off = np.linspace(-0.22, 0.22, len(subjects))
    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(muscles) + 2.0))
    ax.axvline(1.0, color="0.45", ls="--", lw=1.2, zorder=1)
    for j, s in enumerate(subjects):
        xs = [ratio[s].get(m) for m in muscles]
        ax.plot([x for x in xs if x], [y[m] + off[j] for m, x in zip(muscles, xs) if x],
                "o", ms=9, mew=1.3, mec="white", color=colours[s], label=s, zorder=3)
    for m in muscles:
        ax.axhline(y[m], color="0.93", lw=0.8, zorder=0)
    vals = [v for s in subjects for v in ratio[s].values()]
    if vals:
        ax.set_xlim(0, max(vals) * 1.12)
    ax.set_yticks([y[m] for m in muscles], muscles, fontsize=11)
    ax.set_ylim(-0.7, len(muscles) - 0.3)
    ax.set_xlabel(f"{names.get(b, b)} threshold  /  {names.get(a, a)} threshold", fontsize=11,
                  color="0.2")
    ax.tick_params(labelsize=10, colors="0.3"); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("0.3")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=len(subjects), frameon=False,
              fontsize=11)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()
    for s in subjects:
        v = np.array(list(ratio[s].values()), float)
        if len(v):
            print(f"{s}: {names.get(b, b)} needs {np.nanmean(v):.2f} +- {np.nanstd(v):.2f} x the "
                  f"current of {names.get(a, a)}  ({len(v)} muscles)")
    return ratio


# ---------------------------------------------------------------------------
# 2. the shape of the recruitment, on a common axis
# ---------------------------------------------------------------------------
def fig_recruitment(curves, muscles, subjects, protocols=("burst", "arcex"), colours=None,
                    names=None, xmax=2.0, title=None, save=None):
    """One panel per muscle: response vs intensity, both axes normalised within participant.

    curves : {(subject, protocol): {muscle: (x in x MT, y in %)}} from `recruitment`.
    Colour = participant, line style = protocol (solid = first, dashed = second). Everything
    crosses 100 % at its own maximum and 1.0 at its own threshold, so what is left to compare
    is the STEEPNESS: how fast the response grows once the muscle starts responding.
    """
    names = names or {"burst": "30 Hz burst", "arcex": "ARC-EX"}
    colours = colours or dict(zip(subjects, ["#1f3b73", "#e6550d", "#2ca25f", "#9467bd"]))
    ls = dict(zip(protocols, ["-", "--"]))
    ncol = min(4, len(muscles))
    fig, axes, _ = _grid(len(muscles), ncol=ncol)
    for i, m in enumerate(muscles):
        ax = axes[i]
        for s in subjects:
            for p in protocols:
                xy = curves.get((s, p), {}).get(m)
                if xy is None:
                    continue
                x, y = xy
                k = x <= xmax
                ax.plot(x[k], y[k], ls[p], color=colours[s], lw=1.7, alpha=0.9)
        ax.axvline(1.0, color="0.55", ls=":", lw=1.2, zorder=0)
        ax.set_title(m, fontsize=12, fontweight="bold")
        ax.set_xlim(0, xmax); ax.set_ylim(-5, 105)
        ax.tick_params(labelsize=10, colors="0.3")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i % ncol == 0:
            ax.set_ylabel("pulse-1 p2p (% of its max)", fontsize=10, color="0.2")
        if i >= len(muscles) - ncol:
            ax.set_xlabel("intensity (x motor threshold)", fontsize=10, color="0.2")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colours[s], lw=2, label=s) for s in subjects]
    handles += [Line2D([], [], color="0.35", lw=2, ls=ls[p], label=names.get(p, p))
                for p in protocols]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    _finish(fig, axes, len(muscles), ncol, title, save)


# ---------------------------------------------------------------------------
# 3. selectivity: how many muscles come in as the current is raised
# ---------------------------------------------------------------------------
def fig_selectivity(MT, subjects, muscles, protocols=("burst", "arcex"), levels=(1.0, 1.2, 1.5),
                    colours=None, names=None, title=None, save=None):
    """How many of the muscles are recruited at 1.0, 1.2, 1.5 x the LOWEST threshold of that
    recording - i.e. how much of the arm comes in once anything does.

    A selective protocol recruits one or two muscles at 1.0 and adds few by 1.5; a broad one
    brings most of them in at once. Referencing to the recording's own lowest threshold keeps
    it comparable between people.
    """
    names = names or {"burst": "30 Hz burst", "arcex": "ARC-EX"}
    colours = colours or {protocols[0]: "#1f3b73", protocols[1]: "#e6550d"}
    fig, ax = plt.subplots(figsize=(1.7 * len(subjects) * len(protocols) + 2.5, 4.4))
    width = 0.8 / len(levels)
    xt, xl = [], []
    pos = 0
    table = {}
    for s in subjects:
        for p in protocols:
            th = [v for m, v in MT.get((s, p), {}).items() if m in muscles and v]
            if not th:
                pos += 1
                continue
            lo = min(th)
            fr = [sum(1 for v in th if v <= lv * lo) / len(muscles) * 100 for lv in levels]
            table[(s, p)] = (lo, len(th), fr)
            for k, (lv, f) in enumerate(zip(levels, fr)):
                ax.bar(pos + (k - (len(levels) - 1) / 2) * width, f, width=width * 0.92,
                       color=colours[p], alpha=0.45 + 0.25 * k, zorder=2,
                       label=f"{lv:g} x MT" if (s, p) == (subjects[0], protocols[0]) else None)
            xt.append(pos); xl.append(f"{s}\n{names.get(p, p)}")
            pos += 1
    ax.set_xticks(xt, xl, fontsize=10)
    ax.set_ylabel(f"% of the {len(muscles)} muscles responding", fontsize=11, color="0.2")
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=10, colors="0.3")
    ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10, ncol=len(levels), loc="upper left")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()
    for (s, p), (lo, n, fr) in table.items():
        print(f"{s:5s} {names.get(p, p):12s} lowest threshold {lo:g} mA, {n}/{len(muscles)} "
              f"muscles with one; recruited: "
              + ", ".join(f"{lv:g}xMT {f:.0f}%" for lv, f in zip(levels, fr)))
    return table


# ---------------------------------------------------------------------------
# 4. the headline grid: a few muscles, every participant, both protocols
# ---------------------------------------------------------------------------
def fig_muscle_grid(curves, MT, muscles, subjects, protocols=("burst", "arcex"), colours=None,
                    names=None, xmax=2.0, title=None, save=None):
    """One row per muscle, one column per participant, both protocols in every panel.

    Read DOWN a column for one participant, ACROSS a row for one muscle. Both axes are
    normalised inside each participant - x is intensity as a multiple of that muscle's own motor
    threshold, y is its response as % of its own biggest - so the panels are directly comparable
    even though the mA behind them are not. The mA each curve started from is printed in the
    corner of its panel.
    """
    names = names or {"burst": "30 Hz burst", "arcex": "ARC-EX"}
    colours = colours or {protocols[0]: "#1f3b73", protocols[1]: "#e6550d"}
    ls = {protocols[0]: "-", protocols[1]: "--"}
    nr, nc = len(muscles), len(subjects)
    fig, axes = plt.subplots(nr, nc, figsize=(3.9 * nc, 2.9 * nr), squeeze=False,
                             sharex=True, sharey=True)
    for i, m in enumerate(muscles):
        for j, s in enumerate(subjects):
            ax = axes[i][j]
            ax.axvline(1.0, color="0.6", ls=":", lw=1.2, zorder=1)
            ax.axhline(50, color="0.93", lw=0.8, zorder=0)
            lab = []
            for p in protocols:
                xy = curves.get((s, p), {}).get(m)
                th = MT.get((s, p), {}).get(m)
                if xy is None:
                    continue
                x, y = xy
                k = x <= xmax
                ax.plot(x[k], y[k], ls[p], color=colours[p], lw=2.0, alpha=0.95, zorder=3)
                lab.append(f"{names.get(p, p)} {th:g} mA" if th else names.get(p, p))
            if lab:
                ax.annotate("\n".join(lab), (0.03, 0.97), xycoords="axes fraction", va="top",
                            ha="left", fontsize=8.5, color="0.35")
            else:
                ax.annotate("no threshold", (0.5, 0.5), xycoords="axes fraction", ha="center",
                            va="center", fontsize=10, color="0.6")
            if i == 0:
                ax.set_title(s, fontsize=13, fontweight="bold", pad=8)
            if j == 0:
                ax.set_ylabel(f"{m}\n\n% of its max", fontsize=10, color="0.2")
            if i == nr - 1:
                ax.set_xlabel("intensity (x motor threshold)", fontsize=10, color="0.2")
            ax.set_xlim(0, xmax); ax.set_ylim(-5, 108)
            ax.tick_params(labelsize=9, colors="0.3")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colours[p], lw=2.2, ls=ls[p], label=names.get(p, p))
               for p in protocols]
    handles.append(Line2D([], [], color="0.6", ls=":", lw=1.2, label="motor threshold (x1)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.045)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()


# ---------------------------------------------------------------------------
# 5. the three participants in ONE plot
# ---------------------------------------------------------------------------
def fig_across_subjects(curves, muscles, subjects, protocols=("burst", "arcex"), colours=None,
                        names=None, ref=None, band=True, title=None, save=None):
    """One panel per protocol, one line per participant: the muscles averaged.

    The problem this solves: "% of its own max" makes every curve's reference the top of its own
    sweep, and the sweeps do not go equally far - one participant's 100 % can be a saturated
    response and another's a barely supra-threshold one. Here every curve is put on a common
    x grid that ALL the recordings reach, and normalised to its own response AT THAT POINT, so
    100 % means the same thing everywhere. `ref` overrides that shared endpoint (in x MT).

    Line = mean over `muscles`, band = +-SD across them, colour = participant.
    Returns the reference used and the per-subject values at it.
    """
    names = names or {"burst": "30 Hz burst", "arcex": "ARC-EX"}
    colours = colours or dict(zip(subjects, ["#1f3b73", "#e6550d", "#2ca25f", "#9467bd"]))

    have = [(s, p, m) for s in subjects for p in protocols
            for m in muscles if curves.get((s, p), {}).get(m) is not None]
    if not have:
        raise ValueError("none of those muscles has a curve in any recording")
    reach = {(s, p, m): float(np.nanmax(curves[(s, p)][m][0])) for s, p, m in have}
    lim = min(reach, key=reach.get)
    xmax = ref or reach[lim]
    grid = np.linspace(0, xmax, 60)

    fig, axes = plt.subplots(1, len(protocols), figsize=(5.6 * len(protocols), 4.6),
                             squeeze=False, sharey=True)
    axes = axes[0]
    out = {}
    for ax, p in zip(axes, protocols):
        for s in subjects:
            ys = []
            for m in muscles:
                xy = curves.get((s, p), {}).get(m)
                if xy is None:
                    continue
                x, y = xy
                k = np.argsort(x)
                yi = np.interp(grid, x[k], y[k], left=np.nan, right=np.nan)
                at = np.interp(xmax, x[k], y[k])
                if np.isfinite(at) and at > 0:
                    ys.append(100.0 * yi / at)
            if not ys:
                continue
            a = np.vstack(ys)
            import warnings as _w
            with _w.catch_warnings():        # the low end of the grid is below some ladders
                _w.simplefilter("ignore", RuntimeWarning)
                mu, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
            out[(s, p)] = (len(ys), float(mu[-1]))
            ax.plot(grid, mu, "-", color=colours[s], lw=2.6, zorder=3, label=s)
            if band and len(ys) > 1:
                ax.fill_between(grid, mu - sd, mu + sd, color=colours[s], alpha=0.14, lw=0,
                                zorder=2)
        ax.axvline(1.0, color="0.55", ls=":", lw=1.3, zorder=1)
        ax.axhline(100, color="0.85", lw=0.9, zorder=0)
        ax.set_title(names.get(p, p), fontsize=13, fontweight="bold")
        ax.set_xlabel("intensity (x motor threshold)", fontsize=11, color="0.2")
        ax.set_xlim(0, xmax)
        ax.tick_params(labelsize=10, colours="0.3") if False else ax.tick_params(labelsize=10,
                                                                                colors="0.3")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(f"% of response at {xmax:.2f} x MT", fontsize=11.5, color="0.25")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(subjects), frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, -0.02, f"line = mean of {len(muscles)} muscles, band = SD",
             ha="center", fontsize=9.5, color="0.45")
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()
    print(f"common reference: {xmax:.2f} x MT  (the furthest EVERY recording reaches)")
    if ref is None:
        print(f"  set by {lim[0]} {names.get(lim[1], lim[1])} {lim[2]} - its sweep stops there")
        short = {f"{s} {names.get(p, p)}": round(min(reach[(s, p, m)] for m in muscles
                                                     if (s, p, m) in reach), 2)
                 for s in subjects for p in protocols
                 if any((s, p, m) in reach for m in muscles)}
        print("  each recording reaches: " + ", ".join(f"{k} {v}x" for k, v in short.items()))
    return xmax, out


# ---------------------------------------------------------------------------
# 6. one muscle, the participants side by side: how big, and how it holds up
# ---------------------------------------------------------------------------
def fig_bars(runs, MT, muscle, subjects, protocols=("burst", "arcex"), steps=0, colours=None,
             names=None, min_snr_ratio=3.0, title=None, save=None, **kw):
    """One muscle, one figure: participants side by side, a bar per protocol.

    Left   - the 1st pulse as a multiple of that recording's own baseline noise. NOT in mV:
             millivolts are not comparable between people (different gain, placement, anatomy),
             but "how far above its own noise" is.
    Middle - the 2nd pulse as % of the 1st.
    Right  - the mean of pulses 2..N as % of the 1st.

    Each muscle is taken at ITS own motor threshold in that recording, `steps` intensities up.
    The two ratio panels divide by the 1st pulse, so a small 1st pulse makes them explode or
    collapse: a bar whose 1st pulse is under `min_snr_ratio` x its noise is left out of those two
    panels and listed underneath, rather than drawn as if it meant something.
    runs : {(subject, protocol): csv}      MT : {(subject, protocol): {muscle: mA}}
    Returns the numbers behind the bars.
    """
    import warnings as _w
    from .io import load_run
    from .burst import burst_p2p, noise_p2p
    from .paper import step_up
    names = names or {"burst": "30 Hz burst", "arcex": "ARC-EX"}
    colours = colours or {protocols[0]: "#1f3b73", protocols[1]: "#e6550d"}
    hatch = {protocols[0]: "", protocols[1]: "///"}

    val = {}
    for s in subjects:
        for p in protocols:
            csv = runs.get((s, p))
            th = MT.get((s, p), {}).get(muscle)
            if not csv or not th:
                continue
            amp = step_up({muscle: th}, csv, steps, verbose=False).get(muscle) if steps else th
            if not amp:
                continue
            meta, t, sig = load_run(csv)
            chans = [c for c in sig if c != "Trigger A"]
            ch = next((c for c in chans if pretty(c) == muscle), None)
            amps = [m["amp_ma"] for m in meta]
            if ch is None or amp not in amps:
                continue
            res = burst_p2p(meta, t, sig, chans, min_snr=None, max_edge_frac=None,
                            **{k: v for k, v in kw.items()
                               if k not in ("min_snr", "max_edge_frac")})
            w = list(res["amps"]).index(amp)
            y = np.asarray(res["p2p"][ch][w], float)
            b = np.asarray(noise_p2p(t, sig, chans, win_len_ms=res["win"][0][1] - res["win"][0][0])[ch],
                           float)
            b = float(b[w]) if b.ndim else float(b)
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                p1, rest = float(y[0]), float(np.nanmean(y[1:]))
            if not np.isfinite(p1) or p1 <= 0:
                continue
            val[(s, p)] = dict(amp=amp, snr=(p1 / b if b > 0 else np.nan),
                               p2=100 * float(y[1]) / p1, rest=100 * rest / p1, p1_mV=p1)

    panels = [("snr", "1st pulse", "x baseline noise", None),
              ("p2", "2nd pulse", "% of 1st pulse", 100),
              ("rest", "mean of pulses 2-10", "% of 1st pulse", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    x = np.arange(len(subjects)); w = 0.8 / len(protocols)
    for ax, (key, ttl, ylab, line) in zip(axes, panels):
        for j, p in enumerate(protocols):
            h = [val.get((s, p), {}).get(key, np.nan) if
                 (key == "snr" or val.get((s, p), {}).get("snr", 0) >= min_snr_ratio)
                 else np.nan for s in subjects]
            ax.bar(x + (j - (len(protocols) - 1) / 2) * w, h, width=w * 0.9,
                   facecolor=colours[p], alpha=0.85, hatch=hatch[p],
                   edgecolor=("white" if hatch[p] else "none"), lw=0, zorder=2,
                   label=names.get(p, p) if key == "snr" else None)
            for xi, v in zip(x + (j - (len(protocols) - 1) / 2) * w, h):
                if np.isfinite(v):
                    ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=8.5, color="0.35")
        if key == "snr":
            # one participant can be 10x another here, which flattens the rest on a linear
            # axis; log keeps every bar readable and puts the noise floor at 1
            ax.set_yscale("log")
            ax.axhline(1, color="0.4", lw=0.9, ls=(0, (2, 3)), zorder=1)
            ax.set_ylim(bottom=0.9)
        if line:
            ax.axhline(line, color="0.4", lw=0.9, ls=(0, (2, 3)), zorder=1)
        ax.set_xticks(x, subjects, fontsize=12)
        ax.set_title(ttl, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=11, color="0.25")
        ax.tick_params(labelsize=10, colors="0.3")
        ax.grid(True, axis="y", alpha=0.22); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=len(protocols), frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.995))
    weak = [f"{s} {names.get(p, p)}" for (s, p), v in sorted(val.items())
            if v["snr"] < min_snr_ratio]
    if weak:
        fig.text(0.5, -0.02, "blank = 1st pulse too close to noise to divide by  ("
                 + ", ".join(weak) + ")", ha="center", fontsize=9.5, color="#d62728")
    fig.suptitle(title or muscle, fontsize=15, fontweight="bold", y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show()
    return val
