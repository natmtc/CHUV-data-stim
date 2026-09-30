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

# figure-ready defaults: Arial, larger type, no grid. Scoped with plt.rc_context so importing
# this module never changes anyone else's figures.
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
}
# two protocols, and the conditions of the vibration figure
PROTOCOL_COLOURS = ["#9A9A9A", "#C0392B"]                       # 30 Hz grey, ARC-EX red
CONDITION_COLOURS = ["#C6C6C6", "#9A9A9A", "#C0392B", "#2E6F95"]  # baseline, off, ON, ON-2


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
    colours = colours or dict(zip(subjects, ["#4C4C4C", "#C0392B", "#2E6F95", "#8A6FA8"]))

    have = [(s, p, m) for s in subjects for p in protocols
            for m in muscles if curves.get((s, p), {}).get(m) is not None]
    if not have:
        raise ValueError("none of those muscles has a curve in any recording")
    # Normalise every curve to its response AT THRESHOLD - a point every recording has by
    # definition - and then let each run as far as its own sweep went. Truncating everyone to
    # the shortest sweep put the whole figure below threshold, where there is nothing to see.
    reach = {(s, p, m): float(np.nanmax(curves[(s, p)][m][0])) for s, p, m in have}
    at_x = ref or 1.0
    xmax = max(reach.values())
    grid = np.linspace(0, xmax, 90)

    rc = plt.rc_context(PAPER_RC); rc.__enter__()
    fig, axes = plt.subplots(1, len(protocols), figsize=(5.8 * len(protocols), 4.8),
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
                at = np.interp(at_x, x[k], y[k])
                if np.isfinite(at) and at > 0:
                    ys.append(100.0 * yi / at)
            if not ys:
                continue
            a = np.vstack(ys)
            import warnings as _w
            with _w.catch_warnings():        # the low end of the grid is below some ladders
                _w.simplefilter("ignore", RuntimeWarning)
                mu, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
            # only draw where EVERY muscle contributes, so "mean of N muscles" stays true -
            # otherwise the line quietly becomes one muscle where the others ran out
            full = np.sum(np.isfinite(a), axis=0) == len(ys)
            mu = np.where(full, mu, np.nan); sd = np.where(full, sd, np.nan)
            out[(s, p)] = (len(ys), float(mu[-1]))
            ax.plot(grid, mu, "-", color=colours[s], lw=2.6, zorder=3, label=s)
            if band and len(ys) > 1:
                ax.fill_between(grid, mu - sd, mu + sd, color=colours[s], alpha=0.14, lw=0,
                                zorder=2)
        ax.axvline(at_x, color="0.45", ls="--", lw=1.4, zorder=1)
        ax.annotate("threshold", (at_x, 0.98), xycoords=("data", "axes fraction"), ha="center",
                    va="top", fontsize=9.5, color="0.45",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.axhline(100, color="0.85", lw=0.9, zorder=0)
        ax.set_title(names.get(p, p), fontweight="bold")
        ax.set_xlabel("intensity (x motor threshold)", color="0.25")
        ax.set_xlim(0, xmax)
        ax.tick_params(colors="0.25")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("% of the response at threshold", color="0.25")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(subjects), frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, -0.10, "intensity is a multiple of each muscle's own motor threshold, so the "
             "current each protocol needs is divided out\n"
             f"line = mean of all {len(muscles)} muscles, drawn only where every one reaches; "
             "band = SD", ha="center", fontsize=12, color="0.45", linespacing=1.5)
    if title:
        fig.suptitle(title, fontweight="bold", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show(); rc.__exit__(None, None, None)
    print(f"every curve = 100 % at its own threshold ({at_x:g} x MT); each runs as far as its "
          f"sweep went:")
    for s_ in subjects:
        for p_ in protocols:
            got = [reach[(s_, p_, m)] for m in muscles if (s_, p_, m) in reach]
            if got:
                print(f"   {s_:5s} {names.get(p_, p_):12s} up to {min(got):.2f} x MT"
                      + ("   <- nothing above threshold" if min(got) <= 1.01 else ""))
    return at_x, out


# ---------------------------------------------------------------------------
# 6. one muscle, the participants side by side: how big, and how it holds up
# ---------------------------------------------------------------------------
def fig_bars(runs, MT, muscle, subjects, protocols=("burst", "arcex"), steps=0, colours=None,
             hatches=None, linestyles=None, names=None, min_snr_ratio=1.5, rest_from=2,
             with_p1=True, traces=True, trace_ms=None, title=None, save=None, **kw):
    """One muscle, one figure: participants side by side, a bar per protocol.

    The ratio panels are % of that condition's OWN 1st pulse, so 100 % = no change along the
    train: the 2nd pulse, and the mean of pulses `rest_from`..N.

    traces adds a row underneath: the EMG of that muscle at the very intensity the bars are
    measured from, one panel per participant, every condition overlaid in the bar colours - so the
    numbers can be read against the signal they came from.

    A hatch separates two conditions that share a colour in the BARS, but a line has no hatch:
    `linestyles` ({condition: "-", "--", ...}) is what separates them in the trace row. Without
    it, two conditions of the same colour are drawn on top of each other and cannot be told
    apart.

    with_p1 adds a first panel with the 1st pulse itself, in mV - the quantity those percentages
    are a percentage OF, so a ratio taken off a near-noise response is visible rather than hidden.
    Read it WITHIN a participant only (30 Hz against ARC-EX, same session, same electrodes);
    millivolts do not carry between people.

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
    colours = colours or dict(zip(protocols, PROTOCOL_COLOURS))
    hatch = {p_: _hatch(hatches, list(protocols), p_) for p_ in protocols}

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
                p1, rest = float(y[0]), float(np.nanmean(y[rest_from - 1:]))
            if not np.isfinite(p1) or p1 <= 0:
                continue
            onset = res["pulse_ms"]; ipi = res["ipi_ms"]
            keep = (t >= onset[0] - 20) & (t <= onset[-1] + ipi)
            val[(s, p)] = dict(amp=amp, snr=(p1 / b if b > 0 else np.nan),
                               p2=100 * float(y[1]) / p1, rest=100 * rest / p1, p1_mV=p1,
                               t=t[keep] - onset[0], y=sig[ch][w][keep])

    n_pul = kw.get("n_pulses", 10)
    panels = ([("p1_mV", "1st pulse", "mV  (within a subject only)", None)] if with_p1 else []) + \
             [("p2", "2nd pulse", "% of 1st pulse", 100),
              ("rest", f"mean of pulses {rest_from}-{n_pul}", "% of 1st pulse", 100)]
    rc = plt.rc_context(PAPER_RC); rc.__enter__()
    nrow = 2 if traces else 1
    fig = plt.figure(figsize=(max(5.4 * len(panels), 3.6 * len(subjects)),
                              4.6 + (3.0 if traces else 0)))
    gs = fig.add_gridspec(nrow, len(panels), height_ratios=[1, 0.62] if traces else [1],
                          hspace=0.55)
    axes, base_ax = [], None
    for k, (key, _, _, _) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k], sharey=base_ax if key != "p1_mV" else None)
        if key != "p1_mV" and base_ax is None:
            base_ax = ax
        axes.append(ax)
    x = np.arange(len(subjects)); w = 0.8 / len(protocols)
    for ax, (key, ttl, ylab, line) in zip(axes, panels):
        for j, p in enumerate(protocols):
            h = [val.get((s, p), {}).get(key, np.nan)
                 if (key == "p1_mV" or val.get((s, p), {}).get("snr", 0) >= min_snr_ratio)
                 else np.nan for s in subjects]
            ax.bar(x + (j - (len(protocols) - 1) / 2) * w, h, width=w * 0.9,
                   facecolor=colours[p], alpha=0.85, hatch=hatch[p],
                   edgecolor=("white" if hatch[p] else "none"), lw=0, zorder=2,
                   label=names.get(p, p) if ax is axes[0] else None)
            for xi, v, s_ in zip(x + (j - (len(protocols) - 1) / 2) * w, h, subjects):
                # a missing bar has two quite different causes and they were saying the same
                # thing: the response can be REJECTED for sitting too close to noise, which is
                # not the same as the pulse having no value at all
                if not np.isfinite(v) and (s_, p) in val:
                    weak_ = val[(s_, p)].get("snr", np.nan)
                    why = (f"< {min_snr_ratio:g}x noise" if np.isfinite(weak_)
                           and weak_ < min_snr_ratio else "pulse voided")
                    ax.annotate(why, (xi, 2), ha="center", va="bottom", fontsize=10,
                                color="0.45", style="italic", rotation=90)
        if line:
            ax.axhline(line, color="0.4", lw=0.9, ls=(0, (2, 3)), zorder=1)
        # names like "P05 · lidocaine" collide once there are more than a couple of them
        long_ = max((len(str(s_)) for s_ in subjects), default=0) * len(subjects) > 24
        ax.set_xticks(x, subjects, fontsize=12,
                      **(dict(rotation=20, ha="right") if long_ else {}))
        ax.set_title(ttl, fontsize=13, fontweight="bold")
        if ax is axes[0] or (with_p1 and ax is axes[1]):
            ax.set_ylabel(ylab, fontsize=11.5, color="0.25")
        ax.tick_params(colors="0.25")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=len(protocols), frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.995))
    weak = [f"{s} {names.get(p, p)}" for (s, p), v in sorted(val.items())
            if v["snr"] < min_snr_ratio]
    if weak:
        fig.text(0.5, -0.02, "blank = 1st pulse under "
                 f"{min_snr_ratio:g}x noise, too small to divide by  (" + ", ".join(weak) + ")",
                 ha="center", fontsize=9.5, color="#d62728")
    fig.suptitle(title or muscle, fontweight="bold", y=1.09)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    if traces:
        # one panel per participant: the trains the bars were measured from
        # one column per participant, not per bar panel: with more participants than panels the
        # last ones used to be dropped without a word
        tgs = gs[1, :].subgridspec(1, len(subjects), wspace=0.25)
        tax = [fig.add_subplot(tgs[0, k]) for k in range(len(subjects))]
        for ax, s_ in zip(tax, subjects):
            ys = []
            for p_ in protocols:
                v = val.get((s_, p_))
                if not v or "y" not in v:
                    continue
                m_ = v["t"] <= (trace_ms or v["t"].max())
                ls_ = (linestyles or {}).get(p_, "-") if isinstance(linestyles, dict) else "-"
                ax.plot(v["t"][m_], v["y"][m_], ls=ls_, color=colours[p_],
                        lw=1.3 if ls_ != "-" else 1.0, alpha=0.9)
                ys.append(v["y"][m_])
            ax.set_title(s_, fontweight="bold", pad=4)
            ax.set_xlabel("time (ms)", color="0.25")
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.set_yticks([]); ax.tick_params(colors="0.25")
            if not ys:
                continue
            # each participant on its own scale, with its own bar: millivolts do not compare
            # between people, and one shared scale flattens whoever responds least
            lo = float(np.nanmin(np.concatenate(ys))); hi = float(np.nanmax(np.concatenate(ys)))
            pad = (hi - lo) * 0.12
            ax.set_ylim(lo - pad, hi + pad)
            bar = float(f"{(hi - lo) / 3:.1g}") or 0.05
            x0 = ax.get_xlim()[0]
            ax.plot([x0, x0], [lo, lo + bar], color="0.25", lw=3, solid_capstyle="butt",
                    clip_on=False)
            ax.annotate(f"{bar:g} mV", (x0, lo + bar / 2), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=13, color="0.25")
    plt.show(); rc.__exit__(None, None, None)
    print(f"{muscle} - the 1st pulse each bar is a % OF:")
    for (s_, p_), v in sorted(val.items()):
        note = ("   <- 1st pulse too small, bars blank" if v["snr"] < min_snr_ratio else
                "   <- 2nd pulse voided by a dropout" if not np.isfinite(v["p2"]) else "")
        print(f"   {s_:5s} {names.get(p_, p_):12s} {v['amp']:5g} mA   {v['p1_mV']:.4f} mV "
              f"= {v['snr']:5.1f} x its noise" + note)
    return val


# ---------------------------------------------------------------------------
# 7. tendon vibration: every condition as a change from that session's baseline
# ---------------------------------------------------------------------------
def condition_table(runs, MT, muscles, conditions, base="Baseline", match="common",
                    group=None, **kw):
    """Per subject, muscle and condition: pulse-1 p2p at one intensity, and the same as a % of
    the SAME subject's baseline.

    `pct` compares the 1st pulses, `pct_all` the average of all N pulses - the first included -
    so a manipulation that leaves the first response alone but changes the rest of the train is
    not missed.

    Each reading also carries how its own train behaves: `dep2` (2nd pulse as % of the 1st) and
    `dep_rest` (mean of pulses 2..N as % of the 1st). Those are properties of the train itself,
    not comparisons with a control, so they are read per condition rather than as a change.

    Vibration is a within-session manipulation, so the only honest quantity is the change from
    that session's own baseline - between participants you then compare the *changes*, never the
    amplitudes. runs : {(subject, condition): csv}   MT : {subject: {muscle: mA}} from baseline.

    Every condition of a muscle must be read at ONE intensity, or the bars compare vibration
    with current. `match` says how that intensity is found:

      "common" (default) the lowest intensity that is >= the baseline threshold AND was
                         delivered in every block of that participant. Blocks were not always
                         run on the same ladder - P03's VIB blocks step 10 mA where its
                         baseline steps 5 - so the exact threshold can be missing from a block
                         where the muscle responds perfectly well. Moving up one step keeps the
                         comparison matched and keeps the muscle. The mA used is in the table.
      "exact"            the baseline threshold itself, and the muscle is dropped from any
                         block that never delivered it.

    `group` : {condition: group name}. Conditions in different groups are never compared and
    never share an intensity - 30 Hz and ARC-EX sit 70 mA apart, so "one intensity for every
    block" only makes sense inside a protocol. With groups, `MT` may be keyed by (subject,
    group) so each protocol brings its own thresholds, and `base` may be a
    {condition: its baseline condition} dict so each protocol is a % of its own control.
    """
    import warnings as _w
    from .io import load_run
    from .burst import burst_p2p, noise_p2p
    kw2 = {k: v for k, v in kw.items() if k not in ("min_snr", "max_edge_frac")}
    res_of, amps_of, chans_of, noise_of = {}, {}, {}, {}
    for (s, cond), csv in runs.items():
        meta, t, sig = load_run(csv)
        chans = [c for c in sig if c != "Trigger A"]
        r = burst_p2p(meta, t, sig, chans, min_snr=None, max_edge_frac=None, **kw2)
        res_of[(s, cond)], amps_of[(s, cond)], chans_of[(s, cond)] = r, list(r["amps"]), chans
        # the noise each reading has to beat, so a percentage built on a near-noise response can
        # be recognised as such later instead of being averaged in with the rest
        noise_of[(s, cond)] = noise_p2p(t, sig, chans,
                                        win_len_ms=r["win"][0][1] - r["win"][0][0])

    grp_of = group or {c: "" for c in conditions}
    base_of = base if isinstance(base, dict) else {c: base for c in conditions}
    out, missing, moved = {}, [], []
    subjects = sorted({s for s, _ in runs})
    for s in subjects:
      for g in dict.fromkeys(grp_of.get(c, "") for c in conditions):
        blocks = [(s, c) for c in conditions if grp_of.get(c, "") == g and (s, c) in runs]
        if not blocks:
            continue
        shared = set.intersection(*(set(amps_of[b]) for b in blocks))
        for m in muscles:
            th = (MT.get((s, g)) or MT.get(s) or {}).get(m)
            if not th:
                missing.append((s, g or "-", m, "no threshold in the control block")); continue
            if match == "exact":
                use = th if all(th in amps_of[b] for b in blocks) else None
            else:
                above = sorted(a for a in shared if a >= th)
                use = above[0] if above else None
            if use is None:
                have = ", ".join(f"{c}: {amps_of[(s, c)][0]:g}-{amps_of[(s, c)][-1]:g} step "
                                 f"{amps_of[(s, c)][1] - amps_of[(s, c)][0]:g}"
                                 for _, c in blocks)
                missing.append((s, g or "all blocks", m,
                                f"{th:g} mA threshold, but no intensity at or above it was "
                                f"delivered in every block ({have})")); continue
            if use != th:
                moved.append((f"{s} {g}".strip(), m, th, use))
            for _, cond in blocks:
                ch = next((c for c in chans_of[(s, cond)] if pretty(c) == m), None)
                if ch is None:
                    continue
                with _w.catch_warnings():
                    _w.simplefilter("ignore", RuntimeWarning)
                    y = np.asarray(res_of[(s, cond)]["p2p"][ch][amps_of[(s, cond)].index(use)],
                                   float)
                    nb = np.asarray(noise_of[(s, cond)][ch], float)
                    nb = float(nb[amps_of[(s, cond)].index(use)]) if nb.ndim else float(nb)
                    p1_, rest_ = float(y[0]), float(np.nanmean(y[1:]))
                    p2_ = float(y[1]) if len(y) > 1 else np.nan
                    all_ = float(np.nanmean(y))        # all N pulses, the 1st included
                    out[(s, cond, m)] = dict(
                        amp=use, p1=p1_, p2=p2_, rest=rest_, mean_all=all_,
                        snr=(p1_ / nb if nb > 0 else np.nan),
                        # how the train behaves after its own first pulse - nothing to do with
                        # the control condition, so these stand on their own
                        dep2=(100 * p2_ / p1_ if p1_ > 0 else np.nan),
                        dep_rest=(100 * rest_ / p1_ if p1_ > 0 else np.nan))
    for (s, cond, m), v in out.items():
        b = out.get((s, base_of.get(cond, base if isinstance(base, str) else cond), m))
        v["pct"] = 100 * v["p1"] / b["p1"] if b and np.isfinite(b["p1"]) and b["p1"] > 0 else np.nan
        # the same comparison on the whole train rather than its first pulse: the average of all
        # N pulses, as a % of the control's average. A manipulation can leave the first response
        # alone and still change what the train delivers overall.
        v["pct_all"] = (100 * v["mean_all"] / b["mean_all"]
                        if b and np.isfinite(b["mean_all"]) and b["mean_all"] > 0 else np.nan)
    if moved:
        print("read one step above threshold, so that every block has the same intensity:")
        for s_, m_, th_, use_ in moved:
            print(f"   {s_:16s} {m_:22s} threshold {th_:g} mA -> read at {use_:g} mA")
    if missing:
        print("not comparable, so left blank:")
        for s_, c_, m_, why in missing:
            print(f"   {s_}  {c_:18s} {m_:22s} {why}")
    return out


# kept so the vibration notebook's name still works: the table is not vibration-specific
vibration_table = condition_table


def _colour(colours, conditions, cond):
    """colours may be a list in condition order, or a {condition: colour} dict."""
    if isinstance(colours, dict):
        return colours[cond]
    return colours[conditions.index(cond) % len(colours)]


def _darker(c, f=0.42):
    """A darker shade of a colour, for marks that have to read against a bar of that colour."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(c)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def _hatch(hatches, conditions, cond):
    """Bar fill pattern per condition - the channel to use when colour already means something
    else. hatches may be a {condition: pattern} dict, a list in condition order, or None."""
    if not hatches:
        return ""
    if isinstance(hatches, dict):
        return hatches.get(cond, "")
    return hatches[conditions.index(cond) % len(hatches)]


def fig_vibration(tab, subjects, muscles, conditions, vibrated=None, site=None, colours=None,
                  hatches=None, key="pct", ylabel="% of baseline", ref=100, ylim=None,
                  titles=None, title=None, save=None):
    """One panel per participant: every muscle, a bar per condition, as % of that participant's
    own baseline. 100 % = unchanged by the manipulation.

    key : which value in each entry to plot - "pct" (the 1st pulse, the default), "pct_all"
          (the average of the whole train), or any other key condition_table stores.

    hatches : {condition: fill pattern} (e.g. "///") - the channel to carry a second factor
              when colour is already spoken for, such as polarity under a per-protocol colour.

    ylim : (low, high) to fix the axis. A bar past `high` is drawn up to the top with a caret
           and its real value printed above it, so one outlier cannot flatten every other bar
           into a stripe. Without it the axis follows the tallest bar, as usual.

    vibrated : {subject: muscle} - the muscle the vibrator was actually on, marked on the axis.
    site     : {subject: "left wrist extensor"} - where it was in words, for the panel title.
               The vibrator sat on a different muscle and a different side in each participant,
               so the comparison across people is of the CHANGE at the vibrated site, never of
               one participant's "VIB ON extensors" against another's.
    """
    colours = colours or CONDITION_COLOURS
    rc = plt.rc_context(PAPER_RC); rc.__enter__()
    fig, axes = plt.subplots(len(subjects), 1, figsize=(1.35 * len(muscles) + 3.5,
                                                        3.7 * len(subjects)), squeeze=False)
    axes = axes.ravel()
    x = np.arange(len(muscles)); w = 0.8 / len(conditions)
    for ax, s in zip(axes, subjects):
        for j, cond in enumerate(conditions):
            h = [tab.get((s, cond, m), {}).get(key, np.nan) for m in muscles]
            if not np.isfinite(h).any():
                continue
            col = _colour(colours, conditions, cond)
            xs = x + (j - (len(conditions) - 1) / 2) * w
            top = ylim[1] if ylim else None
            # clip an outlier to the top rather than let it flatten every other bar, and say
            # in the figure what it really is - a bar drawn to the ceiling is not a value
            shown = [min(v, top) if (top is not None and np.isfinite(v)) else v for v in h]
            hh = _hatch(hatches, conditions, cond)
            ax.bar(xs, shown, width=w * 0.9, color=col, alpha=0.9, zorder=2, hatch=hh,
                   edgecolor="white" if hh else "none", lw=0)
            if top is not None:
                for xi, v in zip(xs, h):
                    if np.isfinite(v) and v > top:
                        ax.plot([xi], [top], "^", ms=6, color=col, clip_on=False, zorder=4)
                        ax.annotate(f"{v:.0f}", (xi, top), xytext=(0, 7),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=10, color=col, fontweight="bold",
                                    rotation=90, annotation_clip=False, zorder=4)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if ref is not None:
            ax.axhline(ref, color="0.35", lw=1.0, ls=(0, (2, 3)), zorder=1)
        vib = (vibrated or {}).get(s)
        ax.set_xticks(x, [(m + "  *" if m == vib else m) for m in muscles], rotation=20,
                      ha="right")
        ax.set_ylabel(ylabel, color="0.25")
        where = (site or {}).get(s)
        ax.set_title((titles or {}).get(s, f"{s}" + (f"   vibrator on the {where}" if where else
                     (f"   vibrator on {vib}" if vib else ""))),
                     fontweight="bold", loc="left")
        ax.tick_params(colors="0.25")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    from matplotlib.patches import Patch
    drawn = [c_ for c_ in conditions if any((s_, c_, m_) in tab for s_ in subjects
                                            for m_ in muscles)]
    if not drawn:
        # a new session before any threshold is picked: say so rather than die on an empty legend
        for ax in axes:
            ax.annotate("nothing to draw yet — no thresholds picked for these muscles",
                        (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                        color="#C0392B", fontsize=12)
        if title:
            fig.suptitle(title, fontweight="bold", y=1.04)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show(); rc.__exit__(None, None, None)
        print("no bars: pick the thresholds in the last section, then re-run from section 2")
        return
    fig.legend([Patch(facecolor=_colour(colours, conditions, c_), alpha=0.9,
                      hatch=_hatch(hatches, conditions, c_),
                      edgecolor="white" if _hatch(hatches, conditions, c_) else "none")
                for c_ in drawn], drawn, loc="upper center", ncol=len(drawn), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
    plt.show(); rc.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# 8. double-check: is the trace being used really the threshold?
# ---------------------------------------------------------------------------
def check_thresholds(runs, MT, muscle, keys=None, xlim=(-20, 130), names=None, save_dir=None):
    """For one muscle, every recording of the comparison: all its intensities stacked, with the
    intensity the analysis uses drawn in orange.

    This is the check to run when a bar looks wrong. The orange trace should be the LOWEST one
    with a clear response - if the traces below it already respond, the threshold is too high; if
    the orange one is still noise, it is too low. Fix it in notebooks/motor_thresholds/ and re-run.

    runs : {key: csv}   MT : {key: {muscle: mA}}   keys : which to show, default all of runs.
    """
    from .plots import waterfall
    names = names or {}
    for k in (keys or list(runs)):
        csv = runs[k]
        th = (MT.get(k) or {}).get(muscle)
        meta, t, sig = load_run(csv)
        chans = resolve_muscles([c for c in sig if c != "Trigger A"], [muscle])
        lab = " · ".join(names.get(part, str(part)) for part in
                         (k if isinstance(k, tuple) else (k,)))
        if not th:
            print(f"{lab}: no threshold for {muscle} - nothing to check\n"); continue
        waterfall(meta, t, sig, chans, xlim=xlim, highlight=th,
                  highlight_label="the trace used", title=lab,
                  save=(f"{save_dir}/check_{lab.replace(' · ', '_')}.png" if save_dir else None))
        amps = sorted({m["amp_ma"] for m in meta})
        if th not in amps:
            step = amps[1] - amps[0] if len(amps) > 1 else 0
            print(f"{lab}: {th:g} mA is NOT in this recording "
                  f"({amps[0]:g}-{amps[-1]:g}, step {step:g}) - nothing to highlight, and this "
                  f"condition cannot be compared at that threshold")
            continue
        i = amps.index(th)
        print(f"{lab}: using {th:g} mA"
              + (f"  (the one below is {amps[i-1]:g} mA" if i else "  (lowest tested")
              + (f", above {amps[i+1]:g} mA)" if i + 1 < len(amps) else ", highest tested)"))


# ---------------------------------------------------------------------------
# 9. has a threshold been re-picked since the figures were built?
# ---------------------------------------------------------------------------
def mt_stamp(runs):
    """When each recording's saved thresholds were last written. Take this where MT is built."""
    import os
    from .threshold import mt_file
    return {k: (os.path.getmtime(mt_file(v)) if os.path.exists(mt_file(v)) else None)
            for k, v in runs.items()}


def warn_if_stale(runs, stamp, used=None):
    """Say so if a threshold file has been saved since `stamp` was taken.

    The figures read MT from memory, not from disk, so a pick saved after MT was built changes
    nothing until the threshold cell is re-run. `used` names the recordings the figures actually
    take thresholds FROM (e.g. only the baselines), so re-picking anything else is reported as
    having no effect rather than as something to re-run for.
    """
    now, newer = mt_stamp(runs), []
    for k, t in now.items():
        was = stamp.get(k)
        if t and (was is None or t > was + 0.5):
            newer.append(k)
    if not newer:
        return False
    matters = [k for k in newer if used is None or k in used]
    idle = [k for k in newer if k not in matters]
    if matters:
        print("!! saved since the thresholds were built, RE-RUN the threshold cell: "
              + ", ".join(str(k) for k in matters))
    if idle:
        print("   (also re-picked, but these figures do not take thresholds from them, so "
              "nothing changes: " + ", ".join(str(k) for k in idle) + ")")
    return bool(matters)


# ---------------------------------------------------------------------------
# 10. the first thing: every chosen threshold, all muscles, one row per participant
# ---------------------------------------------------------------------------
def fig_threshold_grid(recordings, muscles, xlim=(-20, 130), gain_frac=0.9, title=None,
                       save=None):
    """One row per recording, one column per muscle: every intensity stacked, with the threshold
    SAVED FOR THAT RECORDING drawn in orange.

    The thresholds are read from `results/<session>/mt_*.csv` directly - not from any MT built
    elsewhere - so this shows what you actually picked, and nothing can quietly substitute a
    different value.

    recordings : {row label: csv}. Returns {row label: {muscle: mA}} as read from disk.
    """
    from .threshold import mt_file, load_threshold_csv
    import os
    rows = list(recordings)
    fig, axes = plt.subplots(len(rows), len(muscles), squeeze=False,
                             figsize=(3.3 * len(muscles) + 1.2, 2.9 * len(rows)))
    out = {}
    with plt.rc_context(PAPER_RC):
        for r, lab in enumerate(rows):
            csv = recordings[lab]
            picked = load_threshold_csv(mt_file(csv)) if os.path.exists(mt_file(csv)) else {}
            out[lab] = picked
            meta, t, sig = load_run(csv)
            amps = np.array([m["amp_ma"] for m in meta])
            step = float(np.median(np.diff(np.unique(amps)))) if len(np.unique(amps)) > 1 else 10
            tmask = (t >= xlim[0]) & (t <= xlim[1])
            for c, m in enumerate(muscles):
                ax = axes[r][c]
                ch = next((x for x in sig if x != "Trigger A" and pretty(x) == m), None)
                th = picked.get(m)
                if ch is None:
                    ax.axis("off"); continue
                peak = np.percentile(np.abs(sig[ch][:, tmask]), 99.5)
                gain = (gain_frac * step) / peak if peak > 0 else 1.0
                for w in range(len(amps)):
                    on = th is not None and amps[w] == th
                    ax.plot(t[tmask], sig[ch][w, tmask] * gain + amps[w],
                            color="#C0392B" if on else "#9A9A9A",
                            lw=1.8 if on else 0.7, alpha=1.0 if on else 0.55,
                            zorder=4 if on else 2)
                ax.set_ylim(amps.min() - step, amps.max() + step)
                ax.set_xlim(*xlim)
                ax.set_yticks(np.unique(amps)[::max(1, len(np.unique(amps)) // 5)])
                ax.tick_params(labelsize=11, colors="0.3")
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if r == 0:
                    ax.set_title(m, fontsize=13, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{lab}\nmA", fontsize=12, color="0.2")
                if r == len(rows) - 1:
                    ax.set_xlabel("time (ms)", fontsize=12, color="0.25")
                ax.annotate("not picked" if th is None else f"{th:g} mA", (0.97, 0.03),
                            xycoords="axes fraction", ha="right", va="bottom", fontsize=12,
                            color="0.5" if th is None else "#C0392B", fontweight="bold")
        if title:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
        plt.show()
    for lab in rows:
        print(f"{lab:26s}" + "  ".join(f"{m.split(' (')[0][:9]}{m[-3:]} "
                                       f"{(f'{out[lab][m]:g}' if m in out[lab] else '--'):>4s}"
                                       for m in muscles))
    return out

# ---------------------------------------------------------------------------
# 11. how every muscle's peaks are detected, each at its own mA: one grid, laid
#     out like the waterfall grid above - a row per recording, a column per muscle
# ---------------------------------------------------------------------------
def _detection_panel(ax, t, sig, ch, res, raw, flags, w, th, note, xlim_ms, cmap):
    """Draw one muscle's 10 pulses re-aligned on their own onset, with the max/min taken."""
    onset, ipi = res["pulse_ms"], res["ipi_ms"]
    a_rel, b_rel = np.array(res["wins"][ch][0]) - onset[0]
    hi = xlim_ms or (b_rel + 2)
    ax.axvspan(a_rel, min(b_rel, hi), color="#2ca25f", alpha=0.10, zorder=0)
    at = res.get("anchor_t", {}).get(ch)
    if at is not None and np.isfinite(at[w]).all():
        for c_ in at[w]:
            ax.axvspan(c_ - res["anchor_win_ms"], c_ + res["anchor_win_ms"],
                       color="#C0392B", alpha=0.10, zorder=0)
    for k in range(len(onset)):
        seg = (t >= onset[k] - 2) & (t <= onset[k] + ipi)
        ax.plot(t[seg] - onset[k], sig[ch][w][seg], lw=0.9, alpha=0.85,
                color=cmap(k / max(len(onset) - 1, 1)), zorder=2)
    for key, mk in (("max", "v"), ("min", "^")):
        tt = raw[f"t{key}"][ch][w] - onset[:len(raw[f"t{key}"][ch][w])]
        yy = raw[f"y{key}"][ch][w]
        ax.plot(tt, yy, mk, ms=6, color="#333333", mec="white", mew=0.7, zorder=5)
        bad = np.asarray(flags[ch])[w][:len(tt)]
        if bad.any():
            ax.plot(tt[bad], yy[bad], mk, ms=10, mfc="none", mec="#C0392B", mew=1.6, zorder=6)
    ax.axvline(0, color="#C0392B", lw=1.2, alpha=0.8, zorder=1)
    ax.set_xlim(-2, hi)
    n_bad = int(np.sum(np.asarray(flags[ch])[w]))
    ax.annotate(f"{th:g} mA", (0.03, 0.03), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=11, color="0.35", fontweight="bold")
    ax.annotate(f"{n_bad} flagged" if n_bad else "clean", (0.97, 0.03),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=11,
                color="#C0392B" if n_bad else "#2ca25f", fontweight="bold")
    if note:
        ax.annotate(note, (0.5, 0.99), xycoords="axes fraction", ha="center", va="top",
                    fontsize=10, color="#C0392B")


def _blank_panel(ax, msg):
    ax.annotate(msg, (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
                color="0.5", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)


def fig_detection_grid(recordings, MT, muscles, edge_ms=1.0, jitter_ms=0.5, xlim_ms=None,
                       nearest=True, title=None, save=None, **kw):
    """ONE figure, laid out like fig_threshold_grid: a row per recording, a column per muscle,
    each panel that muscle's 10 pulses at ITS own threshold with the max (v) and min (^) the
    peak-to-peak is made of.

    recordings : {row label: csv}
    MT         : {row label: {muscle: mA}}, or one {muscle: mA} used for every row.

    Shaded green = the response window. Shaded red = the only places a max/min can be taken,
    +-`anchor_win_ms` around the train's own average latency; a peak outside those is skipped
    however clear it looks. Red ring = flagged (on a window border, or at a different latency).

    An intensity picked in one recording may not exist in another (the ladders differ between
    blocks); `nearest` then shows the closest intensity that was recorded, said so on the panel,
    rather than leaving it blank.
    """
    from .burst import burst_p2p, detection_flags
    rows = list(recordings)
    per_row = MT if rows and rows[0] in MT else {lab: MT for lab in rows}
    cmap = plt.get_cmap("viridis")
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(rows), len(muscles), squeeze=False,
                                 figsize=(3.3 * len(muscles) + 1.2, 2.9 * len(rows)))
        for r, lab in enumerate(rows):
            meta, t, sig = load_run(recordings[lab])
            chans = [c for c in sig if c != "Trigger A"]
            res = burst_p2p(meta, t, sig, chans, **kw)
            raw = burst_p2p(meta, t, sig, chans,
                            **{**kw, "min_snr": None, "max_edge_frac": None})
            flags, _ = detection_flags(res, chans, edge_ms, jitter_ms)
            amps = list(res["amps"])
            for c, m in enumerate(muscles):
                ax = axes[r][c]
                ch = next((x for x in chans if pretty(x) == m), None)
                th = per_row.get(lab, {}).get(m)
                # the threshold may come from another block, whose ladder can differ - say so
                # and show the nearest intensity this recording actually delivered
                note = None
                if ch is not None and th and th not in amps and nearest and amps:
                    got = min(amps, key=lambda a: abs(a - th))
                    note, th = f"{th:g} mA not recorded — showing {got:g}", got
                if ch is None or not th or th not in amps:
                    _blank_panel(ax, "no channel" if ch is None else
                                     "not picked" if not th else
                                     f"{th:g} mA not recorded here\n"
                                     f"{min(amps):g}–{max(amps):g} mA in this block")
                else:
                    _detection_panel(ax, t, sig, ch, res, raw, flags, amps.index(th), th, note,
                                     xlim_ms, cmap)
                    ax.tick_params(labelsize=11, colors="0.3")
                    for sp in ("top", "right"):
                        ax.spines[sp].set_visible(False)
                if r == 0:
                    ax.set_title(m, fontsize=13, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{lab}\nEMG (mV)", fontsize=12, color="0.2")
                if r == len(rows) - 1:
                    ax.set_xlabel("ms from pulse onset", fontsize=12, color="0.25")
        if title:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
        if save:
            import os
            os.makedirs(os.path.dirname(save), exist_ok=True)
            fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
        plt.show()


def fig_detection_row(csv, MT, muscles, edge_ms=1.0, jitter_ms=0.5, xlim_ms=None,
                      nearest=True, title=None, save=None, **kw):
    """One recording's muscles side by side - fig_detection_grid with a single row."""
    return fig_detection_grid({"": csv}, MT, muscles, edge_ms=edge_ms,
                              jitter_ms=jitter_ms, xlim_ms=xlim_ms, nearest=nearest,
                              title=title, save=save, **kw)


# ---------------------------------------------------------------------------
# 12. group summary: one number per muscle, averaged over participants
# ---------------------------------------------------------------------------
def fig_group_summary(tab, subjects, muscles, series, key="pct", base=None, colours=None,
                      hatches=None, min_snr_ratio=None, ylabel="% of control", ref=100,
                      ylim=None, labels=None, reference=None, err="sd", ax=None, legend=True,
                      title=None, save=None):
    """Mean +- SD over participants, per muscle and per series, with every participant's own
    value drawn on top of its bar.

    Built to be extended: add a participant to `subjects` and the bars, the error bars and the
    n underneath all follow. With two or three participants the SD is not worth much on its own,
    which is exactly why the individual points are always drawn - the bar is a summary of them,
    never a replacement.

    tab      : {(subject, condition, muscle): {key: value, "snr": ...}} from condition_table, or
               any dict of that shape.
    series   : the conditions to summarise, one bar each.
    base     : {series: its control condition} - a series whose control is itself is skipped,
               since it is 100 % by construction. None summarises every series given.
    reference : ("label", value) draws a flat bar of that height first in every muscle - the
               quantity everything else is a percentage OF, so the drop is read against something
               rather than against the top of the axis. It carries no error bar because it is
               that value by construction.

    err : "sd" (default) or "sem" - what the error bar shows. Say which in the caption.

    min_snr_ratio : drop a participant's value when EITHER the reading or the control it is
               divided by has a 1st pulse under this many times its own noise, and say so
               underneath. The denominator is where the damage is done. A percentage of a near-noise response
               is arithmetic, not physiology, and it would drag an average of two or three
               participants anywhere it liked.
    """
    colours = colours or PROTOCOL_COLOURS
    labels = labels or {}
    keep = [c for c in series if not (base and base.get(c) == c)]
    vals, dropped = {}, []
    for c in keep:
        for m in muscles:
            got = []
            for s in subjects:
                v = tab.get((s, c, m))
                if not v or not np.isfinite(v.get(key, np.nan)):
                    continue
                # a ratio is only as good as what it divides BY, so the control reading has
                # to clear the bar too - that is where the near-noise denominators hide
                snrs = [v.get("snr", np.nan)]
                ctrl = tab.get((s, base.get(c), m)) if base else None
                if ctrl:
                    snrs.append(ctrl.get("snr", np.nan))
                worst = min((r for r in snrs if np.isfinite(r)), default=np.nan)
                if min_snr_ratio is not None and np.isfinite(worst) and worst < min_snr_ratio:
                    dropped.append((s, c, m, worst)); continue
                got.append((s, float(v[key])))
            vals[(c, m)] = got

    nbar = len(keep) + (1 if reference else 0)
    own = ax is None                       # drawing our own figure, or into someone else's panel
    with plt.rc_context(PAPER_RC):
        if own:
            fig, ax = plt.subplots(figsize=(1.9 * len(muscles) + 3.0, 4.8))
        else:
            fig = ax.figure
        x = np.arange(len(muscles)); w = 0.8 / max(nbar, 1)
        if reference:
            rlab, rval = reference
            ax.bar(x - (nbar - 1) / 2 * w, [rval] * len(muscles), width=w * 0.88,
                   color="#D9D9D9", alpha=0.95, zorder=2)
        top = ylim[1] if ylim else None
        off = 1 if reference else 0
        for j, c in enumerate(keep):
            xs = x + (j + off - (nbar - 1) / 2) * w
            mu = [np.mean([v for _, v in vals[(c, m)]]) if vals[(c, m)] else np.nan
                  for m in muscles]
            sd = [(np.std([v for _, v in vals[(c, m)]], ddof=1)
                   / (np.sqrt(len(vals[(c, m)])) if err == "sem" else 1.0))
                  if len(vals[(c, m)]) > 1 else np.nan for m in muscles]
            col, hh = _colour(colours, keep, c), _hatch(hatches, keep, c)
            shown = [min(v, top) if (top is not None and np.isfinite(v)) else v for v in mu]
            ax.bar(xs, shown, width=w * 0.88, color=col, alpha=0.9, zorder=2, hatch=hh,
                   edgecolor="white" if hh else "none", lw=0)
            for xi, m_, mv, sv in zip(xs, muscles, mu, sd):
                if np.isfinite(mv) and np.isfinite(sv) and (top is None or mv <= top):
                    ax.errorbar(xi, mv, yerr=sv, fmt="none", ecolor=_darker(col, 0.25),
                                elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=4)
                if top is not None and np.isfinite(mv) and mv > top:
                    ax.plot([xi], [top], "^", ms=6, color=col, clip_on=False, zorder=5)
                    ax.annotate(f"{mv:.0f}", (xi, top), xytext=(0, 7), rotation=90,
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=10, color=col, fontweight="bold",
                                annotation_clip=False, zorder=5)
                # every participant on top of the bar: with a handful of people the points ARE
                # the result. They take a darker shade of the bar so it stays obvious which
                # series each one belongs to when the bars are close together.
                for k, (s_, v_) in enumerate(vals[(c, m_)]):
                    ax.plot(xi + (k - (len(vals[(c, m_)]) - 1) / 2) * w * 0.30,
                            min(v_, top) if top is not None else v_, "o", ms=5.5,
                            mfc=_darker(col), mec="white", mew=0.9, zorder=6)
        if ref is not None:
            ax.axhline(ref, color="0.35", lw=1.0, ls=(0, (2, 3)), zorder=1)
        if ylim:
            ax.set_ylim(*ylim)
        # n belongs on the bar, not under the muscle: the two series can lose different
        # participants to the noise filter, and one number for both would be wrong
        for j, c in enumerate(keep):
            col = _colour(colours, keep, c)
            for xi, m in zip(x + (j + off - (nbar - 1) / 2) * w, muscles):
                # white inside the bar: a dark label on a dark bar is unreadable, and these
                # bars are coloured by protocol so their shade is not ours to choose
                ax.annotate(f"n={len(vals[(c, m)])}", (xi, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=9, color="white", fontweight="bold", zorder=7)
        ax.set_xticks(x, muscles, rotation=20, ha="right")
        ax.set_ylabel(ylabel, color="0.25")
        ax.tick_params(colors="0.25")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        handles = ([Patch(facecolor="#D9D9D9", alpha=0.95)] if reference else []) + \
                  [Patch(facecolor=_colour(colours, keep, c), alpha=0.9,
                         hatch=_hatch(hatches, keep, c),
                         edgecolor="white" if _hatch(hatches, keep, c) else "none")
                   for c in keep]
        names_ = ([reference[0]] if reference else []) + [labels.get(c, c) for c in keep]
        marks = [Line2D([], [], marker="o", ls="none", ms=5.5, mfc="0.35", mec="white", mew=0.9)]
        if legend and own:
            fig.legend(handles + marks, names_ + ["one participant"],
                       loc="upper center", ncol=len(names_) + 1, frameon=False,
                       bbox_to_anchor=(0.5, 1.02))
        elif legend:
            ax.legend(handles + marks, names_ + ["one participant"], loc="lower left",
                      ncol=len(names_) + 1, frameon=False, fontsize=11,
                      bbox_to_anchor=(0.0, 1.0))
        if title and own:
            fig.suptitle(title, fontweight="bold", y=1.12)
        elif title:
            ax.set_title(title, fontweight="bold", pad=34, loc="left")
        if own:
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        if save and own:
            import os
            os.makedirs(os.path.dirname(save), exist_ok=True)
            fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
        if own:
            plt.show()
    if dropped:
        print(f"left out of the average - the reading or its control is under "
              f"{min_snr_ratio:g}x its own noise:")
        for s_, c_, m_, r_ in dropped:
            print(f"   {s_:16s} {c_:20s} {m_:22s} {r_:.1f} x")
    for c in keep:
        for m in muscles:
            got = vals[(c, m)]
            if got:
                print(f"{c:20s} {m:22s} n={len(got)}  "
                      + "  ".join(f"{s_} {v_:.0f}%" for s_, v_ in got)
                      + f"   mean {np.mean([v for _, v in got]):.0f}%"
                      + (f" +- {np.std([v for _, v in got], ddof=1):.0f}" if len(got) > 1 else ""))
    return vals


def fig_group_panels(tab, subjects, muscles, panels, key="dep_rest", colours=None, hatches=None,
                     min_snr_ratio=None, err="sd", reference=None, labels=None, ylabel="%",
                     ref=100, ylim=None, quiet=True, title=None, save=None):
    """Several fig_group_summary panels side by side in ONE figure, sharing a y axis.

    panels : {panel title: [conditions in that panel]} - e.g. one panel per protocol, each
             holding that protocol's before and with-lidocaine conditions, so the pair can be
             compared without reading across two separate figures.
    """
    out = {}
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(1.6 * len(muscles) * len(panels) + 3.0,
                                                          5.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (ttl, series) in zip(axes, panels.items()):
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
                out[ttl] = fig_group_summary(tab, subjects, muscles, series, key=key,
                                             colours=colours, hatches=hatches,
                                             min_snr_ratio=min_snr_ratio, err=err,
                                             reference=reference, labels=labels, ylabel=ylabel,
                                             ref=ref, ylim=ylim, ax=ax, title=ttl)
            if not quiet:
                print(buf.getvalue(), end="")
        for ax in axes[1:]:
            ax.set_ylabel("")
        if title:
            fig.suptitle(title, fontweight="bold", y=1.10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if save:
            import os
            os.makedirs(os.path.dirname(save), exist_ok=True)
            fig.savefig(save, dpi=300, bbox_inches="tight"); print("saved", save)
        plt.show()
    for ttl, vals in out.items():
        for (c, m), got in vals.items():
            if got:
                print(f"{ttl:10s} {c:22s} {m:22s} n={len(got)}  "
                      + "  ".join(f"{s_} {v_:.0f}%" for s_, v_ in got)
                      + f"   mean {np.mean([v for _, v in got]):.0f}%"
                      + (f" +- {np.std([v for _, v in got], ddof=1):.0f}" if len(got) > 1 else ""))
    return out
