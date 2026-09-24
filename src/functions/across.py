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
