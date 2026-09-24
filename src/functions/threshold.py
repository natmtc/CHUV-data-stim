"""Click-to-pick motor threshold selector for train recordings + CSV export.

The automatic threshold (`paper.motor_thresholds`) is the lowest intensity whose train clears
the response criterion. It gets fooled by a smooth artifact ramp and by a single noisy sweep,
so this is the manual equivalent: the intensities of one muscle stacked as a waterfall, and you
click the first trace that carries a real response. Same A/B/C rhythm as `latency_picker`.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display

from .labels import pretty
from .io import detect_pulses


def threshold_picker(meta, t, sig, muscles, xlim=(-20, 130), picks=None, suggest=None,
                     gain_frac=0.9):
    """Click-to-pick motor threshold, one muscle at a time.

    Requires `%matplotlib widget`. Every intensity of the recording is drawn stacked at its own
    mA, exactly like `waterfall`. Click ON (or next to) the lowest trace that shows a response:
    that intensity becomes the muscle's motor threshold and is drawn in orange. `No response`
    leaves the muscle out, `Clear` undoes the pick.

    picks   : a previous session's dict, `picks[channel] = mA or nan`, to resume or correct.
    suggest : {muscle or label: mA} drawn as a green dotted line - the automatic threshold, so
              you can see where the detector put it before overriding it.

    Returns `picks`, mutated live as you click.
    """
    amps = np.array([m["amp_ma"] for m in meta])
    step = float(np.median(np.diff(np.unique(amps)))) if len(np.unique(amps)) > 1 else 10.0
    pulses = detect_pulses(t, sig["Trigger A"])
    t1 = pulses[0][1] if pulses else 0.0
    tmask = (t >= xlim[0]) & (t <= xlim[1])
    respmask = tmask & (t > t1)                     # scale on the response, not the artifact
    if not respmask.any():
        respmask = tmask

    if picks is None or set(picks) != set(muscles):
        picks = {m: np.nan for m in muscles}

    def _suggested(m):
        if not suggest:
            return None
        for k in (m, pretty(m)):
            if k in suggest:
                return suggest[k]
        return None

    state = {"mi": 0}

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    try:
        fig.canvas.header_visible = False
        fig.canvas.toolbar_position = "right"
    except Exception:
        pass

    def draw():
        m = muscles[state["mi"]]
        ax.clear()
        peak = np.percentile(np.abs(sig[m][:, respmask]), 99.5)
        gain = (gain_frac * step) / peak if peak > 0 else 1.0
        cur = picks[m]
        for w in range(len(amps)):
            on = cur == cur and amps[w] == cur
            ax.plot(t[tmask], sig[m][w, tmask] * gain + amps[w],
                    color="#e6550d" if on else "#1f3b73",
                    lw=2.2 if on else 0.9, alpha=1.0 if on else 0.65, zorder=5 if on else 2)
        s = _suggested(m)
        if s is not None:
            ax.axhline(s, color="#2ca25f", ls=":", lw=1.6, zorder=1)
            ax.annotate(f"detected {s} mA", (xlim[0], s), xytext=(4, 4),
                        textcoords="offset points", color="#2ca25f", fontsize=9,
                        fontweight="bold")
        for a0, a1 in pulses:
            if a1 >= xlim[0] and a0 <= xlim[1]:
                ax.axvspan(a0, a1, color="red", alpha=0.10, zorder=0)
                ax.axvline(a0, color="red", lw=1.0, alpha=0.8, zorder=1)
        ax.set_yticks(np.unique(amps))
        ax.set_ylim(amps.min() - step, amps.max() + step)
        ax.set_xlim(*xlim)
        got = "no response" if cur != cur else f"{cur:g} mA"
        ax.set_title(f"{pretty(m)}   ({state['mi'] + 1}/{len(muscles)})   "
                     f"motor threshold = {got}", fontweight="bold")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Stim amplitude (mA)")
        ax.grid(True, axis="x", alpha=0.25)
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass

    def onclick(event):
        if event.inaxes != ax or event.ydata is None:
            return
        picks[muscles[state["mi"]]] = float(np.unique(amps)[
            int(np.argmin(np.abs(np.unique(amps) - event.ydata)))])
        draw()

    fig.canvas.mpl_connect("button_press_event", onclick)

    mdrop = W.Dropdown(options=[(pretty(m), i) for i, m in enumerate(muscles)], value=0,
                       description="Muscle")
    b_prev = W.Button(description="◀ Prev"); b_next = W.Button(description="Next ▶")
    b_none = W.Button(description="No response", button_style="warning")
    b_take = W.Button(description="Take detected", button_style="info")

    def go(d):
        state["mi"] = int(np.clip(state["mi"] + d, 0, len(muscles) - 1))
        mdrop.value = state["mi"]                    # fires on_muscle, which redraws

    def set_none(_):
        picks[muscles[state["mi"]]] = np.nan
        draw()

    def take_detected(_):
        s = _suggested(muscles[state["mi"]])
        picks[muscles[state["mi"]]] = float(s) if s else np.nan
        draw()

    def on_muscle(ch):
        if ch["new"] is None:
            return
        state["mi"] = int(ch["new"]); draw()

    b_prev.on_click(lambda _: go(-1)); b_next.on_click(lambda _: go(+1))
    b_none.on_click(set_none); b_take.on_click(take_detected)
    mdrop.observe(on_muscle, names="value")

    display(W.HBox([mdrop, b_prev, b_next, b_take, b_none]))
    draw()
    return picks


def mt_file(csv_path, out_dir="results"):
    """Where the picks for one recording live: results/mt_<source-filename>.csv."""
    src = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(out_dir, f"mt_{src}.csv")


def save_threshold_csv(picks, muscles, csv_path, meta=None, out_dir="results", overwrite=False):
    """Write the picked thresholds to results/mt_<source-filename>.csv.

    The name carries the source file's timestamp, so polarities, protocols and lidocaine
    conditions never overwrite each other. A save is refused if the target already holds real
    thresholds and the picks in memory are all empty (pass overwrite=True to force), and if the
    picked intensities do not exist in `meta` - that means you picked on a different recording.
    Returns the path.
    """
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    out_csv = mt_file(csv_path, out_dir)

    if meta is not None:
        have = {m["amp_ma"] for m in meta}
        bad = {pretty(m): v for m, v in picks.items() if v == v and v not in have}
        if bad:
            raise ValueError(
                f"REFUSED: {bad} are not intensities of {os.path.basename(csv_path)} "
                f"({sorted(have)} mA). You picked on a different recording - re-run the load "
                f"cell for the file you want, then pick again.")

    new_has = any(v == v for v in picks.values())
    if os.path.exists(out_csv) and not overwrite:
        if pd.read_csv(out_csv)["mt_ma"].notna().any() and not new_has:
            print(f"⚠ NOT saved: {out_csv} already has thresholds and the current picks are "
                  f"all empty. Pass overwrite=True to force.")
            return out_csv

    rows = []
    for m in muscles:
        v = picks.get(m, np.nan)
        row = dict(source_file=os.path.splitext(os.path.basename(csv_path))[0],
                   muscle=pretty(m), channel=m, mt_ma=(v if v == v else np.nan))
        if meta is not None:
            row["electrode"], row["mode"] = meta[0]["electrode"], meta[0]["mode"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def load_threshold_csv(csv_path, by="muscle"):
    """Read a saved results/mt_*.csv back.

    by="muscle" -> {pretty label: mA}, what the analysis functions expect (muscles with no
    response are left out); by="channel" -> {channel: mA or nan}, what `threshold_picker`
    takes as `picks` to resume editing.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if by == "channel":
        return {c: (float(v) if pd.notna(v) else float("nan"))
                for c, v in zip(df["channel"], df["mt_ma"])}
    return {m: float(v) for m, v in zip(df["muscle"], df["mt_ma"]) if pd.notna(v)}
