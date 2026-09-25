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
from IPython.display import display, clear_output

from .labels import pretty
from .io import detect_pulses

# Widgets and figures live in the kernel, not in the cell, so re-running the pick cell builds a
# second set and the frontend shows both. Each call closes the one before it - and the handle on
# it is kept in the IPython namespace, NOT at module level, because `%autoreload` re-imports this
# module whenever it changes and would throw a module-level handle away.
def _open(key):
    """The widgets a previous call put on screen FOR THIS RECORDING. Keyed, so the eight pick
    cells each own their own live picker - closing one must not kill the others."""
    try:
        from IPython import get_ipython
        ns = get_ipython().user_ns
    except Exception:
        ns = globals()
    reg = ns.setdefault("_tscs_picker_open", {})
    return reg.setdefault(str(key), {"widgets": [], "fig": None})


def _close_previous(key):
    reg = _open(key)
    for w in reg["widgets"]:
        try:
            w.close()
        except Exception:
            pass
    reg["widgets"] = []
    if reg["fig"] is not None:
        try:
            plt.close(reg["fig"])
        except Exception:
            pass
        reg["fig"] = None


def threshold_picker(meta, t, sig, muscles, xlim=(-20, 130), picks=None, suggest=None,
                     gain_frac=0.9, mode="auto", on_save=None, key=None):
    """Pick the motor threshold, one muscle at a time.

    Every intensity of the recording is drawn stacked at its own mA, exactly like `waterfall`.
    Choose the lowest trace that carries a real response: it turns orange and becomes that
    muscle's threshold. `Take detected` accepts the automatic value, `No response` leaves the
    muscle out of the analysis.

    mode : "buttons" needs nothing but ipywidgets - the intensity is chosen from a slider, and
           the figure is redrawn inline. This is the one that works everywhere, VS Code included.
           "click" additionally lets you click the trace itself, but needs `%matplotlib widget`
           (ipympl) to be active AND rendering. "auto" (default) uses "click" when the ipympl
           backend is live and falls back to "buttons" otherwise.
    key     : identifies this picker, so re-running its cell replaces it and leaves the other
              recordings' pickers alone. Pass the recording's path.
    on_save : called with `picks` by a Save button next to the controls, so one cell is the
              whole job - open, pick, save. Whatever it returns is shown beside the button.
    picks   : a previous session's dict, `picks[channel] = mA or nan`, to resume or correct.
    suggest : {muscle or label: mA} drawn as a green dotted line - the automatic threshold, so
              you can see where the detector put it before overriding it.

    Returns `picks`, mutated live as you pick.
    """
    import matplotlib
    live = ("ipympl" in matplotlib.get_backend().lower()) if mode == "auto" else (mode == "click")
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

    state = {"mi": 0, "mute": False}
    steps = np.unique(amps)

    def _plot(ax, m):
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

    # ---- the two ways of getting that figure on screen ---------------------------------
    _close_previous(key)
    # An Image widget, not an Output: the picture is the widget's VALUE, so a redraw swaps it
    # in place. Output + display() appends, and clear_output has to race the new output - that
    # is what stacked a second picker under the first on every re-run.
    img = W.Image(format="png", layout=W.Layout(width="950px"))
    fig = ax = None
    if live:                       # ipympl: one canvas, redrawn in place, clickable
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        _open(key)["fig"] = fig
        try:
            fig.canvas.header_visible = False
            fig.canvas.toolbar_position = "right"
        except Exception:
            pass

        def onclick(event):
            if event.inaxes != ax or event.ydata is None:
                return
            _set(float(steps[int(np.argmin(np.abs(steps - event.ydata)))]))

        fig.canvas.mpl_connect("button_press_event", onclick)

    def draw():
        m = muscles[state["mi"]]
        if live:
            ax.clear(); _plot(ax, m)
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
        else:
            # A bare Figure, never handed to pyplot: pyplot figures belong to the inline
            # backend, which draws them again at the end of the cell.
            import io
            from matplotlib.figure import Figure
            f = Figure(figsize=(9.5, 6.5), layout="constrained")
            _plot(f.add_subplot(111), m)
            buf = io.BytesIO()
            f.savefig(buf, format="png", dpi=100)
            img.value = buf.getvalue()          # swap the picture, add nothing
        cur = picks[m]
        state["mute"] = True                         # move the slider without re-firing it
        sl.value = float(cur) if cur == cur else float(steps[0])
        state["mute"] = False

    # ---- controls (these work with or without ipympl) ----------------------------------
    mdrop = W.Dropdown(options=[(pretty(m), i) for i, m in enumerate(muscles)], value=0,
                       description="Muscle")
    sl = W.SelectionSlider(options=[(f"{a:g} mA", float(a)) for a in steps], value=float(steps[0]),
                           description="threshold", continuous_update=False,
                           style={"description_width": "initial"},
                           layout=W.Layout(width="520px"))
    b_prev = W.Button(description="◀ Prev"); b_next = W.Button(description="Next ▶")
    b_none = W.Button(description="No response", button_style="warning")
    b_take = W.Button(description="Take detected", button_style="info")
    b_save = W.Button(description="Save", button_style="success", icon="save")
    status = W.HTML("")

    def _set(v):
        picks[muscles[state["mi"]]] = v
        draw()

    def do_save(_):
        n = sum(1 for v in picks.values() if v == v)
        try:
            where = on_save(picks)
            status.value = (f"<span style='color:#2ca25f'><b>saved</b> {n} of {len(muscles)} "
                            f"muscles &rarr; {where}</span>")
        except Exception as e:                       # a refused save must not kill the picker
            status.value = f"<span style='color:#d62728'><b>not saved</b> - {e}</span>"

    def on_slider(ch):
        if state["mute"] or ch["new"] is None:
            return
        _set(float(ch["new"]))

    def go(d):
        state["mi"] = int(np.clip(state["mi"] + d, 0, len(muscles) - 1))
        mdrop.value = state["mi"]                    # fires on_muscle, which redraws

    def on_muscle(ch):
        if ch["new"] is None:
            return
        state["mi"] = int(ch["new"]); draw()

    b_prev.on_click(lambda _: go(-1)); b_next.on_click(lambda _: go(+1))
    b_none.on_click(lambda _: _set(np.nan))
    b_save.on_click(do_save)
    b_take.on_click(lambda _: _set(float(_suggested(muscles[state["mi"]]) or np.nan)))
    sl.observe(on_slider, names="value")
    mdrop.observe(on_muscle, names="value")

    row = [mdrop, b_prev, b_next, b_take, b_none] + ([b_save] if on_save else [])
    panel = W.VBox([W.HBox(row), sl] + ([] if live else [img])
                   + ([W.HBox([status])] if on_save else []))
    _open(key)["widgets"] = [panel, img, sl, mdrop, b_prev, b_next, b_take, b_none,
                             b_save, status]
    clear_output(wait=True)      # drop whatever this cell showed on its last run
    display(panel)
    draw()
    return picks


def mt_file(csv_path, out_dir="results"):
    """Where the picks for one recording live: results/mt_<source-filename>.csv."""
    from .io import result_path
    return result_path(csv_path, "mt", out_dir)


def save_threshold_csv(picks, muscles, csv_path, meta=None, out_dir="results", overwrite=False):
    """Write the picked thresholds to results/mt_<source-filename>.csv.

    The name carries the source file's timestamp, so polarities, protocols and lidocaine
    conditions never overwrite each other. A save is refused if the target already holds real
    thresholds and the picks in memory are all empty (pass overwrite=True to force), and if the
    picked intensities do not exist in `meta` - that means you picked on a different recording.
    Returns the path.
    """
    import pandas as pd
    out_csv = mt_file(csv_path, out_dir)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if meta is not None:
        have = {m["amp_ma"] for m in meta}
        bad = {pretty(m): v for m, v in picks.items() if v == v and v not in have}
        if bad:
            raise ValueError(
                f"REFUSED: {bad} are not intensities of {os.path.basename(csv_path)} "
                f"({sorted(have)} mA). You picked on a different recording - re-run the load "
                f"cell for the file you want, then pick again.")

    new_has = any(v == v for v in picks.values())
    if not new_has and not overwrite:
        # an all-empty file would replace the detected thresholds with nothing, so a stray
        # run of the save cell must not create one
        print(f"⚠ NOT saved: nothing is picked yet. Pick in B first "
              f"(pass overwrite=True if you really mean 'no muscle responds here').")
        return out_csv
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
