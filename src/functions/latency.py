"""Interactive click-to-pick latency selector + CSV export."""
import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display

from .labels import pretty
from .io import detect_stim, detect_pulses, result_path


def latency_picker(meta, t, sig, muscles, xlim=(-20, 80), resp_end=50.0, manual_peaks=None):
    """Click-to-pick latency selector (one trace at a time).

    Requires `%matplotlib widget` to be active in the notebook. Click on the trace
    where the response starts, or use the buttons (Set NaN / Prev / Next) and the
    Muscle dropdown. Returns `manual_peaks[muscle] = [{amp, t_peak}, ...]`, mutated
    live as you click; pass it back in to resume a previous session.
    """
    pulses = detect_pulses(t, sig["Trigger A"])
    t0, t1 = pulses[0] if pulses else (0.0, 0.0)
    art = t1                       # end of the FIRST pulse's artifact
    PRE = (t >= -90) & (t <= -10)
    amps_list = [m["amp_ma"] for m in meta]

    # start fresh unless a matching prior session is passed in (same muscles & #windows)
    matches = (manual_peaks is not None
               and set(manual_peaks) == set(muscles)
               and all(len(manual_peaks[m]) == len(amps_list) for m in muscles))
    if not matches:
        manual_peaks = {m: [dict(amp=a, t_peak=np.nan) for a in amps_list] for m in muscles}

    state = {"mi": 0, "wi": 0}

    def suggest(m, w):
        y = sig[m][w]; base, sd = y[PRE].mean(), y[PRE].std()
        win = np.where((t > art) & (t <= resp_end))[0]
        over = win[np.abs(y[win] - base) > 5 * sd]
        return float(t[over[0]]) if len(over) else np.nan

    fig, ax = plt.subplots(figsize=(9.5, 5))
    try:
        fig.canvas.header_visible = False
        fig.canvas.toolbar_position = "right"
    except Exception:
        pass

    def draw():
        m = muscles[state["mi"]]; w = state["wi"]; a = amps_list[w]
        ax.clear()
        mask = (t >= xlim[0]) & (t <= xlim[1])
        ax.plot(t[mask], sig[m][w, mask], color="#1f3b73", lw=1.6)
        for a0, a1 in pulses:      # every pulse of the train gets its own band
            if a1 >= xlim[0] and a0 <= xlim[1]:
                ax.axvspan(a0, a1, color="red", alpha=0.12)
                ax.axvline(a0, color="red", lw=1.2)
        s = suggest(m, w)
        if s == s: ax.axvline(s, color="green", ls=":", lw=1.3)
        cur = manual_peaks[m][w]["t_peak"]
        if cur == cur:
            ax.axvline(cur, color="#e6550d", lw=2)
            ax.plot(cur, np.interp(cur, t, sig[m][w]), "o", color="#e6550d", ms=9, zorder=5)
        lat = "NaN" if cur != cur else f"{cur:.1f} ms"
        ax.set_title(f"{pretty(m)}  |  {a} mA   ({w+1}/{len(amps_list)})   latency = {lat}",
                     fontweight="bold")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("EMG (a.u.)"); ax.grid(alpha=0.25)
        ax.set_xlim(*xlim)   # keep the window (artifact band must not widen it)
        # force a repaint — needed when the redraw is triggered by an ipywidgets
        # callback (dropdown / buttons) rather than by a canvas click event
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.xdata <= art:
            return
        manual_peaks[muscles[state["mi"]]][state["wi"]]["t_peak"] = float(event.xdata)
        if state["wi"] < len(amps_list) - 1:      # auto-advance after a pick
            state["wi"] += 1
        draw()

    fig.canvas.mpl_connect("button_press_event", onclick)

    mdrop = W.Dropdown(options=[(pretty(m), i) for i, m in enumerate(muscles)],
                       value=0, description="Muscle")
    b_prev = W.Button(description="◀ Prev"); b_next = W.Button(description="Next ▶")
    b_nan  = W.Button(description="Set NaN (no response)", button_style="warning")

    def go_prev(_):
        state["wi"] = max(0, state["wi"] - 1); draw()
    def go_next(_):
        state["wi"] = min(len(amps_list) - 1, state["wi"] + 1); draw()
    def set_nan(_):
        manual_peaks[muscles[state["mi"]]][state["wi"]]["t_peak"] = np.nan
        if state["wi"] < len(amps_list) - 1: state["wi"] += 1
        draw()
    def on_muscle(ch):
        if ch["new"] is None: return
        state["mi"] = int(ch["new"]); state["wi"] = 0; draw()

    b_prev.on_click(go_prev); b_next.on_click(go_next); b_nan.on_click(set_nan)
    mdrop.observe(on_muscle, names="value")

    display(W.HBox([mdrop, b_prev, b_next, b_nan]))
    draw()
    return manual_peaks


def save_latency_csv(manual_peaks, muscles, csv_path, meta=None, out_dir="results",
                     overwrite=False):
    """Write the picked latencies to results/latency_<source-filename>.csv.

    The output name matches the FULL source filename (which carries a unique
    timestamp), so different protocols / electrodes / lidocaine conditions never
    overwrite each other. Identifying info is also stored as columns.

    Safety: if the target already holds real (non-NaN) latencies and the data you
    are about to write is entirely empty, the save is refused (pass overwrite=True
    to force) so a fresh/empty session can never wipe good picks. Returns the path.
    """
    import pandas as pd
    src = os.path.splitext(os.path.basename(csv_path))[0]   # unique stem incl. timestamp
    out_csv = result_path(csv_path, "latency", out_dir)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if meta is not None:      # the picks must belong to THIS recording (same intensities)
        want = [m["amp_ma"] for m in meta]
        for m in muscles:
            have = [r["amp"] for r in manual_peaks[m]]
            if have != want:
                raise ValueError(
                    f"REFUSED: the picks in memory were made on a recording with intensities "
                    f"{have} mA, but {src} has {want} mA. You picked on a different file "
                    f"(meta/t/sig were overwritten?). Re-run cell A for the file you want, "
                    f"then pick again.")

    new_has = any(r["t_peak"] == r["t_peak"] for m in muscles for r in manual_peaks[m])
    if os.path.exists(out_csv) and not overwrite:
        old_has = pd.read_csv(out_csv)["latency_ms"].notna().any()
        if old_has and not new_has:
            print(f"⚠ NOT saved: {out_csv} already has real latencies and the current "
                  f"picks are all empty. Pass overwrite=True to force.")
            return out_csv

    rows = []
    for m in muscles:
        for i, r in enumerate(manual_peaks[m]):
            row = dict(source_file=src, muscle=pretty(m), channel=m,
                       amp_ma=r["amp"], latency_ms=r["t_peak"])
            if meta is not None:                # electrode & mode identify the recording
                row["electrode"] = meta[i]["electrode"]
                row["mode"] = meta[i]["mode"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def load_latency_csv(csv_path):
    """Read a saved results/latency_*.csv back into a manual_peaks dict so you can
    re-plot or keep editing: manual_peaks[channel] = [{amp, t_peak}, ...] by amplitude."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    manual_peaks = {}
    for ch, sub in df.groupby("channel", sort=False):
        sub = sub.sort_values("amp_ma")
        manual_peaks[ch] = [dict(amp=int(a), t_peak=(float(v) if pd.notna(v) else float("nan")))
                            for a, v in zip(sub["amp_ma"], sub["latency_ms"])]
    return manual_peaks
