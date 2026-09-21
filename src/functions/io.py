"""Loading recruitment CSVs and detecting the stimulus artifact window."""
import csv
import ast
import numpy as np


def load_run(path):
    """Return (meta, time_ms, signals) for one recruitment CSV.
    meta: list of dicts (one per window); signals[muscle] -> 2D array [n_windows, n_samples].

    Columns are located BY NAME so it works across protocols (Burst/Modulated add extra
    parameter columns like frequency_hz / burst_duration_ms before time_ms)."""
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, rows = rows[0], rows[1:]
    ti = header.index("time_ms")                # everything after time_ms is a signal channel
    sig_cols = header[ti + 1:]

    def col(name):
        return header.index(name)

    meta = []
    for r in rows:
        d = dict(window=int(r[col("window_id")]), amp_ma=int(float(r[col("amplitude_ma")])),
                 electrode=r[col("electrode")], pw_us=int(float(r[col("pulse_width_us")])),
                 mode=r[col("mode")], trigger_time_s=float(r[col("trigger_time_s")]))
        for extra in ("frequency_hz", "burst_duration_ms"):   # present for Burst/Modulated
            if extra in header:
                d[extra] = r[col(extra)]
        meta.append(d)

    t = np.array(ast.literal_eval(rows[0][ti]))
    signals = {name: np.array([ast.literal_eval(r[ti + 1 + j]) for r in rows])
               for j, name in enumerate(sig_cols)}
    return meta, t, signals


def detect_pulses(t, trig, thr_frac=0.4, min_gap_ms=1.0):
    """Return [(start_ms, end_ms), ...] — one entry PER PULSE in the Trigger A channel.

    Single_Pulse gives one entry; Paired_Pulse two; Burst/Modulated one per pulse of
    the train (e.g. 30 Hz x 1000 ms -> 30 entries ~33 ms apart). Suprathreshold samples
    closer than `min_gap_ms` are merged so a ringing trigger is not split in two.
    """
    prof = np.abs(trig - np.median(trig, axis=1, keepdims=True)).mean(0)
    sup = prof > thr_frac * prof.max()
    if not sup.any():
        return []
    idx = np.where(sup)[0]
    # split the suprathreshold samples wherever the time gap exceeds min_gap_ms
    breaks = np.where(np.diff(t[idx]) > min_gap_ms)[0]
    groups = np.split(idx, breaks + 1)
    return [(float(t[g[0]]), float(t[g[-1]])) for g in groups]


def detect_stim(t, trig):
    """Artifact window of the FIRST stimulus pulse, as (start_ms, end_ms).

    For trains this is the first pulse only — NOT the whole train — so the response
    region after it stays usable. Use `detect_pulses` for every pulse.
    """
    pulses = detect_pulses(t, trig)
    return pulses[0] if pulses else (0.0, 0.0)
