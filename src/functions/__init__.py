"""tSCS EMG analysis helpers (P04, 17-07-2026 lidocaine session)."""
from .style import set_style
from .labels import LABELS, pretty
from .io import load_run, detect_stim, detect_pulses
from .plots import waterfall, waterfall_overlay, plot_latency
from .latency import latency_picker, save_latency_csv, load_latency_csv
from .threshold import (threshold_picker, save_threshold_csv, load_threshold_csv, mt_file)
from .across import recruitment, fig_threshold_ratio, fig_recruitment, fig_selectivity
from .burst import (burst_p2p, normalize_burst, artifact_extent, clipped_channels,
                    plot_burst_windows, plot_burst_p2p, compare_burst_p2p, save_burst_csv,
                    first_vs_rest, plot_first_vs_rest, compare_first_vs_rest,
                    noise_p2p, responding_table, motor_threshold,
                    compare_at_intensity, diagnostics, train_metric, summary_heatmap, summary_curves,
                    detection_flags, detection_report, plot_pulse_overlay, resolve_muscles)

__all__ = [
    "set_style",
    "LABELS", "pretty",
    "load_run", "detect_stim", "detect_pulses",
    "waterfall", "waterfall_overlay", "plot_latency",
    "latency_picker", "save_latency_csv", "load_latency_csv",
    "threshold_picker", "save_threshold_csv", "load_threshold_csv", "mt_file",
    "recruitment", "fig_threshold_ratio", "fig_recruitment", "fig_selectivity",
    "burst_p2p", "normalize_burst", "artifact_extent", "clipped_channels",
    "plot_burst_windows", "plot_burst_p2p",
    "compare_burst_p2p", "save_burst_csv",
    "first_vs_rest", "plot_first_vs_rest", "compare_first_vs_rest",
    "noise_p2p", "responding_table", "motor_threshold", "compare_at_intensity", "diagnostics", "train_metric", "summary_heatmap", "summary_curves",
    "detection_flags", "detection_report", "plot_pulse_overlay",
]
