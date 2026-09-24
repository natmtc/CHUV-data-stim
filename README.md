# CHUV tSCS EMG analysis

EMG responses to transcutaneous spinal cord stimulation (single pulse, 30 Hz bursts, ARC-EX),
before/after lidocaine, with tendon vibration, and with changed polarity.

## Layout

```
src/functions/                 analysis code (loading, peak-to-peak, latency picker, figures)
notebooks/original_polarity/   P04 - lidocaine (17-07-2026), tendon vibration (16-07-2026)
notebooks/changed_polarity/    NTA - cathodic vs anodic, before/after lidocaine (24-07-2026)
results/                       picked latencies and peak-to-peak tables (CSV)
figures/                       saved figures
tools/strip_outputs.py         git filter that keeps notebook outputs out of the repo
tSCS_CHUV_data/                raw recordings - not tracked
```

## Setup

```bash
python3 -m venv .venv                          # Python 3.14
.venv/bin/pip install -r requirements.txt
git config filter.nbstrip.clean "python3 tools/strip_outputs.py"
```

The last line is required: notebook outputs are stripped from every commit, so the `.ipynb` files
in the repo hold code and text only. Run a notebook to see its figures. Without the filter
configured, your first commit puts every figure back into the repo.

Notebooks can be run from any folder - their first cell walks up to the repo root, so `src/`,
`results/` and `tSCS_CHUV_data/` always resolve the same way.

## The notebooks

Each **train** notebook (bursts, ARC-EX) follows the same sections: config, detection
diagnostics, reliability of the automatic peak-to-peak, waterfalls, every intensity, recruitment
curves. `tSCS_EMG_Burst_vs_ARCex.ipynb` is the comparison that produces the paper figures.

The **single-pulse** notebooks use a different pipeline: latencies are picked by hand once per
recording (section 2, run it once and comment the cells again) and stored in `results/`.

Two things are picked by hand, both with the same **A load / B pick / C save** blocks, both saved
per recording so they survive a re-run:

| picked | notebooks | saved to |
|---|---|---|
| response latency, per muscle x intensity | single pulse | `results/latency_<recording>.csv` |
| motor threshold, per muscle | `Burst_vs_ARCex` | `results/mt_<recording>.csv` |

The threshold picker is one cell per recording - `pick(("burst", "cathodic", "before"))`. It
stacks every intensity of one muscle as a waterfall; a slider sets the threshold to the lowest
trace that carries a real response, and **Save** writes the file. No `%matplotlib widget` needed,
so it works in VS Code too. Every muscle is then analysed **at** its threshold - the trace you
pick is the trace the figures are drawn from (`MT_STEPS` can step above it instead). Where a `mt_*.csv` exists it replaces the detected threshold
for that recording, so re-run the threshold cell after saving and every figure follows.

All train notebooks share the same peak-detection settings, set in each config cell:

| setting | value | meaning |
|---|---|---|
| `RESP_START_MS` / `GUARD_MS` | 8.0 / 1.0 | response window, after each pulse and before the next |
| `MIN_SNR` / `SNR_ON` | 1.2 / median | a train is a response if its median pulse clears 1.2x baseline |
| `ANCHOR` / `ANCHOR_WIN_MS` | mean / 3.0 | each pulse's max/min is taken near the train's own latency |
| `MAX_EDGE_FRAC` / `EDGE_MS` | 0.5 / 1.0 | reject a train whose peaks sit on the window borders |
| `JITTER_MS` | 0.5 | flag a pulse whose latency differs from the train median |

Change them in one notebook and the others no longer agree - keep them in step.

## Sharing outputs

Because outputs are stripped from the repo, send a rendered copy instead:

```bash
.venv/bin/jupyter nbconvert --to html --execute notebooks/changed_polarity/tSCS_EMG_Burst_vs_ARCex.ipynb
```

One self-contained HTML file with every figure embedded. Add `--no-input` to hide the code.
