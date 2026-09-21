# CHUV tSCS EMG analysis

EMG responses to transcutaneous spinal cord stimulation (single pulse, 30 Hz bursts, ARC-EX),
before/after lidocaine, with tendon vibration, and with changed polarity.

```
src/functions/                 analysis code (loading, peak-to-peak, latency picker, figures)
notebooks/original_polarity/   P04 - lidocaine (17-07-2026), tendon vibration (16-07-2026)
notebooks/changed_polarity/    NTA - cathodic vs anodic, before/after lidocaine (24-07-2026)
results/                       picked latencies and peak-to-peak tables (CSV)
figures/                       saved figures
tools/strip_outputs.py         git filter that keeps notebook outputs out of the repo
tSCS_CHUV_data/                raw recordings - not tracked
```

Notebooks run from any folder (their first cell moves to the repo root). After cloning, run once:
`git config filter.nbstrip.clean "python3 tools/strip_outputs.py"`
