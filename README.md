# CHUV tSCS EMG analysis

EMG responses to transcutaneous spinal cord stimulation (single pulse, 30 Hz bursts, ARC-EX),
before/after lidocaine, with tendon vibration, and with changed polarity.

- `analysis/functions/` – analysis code
- `analysis/original_polarity/` – notebooks, P04 (lidocaine 17-07-2026, vibration 16-07-2026)
- `analysis/changed_polarity/` – notebooks, NTA (cathodic vs anodic, lidocaine, 24-07-2026)
- `analysis/results/` – picked latencies and peak-to-peak tables

Raw recordings and session logs are not tracked. Notebooks are saved without outputs; after
cloning, run once: `git config filter.nbstrip.clean "python3 tools/strip_outputs.py"`.
