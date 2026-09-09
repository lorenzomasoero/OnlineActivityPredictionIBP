# Proprietary experiment — data availability

The paper reports one experiment on a **proprietary dataset of 1,774 experiments
from a large technology company**. That raw data **cannot be released** and is
**not included** here, and neither is the code that ingests the raw proprietary
logs. This is a known limitation: the proprietary experiment is **not
reproducible end-to-end** from the files in this package.

What we provide instead, so the code path is still exercisable and inspectable:

- `mock_proprietary_data.npy` — a small **synthetic** input in the same format as
  the real proprietary input (daily trigger counts; pilot `D0=7`, follow-up
  `D1=21`). It contains no real user data. Running `scripts/run_proprietary.py`
  on it exercises the identical fitting/evaluation pipeline used in the paper and
  writes a CSV + PDF, clearly labelled as produced from mock data.
- `proprietary_results_summary.csv` — the privacy-safe aggregate numbers that
  appear in the paper (per-model accuracy), with no per-user or re-identifiable
  rows. The paper's proprietary table/figures are rendered from these aggregates.

The numbers printed in the manuscript come from the real proprietary data, not
from the mock input. Re-running on the mock input will produce different numbers.
