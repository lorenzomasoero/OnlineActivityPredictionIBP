# Online Activity Prediction via Generalized Indian Buffet Process Models

Code companion for predicting future users' online activity using Bayesian nonparametric models based on generalized Indian buffet processes.

## Paper

> Masoero, L., Camerlenghi, F., Favaro, S., and Broderick, T. (2026).
> "Online activity prediction via generalized Indian buffet process models."
> *Annals of Applied Statistics* (under review).

## Installation

```bash
git clone https://github.com/your-username/OnlineActivityPredictionIBP.git
cd OnlineActivityPredictionIBP
pip install -r requirements.txt
```

**Optional dependencies** (uncomment in `requirements.txt` if needed):
- `cmdstanpy`, `arviz` — for the beta-geometric model
- `cvxopt` — for the unseen species LP estimator
- `jupyter` — for running notebooks

## Quick Start

**Load pre-fitted results and inspect:**

```python
import numpy as np
results = np.load('results/uci/uci_all_results.npy', allow_pickle=True).item()
exp = results[6]
print(f"NB-SSP accuracy: {exp['NB_SSP_regression']['accuracy_v']:.3f}")
```

**Fit TG-SSP on a single experiment:**

```python
from models.tg_ssp import GD
gd = GD()
params = gd.regression(counts_short, num_its=5, norm=2, status=False)
prediction = gd.mean(D0=7, M=21, K=N_pilot, parameters=params)
```

**Generate the decision figure:**

```bash
python plotting/decision_figure.py
```

## Repository Structure

```
OnlineActivityPredictionIBP/
├── models/              # BNP model implementations
│   ├── tg_ssp.py       # TG-SSP (GD class)
│   ├── nb_ssp.py       # NB-SSP (NegBintSBSP class)
│   ├── ibp.py          # IBP
│   └── ...             # Competitors (BB, Jackknife, Good-Toulmin)
├── experiments/         # Fitting and evaluation scripts
│   ├── preprocessing/   # Data preprocessing
│   ├── fitting/         # Model fitting (UCI, REES46, ASOS)
│   └── evaluation/      # D_M computation, hitting-time MAE
├── plotting/            # Figure generation scripts
├── notebooks/           # Jupyter notebooks for exploration
├── data/                # Preprocessed data + download instructions
│   ├── preprocessed/    # Small .npy files (committed)
│   └── raw/             # Raw data (gitignored, user downloads)
├── results/             # Pre-fitted results (committed)
└── output/              # Generated figures
```

## Datasets

This repository uses three datasets:

- **UCI Online Retail II** — Transaction records from a UK online retailer (2009–2011). 13 experiment windows with D0=7 pilot days and D1=21 follow-up days.
- **REES46** — eCommerce behavior data from a multi-category store (Oct 2019 – Apr 2020). Rolling-window experiments with k=21, 50, 100 day horizons.
- **ASOS** — Treatment arm first-trigger counts from an A/B test.

Preprocessed data is included in `data/preprocessed/`. For raw data download instructions, see [`data/README.md`](data/README.md).

## Reproducing Paper Figures

All figures can be reproduced from the committed results without re-fitting:

```bash
# Case study trajectories (UCI, REES46, ASOS)
python plotting/case_studies.py

# Hitting-time decision figure
python plotting/decision_figure.py

# Cross-dataset comparison
python plotting/cross_dataset.py

# Power-law visualizations
python plotting/powerlaw_viz.py

# CI coverage analysis
python plotting/ci_coverage.py

# Accuracy figures
python plotting/accuracy_figures.py
```

Figures are saved to `output/`.

To re-fit models from scratch (requires downloading raw data first):

```bash
# Preprocess
python experiments/preprocessing/preprocess_uci.py
python experiments/preprocessing/preprocess_rees46_rolling.py

# Fit
python experiments/fitting/fit_uci.py
python experiments/fitting/fit_rees46.py
python experiments/fitting/fit_asos.py

# Evaluate
python experiments/evaluation/hitting_time_mae.py
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
