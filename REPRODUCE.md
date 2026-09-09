# Reproducing the analyses

Lorenzo Masoero (masoerl) — last updated 2026-09-04

This guide explains how to install the code, run each analysis, and trace every
number and figure in the paper back to a committed script and CSV. It accompanies
the paper "Online activity prediction via generalized Indian buffet process models"
(Annals of Applied Statistics).

## 1. Requirements

- **Python 3.11** (the package requires >= 3.10; note some systems default to 3.9).
- Core dependencies (NumPy, SciPy, Matplotlib) install from `pyproject.toml`.
- Optional: Julia (+ `julia/Project.toml`) for the two simulation figures;
  CmdStanPy/ArviZ for HBG; CVXOPT for the LP/UnseenEST benchmark; pytest for tests.

## 2. Install (verified)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .                 # core: numpy, scipy, matplotlib
# optional extras:
pip install -e '.[hbg,unseen,test]'
# Julia env (only for the two simulation figures):
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

Verify the install:

```bash
python -m pytest -q                # 14 tests should pass
```

One-command reproduction of the paper's numbers (seed 0):

```bash
make repro          # runs the fitting engine on all data experiments -> CSVs
make figures-sim    # simulation figures (zipf, nb-prediction, param est/sens, inversion)
```

The reported accuracy numbers come from **`paper_fitting/`** (the fitting engine of
record); `make repro` writes `paper_fitting/out/*_all_results.npy` and
`paper_fitting/values/cross_dataset_accuracy.csv`.

## 3. Pipeline overview

```
raw data (downloaded)  --preprocessing/-->  prepared inputs (data/*.npy)
prepared inputs        --scripts/run_*.py-->  results (.npy) + CSV + figures (output/)
result CSVs            --reproducibility/gen_paper_values.py-->  latex/values/generated_values.tex
generated_values.tex   --\input-->  the manuscript (numbers/tables, no hand-typed values)
```

- `preprocessing/` turns raw public data into the prepared inputs under `data/` and
  emits the dataset-characteristics CSVs behind the §3 / appendix numbers.
- `scripts/run_*.py` fit models and evaluate; each writes a raw `.npy`, a tidy CSV of
  the paper-facing numbers, and its figure(s), all under `output/`.
- `reproducibility/gen_paper_values.py` turns the CSVs into LaTeX macros the paper
  `\input`s; rounding happens in LaTeX, not in the CSVs.

## 4. Run the analyses

```bash
# public-data experiments (use bundled prepared inputs by default)
python scripts/run_uci.py
python scripts/run_asos.py
python scripts/run_rees46.py
# hitting-time analysis needs the ASOS + REES46 k=100 fits first:
python scripts/run_rees46.py --experiments data/rees46/experiments_rolling_k100.npy \
                             --output-dir output/rees46_k100
python scripts/run_hitting_times.py
# simulations / illustrations
python scripts/run_zipf.py
python scripts/run_nb_prediction.py
python scripts/run_parameter_estimation.py
python scripts/run_parameter_sensitivity.py
python scripts/run_inversion.py
python scripts/run_model_comparison.py       # needs Julia
python scripts/run_interval_comparison.py     # needs Julia
```

Every script accepts `--help` for inputs, output directory, seeds, and settings.
See `README.md` for the full script -> figure map and `data/README.md` for input
provenance.

## 5. Determinism

The fitting runners take `--num-restarts` and `--seed`. To reproduce the exact
numbers in the paper, use the seeds/settings recorded in each script's header (and
in the raw output). Running without a fixed seed can shift the fitted accuracies.

## 6. Data availability

- **Public** (REES46, UCI, ASOS): small prepared inputs are bundled under `data/`;
  raw downloads + preprocessing are documented in `data/README.md` and
  `preprocessing/`.
- **Proprietary** (1,774-experiment dataset): **not reproducible** — the raw data
  cannot be released. See `data/proprietary/README.md`. We ship only a synthetic
  mock input (so `scripts/run_proprietary.py` exercises the code path) and the
  privacy-safe aggregate CSV behind the paper's proprietary numbers. This is a
  stated known limitation.

## 7. Numbers <-> source

`reproducibility/` holds the tooling that maps paper numbers to CSVs:
- `gen_rees46_experiment_counts.py` -> `rees46_experiment_counts.csv` (appendix rolling-window counts).
- `gen_paper_values.py` -> `latex/values/generated_values.tex` (macros used by the manuscript).
- `07_cross_dataset_figures.py` -> the cross-dataset accuracy table/figure.

See `REPRODUCIBILITY.md` for the standard and provenance convention.
