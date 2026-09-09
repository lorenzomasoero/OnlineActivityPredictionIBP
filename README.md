# Online activity prediction via generalized Indian buffet process models — code companion

Reproduction code for the paper *"Online activity prediction via generalized Indian
buffet process models"* (Beraha, Favaro, Masoero; Annals of Applied Statistics).

The package fits and evaluates the paper's models — Be-SSP, TG-SSP, NB-SSP, the
three-parameter Indian buffet process (IBP), beta-binomial (BB), and hierarchical
beta-geometric (HBG) — on the public UCI, REES46, and ASOS datasets, and regenerates
the paper's tables and figures. Small prepared inputs are bundled; raw data is
downloaded/preprocessed, and the separate proprietary experiment is provided
results-only (see Data availability).

## What reproduces what

Every reported number comes from a committed CSV; every figure from a committed
script. The paper's cross-dataset accuracy numbers are produced by the **fitting
engine** in `paper_fitting/`; the simulation figures by the **simulation runners** in
`scripts/`.

| Paper artifact | How to regenerate | Output |
|---|---|---|
| Table `tab:cross_accuracy` (UCI/REES46/ASOS accuracy) | `make fit && make csv` | `paper_fitting/values/cross_dataset_accuracy.csv` |
| Data-experiment accuracy figures | `make figures-data` | `paper_fitting/out/accuracy_*.pdf` |
| Table `tab:ht_main` (hitting-time MAE) | `python scripts/run_hitting_times.py` | `results/hitting_time_mae_{asos,k100}.csv` |
| Appendix REES46 rolling-window counts (186/157/107) | `python reproducibility/gen_rees46_experiment_counts.py` | `results/rees46_experiment_counts.csv` |
| Appendix TG-SSP bimodality (REES46 28-day) | `bash paper_fitting/bimodality_sweep.sh` then `python paper_fitting/bimodality_report.py` | `paper_fitting/bimodality_tg.csv`, `paper_fitting/out/bimodality_hist.pdf` |
| Simulation figures (DG1/DG2, zipf, NB prediction, parameter est./sens., inversion) | `make figures-sim` | PDFs under `output/` |
| Trajectory / dataset-descriptive figures | `python scripts/run_uci.py`, `run_rees46.py` | PDFs under `output/` |
| Proprietary experiment (code path on synthetic mock) | `make proprietary` | mock results; real numbers in `data/proprietary/proprietary_results_summary.csv` |

Quick end-to-end (paper numbers + data-experiment figures, seed 0):

```bash
make repro
```

## Requirements & install

- **Python 3.11** (requires >= 3.10; some systems default to 3.9).
- Core: NumPy, SciPy, Matplotlib (from `pyproject.toml`).
- Optional: Julia (`julia/Project.toml`) for the model-/interval-comparison
  simulations; CmdStanPy + ArviZ for HBG; CVXOPT for the LP/UnseenEST benchmark.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .            # add '.[hbg,unseen,test]' for optional pieces
julia --project=julia -e 'using Pkg; Pkg.instantiate()'   # only for the two Julia figures
make test                   # 14 tests
```

See `REPRODUCE.md` for the step-by-step guide and `REPRODUCIBILITY.md` for the
provenance standard.

## Layout & the two model implementations

```text
paper_fitting/     fitting engine of record — reproduces the paper's accuracy numbers
                   (fit_*.py + utils/); export_paper_csv.py, make_figures.py
src/activity_prediction/   models, evaluation helpers, and plotting (used for the
                   simulation figures and all plotting)
scripts/           one runner per experiment (simulations + dataset descriptives)
preprocessing/     raw -> prepared inputs + dataset-characteristics stats
reproducibility/   CSV/count generators consumed by the paper
data/              prepared public inputs (see data/README.md); data/proprietary/ (mock + aggregates)
julia/  tests/  Makefile  pyproject.toml
```

Note: the paper's **cross-dataset accuracy** (Table `tab:cross_accuracy`) is produced
by `paper_fitting/` (the implementation used for the published results). The
`src/activity_prediction` model classes are an independent implementation used for the
plotting and simulation studies; they are not the source of the accuracy table.

## Data availability

- **Public** (UCI, REES46, ASOS): small prepared inputs are bundled under `data/`;
  raw downloads and preprocessing are documented in `data/README.md` and `preprocessing/`.
- **Proprietary** (1,774-experiment dataset): **cannot be released**; the experiment is
  **not reproducible end-to-end** (known limitation). We ship only the privacy-safe
  aggregate results (`data/proprietary/proprietary_results_summary.csv`) and a synthetic
  mock so `make proprietary` exercises the code path — its numbers are not the paper's.

## Reproducibility notes

- Fits use random restarts; pass `--seed` (or `SEED=`) for determinism. `make repro`
  uses seed 0. On the small REES46 28-day benchmark, TG-SSP accuracy is bimodal across
  seeds — the paper reports the typical (low) mode; see the bimodality appendix.
- Numbers are rounded in LaTeX, not in the CSVs.

## License

See `LICENSE`.
