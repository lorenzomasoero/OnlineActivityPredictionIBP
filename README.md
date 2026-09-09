# Activity Prediction: AoAS code

This directory contains the fitting, evaluation, and figure code for the
non-proprietary experiments in the AoAS paper. Small prepared inputs for ASOS,
UCI, and REES46 are included; raw data, fitted results, generated figures,
notebooks, and the separate proprietary experiment are not. Dataset runners
use the bundled inputs by default and every path can be overridden on the
command line.

The Python import package is `activity_prediction`. Its model classes use the
paper names: `BeSSP`, `TGSSP`, `NBSSP`, `IBP`, `BetaBinomial`, and
`HierarchicalBetaGeometric`.

## Requirements

- Python 3.10 or newer.
- The core dependencies installed from `pyproject.toml`: NumPy, SciPy, and
  Matplotlib.
- Julia and the environment in `julia/Project.toml` only for the model- and
  interval-comparison simulations.
- Optional: CmdStanPy, ArviZ, and a CmdStan installation for HBG; CVXOPT for
  the LP/UnseenEST benchmark; pytest for the test suite.

From this directory, install the core package with:

```bash
python -m pip install -e .
```

Install optional Python dependencies as needed:

```bash
python -m pip install -e '.[hbg,unseen,test]'
```

Prepare the Julia environment once before using its two wrappers:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## Repository structure

```text
AoAS_Code/
├── src/activity_prediction/   models, experiment helpers, and plotting
├── scripts/                   one Python entry point per experiment
├── data/                      prepared ASOS, UCI, and REES46 inputs
├── julia/                     Julia kernels called by Python wrappers
├── tests/                     focused model and evaluation tests
└── pyproject.toml             package metadata and dependencies
```

## Experiment scripts

Each script performs the experiment and writes both its raw result file(s) and
paper PDF(s). Run `python scripts/<name>.py --help` for its input schema and
configuration flags.

| Script | Experiment and paper output |
|---|---|
| `run_parameter_sensitivity.py` | Parameter sensitivity; `u_simu.pdf`. |
| `run_inversion.py` | Duration-band inversion illustration; `inversion_ci.pdf`. |
| `run_model_comparison.py` | Be-SSP versus TG-SSP simulation; calls Julia and writes `comparison_accuracy.pdf`. |
| `run_interval_comparison.py` | Posterior versus inversion intervals; calls Julia and writes `interval_comparison_length.pdf`. |
| `run_zipf.py` | Zipf-Poisson prediction study; `PAPER_zipf_accuracy.pdf` and `PAPER_zipf_sums.pdf`. |
| `run_parameter_estimation.py` | NB-SSP likelihood-versus-curve study; `PAPER_model_accuracy_sample_size.pdf` and `APPENDIX_log_like.pdf`. |
| `run_nb_prediction.py` | NB-SSP future-trigger illustration; `PAPER_prediction_sum_synthetic.pdf`. |
| `run_asos.py` | ASOS first-trigger fits; `PAPER_ASOS_accuracy_v2.pdf` and `fig_appendix_asos_selected.pdf`. |
| `run_uci.py` | UCI fits and trajectories; `fig_case_study_uci.pdf` and `fig_appendix_uci_accumulation.pdf`. |
| `run_rees46.py` | REES46 fits; three `fig_rees46_*.pdf` descriptives and `fig_appendix_rees46_accumulation.pdf`. |
| `run_hitting_times.py` | ASOS/REES46 duration-to-target analysis; `fig_hitting_time_mae_asos.pdf`. |

The data runners consume the prepared files under `data/`; their formats and
sources are summarized in [`data/README.md`](data/README.md). With the package
installed, the main dataset experiments can therefore be launched directly:

```bash
python scripts/run_asos.py
python scripts/run_uci.py
python scripts/run_rees46.py
```

The hitting-time experiment consumes fitted ASOS results and a REES46 fit over
the bundled rolling `k=100` windows. Produce those prerequisites, then run it:

```bash
python scripts/run_asos.py
python scripts/run_rees46.py \
  --experiments data/rees46/experiments_rolling_k100.npy \
  --output-dir output/rees46_k100
python scripts/run_hitting_times.py
```

All generated `.npy` results and PDFs go below `output/` by default. Use
`python scripts/<name>.py --help` to override inputs, output directories, or
experiment settings.

The Julia wrappers default to the bundled entry points and accept `--julia`
when the executable is not named `julia`. The Python synthetic runners expose
seed and simulation-size controls; their defaults follow the paper settings.
Where an executable historical driver was absent, the script documents its
paper-based reconstruction and records the relevant numerical assumptions in
the raw output.

## Tests

```bash
python -m pytest
```

## License

See [`LICENSE`](LICENSE) for the combined notices retained from the public
source trees.
