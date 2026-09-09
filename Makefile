# Reproduction targets. See REPRODUCE.md for the full guide.
# Numbers come from the fitting engine (paper_fitting/, Lorenzo's implementation);
# simulation figures come from the sim runners (scripts/). Uses seed 0.

PY ?= python
OUTDIR ?= $(CURDIR)/paper_fitting/out
export OUTDIR

.PHONY: install test fit csv figures-data figures-sim proprietary repro clean

install:
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

# Data-experiment fits on Lorenzo's engine (seed 0) -> OUTDIR/*_all_results.npy
fit:
	SEED=0 $(PY) paper_fitting/fit_uci.py
	SEED=0 $(PY) paper_fitting/fit_rees46.py
	SEED=0 $(PY) paper_fitting/fit_rees46_rolling.py --k 21
	SEED=0 $(PY) paper_fitting/fit_asos.py

# Cross-dataset accuracy table -> paper_fitting/values/cross_dataset_accuracy.csv
csv:
	$(PY) paper_fitting/export_paper_csv.py

# Data-experiment accuracy figures from the fitting-engine results (needs `make fit`).
figures-data:
	$(PY) paper_fitting/make_figures.py

# Proprietary code path on a SYNTHETIC mock (real proprietary data is not shipped).
proprietary:
	$(PY) paper_fitting/make_mock_proprietary.py
	$(PY) paper_fitting/run_proprietary.py

# Simulation figures (DG1/DG2, zipf, nb-prediction, parameter est/sens, inversion).
# These do not involve the fitting divergence; model/interval comparison need Julia.
figures-sim:
	$(PY) scripts/run_zipf.py
	$(PY) scripts/run_nb_prediction.py
	$(PY) scripts/run_parameter_estimation.py
	$(PY) scripts/run_parameter_sensitivity.py
	$(PY) scripts/run_inversion.py

# End-to-end: regenerate the paper's numbers from committed data.
repro: fit csv figures-data
	@echo "Done: fits + cross_dataset_accuracy.csv + data-experiment figures regenerated."
	@echo "Simulation figures (optional, some need Julia): 'make figures-sim'."

clean:
	rm -rf $(CURDIR)/paper_fitting/out $(CURDIR)/output
