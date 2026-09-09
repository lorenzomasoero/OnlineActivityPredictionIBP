#!/usr/bin/env python3
"""
run_proprietary.py -- proprietary-experiment code path, on a SYNTHETIC MOCK input.

IMPORTANT: The paper's proprietary experiment uses a proprietary dataset of 1,774
experiments that CANNOT be released. This script does NOT use that data. It runs the
identical fitting/evaluation pipeline on `data/proprietary/mock_proprietary_data.npy`
(synthetic; generate with make_mock_proprietary.py) purely to demonstrate the code path.

The accuracy numbers produced here are on MOCK data and are meaningless as science.
The paper's reported proprietary numbers are the privacy-safe aggregates in
`data/proprietary/proprietary_results_summary.csv`; they are NOT reproducible from
this package (known limitation).
"""
import os, sys, time, csv, warnings
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
np.random.seed(int(os.environ.get("SEED", "0")))
from utils_GD import *
from utils_IBP import *
from utils_NBP import *
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK = os.path.join(HERE, "..", "data", "proprietary", "mock_proprietary_data.npy")
OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "out"))
os.makedirs(OUTDIR, exist_ok=True)
NUM_ITS, WIDTH = 5, 0.95

BANNER = "=" * 72
print(BANNER)
print("MOCK proprietary run -- synthetic data. Numbers are NOT the paper's results.")
print("Paper proprietary numbers: data/proprietary/proprietary_results_summary.csv")
print(BANNER)

if not os.path.exists(MOCK):
    sys.exit(f"Mock input missing: {MOCK}\nRun: python paper_fitting/make_mock_proprietary.py")

experiments = np.load(MOCK, allow_pickle=True)
all_results = {}
for exp in experiments:
    D0, D1 = exp["D0"], exp["D1"]
    N_pilot, U_true = exp["N_pilot"], exp["U_true"]
    counts_short = np.array(exp["cumulative_users"])[:D0 + 1].astype(int)
    r = {"exp_id": exp["exp_id"], "D0": D0, "D1": D1, "N_pilot": N_pilot, "U_true": U_true}
    try:
        gd = GD(); p = gd.regression(counts_short, NUM_ITS, 2, False)
        r["TG_SSP"] = {"U_hat": float(gd.mean(D0, D1, N_pilot, p)[-1])}
    except Exception as e:
        r["TG_SSP"] = {"error": str(e)}
    try:
        ibp = IBP(); p = ibp.regression(counts_short, NUM_ITS, 2, False)
        r["IBP"] = {"U_hat": float(ibp.mean(D0, D1, p)[-1])}
    except Exception as e:
        r["IBP"] = {"error": str(e)}
    try:
        nbp = NegBintSBSP(); p = nbp.fit_regression(D0=0, N0=0, observed_counts=counts_short[1:], num_its=NUM_ITS)
        r["NB_SSP_regression"] = {"U_hat": float(nbp.mean_number_new_users(D0, D1, N_pilot, p)[-1])}
    except Exception as e:
        r["NB_SSP_regression"] = {"error": str(e)}
    for m in ("TG_SSP", "IBP", "NB_SSP_regression"):
        if "U_hat" in r.get(m, {}):
            r[m]["accuracy_v"] = float(1 - min(abs(U_true - r[m]["U_hat"]) / max(U_true, 1), 1))
    all_results[int(exp["exp_id"])] = r

np.save(os.path.join(OUTDIR, "proprietary_mock_results.npy"), all_results)

# MOCK summary CSV (clearly labelled)
summ = os.path.join(OUTDIR, "proprietary_mock_summary.csv")
with open(summ, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "median_v_MOCK", "mean_v_MOCK", "n", "note"])
    for m, label in (("TG_SSP", "TG-SSP"), ("IBP", "IBP"), ("NB_SSP_regression", "NB-SSP")):
        vals = [rr[m]["accuracy_v"] for rr in all_results.values()
                if m in rr and "accuracy_v" in rr[m]]
        if vals:
            w.writerow([label, f"{np.median(vals):.4f}", f"{np.mean(vals):.4f}", len(vals),
                        "MOCK synthetic data -- not the paper's numbers"])
print(f"Wrote {summ} (MOCK). Paper numbers are in data/proprietary/proprietary_results_summary.csv.")
