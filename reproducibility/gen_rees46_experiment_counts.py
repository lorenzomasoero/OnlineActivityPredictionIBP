#!/usr/bin/env python3
"""
Derive REES46 rolling-experiment counts from the committed processed artifacts
and emit a single auditable CSV that the paper consumes for its reported n.

Input : data/rees46_processed/experiments_rolling_k{21,50,100}.npy
Output: results/rees46_experiment_counts.csv
        columns: dataset,D0,k,window_days,D_total,n_experiments

The count identity is  n_experiments = D_total - window_days + 1,
with window_days = D0 + k. D_total is read back from the artifacts
(max window_end) so the CSV is fully determined by committed data --
no manual numbers. Re-run whenever the processed data changes.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # SubmissionAOAS/
PROC_DIR = os.path.join(ROOT, "data", "rees46_processed")
OUT_CSV = os.path.join(ROOT, "results", "rees46_experiment_counts.csv")

D0 = 7
K_VALUES = [21, 50, 100]

rows = []
for k in K_VALUES:
    exps = np.load(os.path.join(PROC_DIR, f"experiments_rolling_k{k}.npy"),
                   allow_pickle=True)
    n = len(exps)
    window_days = D0 + k
    d_total = max(e["window_end"] for e in exps)
    # sanity: the rolling identity must hold exactly
    assert n == d_total - window_days + 1, (
        f"k={k}: n={n} != D_total-W+1={d_total - window_days + 1}")
    rows.append({
        "dataset": "REES46",
        "D0": D0,
        "k": k,
        "window_days": window_days,
        "D_total": d_total,
        "n_experiments": n,
    })

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(df.to_string(index=False))
print(f"\nWrote {OUT_CSV}")
