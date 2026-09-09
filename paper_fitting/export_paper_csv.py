#!/usr/bin/env python3
"""Export the paper's cross-dataset accuracy table from the fitting-engine outputs
(seed 0, in OUTDIR/ default ./out) to a committed CSV the manuscript consumes.

ASOS TG-SSP uses the geometric marginal fit (TG_SSP_geom); all other TG rows use
the curve fit (TG_SSP). NB uses NB_SSP_regression.
Writes: <repo>/results/cross_dataset_accuracy.csv
"""
import os, csv, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(HERE, "out"))
DEST = os.environ.get("DEST", os.path.join(HERE, "values", "cross_dataset_accuracy.csv"))

SOURCES = [
    ("UCI",            13,  "uci_all_results.npy",            "TG_SSP"),
    ("REES46 (28-day)", 7,  "rees46_all_results.npy",         "TG_SSP"),
    ("REES46 (k=21)", 186,  "rees46_rolling_k21_results.npy", "TG_SSP"),
    ("ASOS",          144,  "asos_all_results.npy",           "TG_SSP_geom"),
]

def recs(path):
    a = np.load(path, allow_pickle=True); a = a.item() if a.dtype==object and a.shape==() else a
    return list(a.values()) if isinstance(a, dict) else list(a)

def med(rs, key):
    v = [float(r[key]["accuracy_v"]) for r in rs
         if isinstance(r, dict) and key in r and isinstance(r[key], dict) and "accuracy_v" in r[key]]
    return (float(np.median(v)), float(np.mean(v)), len(v)) if v else (None, None, 0)

rows = []
for label, n_exp, fname, tgkey in SOURCES:
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        print(f"  skip {label}: {fname} missing"); continue
    rs = recs(p)
    for model, key in (("TG-SSP", tgkey), ("IBP", "IBP"), ("NB-SSP", "NB_SSP_regression")):
        m, mn, n = med(rs, key)
        if m is None: continue
        rows.append({"dataset": label, "model": model, "n": n,
                     "median_v": f"{m:.4f}", "mean_v": f"{mn:.4f}", "source_key": key})

os.makedirs(os.path.dirname(DEST), exist_ok=True)
with open(DEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset","model","n","median_v","mean_v","source_key"])
    w.writeheader(); w.writerows(rows)
print(f"Wrote {DEST} ({len(rows)} rows)")
for r in rows:
    print(f"  {r['dataset']:<16}{r['model']:<8} n={r['n']:<4} median={r['median_v']}")
