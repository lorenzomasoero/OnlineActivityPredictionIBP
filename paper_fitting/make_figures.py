#!/usr/bin/env python3
"""Regenerate the data-experiment accuracy figures from the fitting-engine results
(OUTDIR/*_all_results.npy) using the package plotting. Light (plotting only).

For ASOS, TG-SSP is the geometric fit (TG_SSP_geom); elsewhere the curve fit (TG_SSP).
Writes PDFs into OUTDIR.
"""
import os, numpy as np
from activity_prediction.plotting import plot_accuracy_boxplot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR", os.path.join(HERE, "out"))

# canonical paper label -> source key per dataset
DATASETS = [
    ("uci_all_results.npy",            "UCI",            "TG_SSP"),
    ("rees46_all_results.npy",         "REES46 28-day",  "TG_SSP"),
    ("rees46_rolling_k21_results.npy", "REES46 k=21",    "TG_SSP"),
    ("asos_all_results.npy",           "ASOS",           "TG_SSP_geom"),
]

def collect(path, tgkey):
    a = np.load(path, allow_pickle=True); a = a.item() if a.dtype==object and a.shape==() else a
    recs = list(a.values()) if isinstance(a, dict) else list(a)
    data = {"TG-SSP": [], "IBP": [], "NB-SSP": []}
    keymap = {"TG-SSP": tgkey, "IBP": "IBP", "NB-SSP": "NB_SSP_regression"}
    for r in recs:
        if not isinstance(r, dict): continue
        for label, k in keymap.items():
            if k in r and isinstance(r[k], dict) and "accuracy_v" in r[k]:
                data[label].append(float(r[k]["accuracy_v"]))
    return {k: v for k, v in data.items() if v}

for fname, label, tgkey in DATASETS:
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        print(f"  skip {label}: {fname} missing"); continue
    data = collect(p, tgkey)
    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    outpdf = os.path.join(OUT, f"accuracy_{slug}.pdf")
    plot_accuracy_boxplot(data, outpdf, methods=["TG-SSP", "IBP", "NB-SSP"],
                          title=f"{label}: prediction accuracy")
    print(f"  wrote {outpdf}  (medians: " +
          ", ".join(f"{k}={np.median(v):.3f}" for k, v in data.items()) + ")")
