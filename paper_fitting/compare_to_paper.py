#!/usr/bin/env python3
"""Compare the fitting-engine outputs (in OUTDIR, default ./out) to the paper's
tab:cross_accuracy. ASOS TG-SSP uses the geometric fit (TG_SSP_geom), as in the paper."""
import os, numpy as np

OUT = os.environ.get("OUTDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
PAPER = {
    "UCI":            {"TG": 0.71, "IBP": 0.80, "NB": 0.84},
    "REES46 28-day":  {"TG": 0.58, "IBP": 0.81, "NB": 0.90},
    "REES46 k=21":    {"TG": 0.81, "IBP": 0.81, "NB": 0.80},
    "ASOS":           {"TG": 0.71, "IBP": 0.16, "NB": 0.50},
}
SOURCES = {
    "UCI":           ("uci_all_results.npy",           "TG_SSP"),
    "REES46 28-day": ("rees46_all_results.npy",        "TG_SSP"),
    "REES46 k=21":   ("rees46_rolling_k21_results.npy","TG_SSP"),
    "ASOS":          ("asos_all_results.npy",          "TG_SSP_geom"),  # geometric TG for ASOS
}

def med(path, keys):
    a = np.load(path, allow_pickle=True)
    a = a.item() if a.dtype == object and a.shape == () else a
    recs = list(a.values()) if isinstance(a, dict) else list(a)
    out = {}
    for label, k in keys.items():
        vals = [float(r[k]["accuracy_v"]) for r in recs
                if isinstance(r, dict) and k in r and isinstance(r[k], dict) and "accuracy_v" in r[k]]
        out[label] = float(np.median(vals)) if vals else None
    return out

print(f"{'dataset':<16}{'model':<6}{'paper':>7}{'repro':>8}{'Δ':>8}")
for ds,(fname,tgkey) in SOURCES.items():
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        print(f"{ds}: MISSING {fname}"); continue
    m = med(p, {"TG": tgkey, "IBP": "IBP", "NB": "NB_SSP_regression"})
    for model in ("TG","IBP","NB"):
        pv, rv = PAPER[ds][model], m[model]
        d = f"{rv-pv:+.2f}" if rv is not None else "—"
        rvs = f"{rv:.3f}" if rv is not None else "—"
        print(f"{ds:<16}{model:<6}{pv:>7}{rvs:>8}{d:>8}")
