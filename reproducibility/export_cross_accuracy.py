#!/usr/bin/env python3
"""
Export the cross-dataset accuracy table (paper Table `tab:cross_accuracy`) to a
tidy CSV from the committed experiment result files.

Reads   : output/<dataset>/<name>_results.npy produced by the run_* scripts.
Writes  : output/cross_dataset_accuracy.csv
          columns: dataset,model,n,median_v,mean_v

Each result file is a mapping experiment_id -> record; each record holds one dict
per model with an ``accuracy_v`` field. Model keys are normalised to the paper
names (e.g. "NB-SSP (curve)" -> "NB-SSP"). Run the relevant experiments first
(see REPRODUCE.md), then run this exporter.
"""
import argparse
import csv
import os
import numpy as np

# (dataset label in the paper table, path to its results .npy)
DEFAULT_SOURCES = [
    ("UCI", "output/uci/uci_results.npy"),
    ("REES46 (28-day)", "output/rees46/rees46_results.npy"),
    ("REES46 (k=21)", "output/rees46_k21/rees46_results.npy"),
    ("ASOS", "output/asos/asos_results.npy"),
]

MODEL_ALIASES = {
    "NB-SSP (curve)": "NB-SSP",
    "NB-SSP (reg)": "NB-SSP",
    "TG-SSP": "TG-SSP",
    "IBP": "IBP",
    "Be-SSP": "Be-SSP",
}


def _load_records(path):
    arr = np.load(path, allow_pickle=True)
    obj = arr.item() if arr.dtype == object and arr.shape == () else arr
    if isinstance(obj, dict):
        return list(obj.values())
    return list(obj)


def accuracies_by_model(records):
    """Collect accuracy_v across experiments, keyed by normalised model name."""
    acc = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for key, val in rec.items():
            if isinstance(val, dict) and "accuracy_v" in val:
                name = MODEL_ALIASES.get(key, key)
                acc.setdefault(name, []).append(float(val["accuracy_v"]))
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/cross_dataset_accuracy.csv")
    args = ap.parse_args()

    rows = []
    for label, path in DEFAULT_SOURCES:
        if not os.path.exists(path):
            print(f"  skip {label}: {path} not found (run its experiment first)")
            continue
        acc = accuracies_by_model(_load_records(path))
        for model, vals in acc.items():
            vals = np.asarray(vals, float)
            rows.append({
                "dataset": label,
                "model": model,
                "n": len(vals),
                "median_v": f"{np.median(vals):.4f}",
                "mean_v": f"{np.mean(vals):.4f}",
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "n", "median_v", "mean_v"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['dataset']:<18} {r['model']:<8} n={r['n']:<4} "
              f"median={r['median_v']} mean={r['mean_v']}")


if __name__ == "__main__":
    main()
