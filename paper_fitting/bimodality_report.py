#!/usr/bin/env python3
"""Turn bimodality_tg.csv into appendix artifacts: a per-seed histogram figure and a
per-window summary CSV, plus printed tables. Light (no fitting)."""
import os, csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "bimodality_tg.csv"))))

by_seed = collections.defaultdict(list)
by_win = collections.defaultdict(list)
for r in rows:
    by_seed[int(r["seed"])].append(float(r["tg_accuracy_v"]))
    by_win[int(r["exp_id"])].append(float(r["tg_accuracy_v"]))

seed_meds = np.array([np.median(v) for v in by_seed.values()])
low = seed_meds[seed_meds < 0.52]; high = seed_meds[seed_meds >= 0.52]

# Histogram of per-seed medians
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(seed_meds, bins=np.arange(0.35, 0.65, 0.02), color="#377eb8", alpha=0.8, edgecolor="black")
ax.axvline(np.median(seed_meds), color="black", ls="--", lw=1, label=f"median {np.median(seed_meds):.2f}")
ax.set_xlabel("per-seed median accuracy $v$ (REES46 28-day, TG-SSP)")
ax.set_ylabel("# seeds (of 50)"); ax.legend()
fig.tight_layout()
fig_path = os.path.join(HERE, "out", "bimodality_hist.pdf")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
fig.savefig(fig_path)
print(f"wrote {fig_path}")

# Per-window summary CSV
win_csv = os.path.join(HERE, "values", "bimodality_per_window.csv")
os.makedirs(os.path.dirname(win_csv), exist_ok=True)
with open(win_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["window", "min", "median", "max", "frac_high_optimum"])
    for win in sorted(by_win):
        a = np.array(by_win[win]); w.writerow(
            [win, f"{a.min():.3f}", f"{np.median(a):.3f}", f"{a.max():.3f}", f"{np.mean(a>=0.7):.2f}"])
print(f"wrote {win_csv}")
print(f"\nlow mode n={len(low)} mean={low.mean():.3f} | high mode n={len(high)} mean={high.mean():.3f}")
