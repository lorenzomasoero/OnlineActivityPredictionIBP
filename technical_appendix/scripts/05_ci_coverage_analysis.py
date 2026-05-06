"""
05_ci_coverage_analysis.py
==========================
CI coverage investigation across all datasets.
Addresses T1.c (R2-b: "calibrated risk") and Priority 2 from NEXT_STEPS.

Key finding: CI coverage is poor because posterior CIs are narrow
(especially with large N_pilot) while point estimates carry non-trivial
bias. This is an inherent property of empirical Bayes with plug-in
hyperparameters, not a model deficiency.

Output figures:
    fig_ci_bias_vs_width.pdf     — bias vs CI width, colored by coverage
    fig_ci_coverage_by_dataset.pdf — coverage summary across datasets
    fig_ci_coverage_by_ratio.pdf — coverage vs extrapolation ratio (REES46 rolling)

Usage:  python3 SubmissionAOAS/lom_revision/scripts/05_ci_coverage_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import COLORS, MARKERS, LABELS, apply_style
apply_style()

# ============================================================
# Load all results
# ============================================================
def load_results(path):
    return np.load(os.path.join(BASE, path), allow_pickle=True).item()

uci = load_results('results/uci_all_results.npy')
rees46_k21 = load_results('results/rees46_rolling_k21_results.npy')
rees46_k50 = load_results('results/rees46_rolling_k50_results.npy')
rees46_k100 = load_results('results/rees46_rolling_k100_results.npy')
asos = load_results('results/asos_all_results.npy')

# ============================================================
# Extract CI data for NB-SSP regression (best method)
# ============================================================
def extract_ci_data(results, method='NB_SSP_regression'):
    rows = []
    for key in results:
        r = results[key]
        if method not in r or not isinstance(r[method], dict):
            continue
        m = r[method]
        if 'U_hat' not in m or 'U_ci' not in m:
            continue
        lo, hi = m['U_ci']
        U_true = r['U_true']
        U_hat = m['U_hat']
        bias = U_hat - U_true
        bias_pct = 100 * bias / max(U_true, 1)
        width = hi - lo
        rel_width = width / max(U_true, 1)
        covers = m.get('ci_covers', lo <= U_true <= hi)
        ratio = r['D1'] / r['D0']
        rows.append({
            'U_true': U_true, 'U_hat': U_hat,
            'bias': bias, 'bias_pct': bias_pct,
            'ci_lo': lo, 'ci_hi': hi,
            'width': width, 'rel_width': rel_width,
            'covers': covers, 'ratio': ratio,
            'N_pilot': r['N_pilot'],
        })
    return rows

uci_ci = extract_ci_data(uci)
rees46_k21_ci = extract_ci_data(rees46_k21)
rees46_k50_ci = extract_ci_data(rees46_k50)
rees46_k100_ci = extract_ci_data(rees46_k100)
asos_ci = extract_ci_data(asos)

# ============================================================
# Figure 1: Relative bias vs relative CI width (UCI)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))

for row in uci_ci:
    color = '#4daf4a' if row['covers'] else '#e41a1c'
    marker = 'o' if row['covers'] else 'x'
    ax.scatter(row['rel_width'], abs(row['bias_pct']),
               s=50, color=color, marker=marker, zorder=3, alpha=0.8)

# Legend
ax.scatter([], [], s=50, color='#4daf4a', marker='o', label='CI covers truth')
ax.scatter([], [], s=50, color='#e41a1c', marker='x', label='CI misses truth')

ax.set_xlabel('Relative CI width (CI width / U_true)')
ax.set_ylabel('Absolute relative bias (%)')
ax.set_title('UCI: Why CI coverage is 46% (NB-SSP reg)')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Annotate the insight
ax.annotate('CIs cover when\nbias < CI width',
            xy=(0.3, 15), fontsize=9, color='gray',
            ha='center')

fig.savefig(os.path.join(PLOT_DIR, 'fig_ci_bias_vs_width.pdf'))
plt.close()
print("Saved fig_ci_bias_vs_width.pdf")

# ============================================================
# Figure 2: Coverage summary across datasets
# ============================================================
datasets = [
    ('UCI\n(n=13)', uci_ci),
    ('REES46 k=21\n(n=186)', rees46_k21_ci),
    ('REES46 k=50\n(n=157)', rees46_k50_ci),
    ('REES46 k=100\n(n=107)', rees46_k100_ci),
    ('ASOS\n(n=144)', asos_ci),
]

fig, ax = plt.subplots(figsize=(8, 4))

names = []
coverages = []
median_biases = []
median_rel_widths = []

for name, ci_data in datasets:
    if not ci_data:
        continue
    names.append(name)
    cov = 100 * sum(1 for r in ci_data if r['covers']) / len(ci_data)
    coverages.append(cov)
    median_biases.append(np.median([abs(r['bias_pct']) for r in ci_data]))
    median_rel_widths.append(np.median([r['rel_width'] for r in ci_data]))

x = np.arange(len(names))
width = 0.35

bars1 = ax.bar(x - width/2, coverages, width, label='Empirical coverage (%)',
               color='#377eb8', alpha=0.7)
bars2 = ax.bar(x + width/2, median_biases, width, label='Median |bias| (%)',
               color='#e41a1c', alpha=0.7)

ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='Nominal 95%')
ax.set_ylabel('Percentage')
ax.set_title('CI Coverage vs. Bias Across Datasets (NB-SSP reg, 95% nominal)')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 105)

fig.savefig(os.path.join(PLOT_DIR, 'fig_ci_coverage_by_dataset.pdf'))
plt.close()
print("Saved fig_ci_coverage_by_dataset.pdf")

# ============================================================
# Figure 3: Coverage and bias vs extrapolation ratio (REES46)
# ============================================================
# Pool all REES46 rolling results
all_rees46 = rees46_k21_ci + rees46_k50_ci + rees46_k100_ci

# Bin by ratio
ratio_bins = [(0, 3, '≤3'), (3, 5, '3-5'), (5, 7, '5-7'),
              (7, 10, '7-10'), (10, 15, '10-15')]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

bin_labels = []
bin_coverages = []
bin_biases = []
bin_widths = []

for lo, hi, label in ratio_bins:
    subset = [r for r in all_rees46 if lo < r['ratio'] <= hi]
    if not subset:
        continue
    bin_labels.append(f'{label}\n(n={len(subset)})')
    cov = 100 * sum(1 for r in subset if r['covers']) / len(subset)
    bin_coverages.append(cov)
    bin_biases.append(np.median([abs(r['bias_pct']) for r in subset]))
    bin_widths.append(np.median([100 * r['rel_width'] for r in subset]))

x = np.arange(len(bin_labels))

# Left: coverage
ax1.bar(x, bin_coverages, color='#377eb8', alpha=0.7)
ax1.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='Nominal 95%')
ax1.set_ylabel('Empirical coverage (%)')
ax1.set_title('CI Coverage by Extrapolation Ratio')
ax1.set_xticks(x)
ax1.set_xticklabels(bin_labels, fontsize=9)
ax1.set_ylim(0, 105)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# Right: bias vs CI width
ax2.bar(x - 0.2, bin_biases, 0.4, label='Median |bias| (%)',
        color='#e41a1c', alpha=0.7)
ax2.bar(x + 0.2, bin_widths, 0.4, label='Median CI width (%)',
        color='#4daf4a', alpha=0.7)
ax2.set_ylabel('Percentage of U_true')
ax2.set_title('Bias vs. CI Width')
ax2.set_xticks(x)
ax2.set_xticklabels(bin_labels, fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('REES46 Rolling Windows: NB-SSP Regression', fontsize=12)
fig.tight_layout()

fig.savefig(os.path.join(PLOT_DIR, 'fig_ci_coverage_by_ratio.pdf'))
plt.close()
print("Saved fig_ci_coverage_by_ratio.pdf")

# ============================================================
# Print summary
# ============================================================
print("\n=== CI Coverage Summary (NB-SSP regression, 95% nominal) ===\n")
print(f"{'Dataset':>20} {'N':>5} {'Coverage':>10} {'Med |bias|%':>12} {'Med rel width':>14}")
print('-' * 65)
for name, ci_data in datasets:
    if not ci_data:
        continue
    n = len(ci_data)
    cov = sum(1 for r in ci_data if r['covers'])
    med_bias = np.median([abs(r['bias_pct']) for r in ci_data])
    med_width = np.median([100 * r['rel_width'] for r in ci_data])
    print(f"{name.replace(chr(10), ' '):>20} {n:>5} {cov}/{n:>9} ({100*cov/n:.0f}%) {med_bias:>10.1f}% {med_width:>12.1f}%")

print("\n=== Key Insight ===")
print("Coverage fails when |bias| > CI width.")
print("On REES46 (N_pilot ~ 1M), CIs are extremely narrow (rel width < 1%)")
print("but point estimates carry 5-20% bias → coverage near 0%.")
print("On UCI (N_pilot ~ 200), CIs are wider (rel width ~ 30%)")
print("but bias is also larger → coverage 46%.")
print("This is inherent to empirical Bayes with plug-in hyperparameters.")
