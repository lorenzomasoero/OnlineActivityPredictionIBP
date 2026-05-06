"""
02_powerlaw_visualizations.py
=============================
Produces power-law / heavy-tail visualizations from the UCI Online Retail data.
These address R2-minor-a ("visualize data characteristics") and R2-minor-d ("power-law reference").

Output figures:
    lom_revision/plots/fig_powerlaw_triggers.pdf   — log-log histogram of per-user trigger counts
    lom_revision/plots/fig_cumulative_users.pdf     — cumulative new users over time (3 experiments)

Usage:  cd lom_revision/scripts && python 02_powerlaw_visualizations.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# Setup
# ============================================================
os.makedirs('../plots', exist_ok=True)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================
# Load data
# ============================================================
user_day = pd.read_csv('../data/user_day_aggregated.csv')
experiments = np.load('../data/experiments_metadata.npy', allow_pickle=True)

# ============================================================
# Figure 1: Per-user trigger count distribution (log-log)
# ============================================================
total_triggers = user_day.groupby('CustomerID')['trigger_count'].sum()

fig, ax = plt.subplots(figsize=(5, 4))

# Compute empirical PMF
counts = total_triggers.values
unique_vals, freq = np.unique(counts, return_counts=True)
freq_normalized = freq / freq.sum()

ax.scatter(unique_vals, freq_normalized, s=8, alpha=0.6, color='steelblue', zorder=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Total triggers per customer')
ax.set_ylabel('Proportion of customers')
ax.set_title('Distribution of per-customer trigger counts')
ax.grid(True, alpha=0.3, which='both')

# Annotate key stats
n_total = len(counts)
n_one = (counts == 1).sum()
ax.annotate(f'{100*n_one/n_total:.0f}% of customers\nhave 1 trigger',
            xy=(1, freq_normalized[0]),
            xytext=(5, freq_normalized[0] * 0.3),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

fig.savefig('../plots/fig_powerlaw_triggers.pdf')
plt.close()
print("Saved fig_powerlaw_triggers.pdf")

# ============================================================
# Figure 2: Cumulative new users over time (3 experiments)
# ============================================================
# Pick experiments: sparse (1), medium (6), large (12)
selected = [1, 6, 12]
labels = ['Exp 1 (sparse, Jan)', 'Exp 6 (medium, Jun)', 'Exp 12 (large, Nov)']
linestyles = ['-', '--', '-.']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(6, 4))

for idx, (exp_id, label, ls, col) in enumerate(zip(selected, labels, linestyles, colors)):
    exp = experiments[exp_id]
    cum_users = np.array(exp['cumulative_users'])
    days = np.arange(len(cum_users))
    ax.plot(days, cum_users, linestyle=ls, color=col, linewidth=2, label=label)

ax.axvline(x=7, color='gray', linestyle=':', alpha=0.7, label=r'Pilot end ($D_0=7$)')
ax.set_xlabel('Day within experiment window')
ax.set_ylabel('Cumulative unique customers')
ax.set_title('Cumulative new users over time')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.savefig('../plots/fig_cumulative_users.pdf')
plt.close()
print("Saved fig_cumulative_users.pdf")

# ============================================================
# Figure 3: Days active per customer (complementary view)
# ============================================================
days_active = user_day.groupby('CustomerID').size()

fig, ax = plt.subplots(figsize=(5, 4))

vals, freq = np.unique(days_active.values, return_counts=True)
freq_norm = freq / freq.sum()

ax.bar(vals[:20], freq_norm[:20], color='steelblue', alpha=0.7, edgecolor='white')
ax.set_xlabel('Number of active days')
ax.set_ylabel('Proportion of customers')
ax.set_title('Distribution of customer activity span')
ax.grid(True, alpha=0.3, axis='y')

# Annotate
pct_1 = 100 * (days_active == 1).sum() / len(days_active)
pct_3 = 100 * (days_active <= 3).sum() / len(days_active)
ax.annotate(f'{pct_1:.0f}% active\n1 day only',
            xy=(1, freq_norm[0]),
            xytext=(5, freq_norm[0] * 0.8),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

fig.savefig('../plots/fig_days_active.pdf')
plt.close()
print("Saved fig_days_active.pdf")

print("\nAll visualizations complete.")
