"""
08_rees46_user_histograms.py
============================
Per-user trigger count and days-active histograms for REES46.
Uses the precomputed rees46_user_stats.npy (from 06_rees46_user_stats.py).

Output:
    fig_rees46_triggers_per_user.pdf  — log-log histogram (power-law evidence)
    fig_rees46_days_active.pdf        — bar chart of days active

Usage:  python3 SubmissionAOAS/lom_revision/scripts/08_rees46_user_histograms.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import apply_style
apply_style()

stats = np.load(os.path.join(DATA_DIR, 'rees46_user_stats.npy'),
                allow_pickle=True).item()

n_users = stats['n_users']
print(f"REES46: {n_users:,} users, {stats['total_rows']:,} events")

# ============================================================
# Figure 1: Per-user trigger count (log-log)
# ============================================================
hist = stats['triggers_per_user_histogram']
# hist[k] = number of users with exactly k triggers
# Skip k=0 (no users have 0 triggers)
vals = np.arange(1, len(hist))
freq = hist[1:]
mask = freq > 0

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(vals[mask], freq[mask] / n_users, s=4, alpha=0.4,
           color='#1b9e77', zorder=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Total triggers per user (7 months)')
ax.set_ylabel('Proportion of users')
ax.grid(True, alpha=0.3, which='both')

# Annotate key stats
ax.annotate(f'{stats["triggers_pct_1"]:.0f}% have\n1 trigger',
            xy=(1, freq[0] / n_users),
            xytext=(5, freq[0] / n_users * 0.3),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

# Point median annotation to the actual data
median_idx = int(stats['triggers_median'])
median_y = freq[median_idx - 1] / n_users  # freq is 0-indexed from val=1
ax.annotate(f'median = {median_idx}',
            xy=(median_idx, median_y),
            xytext=(median_idx * 8, median_y * 3),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_rees46_triggers_per_user.pdf'))
plt.close()
print("Saved fig_rees46_triggers_per_user.pdf")

# ============================================================
# Figure 2: Days active per user
# ============================================================
days_hist = stats['days_active_histogram']
max_show = 30
vals_d = np.arange(1, min(max_show + 1, len(days_hist)))
freq_d = days_hist[1:max_show + 1] if len(days_hist) > max_show else days_hist[1:]

fig, ax = plt.subplots(figsize=(5, 4))

ax.bar(vals_d, freq_d / n_users, color='#d95f02', alpha=0.7, edgecolor='white')
ax.set_xlabel('Number of active days (out of 211)')
ax.set_ylabel('Proportion of users')
ax.grid(True, alpha=0.3, axis='y')

ax.annotate(f'{stats["days_pct_1"]:.0f}% active\n1 day only',
            xy=(1, freq_d[0] / n_users),
            xytext=(6, freq_d[0] / n_users * 0.8),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

ax.annotate(f'{stats["days_pct_le3"]:.0f}% active\n≤3 days',
            xy=(3, freq_d[2] / n_users),
            xytext=(10, freq_d[2] / n_users * 2),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_rees46_days_active.pdf'))
plt.close()
print("Saved fig_rees46_days_active.pdf")

print(f"\nKey stats for the paper:")
print(f"  {n_users:,} users across 7 months")
print(f"  {stats['triggers_pct_1']:.0f}% have exactly 1 trigger")
print(f"  {stats['days_pct_1']:.0f}% active on exactly 1 day")
print(f"  {stats['days_pct_le3']:.0f}% active on ≤3 days")
print(f"  Median triggers/user: {stats['triggers_median']:.0f}")
print(f"  Max triggers/user: {stats['triggers_max']:,}")
