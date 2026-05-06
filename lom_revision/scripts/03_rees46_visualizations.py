"""
03_rees46_visualizations.py
===========================
Power-law / heavy-tail visualizations for the REES46 eCommerce dataset.
Addresses R2-minor-a ("visualize data characteristics") and R2-minor-d ("power-law reference").

Uses only the preprocessed metadata (no raw CSVs needed).

Output figures:
    lom_revision/plots/fig_rees46_cumulative_users.pdf  — cumulative new users (3 experiments)
    lom_revision/plots/fig_rees46_new_users_per_day.pdf — daily new user arrivals (power-law decay)
    lom_revision/plots/fig_rees46_triggers_per_day.pdf  — daily trigger volume

Usage:  cd SubmissionAOAS/lom_revision/scripts && python3 03_rees46_visualizations.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Setup
# ============================================================
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
from plot_style import COLORS, MARKERS, LINESTYLES, apply_style
apply_style()

EXP_KEYS = ['exp_0', 'exp_1', 'exp_2']
STYLES = [
    {'linestyle': LINESTYLES[k], 'marker': MARKERS[k], 'markevery': 3, 'markersize': 4}
    for k in EXP_KEYS
]
EXP_COLORS = [COLORS[k] for k in EXP_KEYS]

# ============================================================
# Load data
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')  # SubmissionAOAS/

# 7 non-overlapping 28-day experiments (D0=7, D1=21)
experiments = np.load(os.path.join(BASE, 'data/preprocessed/rees46/experiments_metadata.npy'),
                      allow_pickle=True)
# 186 rolling experiments (D0=7, k=21)
rolling_k21 = np.load(os.path.join(BASE, 'data/rees46_processed/experiments_rolling_k21.npy'),
                       allow_pickle=True)

print(f"Loaded {len(experiments)} non-overlapping experiments")
print(f"Loaded {len(rolling_k21)} rolling experiments (k=21)")

# ============================================================
# Figure 1: Cumulative new users over time (3 experiments)
# ============================================================
# Pick experiments: early (exp 0), middle (exp 3), late (exp 6)
selected = [0, 3, 6]
labels = [
    f'Exp {i} (N_pilot={experiments[i]["N_pilot"]:,})'
    for i in selected
]

fig, ax = plt.subplots(figsize=(6, 4))

for idx, (exp_id, label) in enumerate(zip(selected, labels)):
    exp = experiments[exp_id]
    cum_users = np.array(exp['cumulative_users'])
    days = np.arange(len(cum_users))
    ax.plot(days, cum_users / 1e6,
            color=EXP_COLORS[idx], linewidth=2, label=label,
            **STYLES[idx])

ax.axvline(x=7, color='gray', linestyle=':', alpha=0.7,
           label=r'Pilot end ($D_0=7$)')
ax.set_xlabel('Day within experiment window')
ax.set_ylabel('Cumulative unique users (millions)')
ax.set_title('REES46: Cumulative new users over time')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(PLOT_DIR, 'fig_rees46_cumulative_users.pdf'))
plt.close()
print("Saved fig_rees46_cumulative_users.pdf")

# ============================================================
# Figure 2: New users per day (shows power-law decay)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

for idx, (exp_id, label) in enumerate(zip(selected, labels)):
    exp = experiments[exp_id]
    nupd = np.array(exp['new_users_per_day'])
    days = np.arange(1, len(nupd) + 1)
    ax.plot(days, nupd / 1e3,
            color=EXP_COLORS[idx], linewidth=1.5, label=label,
            **STYLES[idx])

ax.axvline(x=7, color='gray', linestyle=':', alpha=0.7,
           label=r'Pilot end ($D_0=7$)')
ax.set_xlabel('Day within experiment window')
ax.set_ylabel('New users (thousands)')
ax.set_title('REES46: Daily new user arrivals')
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(PLOT_DIR, 'fig_rees46_new_users_per_day.pdf'))
plt.close()
print("Saved fig_rees46_new_users_per_day.pdf")

# ============================================================
# Figure 3: Log-log of new users per day (power-law evidence)
# ============================================================
# Aggregate across all 7 experiments: pool new_users_per_day
# and plot day index vs. new users on log-log scale
fig, ax = plt.subplots(figsize=(5, 4))

for idx, (exp_id, label) in enumerate(zip(selected, labels)):
    exp = experiments[exp_id]
    nupd = np.array(exp['new_users_per_day'])
    days = np.arange(1, len(nupd) + 1)
    ax.scatter(days, nupd, s=20, alpha=0.7,
               color=EXP_COLORS[idx], marker=STYLES[idx]['marker'],
               label=label, zorder=3)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Day (log scale)')
ax.set_ylabel('New users on that day (log scale)')
ax.set_title('REES46: Power-law decay of new user arrivals')
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, which='both')

fig.savefig(os.path.join(PLOT_DIR, 'fig_rees46_powerlaw_decay.pdf'))
plt.close()
print("Saved fig_rees46_powerlaw_decay.pdf")

# ============================================================
# Figure 4: Triggers per day (shows re-trigger volume)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

for idx, (exp_id, label) in enumerate(zip(selected, labels)):
    exp = experiments[exp_id]
    tpd = np.array(exp['triggers_per_day'])
    days = np.arange(1, len(tpd) + 1)
    ax.plot(days, tpd / 1e6,
            color=EXP_COLORS[idx], linewidth=1.5, label=label,
            **STYLES[idx])

ax.axvline(x=7, color='gray', linestyle=':', alpha=0.7,
           label=r'Pilot end ($D_0=7$)')
ax.set_xlabel('Day within experiment window')
ax.set_ylabel('Total triggers (millions)')
ax.set_title('REES46: Daily trigger volume')
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(PLOT_DIR, 'fig_rees46_triggers_per_day.pdf'))
plt.close()
print("Saved fig_rees46_triggers_per_day.pdf")

# ============================================================
# Summary statistics for the paper
# ============================================================
print("\n=== REES46 Summary Statistics ===")
for i, exp in enumerate(experiments):
    nupd = np.array(exp['new_users_per_day'])
    tpd = np.array(exp['triggers_per_day'])
    ratio = tpd.sum() / exp['N_total']
    print(f"Exp {i}: N_pilot={exp['N_pilot']:>10,}  "
          f"N_total={exp['N_total']:>10,}  "
          f"U_true={exp['U_true']:>10,}  "
          f"T_total={tpd.sum():>12,}  "
          f"triggers/user={ratio:.1f}")

# 1-day user fraction (from rolling k=21: users who appear in pilot but not follow-up)
# Approximate: fraction of pilot users who don't return
exp0 = experiments[0]
frac_no_return = exp0['U_true'] / exp0['N_total']
print(f"\nFraction of users first seen after pilot (exp 0): {frac_no_return:.1%}")
print(f"Day 1 new users / total: {exp0['new_users_per_day'][0] / exp0['N_total']:.1%}")

print("\nAll REES46 visualizations complete.")
