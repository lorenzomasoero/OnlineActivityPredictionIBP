"""
09_asos_data_characteristics.py
===============================
ASOS data characteristic figures for parity with REES46/UCI.

Output:
    fig_asos_cumulative_users.pdf  — cumulative new users for 3 representative experiments
    fig_asos_d1_distribution.pdf   — distribution of follow-up lengths

Usage:  python3 SubmissionAOAS/lom_revision/scripts/09_asos_data_characteristics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
PLOT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

from plot_style import apply_style
apply_style()

asos = np.load(os.path.join(BASE, 'data/preprocessed/asos/PAPER_asos_data.npy'),
               allow_pickle=True).item()

# Collect all arms with their stats
arms = []
for exp_id, exp in asos.items():
    for arm_name in ['C', 'T']:
        short = exp['first_trigger_counts_short'][arm_name].astype(int)
        long = exp['first_trigger_counts_long'][arm_name].astype(int)
        d0 = len(short) - 1
        d1 = len(long) - len(short)
        arms.append({
            'exp_id': exp_id, 'arm': arm_name,
            'D0': d0, 'D1': d1,
            'N_pilot': int(short[-1]), 'N_total': int(long[-1]),
            'cumulative': long,
        })

arms.sort(key=lambda a: a['D1'])
print(f"ASOS: {len(arms)} arms across {len(asos)} experiments")
print(f"D1 range: {arms[0]['D1']} to {arms[-1]['D1']}")
print(f"N_pilot range: {min(a['N_pilot'] for a in arms):,} to {max(a['N_pilot'] for a in arms):,}")

# ============================================================
# Figure 1: Cumulative users for 3 representative experiments
# ============================================================
# Pick: short D1 (~10), medium (~50), long (~100+)
selected = []
for target_d1 in [10, 55, 150]:
    best = min(arms, key=lambda a: abs(a['D1'] - target_d1))
    selected.append(best)

fig, ax = plt.subplots(figsize=(6, 4.5))

styles = [
    {'color': '#1b9e77', 'linestyle': '-', 'marker': 'o'},
    {'color': '#d95f02', 'linestyle': '--', 'marker': 's'},
    {'color': '#7570b3', 'linestyle': '-.', 'marker': '^'},
]

for arm, style in zip(selected, styles):
    cum = arm['cumulative']
    days = np.arange(len(cum))
    # Normalize to millions for readability
    scale = 1e6 if cum[-1] > 1e6 else 1e3
    unit = 'M' if scale == 1e6 else 'K'
    label = (f'{arm["exp_id"]}_{arm["arm"]} '
             f'(D₁={arm["D1"]}, N={arm["N_total"]/scale:.1f}{unit})')
    markevery = max(1, len(days) // 10)
    ax.plot(days, cum / scale,
            color=style['color'], linestyle=style['linestyle'],
            marker=style['marker'], markevery=markevery, markersize=4,
            linewidth=1.5, label=label)

ax.axvline(x=7, color='gray', linestyle=':', alpha=0.7, label=r'Pilot end ($D_0=7$)')
ax.set_xlabel('Day within experiment')
ax.set_ylabel(f'Cumulative unique users')
ax.set_title('ASOS: Cumulative new users (3 representative experiments)')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_asos_cumulative_users.pdf'))
plt.close()
print("Saved fig_asos_cumulative_users.pdf")

# ============================================================
# Figure 2: Distribution of D1 (follow-up lengths)
# ============================================================
d1_vals = [a['D1'] for a in arms]

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.hist(d1_vals, bins=30, color='#377eb8', alpha=0.7, edgecolor='white')
ax.axvline(x=np.median(d1_vals), color='#e41a1c', linestyle='--', linewidth=1.5,
           label=f'Median D₁ = {np.median(d1_vals):.0f}')
ax.set_xlabel(r'Follow-up length $D_1$ (days)')
ax.set_ylabel('Number of arms')
ax.set_title(f'ASOS: Distribution of follow-up lengths (144 arms, D₀=7)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, 'fig_asos_d1_distribution.pdf'))
plt.close()
print("Saved fig_asos_d1_distribution.pdf")

# ============================================================
# Summary stats for the report
# ============================================================
print(f"\n=== ASOS Summary Statistics ===")
print(f"Experiments: {len(asos)}")
print(f"Arms: {len(arms)} (C + T per experiment)")
print(f"D0: always 7")
print(f"D1: min={min(d1_vals)}, max={max(d1_vals)}, median={np.median(d1_vals):.0f}")
print(f"D1/D0: min={min(d1_vals)/7:.1f}, max={max(d1_vals)/7:.1f}, median={np.median(d1_vals)/7:.1f}")
print(f"N_pilot: min={min(a['N_pilot'] for a in arms):,}, max={max(a['N_pilot'] for a in arms):,}, median={np.median([a['N_pilot'] for a in arms]):,.0f}")
print(f"N_total: min={min(a['N_total'] for a in arms):,}, max={max(a['N_total'] for a in arms):,}, median={np.median([a['N_total'] for a in arms]):,.0f}")
