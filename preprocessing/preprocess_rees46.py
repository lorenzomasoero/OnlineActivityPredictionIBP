"""
preprocess_rees46.py
====================
Preprocesses the REES46 eCommerce behavior dataset into experiment windows.
Adapted from lom_revision/scripts/03_preprocess_rees46.py to run from SubmissionAOAS.

Usage:
    cd SubmissionAOAS/fitting
    python preprocess_rees46.py
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
DATA_DIR = '../data/REES46'
OUT_DIR = '../data/rees46_processed'
WINDOW_SIZE = 28
D0 = 7
TRIGGER_DEFINITION = 'all'  # view + cart + purchase

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Load & aggregate in chunks (memory-safe)
# ============================================================
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '2019-*.csv')) + glob.glob(os.path.join(DATA_DIR, '2020-*.csv')))
print(f"Found {len(csv_files)} file(s): {[os.path.basename(f) for f in csv_files]}")

CHUNK_SIZE = 2_000_000  # 2M rows at a time — ~200MB per chunk

print("\nAggregating to user-day pairs (chunked, memory-safe)...")
user_day_parts = []

for f in csv_files:
    print(f"  Processing {os.path.basename(f)} in chunks...")
    file_rows = 0
    for chunk in pd.read_csv(f, usecols=['event_time', 'event_type', 'user_id'],
                              parse_dates=['event_time'], chunksize=CHUNK_SIZE):
        chunk['date'] = chunk['event_time'].dt.date
        agg = chunk.groupby(['user_id', 'date']).size().reset_index(name='trigger_count')
        user_day_parts.append(agg)
        file_rows += len(chunk)
    print(f"    {file_rows:,} rows processed")

# Combine per-file aggregations and re-aggregate overlapping user-days
user_day = pd.concat(user_day_parts, ignore_index=True)
del user_day_parts  # free memory
user_day = user_day.groupby(['user_id', 'date'], as_index=False)['trigger_count'].sum()
user_day['date'] = pd.to_datetime(user_day['date'])

print(f"\nUser-day pairs: {len(user_day):,}")
print(f"Unique users: {user_day['user_id'].nunique():,}")

# ============================================================
# Step 2: Compute day indices and stats
# ============================================================
min_date = user_day['date'].min()
user_day['day_index'] = (user_day['date'] - min_date).dt.days

first_trigger = user_day.groupby('user_id')['day_index'].min().reset_index()
first_trigger.columns = ['user_id', 'first_trigger_day']
user_day = user_day.merge(first_trigger, on='user_id')

D_total = user_day['day_index'].max() + 1
N_total = user_day['user_id'].nunique()

print(f"\nUser-day pairs: {len(user_day):,}")
print(f"Unique users: {N_total:,}")
print(f"Day span: {D_total}")

# Per-user stats
days_active = user_day.groupby('user_id').size()
total_triggers = user_day.groupby('user_id')['trigger_count'].sum()
print(f"\nPer-user stats:")
print(f"  Days active: median={days_active.median():.0f}, mean={days_active.mean():.1f}")
print(f"  Total triggers: median={total_triggers.median():.0f}, mean={total_triggers.mean():.1f}")
print(f"  1-day users: {(days_active == 1).sum():,} ({100*(days_active == 1).sum()/N_total:.1f}%)")

# Save aggregated
user_day.to_csv(os.path.join(OUT_DIR, 'user_day_aggregated.csv'), index=False)

# ============================================================
# Step 3: Create experiment windows
# ============================================================
D1 = WINDOW_SIZE - D0
experiments = []
window_start = 0
exp_id = 0

print(f"\nCreating {WINDOW_SIZE}-day windows (D0={D0}, D1={D1})...")

while window_start + WINDOW_SIZE <= D_total:
    window_end = window_start + WINDOW_SIZE
    
    wd = user_day[(user_day['day_index'] >= window_start) &
                  (user_day['day_index'] < window_end)].copy()
    wd['window_day'] = wd['day_index'] - window_start
    
    if len(wd) == 0:
        window_start += WINDOW_SIZE
        continue
    
    # New users per day
    first_in_window = wd.groupby('user_id')['window_day'].min().reset_index()
    first_in_window.columns = ['user_id', 'first_window_day']
    
    new_users_per_day = np.zeros(WINDOW_SIZE, dtype=int)
    for d in range(WINDOW_SIZE):
        new_users_per_day[d] = (first_in_window['first_window_day'] == d).sum()
    cumulative_users = np.concatenate([[0], new_users_per_day.cumsum()])
    
    # Total triggers per day
    triggers_per_day = np.zeros(WINDOW_SIZE, dtype=int)
    for d in range(WINDOW_SIZE):
        triggers_per_day[d] = wd[wd['window_day'] == d]['trigger_count'].sum()
    
    # Build matrix only if manageable
    customers_in_window = sorted(wd['user_id'].unique())
    n_customers = len(customers_in_window)
    store_matrix = (WINDOW_SIZE * n_customers) < 5_000_000  # 5M cells max
    
    if store_matrix:
        cust_to_idx = {c: i for i, c in enumerate(customers_in_window)}
        matrix = np.zeros((WINDOW_SIZE, n_customers), dtype=np.int16)
        for _, row in wd.iterrows():
            matrix[row['window_day'], cust_to_idx[row['user_id']]] = min(row['trigger_count'], 32767)
        np.save(os.path.join(OUT_DIR, f'matrix_exp_{exp_id}.npy'), matrix)
        np.save(os.path.join(OUT_DIR, f'matrix_pilot_exp_{exp_id}.npy'), matrix[:D0, :])
    
    N_pilot = int(cumulative_users[D0])
    N_total_window = int(cumulative_users[-1])
    
    exp = {
        'exp_id': exp_id,
        'window_start': window_start,
        'window_end': window_end,
        'cumulative_users': cumulative_users.tolist(),
        'new_users_per_day': new_users_per_day.tolist(),
        'triggers_per_day': triggers_per_day.tolist(),
        'N_pilot': N_pilot,
        'N_total': N_total_window,
        'U_true': N_total_window - N_pilot,
        'T_true': int(triggers_per_day[D0:].sum()),
        'n_customers': n_customers,
        'D0': D0,
        'D1': D1,
        'has_matrix': store_matrix,
    }
    experiments.append(exp)
    
    print(f"  Exp {exp_id}: days {window_start}-{window_end}, "
          f"N_pilot={N_pilot:,}, U_true={exp['U_true']:,}, "
          f"T_true={exp['T_true']:,}, matrix={'yes' if store_matrix else 'NO'}")
    
    exp_id += 1
    window_start += WINDOW_SIZE

np.save(os.path.join(OUT_DIR, 'experiments_metadata.npy'), experiments)
print(f"\nCreated {len(experiments)} experiments. Saved to {OUT_DIR}/")
