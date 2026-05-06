"""
03_preprocess_rees46.py
=======================
Preprocesses the REES46 eCommerce behavior dataset into experiment windows.

The REES46 dataset has columns:
    event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session

We define a "trigger" as any event (view, cart, or purchase) by a user on a given day.
trigger_count = number of events per user per day.

Input:  lom_revision/data/2019-Oct.csv  (or whichever month is downloaded)
Output: lom_revision/data/rees46/user_day_aggregated.csv
        lom_revision/data/rees46/experiments_metadata.npy
        lom_revision/data/rees46/matrix_exp_{i}.npy
        lom_revision/data/rees46/matrix_pilot_exp_{i}.npy

Usage:  cd lom_revision/scripts && python 03_preprocess_rees46.py
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
DATA_DIR = '../data'
OUT_DIR = '../data/rees46'
WINDOW_SIZE = 28
D0 = 7

# Which event types count as triggers
# Options: 'all' (view+cart+purchase), 'purchase' (purchases only), 'cart_purchase' (cart+purchase)
TRIGGER_DEFINITION = 'all'

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Find and load REES46 CSV(s)
# ============================================================
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '2019-*.csv')))
if not csv_files:
    # Also check for .csv.gz
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '2019-*.csv.gz')))

if not csv_files:
    print("ERROR: No REES46 CSV files found in", DATA_DIR)
    print("Expected files like: 2019-Oct.csv, 2019-Nov.csv, etc.")
    exit(1)

print(f"Found {len(csv_files)} REES46 file(s): {csv_files}")

dfs = []
for f in csv_files:
    print(f"  Loading {f}...")
    df = pd.read_csv(f, usecols=['event_time', 'event_type', 'user_id'],
                     parse_dates=['event_time'])
    print(f"    {len(df):,} rows")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Event types: {df['event_type'].value_counts().to_dict()}")
print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")

# ============================================================
# Step 2: Filter by trigger definition
# ============================================================
if TRIGGER_DEFINITION == 'purchase':
    df = df[df['event_type'] == 'purchase']
elif TRIGGER_DEFINITION == 'cart_purchase':
    df = df[df['event_type'].isin(['cart', 'purchase'])]
# else: keep all events

print(f"\nAfter trigger filter ({TRIGGER_DEFINITION}): {len(df):,} rows, {df['user_id'].nunique():,} users")

# ============================================================
# Step 3: Aggregate to (user_id, date, trigger_count)
# ============================================================
df['date'] = df['event_time'].dt.date
user_day = df.groupby(['user_id', 'date']).size().reset_index(name='trigger_count')
user_day['date'] = pd.to_datetime(user_day['date'])

min_date = user_day['date'].min()
user_day['day_index'] = (user_day['date'] - min_date).dt.days

# First trigger day per user
first_trigger = user_day.groupby('user_id')['day_index'].min().reset_index()
first_trigger.columns = ['user_id', 'first_trigger_day']
user_day = user_day.merge(first_trigger, on='user_id')

D_total = user_day['day_index'].max() + 1
N_total = user_day['user_id'].nunique()

print(f"\n=== AGGREGATED DATA ===")
print(f"User-day pairs: {len(user_day):,}")
print(f"Unique users: {N_total:,}")
print(f"Day span: {D_total}")

# Save
user_day.to_csv(os.path.join(OUT_DIR, 'user_day_aggregated.csv'), index=False)

# ============================================================
# Step 4: Per-user statistics
# ============================================================
total_triggers = user_day.groupby('user_id')['trigger_count'].sum()
days_active = user_day.groupby('user_id').size()

print(f"\n=== PER-USER STATISTICS ===")
print(f"Total triggers: median={total_triggers.median():.0f}, mean={total_triggers.mean():.1f}, max={total_triggers.max()}")
print(f"Days active: median={days_active.median():.0f}, mean={days_active.mean():.1f}, max={days_active.max()}")
print(f"1-day users: {(days_active == 1).sum()} ({100*(days_active == 1).sum()/N_total:.1f}%)")
print(f"<=3-day users: {(days_active <= 3).sum()} ({100*(days_active <= 3).sum()/N_total:.1f}%)")

# ============================================================
# Step 5: Create experiment windows
# ============================================================
D1 = WINDOW_SIZE - D0
experiments = []
window_start = 0
exp_id = 0

print(f"\n=== CREATING EXPERIMENT WINDOWS ===")
print(f"Window size: {WINDOW_SIZE} days, D0={D0}, D1={D1}")

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
    
    # Build user x day matrix
    # For large datasets, this can be memory-intensive. Use sparse if needed.
    customers_in_window = sorted(wd['user_id'].unique())
    n_customers = len(customers_in_window)
    
    # Only build full matrix if manageable (< 1M cells)
    store_matrix = (WINDOW_SIZE * n_customers) < 1_000_000
    
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
          f"N_pilot={N_pilot:,}, N_total={N_total_window:,}, "
          f"U_true={exp['U_true']:,}, T_true={exp['T_true']:,}, "
          f"matrix={'yes' if store_matrix else 'NO (too large)'}")
    
    exp_id += 1
    window_start += WINDOW_SIZE

np.save(os.path.join(OUT_DIR, 'experiments_metadata.npy'), experiments)

print(f"\nCreated {len(experiments)} experiments")
print(f"Files saved to {OUT_DIR}/")
print("Done.")
