"""
01_preprocess_uci.py
====================
Preprocesses the UCI Online Retail dataset into experiment windows
suitable for BNP model fitting.

Input:  lom_revision/data/OnlineRetail.csv
Output: lom_revision/data/user_day_aggregated.csv
        lom_revision/data/experiments_metadata.npy
        lom_revision/data/matrix_exp_{i}.npy          (full 28-day matrix)
        lom_revision/data/matrix_pilot_exp_{i}.npy     (first 7 days only)

Usage:  python 01_preprocess_uci.py
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# Configuration
# ============================================================
RAW_PATH = '../data/OnlineRetail.csv'
OUT_DIR = '../data'
WINDOW_SIZE = 28   # days per experiment window
D0 = 7             # pilot days

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Load and clean
# ============================================================
print("Loading raw data...")
df = pd.read_csv(RAW_PATH, encoding='latin-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"  Raw rows: {len(df):,}")

# Drop rows without CustomerID
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)

# Drop cancellations and negative quantities
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]

print(f"  After cleaning: {len(df):,} rows, {df['CustomerID'].nunique():,} customers")

# ============================================================
# Step 2: Aggregate to (CustomerID, date, trigger_count)
# ============================================================
df['date'] = df['InvoiceDate'].dt.date
user_day = df.groupby(['CustomerID', 'date']).size().reset_index(name='trigger_count')
user_day['date'] = pd.to_datetime(user_day['date'])

min_date = user_day['date'].min()
user_day['day_index'] = (user_day['date'] - min_date).dt.days

# First trigger day per customer
first_trigger = user_day.groupby('CustomerID')['day_index'].min().reset_index()
first_trigger.columns = ['CustomerID', 'first_trigger_day']
user_day = user_day.merge(first_trigger, on='CustomerID')

user_day.to_csv(os.path.join(OUT_DIR, 'user_day_aggregated.csv'), index=False)
print(f"  Saved user_day_aggregated.csv ({len(user_day):,} rows)")

# ============================================================
# Step 3: Create experiment windows
# ============================================================
D_total = user_day['day_index'].max() + 1
D1 = WINDOW_SIZE - D0

experiments = []
window_start = 0
exp_id = 0

while window_start + WINDOW_SIZE <= D_total:
    window_end = window_start + WINDOW_SIZE
    
    wd = user_day[(user_day['day_index'] >= window_start) & 
                  (user_day['day_index'] < window_end)].copy()
    wd['window_day'] = wd['day_index'] - window_start
    
    if len(wd) == 0:
        window_start += WINDOW_SIZE
        continue
    
    # New users per day
    first_in_window = wd.groupby('CustomerID')['window_day'].min().reset_index()
    first_in_window.columns = ['CustomerID', 'first_window_day']
    
    new_users_per_day = np.zeros(WINDOW_SIZE, dtype=int)
    for d in range(WINDOW_SIZE):
        new_users_per_day[d] = (first_in_window['first_window_day'] == d).sum()
    cumulative_users = np.concatenate([[0], new_users_per_day.cumsum()])
    
    # Total triggers per day
    triggers_per_day = np.zeros(WINDOW_SIZE, dtype=int)
    for d in range(WINDOW_SIZE):
        day_data = wd[wd['window_day'] == d]
        triggers_per_day[d] = day_data['trigger_count'].sum()
    
    # User x day matrix
    customers_in_window = sorted(wd['CustomerID'].unique())
    cust_to_idx = {c: i for i, c in enumerate(customers_in_window)}
    matrix = np.zeros((WINDOW_SIZE, len(customers_in_window)), dtype=int)
    for _, row in wd.iterrows():
        matrix[row['window_day'], cust_to_idx[row['CustomerID']]] = row['trigger_count']
    
    N_pilot = cumulative_users[D0]
    N_total_window = cumulative_users[-1]
    
    exp = {
        'exp_id': exp_id,
        'window_start': window_start,
        'window_end': window_end,
        'cumulative_users': cumulative_users.tolist(),
        'new_users_per_day': new_users_per_day.tolist(),
        'triggers_per_day': triggers_per_day.tolist(),
        'N_pilot': int(N_pilot),
        'N_total': int(N_total_window),
        'U_true': int(N_total_window - N_pilot),
        'T_true': int(triggers_per_day[D0:].sum()),
        'n_customers': len(customers_in_window),
        'D0': D0,
        'D1': D1,
    }
    experiments.append(exp)
    
    np.save(os.path.join(OUT_DIR, f'matrix_exp_{exp_id}.npy'), matrix)
    np.save(os.path.join(OUT_DIR, f'matrix_pilot_exp_{exp_id}.npy'), matrix[:D0, :])
    
    exp_id += 1
    window_start += WINDOW_SIZE

np.save(os.path.join(OUT_DIR, 'experiments_metadata.npy'), experiments)

print(f"\n  Created {len(experiments)} experiment windows")
for e in experiments:
    print(f"    Exp {e['exp_id']}: days {e['window_start']}-{e['window_end']}, "
          f"N_pilot={e['N_pilot']}, U_true={e['U_true']}, T_true={e['T_true']}")

print("\nDone.")
