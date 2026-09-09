"""
preprocess_rees46_rolling.py  (v3 — streaming)
===============================================
Rolling-window preprocessor for REES46.
D0=7 pilot days, predict over horizons k=21,50,100.

Memory strategy: single pass builds {user_id: first_day} dict (~1.7GB for 15.6M users).
Then derive new_users_per_day[d] = count of users with first_day == d.
Rolling windows are just slices of this array — O(1) per window.

Peak memory: ~1.8GB. Safe with 7GB+ available.
"""

import numpy as np
import pandas as pd
import os
import glob
import time

DATA_DIR = '../data/REES46'
OUT_DIR = '../data/rees46_processed'
D0 = 7
K_VALUES = [21, 50, 100]
CHUNK_SIZE = 2_000_000

os.makedirs(OUT_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '2019-*.csv')) +
                   glob.glob(os.path.join(DATA_DIR, '2020-*.csv')))
print(f"Found {len(csv_files)} CSV files")

# ============================================================
# Pass 1: Build user_id -> first_day dict
# ============================================================
print("\nPass 1: finding first activity day per user (chunked)...")
t0 = time.time()

# Find min_date
min_date = None
for f in csv_files:
    chunk = pd.read_csv(f, usecols=['event_time'], nrows=10, parse_dates=['event_time'])
    d = chunk['event_time'].min()
    if min_date is None or d < min_date:
        min_date = d
min_date = pd.Timestamp(min_date.date(), tz=min_date.tz)
print(f"  min_date = {min_date.date()}")

first_day = {}  # user_id -> earliest day_index
total_rows = 0

for f in csv_files:
    print(f"  {os.path.basename(f)}...", end=" ", flush=True)
    file_rows = 0
    for chunk in pd.read_csv(f, usecols=['event_time', 'user_id'],
                              parse_dates=['event_time'], chunksize=CHUNK_SIZE):
        chunk['day'] = (chunk['event_time'].dt.normalize() - min_date).dt.days
        for uid, day in zip(chunk['user_id'].values, chunk['day'].values):
            day = int(day)
            if uid not in first_day or day < first_day[uid]:
                first_day[uid] = day
        file_rows += len(chunk)
    total_rows += file_rows
    print(f"{file_rows:,} rows")

N_users = len(first_day)
D_total = max(first_day.values()) + 1
print(f"  Total rows: {total_rows:,}, Users: {N_users:,}, Days: {D_total}")
print(f"  Pass 1 done in {time.time()-t0:.0f}s")

# ============================================================
# Derive new_users_per_day
# ============================================================
print("\nBuilding new_users_per_day array...")
new_users_per_day = np.zeros(D_total, dtype=np.int32)
for d in first_day.values():
    new_users_per_day[d] += 1
del first_day  # free ~1.7GB

print(f"  Total new users check: {new_users_per_day.sum():,} (should be {N_users:,})")
print(f"  Day 0: {new_users_per_day[0]:,}, Day 1: {new_users_per_day[1]:,}, ...")

# ============================================================
# Build rolling experiments
# ============================================================
print(f"\nCreating rolling experiments (D0={D0}, k={K_VALUES})...")
cumulative_new = np.concatenate([[0], new_users_per_day.cumsum()])

for k in K_VALUES:
    W = D0 + k
    n_exp = D_total - W + 1
    print(f"\n  k={k}: {n_exp} experiments (window={W} days)")
    experiments = []

    for t in range(n_exp):
        # Users whose first_day is in [t, t+D0): pilot new users
        N_pilot = int(cumulative_new[t + D0] - cumulative_new[t])
        # Users whose first_day is in [t, t+W): total new users in window
        N_total_w = int(cumulative_new[t + W] - cumulative_new[t])
        U_true = N_total_w - N_pilot

        # Cumulative curve within window (for model fitting)
        # counts_short[j] = cumulative new users from day t to day t+j (inclusive)
        cumulative_users = (cumulative_new[t:t+W+1] - cumulative_new[t]).tolist()

        exp = {
            'exp_id': t, 'k': k, 'D0': D0, 'D1': k,
            'window_start': t, 'window_end': t + W,
            'cumulative_users': cumulative_users,
            'N_pilot': N_pilot, 'N_total': N_total_w,
            'U_true': U_true,
        }
        experiments.append(exp)

        if t % 50 == 0 or t == n_exp - 1:
            print(f"    exp {t}/{n_exp}: days {t}-{t+W}, N_pilot={N_pilot:,}, U_true={U_true:,}")

    outfile = os.path.join(OUT_DIR, f'experiments_rolling_k{k}.npy')
    np.save(outfile, experiments)
    print(f"  Saved {len(experiments)} experiments to {outfile}")

print(f"\nDone. Total experiments: {sum(D_total - D0 - k + 1 for k in K_VALUES)}")
