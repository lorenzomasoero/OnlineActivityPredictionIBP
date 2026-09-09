"""
06_rees46_user_stats.py
=======================
Extracts per-user summary statistics from raw REES46 CSVs.
Produces a small .npy file with distributions needed for
the power-law trigger count histogram (like the UCI version).

Run on the dev-dsk where raw CSVs live:
    cd ~/PredictNewCustomers/SubmissionAOAS/lom_revision/scripts
    nice -n 19 python3 06_rees46_user_stats.py

Output:
    ../data/rees46_user_stats.npy  (~small, commit this)

Peak memory: ~2GB (user_id -> counts dict for ~15M users)
Runtime: ~10-15 min
"""

import numpy as np
import pandas as pd
import os
import glob
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'REES46')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

CHUNK_SIZE = 2_000_000

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '2019-*.csv')) +
                   glob.glob(os.path.join(DATA_DIR, '2020-*.csv')))
print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")

if not csv_files:
    print("ERROR: No CSV files found. Are you on the dev-dsk?")
    exit(1)

# ============================================================
# Pass 1: per-user total triggers and days active
# ============================================================
print("\nPass 1: counting triggers and active days per user...")
t0 = time.time()

user_triggers = defaultdict(int)   # user_id -> total event count
user_days = defaultdict(set)       # user_id -> set of active days
total_rows = 0

for f in csv_files:
    fname = os.path.basename(f)
    print(f"  {fname}...", end=" ", flush=True)
    file_rows = 0
    for chunk in pd.read_csv(f, usecols=['event_time', 'user_id'],
                              parse_dates=['event_time'],
                              chunksize=CHUNK_SIZE):
        dates = chunk['event_time'].dt.date
        for uid, date in zip(chunk['user_id'].values, dates.values):
            user_triggers[uid] += 1
            user_days[uid].add(date)
        file_rows += len(chunk)
    total_rows += file_rows
    print(f"{file_rows:,} rows")

elapsed = time.time() - t0
n_users = len(user_triggers)
print(f"\nTotal rows: {total_rows:,}")
print(f"Unique users: {n_users:,}")
print(f"Elapsed: {elapsed:.0f}s")

# ============================================================
# Compute distributions
# ============================================================
print("\nComputing distributions...")

triggers_per_user = np.array(list(user_triggers.values()))
days_per_user = np.array([len(d) for d in user_days.values()])

# Free memory
del user_triggers, user_days

stats = {
    'n_users': n_users,
    'total_rows': total_rows,
    'n_files': len(csv_files),
    # Per-user trigger count distribution (for log-log histogram)
    'triggers_per_user_histogram': np.bincount(
        np.minimum(triggers_per_user, 10000)  # cap at 10K for bincount
    ),
    # Per-user days active distribution
    'days_active_histogram': np.bincount(days_per_user),
    # Summary stats
    'triggers_mean': float(np.mean(triggers_per_user)),
    'triggers_median': float(np.median(triggers_per_user)),
    'triggers_max': int(np.max(triggers_per_user)),
    'triggers_pct_1': float(100 * (triggers_per_user == 1).sum() / n_users),
    'days_mean': float(np.mean(days_per_user)),
    'days_median': float(np.median(days_per_user)),
    'days_max': int(np.max(days_per_user)),
    'days_pct_1': float(100 * (days_per_user == 1).sum() / n_users),
    'days_pct_le3': float(100 * (days_per_user <= 3).sum() / n_users),
}

outfile = os.path.join(OUT_DIR, 'rees46_user_stats.npy')
np.save(outfile, stats)
print(f"\nSaved to {outfile}")

# Print summary
print(f"\n=== REES46 Per-User Statistics (all 7 months) ===")
print(f"Users: {n_users:,}")
print(f"Triggers/user: mean={stats['triggers_mean']:.1f}, "
      f"median={stats['triggers_median']:.0f}, max={stats['triggers_max']:,}")
print(f"  {stats['triggers_pct_1']:.1f}% of users have exactly 1 trigger")
print(f"Days active/user: mean={stats['days_mean']:.1f}, "
      f"median={stats['days_median']:.0f}, max={stats['days_max']}")
print(f"  {stats['days_pct_1']:.1f}% active on exactly 1 day")
print(f"  {stats['days_pct_le3']:.1f}% active on ≤3 days")
