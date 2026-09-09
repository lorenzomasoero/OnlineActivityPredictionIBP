#!/usr/bin/env python3
"""Generate a SYNTHETIC mock proprietary dataset so run_proprietary.py can exercise
the code path without any real proprietary data. No real user data is used.

Format matches the cumulative-first-trigger experiment records the engine consumes
(D0=7 pilot, D1=21 follow-up). Deterministic (seed 0).
Writes: data/proprietary/mock_proprietary_data.npy
"""
import os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "..", "data", "proprietary", "mock_proprietary_data.npy")
rng = np.random.default_rng(0)

D0, D1, N_EXP = 7, 21, 40
experiments = []
for i in range(N_EXP):
    # sub-linear cumulative new-user arrivals with per-experiment scale + noise
    scale = rng.uniform(200, 5000)
    days = np.arange(D0 + D1 + 1)
    growth = scale * np.power(days, rng.uniform(0.6, 0.9))  # concave
    noise = rng.normal(0, 0.01 * scale, size=days.shape).cumsum()
    cum = np.maximum.accumulate(np.clip(growth + noise, 0, None)).round().astype(int)
    cum[0] = 0
    N_pilot = int(cum[D0])
    experiments.append({
        "exp_id": i, "D0": D0, "D1": D1,
        "N_pilot": N_pilot, "N_total": int(cum[-1]),
        "U_true": int(cum[-1] - N_pilot),
        "cumulative_users": cum,
    })

os.makedirs(os.path.dirname(DEST), exist_ok=True)
np.save(DEST, np.array(experiments, dtype=object))
print(f"Wrote {DEST}: {N_EXP} synthetic experiments (D0={D0}, D1={D1}). MOCK — not real data.")
