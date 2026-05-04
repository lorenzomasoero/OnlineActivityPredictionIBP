"""
preprocess_uci.py
=================
Preprocesses the UCI Online Retail dataset into experiment windows.

Downloads and processes the UCI Online Retail II dataset, creating:
  - Per-experiment trigger matrices (days x customers)
  - Experiment metadata (D0, D1, N_pilot, U_true, cumulative_users)

Output:
    data/preprocessed/uci/experiments_metadata.npy
    data/preprocessed/uci/matrix_exp_{i}.npy
    data/preprocessed/uci/matrix_pilot_exp_{i}.npy

Usage:
    python experiments/preprocessing/preprocess_uci.py

TODO: Implement UCI preprocessing pipeline. The preprocessed data files
      are included in data/preprocessed/uci/ for convenience.
"""

import numpy as np
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'raw')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'uci')

D0 = 7  # pilot days

if __name__ == '__main__':
    # TODO: Port UCI preprocessing from the original pipeline.
    # Preprocessed data is already available in data/preprocessed/uci/.
    print("UCI preprocessing not yet implemented.")
    print("Preprocessed data is available in data/preprocessed/uci/.")
