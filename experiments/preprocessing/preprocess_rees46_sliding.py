"""
preprocess_rees46_sliding.py
============================
Sliding-window preprocessor for REES46.
Creates overlapping experiment windows with D0=5 pilot days, D1=9 follow-up days,
sliding by 3 days.

Reads raw CSV files from data/raw/rees46/ and outputs preprocessed metadata
to data/preprocessed/rees46/.

Usage:
    python experiments/preprocessing/preprocess_rees46_sliding.py

TODO: Implement sliding-window logic (see preprocess_rees46_rolling.py for
      the rolling-window variant which is the primary pipeline).
"""

import numpy as np
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'raw', 'rees46')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'rees46')

D0 = 5
D1 = 9
SLIDE = 3

if __name__ == '__main__':
    # TODO: Port sliding-window preprocessing from the original pipeline.
    # The rolling-window variant (preprocess_rees46_rolling.py) is the primary
    # preprocessing script used for the paper results.
    print("Sliding-window preprocessing not yet implemented.")
    print("Use preprocess_rees46_rolling.py for the primary pipeline.")
