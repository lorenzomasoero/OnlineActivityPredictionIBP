"""
draw_from_zipf.py
=================
Draws synthetic datasets from Zipf-distributed activity rates.
Used for power-law simulation studies.

Generates synthetic user activity data where individual activity rates
follow a Zipf (power-law) distribution, then evaluates how well SSP models
recover the true number of new users.

Usage:
    python experiments/synthetic/draw_from_zipf.py

TODO: Port Zipf drawing code from the original codebase.
"""

import numpy as np
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

if __name__ == '__main__':
    # TODO: Implement Zipf-based synthetic data generation.
    print("Zipf synthetic data generation not yet implemented.")
