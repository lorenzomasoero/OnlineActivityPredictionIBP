"""
draw_from_model.py
==================
Draws synthetic datasets from SSP generative models (TG-SSP, NB-SSP, IBP).
Used for simulation studies and model validation.

Generates:
  - Synthetic trigger matrices
  - Cumulative user curves
  - Ground-truth parameters for recovery experiments

Usage:
    python experiments/synthetic/draw_from_model.py

TODO: Port synthetic drawing code from the original codebase.
"""

import numpy as np
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

if __name__ == '__main__':
    # TODO: Implement synthetic data generation from SSP models.
    print("Synthetic data generation not yet implemented.")
