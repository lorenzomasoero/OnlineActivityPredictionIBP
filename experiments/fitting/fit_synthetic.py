"""
fit_synthetic.py
================
Fits models on synthetic data drawn from known SSP processes.
Used to validate model recovery and calibration.

Combines synthetic fitting scripts for:
  - TG-SSP synthetic experiments
  - NB-SSP synthetic experiments
  - IBP synthetic experiments

Usage:
    python experiments/fitting/fit_synthetic.py

TODO: Port synthetic fitting pipeline from the original codebase.
"""

import numpy as np
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from models.tg_ssp import GD
from models.ibp import IBP
from models.nb_ssp import NegBintSBSP

if __name__ == '__main__':
    # TODO: Implement synthetic data fitting experiments.
    print("Synthetic fitting not yet implemented.")
