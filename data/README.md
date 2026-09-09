# Prepared experiment data

This directory contains the small, prepared inputs used by the three
non-proprietary data experiments. They are included so the corresponding
runners work without a separate download or preprocessing step. The files
are copied byte-for-byte from `AoAS/code/`; no fitting output is bundled.

## Contents

- `asos/PAPER_asos_data.npy`: cumulative first-trigger counts for the ASOS
  experiment arms. The underlying ASOS Digital Experiments Dataset is
  described by Liu et al. (2021) and distributed at
  <https://osf.io/64jsb/>.
- `uci/experiments_metadata.npy` and `uci/matrix_exp_0.npy` through
  `matrix_exp_12.npy`: metadata and anonymous days-by-users activity matrices
  for 13 windows of the UCI Online Retail II data. The raw dataset is
  available from <https://archive.ics.uci.edu/dataset/502/online+retail+ii>.
- `rees46/experiments_metadata.npy`: seven non-overlapping 28-day REES46
  windows used by `run_rees46.py` by default.
- `rees46/experiments_rolling_k100.npy`: rolling windows with a 100-day
  follow-up, used to create the REES46 input for `run_hitting_times.py`.
- `rees46/rees46_user_stats.npy`: aggregate user-count, triggers-per-user, and
  active-days histograms used in the REES46 descriptive figures. REES46 hosts
  the raw monthly event logs at <https://data.rees46.com/datasets/marketplace/>.

NumPy object files are loaded with `allow_pickle=True`; only load replacements
from a trusted source. Raw transaction/event files and the paper's separate
1,774-experiment proprietary dataset are not included.
