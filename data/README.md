# Data

This directory contains preprocessed experiment data and instructions for
obtaining the raw datasets.

## Preprocessed Data (included)

The `preprocessed/` directory contains ready-to-use experiment metadata and
matrices. These are sufficient to run all fitting and evaluation scripts.

- **`preprocessed/uci/`** — UCI Online Retail II experiment windows (13 experiments,
  D0=7 pilot days, D1=21 follow-up days). Includes trigger matrices and metadata.
- **`preprocessed/rees46/`** — REES46 eCommerce rolling-window experiments
  (k=21, 50, 100 day horizons). Metadata only (no raw user data).
- **`preprocessed/asos/`** — ASOS treatment arm first-trigger counts.

## Raw Data Download Instructions

Raw data is **not** included in this repository due to size. Download it into
`data/raw/` if you want to re-run preprocessing from scratch.

### REES46 eCommerce Behavior Data

Source: [Kaggle — eCommerce behavior data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

Direct download (hosted by data provider):

```bash
mkdir -p data/raw/rees46
cd data/raw/rees46

# All 7 months (Oct 2019 – Apr 2020)
wget https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz
wget https://data.rees46.com/datasets/marketplace/2019-Nov.csv.gz
wget https://data.rees46.com/datasets/marketplace/2019-Dec.csv.gz
wget https://data.rees46.com/datasets/marketplace/2020-Jan.csv.gz
wget https://data.rees46.com/datasets/marketplace/2020-Feb.csv.gz
wget https://data.rees46.com/datasets/marketplace/2020-Mar.csv.gz
wget https://data.rees46.com/datasets/marketplace/2020-Apr.csv.gz

# Decompress
gunzip *.csv.gz
```

Then run preprocessing:

```bash
python experiments/preprocessing/preprocess_rees46_rolling.py
```

**Note:** Raw CSVs are ~5–8 GB each (~40 GB total uncompressed). You can delete
them after preprocessing — only the small `.npy` metadata files are needed.

### UCI Online Retail II

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

Download the Excel file and place it in `data/raw/`:

```bash
mkdir -p data/raw
cd data/raw
wget https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip online+retail+ii.zip
```

Then run preprocessing:

```bash
python experiments/preprocessing/preprocess_uci.py
```

### ASOS

The ASOS dataset is not publicly available. It was provided by the original
authors of:

> Liu, Y., Mao, J., & Bayesian, B. (2021). *Predicting online activity using
> Bayesian nonparametric models.* Working paper.

To obtain access, contact the original authors directly.
