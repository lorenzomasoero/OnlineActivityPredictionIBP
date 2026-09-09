# Reproducibility

Lorenzo Masoero (masoerl) — last updated 2026-09-04

This package reproduces the non-proprietary results of the AoAS paper "Online
activity prediction via generalized Indian buffet process models". It is a single
code package (models, experiment runners, plotting, and reproducibility tooling);
it is not organized by author.

## Standard

- **No hand-typed numbers in the paper.** Every number and table cell is read from
  a committed CSV; every figure is produced by a committed script reading committed
  data.
- Each experiment runner writes both its raw result file and a **tidy CSV** of the
  paper-facing numbers under `output/`. `scripts/gen_paper_values.py` turns those
  CSVs into `latex/values/generated_values.tex`, which the manuscript `\input`s.
- Rounding happens in LaTeX, not in the CSV; the CSV keeps full precision.

## How to reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # requires Python >= 3.10
pip install -e .                     # core deps: numpy, scipy, matplotlib
# optional extras: '.[hbg,unseen,test]'; Julia env for the two simulation figures
make repro                           # regenerate CSVs + figures from committed public data
```

See `REPRODUCE.md` for the full step-by-step reproduction guide (verified install,
per-experiment commands, and the numbers-to-source map).

See `README.md` for the per-experiment script → figure map and per-dataset input
notes, and `data/README.md` for the prepared public inputs.

## Data availability / privacy

- **Public data** (REES46, UCI, ASOS): small prepared inputs are bundled under
  `data/`; raw downloads and preprocessing are documented, not shipped.
- **Proprietary data:** the 1,774-experiment proprietary dataset **cannot be
  released** — see `data/proprietary/README.md`. That experiment is **not
  reproducible end-to-end**; we ship only a synthetic mock input (so the code path
  runs) and the privacy-safe aggregate result CSV behind the paper's numbers. This
  is stated as a known limitation in the paper.

## Provenance

Each CSV is committed before it is cited; scripts and their outputs are re-run and
re-committed together. The camera-ready is pinned to a tagged commit, whose hash is
recorded here and in the paper at freeze time.
