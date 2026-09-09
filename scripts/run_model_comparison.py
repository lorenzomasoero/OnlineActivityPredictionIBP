#!/usr/bin/env python3
"""Run and plot the Be-SSP versus TG-SSP model-comparison experiment.

The bundled Julia program performs the simulation and writes two CSV files.
This Python entry point invokes it without a shell, reads those files, and
writes ``comparison_accuracy.pdf`` to the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from activity_prediction.experiments import prediction_accuracy
from activity_prediction.plotting.style import PAPER_RC, save_figure


DEFAULT_ENTRYPOINT = (
    Path(__file__).resolve().parents[1] / "julia" / "model_comparison.jl"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for raw CSV files and comparison_accuracy.pdf",
    )
    parser.add_argument(
        "--julia",
        default="julia",
        help="Julia executable (default: julia from PATH)",
    )
    parser.add_argument(
        "--julia-entrypoint",
        type=Path,
        default=DEFAULT_ENTRYPOINT,
        help="model_comparison.jl to run (default: bundled entry point)",
    )
    return parser


def _read_results(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"alpha", "be_ssp", "tg_ssp", "oracle", "true_n"}
    if not rows:
        raise ValueError(f"Julia produced an empty result file: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in required
    }


def _accuracy(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            prediction_accuracy(truth, estimate)
            for estimate, truth in zip(predicted, observed)
        ],
        dtype=float,
    )


def _plot(
    be_generated: dict[str, np.ndarray],
    tg_generated: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    methods = (
        ("Oracle", "oracle", "#4d4d4d"),
        ("Be-SSP", "be_ssp", "#e41a1c"),
        ("TG-SSP", "tg_ssp", "#ff7f00"),
    )
    datasets = (be_generated, tg_generated)
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(1, 4, figsize=(12.2, 3.0))
        for data_index, data in enumerate(datasets):
            for label, column, color in methods:
                absolute_error = np.abs(data[column] - data["true_n"])
                axes[data_index].scatter(
                    data["alpha"], absolute_error, s=15, alpha=0.35,
                    color=color, label=label,
                )
                axes[data_index + 2].scatter(
                    data["alpha"], _accuracy(data[column], data["true_n"]),
                    s=15, alpha=0.35, color=color, label=label,
                )

        titles = (
            "Absolute error — Be-SSP data",
            "Absolute error — TG-SSP data",
            "Prediction accuracy — Be-SSP data",
            "Prediction accuracy — TG-SSP data",
        )
        for axis, title in zip(axes, titles):
            axis.set_xlabel(r"$\alpha$")
            axis.set_title(title)
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Absolute error")
        axes[2].set_ylabel(r"Prediction accuracy $v$")
        axes[2].set_ylim(-0.02, 1.02)
        axes[3].set_ylim(-0.02, 1.02)
        handles, labels = axes[2].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0.14, 1, 1))
        return save_figure(figure, output_path)


def main() -> None:
    args = _parser().parse_args()
    entrypoint = args.julia_entrypoint.resolve()
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Julia entry point not found: {entrypoint}")

    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            args.julia,
            f"--project={entrypoint.parent}",
            str(entrypoint),
            str(results_dir),
        ],
        check=True,
    )

    be_generated = _read_results(results_dir / "comparison_data_from_bern.csv")
    tg_generated = _read_results(results_dir / "comparison_data_from_geom.csv")
    _plot(be_generated, tg_generated, output_dir / "comparison_accuracy.pdf")


if __name__ == "__main__":
    main()
