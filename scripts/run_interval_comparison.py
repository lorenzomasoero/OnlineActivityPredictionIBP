#!/usr/bin/env python3
"""Run and plot the Julia duration-interval comparison experiment.

The bundled Julia program simulates the posterior and trajectory-inversion
intervals. This wrapper invokes it without a shell, parses its CSV vector
columns, and writes ``interval_comparison_length.pdf``.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from activity_prediction.plotting.style import PAPER_RC, save_figure


DEFAULT_ENTRYPOINT = (
    Path(__file__).resolve().parents[1] / "julia" / "interval_comparison.jl"
)
TARGET_MULTIPLIERS = (1.5, 2.0, 5.0)
TAIL_PARAMETERS = (0.8, 1.0, 1.2, 1.4)
FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for raw CSV files and interval_comparison_length.pdf",
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
        help="interval_comparison.jl to run (default: bundled entry point)",
    )
    return parser


def _parse_vector(value: str, *, source: Path, column: str) -> np.ndarray:
    parsed = np.asarray([float(item) for item in FLOAT_PATTERN.findall(value)])
    if parsed.size != len(TARGET_MULTIPLIERS):
        raise ValueError(
            f"expected {len(TARGET_MULTIPLIERS)} values in {column!r} from "
            f"{source}, found {parsed.size}: {value!r}"
        )
    return parsed


def _read_lengths(results_dir: Path) -> dict[float, dict[str, np.ndarray]]:
    collected: defaultdict[float, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"Posterior": [], "Inversion": []}
    )
    paths = [
        results_dir / f"zipf_interval_comparison_tail{tail}.csv"
        for tail in TAIL_PARAMETERS
    ]
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Julia did not produce the expected interval files: "
            + ", ".join(str(path) for path in missing_paths)
        )

    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Julia produced an empty result file: {path}")
        required = {"tail", "post_length", "inverse_length"}
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in rows:
            tail = float(row["tail"])
            collected[tail]["Posterior"].append(
                _parse_vector(row["post_length"], source=path, column="post_length")
            )
            collected[tail]["Inversion"].append(
                _parse_vector(row["inverse_length"], source=path, column="inverse_length")
            )

    return {
        tail: {method: np.vstack(values) for method, values in methods.items()}
        for tail, methods in collected.items()
    }


def _plot(lengths: dict[float, dict[str, np.ndarray]], output_path: Path) -> Path:
    tails = sorted(lengths)
    colors = {"Posterior": "#377eb8", "Inversion": "#ff7f00"}
    offsets = {"Posterior": -0.18, "Inversion": 0.18}
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), sharey=False)
        for target_index, (axis, multiplier) in enumerate(
            zip(axes, TARGET_MULTIPLIERS)
        ):
            for method in ("Posterior", "Inversion"):
                samples = [lengths[tail][method][:, target_index] for tail in tails]
                positions = np.arange(len(tails), dtype=float) + offsets[method]
                boxes = axis.boxplot(
                    samples,
                    positions=positions,
                    widths=0.31,
                    patch_artist=True,
                    showfliers=True,
                    flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35},
                    medianprops={"color": "black", "linewidth": 1.3},
                )
                for box in boxes["boxes"]:
                    box.set_facecolor(colors[method])
                    box.set_alpha(0.5)
            axis.set_yscale("log")
            axis.set_xticks(np.arange(len(tails)))
            axis.set_xticklabels([f"{tail:g}" for tail in tails])
            axis.set_xlabel(r"Zipf exponent $\tau$")
            axis.set_title(rf"$M={multiplier:g}N_D$")
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Interval length (days)")
        figure.legend(
            handles=[Patch(facecolor=color, alpha=0.5, label=method) for method, color in colors.items()],
            loc="lower center",
            ncol=2,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.15, 1, 1))
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

    lengths = _read_lengths(results_dir)
    _plot(lengths, output_dir / "interval_comparison_length.pdf")


if __name__ == "__main__":
    main()
