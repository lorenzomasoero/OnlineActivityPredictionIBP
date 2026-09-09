#!/usr/bin/env python3
"""Run the ASOS first-trigger experiment and render its paper figures.

ASOS exposes cumulative first-trigger curves, but not per-user re-trigger
counts.  The paper therefore uses the TG-SSP geometric marginal likelihood
and curve fits for the three-parameter IBP and NB-SSP.  Every raw fit remains
available in ``asos_results.npy`` under ``all_fits``.

With no path arguments, the script reads the bundled prepared ASOS data and
writes results below ``output/asos`` in the project directory.

Outputs written below ``--output-dir``:

* ``asos_results.npy`` -- fits, trajectories, intervals, and accuracy metrics;
* ``PAPER_ASOS_accuracy_v2.pdf`` -- the main-text accuracy comparison;
* ``fig_appendix_asos_selected.pdf`` -- the stratified per-arm appendix plot.

This module has no work at import time.  Running it performs the fits.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from activity_prediction.experiments import (
    fit_first_trigger_models,
    prediction_accuracy,
)
from activity_prediction.plotting import plot_accuracy_boxplot
from activity_prediction.plotting.style import COLORS, PAPER_RC, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "asos" / "PAPER_asos_data.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "asos"

PAPER_FIT_SOURCE = {
    "TG-SSP": "TG-SSP",
    "IBP": "IBP",
    "NB-SSP": "NB-SSP (curve)",
}
PANEL_METHODS = ("TG-SSP", "IBP", "NB-SSP")
METHOD_SHORT = {"TG-SSP": "TG", "IBP": "IBP", "NB-SSP": "NB"}
RATIO_BINS = ((0, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 300))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="preprocessed ASOS .npy dictionary (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory in which results and figures will be written "
        "(default: %(default)s)",
    )
    parser.add_argument("--pilot-days", type=int, default=7)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--selected-per-bin",
        type=int,
        default=3,
        help="number of appendix arms selected within each extrapolation bin",
    )
    return parser


def _load_asos(path: Path) -> Mapping[object, Mapping[str, Any]]:
    loaded = np.load(path, allow_pickle=True)
    value: Any = loaded.item() if loaded.shape == () else loaded.tolist()
    if not isinstance(value, Mapping) or not value:
        raise ValueError("ASOS input must contain a nonempty experiment mapping")
    if not all(isinstance(record, Mapping) for record in value.values()):
        raise ValueError("each ASOS experiment must be a mapping")
    return value


def _curve(values: Any, *, name: str) -> np.ndarray:
    curve = np.asarray(values)
    if curve.ndim != 1 or len(curve) < 2:
        raise ValueError(f"{name} must be a one-dimensional cumulative curve")
    if not np.issubdtype(curve.dtype, np.number) or np.any(~np.isfinite(curve)):
        raise ValueError(f"{name} must contain finite numeric counts")
    if np.any(curve < 0) or np.any(curve != np.floor(curve)):
        raise ValueError(f"{name} must contain nonnegative integer counts")
    if curve[0] != 0:
        raise ValueError(f"{name} must start at zero")
    # Cumulative first-trigger curves must be nondecreasing. A few ASOS arms
    # (e.g. b3280a/T: a single -5,516 dip at index 39 of a ~3.07M curve) contain
    # tiny non-monotone glitches from measurement noise. Enforce monotonicity via
    # a running max rather than dropping the arm; this matches the original
    # analysis (which retained all arms) and shifts the curve negligibly.
    curve = np.maximum.accumulate(curve)
    return curve.astype(int, copy=False)


def _evaluate_fit(
    source: Mapping[str, Any], *, true_new_users: int
) -> dict[str, Any]:
    record = dict(source)
    trajectory = np.asarray(record["expected_new_users"], dtype=float).reshape(-1)
    if not len(trajectory):
        raise ValueError("model returned an empty follow-up trajectory")
    estimate = float(trajectory[-1])
    record["expected_new_users"] = trajectory
    record["U_hat"] = estimate
    record["accuracy_v"] = prediction_accuracy(true_new_users, estimate)

    interval = record.get("new_user_interval")
    if interval is not None:
        lower, upper = (
            np.asarray(endpoint, dtype=float).reshape(-1) for endpoint in interval
        )
        if len(lower) and len(upper):
            record["U_ci"] = (float(lower[-1]), float(upper[-1]))
            record["ci_covers"] = bool(lower[-1] <= true_new_users <= upper[-1])
    return record


def _quantile_picks(
    records: list[tuple[object, Mapping[str, Any], float]], count: int
) -> list[tuple[object, Mapping[str, Any], float]]:
    ordered = sorted(records, key=lambda item: item[1]["NB-SSP"]["accuracy_v"])
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[index] for index in dict.fromkeys(indices)]


def _selected_arms(
    results: Mapping[object, Mapping[str, Any]], count_per_bin: int
) -> list[tuple[object, Mapping[str, Any], float]]:
    available = [
        (key, record, float(record["D1"]) / float(record["D0"]))
        for key, record in results.items()
        if all(
            method in record and "accuracy_v" in record[method]
            for method in PANEL_METHODS
        )
    ]
    selected: list[tuple[object, Mapping[str, Any], float]] = []
    for lower, upper in RATIO_BINS:
        bucket = [item for item in available if lower < item[2] <= upper]
        selected.extend(_quantile_picks(bucket, count_per_bin))
    return selected


def _draw_accuracy_panel(
    axis: plt.Axes, record: Mapping[str, Any], title: str
) -> None:
    values = [float(record[method]["accuracy_v"]) for method in PANEL_METHODS]
    coverages = [record[method].get("ci_covers") for method in PANEL_METHODS]
    x_positions = np.arange(len(PANEL_METHODS))
    axis.bar(
        x_positions,
        values,
        color=[COLORS[method] for method in PANEL_METHODS],
        alpha=0.75,
        width=0.6,
        edgecolor="white",
    )
    axis.axhline(0.8, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
    axis.set_ylim(0, 1.12)
    axis.set_xticks(x_positions, [METHOD_SHORT[method] for method in PANEL_METHODS])
    axis.tick_params(axis="both", labelsize=7)
    axis.set_title(title, fontsize=8, pad=2)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)

    for position, (accuracy, covers) in enumerate(zip(values, coverages, strict=True)):
        axis.text(
            position,
            accuracy + 0.02,
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="#333333",
        )
        if covers is not None:
            axis.plot(
                position,
                1.06,
                "o",
                color="#2ca02c" if covers else "#d62728",
                markersize=4,
                clip_on=False,
            )


def _plot_selected_accuracy(
    results: Mapping[object, Mapping[str, Any]],
    output_path: Path,
    *,
    count_per_bin: int,
) -> Path:
    selected = _selected_arms(results, count_per_bin)
    if not selected:
        raise ValueError("no ASOS arms have all three paper-model fits")

    columns = min(6, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(2.5 * columns, 2.6 * rows),
            squeeze=False,
        )
        for axis, (key, record, ratio) in zip(
            axes.flat, selected, strict=False
        ):
            _draw_accuracy_panel(
                axis,
                record,
                f"{key}\n$D_0$={record['D0']} $D_1$={record['D1']} "
                f"($\\times${ratio:.1f})",
            )
        for axis in axes.flat[len(selected) :]:
            axis.set_visible(False)

        figure.text(
            0.01,
            0.5,
            r"Accuracy $v_{D_0}^{(D_1)}$",
            va="center",
            rotation="vertical",
        )
        method_handles = [
            Patch(color=COLORS[method], alpha=0.75, label=method)
            for method in PANEL_METHODS
        ]
        coverage_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                markerfacecolor=color,
                markersize=6,
                label=label,
            )
            for color, label in (
                ("#2ca02c", "CI covers"),
                ("#d62728", "CI misses"),
            )
        ]
        figure.legend(
            handles=method_handles + coverage_handles,
            loc="lower center",
            ncol=5,
            frameon=False,
        )
        figure.suptitle(
            f"ASOS — selected arms ({len(selected)} of {len(results)}), "
            "stratified by extrapolation ratio",
            y=1.01,
        )
        figure.tight_layout(rect=(0.03, 0.05, 1, 1))
        return save_figure(figure, output_path)


def main() -> None:
    args = _parser().parse_args()
    if args.pilot_days < 2:
        raise ValueError("--pilot-days must be at least two")
    if args.num_restarts < 1:
        raise ValueError("--num-restarts must be positive")
    if args.selected_per_bin < 1:
        raise ValueError("--selected-per-bin must be positive")
    if not 0 < args.coverage < 1:
        raise ValueError("--coverage must lie in (0, 1)")

    source = _load_asos(args.input)
    results: dict[object, dict[str, Any]] = {}
    arm_index = 0
    for experiment_id, experiment in source.items():
        short_by_arm = experiment.get("first_trigger_counts_short")
        long_by_arm = experiment.get("first_trigger_counts_long")
        if not isinstance(short_by_arm, Mapping) or not isinstance(long_by_arm, Mapping):
            raise ValueError(
                f"ASOS experiment {experiment_id!r} has no first-trigger mappings"
            )
        if set(short_by_arm) != set(long_by_arm):
            raise ValueError(
                f"ASOS experiment {experiment_id!r} has inconsistent arm keys"
            )

        for arm in short_by_arm:
            pilot_curve = _curve(
                short_by_arm[arm], name=f"{experiment_id}/{arm} pilot curve"
            )
            full_curve = _curve(
                long_by_arm[arm], name=f"{experiment_id}/{arm} full curve"
            )
            if len(pilot_curve) != args.pilot_days + 1:
                raise ValueError(
                    f"{experiment_id}/{arm} pilot has {len(pilot_curve) - 1} "
                    f"days, expected {args.pilot_days}"
                )
            if len(full_curve) <= len(pilot_curve):
                raise ValueError(f"{experiment_id}/{arm} has no follow-up days")
            if not np.array_equal(full_curve[: len(pilot_curve)], pilot_curve):
                raise ValueError(
                    f"{experiment_id}/{arm} pilot is not a prefix of the full curve"
                )

            followup_days = len(full_curve) - len(pilot_curve)
            observed_users = int(pilot_curve[-1])
            true_new_users = int(full_curve[-1] - observed_users)
            if observed_users <= 0 or true_new_users <= 0:
                raise ValueError(f"{experiment_id}/{arm} has an empty pilot or follow-up")

            fit_seed = None if args.seed is None else args.seed + arm_index
            raw_fits = fit_first_trigger_models(
                pilot_curve,
                followup_days,
                coverage=args.coverage,
                num_restarts=args.num_restarts,
                seed=fit_seed,
            )
            all_fits = {
                name: _evaluate_fit(fit, true_new_users=true_new_users)
                for name, fit in raw_fits.items()
            }

            key = f"{experiment_id}_{arm}"
            record: dict[str, Any] = {
                "exp_id": experiment_id,
                "arm": arm,
                "D0": args.pilot_days,
                "D1": followup_days,
                "N_pilot": observed_users,
                "N_total": int(full_curve[-1]),
                "U_true": true_new_users,
                "cumulative_users": full_curve,
                "paper_fit_strategy": dict(PAPER_FIT_SOURCE),
                "all_fits": all_fits,
            }
            for display_name, source_name in PAPER_FIT_SOURCE.items():
                record[display_name] = dict(all_fits[source_name])
            results[key] = record
            arm_index += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "asos_results.npy"
    np.save(result_path, results, allow_pickle=True)

    accuracy_data = {
        method: [record[method]["accuracy_v"] for record in results.values()]
        for method in ("NB-SSP", "TG-SSP", "IBP")
    }
    plot_accuracy_boxplot(
        accuracy_data,
        args.output_dir / "PAPER_ASOS_accuracy_v2.pdf",
        methods=("NB-SSP", "TG-SSP", "IBP"),
        ylabel=r"Prediction accuracy $v_{7}^{(D_1)}$",
    )
    _plot_selected_accuracy(
        results,
        args.output_dir / "fig_appendix_asos_selected.pdf",
        count_per_bin=args.selected_per_bin,
    )

    print(f"Saved {result_path}")
    for method, values in accuracy_data.items():
        print(
            f"{method}: median v={np.median(values):.3f}, "
            f"mean v={np.mean(values):.3f}, n={len(values)}"
        )


if __name__ == "__main__":
    main()
