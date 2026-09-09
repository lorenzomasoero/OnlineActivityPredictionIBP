#!/usr/bin/env python3
"""Run the UCI Online Retail experiment and render its paper figures.

The input is the preprocessed collection of non-overlapping UCI experiment
windows: one metadata file plus one days-by-users matrix per window.  The
paper comparison uses curve fitting for TG-SSP, the three-parameter IBP, and
NB-SSP.  Maximum-marginal-likelihood fits and the remaining benchmarks are
also retained in ``uci_results.npy`` under ``all_fits``.

With no path arguments, the script reads the bundled prepared UCI data and
writes results below ``output/uci`` in the project directory.

Outputs written below ``--output-dir``:

* ``uci_results.npy`` -- fitted models, predictions, and accuracy metrics;
* ``fig_case_study_uci.pdf`` -- the three UCI case-study trajectories;
* ``fig_appendix_uci_accumulation.pdf`` -- trajectories for every window.

This module has no work at import time.  Running it performs the fits.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np

from activity_prediction.experiments import (
    fit_daily_activity_models,
    prediction_accuracy,
)
from activity_prediction.plotting import plot_trajectory_panels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "uci"
DEFAULT_METADATA = DEFAULT_DATA_DIR / "experiments_metadata.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "uci"

PAPER_FIT_SOURCE = {
    "TG-SSP": "TG-SSP (curve)",
    "IBP": "IBP (curve)",
    "NB-SSP": "NB-SSP (curve)",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="preprocessed .npy file containing the UCI experiment records "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--matrices-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing one full activity matrix per experiment "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory in which results and figures will be written "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--matrix-pattern",
        default="matrix_exp_{exp_id}.npy",
        help="matrix filename pattern; it must contain the field {exp_id}",
    )
    parser.add_argument("--pilot-days", type=int, default=7)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--case-study-ids",
        nargs="*",
        help="experiment IDs for the case-study plot (paper default: 2 6 12)",
    )
    parser.add_argument("--include-hbg", action="store_true")
    parser.add_argument("--include-lp", action="store_true")
    return parser


def _load_metadata(path: Path) -> list[Mapping[str, Any]]:
    loaded = np.load(path, allow_pickle=True)
    value: Any = loaded.item() if loaded.shape == () else loaded.tolist()
    if isinstance(value, Mapping) and "experiments" in value:
        value = value["experiments"]
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("metadata must contain a sequence of experiment records")
    records = list(value)
    if not records or not all(isinstance(record, Mapping) for record in records):
        raise ValueError("metadata must contain at least one mapping record")
    return records


def _activity_matrix(path: Path) -> np.ndarray:
    matrix = np.load(path, allow_pickle=False)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"{path} must contain a nonempty days-by-users matrix")
    if not np.issubdtype(matrix.dtype, np.number) or np.any(~np.isfinite(matrix)):
        raise ValueError(f"{path} must contain finite numeric counts")
    if np.any(matrix < 0) or np.any(matrix != np.floor(matrix)):
        raise ValueError(f"{path} must contain nonnegative integer counts")
    return matrix.astype(int, copy=False)


def _cumulative_users(matrix: np.ndarray) -> np.ndarray:
    active = np.cumsum(matrix > 0, axis=0) > 0
    return np.concatenate(([0], np.count_nonzero(active, axis=1)))


def _new_user_trajectory(
    fit: Mapping[str, Any],
    *,
    pilot_days: int,
    followup_days: int,
    observed_users: int,
) -> np.ndarray | None:
    if "expected_new_users" in fit:
        trajectory = np.asarray(fit["expected_new_users"], dtype=float).reshape(-1)
        return trajectory[:followup_days]
    if "cumulative_users" in fit:
        cumulative = np.asarray(fit["cumulative_users"], dtype=float).reshape(-1)
        stop = pilot_days + followup_days + 1
        if len(cumulative) >= stop:
            return cumulative[pilot_days + 1 : stop] - observed_users
    return None


def _evaluate_fits(
    fits: Mapping[str, Mapping[str, Any]],
    *,
    pilot_days: int,
    followup_days: int,
    observed_users: int,
    true_new_users: int,
    true_followup_triggers: int,
) -> dict[str, dict[str, Any]]:
    evaluated: dict[str, dict[str, Any]] = {}
    for name, source in fits.items():
        record = dict(source)
        trajectory = _new_user_trajectory(
            record,
            pilot_days=pilot_days,
            followup_days=followup_days,
            observed_users=observed_users,
        )
        if trajectory is not None and len(trajectory):
            estimate = float(trajectory[-1])
            record["expected_new_users"] = trajectory
            record["U_hat"] = estimate
            record["accuracy_v"] = prediction_accuracy(true_new_users, estimate)

            interval = record.get("new_user_interval")
            if interval is not None:
                lower, upper = (
                    np.asarray(endpoint, dtype=float).reshape(-1)
                    for endpoint in interval
                )
                if len(lower) and len(upper):
                    record["U_ci"] = (float(lower[-1]), float(upper[-1]))
                    record["ci_covers"] = bool(
                        lower[-1] <= true_new_users <= upper[-1]
                    )

        trigger_trajectory = record.get("expected_total_triggers")
        if trigger_trajectory is not None:
            trigger_trajectory = np.asarray(trigger_trajectory, dtype=float).reshape(-1)
            if len(trigger_trajectory):
                trigger_estimate = float(trigger_trajectory[-1])
                record["T_hat"] = trigger_estimate
                record["trigger_accuracy_v"] = prediction_accuracy(
                    true_followup_triggers, trigger_estimate
                )
        evaluated[name] = record
    return evaluated


def _paper_case_ids(
    results: Mapping[object, Mapping[str, Any]], requested: Sequence[str] | None
) -> list[object]:
    by_string = {str(key): key for key in results}
    if requested:
        missing = [value for value in requested if value not in by_string]
        if missing:
            raise KeyError(f"unknown UCI experiment IDs: {', '.join(missing)}")
        return [by_string[value] for value in requested]

    historical = [by_string[value] for value in ("2", "6", "12") if value in by_string]
    if len(historical) == 3:
        return historical

    ordered = sorted(results, key=lambda key: float(results[key]["N_pilot"]))
    indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
    return [ordered[index] for index in indices]


def main() -> None:
    args = _parser().parse_args()
    if args.pilot_days < 2:
        raise ValueError("--pilot-days must be at least two")
    if args.num_restarts < 1:
        raise ValueError("--num-restarts must be positive")
    if not 0 < args.coverage < 1:
        raise ValueError("--coverage must lie in (0, 1)")
    if "{exp_id}" not in args.matrix_pattern:
        raise ValueError("--matrix-pattern must contain {exp_id}")

    metadata = _load_metadata(args.metadata)
    results: dict[object, dict[str, Any]] = {}
    for index, experiment in enumerate(metadata):
        if "exp_id" not in experiment:
            raise ValueError(f"metadata record {index} has no exp_id")
        experiment_id = experiment["exp_id"]
        if isinstance(experiment_id, np.generic):
            experiment_id = experiment_id.item()
        matrix_path = args.matrices_dir / args.matrix_pattern.format(
            exp_id=experiment_id
        )
        matrix = _activity_matrix(matrix_path)
        if matrix.shape[0] <= args.pilot_days:
            raise ValueError(
                f"experiment {experiment_id!r} has no days after its pilot"
            )

        followup_days = matrix.shape[0] - args.pilot_days
        curve = _cumulative_users(matrix)
        observed_users = int(curve[args.pilot_days])
        true_new_users = int(curve[-1] - observed_users)
        true_followup_triggers = int(matrix[args.pilot_days :].sum())
        if observed_users <= 0 or true_new_users <= 0 or true_followup_triggers <= 0:
            raise ValueError(
                f"experiment {experiment_id!r} has an empty pilot or follow-up"
            )

        fit_seed = None if args.seed is None else args.seed + index
        raw_fits = fit_daily_activity_models(
            matrix[: args.pilot_days],
            followup_days,
            coverage=args.coverage,
            num_restarts=args.num_restarts,
            seed=fit_seed,
            include_hbg=args.include_hbg,
            include_lp=args.include_lp,
        )
        all_fits = _evaluate_fits(
            raw_fits,
            pilot_days=args.pilot_days,
            followup_days=followup_days,
            observed_users=observed_users,
            true_new_users=true_new_users,
            true_followup_triggers=true_followup_triggers,
        )

        record: dict[str, Any] = {
            "exp_id": experiment_id,
            "D0": args.pilot_days,
            "D1": followup_days,
            "N_pilot": observed_users,
            "U_true": true_new_users,
            "T_true": true_followup_triggers,
            "cumulative_users": curve,
            "paper_fit_strategy": dict(PAPER_FIT_SOURCE),
            "all_fits": all_fits,
        }
        for display_name, source_name in PAPER_FIT_SOURCE.items():
            record[display_name] = dict(all_fits[source_name])
        results[experiment_id] = record

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "uci_results.npy"
    np.save(result_path, results, allow_pickle=True)

    methods = tuple(PAPER_FIT_SOURCE)
    plot_trajectory_panels(
        results,
        args.output_dir / "fig_case_study_uci.pdf",
        experiment_ids=_paper_case_ids(results, args.case_study_ids),
        methods=methods,
    )
    plot_trajectory_panels(
        results,
        args.output_dir / "fig_appendix_uci_accumulation.pdf",
        methods=methods,
    )

    print(f"Saved {result_path}")
    for method in methods:
        values = [record[method]["accuracy_v"] for record in results.values()]
        print(
            f"{method}: median v={np.median(values):.3f}, "
            f"mean v={np.mean(values):.3f}, n={len(values)}"
        )


if __name__ == "__main__":
    main()
