#!/usr/bin/env python3
"""Fit the REES46 experiment windows and write their manuscript figures.

Inputs are the two prepared, public REES46 artifacts produced from the raw
monthly event CSV files:

* ``--experiments`` is a ``.npy`` sequence (or mapping) of non-overlapping
  experiment records. Each record must contain ``cumulative_users`` including
  day zero, ``D0``, and ``D1``. The paper uses seven 28-day windows with
  ``D0=7`` and ``D1=21``.
* ``--user-stats`` is the ``.npy`` mapping with ``n_users``,
  ``triggers_per_user_histogram``, and ``days_active_histogram`` computed over
  all public REES46 months.

Only cumulative first-trigger curves are available for these large windows,
so TG-SSP, IBP, and NB-SSP are fit with the paper's curve criteria. The script
writes the fitted results and all REES46-specific active manuscript figures to
``output/rees46`` by default. The two inputs default to the bundled prepared
artifacts below ``data/rees46``; all three paths can be overridden.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.stats import poisson

from activity_prediction.experiments import prediction_accuracy
from activity_prediction.models import IBP, NBSSP, TGSSP
from activity_prediction.plotting import (
    plot_rees46_cumulative_users,
    plot_rees46_days_active,
    plot_rees46_triggers_per_user,
    plot_trajectory_panels,
)
from activity_prediction.plotting.style import load_plot_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "rees46"
DEFAULT_EXPERIMENTS = DEFAULT_DATA_DIR / "experiments_metadata.npy"
DEFAULT_USER_STATS = DEFAULT_DATA_DIR / "rees46_user_stats.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "rees46"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiments",
        type=Path,
        default=DEFAULT_EXPERIMENTS,
        help="prepared REES46 experiment metadata (.npy, .npz, or .json; "
        "default: %(default)s)",
    )
    parser.add_argument(
        "--user-stats",
        type=Path,
        default=DEFAULT_USER_STATS,
        help="prepared REES46 per-user histogram mapping (.npy or .npz; "
        "default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for rees46_results.npy and the paper PDFs "
        "(default: %(default)s)",
    )
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--selected-windows",
        type=int,
        nargs=3,
        default=(0, 3, 6),
        metavar=("FIRST", "MIDDLE", "LAST"),
        help="zero-based record positions shown in the cumulative-user figure",
    )
    return parser


def _records(data: Any) -> list[dict[str, Any]]:
    value = load_plot_data(data)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping) and "experiments" in value:
        value = value["experiments"]

    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if "cumulative_users" in value:
            value = [value]
        else:
            for key, record in value.items():
                if not isinstance(record, Mapping):
                    raise ValueError("every experiment value must be a mapping")
                copied = dict(record)
                copied.setdefault("exp_id", key)
                records.append(copied)
            value = None
    if value is not None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("experiments must contain a sequence or mapping of records")
        for index, record in enumerate(value):
            if not isinstance(record, Mapping):
                raise ValueError("every experiment must be a mapping")
            copied = dict(record)
            copied.setdefault("exp_id", index)
            records.append(copied)
    if not records:
        raise ValueError("experiments contains no records")
    return [_validate_record(record) for record in records]


def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
    curve = np.asarray(record.get("cumulative_users"), dtype=float)
    if (
        curve.ndim != 1
        or curve.size < 3
        or np.any(~np.isfinite(curve))
        or np.any(curve < 0)
        or np.any(np.diff(curve) < 0)
        or np.any(curve != np.floor(curve))
        or curve[0] != 0
    ):
        raise ValueError(
            f"experiment {record.get('exp_id')!r} has an invalid cumulative_users curve"
        )
    pilot_days = int(record.get("D0", -1))
    followup_days = int(record.get("D1", -1))
    if pilot_days < 2 or followup_days < 1 or curve.size < pilot_days + followup_days + 1:
        raise ValueError(
            f"experiment {record.get('exp_id')!r} must have D0>=2, D1>=1, "
            "and a complete observed curve"
        )
    record["cumulative_users"] = curve[: pilot_days + followup_days + 1].astype(int)
    record["D0"] = pilot_days
    record["D1"] = followup_days
    record["N_pilot"] = int(curve[pilot_days])
    record["N_total"] = int(curve[pilot_days + followup_days])
    record["U_true"] = record["N_total"] - record["N_pilot"]
    if record["N_pilot"] <= 0 or record["U_true"] <= 0:
        raise ValueError(
            f"experiment {record.get('exp_id')!r} must observe users in both periods"
        )
    return record


def _prediction(
    parameters: dict[str, float],
    expected: np.ndarray,
    interval: tuple[np.ndarray, np.ndarray],
    truth: int,
) -> dict[str, object]:
    lower, upper = (np.asarray(endpoint, dtype=float) for endpoint in interval)
    final_expected = float(np.asarray(expected, dtype=float)[-1])
    return {
        "parameters": parameters,
        "expected_new_users": np.asarray(expected, dtype=float),
        "new_user_interval": (lower, upper),
        "accuracy_v": prediction_accuracy(truth, final_expected),
        "ci_covers": bool(lower[-1] <= truth <= upper[-1]),
    }


def _fit_record(
    record: Mapping[str, Any],
    *,
    coverage: float,
    num_restarts: int,
    seed: int | None,
) -> dict[str, Any]:
    pilot_days = int(record["D0"])
    followup_days = int(record["D1"])
    observed_users = int(record["N_pilot"])
    truth = int(record["U_true"])
    pilot_curve = np.asarray(record["cumulative_users"], dtype=int)[: pilot_days + 1]

    tg = TGSSP()
    tg_parameters = tg.fit_curve(
        pilot_curve, num_restarts=num_restarts, seed=seed
    )
    tg_expected = tg.expected_new_users(
        pilot_days, followup_days, observed_users, **tg_parameters
    )

    ibp = IBP()
    ibp_parameters = ibp.fit_curve(
        pilot_curve, num_restarts=num_restarts, seed=seed
    )
    ibp_expected = ibp.expected_new_users(
        pilot_days, followup_days, **ibp_parameters
    )

    nb = NBSSP()
    nb_parameters = nb.fit_curve(
        pilot_curve, num_restarts=num_restarts, seed=seed
    )
    nb_expected = nb.expected_new_users(
        pilot_days, followup_days, observed_users, **nb_parameters
    )

    fitted = dict(record)
    fitted["TG-SSP"] = _prediction(
        tg_parameters,
        tg_expected,
        tg.new_user_interval(
            pilot_days,
            followup_days,
            observed_users,
            coverage=coverage,
            **tg_parameters,
        ),
        truth,
    )
    fitted["IBP"] = _prediction(
        ibp_parameters,
        ibp_expected,
        poisson.interval(coverage, ibp_expected),
        truth,
    )
    fitted["NB-SSP (curve)"] = _prediction(
        nb_parameters,
        nb_expected,
        nb.new_user_interval(
            pilot_days,
            followup_days,
            observed_users,
            coverage=coverage,
            **nb_parameters,
        ),
        truth,
    )
    return fitted


def main() -> None:
    args = _parser().parse_args()
    if args.num_restarts < 1:
        raise ValueError("--num-restarts must be positive")
    if not 0 < args.coverage < 1:
        raise ValueError("--coverage must lie between zero and one")

    experiments = _records(args.experiments)
    selected = tuple(args.selected_windows)
    if any(index < 0 or index >= len(experiments) for index in selected):
        raise IndexError(
            f"--selected-windows must index the {len(experiments)} input records"
        )
    pilot_days = {record["D0"] for record in experiments}
    if len(pilot_days) != 1:
        raise ValueError("the cumulative-user figure requires a common D0")

    results: dict[object, dict[str, Any]] = {}
    for index, record in enumerate(experiments):
        seed = None if args.seed is None else args.seed + index
        results[record["exp_id"]] = _fit_record(
            record,
            coverage=args.coverage,
            num_restarts=args.num_restarts,
            seed=seed,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "rees46_results.npy", results, allow_pickle=True)

    plot_rees46_triggers_per_user(
        args.user_stats, output_dir / "fig_rees46_triggers_per_user.pdf"
    )
    plot_rees46_days_active(
        args.user_stats, output_dir / "fig_rees46_days_active.pdf"
    )
    plot_rees46_cumulative_users(
        experiments,
        output_dir / "fig_rees46_cumulative_users.pdf",
        selected=selected,
        pilot_days=pilot_days.pop(),
    )
    plot_trajectory_panels(
        results,
        output_dir / "fig_appendix_rees46_accumulation.pdf",
        experiment_ids=list(results),
        methods=("TG-SSP", "IBP", "NB-SSP"),
    )


if __name__ == "__main__":
    main()
