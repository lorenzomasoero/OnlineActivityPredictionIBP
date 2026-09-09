#!/usr/bin/env python3
"""Run the public ASOS/REES46 duration-to-target experiment and plot it.

The experiment consumes fitted-result mappings emitted by the ASOS and
REES46 dataset runners. From the ``AoAS_Code`` directory, create the default
prerequisites and then run this script with:

.. code-block:: console

   python scripts/run_asos.py
   python scripts/run_rees46.py --experiments data/rees46/experiments_rolling_k100.npy --output-dir output/rees46_k100
   python scripts/run_hitting_times.py

The default inputs are ``output/asos/asos_results.npy`` and
``output/rees46_k100/rees46_results.npy``. Each experiment record needs
``cumulative_users``, ``D0``, ``D1``, and the fitted ``TG-SSP``, ``NB-SSP``
(or ``NB-SSP (curve)``), and ``IBP`` records. Model records must contain the
named ``parameters`` mapping used by :mod:`activity_prediction`.

The script derives true and predicted hitting days, saves the derived records,
and writes the active manuscript figure ``fig_hitting_time_mae_asos.pdf``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from activity_prediction.experiments import first_hitting_day, model_hitting_day
from activity_prediction.models import predict_jackknife
from activity_prediction.plotting.style import (
    COLORS,
    PAPER_RC,
    canonical_method,
    load_plot_data,
    save_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASOS_RESULTS = PROJECT_ROOT / "output" / "asos" / "asos_results.npy"
DEFAULT_REES_RESULTS = (
    PROJECT_ROOT / "output" / "rees46_k100" / "rees46_results.npy"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "hitting_times"

METHODS = ("NB-SSP", "TG-SSP", "IBP", "Linear", "J3")
METHOD_COLORS = {
    "NB-SSP": COLORS["NB-SSP"],
    "TG-SSP": COLORS["TG-SSP"],
    "IBP": COLORS["IBP"],
    "Linear": "#a65628",
    "J3": COLORS["J3"],
}
TRUE_DAY_COLOR = "#d62728"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--asos-results",
        type=Path,
        default=DEFAULT_ASOS_RESULTS,
        help=(
            "prepared ASOS fitted-result mapping produced by run_asos.py "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--rees-results",
        type=Path,
        default=DEFAULT_REES_RESULTS,
        help=(
            "prepared REES46 k=100 fitted-result mapping produced by "
            "run_rees46.py (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "directory for derived results and fig_hitting_time_mae_asos.pdf "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        nargs="+",
        default=(1.5, 2.0, 3.0),
        help="participation multipliers to evaluate",
    )
    parser.add_argument(
        "--figure-eta",
        type=float,
        default=2.0,
        help="multiplier displayed in the manuscript figure (default: 2)",
    )
    parser.add_argument(
        "--case-study",
        default="ee6ff7_C",
        help="ASOS experiment ID for the left panel (default: paper case study)",
    )
    return parser


def _record_mapping(data: Any) -> dict[object, dict[str, Any]]:
    value = load_plot_data(data)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping) and "experiments" in value:
        value = value["experiments"]

    records: dict[object, dict[str, Any]] = {}
    if isinstance(value, Mapping) and "cumulative_users" in value:
        key = value.get("exp_id", 0)
        records[key] = dict(value)
    elif isinstance(value, Mapping):
        for key, record in value.items():
            if not isinstance(record, Mapping):
                raise ValueError("each fitted experiment must be a mapping")
            copied = dict(record)
            copied.setdefault("exp_id", key)
            records[key] = copied
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, record in enumerate(value):
            if not isinstance(record, Mapping):
                raise ValueError("each fitted experiment must be a mapping")
            copied = dict(record)
            copied.setdefault("exp_id", index)
            if "arm" in copied:
                key: object = f"{copied['exp_id']}_{copied['arm']}"
            else:
                key = copied["exp_id"]
            if key in records:
                key = index
            records[key] = copied
    else:
        raise ValueError("fitted results must contain experiment records")
    if not records:
        raise ValueError("fitted results contains no experiments")
    return records


def _validated_curve(record: Mapping[str, Any]) -> tuple[np.ndarray, int, int, int]:
    curve = np.asarray(record.get("cumulative_users"), dtype=float)
    pilot_days = int(record.get("D0", record.get("pilot_days", -1)))
    followup_days = int(record.get("D1", len(curve) - pilot_days - 1))
    if (
        curve.ndim != 1
        or pilot_days < 1
        or followup_days < 1
        or len(curve) < pilot_days + followup_days + 1
        or np.any(~np.isfinite(curve))
        or np.any(curve < 0)
        or np.any(np.diff(curve) < 0)
    ):
        raise ValueError(f"experiment {record.get('exp_id')!r} has invalid curve metadata")
    curve = curve[: pilot_days + followup_days + 1]
    observed_users = int(curve[pilot_days])
    if observed_users <= 0:
        raise ValueError(f"experiment {record.get('exp_id')!r} has no pilot users")
    return curve, pilot_days, followup_days, observed_users


def _model_parameters(
    record: Mapping[str, Any], method: str
) -> Mapping[str, float] | None:
    for key, payload in record.items():
        if canonical_method(key) != method or not isinstance(payload, Mapping):
            continue
        parameters = payload.get("parameters")
        if isinstance(parameters, Mapping):
            return parameters
    return None


def _model_day(
    record: Mapping[str, Any],
    method: str,
    *,
    pilot_days: int,
    followup_days: int,
    observed_users: int,
    eta: float,
) -> int | None:
    parameters = _model_parameters(record, method)
    if parameters is None:
        return None
    try:
        return model_hitting_day(
            method,
            parameters,
            pilot_days=pilot_days,
            max_followup_days=followup_days,
            observed_users=observed_users,
            participation_multiplier=eta,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return None


def _jackknife_day(
    curve: np.ndarray,
    pilot_days: int,
    followup_days: int,
    target: int,
) -> int | None:
    try:
        prediction = predict_jackknife(
            pilot_days,
            followup_days,
            np.asarray([curve[pilot_days]], dtype=float),
            curve,
            order=3,
        )
    except (TypeError, ValueError, FloatingPointError, ZeroDivisionError):
        return None
    crossing = np.flatnonzero(prediction[pilot_days + 1 :] >= target)
    return pilot_days + int(crossing[0]) + 1 if len(crossing) else None


def _derive_predictions(
    records: Mapping[object, Mapping[str, Any]], etas: Sequence[float]
) -> dict[object, dict[float, dict[str, int | None]]]:
    predictions: dict[object, dict[float, dict[str, int | None]]] = {}
    for experiment_id, record in records.items():
        curve, pilot_days, followup_days, observed_users = _validated_curve(record)
        by_eta: dict[float, dict[str, int | None]] = {}
        for eta in etas:
            target = ceil(eta * observed_users)
            result: dict[str, int | None] = {
                "target_users": target,
                "true_day": first_hitting_day(curve, target),
                "last_observed_day": pilot_days + followup_days,
            }
            for method in ("NB-SSP", "TG-SSP", "IBP"):
                result[method] = _model_day(
                    record,
                    method,
                    pilot_days=pilot_days,
                    followup_days=followup_days,
                    observed_users=observed_users,
                    eta=eta,
                )
            daily_rate = observed_users / pilot_days
            result["Linear"] = pilot_days + ceil(
                (target - observed_users) / daily_rate
            )
            result["J3"] = _jackknife_day(
                curve, pilot_days, followup_days, target
            )
            by_eta[float(eta)] = result
        predictions[experiment_id] = by_eta
    return predictions


def _planning_errors(
    predictions: Mapping[object, Mapping[float, Mapping[str, int | None]]],
    eta: float,
    *,
    clamp_linear: bool,
) -> dict[str, list[float]]:
    errors = {method: [] for method in METHODS}
    for by_eta in predictions.values():
        result = by_eta[eta]
        true_day = result["true_day"]
        if true_day is None:
            continue
        for method in METHODS:
            predicted_day = result.get(method)
            if predicted_day is not None:
                if method == "Linear" and clamp_linear:
                    predicted_day = min(
                        predicted_day, int(result["last_observed_day"])
                    )
                errors[method].append(float(predicted_day - true_day))
    return errors


def _millions(value: float, _position: float) -> str:
    return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")


def _plot_trajectory(
    axis: plt.Axes,
    record: Mapping[str, Any],
    prediction: Mapping[str, int | None],
) -> None:
    curve, pilot_days, followup_days, _ = _validated_curve(record)
    target = int(prediction["target_users"])
    true_day = prediction["true_day"]
    days = np.arange(pilot_days + followup_days + 1)
    axis.plot(days, curve, color="black", linewidth=1.7, zorder=5)
    axis.axvspan(0, pilot_days, alpha=0.06, color="gray")
    axis.axvline(pilot_days, color="gray", linestyle=":", linewidth=0.8)
    axis.axhline(target, color="#555555", linestyle="--", linewidth=1.1)

    for label, day, color, marker in (
        ("True", true_day, TRUE_DAY_COLOR, "D"),
        ("NB-SSP", prediction.get("NB-SSP"), METHOD_COLORS["NB-SSP"], "o"),
        ("Linear", prediction.get("Linear"), METHOD_COLORS["Linear"], "o"),
    ):
        if day is None:
            continue
        axis.vlines(day, 0, target, color=color, linestyle=":", alpha=0.75)
        axis.scatter(day, target, color=color, marker=marker, s=40, zorder=6)

    legend = [
        Line2D([0], [0], color="black", linewidth=1.7, label="Truth"),
        Line2D([0], [0], color=TRUE_DAY_COLOR, marker="D", linestyle=":", label=r"True $D_\eta$"),
        Line2D([0], [0], color=METHOD_COLORS["NB-SSP"], marker="o", linestyle=":", label="NB-SSP"),
        Line2D([0], [0], color=METHOD_COLORS["Linear"], marker="o", linestyle=":", label="Linear"),
    ]
    axis.legend(handles=legend, loc="lower right", framealpha=0.9)
    axis.set_xlabel("Day")
    axis.set_ylabel("Cumulative distinct users (millions)")
    axis.yaxis.set_major_formatter(FuncFormatter(_millions))
    axis.set_title("(a) ASOS case study")
    axis.grid(alpha=0.15)


def _plot_errors(axis: plt.Axes, errors: Mapping[str, Sequence[float]], title: str) -> None:
    methods = [method for method in METHODS if errors[method]]
    if not methods:
        raise ValueError(f"no uncensored hitting-time predictions for {title}")
    boxes = axis.boxplot(
        [errors[method] for method in methods],
        labels=methods,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
        medianprops={"color": "black", "linewidth": 1.6},
    )
    for box, method in zip(boxes["boxes"], methods):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_alpha(0.45)
    axis.axhline(0, color="black", alpha=0.4, linewidth=1.0)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.grid(axis="y", alpha=0.2)


def _plot(
    asos_records: Mapping[object, Mapping[str, Any]],
    asos_predictions: Mapping[object, Mapping[float, Mapping[str, int | None]]],
    rees_predictions: Mapping[object, Mapping[float, Mapping[str, int | None]]],
    *,
    case_study: str,
    eta: float,
    output_path: Path,
) -> Path:
    case_key = next((key for key in asos_records if str(key) == case_study), None)
    if case_key is None:
        raise KeyError(f"ASOS case-study experiment not found: {case_study}")

    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(
            1, 3, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": (1.35, 1, 1)}
        )
        _plot_trajectory(
            axes[0], asos_records[case_key], asos_predictions[case_key][eta]
        )
        _plot_errors(
            axes[1],
            _planning_errors(asos_predictions, eta, clamp_linear=False),
            rf"(b) ASOS ($\eta={eta:g}$)",
        )
        axes[1].set_ylabel("Planning error (days)")
        _plot_errors(
            axes[2],
            _planning_errors(rees_predictions, eta, clamp_linear=True),
            rf"(c) REES46 ($\eta={eta:g}$)",
        )
        figure.tight_layout()
        return save_figure(figure, output_path)


def main() -> None:
    args = _parser().parse_args()
    etas = tuple(dict.fromkeys(float(eta) for eta in args.eta))
    if any(not np.isfinite(eta) or eta <= 1 for eta in etas):
        raise ValueError("all --eta values must be finite and greater than one")
    figure_eta = float(args.figure_eta)
    if not np.isfinite(figure_eta) or figure_eta <= 1:
        raise ValueError("--figure-eta must be finite and greater than one")
    if figure_eta not in etas:
        etas = (*etas, figure_eta)

    asos_records = _record_mapping(args.asos_results)
    rees_records = _record_mapping(args.rees_results)
    asos_predictions = _derive_predictions(asos_records, etas)
    rees_predictions = _derive_predictions(rees_records, etas)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "hitting_time_predictions.npy",
        {"ASOS": asos_predictions, "REES46": rees_predictions},
        allow_pickle=True,
    )
    _plot(
        asos_records,
        asos_predictions,
        rees_predictions,
        case_study=args.case_study,
        eta=figure_eta,
        output_path=output_dir / "fig_hitting_time_mae_asos.pdf",
    )


if __name__ == "__main__":
    main()
