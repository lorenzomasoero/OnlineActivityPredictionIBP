"""Truth-versus-prediction trajectory panels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .style import (
    COLORS,
    LINESTYLES,
    MARKERS,
    PAPER_RC,
    canonical_method,
    load_plot_data,
    ordered_methods,
    save_figure,
)


def plot_trajectory_panels(
    data: Any,
    output_path: str | Path,
    *,
    experiment_ids: Sequence[object] | None = None,
    methods: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Plot cumulative-user truth and fitted prediction trajectories.

    Each experiment record must contain ``cumulative_users`` (or ``truth``)
    and ``D0`` (or ``pilot_days``). Model records may contain
    ``U_trajectory`` and/or ``U_hat``; both are interpreted as numbers of new
    users after the pilot, as in the stored AoAS results.
    """
    experiments = _experiment_mapping(load_plot_data(data))
    selected_experiments = _select_experiments(experiments, experiment_ids)
    available = {
        canonical
        for record in selected_experiments.values()
        for key in record
        if (canonical := canonical_method(key)) is not None
    }
    selected_methods = ordered_methods(methods, available)
    if not selected_methods:
        raise ValueError("selected experiments contain no requested model predictions")

    count = len(selected_experiments)
    columns = min(3, count)
    rows = math.ceil(count / columns)
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.05 * columns, 2.75 * rows),
            squeeze=False,
        )
        legend_handles: dict[str, Any] = {}

        for axis, (experiment_id, record) in zip(axes.flat, selected_experiments.items()):
            truth = np.asarray(
                record.get("cumulative_users", record.get("truth")), dtype=float
            )
            if truth.ndim != 1 or len(truth) < 2 or not np.all(np.isfinite(truth)):
                raise ValueError(f"experiment {experiment_id!r} has an invalid truth trajectory")
            pilot_days = int(record.get("D0", record.get("pilot_days", -1)))
            if not 0 <= pilot_days < len(truth) - 1:
                raise ValueError(f"experiment {experiment_id!r} has an invalid pilot horizon")
            observed_users = truth[pilot_days]
            final_day = len(truth) - 1

            truth_line, = axis.plot(
                np.arange(len(truth)),
                truth,
                color="black",
                linewidth=1.6,
                label="Truth",
                zorder=5,
            )
            legend_handles.setdefault("Truth", truth_line)
            axis.axvspan(0, pilot_days, color="gray", alpha=0.08)
            axis.axvline(pilot_days, color="gray", linestyle=":", linewidth=0.9)

            for method in selected_methods:
                prediction = _method_prediction(record, method)
                if prediction is None:
                    continue
                trajectory = prediction.get(
                    "U_trajectory",
                    prediction.get(
                        "new_user_trajectory",
                        prediction.get(
                            "expected_new_users", prediction.get("trajectory")
                        ),
                    ),
                )
                final_prediction: float | None = None
                if trajectory is not None:
                    trajectory = np.asarray(trajectory, dtype=float).reshape(-1)
                    trajectory = trajectory[np.isfinite(trajectory)]
                    if trajectory.size:
                        prediction_days = np.arange(
                            pilot_days + 1, pilot_days + 1 + len(trajectory)
                        )
                        line, = axis.plot(
                            prediction_days,
                            observed_users + trajectory,
                            color=COLORS[method],
                            linestyle=LINESTYLES[method],
                            marker=MARKERS[method],
                            markevery=max(1, len(trajectory) // 6),
                            markersize=3.2,
                            linewidth=1.2,
                            label=method,
                        )
                        legend_handles.setdefault(method, line)
                        final_prediction = float(observed_users + trajectory[-1])
                elif "U_hat" in prediction or "new_users" in prediction:
                    estimate = float(prediction.get("U_hat", prediction.get("new_users")))
                    points = axis.scatter(
                        [final_day],
                        [observed_users + estimate],
                        color=COLORS[method],
                        marker=MARKERS[method],
                        s=34,
                        label=method,
                        zorder=4,
                    )
                    legend_handles.setdefault(method, points)
                    axis.plot(
                        [pilot_days, final_day],
                        [observed_users, observed_users + estimate],
                        color=COLORS[method],
                        linestyle=LINESTYLES[method],
                        alpha=0.45,
                    )
                    final_prediction = observed_users + estimate

                interval = prediction.get("U_ci", prediction.get("new_user_interval"))
                if interval is not None and final_prediction is not None:
                    lower, upper = np.asarray(interval, dtype=float).reshape(2, -1)[:, -1]
                    axis.errorbar(
                        final_day,
                        final_prediction,
                        yerr=[
                            [max(final_prediction - (observed_users + lower), 0.0)],
                            [max(observed_users + upper - final_prediction, 0.0)],
                        ],
                        fmt="none",
                        ecolor=COLORS[method],
                        elinewidth=1.2,
                        capsize=2.5,
                        alpha=0.75,
                    )

            axis.set_xlabel("Day")
            axis.set_ylabel("Cumulative distinct users")
            axis.set_title(str(record.get("title", f"Exp {experiment_id}")))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_formatter(FuncFormatter(_compact_tick_label))
            axis.grid(alpha=0.22)

        for axis in axes.flat[count:]:
            axis.set_visible(False)
        if title:
            figure.suptitle(title)
        figure.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0.12, 1, 0.96 if title else 1))
        return save_figure(figure, output_path)


def _experiment_mapping(data: Any) -> dict[object, Mapping[str, Any]]:
    if isinstance(data, Mapping) and "experiments" in data:
        data = data["experiments"]
    if isinstance(data, Mapping) and (
        "cumulative_users" in data or "truth" in data
    ):
        return {data.get("exp_id", 0): data}
    if isinstance(data, Mapping):
        experiments = {
            key: value for key, value in data.items() if isinstance(value, Mapping)
        }
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, np.ndarray)):
        experiments = {
            record.get("exp_id", index): record
            for index, record in enumerate(data)
            if isinstance(record, Mapping)
        }
    elif isinstance(data, np.ndarray) and data.dtype == object:
        return _experiment_mapping(data.tolist())
    else:
        raise ValueError("trajectory data must contain experiment records")
    if not experiments:
        raise ValueError("trajectory data contains no experiment records")
    return experiments


def _select_experiments(
    experiments: Mapping[object, Mapping[str, Any]],
    experiment_ids: Sequence[object] | None,
) -> dict[object, Mapping[str, Any]]:
    if experiment_ids is None:
        return dict(experiments)
    selected: dict[object, Mapping[str, Any]] = {}
    by_string = {str(key): key for key in experiments}
    for requested in experiment_ids:
        key = requested if requested in experiments else by_string.get(str(requested))
        if key is None:
            raise KeyError(f"unknown experiment id: {requested}")
        selected[key] = experiments[key]
    if not selected:
        raise ValueError("experiment_ids must select at least one experiment")
    return selected


def _method_prediction(
    record: Mapping[str, Any], method: str
) -> Mapping[str, Any] | None:
    for key, value in record.items():
        if canonical_method(key) == method and isinstance(value, Mapping):
            if "error" not in value:
                return value
    return None


def _compact_tick_label(value: float, _position: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if absolute >= 10_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


__all__ = ["plot_trajectory_panels"]
