"""Public REES46 descriptive figures used in the manuscript."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .style import PAPER_RC, load_plot_data, save_figure


def plot_rees46_cumulative_users(
    experiments: Any,
    output_path: str | Path,
    *,
    selected: Sequence[int] = (0, 3, 6),
    pilot_days: int = 7,
) -> Path:
    """Plot cumulative distinct-user curves for selected REES46 windows."""

    records = load_plot_data(experiments)
    if isinstance(records, np.ndarray):
        records = records.tolist()
    if not isinstance(records, Sequence):
        raise ValueError("experiments must be a sequence of records")
    colors = ("#1b9e77", "#d95f02", "#7570b3")
    with plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(5.6, 3.7))
        for color, index in zip(colors, selected, strict=False):
            record = records[index]
            curve = np.asarray(record["cumulative_users"], dtype=float)
            label = f"Exp {index} ($N_{{D_0}}$={int(record['N_pilot']):,})"
            axis.plot(np.arange(len(curve)), curve / 1e6, color=color, linewidth=1.7, label=label)
        axis.axvline(pilot_days, color="gray", linestyle=":", linewidth=1.0, label="Pilot end")
        axis.set_xlabel("Day within experiment window")
        axis.set_ylabel("Cumulative distinct users (millions)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        return save_figure(figure, output_path)


def plot_rees46_triggers_per_user(stats: Any, output_path: str | Path) -> Path:
    """Plot the log--log per-user trigger-count distribution."""

    values = _stats_mapping(stats)
    histogram = np.asarray(values["triggers_per_user_histogram"], dtype=float)
    users = float(values["n_users"])
    support = np.arange(1, len(histogram))
    frequencies = histogram[1:]
    mask = frequencies > 0
    with plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(4.5, 3.5))
        axis.scatter(support[mask], frequencies[mask] / users, s=5, alpha=0.45, color="#1b9e77")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Total triggers per user")
        axis.set_ylabel("Proportion of users")
        axis.grid(alpha=0.25, which="both")
        figure.tight_layout()
        return save_figure(figure, output_path)


def plot_rees46_days_active(
    stats: Any,
    output_path: str | Path,
    *,
    max_days: int = 30,
) -> Path:
    """Plot the distribution of active days per REES46 user."""

    values = _stats_mapping(stats)
    histogram = np.asarray(values["days_active_histogram"], dtype=float)
    users = float(values["n_users"])
    limit = min(max_days, len(histogram) - 1)
    support = np.arange(1, limit + 1)
    with plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(4.5, 3.5))
        axis.bar(support, histogram[1 : limit + 1] / users, color="#d95f02", alpha=0.72)
        axis.set_xlabel("Number of active days")
        axis.set_ylabel("Proportion of users")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        return save_figure(figure, output_path)


def _stats_mapping(stats: Any) -> Mapping[str, Any]:
    values = load_plot_data(stats)
    if not isinstance(values, Mapping):
        raise ValueError("stats must be a mapping or a path containing one")
    required = {"n_users", "triggers_per_user_histogram", "days_active_histogram"}
    if not required.issubset(values):
        raise ValueError(f"stats is missing keys: {sorted(required - set(values))}")
    if float(values["n_users"]) <= 0:
        raise ValueError("n_users must be positive")
    return values


__all__ = [
    "plot_rees46_cumulative_users",
    "plot_rees46_days_active",
    "plot_rees46_triggers_per_user",
]
