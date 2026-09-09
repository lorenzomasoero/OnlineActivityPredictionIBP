"""Prediction-accuracy boxplots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    COLORS,
    PAPER_RC,
    canonical_method,
    load_plot_data,
    ordered_methods,
    save_figure,
)


def plot_accuracy_boxplot(
    data: Any,
    output_path: str | Path,
    *,
    methods: Sequence[str] | None = None,
    title: str | None = None,
    ylabel: str = "Prediction accuracy $v$",
) -> Path:
    """Plot accuracy distributions from records, a mapping, or a data path.

    Accepted mappings may be method-to-values or experiment-to-result records.
    Method result records use ``accuracy_v``, ``accuracy``, or ``v``. Tidy
    records with ``method`` plus one of those metric fields are also accepted.
    """
    values = _collect_accuracy(load_plot_data(data))
    selected = ordered_methods(methods, set(values))
    if not selected:
        raise ValueError("data contains no requested paper-model accuracy values")

    with plt.rc_context(PAPER_RC):
        width = max(4.8, 0.72 * len(selected) + 1.4)
        figure, axis = plt.subplots(figsize=(width, 3.6))
        import matplotlib as _mpl
        # matplotlib renamed boxplot's `labels` -> `tick_labels` in 3.9 (removed 3.11).
        _lbl = {("tick_labels" if tuple(map(int, _mpl.__version__.split(".")[:2])) >= (3, 9)
                 else "labels"): selected}
        boxplot = axis.boxplot(
            [values[method] for method in selected],
            **_lbl,
            patch_artist=True,
            widths=0.58,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.6},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
        )
        for patch, method in zip(boxplot["boxes"], selected):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        axis.set_ylabel(ylabel)
        if title:
            axis.set_title(title)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.22)
        if len(selected) > 7:
            axis.tick_params(axis="x", rotation=35)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
        figure.tight_layout()
        return save_figure(figure, output_path)


def _collect_accuracy(data: Any) -> dict[str, np.ndarray]:
    collected: defaultdict[str, list[float]] = defaultdict(list)

    def add(method: object, payload: Any) -> bool:
        canonical = canonical_method(method)
        if canonical is None:
            return False
        metric = payload
        if isinstance(payload, Mapping):
            for key in ("accuracy_v", "accuracy", "v", "values"):
                if key in payload:
                    metric = payload[key]
                    break
            else:
                return False
        try:
            array = np.asarray(metric, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return False
        collected[canonical].extend(array[np.isfinite(array)].tolist())
        return array.size > 0

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if "method" in node:
                add(node["method"], node)
                return
            for key, value in node.items():
                if add(key, value):
                    continue
                if isinstance(value, Mapping):
                    visit(value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
                    for item in value:
                        if isinstance(item, Mapping):
                            visit(item)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, np.ndarray)):
            for item in node:
                visit(item)

    visit(data)
    return {
        method: np.asarray(method_values, dtype=float)
        for method, method_values in collected.items()
        if method_values
    }


__all__ = ["plot_accuracy_boxplot"]
