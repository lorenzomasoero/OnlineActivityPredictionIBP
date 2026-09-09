"""Shared, side-effect-free style and data-loading helpers for paper plots."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = (
    "Be-SSP",
    "TG-SSP",
    "NB-SSP",
    "IBP",
    "BB",
    "HBG",
    "GT",
    "J1",
    "J2",
    "J3",
    "J4",
    "LP",
)

METHOD_ALIASES = {
    "Be-SSP": "Be-SSP",
    "BeSSP": "Be-SSP",
    "Be_SSP": "Be-SSP",
    "Be-SSP (curve)": "Be-SSP",
    "SSP": "Be-SSP",
    "TG-SSP": "TG-SSP",
    "TGSSP": "TG-SSP",
    "TG_SSP": "TG-SSP",
    "TG-SSP (curve)": "TG-SSP",
    "TG_SSP_geom": "TG-SSP",
    "SSP_geom": "TG-SSP",
    "NB-SSP": "NB-SSP",
    "NBSSP": "NB-SSP",
    "NB_SSP": "NB-SSP",
    "NB-SSP (curve)": "NB-SSP",
    "NB_SSP_MLE": "NB-SSP",
    "NB_SSP_regression": "NB-SSP",
    "IBP": "IBP",
    "BB": "BB",
    "BetaBinomial": "BB",
    "HBG": "HBG",
    "BG": "HBG",
    "Beta-Geometric": "HBG",
    "HierarchicalBetaGeometric": "HBG",
    "GT": "GT",
    "Good-Toulmin": "GT",
    "Good--Toulmin": "GT",
    "GoodToulmin": "GT",
    "J1": "J1",
    "J2": "J2",
    "J3": "J3",
    "jackknife": "J3",
    "Jackknife (J3)": "J3",
    "J4": "J4",
    "LP": "LP",
    "UnseenEST": "LP",
}

COLORS = {
    "Be-SSP": "#e41a1c",
    "TG-SSP": "#ff7f00",
    "NB-SSP": "#4daf4a",
    "IBP": "#377eb8",
    "BB": "#984ea3",
    "HBG": "#a65628",
    "GT": "#737373",
    "J1": "#f781bf",
    "J2": "#e6550d",
    "J3": "#756bb1",
    "J4": "#31a354",
    "LP": "#1b9e77",
}

MARKERS = {
    "Be-SSP": "o",
    "TG-SSP": "s",
    "NB-SSP": "^",
    "IBP": "D",
    "BB": "P",
    "HBG": "X",
    "GT": "v",
    "J1": "<",
    "J2": ">",
    "J3": "h",
    "J4": "8",
    "LP": "p",
}

LINESTYLES = {
    label: linestyle
    for label, linestyle in zip(
        METHOD_ORDER,
        ("-", "--", "-.", ":", "--", "-.", ":", "-", "--", "-.", ":", "-"),
    )
}

PAPER_RC = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def canonical_method(name: object) -> str | None:
    """Translate a stored result key to the exact display label in the paper."""
    return METHOD_ALIASES.get(str(name))


def load_plot_data(data: Any) -> Any:
    """Return in-memory data, loading a supported file when given a path."""
    if not isinstance(data, (str, Path)):
        return data

    path = Path(data)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=True)
        return value.item() if value.shape == () else value
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    raise ValueError(f"unsupported plot-data format: {suffix or '<none>'}")


def save_figure(figure: plt.Figure, output_path: str | Path) -> Path:
    """Save and close a figure at the caller-provided path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, pad_inches=0.02)
    plt.close(figure)
    return path


def ordered_methods(methods: object, available: set[str]) -> list[str]:
    """Resolve an optional method selection in deterministic paper order."""
    if methods is None:
        return [method for method in METHOD_ORDER if method in available]
    if isinstance(methods, str):
        methods = [methods]
    selected: list[str] = []
    for method in methods:
        canonical = canonical_method(method)
        if canonical is None:
            raise ValueError(f"unknown paper method: {method}")
        if canonical in available and canonical not in selected:
            selected.append(canonical)
    return selected
