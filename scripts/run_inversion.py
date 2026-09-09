#!/usr/bin/env python3
"""Compute the TG-SSP inversion illustration and write ``inversion_ci.pdf``.

The defaults reproduce the setup in the paper's historical
``code/experiments/Report.ipynb``: two pilot days, ``alpha=0.1``, ``c=2500``,
``beta=0.5``, a 29-day forecast, and target ``M=2200``.  Counts in the pilot
are simulated from the paper's sequential TG-SSP unseen-user law.  The shaded
region is the historical notebook's very high-coverage marginal predictive
band, which is sliced at ``M`` to obtain the displayed interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln

from activity_prediction.models import TGSSP
from activity_prediction.plotting.style import PAPER_RC, save_figure


def _psi_terms(alpha: float, total_days: int) -> np.ndarray:
    days = np.arange(1, total_days + 1, dtype=float)
    return alpha * np.exp(betaln(1.0 - alpha, days))


def _simulate_pilot_curve(
    rng: np.random.Generator,
    *,
    pilot_days: int,
    alpha: float,
    c: float,
    beta: float,
) -> np.ndarray:
    """Sample cumulative first-trigger counts through the pilot."""
    terms = _psi_terms(alpha, pilot_days)
    cumulative_psi = np.concatenate(([0.0], np.cumsum(terms)))
    cumulative_users = np.zeros(pilot_days + 1, dtype=int)

    for day in range(1, pilot_days + 1):
        success_probability = (beta + cumulative_psi[day - 1]) / (
            beta + cumulative_psi[day]
        )
        shape = cumulative_users[day - 1] + c + 1.0
        new_users = int(rng.negative_binomial(shape, success_probability))
        if day == 1:
            while new_users == 0:
                new_users = int(rng.negative_binomial(shape, success_probability))
        cumulative_users[day] = cumulative_users[day - 1] + new_users
    return cumulative_users


def _slice_band(
    days: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: float,
) -> tuple[float, float] | None:
    left = np.flatnonzero(upper >= target)
    right = np.flatnonzero(lower <= target)
    if not len(left) or not len(right) or left[0] > right[-1]:
        return None
    return float(days[left[0]]), float(days[right[-1]])


def run_experiment(
    output_dir: Path,
    *,
    seed: int,
    pilot_days: int,
    followup_days: int,
    alpha: float,
    c: float,
    beta: float,
    target_users: float,
    coverage: float,
) -> tuple[Path, Path]:
    """Simulate the pilot and write its inversion PDF and raw band."""
    if pilot_days < 1 or followup_days < 1:
        raise ValueError("pilot_days and followup_days must be positive")
    if target_users <= 0:
        raise ValueError("target_users must be positive")

    rng = np.random.default_rng(seed)
    observed_curve = _simulate_pilot_curve(
        rng,
        pilot_days=pilot_days,
        alpha=alpha,
        c=c,
        beta=beta,
    )
    observed_users = int(observed_curve[-1])
    model = TGSSP()
    expected = model.expected_new_users(
        pilot_days,
        followup_days,
        observed_users,
        alpha=alpha,
        c=c,
        beta=beta,
    )
    lower_new, upper_new = model.new_user_interval(
        pilot_days,
        followup_days,
        observed_users,
        coverage=coverage,
        alpha=alpha,
        c=c,
        beta=beta,
    )

    forecast_days = pilot_days + np.arange(0, followup_days + 1)
    mean_users = np.concatenate(([observed_users], observed_users + expected))
    lower_users = np.concatenate(([observed_users], observed_users + lower_new))
    upper_users = np.concatenate(([observed_users], observed_users + upper_new))
    interval = _slice_band(
        forecast_days,
        lower_users,
        upper_users,
        target_users,
    )

    with plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(5.2, 4.2))
        pilot_axis = np.arange(pilot_days + 1)
        axis.plot(pilot_axis, observed_curve, color="black", linewidth=1.8)
        axis.plot(
            forecast_days,
            mean_users,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        axis.fill_between(
            forecast_days,
            lower_users,
            upper_users,
            color="gray",
            alpha=0.2,
        )
        axis.axvline(pilot_days, color="forestgreen", linestyle=":", linewidth=1.2)
        axis.axhline(target_users, color="red", linestyle="-.", linewidth=1.2)

        if interval is not None:
            left, right = interval
            axis.scatter(
                [left, right],
                [target_users, target_users],
                color="gold",
                edgecolor="goldenrod",
                marker="*",
                s=90,
                zorder=5,
            )
            axis.vlines(
                [left, right],
                0,
                target_users,
                color="gray",
                linestyle="--",
                linewidth=1.0,
            )
            bracket_height = max(0.04 * target_users, 1.0)
            axis.annotate(
                "",
                xy=(left, bracket_height),
                xytext=(right, bracket_height),
                arrowprops={"arrowstyle": "|-|", "color": "darkorange", "linewidth": 1.5},
            )
            axis.text(
                0.5 * (left + right),
                1.25 * bracket_height,
                r"$D_M$",
                ha="center",
                va="bottom",
                fontsize=13,
            )

        axis.text(-0.02, target_users, r"$M$", ha="right", va="center", fontsize=13)
        axis.set_xlim(0, pilot_days + followup_days)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("day")
        axis.set_ylabel(r"$N_D$")
        axis.set_yticks([])
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure_path = save_figure(figure, output_dir / "inversion_ci.pdf")

    result_path = output_dir / "inversion_ci.npz"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    interval_array = (
        np.asarray(interval, dtype=float)
        if interval is not None
        else np.asarray([np.nan, np.nan])
    )
    np.savez_compressed(
        result_path,
        seed=seed,
        alpha=alpha,
        c=c,
        beta=beta,
        coverage=coverage,
        target_users=target_users,
        pilot_days=pilot_days,
        followup_days=followup_days,
        observed_days=np.arange(pilot_days + 1),
        observed_users=observed_curve,
        forecast_days=forecast_days,
        mean_users=mean_users,
        lower_users=lower_users,
        upper_users=upper_users,
        inversion_interval=interval_array,
    )
    return figure_path, result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the computed TG-SSP inversion illustration."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20230903)
    parser.add_argument("--pilot-days", type=int, default=2)
    parser.add_argument("--followup-days", type=int, default=29)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=2500.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--target-users", type=float, default=2200.0)
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.9999,
        help="Marginal predictive-band coverage (the notebook used 0.9999).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args.output_dir,
        seed=args.seed,
        pilot_days=args.pilot_days,
        followup_days=args.followup_days,
        alpha=args.alpha,
        c=args.c,
        beta=args.beta,
        target_users=args.target_users,
        coverage=args.coverage,
    )


if __name__ == "__main__":
    main()
