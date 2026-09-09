#!/usr/bin/env python3
r"""Reproduce the paper's SB-SP parameter-sensitivity illustration.

This is a direct, script-level port of ``code/Untitled.ipynb``.  The
historical notebook uses seven pilot days and seven additional days while
labelling the response as :math:`\hat U_7^{(14)}`; here the superscript is
therefore treated as the ending day, exactly as in that figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from activity_prediction.models import TGSSP
from activity_prediction.plotting.style import PAPER_RC, save_figure


def _final_prediction(
    model: TGSSP,
    *,
    pilot_days: int,
    followup_days: int,
    observed_users: int,
    alpha: float,
    c: float,
    beta: float,
) -> float:
    return float(
        model.expected_new_users(
            pilot_days,
            followup_days,
            observed_users,
            alpha=alpha,
            c=c,
            beta=beta,
        )[-1]
    )


def run_experiment(
    output_dir: Path,
    *,
    pilot_days: int,
    followup_days: int,
    observed_users: int,
    grid_size: int,
) -> tuple[Path, Path]:
    """Compute the sensitivity panels and write their PDF and raw curves."""
    if pilot_days < 1 or followup_days < 1 or observed_users < 0:
        raise ValueError("days must be positive and observed_users non-negative")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")

    model = TGSSP()
    alpha_grid = np.linspace(0.1, 0.9, grid_size)
    c_grid = np.linspace(1.0, 100.0, grid_size)
    beta_grid = np.linspace(1.0, 20.0, grid_size)
    alpha_curves: list[np.ndarray] = []
    c_curves: list[np.ndarray] = []
    beta_curves: list[np.ndarray] = []

    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

        for c, beta in ((200.0, 500.0), (200.0, 1000.0), (2000.0, 500.0), (2000.0, 1000.0)):
            values = np.asarray([
                _final_prediction(
                    model,
                    pilot_days=pilot_days,
                    followup_days=followup_days,
                    observed_users=observed_users,
                    alpha=alpha,
                    c=c,
                    beta=beta,
                )
                for alpha in alpha_grid
            ])
            alpha_curves.append(values)
            axes[0].plot(
                alpha_grid,
                values,
                color="red" if c == 200.0 else "steelblue",
                linestyle="-" if beta == 500.0 else "-.",
                label=rf"$c$: {c:g}, $\beta$: {beta:g}",
            )

        for alpha, beta in ((0.25, 500.0), (0.25, 1000.0), (0.5, 500.0), (0.5, 1000.0)):
            values = np.asarray([
                _final_prediction(
                    model,
                    pilot_days=pilot_days,
                    followup_days=followup_days,
                    observed_users=observed_users,
                    alpha=alpha,
                    c=c,
                    beta=beta,
                )
                for c in c_grid
            ])
            c_curves.append(values)
            axes[1].plot(
                c_grid,
                values,
                color="forestgreen" if alpha == 0.25 else "orange",
                linestyle="-" if beta == 500.0 else "-.",
                label=rf"$\alpha$: {alpha:g}, $\beta$: {beta:g}",
            )

        for alpha, c in ((0.25, 200.0), (0.25, 2000.0), (0.5, 200.0), (0.5, 2000.0)):
            values = np.asarray([
                _final_prediction(
                    model,
                    pilot_days=pilot_days,
                    followup_days=followup_days,
                    observed_users=observed_users,
                    alpha=alpha,
                    c=c,
                    beta=beta,
                )
                for beta in beta_grid
            ])
            beta_curves.append(values)
            axes[2].plot(
                beta_grid,
                values,
                color="gray" if alpha == 0.25 else "brown",
                linestyle="-" if c == 200.0 else "-.",
                label=rf"$\alpha$: {alpha:g}, $c$: {c:g}",
            )

        axes[0].set_yscale("log")
        axes[2].set_yscale("log")
        axes[0].set_xlabel(r"$\alpha$")
        axes[1].set_xlabel(r"$c$")
        axes[2].set_xlabel(r"$\beta$")
        ending_day = pilot_days + followup_days
        axes[0].set_ylabel(rf"$\hat{{U}}_{{{pilot_days}}}^{{{ending_day}}}$")
        for axis in axes:
            axis.legend(frameon=True)
        figure.tight_layout()

        figure_path = save_figure(figure, output_dir / "u_simu.pdf")

    result_path = output_dir / "u_simu.npz"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_path,
        pilot_days=pilot_days,
        followup_days=followup_days,
        observed_users=observed_users,
        alpha_grid=alpha_grid,
        c_grid=c_grid,
        beta_grid=beta_grid,
        alpha_curves=np.stack(alpha_curves),
        c_curves=np.stack(c_curves),
        beta_curves=np.stack(beta_curves),
        alpha_curve_c=np.asarray([200.0, 200.0, 2000.0, 2000.0]),
        alpha_curve_beta=np.asarray([500.0, 1000.0, 500.0, 1000.0]),
        c_curve_alpha=np.asarray([0.25, 0.25, 0.5, 0.5]),
        c_curve_beta=np.asarray([500.0, 1000.0, 500.0, 1000.0]),
        beta_curve_alpha=np.asarray([0.25, 0.25, 0.5, 0.5]),
        beta_curve_c=np.asarray([200.0, 2000.0, 200.0, 2000.0]),
    )
    return figure_path, result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic SB-SP parameter-sensitivity experiment."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-days", type=int, default=7)
    parser.add_argument("--followup-days", type=int, default=7)
    parser.add_argument("--observed-users", type=int, default=150)
    parser.add_argument("--grid-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args.output_dir,
        pilot_days=args.pilot_days,
        followup_days=args.followup_days,
        observed_users=args.observed_users,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
