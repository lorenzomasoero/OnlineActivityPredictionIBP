#!/usr/bin/env python3
"""Run the NB-SSP future-trigger illustration from the paper.

The paper defaults are ``D=200``, ``D0=20`` and
``(alpha, c, beta, r)=(0.5, 50, 0.1, 5)``.  One NB-SSP dataset is generated,
the model is fitted by maximum marginal likelihood on the pilot, and the
true, oracle, and fitted trigger trajectories are plotted in
``PAPER_prediction_sum_synthetic.pdf``.

The sampler ports the intended distribution in the legacy sequential NB-SSP
implementation.  It deliberately corrects that code's apparent new-user
sample-size bug: every newly observed user receives a count, not just one user
per day.  The infinite support of that count distribution is truncated at
``--max-new-count`` (default 200), matching the legacy numerical strategy but
making the approximation explicit and configurable.  These choices,
parameters, and final predictions are recorded next to the figure in
``PAPER_prediction_sum_synthetic.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import betaln, gammaln, logsumexp

from activity_prediction.models import NBSSP
from activity_prediction.plotting.style import PAPER_RC, save_figure


def _psi(
    alpha: float,
    r: float,
    start_days: int,
    additional_days: int,
) -> float:
    return float(
        alpha
        * (
            beta_function(r * start_days + 1.0, -alpha)
            - beta_function(r * (start_days + additional_days) + 1.0, -alpha)
        )
    )


def _new_user_count_probabilities(
    *,
    day: int,
    alpha: float,
    r: float,
    max_new_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.arange(1, max_new_count + 1, dtype=float)
    log_weights = (
        gammaln(counts + r)
        - gammaln(counts + 1.0)
        - gammaln(r)
        + betaln(counts - alpha, r * day + 1.0)
    )
    probabilities = np.exp(log_weights - logsumexp(log_weights))
    return counts.astype(int), probabilities


def _simulate_nb_ssp_activity(
    rng: np.random.Generator,
    *,
    days: int,
    alpha: float,
    c: float,
    beta: float,
    r: float,
    max_new_count: int,
) -> np.ndarray:
    """Generate a days-by-users NB-SSP count matrix sequentially."""
    if days < 1 or max_new_count < 1:
        raise ValueError("days and max_new_count must be positive")
    if not 0 < alpha < 1 or c <= 0 or beta <= 0 or r <= 0:
        raise ValueError("NB-SSP parameters are outside their support")

    rows: list[np.ndarray] = []
    user_totals = np.empty(0, dtype=np.int64)

    for day in range(1, days + 1):
        number_seen = user_totals.size
        paper_probability = _psi(alpha, r, day - 1, 1) / (
            beta + _psi(alpha, r, 0, day)
        )
        success_probability = float(np.clip(1.0 - paper_probability, 1e-12, 1.0))
        number_new = int(
            rng.negative_binomial(number_seen + c + 1.0, success_probability)
        )

        if number_seen:
            propensities = rng.beta(
                user_totals - alpha,
                r * (day - 1) + 1.0,
            )
            old_counts = rng.negative_binomial(
                r,
                np.clip(1.0 - propensities, 1e-12, 1.0),
            ).astype(np.int64)
        else:
            old_counts = np.empty(0, dtype=np.int64)

        if number_new:
            support, probabilities = _new_user_count_probabilities(
                day=day,
                alpha=alpha,
                r=r,
                max_new_count=max_new_count,
            )
            new_counts = rng.choice(
                support,
                size=number_new,
                replace=True,
                p=probabilities,
            ).astype(np.int64)
        else:
            new_counts = np.empty(0, dtype=np.int64)

        rows.append(np.concatenate((old_counts, new_counts)))
        user_totals = np.concatenate((user_totals + old_counts, new_counts))

    if not user_totals.size:
        raise RuntimeError(
            "the simulated dataset contains no users; choose another seed or parameters"
        )

    matrix = np.zeros((days, user_totals.size), dtype=np.int32)
    for day_index, row in enumerate(rows):
        matrix[day_index, : row.size] = row
    return matrix


def _expected_components(
    model: NBSSP,
    *,
    pilot_days: int,
    followup_days: int,
    observed_users: int,
    pilot_total_triggers: int,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = model.expected_total_triggers(
        pilot_days,
        followup_days,
        observed_users,
        pilot_total_triggers,
        **parameters,
    )
    horizons = np.arange(1, followup_days + 1, dtype=float)
    previously_seen = (horizons / pilot_days) * (
        pilot_total_triggers - parameters["alpha"] * observed_users
    )
    new_users = total - previously_seen
    return previously_seen, new_users, total


def run_experiment(
    output_dir: Path,
    *,
    seed: int,
    days: int,
    pilot_days: int,
    alpha: float,
    c: float,
    beta: float,
    r: float,
    num_restarts: int,
    max_new_count: int,
) -> tuple[Path, Path, Path]:
    """Generate data and write the prediction PDF, raw curves, and metadata."""
    if not 1 <= pilot_days < days:
        raise ValueError("pilot_days must lie between one and days - 1")
    if num_restarts < 1:
        raise ValueError("num_restarts must be positive")

    rng = np.random.default_rng(seed)
    activity = _simulate_nb_ssp_activity(
        rng,
        days=days,
        alpha=alpha,
        c=c,
        beta=beta,
        r=r,
        max_new_count=max_new_count,
    )
    observed_mask = activity[:pilot_days].sum(axis=0) > 0
    pilot_activity = activity[:pilot_days, observed_mask]
    if pilot_activity.shape[1] == 0:
        raise RuntimeError("the pilot contains no observed users")

    model = NBSSP()
    fitted = model.fit_marginal_likelihood(
        pilot_activity,
        num_restarts=num_restarts,
        seed=seed,
    )
    oracle = {"alpha": alpha, "c": c, "beta": beta, "r": r}
    followup_days = days - pilot_days
    observed_users = int(observed_mask.sum())
    pilot_total_triggers = int(pilot_activity.sum())

    oracle_components = _expected_components(
        model,
        pilot_days=pilot_days,
        followup_days=followup_days,
        observed_users=observed_users,
        pilot_total_triggers=pilot_total_triggers,
        parameters=oracle,
    )
    fitted_components = _expected_components(
        model,
        pilot_days=pilot_days,
        followup_days=followup_days,
        observed_users=observed_users,
        pilot_total_triggers=pilot_total_triggers,
        parameters=fitted,
    )

    followup = activity[pilot_days:]
    true_seen = np.cumsum(followup[:, observed_mask].sum(axis=1))
    true_new = np.cumsum(followup[:, ~observed_mask].sum(axis=1))
    true_components = (true_seen, true_new, true_seen + true_new)
    plot_days = np.arange(pilot_days, days + 1)

    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True)
        titles = (
            r"$S_{D_0}^{(d)}$",
            r"$\sum_j j U_{D_0}^{(d,j)}$",
            r"$T_{D_0}^{(d)}$",
        )
        for index, (axis, title) in enumerate(zip(axes, titles, strict=True)):
            true_values = np.concatenate(([0.0], true_components[index]))
            oracle_values = np.concatenate(([0.0], oracle_components[index]))
            fitted_values = np.concatenate(([0.0], fitted_components[index]))
            axis.plot(plot_days, true_values, color="blue", linewidth=2.0, label="True")
            axis.plot(plot_days, oracle_values, color="green", linewidth=2.0, label="Oracle")
            axis.plot(
                plot_days,
                fitted_values,
                color="red",
                linewidth=2.0,
                linestyle=":",
                label="Fitted",
            )
            axis.set_title(title)
            axis.set_xlabel(r"# Days $d$")
            axis.grid(alpha=0.18)
        axes[0].set_ylabel("# Triggers")
        axes[0].legend(frameon=True)
        figure.tight_layout()
        figure_path = save_figure(
            figure,
            output_dir / "PAPER_prediction_sum_synthetic.pdf",
        )

    metadata = {
        "seed": seed,
        "days": days,
        "pilot_days": pilot_days,
        "num_restarts": num_restarts,
        "max_new_count": max_new_count,
        "sampler": {
            "legacy_distribution": "sequential NB-SSP draw",
            "corrected_new_user_sample_size": True,
            "new_user_count_support": [1, max_new_count],
        },
        "true_parameters": oracle,
        "fitted_parameters": fitted,
        "observed_users": observed_users,
        "pilot_total_triggers": pilot_total_triggers,
        "final_true_components": {
            "previously_seen_user_triggers": float(true_components[0][-1]),
            "new_user_triggers": float(true_components[1][-1]),
            "total": float(true_components[2][-1]),
        },
        "final_oracle_components": {
            "previously_seen_user_triggers": float(oracle_components[0][-1]),
            "new_user_triggers": float(oracle_components[1][-1]),
            "total": float(oracle_components[2][-1]),
        },
        "final_fitted_components": {
            "previously_seen_user_triggers": float(fitted_components[0][-1]),
            "new_user_triggers": float(fitted_components[1][-1]),
            "total": float(fitted_components[2][-1]),
        },
    }
    metadata_path = output_dir / "PAPER_prediction_sum_synthetic.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    result_path = output_dir / "PAPER_prediction_sum_synthetic.npz"
    np.savez_compressed(
        result_path,
        seed=seed,
        days=days,
        pilot_days=pilot_days,
        plot_days=plot_days,
        component_names=np.asarray(
            ["previously_seen_user_triggers", "new_user_triggers", "total"]
        ),
        true_components=np.stack(true_components),
        oracle_components=np.stack(oracle_components),
        fitted_components=np.stack(fitted_components),
        true_parameters=np.asarray([alpha, c, beta, r]),
        fitted_parameters=np.asarray(
            [fitted["alpha"], fitted["c"], fitted["beta"], fitted["r"]]
        ),
        parameter_names=np.asarray(["alpha", "c", "beta", "r"]),
        observed_users=observed_users,
        pilot_total_triggers=pilot_total_triggers,
        max_new_count=max_new_count,
    )
    return figure_path, result_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper's synthetic NB-SSP trigger-prediction experiment."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20230903)
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--pilot-days", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=5.0)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument(
        "--max-new-count",
        type=int,
        default=200,
        help="Truncation of the new-user trigger-count distribution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args.output_dir,
        seed=args.seed,
        days=args.days,
        pilot_days=args.pilot_days,
        alpha=args.alpha,
        c=args.c,
        beta=args.beta,
        r=args.r,
        num_restarts=args.num_restarts,
        max_new_count=args.max_new_count,
    )


if __name__ == "__main__":
    main()
