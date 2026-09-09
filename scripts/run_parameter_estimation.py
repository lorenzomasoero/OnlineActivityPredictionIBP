#!/usr/bin/env python3
"""Run the NB-SSP parameter-estimation study and render its paper figures.

The supplementary material specifies the comparison completely: draw NB-SSP
data for 500 days, fit at pilot horizons 5, 10, 20, and 50 by either maximum
marginal likelihood or the paper's curve criterion, and evaluate prediction of
the remaining new users.  It uses 100 repetitions and the three parameter sets
shown in the published figure.  A separate 365-day draw at
``(alpha, c, beta, r) = (0.5, 30, 2, 5)`` supplies the likelihood profiles.

No executable legacy driver for this study remains.  This reconstruction uses
the paper's sequential predictive representation.  New-user arrivals follow
the stated negative-binomial law; existing users follow the conjugate
beta--negative-binomial update; and a new user's positive first-day count is
drawn by augmenting its beta-function mass with the latent activity rate.  A
bounded rejection sampler draws that rate and an inverse CDF draw produces the
positive negative-binomial count, avoiding the finite count cap in the legacy
model class.

One invocation performs simulation, fitting, evaluation, likelihood profiling,
and plotting.  It writes ``parameter_estimation_results.npz``,
``PAPER_model_accuracy_sample_size.pdf``, and ``APPENDIX_log_like.pdf`` below
``--output-dir``.  Importing this module does no work.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta as beta_function
from scipy.special import betaln, gammaln
from scipy.stats import nbinom

from activity_prediction.experiments import prediction_accuracy
from activity_prediction.models import NBSSP
from activity_prediction.plotting.style import PAPER_RC, save_figure


METHODS = ("Regression", "Likelihood")
PARAMETER_NAMES = ("alpha", "c", "beta", "r")
DEFAULT_PARAMETER_SETS = (
    (0.25, 50.0, 2.0, 5.0),
    (0.50, 50.0, 5.0, 5.0),
    (0.75, 50.0, 20.0, 5.0),
)
DEFAULT_PROFILE_PARAMETERS = (0.5, 30.0, 2.0, 5.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory in which the result archive and figures are written",
    )
    parser.add_argument("--seed", type=int, default=20230903)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--total-days", type=int, default=500)
    parser.add_argument(
        "--pilot-days",
        type=int,
        nargs="+",
        default=(5, 10, 20, 50),
        metavar="D0",
    )
    parser.add_argument("--profile-days", type=int, default=365)
    parser.add_argument("--profile-points", type=int, default=160)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument(
        "--parameter-set",
        type=float,
        nargs=4,
        action="append",
        metavar=("ALPHA", "C", "BETA", "R"),
        help=(
            "NB-SSP parameters in paper order; repeat for multiple panels "
            "(defaults reproduce the three published panels)"
        ),
    )
    parser.add_argument(
        "--profile-parameters",
        type=float,
        nargs=4,
        default=DEFAULT_PROFILE_PARAMETERS,
        metavar=("ALPHA", "C", "BETA", "R"),
    )
    return parser


def _validate_parameters(values: Sequence[float]) -> tuple[float, float, float, float]:
    alpha, c, beta, r = map(float, values)
    if not np.all(np.isfinite((alpha, c, beta, r))):
        raise ValueError("NB-SSP parameters must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if c <= 0.0 or beta <= 0.0 or r <= 0.0:
        raise ValueError("c, beta, and r must be positive")
    return alpha, c, beta, r


def _psi(
    alpha: float,
    r: float,
    start_days: int | np.ndarray,
    additional_days: int | np.ndarray,
) -> np.ndarray:
    start = np.asarray(start_days, dtype=float)
    additional = np.asarray(additional_days, dtype=float)
    return alpha * (
        beta_function(r * start + 1.0, -alpha)
        - beta_function(r * (start + additional) + 1.0, -alpha)
    )


def _draw_arrival_curve(
    generator: np.random.Generator,
    *,
    total_days: int,
    alpha: float,
    c: float,
    beta: float,
    r: float,
) -> np.ndarray:
    """Draw cumulative distinct-user counts from the NB-SSP predictive law."""
    cumulative = np.zeros(total_days + 1, dtype=np.int64)
    for day in range(1, total_days + 1):
        increment_psi = float(_psi(alpha, r, day - 1, 1))
        total_psi = float(_psi(alpha, r, 0, day))
        failure_probability = increment_psi / (beta + total_psi)
        if not 0.0 <= failure_probability < 1.0:
            raise FloatingPointError(
                f"invalid new-user probability {failure_probability} on day {day}"
            )
        new_users = generator.negative_binomial(
            cumulative[day - 1] + c + 1.0,
            1.0 - failure_probability,
        )
        cumulative[day] = cumulative[day - 1] + int(new_users)
    return cumulative


def _draw_new_user_counts(
    generator: np.random.Generator,
    day: int,
    number_users: int,
    *,
    alpha: float,
    r: float,
) -> np.ndarray:
    """Draw positive first-day counts without truncating their support.

    A new user's latent rate has density proportional to
    ``theta**(-1-alpha) * (1-theta)**(r*(day-1))`` times its probability of a
    positive NB count.  A ``Beta(1-alpha, r*(day-1)+1)`` proposal leaves the
    bounded rejection ratio ``[1-(1-theta)**r] / theta``.  Conditional counts
    are then sampled by transforming uniforms through the NB inverse CDF above
    its mass at zero.
    """
    accepted: list[np.ndarray] = []
    remaining = number_users
    envelope = max(1.0, r)
    while remaining:
        batch_size = max(32, 2 * remaining)
        proposals = generator.beta(
            1.0 - alpha,
            r * (day - 1) + 1.0,
            size=batch_size,
        )
        proposals = np.clip(
            proposals, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
        )
        positive_probability = -np.expm1(r * np.log1p(-proposals))
        acceptance = positive_probability / (envelope * proposals)
        selected = proposals[generator.random(batch_size) < acceptance]
        if selected.size:
            selected = selected[:remaining]
            accepted.append(selected)
            remaining -= selected.size

    latent_rates = np.concatenate(accepted)
    probability_zero = np.power(1.0 - latent_rates, r)
    tail_uniforms = np.maximum(
        generator.random(number_users), np.nextafter(0.0, 1.0)
    )
    conditional_survival = (1.0 - probability_zero) * tail_uniforms
    counts = nbinom.isf(
        conditional_survival,
        n=r,
        p=1.0 - latent_rates,
    )
    if np.any(~np.isfinite(counts)) or np.any(counts < 1):
        raise FloatingPointError("failed to draw a positive new-user count")
    return counts.astype(np.int64)


def _draw_pilot_activity(
    generator: np.random.Generator,
    cumulative_users: np.ndarray,
    *,
    pilot_days: int,
    alpha: float,
    r: float,
) -> np.ndarray:
    """Draw daily counts for users observed by ``pilot_days``."""
    number_users = int(cumulative_users[pilot_days])
    activity = np.zeros((pilot_days, number_users), dtype=np.int64)
    cumulative_triggers = np.zeros(number_users, dtype=np.int64)

    for day in range(1, pilot_days + 1):
        previous_users = int(cumulative_users[day - 1])
        current_users = int(cumulative_users[day])
        if previous_users:
            latent_rates = generator.beta(
                cumulative_triggers[:previous_users] - alpha,
                r * (day - 1) + 1.0,
            )
            latent_rates = np.minimum(latent_rates, np.nextafter(1.0, 0.0))
            old_counts = generator.negative_binomial(r, 1.0 - latent_rates)
            activity[day - 1, :previous_users] = old_counts
            cumulative_triggers[:previous_users] += old_counts

        number_new = current_users - previous_users
        if number_new:
            new_counts = _draw_new_user_counts(
                generator,
                day,
                number_new,
                alpha=alpha,
                r=r,
            )
            activity[day - 1, previous_users:current_users] = new_counts
            cumulative_triggers[previous_users:current_users] = new_counts
    return activity


def _draw_nb_ssp(
    generator: np.random.Generator,
    *,
    total_days: int,
    matrix_days: int,
    alpha: float,
    c: float,
    beta: float,
    r: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw an arrival curve and the requested daily-count prefix."""
    cumulative = _draw_arrival_curve(
        generator,
        total_days=total_days,
        alpha=alpha,
        c=c,
        beta=beta,
        r=r,
    )
    activity = _draw_pilot_activity(
        generator,
        cumulative,
        pilot_days=matrix_days,
        alpha=alpha,
        r=r,
    )
    return cumulative, activity


def _negative_log_marginal_likelihood(
    activity_matrix: np.ndarray,
    *,
    alpha: float,
    c: float,
    beta: float,
    r: float,
) -> float:
    """Evaluate Theorem 4.1 in the canonical ``(alpha, c, beta, r)`` order."""
    matrix = np.asarray(activity_matrix, dtype=float)
    matrix = matrix[:, matrix.sum(axis=0) > 0]
    pilot_days, observed_users = matrix.shape
    user_totals = matrix.sum(axis=0)
    psi_pilot = float(_psi(alpha, r, 0, pilot_days))

    log_likelihood = observed_users * np.log(alpha)
    log_likelihood += (c + 1.0) * np.log(beta)
    log_likelihood += gammaln(observed_users + c + 1.0) - gammaln(c + 1.0)
    log_likelihood -= (observed_users + c + 1.0) * np.log(beta + psi_pilot)
    log_likelihood += np.sum(
        gammaln(matrix + r) - gammaln(matrix + 1.0) - gammaln(r)
    )
    log_likelihood += np.sum(
        betaln(user_totals - alpha, r * pilot_days + 1.0)
    )
    return -float(log_likelihood)


def _fit_one_dataset(
    cumulative_users: np.ndarray,
    activity: np.ndarray,
    *,
    pilot_days: np.ndarray,
    total_days: int,
    num_restarts: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accuracies = np.full((len(pilot_days), len(METHODS)), np.nan)
    predictions = np.full_like(accuracies, np.nan)
    fitted_parameters = np.full(
        (len(pilot_days), len(METHODS), len(PARAMETER_NAMES)), np.nan
    )
    failure_messages = np.full((len(pilot_days), len(METHODS)), "", dtype="U512")

    for pilot_index, horizon in enumerate(pilot_days):
        observed_users = int(cumulative_users[horizon])
        true_new_users = int(cumulative_users[total_days] - observed_users)
        if observed_users == 0 or true_new_users == 0:
            failure_messages[pilot_index, :] = "empty pilot or follow-up truth"
            continue

        method_operations = (
            (
                "Regression",
                lambda horizon=horizon: NBSSP.fit_curve(
                    cumulative_users[: horizon + 1],
                    start_day=1,
                    num_restarts=num_restarts,
                    seed=int(generator.integers(0, np.iinfo(np.int32).max)),
                ),
            ),
            (
                "Likelihood",
                lambda horizon=horizon, observed_users=observed_users: (
                    NBSSP.fit_marginal_likelihood(
                        activity[:horizon, :observed_users],
                        num_restarts=num_restarts,
                        seed=int(generator.integers(0, np.iinfo(np.int32).max)),
                    )
                ),
            ),
        )
        for method_index, (method, fit) in enumerate(method_operations):
            try:
                parameters = fit()
                estimate = float(
                    NBSSP.expected_new_users(
                        int(horizon),
                        total_days - int(horizon),
                        observed_users,
                        **parameters,
                    )[-1]
                )
                predictions[pilot_index, method_index] = estimate
                accuracies[pilot_index, method_index] = prediction_accuracy(
                    true_new_users, estimate
                )
                fitted_parameters[pilot_index, method_index] = [
                    parameters[name] for name in PARAMETER_NAMES
                ]
            except Exception as error:  # retain the other method and horizons
                failure_messages[pilot_index, method_index] = (
                    f"{type(error).__name__}: {error}"
                )
                warnings.warn(
                    f"{method} fit at D0={horizon} failed: {error}", stacklevel=2
                )
    return accuracies, predictions, fitted_parameters, failure_messages


def _profile_grids(
    parameters: tuple[float, float, float, float], points: int
) -> dict[str, np.ndarray]:
    alpha, c, beta, r = parameters
    return {
        "alpha": np.linspace(max(0.01, 0.05 * alpha), min(0.99, 1.95 * alpha), points),
        "c": np.linspace(max(0.1, 0.02 * c), 10.0 * c / 3.0, points),
        "beta": np.linspace(max(0.05, 0.025 * beta), 10.0 * beta, points),
        "r": np.linspace(max(0.05, 0.02 * r), 4.0 * r, points),
    }


def _likelihood_profiles(
    activity: np.ndarray,
    parameters: tuple[float, float, float, float],
    *,
    points: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    profile_names = ("beta", "alpha", "c", "r")
    grids_by_name = _profile_grids(parameters, points)
    grids = np.vstack([grids_by_name[name] for name in profile_names])
    values = np.empty_like(grids)
    base = dict(zip(PARAMETER_NAMES, parameters, strict=True))
    for profile_index, name in enumerate(profile_names):
        for point_index, coordinate in enumerate(grids[profile_index]):
            candidate = dict(base)
            candidate[name] = float(coordinate)
            values[profile_index, point_index] = _negative_log_marginal_likelihood(
                activity, **candidate
            )
    return profile_names, grids, values


def _plot_accuracy(
    accuracies: np.ndarray,
    parameter_sets: np.ndarray,
    pilot_days: np.ndarray,
    output_path: Path,
) -> None:
    colors = {"Regression": "#377eb8", "Likelihood": "#ff7f00"}
    line_styles = {"Regression": "-.", "Likelihood": ":"}
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(
            1,
            len(parameter_sets),
            figsize=(4.2 * len(parameter_sets), 3.8),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for parameter_index, (axis, parameters) in enumerate(
            zip(axes.flat, parameter_sets, strict=True)
        ):
            for method_index, method in enumerate(METHODS):
                samples = accuracies[parameter_index, :, :, method_index]
                median = np.nanmedian(samples, axis=0)
                lower = np.nanquantile(samples, 0.1, axis=0)
                upper = np.nanquantile(samples, 0.9, axis=0)
                axis.errorbar(
                    pilot_days,
                    median,
                    yerr=np.vstack((median - lower, upper - median)),
                    color=colors[method],
                    linestyle=line_styles[method],
                    marker="o" if method == "Likelihood" else "s",
                    markersize=3.5,
                    linewidth=1.5,
                    capsize=0,
                    label=method,
                )
            alpha, c, beta, r = parameters
            axis.set_title(
                rf"$\beta,\alpha,c,r=({beta:g},{alpha:g},{c:g},{r:g})$"
            )
            axis.set_xticks(pilot_days)
            axis.set_xlabel(r"Pilot sample size $D_0$")
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.22)
        axes[0, 0].set_ylabel(r"Accuracy $v_{D_0}^{(D_1)}$")
        axes[0, 0].legend(frameon=False, loc="lower right")
        figure.tight_layout()
        save_figure(figure, output_path)


def _plot_likelihood_profiles(
    profile_names: tuple[str, ...],
    grids: np.ndarray,
    values: np.ndarray,
    true_parameters: tuple[float, float, float, float],
    output_path: Path,
) -> None:
    true_by_name = dict(zip(PARAMETER_NAMES, true_parameters, strict=True))
    labels = {"alpha": r"$\alpha$", "c": r"$c$", "beta": r"$\beta$", "r": r"$r$"}
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(2, 2, figsize=(8.4, 5.7))
        for axis, name, grid, profile in zip(
            axes.flat, profile_names, grids, values, strict=True
        ):
            axis.plot(grid, profile, color="#377eb8", linewidth=2.0)
            truth = true_by_name[name]
            truth_value = np.interp(truth, grid, profile)
            axis.plot(
                truth,
                truth_value,
                marker="X",
                color="#e41a1c",
                markersize=8,
                linestyle="none",
                label="True",
            )
            axis.set_xlabel(labels[name])
            axis.set_ylabel(r"$-\log p(Z_{1:D})$")
            axis.grid(alpha=0.18)
        axes[1, 1].legend(frameon=False, loc="upper right")
        figure.tight_layout()
        save_figure(figure, output_path)


def main() -> None:
    args = _parser().parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.total_days < 3 or args.profile_days < 2:
        raise ValueError("simulation horizons are too short")
    if args.profile_points < 8:
        raise ValueError("--profile-points must be at least eight")
    if args.num_restarts < 1:
        raise ValueError("--num-restarts must be positive")
    pilot_days = np.asarray(args.pilot_days, dtype=int)
    if (
        pilot_days.ndim != 1
        or not pilot_days.size
        or np.any(pilot_days < 2)
        or np.any(pilot_days >= args.total_days)
        or np.any(np.diff(pilot_days) <= 0)
    ):
        raise ValueError(
            "--pilot-days must be strictly increasing integers in [2, total-days)"
        )

    parameter_sets = np.asarray(
        args.parameter_set or DEFAULT_PARAMETER_SETS, dtype=float
    )
    validated_sets = np.asarray(
        [_validate_parameters(values) for values in parameter_sets], dtype=float
    )
    profile_parameters = _validate_parameters(args.profile_parameters)

    accuracies = np.full(
        (
            len(validated_sets),
            args.repetitions,
            len(pilot_days),
            len(METHODS),
        ),
        np.nan,
    )
    predictions = np.full_like(accuracies, np.nan)
    fitted_parameters = np.full(
        accuracies.shape + (len(PARAMETER_NAMES),), np.nan
    )
    failure_messages = np.full(accuracies.shape, "", dtype="U512")
    observed_users = np.zeros(
        (len(validated_sets), args.repetitions, len(pilot_days)), dtype=np.int64
    )
    true_new_users = np.zeros_like(observed_users)

    number_datasets = len(validated_sets) * args.repetitions
    seed_sequences = np.random.SeedSequence(args.seed).spawn(number_datasets + 1)
    replicate_seeds = np.asarray(
        [int(sequence.generate_state(1)[0]) for sequence in seed_sequences[1:]],
        dtype=np.uint32,
    ).reshape(len(validated_sets), args.repetitions)

    for parameter_index, values in enumerate(validated_sets):
        alpha, c, beta, r = map(float, values)
        for repetition in range(args.repetitions):
            sequence_index = 1 + parameter_index * args.repetitions + repetition
            generator = np.random.default_rng(seed_sequences[sequence_index])
            cumulative, activity = _draw_nb_ssp(
                generator,
                total_days=args.total_days,
                matrix_days=int(pilot_days[-1]),
                alpha=alpha,
                c=c,
                beta=beta,
                r=r,
            )
            for pilot_index, horizon in enumerate(pilot_days):
                observed_users[parameter_index, repetition, pilot_index] = cumulative[
                    horizon
                ]
                true_new_users[
                    parameter_index, repetition, pilot_index
                ] = cumulative[args.total_days] - cumulative[horizon]
            fitted = _fit_one_dataset(
                cumulative,
                activity,
                pilot_days=pilot_days,
                total_days=args.total_days,
                num_restarts=args.num_restarts,
                generator=generator,
            )
            accuracies[parameter_index, repetition] = fitted[0]
            predictions[parameter_index, repetition] = fitted[1]
            fitted_parameters[parameter_index, repetition] = fitted[2]
            failure_messages[parameter_index, repetition] = fitted[3]

    profile_generator = np.random.default_rng(seed_sequences[0])
    profile_curve, profile_activity = _draw_nb_ssp(
        profile_generator,
        total_days=args.profile_days,
        matrix_days=args.profile_days,
        alpha=profile_parameters[0],
        c=profile_parameters[1],
        beta=profile_parameters[2],
        r=profile_parameters[3],
    )
    profile_names, profile_grids, profile_values = _likelihood_profiles(
        profile_activity,
        profile_parameters,
        points=args.profile_points,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "parameter_estimation_results.npz"
    np.savez_compressed(
        result_path,
        method_names=np.asarray(METHODS),
        parameter_names=np.asarray(PARAMETER_NAMES),
        parameter_sets=validated_sets,
        pilot_days=pilot_days,
        total_days=np.asarray(args.total_days),
        accuracies=accuracies,
        predictions=predictions,
        fitted_parameters=fitted_parameters,
        observed_users=observed_users,
        true_new_users=true_new_users,
        failure_messages=failure_messages,
        profile_parameter_names=np.asarray(profile_names),
        profile_parameters=np.asarray(profile_parameters),
        profile_days=np.asarray(args.profile_days),
        profile_cumulative_users=profile_curve,
        profile_grids=profile_grids,
        profile_negative_log_likelihood=profile_values,
        seed=np.asarray(args.seed),
        replicate_seeds=replicate_seeds,
        repetitions=np.asarray(args.repetitions),
        num_restarts=np.asarray(args.num_restarts),
        reconstruction_note=np.asarray(
            "Sequential NB-SSP predictive sampler reconstructed from the paper; "
            "new-user rates use bounded rejection and positive counts use the "
            "negative-binomial inverse CDF without a finite count cap."
        ),
    )
    _plot_accuracy(
        accuracies,
        validated_sets,
        pilot_days,
        args.output_dir / "PAPER_model_accuracy_sample_size.pdf",
    )
    _plot_likelihood_profiles(
        profile_names,
        profile_grids,
        profile_values,
        profile_parameters,
        args.output_dir / "APPENDIX_log_like.pdf",
    )
    print(f"wrote {result_path}")
    print(f"wrote {args.output_dir / 'PAPER_model_accuracy_sample_size.pdf'}")
    print(f"wrote {args.output_dir / 'APPENDIX_log_like.pdf'}")


if __name__ == "__main__":
    main()
