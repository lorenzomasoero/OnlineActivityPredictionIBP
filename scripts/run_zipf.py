#!/usr/bin/env python3
"""Run the paper's Zipf--Poisson study and render both paper figures.

The manuscript specifies the data-generating process exactly: user ``i`` has
daily activation probability ``i**(-tau)``; on an active day its trigger count
is zero-truncated Poisson with rate ``1 + m_previous / day``.  The historical
``draw_from_zipf.py`` retained in this repository is only a TODO, so this file
is a faithful reconstruction from Section 5.1 rather than a transcription of
an executable legacy script.  To avoid a dense population-by-day Bernoulli
array, it first draws every user's Binomial number of active days and then
places those days uniformly.  This is distributionally identical to the
independent Bernoulli construction in the paper.

The displayed historical figure contains NB-SSP, Be-SSP, TG-SSP, IBP, BB, LP,
GT, J1, and J4.  Those are reproduced here.  LP uses the same rare-histogram
moment program as UnseenEST, solved with SciPy/HiGHS so this core experiment
does not require CVXOPT.  HBG is discussed in the manuscript but is not shown
in that figure and requires the optional Stan toolchain, so it is not run.

One invocation performs simulation, fitting, evaluation, and plotting.  It
writes ``zipf_results.npz``, ``PAPER_zipf_accuracy.pdf``, and
``PAPER_zipf_sums.pdf`` below ``--output-dir``.  Importing this module does no
work.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.optimize import linprog
from scipy.stats import binom

from activity_prediction.experiments import prediction_accuracy
from activity_prediction.models import (
    BeSSP,
    BetaBinomial,
    IBP,
    NBSSP,
    TGSSP,
    predict_good_toulmin,
    predict_jackknife,
)
from activity_prediction.plotting.style import COLORS, PAPER_RC, save_figure


METHODS = (
    "NB-SSP",
    "Be-SSP",
    "TG-SSP",
    "IBP",
    "BB",
    "LP",
    "GT",
    "J1",
    "J4",
)
DEFAULT_TAILS = (0.6, 0.7, 0.8, 0.9)


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
    parser.add_argument("--population-size", type=int, default=1_000_000)
    parser.add_argument("--pilot-days", type=int, default=5)
    parser.add_argument("--followup-days", type=int, default=50)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument(
        "--tail-parameters",
        type=float,
        nargs="+",
        default=DEFAULT_TAILS,
        metavar="TAU",
    )
    parser.add_argument(
        "--lp-kappa",
        type=float,
        default=0.5,
        help="rare-frequency cutoff used by the SciPy UnseenEST program",
    )
    return parser


def _positive_poisson(
    generator: np.random.Generator, rates: np.ndarray
) -> np.ndarray:
    """Draw independent Poisson variables conditional on being positive."""
    rates = np.asarray(rates, dtype=float)
    draws = generator.poisson(rates)
    missing = draws == 0
    while np.any(missing):
        draws[missing] = generator.poisson(rates[missing])
        missing = draws == 0
    return draws


def _draw_zipf_poisson(
    generator: np.random.Generator,
    *,
    tail: float,
    population_size: int,
    total_days: int,
) -> np.ndarray:
    """Draw the nonzero columns of one Zipf--Poisson activity matrix.

    Conditional on a user's Binomial active-day count, its active locations
    are a uniform subset of the horizon.  This replaces a very large dense
    Bernoulli draw without changing its law.  Trigger counts are then sampled
    in chronological order because their Poisson rate depends on cumulative
    past activity.
    """
    user_indices = np.arange(1, population_size + 1, dtype=float)
    activation_probabilities = np.power(user_indices, -tail)
    active_day_counts = generator.binomial(total_days, activation_probabilities)
    retained = np.flatnonzero(active_day_counts)
    retained_counts = active_day_counts[retained]

    active_days = np.zeros((total_days, retained.size), dtype=bool)
    for user, count in enumerate(retained_counts):
        selected = generator.choice(total_days, size=int(count), replace=False)
        active_days[selected, user] = True

    activity = np.zeros(active_days.shape, dtype=np.int32)
    cumulative_triggers = np.zeros(retained.size, dtype=np.int64)
    for zero_based_day in range(total_days):
        active = active_days[zero_based_day]
        if not np.any(active):
            continue
        day = zero_based_day + 1
        rates = 1.0 + cumulative_triggers[active] / day
        counts = _positive_poisson(generator, rates)
        activity[zero_based_day, active] = counts
        cumulative_triggers[active] += counts
    return activity


def _cumulative_users(activity: np.ndarray) -> np.ndarray:
    first_days = np.argmax(activity > 0, axis=0)
    arrivals = np.bincount(first_days, minlength=activity.shape[0])
    return np.concatenate(([0], np.cumsum(arrivals))).astype(int)


def _site_frequency_spectrum(binary_pilot: np.ndarray) -> np.ndarray:
    frequencies = binary_pilot.sum(axis=0)
    observed = frequencies[frequencies > 0]
    return np.bincount(observed, minlength=binary_pilot.shape[0] + 1)[1:]


def _first_trigger_counts(binary_pilot: np.ndarray) -> np.ndarray:
    observed = binary_pilot[:, binary_pilot.any(axis=0)]
    first_days = np.argmax(observed, axis=0) + 1
    return np.bincount(first_days, minlength=binary_pilot.shape[0] + 1)[1:]


def _scipy_unseen_prediction(
    sample_size: int,
    followup_size: int,
    sfs: np.ndarray,
    *,
    kappa: float,
) -> float:
    """Return the UnseenEST future-new-user mean using SciPy's LP solver.

    The unknown rare-probability histogram is fitted by weighted absolute
    moment error, with its first moment fixed at the empirical rare mass.  The
    frequent part of the spectrum remains empirical, as in the historical
    UnseenEST implementation bundled with the project.
    """
    rare_cutoff = max(1, min(len(sfs), int(sample_size * kappa)))
    rare_spectrum = np.asarray(sfs[:rare_cutoff], dtype=float)
    maximum_probability = rare_cutoff / sample_size
    minimum_probability = 1.0 / (100.0 * sample_size)
    grid_size = int(
        np.ceil(np.log(maximum_probability / minimum_probability) / np.log(1.01))
    ) + 1
    grid = minimum_probability * np.power(1.01, np.arange(grid_size))

    number_moments = rare_cutoff + int(np.ceil(np.sqrt(rare_cutoff)))
    target = np.pad(rare_spectrum, (0, number_moments - rare_cutoff))
    orders = np.arange(1, number_moments + 1)
    moment_matrix = binom.pmf(orders[:, None], sample_size, grid[None, :])

    number_variables = grid_size + 2 * number_moments
    objective = np.zeros(number_variables)
    weights = 1.0 / np.sqrt(target + 1.0)
    objective[grid_size : grid_size + number_moments] = weights
    objective[grid_size + number_moments :] = weights

    constraints = np.zeros((number_moments + 1, number_variables))
    constraints[:number_moments, :grid_size] = moment_matrix
    constraints[
        np.arange(number_moments), grid_size + np.arange(number_moments)
    ] = 1.0
    constraints[
        np.arange(number_moments),
        grid_size + number_moments + np.arange(number_moments),
    ] = -1.0
    constraints[-1, :grid_size] = grid

    rare_mass = float(
        np.inner(np.arange(1, rare_cutoff + 1), rare_spectrum) / sample_size
    )
    right_hand_side = np.concatenate((target, [rare_mass]))
    result = linprog(
        objective,
        A_ub=np.concatenate(
            (np.ones(grid_size), np.zeros(2 * number_moments))
        )[None, :],
        b_ub=np.asarray([65_000_000.0]),
        A_eq=constraints,
        b_eq=right_hand_side,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"UnseenEST linear program failed: {result.message}")

    rare_histogram = result.x[:grid_size]
    frequent_frequencies = np.arange(rare_cutoff + 1, len(sfs) + 1)
    frequent_histogram = np.asarray(sfs[rare_cutoff:], dtype=float)
    frequent_grid = frequent_frequencies / sample_size
    probabilities = np.concatenate((grid, frequent_grid))
    histogram = np.concatenate((rare_histogram, frequent_histogram))
    survival_at_pilot = np.power(1.0 - probabilities, sample_size)
    discovery_in_followup = 1.0 - np.power(1.0 - probabilities, followup_size)
    return float(np.inner(histogram, survival_at_pilot * discovery_in_followup))


def _fit_predictions(
    activity: np.ndarray,
    *,
    pilot_days: int,
    followup_days: int,
    num_restarts: int,
    lp_kappa: float,
    generator: np.random.Generator,
) -> tuple[dict[str, float], float, dict[str, str]]:
    pilot_binary_all = activity[:pilot_days] > 0
    pilot_observed = pilot_binary_all.any(axis=0)
    pilot_activity = activity[:pilot_days, pilot_observed]
    pilot_binary = pilot_binary_all[:, pilot_observed]
    observed_users = int(pilot_observed.sum())
    cumulative = _cumulative_users(activity)[: pilot_days + 1]
    sfs = _site_frequency_spectrum(pilot_binary)
    first_counts = _first_trigger_counts(pilot_binary)

    predictions: dict[str, float] = {}
    failures: dict[str, str] = {}
    nb_total_prediction = np.nan

    def record(method: str, operation: Callable[[], float]) -> None:
        try:
            predictions[method] = float(operation())
        except Exception as error:  # keep independent competitors evaluable
            failures[method] = f"{type(error).__name__}: {error}"
            predictions[method] = np.nan
            warnings.warn(f"{method} fit failed: {error}", stacklevel=2)

    nb_parameters: dict[str, float] | None = None
    try:
        nb_parameters = NBSSP.fit_marginal_likelihood(
            pilot_activity,
            num_restarts=num_restarts,
            seed=int(generator.integers(0, np.iinfo(np.int32).max)),
        )
        predictions["NB-SSP"] = float(
            NBSSP.expected_new_users(
                pilot_days,
                followup_days,
                observed_users,
                **nb_parameters,
            )[-1]
        )
        nb_total_prediction = float(
            NBSSP.expected_total_triggers(
                pilot_days,
                followup_days,
                observed_users,
                int(pilot_activity.sum()),
                **nb_parameters,
            )[-1]
        )
    except Exception as error:
        failures["NB-SSP"] = f"{type(error).__name__}: {error}"
        predictions["NB-SSP"] = np.nan
        warnings.warn(f"NB-SSP fit failed: {error}", stacklevel=2)

    be = BeSSP()

    def be_prediction() -> float:
        parameters = be.fit_marginal_likelihood(
            sfs,
            pilot_days,
            num_restarts=num_restarts,
            seed=int(generator.integers(0, np.iinfo(np.int32).max)),
        )
        return float(
            be.expected_new_users(
                pilot_days, followup_days, observed_users, **parameters
            )[-1]
        )

    record("Be-SSP", be_prediction)

    tg = TGSSP()

    def tg_prediction() -> float:
        parameters = tg.fit_marginal_likelihood(
            first_counts,
            num_restarts=num_restarts,
            seed=int(generator.integers(0, np.iinfo(np.int32).max)),
        )
        return float(
            tg.expected_new_users(
                pilot_days, followup_days, observed_users, **parameters
            )[-1]
        )

    record("TG-SSP", tg_prediction)

    def ibp_prediction() -> float:
        parameters = IBP.fit_marginal_likelihood(
            sfs,
            pilot_days,
            num_restarts=num_restarts,
            seed=int(generator.integers(0, np.iinfo(np.int32).max)),
        )
        return float(
            IBP.expected_new_users(
                pilot_days, followup_days, **parameters
            )[-1]
        )

    record("IBP", ibp_prediction)

    bb = BetaBinomial()

    def bb_prediction() -> float:
        parameters = bb.fit_marginal_likelihood(
            sfs,
            pilot_days,
            num_restarts=num_restarts,
            seed=int(generator.integers(0, np.iinfo(np.int32).max)),
        )
        return float(
            bb.expected_new_users(
                pilot_days,
                followup_days,
                int(sfs[0]),
                **parameters,
            )[-1]
        )

    record("BB", bb_prediction)
    record(
        "LP",
        lambda: _scipy_unseen_prediction(
            pilot_days, followup_days, sfs, kappa=lp_kappa
        ),
    )
    record(
        "GT",
        lambda: predict_good_toulmin(
            pilot_days, followup_days, sfs, cumulative
        )[0][-1]
        - observed_users,
    )
    for order in (1, 4):
        record(
            f"J{order}",
            lambda order=order: predict_jackknife(
                pilot_days,
                followup_days,
                sfs,
                cumulative,
                order=order,
            )[-1]
            - observed_users,
        )

    return predictions, nb_total_prediction, failures


def _plot_new_user_accuracy(
    accuracies: np.ndarray,
    tails: np.ndarray,
    output_path: Path,
) -> None:
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(
            1,
            len(tails),
            figsize=(4.0 * len(tails), 4.2),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for tail_index, (axis, tail) in enumerate(zip(axes.flat, tails, strict=True)):
            values = []
            for method_index in range(len(METHODS)):
                finite = accuracies[tail_index, :, method_index]
                finite = finite[np.isfinite(finite)]
                values.append(finite if finite.size else np.array([np.nan]))
            boxes = axis.boxplot(
                values,
                labels=METHODS,
                vert=False,
                patch_artist=True,
                widths=0.56,
                medianprops={"color": "black", "linewidth": 1.4},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.45},
            )
            for patch, method in zip(boxes["boxes"], METHODS, strict=True):
                patch.set_facecolor(COLORS[method])
                patch.set_alpha(0.75)
                patch.set_edgecolor("black")
            axis.set_title(rf"$\tau={tail:g}$")
            axis.set_xlim(0.0, 1.01)
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axis.grid(axis="x", alpha=0.22)
            axis.set_xlabel(r"Prediction accuracy $v_{D_0}^{(D_1)}$")
        axes[0, 0].set_ylabel("Methods")
        figure.tight_layout()
        save_figure(figure, output_path)


def _plot_total_trigger_survival(
    accuracies: np.ndarray,
    tails: np.ndarray,
    output_path: Path,
) -> None:
    thresholds = np.linspace(0.0, 1.0, 201)
    line_styles = (":", "--", "-.", "-")
    colors = ("#e41a1c", "#4daf4a", "#377eb8", "#ff7f00")
    with plt.rc_context(PAPER_RC):
        figure, axis = plt.subplots(figsize=(6.5, 4.2))
        for tail_index, tail in enumerate(tails):
            values = accuracies[tail_index]
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            survival = np.mean(values[:, None] >= thresholds[None, :], axis=0)
            axis.plot(
                thresholds,
                survival,
                color=colors[tail_index % len(colors)],
                linestyle=line_styles[tail_index % len(line_styles)],
                linewidth=2.0,
                label=rf"$\tau={tail:g}$",
            )
        axis.set_xlim(0.5, 1.0)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel(r"Accuracy threshold $v$")
        axis.set_ylabel(r"$\widehat{\Pr}(\widetilde v_{D_0}^{(D_1)}\geq v)$")
        axis.set_title(r"Prediction accuracy of $T_{D_0}^{(D_1)}$")
        axis.grid(alpha=0.22)
        axis.legend(loc="lower left", ncol=2, frameon=False)
        figure.tight_layout()
        save_figure(figure, output_path)


def main() -> None:
    args = _parser().parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.population_size < 1:
        raise ValueError("--population-size must be positive")
    if args.pilot_days < 2 or args.followup_days < 1:
        raise ValueError("--pilot-days must be at least two and follow-up positive")
    if args.num_restarts < 1:
        raise ValueError("--num-restarts must be positive")
    if not 0.0 < args.lp_kappa < 1.0:
        raise ValueError("--lp-kappa must lie in (0, 1)")
    tails = np.asarray(args.tail_parameters, dtype=float)
    if tails.ndim != 1 or not tails.size or np.any(~np.isfinite(tails)):
        raise ValueError("--tail-parameters must contain finite values")
    if np.any(tails <= 0.0):
        raise ValueError("--tail-parameters must be positive")

    total_days = args.pilot_days + args.followup_days
    number_methods = len(METHODS)
    predictions = np.full(
        (len(tails), args.repetitions, number_methods), np.nan, dtype=float
    )
    accuracies = np.full_like(predictions, np.nan)
    total_predictions = np.full((len(tails), args.repetitions), np.nan)
    total_accuracies = np.full_like(total_predictions, np.nan)
    true_new_users = np.zeros((len(tails), args.repetitions), dtype=np.int64)
    true_total_triggers = np.zeros_like(true_new_users)
    observed_users = np.zeros_like(true_new_users)
    failures = np.full(
        (len(tails), args.repetitions, number_methods), "", dtype="U512"
    )

    seed_sequences = np.random.SeedSequence(args.seed).spawn(
        len(tails) * args.repetitions
    )
    replicate_seeds = np.asarray(
        [int(sequence.generate_state(1)[0]) for sequence in seed_sequences],
        dtype=np.uint32,
    ).reshape(len(tails), args.repetitions)

    for tail_index, tail in enumerate(tails):
        for repetition in range(args.repetitions):
            generator = np.random.default_rng(
                seed_sequences[tail_index * args.repetitions + repetition]
            )
            activity = _draw_zipf_poisson(
                generator,
                tail=float(tail),
                population_size=args.population_size,
                total_days=total_days,
            )
            pilot_active = np.any(activity[: args.pilot_days] > 0, axis=0)
            future_active = np.any(activity[args.pilot_days :] > 0, axis=0)
            observed = int(pilot_active.sum())
            truth = int(np.sum(~pilot_active & future_active))
            trigger_truth = int(activity[args.pilot_days :].sum())
            if observed == 0 or truth == 0 or trigger_truth == 0:
                warnings.warn(
                    f"tau={tail:g}, repetition={repetition}: empty pilot or truth",
                    stacklevel=2,
                )
                continue

            estimates, trigger_estimate, fit_failures = _fit_predictions(
                activity,
                pilot_days=args.pilot_days,
                followup_days=args.followup_days,
                num_restarts=args.num_restarts,
                lp_kappa=args.lp_kappa,
                generator=generator,
            )
            observed_users[tail_index, repetition] = observed
            true_new_users[tail_index, repetition] = truth
            true_total_triggers[tail_index, repetition] = trigger_truth
            total_predictions[tail_index, repetition] = trigger_estimate
            if np.isfinite(trigger_estimate):
                total_accuracies[tail_index, repetition] = prediction_accuracy(
                    trigger_truth, trigger_estimate
                )

            for method_index, method in enumerate(METHODS):
                estimate = estimates[method]
                predictions[tail_index, repetition, method_index] = estimate
                if np.isfinite(estimate):
                    accuracies[tail_index, repetition, method_index] = (
                        prediction_accuracy(truth, estimate)
                    )
                failures[tail_index, repetition, method_index] = fit_failures.get(
                    method, ""
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "zipf_results.npz"
    np.savez_compressed(
        result_path,
        method_names=np.asarray(METHODS),
        tail_parameters=tails,
        predictions=predictions,
        accuracies=accuracies,
        total_trigger_predictions=total_predictions,
        total_trigger_accuracies=total_accuracies,
        observed_users=observed_users,
        true_new_users=true_new_users,
        true_total_triggers=true_total_triggers,
        failure_messages=failures,
        replicate_seeds=replicate_seeds,
        seed=np.asarray(args.seed),
        repetitions=np.asarray(args.repetitions),
        population_size=np.asarray(args.population_size),
        pilot_days=np.asarray(args.pilot_days),
        followup_days=np.asarray(args.followup_days),
        num_restarts=np.asarray(args.num_restarts),
        lp_kappa=np.asarray(args.lp_kappa),
        reconstruction_note=np.asarray(
            "Binomial active-day counts plus uniform locations are equivalent "
            "to daily Bernoulli draws; LP is the UnseenEST moment program "
            "solved by scipy.optimize.linprog."
        ),
    )
    _plot_new_user_accuracy(
        accuracies,
        tails,
        args.output_dir / "PAPER_zipf_accuracy.pdf",
    )
    _plot_total_trigger_survival(
        total_accuracies,
        tails,
        args.output_dir / "PAPER_zipf_sums.pdf",
    )
    print(f"wrote {result_path}")
    print(f"wrote {args.output_dir / 'PAPER_zipf_accuracy.pdf'}")
    print(f"wrote {args.output_dir / 'PAPER_zipf_sums.pdf'}")


if __name__ == "__main__":
    main()
