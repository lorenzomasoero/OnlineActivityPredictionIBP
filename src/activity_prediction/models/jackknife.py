"""Jackknife species-richness extrapolators used in the paper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _require_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _harmonic_number(sample_size: int) -> float:
    if sample_size < 20:
        return float(np.sum(1.0 / np.arange(1, sample_size + 1)))
    return float(
        np.log(sample_size)
        + np.euler_gamma
        + 1.0 / (2 * sample_size)
        - 1.0 / (12 * sample_size**2)
        + 1.0 / (120 * sample_size**4)
    )


def _harmonic_increment(extrapolation_size: int, sample_size: int) -> float:
    return _harmonic_number(extrapolation_size + sample_size - 1) - _harmonic_number(
        sample_size - 1
    )


def _missed_species(
    sample_size: int,
    extrapolation_size: int,
    sfs: np.ndarray,
    order: int,
) -> float:
    if sample_size == 2:
        order = min(order, 3)
    if sample_size == 1:
        order = 1

    padded_sfs = np.concatenate(([0.0], sfs))
    if padded_sfs.size < order + 1:
        padded_sfs = np.pad(padded_sfs, (0, order + 1 - padded_sfs.size))

    delta = _harmonic_increment(extrapolation_size, sample_size)
    n = sample_size

    if order == 1:
        missed = (-1.0 + n) / n * delta * padded_sfs[1]
    elif order == 2:
        missed = (
            (
                ((1 + 2 * (-2 + n) * n) * delta) / (n * (-3 + 2 * n))
                + ((-2 + n) * (-1 + n) * delta**2) / (n * (-3 + 2 * n))
            )
            * padded_sfs[1]
            + (
                -(2 * (-2 + n) ** 2 * delta) / ((-1 + n) * n * (-3 + 2 * n))
                - (2 * (-2 + n) ** 2 * delta**2) / (n * (-3 + 2 * n))
            )
            * padded_sfs[2]
        )
    elif order == 3:
        n = float(n)
        denominator = (
            (-1 + n) ** 2
            * n
            * (-5 + 2 * n)
            * (-3 + 2 * n)
            * (11 - 12 * n + 3 * n**2)
        )
        missed = (
            delta
            * (
                -127
                + 628 * n
                - 1386 * n**2
                + 1722 * n**3
                - 1269 * n**4
                + 546 * n**5
                - 126 * n**6
                + 12 * n**7
                + (-1 + n)
                * (
                    -10
                    - 87 * n
                    + 310 * n**2
                    - 372 * n**3
                    + 210 * n**4
                    - 57 * n**5
                    + 6 * n**6
                )
                * delta
                + 2
                * (-2 + n) ** 2
                * (-1 + n) ** 3
                * (6 - 5 * n + n**2)
                * delta**2
            )
            * padded_sfs[1]
            / denominator
            + delta
            * (
                -2
                * (
                    -248
                    + 402 * n
                    - 61 * n**2
                    - 249 * n**3
                    + 195 * n**4
                    - 57 * n**5
                    + 6 * n**6
                )
                - 2
                * (-1 + n)
                * (
                    -560
                    + 1130 * n
                    - 731 * n**2
                    + 55 * n**3
                    + 127 * n**4
                    - 51 * n**5
                    + 6 * n**6
                )
                * delta
                + 2
                * (-1 + n) ** 2
                * (6 - 5 * n + n**2)
                * (52 - 78 * n + 38 * n**2 - 6 * n**3)
                * delta**2
            )
            * padded_sfs[2]
            / denominator
            + delta
            * (
                6 * (3 - 2 * n) ** 2 * (-3 + n) ** 3
                + 6
                * (-3 + n) ** 3
                * (-1 + n)
                * (15 - 19 * n + 6 * n**2)
                * delta
                + 6
                * (-3 + n) ** 2
                * (-1 + n) ** 2
                * (-3 + 2 * n)
                * (6 - 5 * n + n**2)
                * delta**2
            )
            * padded_sfs[3]
            / denominator
        )
    else:
        n = float(n)
        denominator_12 = (
            (-2 + n)
            * (-1 + n) ** 3
            * n
            * (-7 + 2 * n)
            * (-5 + 2 * n)
            * (-3 + 2 * n)
            * (5 - 5 * n + n**2)
            * (26 - 18 * n + 3 * n**2)
            * (11 - 12 * n + 3 * n**2)
        )
        denominator_3 = (
            (-2 + n)
            * (-1 + n) ** 3
            * n
            * (-5 + 2 * n)
            * (-3 + 2 * n)
            * (5 - 5 * n + n**2)
            * (26 - 18 * n + 3 * n**2)
            * (11 - 12 * n + 3 * n**2)
        )
        missed = (
            delta
            * (
                2
                * (
                    79138
                    - 768163 * n
                    + 3305209 * n**2
                    - 8455556 * n**3
                    + 14456396 * n**4
                    - 17528889 * n**5
                    + 15570611 * n**6
                    - 10302336 * n**7
                    + 5104884 * n**8
                    - 1885839 * n**9
                    + 511449 * n**10
                    - 98781 * n**11
                    + 12849 * n**12
                    - 1008 * n**13
                    + 36 * n**14
                )
                + (-1 + n)
                * (
                    -142984
                    + 984880 * n
                    - 3219708 * n**2
                    + 6550921 * n**3
                    - 9150649 * n**4
                    + 9174455 * n**5
                    - 6742096 * n**6
                    + 3659027 * n**7
                    - 1462263 * n**8
                    + 424335 * n**9
                    - 86856 * n**10
                    + 11874 * n**11
                    - 972 * n**12
                    + 36 * n**13
                )
                * delta
                + 2
                * (-1 + n) ** 2
                * (6 - 5 * n + n**2)
                * (
                    -2288
                    + 3592 * n
                    + 10090 * n**2
                    - 36645 * n**3
                    + 50160 * n**4
                    - 39031 * n**5
                    + 19017 * n**6
                    - 5922 * n**7
                    + 1147 * n**8
                    - 126 * n**9
                    + 6 * n**10
                )
                * delta**2
                + (-4 + n)
                * (-1 + n) ** 3
                * (6 - 5 * n + n**2) ** 2
                * (
                    184
                    - 606 * n
                    + 803 * n**2
                    - 549 * n**3
                    + 204 * n**4
                    - 39 * n**5
                    + 3 * n**6
                )
                * delta**3
            )
            * padded_sfs[1]
            / denominator_12
            + delta
            * (
                2
                * (
                    806336
                    - 4114800 * n
                    + 9788448 * n**2
                    - 14623884 * n**3
                    + 15584978 * n**4
                    - 12651005 * n**5
                    + 8033192 * n**6
                    - 3985285 * n**7
                    + 1516278 * n**8
                    - 429717 * n**9
                    + 87120 * n**10
                    - 11877 * n**11
                    + 972 * n**12
                    - 36 * n**13
                )
                - 2
                * (-1 + n)
                * (
                    -1762976
                    + 8275744 * n
                    - 17540568 * n**2
                    + 22472256 * n**3
                    - 19868590 * n**4
                    + 13296025 * n**5
                    - 7229740 * n**6
                    + 3298055 * n**7
                    - 1236311 * n**8
                    + 360392 * n**9
                    - 76491 * n**10
                    + 10944 * n**11
                    - 936 * n**12
                    + 36 * n**13
                )
                * delta
                - 4
                * (-1 + n) ** 2
                * (6 - 5 * n + n**2)
                * (
                    -117832
                    + 426816 * n
                    - 645654 * n**2
                    + 513018 * n**3
                    - 207385 * n**4
                    + 15319 * n**5
                    + 25646 * n**6
                    - 13195 * n**7
                    + 3075 * n**8
                    - 366 * n**9
                    + 18 * n**10
                )
                * delta**2
                - 2
                * (-4 + n)
                * (-1 + n) ** 3
                * (6 - 5 * n + n**2) ** 2
                * (
                    3176
                    - 8768 * n
                    + 9854 * n**2
                    - 5764 * n**3
                    + 1850 * n**4
                    - 309 * n**5
                    + 21 * n**6
                )
                * delta**3
            )
            * padded_sfs[2]
            / denominator_12
            + delta
            * (
                12
                * (3 - 2 * n) ** 2
                * (-3 + n)
                * (
                    24678
                    - 73356 * n
                    + 88865 * n**2
                    - 54881 * n**3
                    + 16400 * n**4
                    - 464 * n**5
                    - 1289 * n**6
                    + 419 * n**7
                    - 57 * n**8
                    + 3 * n**9
                )
                + 6
                * (-1 + n)
                * (9 - 9 * n + 2 * n**2)
                * (
                    -354348
                    + 1276800 * n
                    - 1961648 * n**2
                    + 1662242 * n**3
                    - 829199 * n**4
                    + 230491 * n**5
                    - 21616 * n**6
                    - 6925 * n**7
                    + 2631 * n**8
                    - 354 * n**9
                    + 18 * n**10
                )
                * delta
                + 6
                * (-1 + n) ** 2
                * (-3 + 2 * n)
                * (6 - 5 * n + n**2)
                * (
                    138996
                    - 425226 * n
                    + 547888 * n**2
                    - 385039 * n**3
                    + 158306 * n**4
                    - 36911 * n**5
                    + 3767 * n**6
                    + 208 * n**7
                    - 87 * n**8
                    + 6 * n**9
                )
                * delta**2
                + 6
                * (-4 + n)
                * (-1 + n) ** 3
                * (-3 + 2 * n)
                * (6 - 5 * n + n**2) ** 2
                * (
                    -1494
                    + 2874 * n
                    - 2121 * n**2
                    + 758 * n**3
                    - 132 * n**4
                    + 9 * n**5
                )
                * delta**3
            )
            * padded_sfs[3]
            / denominator_3
            + delta
            * (
                -12
                * (3 - 2 * n) ** 2
                * (-4 + n) ** 4
                * (-3 + n)
                * (11 - 12 * n + 3 * n**2) ** 2
                - 12
                * (-4 + n) ** 4
                * (-1 + n)
                * (9 - 9 * n + 2 * n**2)
                * (
                    -803
                    + 2196 * n
                    - 2363 * n**2
                    + 1249 * n**3
                    - 324 * n**4
                    + 33 * n**5
                )
                * delta
                - 24
                * (-4 + n) ** 4
                * (-1 + n) ** 2
                * (-3 + 2 * n)
                * (6 - 5 * n + n**2)
                * (143 - 299 * n + 228 * n**2 - 75 * n**3 + 9 * n**4)
                * delta**2
                - 12
                * (-4 + n) ** 4
                * (-1 + n) ** 3
                * (-3 + 2 * n)
                * (6 - 5 * n + n**2) ** 2
                * (11 - 12 * n + 3 * n**2)
                * delta**3
            )
            * padded_sfs[4]
            / denominator_12
        )
    return float(missed)


def predict_jackknife(
    sample_size: int,
    extrapolation_size: int,
    sfs: Sequence[int] | np.ndarray,
    cumulative_counts: Sequence[float] | np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """Return the observed and extrapolated accumulation curve."""
    sample_size = _require_nonnegative_integer(sample_size, name="sample_size")
    if sample_size == 0:
        raise ValueError("sample_size must be positive")
    extrapolation_size = _require_nonnegative_integer(
        extrapolation_size, name="extrapolation_size"
    )
    order = _require_nonnegative_integer(order, name="order")
    if order not in (1, 2, 3, 4):
        raise ValueError("order must be one of 1, 2, 3, or 4")

    frequencies = np.asarray(sfs, dtype=float)
    if frequencies.ndim != 1:
        raise ValueError("sfs must be one-dimensional")
    if frequencies.size > sample_size:
        raise ValueError("sfs cannot have more entries than sample_size")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("sfs must contain finite, non-negative counts")

    observed = np.asarray(cumulative_counts, dtype=float)
    if observed.ndim != 1:
        raise ValueError("cumulative_counts must be one-dimensional")
    if observed.size < sample_size + 1:
        raise ValueError("cumulative_counts must include days 0 through sample_size")
    if not np.all(np.isfinite(observed[: sample_size + 1])):
        raise ValueError("cumulative_counts must contain only finite values")
    if np.any(observed[: sample_size + 1] < 0) or np.any(
        np.diff(observed[: sample_size + 1]) < 0
    ):
        raise ValueError("cumulative_counts must be non-negative and non-decreasing")

    if frequencies.size < order:
        frequencies = np.pad(frequencies, (0, order - frequencies.size))

    prediction = np.zeros(sample_size + extrapolation_size + 1)
    prediction[: sample_size + 1] = observed[: sample_size + 1]
    prediction[sample_size + 1 :] = observed[sample_size] + np.asarray(
        [
            _missed_species(sample_size, step, frequencies, order)
            for step in range(1, extrapolation_size + 1)
        ]
    )
    return prediction
