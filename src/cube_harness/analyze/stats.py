"""Shared statistics helpers for experiment reporting.

Kept dependency-free (only the stdlib ``math``) so it can be imported from
``scripts/`` shells and notebooks without pulling in numpy/pandas.
"""

from __future__ import annotations

import math


def binomial_std_err(mean: float, n: int) -> float:
    """Standard error of a binary outcome ``Bernoulli(p)`` given the sample mean.

    Returns ``sqrt(p * (1 - p) / n)``. Returns ``0.0`` when ``n <= 1``
    (the formula degenerates) or ``n == 0``.

    For non-binary data, use :func:`sample_std_err` or :func:`reward_mean_stderr`.
    """
    if n <= 1:
        return 0.0
    return math.sqrt(mean * (1 - mean) / n)


def sample_std_err(rewards: list[float]) -> float:
    """Sample-based standard error: ``std(rewards, ddof=1) / sqrt(n)``.

    Returns ``0.0`` for ``n <= 1``.
    """
    n = len(rewards)
    if n <= 1:
        return 0.0
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    return math.sqrt(var / n)


def reward_mean_stderr(rewards: list[float]) -> tuple[float, float]:
    """Return ``(mean, standard_error)`` for an experiment's rewards.

    Auto-selects the SE formula by the data shape:
      * All values in {0, 1} → :func:`binomial_std_err` (tighter for binary outcomes).
      * Otherwise → :func:`sample_std_err` (sample-std with Bessel correction).

    This is the canonical aggregate used by both ``scripts/experiments_report.py`` and the
    XRay viewer — same data should produce the same accuracy CI in both tools.
    Returns ``(0.0, 0.0)`` for empty input.
    """
    if not rewards:
        return 0.0, 0.0
    mean = sum(rewards) / len(rewards)
    if all(r in (0.0, 1.0) for r in rewards):
        return mean, binomial_std_err(mean, len(rewards))
    return mean, sample_std_err(rewards)
