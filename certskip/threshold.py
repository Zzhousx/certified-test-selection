# -*- coding: utf-8 -*-
"""Threshold rules.

The threshold is the whole statistical content of the method. Everything else is
plumbing. Three rules are provided and they answer different questions:

    crc  -- split conformal risk control. Guarantees E[L] <= eps, where the
            expectation runs over the calibration draw. A single draw can exceed eps,
            and in practice a large fraction of draws do.

    fpc  -- Exact finite-population hypergeometric inversion. Guarantees, with
            confidence 1-delta, that the unexecuted remainder contains at most an
            eps share of all failures in the finite library. This is the primary
            single-release rule.

    cp   -- Standard independent-sample binomial tolerance reference (legacy name).
            The pipeline checks its rank against the finite-library tail before
            issuing a library-level statement.

The CRC loss uses failures outside calibration as its denominator. The FPC and
finite-checked pipeline statements use all library failures. All are conditioned
on failure, so their effective sample size is the number of failures observed, not
the number of scenarios run.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from fractions import Fraction

from scipy.stats import hypergeom as _hypergeom
from scipy.stats import binom as _binom

__all__ = ["crc_threshold", "cp_threshold", "finite_population_threshold",
           "finite_population_index", "get_rule", "skip_mask",
           "min_failures_for", "SKIP_NOTHING", "binomial_tolerance_index",
           "checked_tolerance_index"]

# A threshold of -inf skips nothing. This is the safe default whenever the rule has
# not seen enough failures to support any claim about the low-scoring region.
SKIP_NOTHING = -np.inf


def _failure_scores(scores: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes)
    if scores.shape != outcomes.shape:
        raise ValueError("scores and outcomes must have the same shape")
    return np.sort(scores[outcomes == 1])


def skip_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return the conservative skip decision for a scalar threshold.

    The comparison is deliberately strict.  If several scenarios share the
    calibration order statistic, placing the entire tie block below the cut can
    miss many more failures than its rank represents.  Treating the tie block as
    executable preserves the rank argument without randomized tie breaking.
    """
    return np.asarray(scores, dtype=float) < float(threshold)


def crc_threshold(scores: np.ndarray, outcomes: np.ndarray, eps: float) -> float:
    """Split conformal risk control threshold.

    tau is the k-th smallest score among the m observed failures, with
    k = floor(eps * (m + 1)) - 1. The (m + 1) is the finite-sample correction: the next
    failure we would meet is a fresh draw, and counting it in the rank makes the bound
    exact at any m rather than only in the limit.

    Deployment skips scores strictly below tau.  This conservative convention
    makes the rule valid when scores are tied: every scenario in the threshold's
    tie block is executed.

    Returns SKIP_NOTHING when k < 0, which happens while m < 1 / eps. Refusing to act
    is correct there: with two or three observed failures there is no evidence about
    the region we would be skipping.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between 0 and 1")
    f = _failure_scores(scores, outcomes)
    m = len(f)
    if m == 0:
        return SKIP_NOTHING
    k = int(np.floor(eps * (m + 1))) - 1
    return float(f[k]) if k >= 0 else SKIP_NOTHING


def cp_threshold(scores: np.ndarray, outcomes: np.ndarray, eps: float,
                 delta: float = 0.10) -> float:
    """Standard iid order-statistic tolerance reference (legacy API name).

    One-based rank j corresponds to Beta(j, m-j+1). A finite-library
    application uses checked_tolerance_index to check the exact tail.
    This function alone states the iid tolerance guarantee.
    """
    f = _failure_scores(scores, outcomes)
    j = binomial_tolerance_index(len(f), eps, delta)
    return float(f[j - 1]) if j else SKIP_NOTHING


@lru_cache(maxsize=32768)
def binomial_tolerance_index(m: int, eps: float, delta: float = .10) -> int:
    """Largest j with BinomCDF(j-1; m, eps) <= delta."""
    if not 0 < eps < 1 or not 0 < delta < 1 or m < 0:
        raise ValueError("invalid tolerance parameters")
    lo, hi = 0, int(m)
    while lo < hi:
        j = (lo + hi + 1) // 2
        if _binom.cdf(j - 1, m, eps) <= delta:
            lo = j
        else:
            hi = j - 1
    return lo


def _validate_population_design(population_size: int,
                                calibration_size: int) -> tuple[int, int]:
    population_size = int(population_size)
    calibration_size = int(calibration_size)
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 0 < calibration_size < population_size:
        raise ValueError(
            "calibration_size must be between 1 and population_size - 1")
    return population_size, calibration_size


def _finite_population_tail(j: int, m: int, population_size: int,
                            calibration_size: int, eps: float) -> tuple[float, int]:
    """Worst conditional violation probability for a candidate failure rank.

    ``j`` is one based.  Conditional on observing ``m`` calibration failures and
    on there being ``M`` failures in the library, the sampled failure ranks are a
    uniform size-``m`` subset of ``1,...,M``.  If ``R_j`` is the j-th sampled rank,
    the strict cut hides at most ``R_j-j`` failures.  A library-level violation is

        R_j - j > eps * M.

    With ``q = j + floor(eps*M)``, this is ``R_j > q``.  Equivalently, at most
    ``j-1`` of the ``m`` sampled ranks occur among the first ``q`` population
    ranks, which is the hypergeometric lower tail evaluated below.
    """
    max_failures = m + population_size - calibration_size
    # The lower tail increases with M while floor(eps*M), hence q, is fixed.
    # Its maximum is at a block's right endpoint; integer arithmetic handles
    # discontinuities in a user-supplied decimal epsilon exactly.
    fraction = Fraction(str(eps))
    num, den = fraction.numerator, fraction.denominator
    blocks = np.arange((num*m)//den, (num*max_failures)//den + 1,
                       dtype=np.int64)
    if (int(blocks[-1])+1)*den < np.iinfo(np.int64).max and num*max_failures < np.iinfo(np.int64).max:
        totals = np.minimum(((blocks + 1)*den - 1)//num, max_failures)
    else:
        totals = np.asarray([min(((int(k)+1)*den-1)//num,max_failures) for k in blocks],dtype=np.int64)
    totals = np.unique(totals[(totals >= m) & (totals <= max_failures)])
    q = j + (num*totals)//den if num*max_failures < np.iinfo(np.int64).max else np.asarray([j+num*int(v)//den for v in totals])
    active = q < totals
    tails = np.zeros(totals.shape, dtype=float)
    if np.any(active):
        tails[active] = _hypergeom.cdf(
            j - 1, totals[active], q[active], m)
    worst_at = int(np.argmax(tails))
    return float(tails[worst_at]), int(totals[worst_at])


@lru_cache(maxsize=8192)
def finite_population_index(m: int, population_size: int,
                            calibration_size: int, eps: float,
                            delta: float = 0.10) -> int:
    """Largest certified one-based failure rank for a finite library.

    The guarantee is uniform over every total number of library failures compatible
    with the observed ``m`` failures.  Returning zero means that no nonempty strict
    cut can be certified.  The predicate is monotone because ``R_j-j`` is
    nondecreasing in ``j`` for every realized set of sampled ranks, so binary search
    is exact rather than heuristic.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between 0 and 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1")
    population_size, calibration_size = _validate_population_design(
        population_size, calibration_size)
    m = int(m)
    if not 0 <= m <= calibration_size:
        raise ValueError("m must be between 0 and calibration_size")
    if m == 0:
        return 0

    lo, hi = 0, m
    while lo < hi:
        mid = (lo + hi + 1) // 2
        tail, _ = _finite_population_tail(
            mid, m, population_size, calibration_size, eps)
        if tail <= delta + 1e-15:
            lo = mid
        else:
            hi = mid - 1
    return lo


@lru_cache(maxsize=32768)
def checked_tolerance_index(m: int, population_size: int,
                            calibration_size: int, eps: float,
                            delta: float = .10) -> int:
    """Binomial tolerance candidate with a finite-library tail check.

    The check uses N, n, m, epsilon and delta, never the target truth M.
    A failed check caps the candidate at the exact finite-population rank.
    """
    population_size, calibration_size = _validate_population_design(
        population_size, calibration_size)
    if not 0 <= m <= calibration_size:
        raise ValueError("m must be between zero and calibration size")
    j = binomial_tolerance_index(m, eps, delta)
    if j == 0:
        return 0
    if _finite_population_tail(j, m, population_size, calibration_size, eps)[0] <= delta:
        return j
    return min(j, finite_population_index(m, population_size, calibration_size, eps, delta))


def finite_population_threshold(scores: np.ndarray, outcomes: np.ndarray,
                                eps: float, population_size: int,
                                calibration_size: int,
                                delta: float = 0.10) -> float:
    """Exact finite-population, high-probability threshold.

    For a precommitted simple random sample of ``calibration_size`` from a library
    of ``population_size`` scenarios, the returned strict cut satisfies

        P(hidden failures / all library failures > eps) <= delta.

    No iid or asymptotic approximation is used.  The calculation conditions on the
    observed failure count and maximizes the exact hypergeometric violation
    probability over every compatible, unknown total failure count.
    """
    f = _failure_scores(scores, outcomes)
    if len(np.asarray(scores)) != int(calibration_size):
        raise ValueError("scores must contain exactly calibration_size observations")
    j = finite_population_index(
        len(f), population_size, calibration_size, float(eps), float(delta))
    return float(f[j - 1]) if j > 0 else SKIP_NOTHING


def get_rule(name: str):
    """Return a callable (scores, outcomes, eps) -> tau."""
    name = name.lower()
    if name == "crc":
        return crc_threshold
    if name in ("cp", "clopper", "clopper-pearson"):
        return cp_threshold
    if name in ("fpc", "finite", "finite-population"):
        return finite_population_threshold
    raise ValueError("unknown threshold rule %r; use 'crc', 'cp', or 'fpc'" % name)


def min_failures_for(eps: float) -> int:
    """Smallest number of observed failures at which the CRC rule can act.

    The conformal index k = floor(eps * (m + 1)) - 1 is non-negative once
    eps * (m + 1) >= 1, which is m >= 1/eps - 1. For eps = 0.05 that is 19 failures,
    not 20. Below this the rule skips nothing.

    The adaptive stopping rule uses this as a feasibility gate: the marginal cannot be
    read before a threshold exists.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between 0 and 1")
    return max(1, int(np.ceil(1.0 / eps - 1.0)))
