# -*- coding: utf-8 -*-
"""The method end to end.

    cs = CertifiedSkip(eps=0.05)
    cs.fit(X_prior, y_prior)              # risk model from an earlier campaign
    result = cs.run_fixed(X_pool, execute, n_calibration=1000)

`execute(indices) -> array of 0/1` is the hook to the simulator. In a real pipeline it
launches runs. In the offline experiments here it is a lookup into recorded outcomes,
which lets us score the method against ground truth.

The fixed-budget path is the certified method: its calibration size is committed before
any current-library outcomes are observed.  A separate ``run_adaptive`` path implements
the paper's marginal-utility heuristic.  Because that stopping time reads observed
failure scores, it deliberately does not claim the fixed-budget certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .threshold import (get_rule, min_failures_for, skip_mask, SKIP_NOTHING,
                        checked_tolerance_index)

__all__ = ["CertifiedSkip", "SkipResult"]


@dataclass
class SkipResult:
    """What one run of the method produced."""
    threshold: float
    calibrated: np.ndarray          # indices actually executed for calibration
    skipped: np.ndarray             # indices the certificate says need not run
    executed: np.ndarray            # calibration plus everything above the threshold
    eps: float
    rule: str
    n_calibration: int
    n_failures_seen: int
    mode: str = "fixed"
    delta: float | None = None
    history: list = field(default_factory=list)   # (n, skippable) after each batch
    execution_complete: bool = False

    @property
    def skip_rate(self) -> float:
        total = len(self.skipped) + len(self.executed)
        return len(self.skipped) / total if total else 0.0

    @property
    def runs_saved(self) -> int:
        """Runs saved relative to executing the full library.

        Calibration scenarios belong to the library and are therefore not an
        additional charge against this baseline.  The method executes exactly
        ``N - len(skipped)`` scenarios.
        """
        return int(len(self.skipped))

    @property
    def total_executed(self) -> int:
        return int(len(self.executed))

    @property
    def penalized_utility(self) -> int:
        """Conservative utility used only by the adaptive budget heuristic.

        This quantity charges every calibration run as certification overhead:
        ``skipped - calibrated``.  It is not the number of runs saved relative
        to full execution.
        """
        return int(len(self.skipped)) - int(self.n_calibration)

    @property
    def net_saving(self) -> int:
        """Backward-compatible alias for actual runs saved."""
        return self.runs_saved

    def certificate(self) -> str:
        total = len(self.skipped) + len(self.executed)
        prefix = "skipped %d of %d scenarios (%.1f%%)" % (
            len(self.skipped), total, 100 * self.skip_rate)
        if self.mode != "fixed":
            return (prefix + "; adaptive stopping used--no fixed-budget finite-sample "
                    "certificate is claimed")
        if not self.execution_complete:
            return (prefix + "; statement covers the declared skip set only; "
                    "retained execution is incomplete")
        if self.rule.lower() == "crc":
            return (prefix + "; over random fixed-budget calibration draws, the "
                    "expected missed-failure share is at most %.0f%%" %
                    (100 * self.eps))
        confidence = 100 * (1.0 - (self.delta if self.delta is not None else 0.10))
        method = ("exact finite-population" if self.rule.lower() in
                  ("fpc", "finite", "finite-population") else
                  "binomial tolerance with finite-library check")
        return (prefix + "; with %.0f%% confidence under fixed-budget calibration "
                "(%s), "
                "the unexecuted remainder contains at most %.0f%% of all "
                "library failures" %
                (confidence, method, 100 * self.eps))


class CertifiedSkip:
    """Certified pre-execution triage of a scenario library."""

    def __init__(self, eps: float = 0.05, rule: str = "crc", delta: float = 0.10,
                 batch_frac: float = 0.01, min_batch: int = 25,
                 window: int = 3, patience: int = 2, max_frac: float = 0.6,
                 model=None, random_state: int = 0):
        if not 0.0 < eps < 1.0:
            raise ValueError("eps must lie strictly between 0 and 1")
        self.eps = eps
        self.rule_name = rule
        self.delta = delta
        self.batch_frac = batch_frac
        self.min_batch = min_batch
        self.window = window
        self.patience = patience
        self.max_frac = max_frac
        self.random_state = random_state
        self.model = model
        self._fitted = False

    # ---------------------------------------------------------------- model
    def fit(self, X, y) -> "CertifiedSkip":
        """Fit the risk model on an earlier campaign."""
        if self.model is None:
            self.model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, random_state=self.random_state)
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        self._fitted = True
        return self

    def score(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before scoring")
        return self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def _threshold(self, scores, outcomes, population_size=None,
                   calibration_size=None) -> float:
        rule = get_rule(self.rule_name)
        if self.rule_name.lower() in ("fpc", "finite", "finite-population"):
            if population_size is None or calibration_size is None:
                raise ValueError("the finite-population rule needs N and n")
            return rule(scores, outcomes, self.eps, population_size,
                        calibration_size, self.delta)
        if self.rule_name.lower().startswith("c") and self.rule_name.lower() != "crc":
            f = np.sort(np.asarray(scores)[np.asarray(outcomes) == 1])
            j = checked_tolerance_index(len(f), population_size,
                                        calibration_size, self.eps, self.delta)
            return float(f[j - 1]) if j else SKIP_NOTHING
        return rule(scores, outcomes, self.eps)

    # ---------------------------------------------------------------- stopping
    def _should_stop(self, history, n_failures) -> bool:
        """Marginal rule with the two guards that make it behave at the start.

        The marginal cannot be read before a threshold exists. While the number of
        observed failures is below 1/eps the conformal index is negative, the
        skippable count is zero, and a naive reading of the flat start would stop
        immediately. We therefore refuse to stop until a threshold is feasible and
        something is actually skippable.
        """
        if n_failures < min_failures_for(self.eps):
            return False
        if len(history) <= self.window or history[-1][1] <= 0:
            return False
        gained = history[-1][1] - history[-1 - self.window][1]
        spent = history[-1][0] - history[-1 - self.window][0]
        return gained / max(spent, 1) < 1.0

    # ---------------------------------------------------------------- main loop
    def run_adaptive(self, X, execute, order=None, scores=None) -> SkipResult:
        """Select a calibration budget with the empirical marginal heuristic.

        This path is intentionally not certified by the fixed-budget theorem:
        the stopping time depends on observed failure scores.  Use ``run_fixed``
        whenever a finite-sample certificate is required.

        `execute(indices)` must return an array of 0/1 outcomes for those indices.
        `order` fixes the random visiting order; supply it to reproduce a draw.
        `scores` lets a caller reuse a scoring pass across many repetitions. Scoring
        is deterministic given the model, and repeating it dominates the runtime of
        any study that repeats draws.
        """
        X = np.asarray(X, dtype=float)
        n_total = len(X)
        scores = self.score(X) if scores is None else np.asarray(scores, dtype=float)
        scores_sorted = np.sort(scores)

        rng = np.random.default_rng(self.random_state)
        if order is None:
            order = rng.permutation(n_total)
        order = np.asarray(order)

        batch = max(self.min_batch, int(n_total * self.batch_frac))
        limit = min(int(n_total * self.max_frac), n_total - 1)

        n, stalls = 0, 0
        history, outcomes = [], np.empty(0, dtype=int)
        tau = SKIP_NOTHING

        while n + batch <= limit:
            fresh = order[n:n + batch]
            outcomes = np.concatenate([outcomes, np.asarray(execute(fresh), dtype=int)])
            n += batch
            calibrated = order[:n]

            tau = self._threshold(scores[calibrated], outcomes,
                                  n_total, n)
            below_total = int(np.searchsorted(scores_sorted, tau, side="left"))
            below_calib = int(skip_mask(scores[calibrated], tau).sum())
            skippable = below_total - below_calib
            history.append((n, skippable))

            n_failures = int(outcomes.sum())
            if self._should_stop(history, n_failures):
                stalls += 1
                if stalls >= self.patience:
                    break
            else:
                stalls = 0

        calibrated = order[:n]
        rest = order[n:]
        is_skipped = skip_mask(scores[rest], tau)
        return SkipResult(
            threshold=float(tau),
            calibrated=calibrated,
            skipped=rest[is_skipped],
            executed=np.concatenate([calibrated, rest[~is_skipped]]),
            eps=self.eps,
            rule=self.rule_name,
            n_calibration=int(n),
            n_failures_seen=int(outcomes.sum()),
            mode="adaptive",
            delta=self.delta,
            history=history)

    # ---------------------------------------------------------------- fixed budget
    def run_fixed(self, X, execute, n_calibration: int, order=None,
                  scores=None) -> SkipResult:
        """Certified triage with a precommitted calibration budget.

        ``n_calibration`` must be chosen before outcomes from this library are
        observed.  It may depend on library size, historical campaigns, or an
        external execution budget, but not on current calibration labels.
        """
        X = np.asarray(X, dtype=float)
        n_total = len(X)
        if not 0 < int(n_calibration) < n_total:
            raise ValueError("n_calibration must be between 1 and len(X)-1")
        n_calibration = int(n_calibration)
        scores = self.score(X) if scores is None else np.asarray(scores, dtype=float)
        rng = np.random.default_rng(self.random_state)
        if order is None:
            order = rng.permutation(n_total)
        order = np.asarray(order)

        calibrated = order[:n_calibration]
        if scores.shape != (n_total,) or not np.isfinite(scores).all():
            raise ValueError("scores must be a finite vector with one value per scenario")
        if order.shape != (n_total,) or not np.array_equal(np.sort(order), np.arange(n_total)):
            raise ValueError("order must be a permutation of all scenario indices")
        if not np.issubdtype(order.dtype, np.integer):
            raise ValueError("order must contain integer indices")
        outcomes = np.asarray(execute(calibrated))
        if outcomes.shape != calibrated.shape or not np.isin(outcomes, [0, 1]).all():
            raise ValueError("executor must return a binary outcome for every calibration scenario")
        outcomes = outcomes.astype(int)
        tau = self._threshold(scores[calibrated], outcomes,
                              n_total, n_calibration)

        rest = order[n_calibration:]
        is_skipped = skip_mask(scores[rest], tau)
        retained = rest[~is_skipped]
        completed = np.asarray(execute(retained))
        if completed.shape != retained.shape or not np.isin(completed, [0, 1]).all():
            raise ValueError("executor must return a binary outcome for every retained scenario")
        return SkipResult(
            threshold=float(tau), calibrated=calibrated,
            skipped=rest[is_skipped],
            executed=np.concatenate([calibrated, rest[~is_skipped]]),
            eps=self.eps, rule=self.rule_name,
            n_calibration=int(n_calibration),
            n_failures_seen=int(outcomes.sum()), mode="fixed", delta=self.delta,
            history=[], execution_complete=True)


def lookup_executor(truth):
    """Offline stand-in for the simulator: read recorded outcomes.

    In production this is replaced by a function that launches the runs.
    """
    truth = np.asarray(truth, dtype=int)

    def execute(indices):
        return truth[np.asarray(indices)]

    return execute
