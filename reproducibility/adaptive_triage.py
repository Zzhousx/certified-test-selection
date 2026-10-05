from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Checkpoint triage with an executor that reveals only requested outcomes."""
import numpy as np
from certskip.threshold import finite_population_index

def validate_outcomes(values, size):
    raw = np.asarray(values)
    if raw.shape != (size,) or not np.isin(raw, [0, 1]).all():
        raise ValueError('executor must return one binary outcome per requested index')
    return raw.astype(np.int8)

def checkpoint_decision(scores, calibration, outcomes, epsilon, delta):
    """This function cannot access outcomes outside calibration."""
    scores = np.asarray(scores, float)
    c = np.asarray(calibration, int)
    y = validate_outcomes(outcomes, len(c))
    f = np.sort(scores[c][y == 1])
    j = finite_population_index(len(f), len(scores), len(c), epsilon, delta)
    tau = float(f[j - 1]) if j else -np.inf
    skip = scores < tau
    skip[c] = False
    return dict(threshold=tau, skipped_indices=np.flatnonzero(skip), rank=j, m=len(f))

def adaptive_triage(scores, execute, checkpoints, epsilon=.05, delta=.1,
                    stop_skip_fraction=.1, rng=None, permutation=None,
                    adjusted=True):
    """Run calibration, select a checkpoint, then execute all retained tests.

    adjusted=False is an explicitly uncertified experimental ablation.
    A caller must freeze scores and design before observing current outcomes.
    """
    s = np.asarray(scores, float)
    if s.ndim != 1 or len(s) < 2 or not np.isfinite(s).all():
        raise ValueError('scores must be a finite vector of length at least two')
    ns = np.asarray(checkpoints)
    if (ns.ndim != 1 or not len(ns) or not np.equal(ns, np.floor(ns)).all()
        or ns[0] < 1 or ns[-1] >= len(s) or np.any(np.diff(ns) <= 0)):
        raise ValueError('checkpoints must be increasing integer sizes inside the library')
    if not 0 < epsilon < 1 or not 0 < delta < 1 or not 0 <= stop_skip_fraction <= 1:
        raise ValueError('invalid risk or stopping setting')
    if permutation is None:
        permutation = (rng or np.random.default_rng()).permutation(len(s))
    p = np.asarray(permutation)
    if p.shape != s.shape or not np.array_equal(np.sort(p), np.arange(len(s))):
        raise ValueError('permutation must contain every library index exactly once')
    p = p.astype(int)
    alpha = delta / len(ns) if adjusted else delta
    observed = []
    history = []
    previous = 0
    for k, n in enumerate(ns.astype(int)):
        added = p[previous:n]
        observed.extend(validate_outcomes(execute(added), len(added)).tolist())
        decision = checkpoint_decision(s, p[:n], observed, epsilon, alpha)
        hit = decision['rank'] > 0 and len(decision['skipped_indices']) / len(s) >= stop_skip_fraction
        history.append(dict(checkpoint=k + 1, n=int(n), m=decision['m'],
                            rank=decision['rank'], skipped=len(decision['skipped_indices']),
                            threshold=decision['threshold'], target_met=bool(hit), delta=alpha))
        previous = n
        if hit or k == len(ns) - 1:
            break
    retain = np.ones(len(s), bool)
    retain[p[:n]] = False
    retain[decision['skipped_indices']] = False
    remaining = np.flatnonzero(retain)
    validate_outcomes(execute(remaining), len(remaining))
    return dict(**decision, n=int(n), checkpoint=k+1, target_met=bool(hit),
                history=history, execution_complete=True,
                certified=bool(adjusted), executed=len(s)-len(decision['skipped_indices']))
