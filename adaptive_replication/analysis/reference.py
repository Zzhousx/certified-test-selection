"""Finite-population and binomial certification reference.

Scores increase with failure risk. Skip only unexecuted tests strictly below
the threshold. Loss is omitted failures divided by all library failures.
Calibration executions count toward total cost.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import ceil

import numpy as np
from scipy.stats import binom, hypergeom


# ---------------------------------------------------------------------------
# Tolerance handling: epsilon is used through floor(eps * M'), so keep it exact.
# ---------------------------------------------------------------------------
def _eps_fraction(eps) -> Fraction:
    f = Fraction(str(eps)) if not isinstance(eps, Fraction) else eps
    if not (0 < f < 1):
        raise ValueError("eps must be in (0, 1)")
    return f


# ---------------------------------------------------------------------------
# Paper's certificate (FPC), Eqs. (3)-(5) with the block-endpoint evaluation of III-D
# ---------------------------------------------------------------------------
def _b_j(j: int, m: int, M_lo: int, M_hi: int, eps: Fraction) -> float:
    """max_{M' in [M_lo, M_hi]} P(X <= j-1), X ~ Hypergeom(M', q = j + floor(eps*M'), m).

    Within a block of M' sharing floor(eps*M'), the tail is nondecreasing in M',
    so only block right endpoints (clipped to M_hi) need evaluation.
    """
    en, ed = eps.numerator, eps.denominator
    k0, k1 = (M_lo * en) // ed, (M_hi * en) // ed
    ks = np.arange(k0, k1 + 1, dtype=np.int64)
    right = -(-(ks + 1) * ed // en) - 1          # largest M' with floor(eps*M') == k
    Mp = np.minimum(right, M_hi)
    q = j + ks
    ok = (Mp >= M_lo) & (q < Mp)                  # B = 0 when q >= M'
    if not ok.any():
        return 0.0
    return float(hypergeom.cdf(j - 1, Mp[ok], q[ok], m).max())


@lru_cache(maxsize=None)
def _jstar_range(m: int, M_lo: int, M_hi: int, eps: Fraction, delta: float) -> int:
    if m <= 0:
        return 0
    lo, hi = 0, m                                  # b_j is nondecreasing in j -> binary search
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _b_j(mid, m, M_lo, M_hi, eps) <= delta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def jstar_fpc(m: int, N: int, n: int, eps, delta: float) -> int:
    """Paper's j* (Eq. 5): compatible totals M' in {m, ..., m + N - n}."""
    return _jstar_range(int(m), int(m), int(m + N - n), _eps_fraction(eps), float(delta))


def jstar_binomial(m: int, eps, delta: float) -> int:
    """Classical order-statistic rule (Neyman-Pearson umbrella; QBCB of Lewis et al. 2021
    with recall target t = 1 - eps): largest j with P(Bin(m, eps) <= j-1) <= delta."""
    e = float(_eps_fraction(eps))
    if m <= 0:
        return 0
    js = np.arange(1, m + 1)
    ok = binom.cdf(js - 1, m, e) <= delta          # nondecreasing in j
    return int(js[ok].max()) if ok.any() else 0


def m_min_binomial(eps, delta: float) -> int:
    """Smallest m with (1-eps)^m <= delta, i.e. the binomial rule accepts j = 1."""
    e = float(_eps_fraction(eps))
    m = 1
    while (1 - e) ** m > delta:
        m += 1
    return m


def m_min_fpc(N: int, n: int, eps, delta: float, m_max: int | None = None) -> int | None:
    """Smallest m for which the paper's certificate accepts j = 1 (None if never <= m_max)."""
    m_max = m_max or n
    for m in range(1, m_max + 1):
        if jstar_fpc(m, N, n, eps, delta) >= 1:
            return m
    return None


# ---------------------------------------------------------------------------
# Planning quantities (use true M for validation; a prior estimate of M for planning)
# ---------------------------------------------------------------------------
def threshold_available(m: int, N: int, n: int, eps, delta: float, rule: str = "fpc") -> bool:
    """j* >= 1  <=>  b_1 <= delta (b_j nondecreasing in j), so one tail evaluation suffices."""
    if m <= 0:
        return False
    if rule == "fpc":
        return _b_j(1, int(m), int(m), int(m + N - n), _eps_fraction(eps)) <= delta
    return (1 - float(_eps_fraction(eps))) ** m <= delta


def predicted_refusal(N: int, M: int, n: int, eps, delta: float, rule: str = "fpc",
                      tail: float = 1e-10) -> float:
    """P(j* = 0) when the calibration sample is uniform of size n (m values with total
    probability < 2*tail are skipped; their mass is counted as refusal on the low side and
    as acceptance on the high side, so the error is below 2*tail)."""
    rv = hypergeom(N, M, n)
    lo, hi = int(rv.ppf(tail)), int(rv.ppf(1 - tail))
    total = float(rv.cdf(lo - 1)) if lo > 0 else 0.0
    for m in range(max(lo, 0), hi + 1):
        if not threshold_available(m, N, n, eps, delta, rule):
            total += rv.pmf(m)
    return float(total)


def prob_rules_differ(N: int, M: int, n: int, eps, delta: float, tail: float = 1e-6) -> float:
    """P(j*_fpc != j*_binomial) under uniform calibration of size n."""
    rv = hypergeom(N, M, n)
    lo, hi = int(rv.ppf(tail)), int(rv.ppf(1 - tail))
    return float(sum(rv.pmf(m) for m in range(max(lo, 0), hi + 1)
                     if jstar_fpc(m, N, n, eps, delta) != jstar_binomial(m, eps, delta)))


def min_calibration_fraction(N: int, M_est: float, eps, delta: float,
                             target_prob: float = 0.5, rule: str = "fpc") -> float | None:
    """Smallest calibration fraction c (grid of 0.5%) with P(threshold available) >= target_prob,
    assuming M_est failures (true M for validation, prior estimate for planning)."""
    M = int(round(M_est))
    for c in np.arange(0.005, 1.0, 0.005):
        n = int(round(c * N))
        if 0 < n < N and 1 - predicted_refusal(N, M, n, eps, delta, rule) >= target_prob:
            return float(round(c, 3))
    return None


# ---------------------------------------------------------------------------
# Decisions on one uniform permutation `perm` of range(N)
# ---------------------------------------------------------------------------
def _loss(labels: np.ndarray, skip: np.ndarray) -> float:
    M = int(labels.sum())
    return 0.0 if M == 0 else float(labels[skip].sum()) / M


def _threshold_skip(scores, labels, calib_idx, j):
    N = len(scores)
    in_c = np.zeros(N, bool)
    in_c[calib_idx] = True
    if j == 0:
        return np.zeros(N, bool), -np.inf
    f = np.sort(scores[calib_idx][labels[calib_idx] == 1])
    tau = f[j - 1]
    return (~in_c) & (scores < tau), tau


def fixed_decision(scores, labels, perm, n, eps, delta, rule="fpc"):
    """Algorithm 1 with a fixed calibration size n. Returns dict with skip mask and metrics."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    N = len(scores)
    C = np.asarray(perm[:n])
    m = int(labels[C].sum())
    j = jstar_fpc(m, N, n, eps, delta) if rule == "fpc" else jstar_binomial(m, eps, delta)
    skip, tau = _threshold_skip(scores, labels, C, j)
    return dict(m=m, j=j, tau=tau, n_cal=n, skip=skip, saving=skip.sum() / N,
                loss=_loss(labels, skip))


def inverse_sampling_decision(scores, labels, perm, K, eps, delta, rule="fpc", n_cap=None):
    """Target-K calibration (generalises Cormack & Grossman's target method):
    execute the permutation until K failures are observed, then apply the certificate
    with m = K and n = n_stop. If K failures are not found within n_cap executions
    (default: whole library), stop calibration and skip nothing.

    Validity: n_stop depends only on the pass/fail pattern of the permutation; given that
    pattern, the identities of the K sampled failures are a uniform K-subset of all M
    failures, and M <= K + N - n_stop. The proof of Theorem 1 (which conditions on m and
    uses only these two facts) therefore applies with m = K, n = n_stop. No delta split.
    Stopping at the cap without a threshold skips nothing, so it cannot cause an exceedance.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    N = len(scores)
    cap = N if n_cap is None else int(n_cap)
    cum = np.cumsum(labels[np.asarray(perm)])
    hit = np.nonzero(cum >= K)[0]
    if len(hit) == 0 or int(hit[0]) + 1 > cap:      # K failures not reached within the cap
        n_used = min(cap, N)
        return dict(m=int(cum[n_used - 1]), j=0, tau=-np.inf, n_cal=n_used,
                    skip=np.zeros(N, bool), saving=0.0, loss=0.0)
    n_stop = int(hit[0]) + 1
    C = np.asarray(perm[:n_stop])
    j = jstar_fpc(K, N, n_stop, eps, delta) if rule == "fpc" else jstar_binomial(K, eps, delta)
    skip, tau = _threshold_skip(scores, labels, C, j)
    return dict(m=K, j=j, tau=tau, n_cal=n_stop, skip=skip, saving=skip.sum() / N,
                loss=_loss(labels, skip))


def ucb_miss_share(m: int, x: int, N: int, n: int, L_u: int, delta: float) -> float:
    """(1-delta) upper confidence bound on H/M for a skip set fixed BEFORE calibration.

    m   : failures in the calibration sample
    x   : calibration failures scoring below the (pre-fixed) threshold
    L_u : unsampled scenarios below the threshold (= |S|)
    For every compatible (M', K') -- K' failures below the threshold among M' failures --
    keep it unless P(Hypergeom(M', K', m) <= x) <= delta; return max (K' - x) / M'.
    Deterministic constraints: 0 <= K'-x <= L_u, 0 <= M'-K'-(m-x) <= N-n-L_u.
    """
    if L_u == 0:
        return 0.0
    if m == 0:
        return 1.0                                   # no failure evidence: bound is vacuous
    Mp = np.arange(max(m, 1), m + (N - n) + 1, dtype=np.int64)
    if L_u is None:                                  # relaxed: drop the two L_u constraints
        lo = np.full_like(Mp, x)
        hi = Mp - (m - x)
    else:
        lo = np.maximum(x, Mp - (m - x) - (N - n - L_u))
        hi = np.minimum(x + L_u, Mp - (m - x))
    feasible = lo <= hi
    Mp, lo, hi = Mp[feasible], lo[feasible], hi[feasible]
    # keep only M' whose smallest K' is not rejected
    keep = hypergeom.cdf(x, Mp, lo, m) > delta
    Mp, lo, hi = Mp[keep], lo[keep], hi[keep]
    if len(Mp) == 0:
        return 0.0
    # binary search per M' for largest K' with cdf > delta (cdf is nonincreasing in K')
    a, b = lo.copy(), hi.copy()
    while True:
        active = a < b
        if not active.any():
            break
        mid = (a + b + 1) // 2
        good = hypergeom.cdf(x, Mp, mid, m) > delta
        a = np.where(active & good, mid, a)
        b = np.where(active & ~good, mid - 1, b)
    return float(((a - x) / Mp).max())


@lru_cache(maxsize=4096)
def _grid_blocks(m: int, x: int, N: int, n: int, delta: float, ratio: float):
    """For a geometric grid m = G_0 < G_1 < ... < G_T = m + N - n, return arrays
    (Kx_b, G_a): Kx_b[i] = (largest K' accepted at M' = G_{i+1}) - x, G_a[i] = G_i,
    plus the exact value at M' = G_0. Acceptance of a fixed K' only grows with M'
    (more unmarked items), so K'(M') <= K'(G_{i+1}) for all M' in [G_i, G_{i+1}]."""
    M_hi = m + (N - n)
    g = [m]
    while g[-1] < M_hi:
        g.append(min(M_hi, max(g[-1] + 1, int(g[-1] * ratio))))
    G = np.asarray(g, dtype=np.int64)
    lo = np.full_like(G, x)
    hi = G - (m - x)
    ok = hypergeom.cdf(x, G, lo, m) > delta      # K' = x is accepted everywhere it is feasible
    a, b = lo.copy(), hi.copy()
    while True:
        active = ok & (a < b)
        if not active.any():
            break
        mid = (a + b + 1) // 2
        good = hypergeom.cdf(x, G, mid, m) > delta
        a = np.where(active & good, mid, a)
        b = np.where(active & ~good, mid - 1, b)
    Kx = np.where(ok, a - x, 0)
    return Kx[1:], G[:-1], int(Kx[0]), int(G[0])


def ucb_miss_share_grid(m: int, x: int, N: int, n: int, L_u: int, delta: float,
                        ratio: float = 1.005) -> float:
    """Fast, valid (conservative) version of ucb_miss_share.
    Block bound: sup_{M' in [G_i, G_{i+1}]} (K'(M') - x)/M' <= min(K'(G_{i+1}) - x, L_u) / G_i.
    Dropping the lower feasibility constraint only enlarges the compatible set.
    In checks at the paper's library sizes (ratio 1.005) the grid bound was never below the
    exact bound and exceeded it by at most ~1 percentage point, at ~20x lower cost."""
    if L_u == 0:
        return 0.0
    if m == 0:
        return 1.0
    Kx_b, G_a, Kx0, G0 = _grid_blocks(int(m), int(x), int(N), int(n), float(delta), float(ratio))
    vals = np.minimum(Kx_b, L_u) / G_a
    v0 = min(Kx0, L_u) / G0
    return float(max(v0, vals.max() if len(vals) else 0.0))


def budget_threshold(scores, B: int, n: int) -> float:
    """Score threshold fixed from scores only: at most B - n scenarios have s >= tau_B,
    so total executions |{s >= tau}| + |C ∩ {s < tau}| <= B for any calibration draw."""
    s = np.sort(np.asarray(scores, float))[::-1]
    cap = B - n
    if cap <= 0:
        return np.inf                                # nothing outside calibration can run
    if cap >= len(s):
        return -np.inf
    # smallest tau with |{s >= tau}| <= cap: strictly above the (cap+1)-th largest score
    t = s[cap]                                       # (cap+1)-th largest
    above = s[s > t]
    return float(above.min()) if len(above) else np.inf


def budget_decision(scores, labels, perm, n, B, delta, exact=False):
    """Fixed-budget design: threshold from scores and budget only, uniform calibration of
    size n, then execute calibration + all scenarios with s >= tau_B. Reports a certified
    (1-delta) upper bound on the missed-failure share alongside the realised loss.
    exact=False uses the geometric-grid bound (valid, much faster, slightly looser)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    N = len(scores)
    tau = budget_threshold(scores, B, n)
    C = np.asarray(perm[:n])
    in_c = np.zeros(N, bool)
    in_c[C] = True
    skip = (~in_c) & (scores < tau)
    m = int(labels[C].sum())
    x = int(labels[C][scores[C] < tau].sum())
    L_u = int(skip.sum())
    if L_u == 0:
        eps_hat = 0.0
    elif exact:
        eps_hat = ucb_miss_share(m, x, N, n, L_u, delta)
    else:
        eps_hat = ucb_miss_share_grid(m, x, N, n, L_u, float(delta))
    executed = N - int(skip.sum())
    return dict(m=m, x=x, tau=tau, n_cal=n, executed=executed, skip=skip,
                saving=skip.sum() / N, loss=_loss(labels, skip), eps_hat=eps_hat)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _synthetic_library(N, M, auc_shift, rng, ties=False):
    labels = np.zeros(N, int)
    labels[rng.choice(N, M, replace=False)] = 1
    scores = rng.normal(0, 1, N) + auc_shift * labels
    if ties:
        scores = np.round(scores, 1)
    return scores, labels


def self_test(mc_reps: int = 4000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    # 1) monotonicity of b_j and binary search vs brute force
    N, n = 400, 120
    eps = Fraction(1, 20)
    for m in range(1, 70):
        bs = [_b_j(j, m, m, m + N - n, eps) for j in range(1, m + 1)]
        assert all(bs[i] <= bs[i + 1] + 1e-15 for i in range(len(bs) - 1))
        brute = max([0] + [j for j in range(1, m + 1) if bs[j - 1] <= 0.1])
        assert brute == jstar_fpc(m, N, n, 0.05, 0.1)
    print("[ok] b_j monotone; binary search matches brute force")

    # 2) closed-form minimum failure counts (reference setting)
    assert m_min_binomial(0.05, 0.10) == 45
    assert m_min_binomial(0.05, 0.01) == 90
    assert m_min_binomial(0.01, 0.10) == 230
    assert m_min_fpc(10000, 2000, 0.05, 0.10) == 45
    print("[ok] m_min(0.05,0.10)=45, (0.05,0.01)=90, (0.01,0.10)=230; FPC(N=1e4,n=2e3)=45")

    # 3) Monte Carlo validity: exceedance frequency <= delta (+ MC slack)
    delta, eps = 0.10, 0.05
    slack = 3 * np.sqrt(delta * (1 - delta) / mc_reps)
    # cases chosen so the certificates are near-binding (informative), incl. ties and a
    # useless scorer (shift 0), where violations would show up first
    cases = [(800, 400, 0.0, False), (600, 200, 1.0, True), (1000, 150, 1.5, False)]
    for (N, M, shift, ties) in cases:
        scores, labels = _synthetic_library(N, M, shift, rng, ties)
        exc = {"fixed_fpc": 0, "fixed_bin": 0, "invK": 0, "budget": 0}
        for _ in range(mc_reps):
            perm = rng.permutation(N)
            d1 = fixed_decision(scores, labels, perm, int(0.3 * N), eps, delta, "fpc")
            d2 = fixed_decision(scores, labels, perm, int(0.3 * N), eps, delta, "binomial")
            d3 = inverse_sampling_decision(scores, labels, perm, 60, eps, delta, "fpc")
            d4 = budget_decision(scores, labels, perm, int(0.1 * N), int(0.6 * N), delta)
            exc["fixed_fpc"] += d1["loss"] > eps
            exc["fixed_bin"] += d2["loss"] > eps
            exc["invK"] += d3["loss"] > eps
            exc["budget"] += d4["loss"] > d4["eps_hat"] + 1e-12
        rates = {k: v / mc_reps for k, v in exc.items()}
        assert all(r <= delta + slack for r in rates.values()), rates
        print(f"[ok] N={N} M={M} ties={ties}: exceedance " +
              ", ".join(f"{k}={v:.3f}" for k, v in rates.items()) + f" (delta={delta})")

    # 3b) grid bound is never below the exact bound
    scores, labels = _synthetic_library(3000, 600, 1.0, rng)
    for _ in range(20):
        perm = rng.permutation(3000)
        g = budget_decision(scores, labels, perm, 300, 1500, 0.1)["eps_hat"]
        e = budget_decision(scores, labels, perm, 300, 1500, 0.1, exact=True)["eps_hat"]
        assert g >= e - 1e-12 and g - e < 0.02, (g, e)
    print("[ok] grid bound >= exact bound (and within 2 pp) on 20 draws")

    # 3c) capped inverse sampling never skips when K failures are not reached within the cap
    scores, labels = _synthetic_library(2000, 40, 1.5, rng)
    for _ in range(50):
        d = inverse_sampling_decision(scores, labels, rng.permutation(2000), 45, 0.05, 0.1,
                                      n_cap=600)
        assert d["saving"] == 0.0 and d["n_cal"] == 600
    print("[ok] capped Target-K skips nothing when fewer than K failures exist")

    # 4) sanity values quoted in the revision brief (paper's Table I libraries, n = round(0.2N))
    lib = {"DS-R": (13565, 158), "DS-G": (13262, 192), "OP-0.10.0": (10000, 306)}
    got = {k: predicted_refusal(N, M, round(0.2 * N), 0.05, 0.10) for k, (N, M) in lib.items()}
    assert abs(got["DS-R"] - 0.9936) < 0.001 and abs(got["DS-G"] - 0.8657) < 0.001
    assert abs(got["OP-0.10.0"] - 0.0062) < 0.001
    print("[ok] predicted refusal at 20% calibration: " +
          ", ".join(f"{k}={v:.4f}" for k, v in got.items()))
    assert jstar_fpc(1149, 12496, 2499, 0.05, 0.1) == 54 and jstar_binomial(1149, 0.05, 0.1) == 48
    print("[ok] SD-A at m=1149: j*_fpc=54, j*_binomial=48")
    print("all self-tests passed")


if __name__ == "__main__":
    self_test()
