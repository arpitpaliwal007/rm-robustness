"""Statistics shared by the probes.

Everything that reports a number reports an uncertainty with it. At HH test-set
sizes (1k-3k pairs per subset) a 2-point accuracy difference is inside the noise,
and half the published claims about reward models are 2-point differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Estimate:
    value: float
    lo: float
    hi: float
    n: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.value:.4f} [{self.lo:.4f}, {self.hi:.4f}] (n={self.n})"

    def as_dict(self) -> dict:
        return {"value": self.value, "ci_lo": self.lo, "ci_hi": self.hi, "n": self.n}


def bootstrap(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(stat(x)), float(lo), float(hi), n)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """CI on mean(a) - mean(b) for paired observations (same items, two scorers).

    Paired resampling matters: the two accuracies share the same test items, so the
    unpaired CIs on each overlap far more than the CI on their difference.
    """
    x, y = np.asarray(a, float), np.asarray(b, float)
    assert len(x) == len(y), "paired_bootstrap_diff needs aligned arrays"
    n = len(x)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = (x[idx].mean(axis=1) - y[idx].mean(axis=1))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(x.mean() - y.mean()), float(lo), float(hi), n)


def accuracy(margins: Sequence[float], ties_count_half: bool = True) -> np.ndarray:
    """Per-item correctness from signed margins r(chosen) - r(rejected)."""
    m = np.asarray(margins, dtype=float)
    correct = (m > 0).astype(float)
    if ties_count_half:
        correct[m == 0] = 0.5
    return correct


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cohens_d(deltas: Sequence[float]) -> float:
    """Standardised effect size for a paired intervention: mean shift in units of
    its own sd. The natural currency for 'how big is this bias' questions."""
    d = np.asarray(deltas, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else float("nan")


def expected_calibration_error(probs: Sequence[float], labels: Sequence[float], n_bins: int = 10) -> float:
    p = np.asarray(probs, float)
    y = np.asarray(labels, float)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import ks_2samp

    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


class IsotonicLengthModel:
    """Monotone fit of reward on length, r ~ g(length).

    Isotonic rather than linear because the empirical reward-length curve saturates:
    a linear fit understates how much of the reward is length at the short end and
    overstates it at the long end. Monotonicity is the assumption we are willing to
    make (the bias is directional); functional form is not.
    """

    def __init__(self, increasing: str = "auto"):
        self.increasing = increasing
        self._iso = None
        self._mean = 0.0

    def fit(self, lengths: Sequence[float], rewards: Sequence[float]) -> "IsotonicLengthModel":
        import warnings

        from sklearn.isotonic import IsotonicRegression

        x = np.asarray(lengths, float)
        y = np.asarray(rewards, float)
        self._mean = float(y.mean())
        self._iso = IsotonicRegression(increasing=self.increasing, out_of_bounds="clip")
        with warnings.catch_warnings():
            # sklearn warns when it cannot confidently infer the direction. That is the
            # honest situation when reward barely depends on length, and the fitted step
            # function is near-constant either way, which is the right answer.
            warnings.simplefilter("ignore")
            self._iso.fit(x, y)
        return self

    def predict(self, lengths: Sequence[float]) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("fit first")
        return np.asarray(self._iso.predict(np.asarray(lengths, float)), dtype=float)

    def residualize(self, lengths: Sequence[float], rewards: Sequence[float]) -> np.ndarray:
        return np.asarray(rewards, float) - self.predict(lengths)

    def r2(self, lengths: Sequence[float], rewards: Sequence[float]) -> float:
        y = np.asarray(rewards, float)
        pred = self.predict(lengths)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_logistic_1d(x: Sequence[float], y: Sequence[float], seed: int = 0):
    """Logistic regression of a binary outcome on one covariate. Returns (model, fn)."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(x, float).reshape(-1, 1)
    yy = np.asarray(y, float)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X, yy)
    return clf


def cross_val_accuracy(x: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Per-item out-of-fold correctness for a logistic model on features `x`.

    Out-of-fold, not in-sample: the length baseline is a *model* and reporting its
    training accuracy against a held-out RM accuracy would flatter it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(x, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, float)
    out = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        # Standardise inside the fold. Raw character counts and 0/1 indicators differ by
        # three orders of magnitude, which makes lbfgs crawl and the penalty meaningless.
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        out[te] = (clf.predict(X[te]) == y[te]).astype(float)
    return out


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Bootstrap that resamples whole clusters.

    Probe items generated from the same topic are not independent observations. Item-level
    bootstrap on templated probe sets produces confidence intervals that are too narrow by
    roughly the square root of the number of templates per topic, which is how a probe set
    of 36 items starts reporting the precision of 36 independent measurements.
    """
    x = np.asarray(values, dtype=float)
    keys = np.asarray(list(clusters))
    uniq = np.unique(keys)
    groups = [x[keys == u] for u in uniq]
    if len(uniq) == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(stat(x)), float(lo), float(hi), len(x))


def ols_intercept_slope(x: Sequence[float], y: Sequence[float], n_boot: int = 2000, seed: int = 0):
    """Simple regression with bootstrap CIs. Used to report an effect *at zero length
    difference*: the intercept of effect-on-delta-length is the part of a probe result
    that length cannot explain."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 5 or x.std() == 0:
        return {"intercept": float(y.mean()) if n else float("nan"), "slope": float("nan"),
                "intercept_ci": [float("nan"), float("nan")], "n": n}
    b, a = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ints = []
    for i in idx:
        if x[i].std() == 0:
            continue
        bb, aa = np.polyfit(x[i], y[i], 1)
        ints.append(aa)
    lo, hi = (np.percentile(ints, [2.5, 97.5]) if ints else (np.nan, np.nan))
    return {"intercept": float(a), "slope": float(b), "intercept_ci": [float(lo), float(hi)], "n": n}
