import numpy as np

from rmrobust import stats as S


def test_bootstrap_covers_the_truth():
    rng = np.random.default_rng(0)
    hits = 0
    for s in range(60):
        x = rng.normal(0.7, 1.0, 300)
        e = S.bootstrap(x, n_boot=400, seed=s)
        hits += e.lo <= 0.7 <= e.hi
    assert hits >= 50, hits


def test_paired_bootstrap_is_tighter_than_unpaired_for_correlated_arms():
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 1, 500)
    a = shared + rng.normal(0.1, 0.05, 500)
    b = shared + rng.normal(0.0, 0.05, 500)
    paired = S.paired_bootstrap_diff(a, b, n_boot=800)
    ea, eb = S.bootstrap(a, n_boot=800), S.bootstrap(b, n_boot=800)
    unpaired_width = (ea.hi - ea.lo) + (eb.hi - eb.lo)
    assert (paired.hi - paired.lo) < 0.2 * unpaired_width


def test_cluster_bootstrap_is_wider_than_item_bootstrap_when_clusters_are_real():
    rng = np.random.default_rng(2)
    topic_effect = rng.normal(0, 1.0, 20)
    vals, clusters = [], []
    for t, eff in enumerate(topic_effect):
        for _ in range(5):  # 5 near-identical templates per topic
            vals.append(eff + rng.normal(0, 0.02))
            clusters.append(t)
    item = S.bootstrap(vals, n_boot=800)
    clus = S.cluster_bootstrap(vals, clusters, n_boot=800)
    # With k near-identical items per topic the item-level interval is too narrow by
    # about sqrt(k); here k = 5, so expect a factor near 2.24.
    ratio = (clus.hi - clus.lo) / (item.hi - item.lo)
    assert 1.7 < ratio < 3.2, ratio


def test_isotonic_residualisation_removes_a_monotone_signal():
    rng = np.random.default_rng(3)
    x = rng.uniform(1, 6, 2000)
    y = np.tanh(x - 3) + rng.normal(0, 0.1, 2000)
    m = S.IsotonicLengthModel().fit(x, y)
    assert m.r2(x, y) > 0.9
    resid = m.residualize(x, y)
    assert abs(np.corrcoef(x, resid)[0, 1]) < 0.1


def test_ols_intercept_recovers_the_effect_at_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 400)
    y = 0.3 + 0.8 * x + rng.normal(0, 0.1, 400)
    r = S.ols_intercept_slope(x, y, n_boot=400)
    assert abs(r["intercept"] - 0.3) < 0.03
    assert r["intercept_ci"][0] <= 0.3 <= r["intercept_ci"][1]


def test_ece_is_zero_for_a_calibrated_predictor():
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)
    assert S.expected_calibration_error(p, y) < 0.02
