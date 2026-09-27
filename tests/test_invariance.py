"""Invariance tests, again against planted ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_invariance.invariance import check_invariance, tetrachoric_matrix
from inspect_invariance.matrix import ResponseMatrix

ITEMS = tuple(f"i{k}" for k in range(8))
THRESHOLDS = [-0.5, -0.2, 0.0, 0.2, 0.5, -0.3, 0.1, 0.3]


def generate(n, loadings, thresholds=THRESHOLDS, seed=11):
    """One-factor probit generation: y* = lam * theta + e, y = 1 where y* > tau."""
    rng = np.random.default_rng(seed)
    lam = np.asarray(loadings, dtype=float)
    resid = np.sqrt(np.clip(1.0 - lam**2, 1e-6, None))
    theta = rng.normal(size=n)
    ystar = theta[:, None] * lam[None, :] + rng.normal(size=(n, len(lam))) * resid[None, :]
    return (ystar > np.asarray(thresholds)[None, :]).astype(int)


def matrix(data, lang):
    return ResponseMatrix(
        language=lang,
        items=ITEMS,
        respondents=tuple(f"r{i}" for i in range(len(data))),
        data=data,
        design="models",
    )


def test_equal_loadings_support_metric_invariance():
    equal = [0.7] * 8
    result = check_invariance({
        "en": matrix(generate(800, equal, seed=1), "en"),
        "hr": matrix(generate(800, equal, seed=2), "hr"),
    })
    assert result.configural_holds
    assert result.metric_holds
    assert abs(result.delta_cfi) <= 0.010


def test_unequal_loadings_break_metric_invariance():
    equal = [0.7] * 8
    unequal = [0.7, 0.15, 0.7, 0.7, 0.15, 0.7, 0.15, 0.7]
    result = check_invariance({
        "en": matrix(generate(800, equal, seed=1), "en"),
        "hr": matrix(generate(800, unequal, seed=2), "hr"),
    })
    assert result.configural_holds, "each language alone should still be one-factor"
    assert not result.metric_holds
    assert result.delta_cfi < -0.010
    assert "metric invariance fails" in result.verdict


def test_the_difference_test_counts_the_actual_restrictions():
    """df_diff must count restrictions, not differenced adjusted df.

    Eight loadings are constrained equal and one factor variance is freed, so
    seven parameters go. The adjusted degrees of freedom are rescaled per model
    and their difference is not the number of restrictions: differencing them
    gave 1 here, which is the bug this pins.
    """
    equal = [0.7] * 8
    result = check_invariance({
        "en": matrix(generate(600, equal, seed=1), "en"),
        "hr": matrix(generate(600, equal, seed=2), "hr"),
    })
    assert result.df_diff == 7


def test_adjusted_statistics_are_not_monotone_in_nesting():
    """Documented, not a defect, and the reason the scaled difference test exists.

    Under maximum likelihood a more constrained model cannot fit better, so
    chi-square is monotone in nesting. A mean-and-variance adjusted statistic
    rescales each model by its own correction factor, so the metric statistic can
    come out below the configural one. Anyone tempted to add a monotonicity
    assertion should read this first.
    """
    equal = [0.7] * 8
    result = check_invariance({
        "en": matrix(generate(600, equal, seed=1), "en"),
        "hr": matrix(generate(600, equal, seed=2), "hr"),
    })
    # Both are finite and positive; their order is not guaranteed either way.
    assert result.configural.chi2 > 0 and result.metric.chi2 > 0
    assert result.df_diff > 0, "the metric model really is more constrained"


def test_tetrachoric_recovers_a_known_correlation():
    """Tetrachoric correlation should recover the latent correlation, where a
    Pearson correlation on the dichotomised variables would be attenuated."""
    rng = np.random.default_rng(3)
    rho = 0.6
    cov = [[1.0, rho], [rho, 1.0]]
    latent = rng.multivariate_normal([0, 0], cov, size=20000)
    binary = (latent > 0.4).astype(int)  # an asymmetric split, which attenuates most

    tet = tetrachoric_matrix(binary)[0, 1]
    pearson = np.corrcoef(binary[:, 0], binary[:, 1])[0, 1]

    assert abs(tet - rho) < 0.06, f"tetrachoric {tet:.3f} should recover {rho}"
    assert pearson < tet - 0.10, "the Pearson correlation should be visibly attenuated"


def test_single_language_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        check_invariance({"en": matrix(generate(200, [0.7] * 8), "en")})


def test_mismatched_item_sets_are_rejected_rather_than_intersected():
    a = matrix(generate(300, [0.7] * 8, seed=1), "en")
    b = ResponseMatrix("hr", ITEMS[:6], a.respondents, generate(300, [0.7] * 6,
                       THRESHOLDS[:6], seed=2), "models")
    with pytest.raises(ValueError, match="different item set"):
        check_invariance({"en": a, "hr": b})


def test_too_few_items_is_rejected():
    small_items = ITEMS[:3]
    data = generate(300, [0.7] * 3, THRESHOLDS[:3])
    m = {
        l: ResponseMatrix(l, small_items, tuple(f"r{i}" for i in range(300)), data, "models")
        for l in ("en", "hr")
    }
    with pytest.raises(ValueError, match="at least 4 items"):
        check_invariance(m)
