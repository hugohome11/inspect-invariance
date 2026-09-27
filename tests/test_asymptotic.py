"""Tests for the asymptotic covariance of the tetrachoric estimates.

This is the quantity the whole weighted estimator rests on, so it is checked
against the thing it claims to be: the actual sampling variability of the
estimator. A closed-form derivative that is subtly wrong would produce a fit that
looks respectable and is silently miscalibrated, which is the failure mode this
package was itself guilty of, so the validation here is a Monte Carlo comparison
rather than an internal consistency check.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspect_invariance.asymptotic import (
    MAX_ITEMS_FULL_COVARIANCE,
    asymptotic_covariance,
    asymptotic_variances,
    correlation_estimates,
    item_pairs,
)

LOADINGS = np.array([0.8, 0.7, 0.6, 0.5, 0.7])
# Thresholds spread wide on purpose: a skewed item is where a naive variance
# estimate goes wrong, so the easy symmetric case would not be a real test.
THRESHOLDS = np.array([-0.9, -0.3, 0.0, 0.4, 0.9])


def _generate(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    resid = np.sqrt(1.0 - LOADINGS**2)
    ystar = np.outer(theta, LOADINGS) + rng.normal(size=(n, len(LOADINGS))) * resid
    return (ystar > THRESHOLDS).astype(int)


def test_pairs_are_upper_triangular_in_row_major_order():
    assert item_pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_correlations_recover_the_generating_loadings():
    data = _generate(4000, 0)
    rho, a, pairs = correlation_estimates(data)
    expected = np.array([LOADINGS[i] * LOADINGS[j] for i, j in pairs])
    assert np.abs(rho - expected).max() < 0.06
    # a is the negated threshold, so it should recover -THRESHOLDS.
    assert np.abs(a + THRESHOLDS).max() < 0.06


def test_the_cheap_diagonal_matches_the_full_matrix():
    """The two routines must agree, or the fit and its correction disagree."""
    data = _generate(600, 1)
    rho, a, pairs = correlation_estimates(data)
    full = asymptotic_covariance(data, rho, a, pairs)
    cheap = asymptotic_variances(data, rho, a, pairs)
    assert np.allclose(np.diag(full), cheap, rtol=1e-8, atol=1e-12)


def test_the_full_covariance_is_symmetric_and_positive_semidefinite():
    data = _generate(600, 2)
    rho, a, pairs = correlation_estimates(data)
    gamma = asymptotic_covariance(data, rho, a, pairs)
    assert np.allclose(gamma, gamma.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(gamma).min()) > -1e-10


def test_variances_shrink_like_one_over_n():
    """Doubling the sample should roughly halve the variance."""
    small, large = _generate(500, 3), _generate(2000, 3)
    ratios = []
    for data in (small, large):
        rho, a, pairs = correlation_estimates(data)
        ratios.append(asymptotic_variances(data, rho, a, pairs).mean())
    assert 3.0 < ratios[0] / ratios[1] < 5.5


@pytest.mark.parametrize("n", [200, 1000])
def test_predicted_variance_matches_the_sampling_variance(n):
    """The decisive test: Monte Carlo against the closed form.

    Anything much outside a ratio of one means the delta-method Jacobian is
    wrong, and every fit statistic downstream would be wrong with it.
    """
    replications = 150
    estimates, predicted = [], []
    for rep in range(replications):
        data = _generate(n, 5000 + rep)
        rho, a, pairs = correlation_estimates(data)
        estimates.append(rho)
        predicted.append(asymptotic_variances(data, rho, a, pairs))

    empirical = np.array(estimates).var(axis=0, ddof=1)
    expected = np.array(predicted).mean(axis=0)
    ratio = expected / empirical
    assert 0.75 < float(np.median(ratio)) < 1.30, f"median ratio {np.median(ratio):.3f}"


def test_a_skewed_item_pair_is_marked_as_imprecise():
    """The point of weighting: a nearly constant item carries little information."""
    rng = np.random.default_rng(7)
    n = 800
    theta = rng.normal(size=n)
    lam = np.array([0.7, 0.7, 0.7, 0.7])
    resid = np.sqrt(1.0 - lam**2)
    ystar = np.outer(theta, lam) + rng.normal(size=(n, 4)) * resid
    # The last item is answered correctly by almost everyone.
    data = (ystar > np.array([0.0, 0.0, 0.0, -2.3])).astype(int)

    rho, a, pairs = correlation_estimates(data)
    variances = asymptotic_variances(data, rho, a, pairs)
    skewed = [k for k, (i, j) in enumerate(pairs) if 3 in (i, j)]
    balanced = [k for k, (i, j) in enumerate(pairs) if 3 not in (i, j)]
    assert variances[skewed].mean() > 2.0 * variances[balanced].mean()


def test_too_many_items_is_refused_rather_than_attempted():
    k = MAX_ITEMS_FULL_COVARIANCE + 1
    rng = np.random.default_rng(9)
    data = (rng.normal(size=(120, k)) > 0).astype(int)
    rho, a, pairs = correlation_estimates(data)
    with pytest.raises(ValueError, match="limit is"):
        asymptotic_covariance(data, rho, a, pairs)


def test_one_respondent_is_refused():
    data = np.array([[1, 0, 1, 0]])
    rho, a, pairs = correlation_estimates(data)
    with pytest.raises(ValueError, match="at least 2 respondents"):
        asymptotic_variances(data, rho, a, pairs)
