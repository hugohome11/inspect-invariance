"""Asymptotic covariance of the tetrachoric correlation estimates.

This module supplies the piece that was missing and that made every fit index in
:mod:`inspect_invariance.invariance` untrustworthy. A tetrachoric correlation
matrix is not a sample covariance matrix. Feeding it to a normal-theory
maximum-likelihood discrepancy and multiplying by ``N - 1`` produces a statistic
that is not distributed as chi-square, and in practice runs four- to sevenfold
inflated, which rejects correctly specified models. The repair is to weight the
residuals by how precisely each correlation is actually estimated, which means
knowing their sampling covariance.

How it is computed, which is exact rather than approximated
----------------------------------------------------------

Every quantity the tetrachoric estimator consumes is a sample mean of an
indicator:

* ``p_i = mean(y_i)``, the univariate proportion for item *i*;
* ``p_ij = mean(y_i * y_j)``, the joint proportion for the pair.

Stack those into one vector ``z``. Its sampling covariance is just the covariance
of the underlying indicator vectors divided by *N*, which is exact and assumes
nothing about normality: the central limit theorem applies to means of bounded
random variables whatever their distribution.

The tetrachoric correlation is then a smooth implicit function of three of those
means. Writing ``a_i = Phi^-1(p_i)`` (the negated threshold) the estimator solves

    Phi_2(a_i, a_j; rho) = p_ij

so the delta method gives the Jacobian in closed form, with no numerical
differencing:

    d(rho)/d(p_ij) = 1 / phi_2(a_i, a_j; rho)
    d(rho)/d(p_i)  = -Phi((a_j - rho a_i) / sqrt(1 - rho^2)) / phi_2(...)
    d(rho)/d(p_j)  = -Phi((a_i - rho a_j) / sqrt(1 - rho^2)) / phi_2(...)

using ``d(Phi_2)/d(a) = phi(a) Phi((b - rho a)/sqrt(1 - rho^2))`` and
``d(a_i)/d(p_i) = 1/phi(a_i)``, which cancel.

That Jacobian propagates the exact covariance of ``z`` into the covariance of the
correlation estimates. Two consequences worth stating. It carries the
threshold-estimation uncertainty rather than conditioning it away, which matters
most for the skewed items that evaluation data is full of. And it needs neither
the inverse nor the determinant of the correlation matrix, so an indefinite
sample matrix is no longer fatal.

Reference for the two-stage estimator and its covariance: Muthen (1984),
Joreskog (1994). The weighting scheme they support is in
:mod:`inspect_invariance.invariance`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = [
    "MAX_ITEMS_FULL_COVARIANCE",
    "item_pairs",
    "correlation_estimates",
    "asymptotic_covariance",
    "asymptotic_variances",
]

MAX_ITEMS_FULL_COVARIANCE = 60
"""Item count above which the full asymptotic covariance is not formed.

The matrix is ``m`` by ``m`` where ``m = k(k - 1)/2``, so it grows as the fourth
power of the item count: 60 items is 1,770 correlations and a 1,770-square matrix,
which is 25 MB and fine. 500 items would be 124,750 correlations and a matrix of
1.6e10 entries, which is not.

This is not only a computational limit. A single common factor over several
hundred heterogeneous items is not a model worth fitting in the first place, and
the sensible unit for an invariance test is a subscale or a subject, not an entire
mixed item bank. :func:`inspect_invariance.invariance.check_invariance` refuses
above this and says so.
"""

_MIN_DENSITY = 1e-12
"""Floor on the bivariate density in the Jacobian denominator.

The density at the thresholds goes to zero as a correlation approaches the
boundary, and the derivative of rho with respect to the observed proportions
diverges there. That divergence is real: the correlation is genuinely almost
unidentified in that corner, and the large variance this produces is the correct
answer rather than a numerical nuisance.
"""


def item_pairs(n_items: int) -> list[tuple[int, int]]:
    """The ``(i, j)`` pairs with ``i < j``, in row-major order.

    This ordering defines the layout of every vector and matrix in this module
    and in the fitting code, so it is defined once here.
    """
    return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]


def _bivariate_density(a: float, b: float, rho: float) -> float:
    """Standard bivariate normal density at ``(a, b)`` with correlation ``rho``."""
    one_minus = max(1.0 - rho * rho, 1e-12)
    quad = (a * a - 2.0 * rho * a * b + b * b) / one_minus
    return math.exp(-0.5 * quad) / (2.0 * math.pi * math.sqrt(one_minus))


def correlation_estimates(
    data: np.ndarray, *, clip: float = 0.999
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Tetrachoric correlations as a vector, with the normal scores behind them.

    Args:
        data: ``(n_respondents, n_items)`` matrix of 0/1 responses.
        clip: bound on the returned correlations.

    Returns:
        ``(rho, a, pairs)`` where ``rho`` has one entry per pair in
        :func:`item_pairs` order, and ``a[i] = Phi^-1(P(y_i = 1))`` is the
        negated threshold for item *i*.
    """
    from .invariance import _tetrachoric_pair  # local import avoids a cycle

    data = np.asarray(data)
    n_items = data.shape[1]
    p = data.mean(axis=0)
    p_safe = np.clip(p, 1e-6, 1.0 - 1e-6)
    a = stats.norm.ppf(p_safe)
    tau = -a

    pairs = item_pairs(n_items)
    rho = np.zeros(len(pairs))
    for index, (i, j) in enumerate(pairs):
        yi, yj = data[:, i], data[:, j]
        n11 = float(np.sum((yi == 1) & (yj == 1)))
        n10 = float(np.sum((yi == 1) & (yj == 0)))
        n01 = float(np.sum((yi == 0) & (yj == 1)))
        n00 = float(np.sum((yi == 0) & (yj == 0)))
        rho[index] = float(
            np.clip(_tetrachoric_pair(n11, n10, n01, n00, tau[i], tau[j]), -clip, clip)
        )
    return rho, a, pairs


def _jacobian(
    rho: np.ndarray, a: np.ndarray, pairs: list[tuple[int, int]], n_items: int
) -> np.ndarray:
    """Derivative of the correlation vector with respect to the sample means.

    The sample-mean vector is laid out as the ``n_items`` univariate proportions
    followed by the pairwise joint proportions in :func:`item_pairs` order, so
    the result is ``(m, n_items + m)`` and very sparse: three non-zeros per row.
    """
    m = len(pairs)
    g = np.zeros((m, n_items + m))
    for index, (i, j) in enumerate(pairs):
        r = float(rho[index])
        ai, aj = float(a[i]), float(a[j])
        root = math.sqrt(max(1.0 - r * r, 1e-12))
        density = max(_bivariate_density(ai, aj, r), _MIN_DENSITY)

        g[index, n_items + index] = 1.0 / density
        g[index, i] = -float(stats.norm.cdf((aj - r * ai) / root)) / density
        g[index, j] = -float(stats.norm.cdf((ai - r * aj) / root)) / density
    return g


def _statistic_matrix(data: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Per-respondent indicators whose means are the sufficient statistics."""
    data = np.asarray(data, dtype=float)
    products = np.empty((data.shape[0], len(pairs)))
    for index, (i, j) in enumerate(pairs):
        products[:, index] = data[:, i] * data[:, j]
    return np.hstack([data, products])


def asymptotic_covariance(
    data: np.ndarray,
    rho: np.ndarray,
    a: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Full sampling covariance of the tetrachoric correlation estimates.

    Returns an ``(m, m)`` matrix already divided by the respondent count, so it
    is the covariance of the estimates themselves and not an ``N``-scaled
    version. That is what the test-statistic corrections expect.

    Raises:
        ValueError: if the item count exceeds
            :data:`MAX_ITEMS_FULL_COVARIANCE`.
    """
    data = np.asarray(data)
    n_respondents, n_items = data.shape
    if n_items > MAX_ITEMS_FULL_COVARIANCE:
        raise ValueError(
            f"{n_items} items would need a "
            f"{len(pairs)}-square asymptotic covariance matrix; the limit is "
            f"{MAX_ITEMS_FULL_COVARIANCE} items. Test invariance within a "
            "subscale or subject rather than across a whole item bank."
        )
    if n_respondents < 2:
        raise ValueError("the asymptotic covariance needs at least 2 respondents")

    z = _statistic_matrix(data, pairs)
    # rowvar=False treats columns as variables. ddof=0 rather than 1: this is a
    # plug-in asymptotic covariance, and the closed-form moments in
    # asymptotic_variances (p(1 - p) and so on) are the ddof=0 versions. The two
    # differ only by N/(N - 1), but the diagonal of this matrix and that function
    # must agree exactly, because the fit uses one and the correction uses the
    # other and a silent mismatch between them is untestable from outside.
    sigma_z = np.cov(z, rowvar=False, ddof=0)
    g = _jacobian(rho, a, pairs, n_items)
    gamma = g @ sigma_z @ g.T / n_respondents
    # Symmetrise to kill accumulated floating-point asymmetry.
    return (gamma + gamma.T) / 2.0


def asymptotic_variances(
    data: np.ndarray,
    rho: np.ndarray,
    a: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Just the diagonal of :func:`asymptotic_covariance`, cheaply.

    Diagonally weighted least squares needs only these, and each one depends on
    three of the sufficient statistics, so this costs ``O(m)`` rather than
    ``O(m^2)`` and has no item-count limit. Used for the fit itself; the full
    matrix is needed only for the test-statistic correction.
    """
    data = np.asarray(data, dtype=float)
    n_respondents, n_items = data.shape
    if n_respondents < 2:
        raise ValueError("the asymptotic covariance needs at least 2 respondents")

    p = data.mean(axis=0)
    variances = np.zeros(len(pairs))
    for index, (i, j) in enumerate(pairs):
        r = float(rho[index])
        ai, aj = float(a[i]), float(a[j])
        root = math.sqrt(max(1.0 - r * r, 1e-12))
        density = max(_bivariate_density(ai, aj, r), _MIN_DENSITY)

        d_pij = 1.0 / density
        d_pi = -float(stats.norm.cdf((aj - r * ai) / root)) / density
        d_pj = -float(stats.norm.cdf((ai - r * aj) / root)) / density

        # Exact covariance of the three Bernoulli means involved. Writing
        # u = y_i, v = y_j, w = y_i y_j, every moment is a proportion: E[uw] =
        # E[u v] = p_ij because w is already the product, and E[w^2] = p_ij.
        yi, yj = data[:, i], data[:, j]
        pij = float(np.mean(yi * yj))
        pi, pj = float(p[i]), float(p[j])
        cov = np.array(
            [
                [pi * (1 - pi), pij - pi * pj, pij * (1 - pi)],
                [pij - pi * pj, pj * (1 - pj), pij * (1 - pj)],
                [pij * (1 - pi), pij * (1 - pj), pij * (1 - pij)],
            ]
        )
        grad = np.array([d_pi, d_pj, d_pij])
        variances[index] = float(grad @ cov @ grad) / n_respondents
    return variances
