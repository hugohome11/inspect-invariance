"""Multi-group measurement invariance for two or more language versions of a form.

The question this answers is not "does the model score lower in Croatian" but
"do the Croatian and English forms measure the same thing at all". If they do
not, the score difference between them has no defined meaning and should not be
reported as a finding about the model.

Invariance is tested as a sequence of nested models, each adding a constraint:

``configural``
    The same one-factor structure holds in every language, with every loading
    free. This is the weakest claim: the forms measure *a* common dimension,
    though not necessarily on the same scale. If configural fit is poor, stop:
    the construct itself differs across languages and nothing further is
    interpretable.

``metric``
    Loadings are additionally held equal across languages. This is the claim that
    a unit of the latent trait buys the same amount of item response everywhere,
    so *differences* and *correlations* are comparable. Failure means at least one
    item discriminates differently in one language.

Scalar invariance, which is what licenses comparing *means*, requires equal item
intercepts. For binary items the intercept is the item threshold, and a
difference in thresholds at equal ability is exactly what uniform DIF is. So this
module deliberately stops at metric, and the scalar-level question is answered
item by item by :mod:`inspect_invariance.dif`, which measures the same thing on a
scale that item review can act on. Reporting a single scalar chi-square would be
less informative than naming the items, not more.

Fit is evaluated on the tetrachoric correlation matrix, which is the correct
input for binary indicators: a Pearson correlation between two dichotomised
variables is attenuated by their difficulty split, and would understate loadings
for hard or easy items.

Limits, stated rather than buried. This is a screening procedure. A full
treatment would estimate thresholds and loadings jointly by WLSMV with a
mean-and-variance-adjusted test statistic, which is what lavaan or Mplus would do
and what a published claim should rest on. What is implemented here is a
maximum-likelihood fit to the tetrachoric correlation matrix, which is adequate
for deciding whether a benchmark is worth a closer look and not adequate as the
final word in a report to a regulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .matrix import ResponseMatrix

__all__ = ["FitStats", "InvarianceResult", "check_invariance", "tetrachoric_matrix"]

# Cheung and Rensvold (2002) for CFI; Chen (2007) for RMSEA.
DELTA_CFI_CUT = 0.010
DELTA_RMSEA_CUT = 0.015


@dataclass(frozen=True)
class FitStats:
    """Fit of one model in the sequence."""

    label: str
    chi2: float
    df: int
    p: float
    cfi: float
    rmsea: float
    converged: bool

    def __str__(self) -> str:
        return (
            f"{self.label:<11} chi2={self.chi2:9.2f}  df={self.df:4d}  "
            f"p={self.p:.4f}  CFI={self.cfi:.3f}  RMSEA={self.rmsea:.3f}"
        )


@dataclass(frozen=True)
class InvarianceResult:
    """Outcome of the configural and metric comparison."""

    languages: tuple[str, ...]
    items: tuple[str, ...]
    configural: FitStats
    metric: FitStats
    delta_cfi: float
    delta_rmsea: float
    chi2_diff: float
    df_diff: int
    p_diff: float

    @property
    def configural_holds(self) -> bool:
        """Whether a common one-factor structure is tenable at all."""
        return self.configural.cfi >= 0.90 and self.configural.rmsea <= 0.10

    @property
    def metric_holds(self) -> bool:
        """Whether loadings can be treated as equal across languages."""
        return (
            self.configural_holds
            and abs(self.delta_cfi) <= DELTA_CFI_CUT
            and abs(self.delta_rmsea) <= DELTA_RMSEA_CUT
        )

    @property
    def verdict(self) -> str:
        if not self.configural_holds:
            return (
                "configural invariance fails: the language versions do not share a "
                "common factor structure, so no cross-language comparison is defined"
            )
        if not self.metric_holds:
            return (
                "metric invariance fails: loadings differ across languages, so "
                "cross-language differences are not on a common scale"
            )
        return (
            "metric invariance holds: loadings are equivalent across languages, so "
            "cross-language comparison of differences is defensible at this level"
        )


def tetrachoric_matrix(data: np.ndarray, *, clip: float = 0.999) -> np.ndarray:
    """Tetrachoric correlation matrix for a 0/1 response matrix.

    Each pair is estimated by maximising the bivariate-normal likelihood of its
    2x2 table over the correlation, holding the thresholds at the values implied
    by the marginals. Constant items yield zero correlation, since a variable
    with no variance cannot covary with anything.
    """
    data = np.asarray(data)
    n_items = data.shape[1]
    out = np.eye(n_items)
    p = data.mean(axis=0)
    # Thresholds on the standard normal scale. A proportion of 1.0 or 0.0 has no
    # finite threshold, so it is nudged inside the open interval.
    p_safe = np.clip(p, 1e-6, 1 - 1e-6)
    tau = stats.norm.ppf(1.0 - p_safe)

    for i in range(n_items):
        if p[i] in (0.0, 1.0):
            continue
        for j in range(i + 1, n_items):
            if p[j] in (0.0, 1.0):
                continue
            n11 = float(np.sum((data[:, i] == 1) & (data[:, j] == 1)))
            n10 = float(np.sum((data[:, i] == 1) & (data[:, j] == 0)))
            n01 = float(np.sum((data[:, i] == 0) & (data[:, j] == 1)))
            n00 = float(np.sum((data[:, i] == 0) & (data[:, j] == 0)))
            rho = _tetrachoric_pair(n11, n10, n01, n00, tau[i], tau[j])
            out[i, j] = out[j, i] = float(np.clip(rho, -clip, clip))
    return out


def check_invariance(matrices: dict[str, ResponseMatrix]) -> InvarianceResult:
    """Run the configural and metric comparison across language versions.

    Args:
        matrices: language code to :class:`ResponseMatrix`. Every matrix must
            carry the same items; use :meth:`ResponseMatrix.aligned_to` first if
            they do not.

    Raises:
        ValueError: if fewer than two languages are supplied, or the item sets
            differ. Comparing forms with different items is not an invariance
            test, and silently intersecting them would hide that.
    """
    if len(matrices) < 2:
        raise ValueError("invariance needs at least two language versions")

    langs = tuple(matrices.keys())
    item_sets = {lang: tuple(m.items) for lang, m in matrices.items()}
    first = item_sets[langs[0]]
    for lang, items in item_sets.items():
        if items != first:
            raise ValueError(
                f"language {lang!r} has a different item set from {langs[0]!r}; "
                f"align them explicitly with ResponseMatrix.aligned_to"
            )

    n_items = len(first)
    if n_items < 4:
        raise ValueError(
            f"a one-factor model needs at least 4 items to be overidentified, got {n_items}"
        )

    corrs = [tetrachoric_matrix(matrices[l].data) for l in langs]
    ns = [matrices[l].n_respondents for l in langs]

    configural = _fit_configural(corrs, ns, n_items)
    metric = _fit_metric(corrs, ns, n_items)

    chi2_diff = max(0.0, metric.chi2 - configural.chi2)
    df_diff = metric.df - configural.df
    p_diff = float(stats.chi2.sf(chi2_diff, df_diff)) if df_diff > 0 else float("nan")

    return InvarianceResult(
        languages=langs,
        items=first,
        configural=configural,
        metric=metric,
        delta_cfi=metric.cfi - configural.cfi,
        delta_rmsea=metric.rmsea - configural.rmsea,
        chi2_diff=chi2_diff,
        df_diff=df_diff,
        p_diff=p_diff,
    )


# ---------------------------------------------------------------- internals


def _tetrachoric_pair(
    n11: float, n10: float, n01: float, n00: float, t_i: float, t_j: float
) -> float:
    total = n11 + n10 + n01 + n00
    if total == 0:
        return 0.0
    if min(n11, n10, n01, n00) == 0:
        # A zero cell drives the estimate to the boundary. Add the conventional
        # 0.5 continuity correction rather than returning a degenerate +/-1.
        n11, n10, n01, n00 = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
        total += 2.0

    def neg_ll(rho_arr: np.ndarray) -> float:
        rho = float(np.clip(rho_arr[0], -0.999, 0.999))
        p11 = _bvn_upper(t_i, t_j, rho)
        p_i = 1.0 - stats.norm.cdf(t_i)
        p_j = 1.0 - stats.norm.cdf(t_j)
        p10 = max(p_i - p11, 1e-12)
        p01 = max(p_j - p11, 1e-12)
        p00 = max(1.0 - p11 - p10 - p01, 1e-12)
        p11 = max(p11, 1e-12)
        return -(n11 * math.log(p11) + n10 * math.log(p10)
                 + n01 * math.log(p01) + n00 * math.log(p00))

    res = optimize.minimize_scalar(
        lambda r: neg_ll(np.array([r])), bounds=(-0.99, 0.99), method="bounded"
    )
    return float(res.x)


def _bvn_upper(h: float, k: float, rho: float) -> float:
    """P(X > h, Y > k) for a standard bivariate normal with correlation rho."""
    return float(stats.multivariate_normal.cdf(
        [-h, -k], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]
    ))


def _implied(loadings: np.ndarray) -> np.ndarray:
    """Correlation matrix implied by a one-factor model with unit factor variance."""
    lam = np.clip(loadings, -0.995, 0.995)
    sigma = np.outer(lam, lam)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def _discrepancy(s: np.ndarray, sigma: np.ndarray) -> float:
    """Maximum-likelihood discrepancy between observed and implied matrices."""
    p = s.shape[0]
    try:
        sign, logdet_sigma = np.linalg.slogdet(sigma)
        if sign <= 0:
            return 1e6
        sign_s, logdet_s = np.linalg.slogdet(s)
        if sign_s <= 0:
            logdet_s = -1e3
        inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return 1e6
    return float(logdet_sigma - logdet_s + np.trace(s @ inv) - p)


def _fit_group(s: np.ndarray, n_items: int) -> tuple[np.ndarray, float, bool]:
    def obj(lam: np.ndarray) -> float:
        return _discrepancy(s, _implied(lam))

    best, best_f, ok = None, np.inf, False
    for start in (0.6, 0.3, 0.8):
        res = optimize.minimize(
            obj, np.full(n_items, start), method="L-BFGS-B",
            bounds=[(-0.995, 0.995)] * n_items,
        )
        if res.fun < best_f:
            best, best_f, ok = res.x, float(res.fun), bool(res.success)
    return best, best_f, ok


def _fit_configural(corrs, ns, n_items) -> FitStats:
    total_f, ok = 0.0, True
    for s, n in zip(corrs, ns):
        _, f, conv = _fit_group(s, n_items)
        total_f += (n - 1) * f
        ok = ok and conv
    g = len(corrs)
    df = g * (n_items * (n_items - 1) // 2) - g * n_items
    return _stats("configural", total_f, df, corrs, ns, n_items, ok)


def _fit_metric(corrs, ns, n_items) -> FitStats:
    def obj(lam: np.ndarray) -> float:
        sigma = _implied(lam)
        return sum((n - 1) * _discrepancy(s, sigma) for s, n in zip(corrs, ns))

    best_f, ok = np.inf, False
    for start in (0.6, 0.3, 0.8):
        res = optimize.minimize(
            obj, np.full(n_items, start), method="L-BFGS-B",
            bounds=[(-0.995, 0.995)] * n_items,
        )
        if res.fun < best_f:
            best_f, ok = float(res.fun), bool(res.success)
    g = len(corrs)
    df = g * (n_items * (n_items - 1) // 2) - n_items
    return _stats("metric", best_f, df, corrs, ns, n_items, ok)


def _stats(label, weighted_f, df, corrs, ns, n_items, converged) -> FitStats:
    chi2 = max(0.0, weighted_f)
    p = float(stats.chi2.sf(chi2, df)) if df > 0 else float("nan")

    # Independence baseline: implied matrix is the identity, so the discrepancy
    # reduces to -log|S| per group.
    base_f = sum((n - 1) * _discrepancy(s, np.eye(n_items)) for s, n in zip(corrs, ns))
    g = len(corrs)
    base_df = g * (n_items * (n_items - 1) // 2)

    d_model = max(chi2 - df, 0.0)
    d_base = max(base_f - base_df, 0.0)
    cfi = 1.0 if d_base <= 0 else float(np.clip(1.0 - d_model / d_base, 0.0, 1.0))

    n_total = sum(ns)
    rmsea = (
        math.sqrt(d_model / (df * (n_total - g))) * math.sqrt(g)
        if df > 0 and n_total > g
        else float("nan")
    )
    return FitStats(label, chi2, df, p, cfi, rmsea, converged)
