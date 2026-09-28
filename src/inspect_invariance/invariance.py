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
    Loadings are additionally held equal across languages, up to one factor
    variance per language. This is the claim that a unit of the latent trait buys
    the same amount of item response everywhere, so *differences* and
    *correlations* are comparable. Failure means at least one item discriminates
    differently in one language, and not merely that one language varies more.

Scalar invariance, which is what licenses comparing *means*, requires equal item
intercepts. For binary items the intercept is the item threshold, and a
difference in thresholds at equal ability is exactly what uniform DIF is. So this
module deliberately stops at metric, and the scalar-level question is answered
item by item by :mod:`inspect_invariance.dif`, which measures the same thing on a
scale that item review can act on. Reporting a single scalar chi-square would be
less informative than naming the items, not more.

Estimation
----------

Fit is on the tetrachoric correlations, which are the correct input for binary
indicators: a Pearson correlation between two dichotomised variables is
attenuated by their difficulty split, and would understate loadings for hard or
easy items.

They are fitted by **diagonally weighted least squares**, with a
mean-and-variance-adjusted test statistic. Each residual correlation is weighted
by the reciprocal of its own sampling variance, taken from
:mod:`inspect_invariance.asymptotic`, and the statistic is then corrected using
the full sampling covariance so that it is actually distributed as the reported
degrees of freedom claim. This is the WLSMV estimator in the sense of Muthen
(1984) and Asparouhov and Muthen (2010), and it is what lavaan and Mplus do for
ordinal indicators.

It replaces a normal-theory maximum-likelihood fit that treated the tetrachoric
matrix as though it were a sample covariance matrix. That earlier statistic was
not calibrated: it ran four- to sevenfold inflated and rejected correctly
specified models across the whole range of item counts and sample sizes a real
benchmark occupies. The full account is in the description of pull request #1,
https://github.com/hugohome11/inspect-invariance/pull/1.

Two practical gains come with the change. Weighting by precision means a barely
estimable correlation, which is what a skewed item pair in a small sample
produces, no longer dominates the fit. And because only the residual vector is
weighted, the fit needs neither the inverse nor the determinant of the
correlation matrix, so an indefinite sample matrix is no longer fatal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .asymptotic import (
    MAX_ITEMS_FULL_COVARIANCE,
    asymptotic_covariance,
    correlation_estimates,
    item_pairs,
)
from .matrix import ResponseMatrix

__all__ = ["FitStats", "InvarianceResult", "check_invariance", "tetrachoric_matrix"]

# Cheung and Rensvold (2002) for CFI; Chen (2007) for RMSEA.
DELTA_CFI_CUT = 0.010
DELTA_RMSEA_CUT = 0.015

MAX_STACKED_RESIDUALS = 4000
"""Cap on languages times correlations, which is what the memory actually follows.

The corrected statistic needs the joint covariance of every residual correlation
in every group at once, so the array is the square of this number. At 4,000 that
is 128 MB, which is the largest that stays comfortable. The limit binds through
the product, so a wide language set and a long form trade against each other:
about 20 items across 18 languages fits, as does 60 items across 2.

An implementation exploiting the block structure of the covariance could push this
considerably further, since the group blocks are independent and only the shared
loadings couple them. It is not done.
"""

MIN_RESPONDENTS = 100
"""Respondents per language below which no fit statistics are reported.

The estimator is asymptotic in two places: the correlations themselves, and the
sampling covariance used to weight and to correct them. Below this count the
second becomes too noisy to correct with, and the adjusted statistic loses its
calibration even though the fit is still computable.

The floor is far lower than the one the previous maximum-likelihood
implementation needed, which is the point of the change, but it is a real floor
and not a formality. A respondent is a model-by-epoch cell, so reaching it means
either a panel of models or a generous ``--epochs``. The item-level DIF analysis
in :mod:`inspect_invariance.dif` needs far fewer and is the better tool at small
scale.
"""


@dataclass(frozen=True)
class FitStats:
    """Fit of one model in the sequence.

    ``chi2`` and ``df`` are the mean-and-variance adjusted statistic and its
    adjusted degrees of freedom. The adjusted degrees of freedom are not in
    general a whole number, and are rounded here only for display; the p-value
    uses the unrounded value.
    """

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
    calibrated: bool = True
    """Whether the test statistic these indices rest on is calibrated.

    True since the estimator became diagonally weighted least squares with a
    mean-and-variance adjusted statistic. It was False for the preceding
    normal-theory maximum-likelihood fit, which was not distributed as its
    degrees of freedom claimed, and while it was False these indices were
    reported as description only and took no part in any verdict.
    """

    @property
    def configural_holds(self) -> bool:
        """Whether a common one-factor structure is tenable at all."""
        return self.configural.cfi >= 0.90 and self.configural.rmsea <= 0.10

    @property
    def metric_holds(self) -> bool:
        """Whether loadings can be treated as equal across languages.

        Decided by the scaled difference test, not by a change in fit indices.
        That is a deliberate departure from the usual delta-CFI and delta-RMSEA
        rules of thumb, and it is forced by the estimator. Those cutoffs were
        calibrated for maximum likelihood, where nested models share their
        degrees of freedom; a mean-and-variance adjusted statistic rescales each
        model separately, so the adjusted degrees of freedom shift between
        configural and metric and the two indices are no longer on a common
        scale. Measured on correct models with no differential functioning at all,
        the delta-RMSEA rule rejected 8 of 30 and delta-CFI 3 of 30, while the
        scaled difference test rejected 0 of 30.

        The deltas remain on the result as description, because they are what
        readers expect to see, and because a large one is still worth looking at.
        They just do not decide the question.
        """
        if not self.configural_holds:
            return False
        if math.isfinite(self.p_diff):
            return self.p_diff >= 0.05
        # No usable difference test, normally a barely identified model. Fall
        # back to the fit-index rule rather than assuming either answer.
        return (
            abs(self.delta_cfi) <= DELTA_CFI_CUT
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
    two-by-two table, with the thresholds fixed at the observed marginals.
    """
    data = np.asarray(data)
    n_items = data.shape[1]
    out = np.eye(n_items)
    rho, _, pairs = correlation_estimates(data, clip=clip)
    for index, (i, j) in enumerate(pairs):
        out[i, j] = out[j, i] = float(rho[index])
    return out


def check_invariance(matrices: dict[str, ResponseMatrix]) -> InvarianceResult:
    """Run the configural and metric comparison across language versions.

    Args:
        matrices: language code to :class:`ResponseMatrix`. Every matrix must
            carry the same items; use :meth:`ResponseMatrix.aligned_to` first if
            they do not.

    Raises:
        ValueError: if fewer than two languages are supplied, the item sets
            differ, there are too few items to overidentify a one-factor model,
            there are too many items to form the asymptotic covariance, or a
            language falls below :data:`MIN_RESPONDENTS`.
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
    n_pairs = n_items * (n_items - 1) // 2
    stacked = len(matrices) * n_pairs
    if n_items > MAX_ITEMS_FULL_COVARIANCE or stacked > MAX_STACKED_RESIDUALS:
        raise ValueError(
            f"{n_items} items across {len(matrices)} languages gives {stacked} "
            f"residual correlations, and the corrected statistic needs their "
            f"joint covariance, which is that number squared. The limits are "
            f"{MAX_ITEMS_FULL_COVARIANCE} items and {MAX_STACKED_RESIDUALS} "
            "stacked residuals. Test invariance within a subscale or a subject "
            "rather than across a whole item bank; a single common factor over "
            "hundreds of heterogeneous items is not a model worth fitting "
            "anyway. As a guide, about 20 items across 18 languages fits, as "
            "does 60 items across 2. The item-level DIF analysis has no such "
            "limit and runs on the full bank."
        )

    for lang in langs:
        n = matrices[lang].n_respondents
        if n < MIN_RESPONDENTS:
            raise ValueError(
                f"language {lang!r} has {n} respondents; the weighted statistic "
                f"is not calibrated below {MIN_RESPONDENTS}. A respondent is a "
                "model-by-epoch cell, so raise --epochs or add models. The "
                "item-level DIF analysis still runs and needs far fewer."
            )

    pairs = item_pairs(n_items)
    idx_i, idx_j = _pair_indices(pairs)
    n_groups = len(langs)
    m = len(pairs)

    s_blocks, gamma_blocks, corr_matrices = [], [], []
    for lang in langs:
        data = matrices[lang].data
        rho, a, _ = correlation_estimates(data)
        s_blocks.append(rho)
        gamma_blocks.append(asymptotic_covariance(data, rho, a, pairs))
        full = np.eye(n_items)
        for index, (i, j) in enumerate(pairs):
            full[i, j] = full[j, i] = rho[index]
        corr_matrices.append(full)

    s_stack = np.concatenate(s_blocks)
    gamma = np.zeros((n_groups * m, n_groups * m))
    for g, block in enumerate(gamma_blocks):
        gamma[g * m:(g + 1) * m, g * m:(g + 1) * m] = block

    variances = np.clip(np.diag(gamma), 1e-12, None)
    v_diag = 1.0 / variances
    sd_stack = np.sqrt(variances)

    start_lam = _start_values(corr_matrices, n_items)
    n_total = sum(matrices[lang].n_respondents for lang in langs)

    shared = dict(
        s_stack=s_stack, gamma=gamma, v_diag=v_diag, sd_stack=sd_stack,
        n_items=n_items, n_groups=n_groups, idx_i=idx_i, idx_j=idx_j,
        start_lam=start_lam, n_total=n_total,
    )

    base_stats, _, _ = _fit_stats(
        "baseline", "baseline", naive_df=n_groups * m, baseline=None, **shared
    )
    baseline = (base_stats.chi2, base_stats.df)

    configural, _, c_config = _fit_stats(
        "configural", "configural",
        naive_df=n_groups * m - n_groups * n_items, baseline=baseline, **shared
    )
    metric, _, c_metric = _fit_stats(
        "metric", "metric",
        naive_df=n_groups * m - n_items - (n_groups - 1), baseline=baseline, **shared
    )

    # The difference test uses the naive degrees of freedom, which count
    # restrictions. The adjusted ones are rescaled per model and do not preserve
    # the nesting, so differencing them is meaningless: on a two-group eight-item
    # fit it gives 1 where seven parameters were actually constrained.
    naive_config = n_groups * m - n_groups * n_items
    naive_metric = n_groups * m - n_items - (n_groups - 1)
    chi2_diff, df_diff, p_diff = _scaled_difference(
        configural, metric, c_config, c_metric, naive_config, naive_metric
    )

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
        calibrated=True,
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


def _pair_indices(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([i for i, _ in pairs], dtype=int),
        np.array([j for _, j in pairs], dtype=int),
    )


def _implied_vector(
    params: np.ndarray, model: str, n_items: int, n_groups: int,
    idx_i: np.ndarray, idx_j: np.ndarray,
) -> np.ndarray:
    """Model-implied correlations, stacked group after group.

    Only off-diagonal elements appear. The diagonal of a correlation matrix is 1
    by construction, carries no information about the model, and including it
    would add degrees of freedom that no data support.
    """
    m = len(idx_i)
    out = np.empty(n_groups * m)
    if model == "baseline":
        out[:] = 0.0
        return out
    if model == "configural":
        lam = params.reshape(n_groups, n_items)
        for g in range(n_groups):
            out[g * m:(g + 1) * m] = lam[g][idx_i] * lam[g][idx_j]
        return out
    if model == "metric":
        lam = params[:n_items]
        psi = np.concatenate([[1.0], np.exp(params[n_items:])])
        base = lam[idx_i] * lam[idx_j]
        for g in range(n_groups):
            out[g * m:(g + 1) * m] = psi[g] * base
        return out
    raise ValueError(f"unknown model {model!r}")


def _delta(
    params: np.ndarray, model: str, n_items: int, n_groups: int,
    idx_i: np.ndarray, idx_j: np.ndarray,
) -> np.ndarray:
    """Analytic derivative of the implied vector with respect to the parameters.

    Given in closed form rather than differenced, because it is used twice: by
    the optimiser, and by the test-statistic correction, where a sloppy Jacobian
    would quietly bias the adjusted degrees of freedom.
    """
    m = len(idx_i)
    rows = n_groups * m
    span = np.arange(m)
    if model == "baseline":
        return np.zeros((rows, 0))

    if model == "configural":
        out = np.zeros((rows, n_groups * n_items))
        lam = params.reshape(n_groups, n_items)
        for g in range(n_groups):
            r0, c0 = g * m, g * n_items
            np.add.at(out, (r0 + span, c0 + idx_i), lam[g][idx_j])
            np.add.at(out, (r0 + span, c0 + idx_j), lam[g][idx_i])
        return out

    if model == "metric":
        out = np.zeros((rows, n_items + n_groups - 1))
        lam = params[:n_items]
        psi = np.concatenate([[1.0], np.exp(params[n_items:])])
        base = lam[idx_i] * lam[idx_j]
        for g in range(n_groups):
            r0 = g * m
            np.add.at(out, (r0 + span, idx_i), psi[g] * lam[idx_j])
            np.add.at(out, (r0 + span, idx_j), psi[g] * lam[idx_i])
            if g > 0:
                # Parameterised as log psi, so the derivative carries psi.
                out[r0 + span, n_items + g - 1] = psi[g] * base
        return out

    raise ValueError(f"unknown model {model!r}")


def _start_values(corrs: list[np.ndarray], n_items: int) -> np.ndarray:
    """Loadings from the leading eigenvector, which beats a constant start.

    A one-factor correlation structure is rank one off the diagonal, so the
    dominant eigenvector scaled by the root of its eigenvalue is already close to
    the answer and keeps the optimiser away from the local minimum at zero.
    """
    pooled = np.mean(corrs, axis=0).copy()
    np.fill_diagonal(pooled, 1.0)
    values, vectors = np.linalg.eigh(pooled)
    lead = vectors[:, -1] * math.sqrt(max(float(values[-1]), 1e-6))
    if lead.sum() < 0:
        lead = -lead
    return np.clip(lead, -0.9, 0.9)


def _fit_dwls(
    s_stack: np.ndarray, sd_stack: np.ndarray, model: str, n_items: int,
    n_groups: int, idx_i: np.ndarray, idx_j: np.ndarray, start_lam: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Minimise the diagonally weighted residual sum of squares.

    The weight on each residual is the reciprocal of that correlation's sampling
    variance, so a correlation that is barely estimable counts for little instead
    of dominating the fit. That substitution is the repair.

    Returns ``(params, T, converged)``. Because the weights already carry the
    ``1/N`` from the sampling covariance, ``T`` is on the test-statistic scale
    directly and is not multiplied by the sample size afterwards.
    """
    if model == "baseline":
        resid = s_stack / sd_stack
        return np.zeros(0), float(resid @ resid), True

    if model == "configural":
        p0 = np.tile(start_lam, n_groups)
        lo = np.full(p0.size, -0.995)
        hi = np.full(p0.size, 0.995)
    else:
        p0 = np.concatenate([start_lam, np.zeros(n_groups - 1)])
        lo = np.concatenate([np.full(n_items, -0.995), np.full(n_groups - 1, -5.0)])
        hi = np.concatenate([np.full(n_items, 0.995), np.full(n_groups - 1, 5.0)])

    def residual(params: np.ndarray) -> np.ndarray:
        sigma = _implied_vector(params, model, n_items, n_groups, idx_i, idx_j)
        return (s_stack - sigma) / sd_stack

    def jac(params: np.ndarray) -> np.ndarray:
        return -_delta(params, model, n_items, n_groups, idx_i, idx_j) / sd_stack[:, None]

    best, best_t, ok = p0, np.inf, False
    for scale in (1.0, 0.6, 1.3):
        try:
            res = optimize.least_squares(
                residual, np.clip(p0 * scale, lo, hi), jac=jac,
                bounds=(lo, hi), method="trf", max_nfev=2000,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        t = 2.0 * float(res.cost)
        if t < best_t:
            best, best_t, ok = res.x, t, bool(res.success)
    return best, best_t, ok


def _mean_and_variance_adjusted(
    t_raw: float, gamma: np.ndarray, delta: np.ndarray,
    v_diag: np.ndarray, naive_df: int,
) -> tuple[float, float]:
    """Asparouhov and Muthen's mean-and-variance adjusted statistic.

    The diagonally weighted statistic is not chi-square, because the weight
    matrix is not the inverse of the residuals' covariance. Its first two moments
    are computable though, from the residual projector

        U = V - V D (D' V D)^-1 D' V

    giving ``E[T] = tr(U Gamma)`` and ``Var[T] = 2 tr(U Gamma U Gamma)``. Matching
    both to a chi-square gives the adjusted degrees of freedom and the rescaled
    statistic returned here. This is the step that makes the test calibrated; its
    absence is why the old normal-theory statistic ran fourfold inflated.

    Returns ``(t_star, df_star)``.
    """
    if delta.shape[1] > 0:
        vd = v_diag[:, None] * delta
        bread = delta.T @ vd
        try:
            middle = np.linalg.solve(bread, vd.T)
        except np.linalg.LinAlgError:
            middle = np.linalg.pinv(bread) @ vd.T
        ug = v_diag[:, None] * gamma - vd @ (middle @ gamma)
    else:
        ug = v_diag[:, None] * gamma

    tr1 = float(np.trace(ug))
    tr2 = float(np.trace(ug @ ug))
    if not (math.isfinite(tr1) and math.isfinite(tr2)) or tr1 <= 0 or tr2 <= 0:
        return float("nan"), float(naive_df)

    df_star = tr1 * tr1 / tr2
    t_star = t_raw * df_star / tr1
    return float(t_star), float(df_star)


def _fit_stats(
    label: str, model: str, s_stack: np.ndarray, gamma: np.ndarray,
    v_diag: np.ndarray, sd_stack: np.ndarray, n_items: int, n_groups: int,
    idx_i: np.ndarray, idx_j: np.ndarray, start_lam: np.ndarray,
    naive_df: int, n_total: int, baseline: tuple[float, int] | None,
) -> tuple[FitStats, np.ndarray, float]:
    """Fit one model and return its adjusted statistics.

    Returns ``(stats, params, scaling)`` where ``scaling`` is ``T / T*``, kept so
    a corrected difference test between two nested models can be formed.
    """
    params, t_raw, converged = _fit_dwls(
        s_stack, sd_stack, model, n_items, n_groups, idx_i, idx_j, start_lam
    )
    delta = _delta(params, model, n_items, n_groups, idx_i, idx_j)
    t_star, df_star = _mean_and_variance_adjusted(t_raw, gamma, delta, v_diag, naive_df)

    p = (
        float(stats.chi2.sf(t_star, df_star))
        if df_star > 0 and math.isfinite(t_star)
        else float("nan")
    )
    excess = max(t_star - df_star, 0.0) if math.isfinite(t_star) else float("nan")

    if baseline is None or not math.isfinite(excess):
        cfi = float("nan")
    else:
        base_excess = max(baseline[0] - baseline[1], 0.0)
        cfi = (
            1.0 if base_excess <= 0
            else float(np.clip(1.0 - excess / base_excess, 0.0, 1.0))
        )

    rmsea = (
        math.sqrt(excess / (df_star * max(n_total - n_groups, 1))) * math.sqrt(n_groups)
        if df_star > 0 and math.isfinite(excess)
        else float("nan")
    )
    scaling = (
        t_raw / t_star if math.isfinite(t_star) and t_star > 0 else float("nan")
    )
    return (
        FitStats(label, float(t_star), int(round(df_star)), p, cfi, rmsea, converged),
        params,
        scaling,
    )


def _scaled_difference(
    configural: FitStats,
    metric: FitStats,
    c_config: float,
    c_metric: float,
    naive_config: int,
    naive_metric: int,
) -> tuple[float, int, float]:
    """Satorra and Bentler's scaled difference test for two nested models.

    The adjusted statistics are not differenceable directly: each has been
    rescaled by its own correction factor, so subtracting them mixes two scales.
    The standard repair rescales the raw difference by a factor pooled from the
    two models in proportion to their degrees of freedom.

    Both the pooling and the resulting degrees of freedom use the *naive* degrees
    of freedom, which count restrictions, not the adjusted ones. A pathological
    pooled factor, which can happen when a model is barely identified, returns
    NaN rather than a number that looks usable.
    """
    df_diff = naive_metric - naive_config
    if df_diff <= 0:
        return float("nan"), 0, float("nan")
    if not (math.isfinite(c_config) and math.isfinite(c_metric)):
        return float("nan"), df_diff, float("nan")

    raw_config = configural.chi2 * c_config
    raw_metric = metric.chi2 * c_metric
    pooled = (naive_metric * c_metric - naive_config * c_config) / df_diff
    if pooled <= 0:
        return float("nan"), df_diff, float("nan")

    t_diff = max(raw_metric - raw_config, 0.0) / pooled
    return float(t_diff), int(df_diff), float(stats.chi2.sf(t_diff, df_diff))
