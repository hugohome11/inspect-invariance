"""Differential item functioning between two language versions of a benchmark.

Two procedures, both standard in operational testing and both implemented here
against their published definitions rather than approximated:

* **Mantel-Haenszel** (Holland and Thayer, 1988), with the ETS delta scale and the
  A/B/C classification used by ETS for operational item review. Detects *uniform*
  DIF only.
* **Logistic regression** (Swaminathan and Rogers, 1990), which additionally
  detects *non-uniform* DIF, with the Nagelkerke ``delta R^2`` effect size and the
  Jodoin and Gierl (2001) classification.

Both condition on an observed matching variable, the total score on the form, so
that the comparison is between reference and focal respondents *of equal
proficiency*. That conditioning is the whole point: an item is not flagged
because one group scores lower on it, but because one group scores lower on it
than their overall performance predicts.

The term "respondent" needs care when the examinee is a model rather than a
person. See ``matrix.py`` for the two designs this package supports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

__all__ = [
    "MHResult",
    "LogisticResult",
    "mantel_haenszel",
    "logistic_dif",
    "ETS_A",
    "ETS_B",
    "ETS_C",
]

ETS_A = "A"  # negligible
ETS_B = "B"  # moderate
ETS_C = "C"  # large

# ETS delta thresholds (Zieky, 1993). Delta is on the ETS scale, where a negative
# value favours the reference group.
_ETS_B_CUT = 1.0
_ETS_C_CUT = 1.5

# Jodoin and Gierl (2001) cutoffs for Nagelkerke delta R^2.
_JG_B_CUT = 0.035
_JG_C_CUT = 0.070


@dataclass(frozen=True)
class MHResult:
    """Mantel-Haenszel DIF statistics for one item."""

    item: str
    alpha: float
    """MH common odds ratio. 1.0 means no uniform DIF."""
    delta: float
    """ETS delta, ``-2.35 * ln(alpha)``. Negative favours the reference group."""
    chi2: float
    """MH chi-square with Holland and Thayer's continuity correction, 1 df."""
    p: float
    classification: str
    """ETS A (negligible), B (moderate) or C (large)."""
    n_reference: int
    n_focal: int
    strata_used: int
    """Score strata that contributed. Strata with no variance carry no information."""

    @property
    def flagged(self) -> bool:
        return self.classification in (ETS_B, ETS_C)


@dataclass(frozen=True)
class LogisticResult:
    """Logistic-regression DIF statistics for one item."""

    item: str
    chi2_total: float
    """M2 against M0, 2 df: any DIF."""
    p_total: float
    chi2_uniform: float
    """M1 against M0, 1 df: uniform DIF."""
    p_uniform: float
    chi2_nonuniform: float
    """M2 against M1, 1 df: non-uniform DIF."""
    p_nonuniform: float
    delta_r2: float
    """Nagelkerke pseudo R-squared increment from M0 to M2."""
    classification: str
    """Jodoin and Gierl A (negligible), B (moderate) or C (large)."""
    converged: bool = True
    note: str = ""

    @property
    def flagged(self) -> bool:
        return self.classification in (ETS_B, ETS_C)

    @property
    def kind(self) -> str:
        """Which form of DIF dominates, for reporting. Only meaningful if flagged.

        Significance alone is not enough to call DIF non-uniform: on a large
        sample the interaction term can clear p < .05 while explaining far less
        than the group main effect. The interaction has to be both significant
        and the larger of the two components before it earns the label.
        """
        if not self.flagged:
            return "none"
        if self.p_nonuniform < 0.05 and self.chi2_nonuniform > self.chi2_uniform:
            return "non-uniform"
        return "uniform"


def mantel_haenszel(
    responses: np.ndarray,
    group: np.ndarray,
    item_index: int,
    item_name: str | None = None,
    *,
    matching: np.ndarray | None = None,
    purify: bool = True,
) -> MHResult:
    """Mantel-Haenszel DIF for a single item.

    Args:
        responses: ``(n_respondents, n_items)`` matrix of 0/1 scored responses.
        group: length ``n_respondents``; 0 marks the reference group, 1 the focal
            group. The reference group is conventionally the source language.
        item_index: column of ``responses`` to test.
        item_name: label for reporting; defaults to the column index.
        matching: optional externally supplied matching variable. Defaults to the
            total score on the form.
        purify: exclude the studied item from its own matching score. This is the
            standard correction; without it the item contaminates the variable it
            is conditioned on, which biases the statistic toward no DIF.

    Returns:
        An :class:`MHResult`.

    Raises:
        ValueError: if the inputs are misshapen or a group is empty.
    """
    responses = np.asarray(responses)
    group = np.asarray(group)
    _validate(responses, group, item_index)

    name = item_name if item_name is not None else f"item_{item_index}"
    scores = _matching_scores(responses, item_index, matching, purify)

    y = responses[:, item_index]
    is_focal = group == 1

    num = 0.0  # sum of A_k * D_k / n_k
    den = 0.0  # sum of B_k * C_k / n_k
    sum_a = 0.0
    sum_ea = 0.0
    sum_va = 0.0
    strata = 0

    for level in np.unique(scores):
        at = scores == level
        n_k = int(at.sum())
        if n_k < 2:
            continue

        ref_at = at & ~is_focal
        foc_at = at & is_focal
        n_ref = int(ref_at.sum())
        n_foc = int(foc_at.sum())
        if n_ref == 0 or n_foc == 0:
            # A stratum containing only one group carries no information about a
            # difference between groups. Dropping it is correct, not a shortcut.
            continue

        a = float(y[ref_at].sum())          # reference, correct
        b = float(n_ref - a)                # reference, incorrect
        c = float(y[foc_at].sum())          # focal, correct
        d = float(n_foc - c)                # focal, incorrect

        n_correct = a + c
        if n_correct == 0 or n_correct == n_k:
            # Everyone in the stratum answered the same way; no information.
            continue

        num += a * d / n_k
        den += b * c / n_k

        # Components of the MH chi-square, under the hypergeometric null.
        sum_a += a
        sum_ea += n_ref * n_correct / n_k
        sum_va += (n_ref * n_foc * n_correct * (n_k - n_correct)) / (n_k * n_k * (n_k - 1))
        strata += 1

    if strata == 0 or den == 0.0 or num == 0.0:
        return MHResult(
            item=name,
            alpha=float("nan"),
            delta=float("nan"),
            chi2=float("nan"),
            p=float("nan"),
            classification=ETS_A,
            n_reference=int((~is_focal).sum()),
            n_focal=int(is_focal.sum()),
            strata_used=strata,
        )

    alpha = num / den
    delta = -2.35 * math.log(alpha)

    if sum_va > 0:
        chi2 = (abs(sum_a - sum_ea) - 0.5) ** 2 / sum_va
        p = float(stats.chi2.sf(chi2, 1))
    else:
        chi2, p = float("nan"), float("nan")

    classification = _ets_class(delta, p)

    return MHResult(
        item=name,
        alpha=float(alpha),
        delta=float(delta),
        chi2=float(chi2),
        p=p,
        classification=classification,
        n_reference=int((~is_focal).sum()),
        n_focal=int(is_focal.sum()),
        strata_used=strata,
    )


def logistic_dif(
    responses: np.ndarray,
    group: np.ndarray,
    item_index: int,
    item_name: str | None = None,
    *,
    matching: np.ndarray | None = None,
    purify: bool = True,
) -> LogisticResult:
    """Logistic-regression DIF for a single item, uniform and non-uniform.

    Fits three nested models of the probability of a correct response:

    * ``M0: logit(p) = b0 + b1 * score``
    * ``M1: logit(p) = b0 + b1 * score + b2 * group``
    * ``M2: logit(p) = b0 + b1 * score + b2 * group + b3 * score * group``

    ``M1`` against ``M0`` tests uniform DIF, ``M2`` against ``M1`` non-uniform DIF,
    and ``M2`` against ``M0`` tests either, on 2 degrees of freedom. The effect
    size is the Nagelkerke pseudo R-squared increment from ``M0`` to ``M2``.

    Arguments are as for :func:`mantel_haenszel`.
    """
    responses = np.asarray(responses)
    group = np.asarray(group)
    _validate(responses, group, item_index)

    name = item_name if item_name is not None else f"item_{item_index}"
    scores = _matching_scores(responses, item_index, matching, purify).astype(float)
    y = responses[:, item_index].astype(float)
    g = (group == 1).astype(float)

    n = len(y)
    if y.sum() in (0.0, float(n)):
        # A constant item cannot show DIF; every model fits it identically.
        return LogisticResult(
            item=name, chi2_total=0.0, p_total=1.0, chi2_uniform=0.0, p_uniform=1.0,
            chi2_nonuniform=0.0, p_nonuniform=1.0, delta_r2=0.0,
            classification=ETS_A, converged=True,
            note="item is constant; no DIF is estimable",
        )

    ones = np.ones(n)
    x0 = np.column_stack([ones, scores])
    x1 = np.column_stack([ones, scores, g])
    x2 = np.column_stack([ones, scores, g, scores * g])

    ll0, ok0 = _fit_logistic(x0, y)
    ll1, ok1 = _fit_logistic(x1, y)
    ll2, ok2 = _fit_logistic(x2, y)
    converged = ok0 and ok1 and ok2

    chi2_total = max(0.0, 2.0 * (ll2 - ll0))
    chi2_uniform = max(0.0, 2.0 * (ll1 - ll0))
    chi2_nonuniform = max(0.0, 2.0 * (ll2 - ll1))

    p_total = float(stats.chi2.sf(chi2_total, 2))
    p_uniform = float(stats.chi2.sf(chi2_uniform, 1))
    p_nonuniform = float(stats.chi2.sf(chi2_nonuniform, 1))

    delta_r2 = _nagelkerke(ll0, ll2, n)
    classification = _jodoin_gierl(delta_r2, p_total)

    return LogisticResult(
        item=name,
        chi2_total=chi2_total, p_total=p_total,
        chi2_uniform=chi2_uniform, p_uniform=p_uniform,
        chi2_nonuniform=chi2_nonuniform, p_nonuniform=p_nonuniform,
        delta_r2=delta_r2,
        classification=classification,
        converged=converged,
        note="" if converged else "one or more models did not converge",
    )


# ---------------------------------------------------------------- internals


def _validate(responses: np.ndarray, group: np.ndarray, item_index: int) -> None:
    if responses.ndim != 2:
        raise ValueError(f"responses must be 2-D, got shape {responses.shape}")
    if len(group) != responses.shape[0]:
        raise ValueError(
            f"group has {len(group)} entries but responses has "
            f"{responses.shape[0]} rows"
        )
    if not 0 <= item_index < responses.shape[1]:
        raise ValueError(
            f"item_index {item_index} out of range for {responses.shape[1]} items"
        )
    uniq = set(np.unique(group).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"group must contain only 0 and 1, got {sorted(uniq)}")
    if 0 not in uniq or 1 not in uniq:
        raise ValueError("both a reference (0) and a focal (1) group are required")
    valid = set(np.unique(responses).tolist())
    if not valid <= {0, 1}:
        raise ValueError("responses must be 0/1 scored; polytomous items are not supported")


def _matching_scores(
    responses: np.ndarray,
    item_index: int,
    matching: np.ndarray | None,
    purify: bool,
) -> np.ndarray:
    if matching is not None:
        m = np.asarray(matching)
        if len(m) != responses.shape[0]:
            raise ValueError("matching variable length does not match responses")
        return m
    total = responses.sum(axis=1)
    if purify:
        return total - responses[:, item_index]
    return total


def _ets_class(delta: float, p: float) -> str:
    if not math.isfinite(delta) or not math.isfinite(p):
        return ETS_A
    significant = p < 0.05
    magnitude = abs(delta)
    if not significant or magnitude < _ETS_B_CUT:
        return ETS_A
    if magnitude < _ETS_C_CUT:
        return ETS_B
    return ETS_C


def _jodoin_gierl(delta_r2: float, p_total: float) -> str:
    if not math.isfinite(delta_r2) or not math.isfinite(p_total):
        return ETS_A
    if p_total >= 0.05 or delta_r2 < _JG_B_CUT:
        return ETS_A
    if delta_r2 < _JG_C_CUT:
        return ETS_B
    return ETS_C


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """Maximum-likelihood logistic fit, returning the log-likelihood.

    A tiny ridge penalty keeps the fit finite under complete separation, which is
    common with short forms and small respondent counts. The penalty is small
    enough not to move the likelihood-ratio tests materially, and its presence is
    reported rather than hidden.
    """
    ridge = 1e-6

    def neg_ll(beta: np.ndarray) -> float:
        eta = np.clip(x @ beta, -35.0, 35.0)
        ll = np.sum(y * eta - np.logaddexp(0.0, eta))
        return -(ll - ridge * float(beta @ beta))

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        return -(x.T @ (y - p) - 2.0 * ridge * beta)

    beta0 = np.zeros(x.shape[1])
    res = optimize.minimize(neg_ll, beta0, jac=grad, method="BFGS")
    eta = np.clip(x @ res.x, -35.0, 35.0)
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    return ll, bool(res.success)


def _nagelkerke(ll_null: float, ll_full: float, n: int) -> float:
    """Nagelkerke pseudo R-squared increment between two nested fits."""
    cox_snell = 1.0 - math.exp(-2.0 * (ll_full - ll_null) / n)
    denom = 1.0 - math.exp(2.0 * ll_null / n)
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, cox_snell / denom))
