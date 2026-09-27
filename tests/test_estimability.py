"""Tests for the guards that stop the package claiming more than it can.

Three separate failures are covered, all of them ones this package existed to
catch in other people's benchmarks and was committing in its own code:

1. a logistic fit that separates completely reports a near-perfect effect size;
2. a design with too few events per parameter does the same more quietly;
3. testing every item without controlling false discovery condemns a clean
   benchmark about once per run.

Together these produced confident verdicts from designs that could not support
any verdict at all, in either direction.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from inspect_invariance.dif import benjamini_hochberg, logistic_dif
from inspect_invariance.matrix import ResponseMatrix
from inspect_invariance.report import analyse


# ------------------------------------------------------------------ BH


def test_bh_matches_worked_example():
    q = benjamini_hochberg([0.001, 0.002, 0.003])
    assert np.allclose(q, [0.003, 0.003, 0.003])


def test_bh_scales_the_smallest_p_by_the_number_of_tests():
    q = benjamini_hochberg([0.02, 0.5, 0.6, 0.9])
    assert q[0] == pytest.approx(0.08)


def test_bh_is_monotone_and_bounded():
    p = list(np.linspace(0.0001, 0.99, 40))
    q = benjamini_hochberg(p)
    assert np.all(q <= 1.0) and np.all(q >= 0.0)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_bh_never_lowers_a_p_value():
    p = [0.01, 0.2, 0.4]
    assert np.all(benjamini_hochberg(p) >= np.array(p) - 1e-12)


def test_bh_ignores_non_estimable_items():
    q = benjamini_hochberg([0.01, float("nan"), 0.9])
    assert math.isnan(q[1])
    # Two real tests, not three, so the smallest is scaled by two.
    assert q[0] == pytest.approx(0.02)


def test_bh_on_nothing_estimable_returns_all_nan():
    q = benjamini_hochberg([float("nan"), float("nan")])
    assert len(q) == 2 and all(math.isnan(x) for x in q)


# ------------------------------------------------- separation and power


def test_complete_separation_is_reported_not_scored():
    """The reference group gets every item right, the focal group every one wrong."""
    responses = np.array([[1, 1, 1, 1]] * 4 + [[0, 0, 0, 0]] * 4)
    group = np.array([0] * 4 + [1] * 4)
    result = logistic_dif(responses, group, 0)
    assert math.isnan(result.delta_r2), "a separated fit has no estimable effect size"
    assert result.classification == "A"
    assert not result.converged
    assert "estimable" in result.note


def test_a_tiny_design_yields_no_effect_size():
    rng = random.Random(3)
    responses = np.array([[rng.randint(0, 1) for _ in range(6)] for _ in range(6)])
    group = np.array([0, 0, 0, 1, 1, 1])
    result = logistic_dif(responses, group, 0)
    assert math.isnan(result.delta_r2)
    assert "too few respondents" in result.note


def test_a_well_powered_design_still_produces_an_effect_size():
    """The guards must not silence a design that can support a conclusion."""
    matrices = _simulate(n_resp=80, shift=2.0, seed=1)
    report = analyse(matrices, reference="eng")
    focal = [lr for lr in report.languages if lr.focal == "yor"][0]
    assert not focal.underpowered
    assert any(np.isfinite(f.lr.delta_r2) for f in focal.findings)


# ------------------------------------------------------------ verdicts


def test_an_underpowered_run_is_not_established_rather_than_defensible():
    """The failure that matters most: silence must not read as a clean result."""
    report = analyse(_simulate(n_resp=3, shift=0.0, seed=5), reference="eng")
    assert report.verdict == "not-established"
    assert not report.comparable
    assert report.underpowered


def test_an_underpowered_run_says_so_in_the_report():
    from inspect_invariance.report import render

    text = render(analyse(_simulate(n_resp=3, shift=0.0, seed=5), reference="eng"))
    assert "NOT ESTABLISHED" in text
    assert "absence of evidence" in text
    assert "No item shows differential functioning" not in text


def test_non_estimable_items_are_counted_apart_from_clean_ones():
    report = analyse(_simulate(n_resp=3, shift=0.0, seed=5), reference="eng")
    for lr in report.languages:
        assert len(lr.estimable) + len(lr.not_estimable) == lr.n_items
        assert len(lr.not_estimable) > 0


def test_planted_dif_still_produces_a_not_defensible_verdict():
    report = analyse(_simulate(n_resp=80, shift=2.5, seed=2), reference="eng")
    assert report.verdict == "not-defensible"


def test_a_clean_benchmark_produces_no_large_dif_flags():
    """Without false-discovery control this flagged items on most seeds.

    Scoped to the DIF layer deliberately. The overall verdict is still wrong on
    this data because the invariance layer rejects a correctly specified model;
    that defect is pinned by ``test_invariance_does_not_reject_a_perfect_model``
    below and is not what this test is about.
    """
    flagged = 0
    for seed in range(8):
        report = analyse(_simulate(n_resp=80, shift=0.0, seed=seed), reference="eng")
        flagged += sum(len(lr.large) for lr in report.languages)
    assert flagged <= 2, f"{flagged} large DIF flags across 8 null runs of 40 tests"


def test_an_uncalibrated_invariance_result_cannot_license_a_comparison():
    """Descriptive fit indices must not drive the verdict in either direction."""
    report = analyse(_simulate(n_resp=80, shift=0.0, seed=0), reference="eng")
    assert report.verdict != "defensible"
    if report.invariance is not None:
        assert not report.invariance.calibrated


def test_the_invariance_section_says_it_is_not_a_test():
    from inspect_invariance.invariance import check_invariance
    from inspect_invariance.report import render

    matrices = _simulate(n_resp=80, shift=0.0, seed=0)
    text = render(analyse(matrices, reference="eng"))
    assert ("NOT A HYPOTHESIS TEST" in text) or ("Not tested:" in text)


def test_an_indefinite_correlation_matrix_is_now_tolerated():
    """The old fit needed log|S| and S inverse, so an indefinite matrix was fatal.

    Diagonally weighted least squares weights only the residual vector, so it
    needs neither. A sample matrix with a negative eigenvalue, which is the norm
    when items outnumber respondents, is now merely imprecise rather than
    unusable, and the weights say so.
    """
    from inspect_invariance.invariance import check_invariance, tetrachoric_matrix

    matrices = _simulate(n_resp=120, shift=0.0, seed=4, n_items=20)
    smallest = min(
        float(np.linalg.eigvalsh(tetrachoric_matrix(m.data)).min())
        for m in matrices.values()
    )
    assert smallest < 0, "this fixture is meant to be indefinite"

    result = check_invariance(matrices)
    assert result.calibrated
    assert np.isfinite(result.configural.chi2)


def test_the_respondent_floor_still_refuses():
    from inspect_invariance.invariance import MIN_RESPONDENTS, check_invariance

    with pytest.raises(ValueError, match="not calibrated below"):
        check_invariance(_simulate(n_resp=MIN_RESPONDENTS - 1, shift=0.0, seed=1))


def test_invariance_is_calibrated_and_accepts_a_perfect_model():
    """Was xfail under the old estimator. Passes since DWLS landed.

    This is the test the whole repair exists to satisfy: data generated from
    exactly the fitted model must not be rejected.
    """
    from inspect_invariance.invariance import check_invariance

    result = check_invariance(_simulate(n_resp=800, shift=0.0, seed=0, n_items=20))
    assert result.calibrated
    assert result.configural_holds and result.metric_holds


def test_findings_carry_the_adjusted_p():
    report = analyse(_simulate(n_resp=80, shift=0.0, seed=0), reference="eng")
    finding = report.languages[0].findings[0]
    assert math.isnan(finding.mh.q) or 0.0 <= finding.mh.q <= 1.0
    assert math.isnan(finding.lr.q) or 0.0 <= finding.lr.q <= 1.0


# ------------------------------------------------------------- helpers


def _simulate(n_resp: int, shift: float, seed: int, n_items: int = 20):
    """Three language versions under a 1PL, with DIF planted in two yor items."""
    rng = random.Random(seed)
    items = tuple(f"it{i:03d}" for i in range(n_items))
    difficulty = [rng.uniform(-1.2, 1.2) for _ in range(n_items)]
    matrices = {}
    for language, ability in {"eng": 0.5, "zul": 0.3, "yor": 0.4}.items():
        rows = []
        for _ in range(n_resp):
            theta = ability + rng.gauss(0, 0.9)
            rows.append(
                [
                    1
                    if rng.random()
                    < 1
                    / (
                        1
                        + math.exp(
                            -(
                                theta
                                - (
                                    difficulty[i]
                                    + (shift if (i < 2 and language == "yor") else 0.0)
                                )
                            )
                        )
                    )
                    else 0
                    for i in range(n_items)
                ]
            )
        matrices[language] = ResponseMatrix(
            language,
            items,
            tuple(f"r{i}" for i in range(n_resp)),
            np.array(rows, dtype=int),
            "models",
        )
    return matrices
