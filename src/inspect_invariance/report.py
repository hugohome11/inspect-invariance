"""Putting the two analyses together into something a reviewer can act on.

The output is deliberately organised around a decision rather than around
statistics: may these language versions be compared, and if not, which items are
responsible. A regulator or a benchmark maintainer needs the item list. The fit
indices are there to justify it, not to be the answer.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace

import numpy as np

from .dif import (
    ETS_A,
    LogisticResult,
    MHResult,
    benjamini_hochberg,
    classify_logistic,
    classify_mh,
    logistic_dif,
    mantel_haenszel,
)
from .invariance import InvarianceResult, check_invariance
from .matrix import ResponseMatrix

__all__ = ["ItemFinding", "LanguageReport", "Analysis", "analyse", "render"]


@dataclass(frozen=True)
class ItemFinding:
    """What both DIF procedures concluded about one item in one language."""

    item: str
    mh: MHResult
    lr: LogisticResult

    @property
    def flagged(self) -> bool:
        return self.mh.flagged or self.lr.flagged

    @property
    def estimable(self) -> bool:
        """Whether either procedure could actually estimate an effect.

        A non-estimable item is not a clean item. Counting the two together is
        the mistake this package exists to catch, so they are kept apart.
        """
        return bool(np.isfinite(self.lr.delta_r2) or np.isfinite(self.mh.delta))

    @property
    def severity(self) -> str:
        """Worst classification across the two procedures."""
        order = {"A": 0, "B": 1, "C": 2}
        worst = max(self.mh.classification, self.lr.classification, key=lambda c: order[c])
        return worst

    @property
    def favours(self) -> str:
        """Which group the item favours, in plain words."""
        if not np.isfinite(self.mh.delta) or abs(self.mh.delta) < 1e-9:
            return "neither"
        return "reference" if self.mh.delta < 0 else "focal"


@dataclass(frozen=True)
class LanguageReport:
    """One focal language compared against the reference language."""

    reference: str
    focal: str
    n_items: int
    findings: tuple[ItemFinding, ...]

    @property
    def flagged(self) -> tuple[ItemFinding, ...]:
        return tuple(f for f in self.findings if f.flagged)

    @property
    def large(self) -> tuple[ItemFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "C")

    @property
    def share_flagged(self) -> float:
        return len(self.flagged) / self.n_items if self.n_items else 0.0

    @property
    def estimable(self) -> tuple[ItemFinding, ...]:
        return tuple(f for f in self.findings if f.estimable)

    @property
    def not_estimable(self) -> tuple[ItemFinding, ...]:
        return tuple(f for f in self.findings if not f.estimable)

    @property
    def underpowered(self) -> bool:
        """True when too little of the item set could be tested to conclude.

        Half is a convention, not a theorem. The point is only that a report
        resting on a minority of its items should not be read as a finding
        about the benchmark.
        """
        return len(self.estimable) < max(2, self.n_items / 2)


@dataclass(frozen=True)
class Analysis:
    """Everything the analysis concluded."""

    reference: str
    invariance: InvarianceResult | None
    languages: tuple[LanguageReport, ...]
    invariance_error: str = ""

    @property
    def underpowered(self) -> bool:
        """True when any language comparison rested on too few testable items."""
        return any(lr.underpowered for lr in self.languages)

    @property
    def verdict(self) -> str:
        """One of ``not-established``, ``not-defensible`` or ``defensible``.

        Three states rather than two, because "we found no broken items" and
        "we could not test the items" are different claims and only one of them
        licenses a cross-language comparison. An under-powered run reports the
        first, and reporting it as the second would be the very error the
        method is meant to expose.

        Evidence of a problem still outranks absence of evidence: a large DIF
        finding is reported even when the rest of the run was thin, since a
        defect that survives a weak design is not made doubtful by the design
        being weak.
        """
        invariance_usable = (
            self.invariance is not None and self.invariance.calibrated
        )

        if invariance_usable and not self.invariance.metric_holds:
            return "not-defensible"
        if any(lr.large for lr in self.languages):
            return "not-defensible"
        if self.underpowered:
            return "not-established"
        if not invariance_usable:
            # Half the analysis did not run, or ran on an uncalibrated
            # statistic. Clean DIF on its own does not establish that the
            # versions share a scale, and reporting "defensible" here would be
            # the same error as treating an under-powered DIF run as a clean
            # one. An uncalibrated invariance result is equally barred from
            # condemning the benchmark: a rejection it cannot support is not
            # evidence either.
            return "not-established"
        return "defensible"

    @property
    def comparable(self) -> bool:
        """Whether cross-language score comparison is defensible.

        Both conditions have to hold. Metric invariance says the forms share a
        scale; the absence of large DIF says no individual item is broken. A
        benchmark can pass the first and fail the second, which is exactly the
        case worth catching, because the aggregate looks fine while specific
        items are mistranslated.

        An under-powered run is not comparable either: see :attr:`verdict`.
        """
        return self.verdict == "defensible"


def analyse(
    matrices: dict[str, ResponseMatrix],
    *,
    reference: str | None = None,
) -> Analysis:
    """Run the full analysis over per-language response matrices.

    Args:
        matrices: language code to :class:`ResponseMatrix`, as produced by
            :func:`inspect_invariance.matrix.from_eval_logs`.
        reference: the language every other is compared against, normally the
            language the benchmark was authored in. Defaults to the first key.

    Raises:
        ValueError: if the reference language is absent, or fewer than two
            languages are supplied.
    """
    if len(matrices) < 2:
        raise ValueError("at least two language versions are needed")

    langs = list(matrices.keys())
    ref = reference or langs[0]
    if ref not in matrices:
        raise ValueError(f"reference language {ref!r} not among {langs}")

    # Every comparison must run on the same items in the same order.
    common = [i for i in matrices[ref].items
              if all(i in m.items for m in matrices.values())]
    if len(common) < 2:
        raise ValueError("language versions share fewer than two items")
    aligned = {l: m.aligned_to(common) for l, m in matrices.items()}

    try:
        inv: InvarianceResult | None = check_invariance(aligned)
        inv_err = ""
    except ValueError as exc:
        # Too few items for a factor model is common on small demo sets, and is
        # not a reason to lose the DIF analysis, which needs only two items.
        inv, inv_err = None, str(exc)

    reports = []
    ref_matrix = aligned[ref]
    for lang, matrix in aligned.items():
        if lang == ref:
            continue
        stacked = np.vstack([ref_matrix.data, matrix.data])
        group = np.concatenate([
            np.zeros(ref_matrix.n_respondents, dtype=int),
            np.ones(matrix.n_respondents, dtype=int),
        ])
        raw = [
            (
                mantel_haenszel(stacked, group, idx, item),
                logistic_dif(stacked, group, idx, item),
            )
            for idx, item in enumerate(common)
        ]

        # Every item of the form is tested against the same reference group, so
        # the per-item p-values are a family and have to be corrected as one.
        # Without this a twenty-item benchmark yields about one "finding" per
        # run from noise alone, and the verdict condemns a clean instrument.
        mh_q = benjamini_hochberg([m.p for m, _ in raw])
        lr_q = benjamini_hochberg([l.p_total for _, l in raw])

        findings = []
        for (mh_res, lr_res), qm, ql in zip(raw, mh_q, lr_q):
            findings.append(
                ItemFinding(
                    item=mh_res.item,
                    mh=replace(
                        mh_res, q=float(qm),
                        classification=classify_mh(mh_res.delta, float(qm)),
                    ),
                    lr=replace(
                        lr_res, q=float(ql),
                        classification=classify_logistic(lr_res.delta_r2, float(ql)),
                    ),
                )
            )
        reports.append(
            LanguageReport(
                reference=ref, focal=lang, n_items=len(common),
                findings=tuple(findings),
            )
        )

    return Analysis(
        reference=ref, invariance=inv, languages=tuple(reports),
        invariance_error=inv_err,
    )


def render(analysis: Analysis) -> str:
    """Render the analysis as plain text."""
    out = io.StringIO()
    w = out.write

    w("MEASUREMENT INVARIANCE REPORT\n")
    w("=" * 78 + "\n\n")
    w(f"Reference language: {analysis.reference}\n")
    focal = ", ".join(lr.focal for lr in analysis.languages)
    w(f"Compared against:   {focal}\n\n")

    w("-" * 78 + "\n")
    w("1. DOES THE BENCHMARK MEASURE THE SAME CONSTRUCT IN EVERY LANGUAGE?\n")
    w("-" * 78 + "\n\n")
    if analysis.invariance is None:
        w(f"  Not tested: {analysis.invariance_error}\n\n")
    else:
        inv = analysis.invariance
        w(f"  {inv.configural}\n")
        w(f"  {inv.metric}\n\n")
        w(f"  scaled difference  chi2={inv.chi2_diff:.2f} on {inv.df_diff} df, "
          f"p={inv.p_diff:.4g}   <- decides metric invariance\n")
        w(f"  change in CFI      {inv.delta_cfi:+.4f}   description only\n")
        w(f"  change in RMSEA    {inv.delta_rmsea:+.4f}   description only\n\n")
        w("  The two deltas are reported because readers expect them, and a large\n"
          "  one is worth a look, but they do not decide the question here. Their\n"
          "  usual cutoffs assume nested models share their degrees of freedom,\n"
          "  which a mean-and-variance adjusted statistic does not: it rescales\n"
          "  each model separately. On correct models with no differential\n"
          "  functioning the delta-RMSEA rule rejected 8 of 30 while the scaled\n"
          "  difference test rejected 0 of 30.\n\n")
        if inv.calibrated:
            w(f"  {inv.verdict}\n\n")
        else:
            w("  NOT A HYPOTHESIS TEST. The indices above are descriptive only.\n"
              "  They come from a normal-theory fit to a tetrachoric correlation\n"
              "  matrix, which is not a sample covariance matrix, so the statistic\n"
              "  is not distributed as they assume. On data with no differential\n"
              "  functioning at all it rejects a correct model on most replications\n"
              "  at any sample size reachable here. Read the numbers as a rough\n"
              "  description of fit, and do not read the comparison as passing or\n"
              "  failing. The item-level analysis below is unaffected and is the\n"
              "  part to act on.\n\n")

    w("-" * 78 + "\n")
    w("2. WHICH ITEMS BEHAVE DIFFERENTLY BETWEEN LANGUAGES?\n")
    w("-" * 78 + "\n\n")

    for lr in analysis.languages:
        w(f"  {lr.reference} against {lr.focal}: "
          f"{len(lr.flagged)} of {lr.n_items} items flagged "
          f"({lr.share_flagged:.0%}), {len(lr.large)} large\n")
        w(f"    {len(lr.estimable)} of {lr.n_items} items were testable"
          f"{'' if not lr.not_estimable else f'; {len(lr.not_estimable)} were not'}\n\n")
        if lr.underpowered:
            w("    TOO FEW TESTABLE ITEMS to conclude anything about this\n"
              "    language version. This is a statement about the design, not\n"
              "    about the benchmark: with this many respondents the procedure\n"
              "    cannot identify an effect either way. Add respondents (more\n"
              "    models, or more epochs) and re-run.\n\n")
        if not lr.flagged:
            if not lr.underpowered:
                w("    No item shows differential functioning beyond the negligible band.\n\n")
            continue
        w(f"    {'item':<14}{'MH delta':>10}{'ETS':>5}{'dR2':>9}{'JG':>4}"
          f"{'form':>13}   favours\n")
        w("    " + "-" * 70 + "\n")
        for f in sorted(lr.flagged, key=lambda x: -abs(_num(x.mh.delta))):
            w(f"    {f.item:<14}{_num(f.mh.delta):>10.2f}{f.mh.classification:>5}"
              f"{f.lr.delta_r2:>9.4f}{f.lr.classification:>4}{f.lr.kind:>13}"
              f"   {f.favours}\n")
        w("\n")

    w("-" * 78 + "\n")
    w("3. VERDICT\n")
    w("-" * 78 + "\n\n")
    if analysis.verdict == "defensible":
        w("  Cross-language comparison is DEFENSIBLE on this evidence. A score\n"
          "  difference between these language versions can be read as a\n"
          "  difference in the model, not in the instrument.\n")
    elif analysis.verdict == "not-defensible":
        w("  Cross-language comparison is NOT DEFENSIBLE as the benchmark stands.\n"
          "  A score difference between these language versions is confounded with\n"
          "  differences in the instrument. Review the flagged items above, and\n"
          "  re-run once they are repaired or removed.\n")
    else:
        w("  NOT ESTABLISHED. This run does not license a conclusion either way.\n"
          "  Too few items could be tested, so the absence of flagged items below\n"
          "  is an absence of evidence and not evidence of comparability. Do not\n"
          "  read it as a clean result. Add respondents and re-run.\n")
    w("\n")
    return out.getvalue()


def _num(x: float) -> float:
    return x if np.isfinite(x) else 0.0
