"""Putting the two analyses together into something a reviewer can act on.

The output is deliberately organised around a decision rather than around
statistics: may these language versions be compared, and if not, which items are
responsible. A regulator or a benchmark maintainer needs the item list. The fit
indices are there to justify it, not to be the answer.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from .dif import ETS_A, LogisticResult, MHResult, logistic_dif, mantel_haenszel
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


@dataclass(frozen=True)
class Analysis:
    """Everything the analysis concluded."""

    reference: str
    invariance: InvarianceResult | None
    languages: tuple[LanguageReport, ...]
    invariance_error: str = ""

    @property
    def comparable(self) -> bool:
        """Whether cross-language score comparison is defensible.

        Both conditions have to hold. Metric invariance says the forms share a
        scale; the absence of large DIF says no individual item is broken. A
        benchmark can pass the first and fail the second, which is exactly the
        case worth catching, because the aggregate looks fine while specific
        items are mistranslated.
        """
        if self.invariance is not None and not self.invariance.metric_holds:
            return False
        return all(not lr.large for lr in self.languages)


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
        findings = []
        for idx, item in enumerate(common):
            findings.append(
                ItemFinding(
                    item=item,
                    mh=mantel_haenszel(stacked, group, idx, item),
                    lr=logistic_dif(stacked, group, idx, item),
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
        w(f"  change in CFI    {inv.delta_cfi:+.4f}   (invariance is tenable within "
          f"{'+/-0.010' })\n")
        w(f"  change in RMSEA  {inv.delta_rmsea:+.4f}   (invariance is tenable within "
          f"{'+/-0.015' })\n")
        w(f"  chi-square diff  {inv.chi2_diff:.2f} on {inv.df_diff} df, p = {inv.p_diff:.4g}\n\n")
        w(f"  {inv.verdict}\n\n")

    w("-" * 78 + "\n")
    w("2. WHICH ITEMS BEHAVE DIFFERENTLY BETWEEN LANGUAGES?\n")
    w("-" * 78 + "\n\n")

    for lr in analysis.languages:
        w(f"  {lr.reference} against {lr.focal}: "
          f"{len(lr.flagged)} of {lr.n_items} items flagged "
          f"({lr.share_flagged:.0%}), {len(lr.large)} large\n\n")
        if not lr.flagged:
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
    if analysis.comparable:
        w("  Cross-language comparison is DEFENSIBLE on this evidence. A score\n"
          "  difference between these language versions can be read as a\n"
          "  difference in the model, not in the instrument.\n")
    else:
        w("  Cross-language comparison is NOT DEFENSIBLE as the benchmark stands.\n"
          "  A score difference between these language versions is confounded with\n"
          "  differences in the instrument. Review the flagged items above, and\n"
          "  re-run once they are repaired or removed.\n")
    w("\n")
    return out.getvalue()


def _num(x: float) -> float:
    return x if np.isfinite(x) else 0.0
