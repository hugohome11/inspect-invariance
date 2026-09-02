"""Measurement invariance and DIF testing for multilingual LLM benchmarks.

Translating a benchmark does not preserve what it measures. This package tests
whether it did, using the procedures that operational testing uses for exactly
this problem, and reports which items broke.
"""

from .dif import LogisticResult, MHResult, logistic_dif, mantel_haenszel
from .invariance import InvarianceResult, check_invariance, tetrachoric_matrix
from .matrix import ResponseMatrix, from_eval_logs, from_records
from .report import Analysis, ItemFinding, LanguageReport, analyse, render

__version__ = "0.1.0"

__all__ = [
    "mantel_haenszel", "logistic_dif", "MHResult", "LogisticResult",
    "check_invariance", "tetrachoric_matrix", "InvarianceResult",
    "ResponseMatrix", "from_records", "from_eval_logs",
    "analyse", "render", "Analysis", "ItemFinding", "LanguageReport",
    "__version__",
]
