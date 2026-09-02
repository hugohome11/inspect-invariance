"""Matrix construction, benchmark loading, and the end-to-end report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from inspect_invariance.matrix import ResponseMatrix, from_records
from inspect_invariance.report import analyse, render
from inspect_invariance.task import (
    DEMO_BENCHMARK,
    load_benchmark,
    multilingual_task,
    samples_for_language,
)


def records(n_resp=40, n_items=10, break_item=3, seed=5):
    """Flat records for two languages, with one item broken in the focal language."""
    rng = np.random.default_rng(seed)
    out = []
    for lang in ("en", "hr"):
        for r in range(n_resp):
            ability = rng.normal()
            for i in range(n_items):
                eta = ability - (i - n_items / 2) * 0.3
                if lang == "hr" and i == break_item:
                    eta -= 2.0
                p = 1 / (1 + np.exp(-eta))
                out.append({
                    "language": lang,
                    "item_id": f"q{i:03d}",
                    "respondent": f"model#{r}",
                    "correct": int(rng.random() < p),
                })
    return out


def test_from_records_builds_aligned_matrices():
    matrices = from_records(records(), design="models")
    assert set(matrices) == {"en", "hr"}
    assert matrices["en"].n_items == 10
    assert matrices["en"].n_respondents == 40
    assert matrices["en"].items == matrices["hr"].items


def test_missing_responses_are_refused_not_imputed():
    recs = records()
    recs.pop(0)
    with pytest.raises(ValueError, match="missing responses"):
        from_records(recs)


def test_records_missing_a_field_are_reported_clearly():
    with pytest.raises(ValueError, match="missing required field"):
        from_records([{"language": "en", "item_id": "q1", "correct": 1}])


def test_aligned_to_reorders_and_drops():
    matrices = from_records(records(), design="models")
    wanted = ["q005", "q001", "not_an_item"]
    aligned = matrices["en"].aligned_to(wanted)
    assert aligned.items == ("q005", "q001")
    assert aligned.data.shape == (40, 2)


def test_end_to_end_detects_the_broken_item():
    matrices = from_records(records(n_resp=120, break_item=3), design="models")
    analysis = analyse(matrices, reference="en")

    assert len(analysis.languages) == 1
    report = analysis.languages[0]
    assert report.reference == "en" and report.focal == "hr"

    flagged = {f.item for f in report.flagged}
    assert "q003" in flagged, f"the broken item should be flagged, got {flagged}"
    assert not analysis.comparable


def test_render_produces_a_verdict():
    matrices = from_records(records(n_resp=120), design="models")
    text = render(analyse(matrices, reference="en"))
    assert "MEASUREMENT INVARIANCE REPORT" in text
    assert "VERDICT" in text
    assert "q003" in text


def test_unknown_reference_language_is_rejected():
    matrices = from_records(records(), design="models")
    with pytest.raises(ValueError, match="reference language"):
        analyse(matrices, reference="de")


# ---------------------------------------------------------------- benchmark


def test_demo_benchmark_loads_and_is_internally_consistent():
    spec = load_benchmark(DEMO_BENCHMARK)
    assert spec["languages"] == ["en", "hr", "de"]
    assert len(spec["items"]) == 12
    ids = [i["id"] for i in spec["items"]]
    assert len(set(ids)) == len(ids)


def test_demo_benchmark_keeps_its_positive_control():
    """q007 carries a deliberately degraded Croatian stem. The README documents
    it, so a silent repair would make the README wrong."""
    spec = load_benchmark(DEMO_BENCHMARK)
    q007 = next(i for i in spec["items"] if i["id"] == "q007")
    assert "positive_control" in q007
    assert "može biti istinita" in q007["translations"]["hr"]["question"]
    assert "must be true" in q007["translations"]["en"]["question"]


def test_samples_carry_the_metadata_the_analysis_needs():
    spec = load_benchmark(DEMO_BENCHMARK)
    samples = samples_for_language(spec, "hr")
    assert len(samples) == 12
    for s in samples:
        assert s.metadata["language"] == "hr"
        assert s.metadata["item_id"]
        assert s.choices


def test_task_builds_for_every_declared_language():
    for lang in ("en", "hr", "de"):
        t = multilingual_task(language=lang)
        assert t.name == f"demo-reasoning-{lang}"
        assert len(t.dataset) == 12


def test_unknown_language_is_rejected():
    spec = load_benchmark(DEMO_BENCHMARK)
    with pytest.raises(ValueError, match="not in benchmark"):
        samples_for_language(spec, "fr")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s["items"][0].pop("id"), "no id"),
        (lambda s: s["items"][0]["translations"].pop("hr"), "no translation"),
        (lambda s: s["items"][0]["translations"]["hr"]["choices"].pop(), "choice counts"),
    ],
)
def test_broken_benchmarks_are_rejected(tmp_path, mutate, message):
    spec = json.loads(DEMO_BENCHMARK.read_text(encoding="utf-8"))
    mutate(spec)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_benchmark(path)
