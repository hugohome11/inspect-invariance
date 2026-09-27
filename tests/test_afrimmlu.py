"""Tests for the AfriMMLU task.

The parsing and linkage logic is tested against a synthetic dataset injected in
place of the HTTP fetch, so the suite is hermetic and does not depend on Hugging
Face being reachable. One live test is kept at the end and skips itself when the
network is unavailable, because a pinned revision that has stopped resolving is
exactly the failure this package should notice.
"""

from __future__ import annotations

import os

import pytest

from inspect_invariance import afrimmlu as mod


def _tsv(rows: list[dict[str, object]]) -> str:
    lines = ["question\tchoices\tanswer\tsubject"]
    for row in rows:
        lines.append(
            f"{row['question']}\t{row['choices']!r}\t{row['answer']}\t{row['subject']}"
        )
    return "\n".join(lines) + "\n"


def _corpus(**overrides: list[dict[str, object]]) -> dict[str, str]:
    """Three parallel language versions of a four-item set, plus overrides."""
    base = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": "A", "subject": "maths"},
        {"question": "q2", "choices": ["a", "b", "c", "d"], "answer": "B", "subject": "maths"},
        {"question": "q3", "choices": ["a", "b", "c", "d"], "answer": "C", "subject": "law"},
        {"question": "q4", "choices": ["a", "b", "c", "d"], "answer": "D", "subject": "law"},
    ]
    corpus = {lang: _tsv(base) for lang in ("eng", "zul", "yor")}
    for lang, rows in overrides.items():
        corpus[lang] = _tsv(rows)
    return corpus


@pytest.fixture
def fake_fetch(monkeypatch):
    """Install a synthetic corpus in place of the network fetch."""

    def install(corpus: dict[str, str]):
        def _fetch(language: str, split: str) -> str:
            if language not in corpus:
                raise AssertionError(f"unexpected fetch for {language}")
            return corpus[language]

        monkeypatch.setattr(mod, "_fetch", _fetch)

    return install


def test_samples_are_linked_across_languages(fake_fetch):
    fake_fetch(_corpus())
    samples = mod.afrimmlu_samples(["eng", "zul", "yor"], "test")

    assert len(samples) == 12
    item_ids = {s.metadata["item_id"] for s in samples}
    assert len(item_ids) == 4, "the same four items must appear in every language"
    for item_id in item_ids:
        languages = {s.metadata["language"] for s in samples if s.metadata["item_id"] == item_id}
        assert languages == {"eng", "zul", "yor"}


def test_samples_carry_the_metadata_the_analysis_needs(fake_fetch):
    fake_fetch(_corpus())
    sample = mod.afrimmlu_samples(["eng", "zul"], "test")[0]
    assert set(sample.metadata) >= {"item_id", "language", "benchmark"}
    assert sample.metadata["benchmark"] == "afrimmlu-test"
    assert sample.id.endswith(":eng")


def test_item_ids_are_stable_and_carry_the_subject(fake_fetch):
    fake_fetch(_corpus())
    samples = mod.afrimmlu_samples(["eng", "zul"], "test")
    assert samples[0].metadata["item_id"] == "maths/0000"
    assert samples[2].metadata["item_id"] == "law/0002"


def test_a_shifted_answer_key_is_rejected(fake_fetch):
    """The strongest evidence that two files are not parallel."""
    rows = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": "B", "subject": "maths"},
        {"question": "q2", "choices": ["a", "b", "c", "d"], "answer": "B", "subject": "maths"},
        {"question": "q3", "choices": ["a", "b", "c", "d"], "answer": "C", "subject": "law"},
        {"question": "q4", "choices": ["a", "b", "c", "d"], "answer": "D", "subject": "law"},
    ]
    fake_fetch(_corpus(zul=rows))
    with pytest.raises(ValueError, match="not parallel"):
        mod.afrimmlu_samples(["eng", "zul"], "test")


def test_a_mismatched_subject_is_rejected(fake_fetch):
    rows = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": "A", "subject": "law"},
        {"question": "q2", "choices": ["a", "b", "c", "d"], "answer": "B", "subject": "maths"},
        {"question": "q3", "choices": ["a", "b", "c", "d"], "answer": "C", "subject": "law"},
        {"question": "q4", "choices": ["a", "b", "c", "d"], "answer": "D", "subject": "law"},
    ]
    fake_fetch(_corpus(zul=rows))
    with pytest.raises(ValueError, match="not parallel"):
        mod.afrimmlu_samples(["eng", "zul"], "test")


def test_different_option_counts_are_rejected(fake_fetch):
    rows = [
        {"question": "q1", "choices": ["a", "b", "c"], "answer": "A", "subject": "maths"},
        {"question": "q2", "choices": ["a", "b", "c", "d"], "answer": "B", "subject": "maths"},
        {"question": "q3", "choices": ["a", "b", "c", "d"], "answer": "C", "subject": "law"},
        {"question": "q4", "choices": ["a", "b", "c", "d"], "answer": "D", "subject": "law"},
    ]
    fake_fetch(_corpus(zul=rows))
    with pytest.raises(ValueError, match="one to one"):
        mod.afrimmlu_samples(["eng", "zul"], "test")


def test_a_short_language_version_is_rejected(fake_fetch):
    rows = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": "A", "subject": "maths"},
        {"question": "q2", "choices": ["a", "b", "c", "d"], "answer": "B", "subject": "maths"},
    ]
    fake_fetch(_corpus(zul=rows))
    with pytest.raises(ValueError, match="not parallel"):
        mod.afrimmlu_samples(["eng", "zul"], "test")


def test_a_single_language_is_rejected(fake_fetch):
    fake_fetch(_corpus())
    with pytest.raises(ValueError, match="at least two languages"):
        mod.afrimmlu_samples(["eng"], "test")


def test_duplicate_language_codes_collapse(fake_fetch):
    fake_fetch(_corpus())
    samples = mod.afrimmlu_samples(["eng", "zul", "eng"], "test")
    assert {s.metadata["language"] for s in samples} == {"eng", "zul"}


def test_unknown_language_is_rejected(fake_fetch):
    fake_fetch(_corpus())
    with pytest.raises(ValueError, match="unknown AfriMMLU language"):
        mod.load_split("xxx", "test")


def test_unknown_split_is_rejected(fake_fetch):
    fake_fetch(_corpus())
    with pytest.raises(ValueError, match="unknown split"):
        mod.load_split("eng", "train")


def test_an_unparseable_choices_cell_is_reported_not_skipped(monkeypatch):
    broken = (
        "question\tchoices\tanswer\tsubject\n"
        "q1\tnot-a-list\tA\tmaths\n"
    )
    monkeypatch.setattr(mod, "_fetch", lambda language, split: broken)
    with pytest.raises(ValueError, match="not a list literal"):
        mod.load_split("eng", "test")


def test_limit_applies_to_every_language(fake_fetch):
    fake_fetch(_corpus())
    samples = mod.afrimmlu_samples(["eng", "zul", "yor"], "test", limit=2)
    assert len(samples) == 6
    assert len({s.metadata["item_id"] for s in samples}) == 2


def test_task_builds_with_a_comma_separated_language_string(fake_fetch):
    fake_fetch(_corpus())
    built = mod.afrimmlu(languages="eng,zul", split="test")
    assert len(built.dataset) == 8
    assert built.scorer is not None


def test_task_defaults_to_every_declared_language(fake_fetch):
    fake_fetch({lang: _tsv(
        [
            {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": "A", "subject": "maths"},
            {"question": "q2", "choices": ["a", "b", "c", "d"], "answer": "B", "subject": "maths"},
            {"question": "q3", "choices": ["a", "b", "c", "d"], "answer": "C", "subject": "law"},
            {"question": "q4", "choices": ["a", "b", "c", "d"], "answer": "D", "subject": "law"},
        ]
    ) for lang in mod.AFRIMMLU_LANGUAGES})
    built = mod.afrimmlu()
    assert len(built.dataset) == 4 * len(mod.AFRIMMLU_LANGUAGES)


def test_the_revision_is_a_full_commit_sha():
    """The register requires assets pinned to an immutable revision."""
    assert len(mod.AFRIMMLU_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in mod.AFRIMMLU_REVISION)


@pytest.mark.skipif(
    os.environ.get("INSPECT_INVARIANCE_SKIP_NETWORK") == "1",
    reason="network tests disabled",
)
def test_the_pinned_revision_still_resolves():
    """A live check that the pin is real and the columns have not moved."""
    try:
        items = mod.load_split("eng", "dev")
    except OSError as exc:  # offline, DNS failure, HF outage
        pytest.skip(f"Hugging Face unreachable: {exc}")
    assert len(items) > 0
    first = items[0]
    assert set(first) == {"question", "choices", "answer", "subject"}
    assert isinstance(first["choices"], list) and first["choices"]
