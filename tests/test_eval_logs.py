"""Reading Inspect eval logs.

The fixture below mirrors the structure of a real Inspect JSON log, verified
against output from ``inspect eval ... --log-format json`` on inspect_ai 0.3.261:
``eval.model`` identifies the model, and each entry in ``samples`` carries
``epoch``, ``metadata`` (with the ``item_id`` and ``language`` the task writes)
and ``scores``, whose scorer entries hold ``value`` of ``"C"`` or ``"I"``.

This path is worth its own tests because it is the seam between two systems, and
a seam that is only exercised by hand is a seam that breaks quietly.
"""

from __future__ import annotations

import json

import pytest

from inspect_invariance.matrix import from_eval_logs


def log(model: str, language: str, outcomes: dict[str, str], epoch: int = 1) -> dict:
    return {
        "version": 2,
        "status": "success",
        "eval": {"model": model, "task": f"demo-{language}"},
        "samples": [
            {
                "id": f"{item}:{language}",
                "epoch": epoch,
                "metadata": {
                    "item_id": item,
                    "language": language,
                    "benchmark": "demo",
                },
                "scores": {"choice": {"value": value, "answer": "", "explanation": ""}},
            }
            for item, value in outcomes.items()
        ],
    }


def write(tmp_path, name: str, payload: dict):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


ITEMS_A = {"q001": "C", "q002": "I", "q003": "C"}
ITEMS_B = {"q001": "C", "q002": "C", "q003": "I"}


def test_reads_two_languages_from_a_directory(tmp_path):
    write(tmp_path, "en.json", log("openai/gpt-4o", "en", ITEMS_A))
    write(tmp_path, "hr.json", log("openai/gpt-4o", "hr", ITEMS_B))

    matrices = from_eval_logs([tmp_path], design="models")

    assert sorted(matrices) == ["en", "hr"]
    assert matrices["en"].items == ("q001", "q002", "q003")
    assert matrices["en"].data.tolist() == [[1, 0, 1]]
    assert matrices["hr"].data.tolist() == [[1, 1, 0]]


def test_epochs_design_makes_each_run_a_respondent(tmp_path):
    write(tmp_path, "en1.json", log("openai/gpt-4o", "en", ITEMS_A, epoch=1))
    write(tmp_path, "en2.json", log("openai/gpt-4o", "en", ITEMS_B, epoch=2))
    write(tmp_path, "hr1.json", log("openai/gpt-4o", "hr", ITEMS_A, epoch=1))
    write(tmp_path, "hr2.json", log("openai/gpt-4o", "hr", ITEMS_B, epoch=2))

    matrices = from_eval_logs([tmp_path], design="epochs")

    assert matrices["en"].n_respondents == 2
    assert matrices["en"].respondents == ("openai/gpt-4o#1", "openai/gpt-4o#2")


def test_models_design_makes_each_model_a_respondent(tmp_path):
    write(tmp_path, "a.json", log("openai/gpt-4o", "en", ITEMS_A))
    write(tmp_path, "b.json", log("anthropic/claude", "en", ITEMS_B))
    write(tmp_path, "c.json", log("openai/gpt-4o", "hr", ITEMS_A))
    write(tmp_path, "d.json", log("anthropic/claude", "hr", ITEMS_B))

    matrices = from_eval_logs([tmp_path], design="models")

    assert matrices["en"].n_respondents == 2
    assert set(matrices["en"].respondents) == {"openai/gpt-4o", "anthropic/claude"}


def test_samples_without_the_linking_metadata_are_skipped(tmp_path):
    payload = log("openai/gpt-4o", "en", ITEMS_A)
    payload["samples"].append(
        {"id": "stray", "epoch": 1, "metadata": {}, "scores": {"choice": {"value": "C"}}}
    )
    write(tmp_path, "en.json", payload)
    write(tmp_path, "hr.json", log("openai/gpt-4o", "hr", ITEMS_B))

    matrices = from_eval_logs([tmp_path], design="models")
    assert matrices["en"].n_items == 3, "the unlinkable sample must not become an item"


def test_numeric_and_boolean_scores_are_accepted(tmp_path):
    payload = log("openai/gpt-4o", "en", ITEMS_A)
    payload["samples"][0]["scores"] = {"choice": {"value": 1.0}}
    payload["samples"][1]["scores"] = {"choice": {"value": False}}
    write(tmp_path, "en.json", payload)
    write(tmp_path, "hr.json", log("openai/gpt-4o", "hr", ITEMS_B))

    matrices = from_eval_logs([tmp_path], design="models")
    assert matrices["en"].data.tolist() == [[1, 0, 1]]


def test_logs_with_nothing_usable_are_reported_clearly(tmp_path):
    write(tmp_path, "empty.json", {"eval": {"model": "m"}, "samples": []})
    with pytest.raises(ValueError, match="no usable samples"):
        from_eval_logs([tmp_path])


def test_explicit_file_paths_work_as_well_as_directories(tmp_path):
    a = write(tmp_path, "en.json", log("openai/gpt-4o", "en", ITEMS_A))
    b = write(tmp_path, "hr.json", log("openai/gpt-4o", "hr", ITEMS_B))
    matrices = from_eval_logs([a, b], design="models")
    assert sorted(matrices) == ["en", "hr"]
