"""Inspect tasks that run one parallel item set across several languages.

The design constraint that makes the analysis possible is linkage: the same item
must be identifiable in every language. So a benchmark is defined once, as a set
of items each carrying its translations, and one task is generated per language
from that single definition. Every sample then carries ``item_id`` and
``language`` in its metadata, which is what
:func:`inspect_invariance.matrix.from_eval_logs` reads back.

Run every language version::

    inspect eval inspect_invariance/task.py --model openai/gpt-4o --epochs 20

or one at a time::

    inspect eval inspect_invariance/task.py@english --model openai/gpt-4o

Then analyse the logs::

    inspect-invariance analyse ./logs --reference en
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

__all__ = ["load_benchmark", "samples_for_language", "multilingual_task", "DEMO_BENCHMARK"]

DEMO_BENCHMARK = Path(__file__).resolve().parent / "data" / "demo" / "benchmark.json"
"""Path to the bundled demo benchmark.

It lives inside the package rather than beside it so that it ships in the wheel.
Resolving it from the repository root worked from a checkout and silently broke
under ``pip install``, where only ``src/inspect_invariance/`` is packaged.
"""


def load_benchmark(path: str | Path) -> dict[str, Any]:
    """Load and validate a multilingual benchmark definition.

    The file is JSON with the shape::

        {
          "name": "demo-reasoning",
          "reference_language": "en",
          "languages": ["en", "hr", "de"],
          "items": [
            {
              "id": "q001",
              "target": "B",
              "translations": {
                "en": {"question": "...", "choices": ["...", "...", "..."]},
                "hr": {"question": "...", "choices": ["...", "...", "..."]}
              }
            }
          ]
        }

    Raises:
        ValueError: if an item is missing a language, or its choice count differs
            between languages. Both would break the item linkage the whole
            analysis depends on, and both are easy to introduce by hand, so they
            are checked rather than assumed.
    """
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))

    for key in ("languages", "items"):
        if key not in spec:
            raise ValueError(f"benchmark {path.name} is missing required key {key!r}")

    languages = list(spec["languages"])
    if len(languages) < 2:
        raise ValueError("a multilingual benchmark needs at least two languages")

    seen: set[str] = set()
    for item in spec["items"]:
        item_id = item.get("id")
        if not item_id:
            raise ValueError(f"an item in {path.name} has no id")
        if item_id in seen:
            raise ValueError(f"duplicate item id {item_id!r}")
        seen.add(item_id)

        translations = item.get("translations") or {}
        missing = [l for l in languages if l not in translations]
        if missing:
            raise ValueError(f"item {item_id!r} has no translation for {missing}")

        counts = {l: len(translations[l].get("choices") or []) for l in languages}
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"item {item_id!r} has different choice counts across languages: "
                f"{counts}; the options must correspond one to one"
            )

    spec.setdefault("name", path.stem)
    spec.setdefault("reference_language", languages[0])
    return spec


def samples_for_language(spec: dict[str, Any], language: str) -> list[Sample]:
    """Build the Inspect samples for one language version of a benchmark."""
    if language not in spec["languages"]:
        raise ValueError(
            f"language {language!r} not in benchmark: {spec['languages']}"
        )
    samples = []
    for item in spec["items"]:
        tr = item["translations"][language]
        samples.append(
            Sample(
                input=tr["question"],
                choices=list(tr["choices"]),
                target=item["target"],
                id=f"{item['id']}:{language}",
                metadata={
                    "item_id": item["id"],
                    "language": language,
                    "benchmark": spec["name"],
                },
            )
        )
    return samples


def multilingual_task(
    benchmark: str | Path = DEMO_BENCHMARK,
    language: str = "en",
    *,
    cot: bool = False,
) -> Task:
    """One Inspect task: the whole item set in a single language.

    Args:
        benchmark: path to the benchmark definition.
        language: which language version to run.
        cot: whether to let the model reason before answering. Worth knowing that
            turning this on changes what is being measured, and therefore has to
            be held constant across languages or the comparison is confounded by
            the amount of reasoning the model happens to do in each.
    """
    spec = load_benchmark(benchmark)
    return Task(
        dataset=MemoryDataset(
            samples=samples_for_language(spec, language),
            name=f"{spec['name']}-{language}",
        ),
        solver=multiple_choice(cot=cot),
        scorer=choice(),
        name=f"{spec['name']}-{language}",
    )


# Tasks discovered by `inspect eval` on this file. Keep one thin wrapper per
# language rather than generating them dynamically: Inspect resolves tasks by
# name, and a name you can type is worth more than a clever loop.


@task
def english() -> Task:
    return multilingual_task(language="en")


@task
def croatian() -> Task:
    return multilingual_task(language="hr")


@task
def german() -> Task:
    return multilingual_task(language="de")
