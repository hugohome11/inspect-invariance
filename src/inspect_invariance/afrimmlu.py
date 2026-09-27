"""AfriMMLU as a linked multilingual Inspect task.

AfriMMLU (Adelani et al., 2024) is a parallel translation of a 500-item MMLU
subset into sixteen African languages plus English and French. Parallel is the
property that matters here: row *n* of every language file is the same question,
so an item can be identified across languages, which is the precondition for
every analysis in this package and the thing most multilingual benchmarks make
impossible.

One task covers every language at once. That is deliberate. The analysis needs
several languages' responses from the same respondent, and
:func:`inspect_invariance.matrix.from_eval_logs` groups samples by their
``language`` metadata rather than by which file they came from, so a single
evaluation run produces a directly analysable log::

    inspect eval inspect_invariance/afrimmlu.py@afrimmlu --model openai/gpt-5-nano
    inspect-invariance analyse ./logs --reference eng

The dataset is pinned to a commit SHA. It is fetched over HTTP from the Hugging
Face resolve endpoint at that revision rather than through the ``datasets``
library, which keeps the pin visible in the URL and avoids a heavy dependency
for four columns of TSV.

Reference:
    Adelani, D. I., et al. (2024). IrokoBench: A New Benchmark for African
    Languages in the Age of Large Language Models. https://arxiv.org/abs/2406.03368
"""

from __future__ import annotations

import ast
import csv
import io
import os
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

__all__ = [
    "AFRIMMLU_REPO",
    "AFRIMMLU_REVISION",
    "AFRIMMLU_LANGUAGES",
    "REFERENCE_LANGUAGE",
    "load_split",
    "afrimmlu_samples",
    "afrimmlu",
]

AFRIMMLU_REPO = "masakhane/afrimmlu"

AFRIMMLU_REVISION = "96f247619673906ae4c321a3232afc67b98ea57e"
"""Pinned dataset revision.

The register requires external assets to be pinned in version-controlled
storage. Every fetch below interpolates this SHA into the URL, so the data a
reader gets is the data these results were produced from, whatever happens to
the dataset's main branch later.
"""

AFRIMMLU_LANGUAGES: dict[str, str] = {
    "amh": "Amharic",
    "eng": "English",
    "ewe": "Ewe",
    "fra": "French",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lin": "Lingala",
    "lug": "Luganda",
    "orm": "Oromo",
    "sna": "Shona",
    "sot": "Sesotho",
    "swa": "Swahili",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "isiXhosa",
    "yor": "Yoruba",
    "zul": "isiZulu",
}

REFERENCE_LANGUAGE = "eng"
"""The language other versions are compared against.

English is the source the rest were translated from, which makes it the
defensible reference. It is a choice, not a fact about the languages, and
`analyse --reference` overrides it.
"""

SPLITS = ("dev", "val", "test")

_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/data/{lang}/{split}.tsv"

_CACHE_ENV = "INSPECT_INVARIANCE_CACHE"


def _cache_dir() -> Path:
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "inspect_invariance" / AFRIMMLU_REVISION[:12]


def _fetch(language: str, split: str) -> str:
    """Return the raw TSV for one language and split, caching it on disk.

    Cached under the revision SHA, so changing the pin cannot silently serve
    stale data from a previous one.
    """
    cache = _cache_dir() / language
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{split}.tsv"
    if path.exists():
        return path.read_text(encoding="utf-8")

    url = _URL.format(
        repo=AFRIMMLU_REPO, revision=AFRIMMLU_REVISION, lang=language, split=split
    )
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
        raw = response.read().decode("utf-8")
    path.write_text(raw, encoding="utf-8")
    return raw


def load_split(language: str, split: str = "test") -> list[dict[str, object]]:
    """Load one language version of one split as a list of item dicts.

    Each dict carries ``question``, ``choices``, ``answer`` and ``subject``.

    Raises:
        ValueError: on an unknown language or split, or if a row's ``choices``
            cell is not a parseable list. A malformed row is not skipped
            quietly, because a silently shortened language version would bias
            every comparison that language takes part in.
    """
    if language not in AFRIMMLU_LANGUAGES:
        raise ValueError(
            f"unknown AfriMMLU language {language!r}; "
            f"expected one of {sorted(AFRIMMLU_LANGUAGES)}"
        )
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")

    rows = list(csv.DictReader(io.StringIO(_fetch(language, split)), delimiter="\t"))
    items: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        try:
            choices = ast.literal_eval(row["choices"])
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"{language}/{split} row {index}: choices cell is not a list literal"
            ) from exc
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{language}/{split} row {index}: choices is not a list")
        items.append(
            {
                "question": row["question"],
                "choices": [str(c) for c in choices],
                "answer": row["answer"],
                "subject": row["subject"],
            }
        )
    return items


def afrimmlu_samples(
    languages: Sequence[str],
    split: str = "test",
    *,
    limit: int | None = None,
) -> list[Sample]:
    """Build linked Inspect samples across several language versions.

    The reference language is loaded first and defines the item set. Every other
    language is then checked against it, item by item, on the three fields that
    translation must not change: the answer key, the subject, and the number of
    options. A mismatch means the versions are not parallel and the linkage is
    an illusion, so it raises rather than proceeding.

    Args:
        languages: language codes to include; at least two.
        split: ``dev``, ``val`` or ``test``.
        limit: keep only the first *n* items, for a cheap smoke run.

    Raises:
        ValueError: on fewer than two languages, or if the versions are not
            parallel.
    """
    codes = list(dict.fromkeys(languages))
    if len(codes) < 2:
        raise ValueError(
            "a multilingual benchmark needs at least two languages; "
            f"got {codes}"
        )

    reference = REFERENCE_LANGUAGE if REFERENCE_LANGUAGE in codes else codes[0]
    reference_items = load_split(reference, split)
    if limit is not None:
        reference_items = reference_items[:limit]

    item_ids = [
        f"{item['subject']}/{index:04d}"
        for index, item in enumerate(reference_items)
    ]

    samples: list[Sample] = []
    for code in codes:
        items = load_split(code, split)
        if limit is not None:
            items = items[:limit]
        if len(items) != len(reference_items):
            raise ValueError(
                f"{code}/{split} has {len(items)} items but reference "
                f"{reference}/{split} has {len(reference_items)}; the versions "
                "are not parallel"
            )
        for index, (item, reference_item) in enumerate(zip(items, reference_items)):
            if item["answer"] != reference_item["answer"]:
                raise ValueError(
                    f"{code}/{split} item {index} has answer key "
                    f"{item['answer']!r} against {reference_item['answer']!r} in "
                    f"{reference}; the versions are not parallel"
                )
            if item["subject"] != reference_item["subject"]:
                raise ValueError(
                    f"{code}/{split} item {index} has subject {item['subject']!r} "
                    f"against {reference_item['subject']!r} in {reference}; the "
                    "versions are not parallel"
                )
            choices = item["choices"]
            assert isinstance(choices, list)
            if len(choices) != len(reference_item["choices"]):
                raise ValueError(
                    f"{code}/{split} item {index} has {len(choices)} options "
                    f"against {len(reference_item['choices'])} in {reference}; "
                    "the options must correspond one to one"
                )
            samples.append(
                Sample(
                    input=str(item["question"]),
                    choices=list(choices),
                    target=str(item["answer"]),
                    id=f"{item_ids[index]}:{code}",
                    metadata={
                        "item_id": item_ids[index],
                        "language": code,
                        "benchmark": f"afrimmlu-{split}",
                        "subject": item["subject"],
                    },
                )
            )
    return samples


@task
def afrimmlu(
    languages: str | Iterable[str] | None = None,
    split: str = "test",
    limit: int | None = None,
    cot: bool = False,
) -> Task:
    """AfriMMLU administered in every language version, with items linked.

    Running this produces one log whose samples carry ``item_id`` and
    ``language``, which ``inspect-invariance analyse`` reads back to test
    whether the language versions measure the same thing.

    The default is all eighteen language versions of the 500-item test split,
    which is 9,000 samples. Narrow it with ``languages`` or ``limit`` for a
    cheaper run.

    Args:
        languages: comma-separated codes or an iterable; default all eighteen.
        split: ``dev`` (25 items), ``val`` (83) or ``test`` (500).
        limit: keep only the first *n* items of each language version.
        cot: let the model reason before answering. Held constant across
            languages by construction, since one task builds them all; varying
            it between languages would confound the comparison with the amount
            of reasoning the model happens to do in each.
    """
    if languages is None:
        codes: list[str] = list(AFRIMMLU_LANGUAGES)
    elif isinstance(languages, str):
        codes = [c.strip() for c in languages.split(",") if c.strip()]
    else:
        codes = [str(c).strip() for c in languages]

    samples = afrimmlu_samples(codes, split, limit=limit)
    return Task(
        dataset=MemoryDataset(samples=samples, name=f"afrimmlu-{split}"),
        solver=multiple_choice(cot=cot),
        scorer=choice(),
        name=f"afrimmlu-{split}",
    )
