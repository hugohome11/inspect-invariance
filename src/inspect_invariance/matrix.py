"""Turning Inspect eval logs into a response matrix.

Classical test theory assumes a matrix of *respondents* by *items*. A benchmark
run against a model does not obviously supply respondents, and how you fill that
role decides what the analysis can conclude. Two designs are supported, and the
choice is yours to make and to state:

``epochs``
    One model, the benchmark run ``k`` times. Each run is a respondent. This
    treats sampling variability as the source of individual differences, so the
    "ability" being conditioned on is the run-to-run fluctuation of a single
    model. Cheap, and adequate for detecting gross translation failures, but the
    respondents are not independent in the way test theory assumes and the
    ability range is narrow.

``models``
    A panel of models, each run once (or averaged over epochs). Each model is a
    respondent. The ability distribution is then genuine variation in capability,
    which is much closer to the assumptions the statistics were built on. This is
    the design to prefer when you can afford it: a dozen models spanning a real
    capability range gives the matching variable something to match on.

Either way the requirement is the same and it is strict: the *same items* must
appear in every language, linked by a stable ``item_id``. Without that link there
is no common scale and nothing to compare.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

__all__ = ["ResponseMatrix", "from_records", "from_eval_logs", "Design"]

Design = Literal["epochs", "models"]


@dataclass(frozen=True)
class ResponseMatrix:
    """A respondents-by-items 0/1 matrix for one language, with aligned labels."""

    language: str
    items: tuple[str, ...]
    respondents: tuple[str, ...]
    data: np.ndarray
    design: Design

    def __post_init__(self) -> None:
        if self.data.shape != (len(self.respondents), len(self.items)):
            raise ValueError(
                f"data shape {self.data.shape} does not match "
                f"{len(self.respondents)} respondents by {len(self.items)} items"
            )

    @property
    def n_respondents(self) -> int:
        return self.data.shape[0]

    @property
    def n_items(self) -> int:
        return self.data.shape[1]

    def p_values(self) -> np.ndarray:
        """Proportion correct per item, the classical difficulty index."""
        return self.data.mean(axis=0)

    def aligned_to(self, items: Iterable[str]) -> "ResponseMatrix":
        """Reorder columns to a given item order, dropping any item not present.

        Two language versions can only be compared item by item, so both must be
        put in the same order before anything else happens.
        """
        wanted = [i for i in items if i in self.items]
        idx = [self.items.index(i) for i in wanted]
        return ResponseMatrix(
            language=self.language,
            items=tuple(wanted),
            respondents=self.respondents,
            data=self.data[:, idx],
            design=self.design,
        )


def from_records(
    records: Iterable[dict],
    *,
    design: Design = "epochs",
) -> dict[str, ResponseMatrix]:
    """Build one :class:`ResponseMatrix` per language from flat records.

    Each record needs ``language``, ``item_id``, ``respondent`` and ``correct``.
    This is the format :func:`from_eval_logs` produces, and the format to use if
    your scores come from somewhere other than Inspect.

    Raises:
        ValueError: if a language ends up with a ragged matrix, which means some
            respondent did not answer every item. Imputing there would invent
            data, so it is refused rather than patched.
    """
    cells: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    langs: list[str] = []
    items_by_lang: dict[str, list[str]] = defaultdict(list)
    resp_by_lang: dict[str, list[str]] = defaultdict(list)

    for rec in records:
        try:
            lang = str(rec["language"])
            item = str(rec["item_id"])
            resp = str(rec["respondent"])
            correct = int(bool(rec["correct"]))
        except KeyError as exc:
            raise ValueError(f"record missing required field {exc}: {rec!r}") from exc

        if lang not in langs:
            langs.append(lang)
        if item not in items_by_lang[lang]:
            items_by_lang[lang].append(item)
        if resp not in resp_by_lang[lang]:
            resp_by_lang[lang].append(resp)
        cells[lang][(resp, item)] = correct

    out: dict[str, ResponseMatrix] = {}
    for lang in langs:
        items = items_by_lang[lang]
        resps = resp_by_lang[lang]
        data = np.full((len(resps), len(items)), -1, dtype=int)
        for r, resp in enumerate(resps):
            for c, item in enumerate(items):
                val = cells[lang].get((resp, item))
                if val is not None:
                    data[r, c] = val
        if (data < 0).any():
            missing = int((data < 0).sum())
            raise ValueError(
                f"language {lang!r} has {missing} missing responses; every "
                f"respondent must answer every item for a complete-data analysis"
            )
        out[lang] = ResponseMatrix(
            language=lang,
            items=tuple(items),
            respondents=tuple(resps),
            data=data,
            design=design,
        )
    return out


def from_eval_logs(
    paths: Iterable[str | Path],
    *,
    design: Design = "epochs",
) -> dict[str, ResponseMatrix]:
    """Read Inspect ``.eval`` or ``.json`` logs into per-language matrices.

    The task in :mod:`inspect_invariance.task` writes ``language`` and
    ``item_id`` into each sample's metadata, which is what makes the samples
    linkable across languages. Logs produced by some other task must carry the
    same two fields.

    Args:
        paths: log files, or directories to search for ``*.json``.
        design: how respondents were obtained; see the module docstring.
    """
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    records: list[dict] = []
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        model = _dig(payload, "eval", "model") or f.stem
        for sample in payload.get("samples", []) or []:
            meta = sample.get("metadata") or {}
            language = meta.get("language")
            item_id = meta.get("item_id")
            if language is None or item_id is None:
                continue
            correct = _sample_correct(sample)
            if correct is None:
                continue
            epoch = sample.get("epoch", 1)
            respondent = str(model) if design == "models" else f"{model}#{epoch}"
            records.append(
                {
                    "language": language,
                    "item_id": item_id,
                    "respondent": respondent,
                    "correct": correct,
                }
            )

    if not records:
        raise ValueError(
            "no usable samples found; logs must carry metadata.language and "
            "metadata.item_id, and samples must be scored"
        )
    return from_records(records, design=design)


def _dig(payload: dict, *keys: str):
    cur = payload
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _sample_correct(sample: dict) -> int | None:
    """Extract a 0/1 outcome from an Inspect sample's scores."""
    scores = sample.get("scores") or {}
    for payload in scores.values():
        if not isinstance(payload, dict):
            continue
        value = payload.get("value")
        if value in ("C", "I"):
            return 1 if value == "C" else 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value >= 0.5)
    return None
