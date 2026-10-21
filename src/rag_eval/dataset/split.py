"""Stratified dev/test split of the approved items, recorded with hashes.

The test split is held out: it is never used to tune prompts, judges or thresholds.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from rag_eval.dataset.schema import Case


def stratum(case: Case) -> str:
    qtype = case.dimensions.get("question_type", "unknown")
    kind = case.dimensions.get("unanswerable_kind")
    return f"{qtype}/{kind}" if kind else qtype


def stratified_split(
    cases: list[Case], dev_share: float, seed: int
) -> tuple[list[Case], list[Case]]:
    """Split each stratum dev/test in proportion; dev gets round(n * dev_share) per stratum."""
    rng = random.Random(seed)
    groups: dict[str, list[Case]] = defaultdict(list)
    for case in sorted(cases, key=lambda c: c.id):
        groups[stratum(case)].append(case)
    dev: list[Case] = []
    test: list[Case] = []
    for name in sorted(groups):
        group = groups[name]
        rng.shuffle(group)
        n_dev = round(len(group) * dev_share)
        dev += [c.model_copy(update={"split": "dev"}) for c in group[:n_dev]]
        test += [c.model_copy(update={"split": "test"}) for c in group[n_dev:]]
    return sorted(dev, key=lambda c: c.id), sorted(test, key=lambda c: c.id)


def _write(path: Path, cases: list[Case]) -> str:
    content = "".join(c.model_dump_json() + "\n" for c in cases)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def write_split(
    dev: list[Case], test: list[Case], data_dir: Path, seed: int, approved_sha256: str
) -> Path:
    record = {
        "seed": seed,
        "approved_sha256": approved_sha256,
        "dev": {"n": len(dev), "sha256": _write(data_dir / "dev.jsonl", dev)},
        "test": {"n": len(test), "sha256": _write(data_dir / "test.jsonl", test)},
        "strata": {
            name: {
                "dev": sum(stratum(c) == name for c in dev),
                "test": sum(stratum(c) == name for c in test),
            }
            for name in sorted({stratum(c) for c in dev + test})
        },
    }
    path = data_dir / "splits.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
