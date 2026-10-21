"""The question dimensions in data/dimensions.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DimensionValue(_Frozen):
    description: str = ""
    example: str = ""
    target: float = Field(ge=0, le=1)


class Dimension(_Frozen):
    description: str
    values: dict[str, DimensionValue]

    def targets(self) -> dict[str, float]:
        return {name: value.target for name, value in self.values.items()}


class Sizes(_Frozen):
    candidates: int = Field(gt=0)
    approved_target: tuple[int, int]
    split: dict[str, float]


class DimensionSpec(_Frozen):
    dimensions: dict[str, Dimension]
    constraints: tuple[str, ...]
    sizes: Sizes

    def dimension(self, name: str) -> Dimension:
        return self.dimensions[name]


REQUIRED = (
    "question_type",
    "unanswerable_kind",
    "complexity",
    "phrasing",
    "evidence_location",
    "chunk_boundary",
)


def load_dimensions(path: Path) -> DimensionSpec:
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping")
    spec = DimensionSpec.model_validate(cast(dict[str, Any], loaded))
    missing = [name for name in REQUIRED if name not in spec.dimensions]
    if missing:
        raise ValueError(f"{path}: missing dimensions {missing}")
    for name, dimension in spec.dimensions.items():
        total = sum(dimension.targets().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{path}: targets of {name} sum to {total}, not 1")
    return spec
