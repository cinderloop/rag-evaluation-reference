from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval.config import RunConfig, deep_merge, load_config

BASELINE = Path("configs/baseline.yaml")


def test_baseline_loads() -> None:
    config = load_config(BASELINE)
    assert config.name == "baseline"
    assert config.generation.mode == "query"
    assert config.llm.temperature == 0


def test_deep_merge_merges_mappings_and_replaces_other_values() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    override = {"a": {"y": 20, "z": 30}, "b": [9]}
    assert deep_merge(base, override) == {"a": {"x": 1, "y": 20, "z": 30}, "b": [9], "c": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}


def test_mutation_extends_baseline_and_overrides_only_its_fields(tmp_path: Path) -> None:
    mutation = tmp_path / "reduced-top-n.yaml"
    mutation.write_text(
        f"extends: {BASELINE.resolve()}\nname: reduced-top-n\nretrieval:\n  top_n: 1\n",
        encoding="utf-8",
    )
    baseline = load_config(BASELINE)
    mutated = load_config(mutation)
    assert mutated.retrieval.top_n == 1
    assert mutated.retrieval.similarity_threshold == baseline.retrieval.similarity_threshold
    assert mutated.llm == baseline.llm
    assert mutated.config_hash() != baseline.config_hash()


def test_extends_is_relative_to_the_declaring_file(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "base.yaml").write_text(BASELINE.read_text(encoding="utf-8"), encoding="utf-8")
    child = tmp_path / "nested" / "child.yaml"
    child.write_text("extends: ../base.yaml\nname: child\n", encoding="utf-8")
    assert load_config(child).name == "child"


def test_circular_extends_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_config_hash_is_stable_and_order_independent() -> None:
    config = load_config(BASELINE)
    reordered = RunConfig.model_validate(dict(reversed(list(config.model_dump().items()))))
    assert reordered.config_hash() == config.config_hash()


def _baseline_with(**changes: object) -> dict[str, object]:
    data = load_config(BASELINE).model_dump()
    for dotted, value in changes.items():
        *parents, leaf = dotted.split("__")
        target = data
        for key in parents:
            target = target[key]
        target[leaf] = value
    return data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"generation__system_prompt": "Today is {datetime}."}, "variables"),
        ({"chunking__overlap": 1000}, "overlap"),
        ({"anythingllm_image": "mintplexlabs/anythingllm:latest"}, "pattern"),
        ({"llm__model__digest": "abc"}, "pattern"),
        ({"name": "Baseline Config"}, "lowercase"),
        ({"retrieval__similarity_threshold": 1.5}, "less than or equal"),
        ({"generation__mode": "automatic"}, "chat"),
    ],
)
def test_invalid_configs_are_rejected(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RunConfig.model_validate(_baseline_with(**change))


def test_unknown_keys_are_rejected() -> None:
    data = load_config(BASELINE).model_dump()
    data["retrieval"]["topN"] = 4
    with pytest.raises(ValidationError, match="Extra inputs"):
        RunConfig.model_validate(data)
