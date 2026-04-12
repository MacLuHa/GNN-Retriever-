import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.SimpleNamespace(pipeline=lambda *args, **kwargs: None)
if "gliner" not in sys.modules:
    sys.modules["gliner"] = types.SimpleNamespace(GLiNER=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None))
if "simplemma" not in sys.modules:
    sys.modules["simplemma"] = types.SimpleNamespace(lemmatize=lambda token, lang=None: token)
if "httpx" not in sys.modules:
    sys.modules["httpx"] = types.SimpleNamespace(Timeout=object, AsyncClient=object)

HybridEntityExtractor = importlib.import_module(
    "stage3_3_graph_knowledge_base.hybrid_extractor"
).HybridEntityExtractor
ExtractionConfig = importlib.import_module("stage3_3_graph_knowledge_base.config").ExtractionConfig
RelationExtractionResult = importlib.import_module(
    "stage3_3_graph_knowledge_base.models"
).RelationExtractionResult


class FakeLlmExtractor:
    async def extract(self, text: str):
        raise AssertionError("ner_only should not call full entity extraction")

    async def extract_relations(self, text: str, entity_names: list[str]) -> RelationExtractionResult:
        assert entity_names == ["Alice", "Bob", "Carol"]
        return RelationExtractionResult(
            relations=[
                {"from": "Alice", "to": "Bob"},
            ]
        )


class FakeNerExtractor:
    async def extract(self, text: str):
        return [
            SimpleNamespace(name="Alice"),
            SimpleNamespace(name="Bob"),
            SimpleNamespace(name="Carol"),
        ]


def test_ner_only_keeps_only_relation_backed_entities() -> None:
    async def scenario():
        extractor = HybridEntityExtractor(
            ExtractionConfig(mode="ner_only", relation_entity_limit=10),
            FakeLlmExtractor(),
            ner_extractor=FakeNerExtractor(),
        )
        return await extractor.extract("Alice met Bob while Carol stayed aside.")

    extraction, diagnostics = asyncio.run(scenario())

    assert {entity.name for entity in extraction.entities} == {"Alice", "Bob"}
    assert [(relation.from_name, relation.to_name) for relation in extraction.relations] == [("Alice", "Bob")]
    assert diagnostics.mode == "ner_only"
    assert diagnostics.ner_entities_raw == 3
    assert diagnostics.merged_entities == 3


def test_merge_entity_candidates_deduplicates_normalized_names() -> None:
    ner_module = importlib.import_module("stage3_3_graph_knowledge_base.ner_entity_extractor")

    merged = ner_module.merge_entity_candidates(
        ["Anarchism", "Enlightenment"],
        [" anarchism ", "Libertarian socialism"],
    )

    assert [entity.name for entity in merged] == [
        "Anarchism",
        "Enlightenment",
        "Libertarian socialism",
    ]
