import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "httpx" not in sys.modules:
    sys.modules["httpx"] = types.SimpleNamespace(Timeout=object, AsyncClient=object)
if "qdrant_client" not in sys.modules:
    distance = types.SimpleNamespace(COSINE="cosine", DOT="dot", EUCLID="euclid", MANHATTAN="manhattan")
    fake_models = types.SimpleNamespace(Distance=distance)
    sys.modules["qdrant_client"] = types.SimpleNamespace(AsyncQdrantClient=object, models=fake_models)

from stage3_3_graph_knowledge_base.entity_embedding_service import EntityEmbeddingService
from stage3_3_graph_knowledge_base.models import GraphEntityNode


class FakeGraphStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def set_entity_embedding_id(self, entity_id: str, embedding_id: str) -> None:
        self.calls.append((entity_id, embedding_id))


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2]


class FakeQdrantStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[float]]] = []

    async def upsert_entity_vector(self, entity_id: str, entity_name: str, vector: list[float]) -> str:
        self.calls.append((entity_id, entity_name, vector))
        return f"embedding:{entity_id}"


def test_upsert_entities_skips_previously_embedded_entities() -> None:
    async def scenario() -> tuple[int, FakeGraphStore, FakeEmbedder, FakeQdrantStore]:
        graph_store = FakeGraphStore()
        embedder = FakeEmbedder()
        qdrant_store = FakeQdrantStore()
        service = EntityEmbeddingService(graph_store, embedder, qdrant_store)

        updated = await service.upsert_entities(
            [
                GraphEntityNode(entity_id="known", entity_name="Known Entity", embedding_id="embedding:known"),
                GraphEntityNode(entity_id="new", entity_name="New Entity"),
            ]
        )
        return updated, graph_store, embedder, qdrant_store

    updated, graph_store, embedder, qdrant_store = asyncio.run(scenario())

    assert updated == 1
    assert embedder.calls == ["New Entity"]
    assert qdrant_store.calls == [("new", "New Entity", [0.1, 0.2])]
    assert graph_store.calls == [("new", "embedding:new")]
