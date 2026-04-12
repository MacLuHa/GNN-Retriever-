from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def compute_query_metrics(
    ranked_entity_ids: Sequence[str],
    relevant_entity_ids: set[str],
    ks: Sequence[int],
) -> dict[str, float]:
    """Compute ranking metrics for a single query from an ordered ranking list."""
    if not ks:
        raise ValueError("ks must not be empty")

    ordered_ks = sorted({int(k) for k in ks if int(k) > 0})
    if not ordered_ks:
        raise ValueError("ks must contain positive integers")

    total_relevant = len(relevant_entity_ids)
    if total_relevant == 0:
        raise ValueError("relevant_entity_ids must not be empty")

    rel_flags = [1 if entity_id in relevant_entity_ids else 0 for entity_id in ranked_entity_ids]

    metrics: dict[str, float] = {}
    first_rel_rank: int | None = None
    for idx, is_relevant in enumerate(rel_flags, start=1):
        if is_relevant:
            first_rel_rank = idx
            break

    for k in ordered_ks:
        topk_flags = rel_flags[:k]
        rel_in_k = sum(topk_flags)
        precision = rel_in_k / k
        recall = rel_in_k / total_relevant
        hit = 1.0 if rel_in_k > 0 else 0.0

        dcg = 0.0
        for rank, flag in enumerate(topk_flags, start=1):
            if flag:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_rels = min(total_relevant, k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_rels + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        mrr = 0.0
        if first_rel_rank is not None and first_rel_rank <= k:
            mrr = 1.0 / first_rel_rank

        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"hit@{k}"] = hit
        metrics[f"mrr@{k}"] = mrr
        metrics[f"ndcg@{k}"] = ndcg

    return metrics


@dataclass(slots=True)
class EntityRow:
    entity_id: str
    embedding_id: str
    gnn_embedding: list[float] | None


@dataclass(slots=True)
class GoldenRow:
    chunk_id: str
    positive_entity_ids: list[str]


@dataclass(slots=True)
class ChunkRow:
    chunk_id: str
    embedding_id: str
    gnn_embedding: list[float] | None


@dataclass(slots=True)
class EvalRow:
    chunk_id: str
    positive_entity_ids: list[str]
    baseline_vector: Any
    gnn_vector: Any


class BenchmarkRunner:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        qdrant_url: str,
        qdrant_api_key: str | None,
        qdrant_chunk_collection: str,
        qdrant_entity_collection: str,
        qdrant_id_field: str,
    ) -> None:
        try:
            import numpy as np
            from neo4j import GraphDatabase
            from qdrant_client import QdrantClient
            from qdrant_client.models import FieldCondition, Filter, MatchAny
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for benchmark. Install: numpy neo4j qdrant-client"
            ) from exc

        self.np = np
        self.FieldCondition = FieldCondition
        self.Filter = Filter
        self.MatchAny = MatchAny

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.qdrant_chunk_collection = qdrant_chunk_collection
        self.qdrant_entity_collection = qdrant_entity_collection
        self.qdrant_id_field = qdrant_id_field

    def close(self) -> None:
        self.driver.close()

    def _run_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [row.data() for row in result]

    def load_entities(self) -> list[EntityRow]:
        rows = self._run_query(
            """
            MATCH (e:Entity)
            WHERE e.entity_id IS NOT NULL
            RETURN
              toString(e.entity_id) AS entity_id,
              toString(coalesce(e.embedding_id, e.entity_id)) AS embedding_id,
              e.gnn_embedding AS gnn_embedding
            """
        )
        entities: list[EntityRow] = []
        for row in rows:
            entities.append(
                EntityRow(
                    entity_id=str(row["entity_id"]),
                    embedding_id=str(row["embedding_id"]),
                    gnn_embedding=row.get("gnn_embedding"),
                )
            )
        return entities

    def create_golden_dataset(self, limit: int, min_positives: int) -> list[GoldenRow]:
        rows = self._run_query(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.chunk_id IS NOT NULL AND c.embedding_id IS NOT NULL AND e.entity_id IS NOT NULL
            WITH c, collect(DISTINCT toString(e.entity_id)) AS positives
            WHERE size(positives) >= $min_positives
            RETURN toString(c.chunk_id) AS chunk_id, positives
            ORDER BY rand()
            LIMIT $limit
            """,
            {"limit": int(limit), "min_positives": int(min_positives)},
        )
        return [GoldenRow(chunk_id=str(row["chunk_id"]), positive_entity_ids=list(row["positives"])) for row in rows]

    def load_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[ChunkRow]:
        if not chunk_ids:
            return []
        rows = self._run_query(
            """
            MATCH (c:Chunk)
            WHERE toString(c.chunk_id) IN $chunk_ids
            RETURN
              toString(c.chunk_id) AS chunk_id,
              toString(c.embedding_id) AS embedding_id,
              c.gnn_embedding AS gnn_embedding
            """,
            {"chunk_ids": list(chunk_ids)},
        )
        chunks: list[ChunkRow] = []
        for row in rows:
            embedding_id = row.get("embedding_id")
            if embedding_id is None:
                continue
            chunks.append(
                ChunkRow(
                    chunk_id=str(row["chunk_id"]),
                    embedding_id=str(embedding_id),
                    gnn_embedding=row.get("gnn_embedding"),
                )
            )
        return chunks

    def fetch_qdrant_vectors(
        self,
        collection_name: str,
        embedding_ids: Sequence[str],
        batch_size: int,
    ) -> dict[str, Any]:
        vector_map: dict[str, Any] = {}
        ids = [str(x) for x in embedding_ids if x is not None]

        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            if not batch:
                continue

            points, _ = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=self.Filter(
                    must=[
                        self.FieldCondition(
                            key=self.qdrant_id_field,
                            match=self.MatchAny(any=batch),
                        )
                    ]
                ),
                with_payload=[self.qdrant_id_field],
                with_vectors=True,
                limit=len(batch),
            )
            for point in points:
                payload = point.payload or {}
                embedding_id = payload.get(self.qdrant_id_field)
                if embedding_id is None:
                    continue
                vector = point.vector
                if isinstance(vector, dict):
                    vector = next(iter(vector.values()), None)
                if vector is None:
                    continue
                vector_map[str(embedding_id)] = self.np.asarray(vector, dtype=self.np.float32)
        return vector_map


def _normalize_rows(matrix: Any, np: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _load_or_create_golden(
    runner: BenchmarkRunner,
    golden_path: Path,
    golden_mode: str,
    golden_limit: int,
    golden_min_positives: int,
) -> list[GoldenRow]:
    if golden_mode not in {"auto", "create", "reuse"}:
        raise ValueError("golden_mode must be one of: auto, create, reuse")

    should_create = golden_mode == "create" or (golden_mode == "auto" and not golden_path.exists())
    if golden_mode == "reuse" and not golden_path.exists():
        raise FileNotFoundError(f"golden dataset not found at {golden_path}")

    if should_create:
        golden_rows = runner.create_golden_dataset(golden_limit, golden_min_positives)
        if not golden_rows:
            raise RuntimeError("No golden rows created from Neo4j. Check Chunk-Entity MENTIONS graph.")
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        with golden_path.open("w", encoding="utf-8") as f:
            for row in golden_rows:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": row.chunk_id,
                            "positive_entity_ids": row.positive_entity_ids,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        return golden_rows

    rows: list[GoldenRow] = []
    with golden_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            rows.append(
                GoldenRow(
                    chunk_id=str(data["chunk_id"]),
                    positive_entity_ids=[str(x) for x in data["positive_entity_ids"]],
                )
            )
    if not rows:
        raise RuntimeError(f"Golden dataset is empty: {golden_path}")
    return rows


def _evaluate_model(
    model_name: str,
    query_vectors: Any,
    entity_matrix: Any,
    entity_ids: list[str],
    positives_by_query: list[set[str]],
    chunk_ids: list[str],
    ks: list[int],
    batch_size: int,
    np: Any,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    max_k = max(ks)
    entity_matrix_t = entity_matrix.T

    per_query_rows: list[dict[str, Any]] = []
    sum_metrics: dict[str, float] = {}

    for start in range(0, len(chunk_ids), batch_size):
        end = min(start + batch_size, len(chunk_ids))
        batch_q = query_vectors[start:end]
        scores = batch_q @ entity_matrix_t

        for local_idx in range(scores.shape[0]):
            query_idx = start + local_idx
            score_row = scores[local_idx]
            top_indices = np.argpartition(-score_row, max_k - 1)[:max_k]
            top_indices = top_indices[np.argsort(-score_row[top_indices])]
            ranked_ids = [entity_ids[int(i)] for i in top_indices]

            metrics = compute_query_metrics(ranked_ids, positives_by_query[query_idx], ks)
            for key, value in metrics.items():
                sum_metrics[key] = sum_metrics.get(key, 0.0) + value

            row: dict[str, Any] = {
                "model": model_name,
                "chunk_id": chunk_ids[query_idx],
                "num_positives": len(positives_by_query[query_idx]),
            }
            row.update(metrics)
            per_query_rows.append(row)

    num_queries = float(len(chunk_ids))
    summary = {metric: value / num_queries for metric, value in sum_metrics.items()}
    summary["num_queries"] = num_queries
    return summary, per_query_rows


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    runner = BenchmarkRunner(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_chunk_collection=args.qdrant_chunk_collection,
        qdrant_entity_collection=args.qdrant_entity_collection,
        qdrant_id_field=args.qdrant_id_field,
    )

    try:
        golden_rows = _load_or_create_golden(
            runner=runner,
            golden_path=Path(args.golden_path),
            golden_mode=args.golden_mode,
            golden_limit=args.golden_limit,
            golden_min_positives=args.golden_min_positives,
        )

        entities = runner.load_entities()
        if not entities:
            raise RuntimeError("No entities found in Neo4j.")

        np = runner.np

        entity_qdrant = runner.fetch_qdrant_vectors(
            collection_name=args.qdrant_entity_collection,
            embedding_ids=[e.embedding_id for e in entities],
            batch_size=args.qdrant_batch_size,
        )

        entity_gnn: dict[str, Any] = {}
        for entity in entities:
            if entity.gnn_embedding is None:
                continue
            entity_gnn[entity.entity_id] = np.asarray(entity.gnn_embedding, dtype=np.float32)

        entity_baseline: dict[str, Any] = {}
        for entity in entities:
            vec = entity_qdrant.get(entity.embedding_id)
            if vec is not None:
                entity_baseline[entity.entity_id] = vec

        common_entity_ids = sorted(set(entity_baseline.keys()) & set(entity_gnn.keys()))
        if not common_entity_ids:
            raise RuntimeError(
                "No common entities with both baseline Qdrant vectors and gnn_embedding."
            )

        entity_baseline_matrix = np.vstack([entity_baseline[eid] for eid in common_entity_ids]).astype(np.float32)
        entity_gnn_matrix = np.vstack([entity_gnn[eid] for eid in common_entity_ids]).astype(np.float32)

        entity_baseline_matrix = _normalize_rows(entity_baseline_matrix, np)
        entity_gnn_matrix = _normalize_rows(entity_gnn_matrix, np)

        chunk_rows = runner.load_chunks_by_ids([row.chunk_id for row in golden_rows])
        chunks_by_id = {row.chunk_id: row for row in chunk_rows}

        chunk_qdrant = runner.fetch_qdrant_vectors(
            collection_name=args.qdrant_chunk_collection,
            embedding_ids=[row.embedding_id for row in chunk_rows],
            batch_size=args.qdrant_batch_size,
        )

        eval_rows: list[EvalRow] = []
        common_entity_set = set(common_entity_ids)
        for row in golden_rows:
            chunk = chunks_by_id.get(row.chunk_id)
            if chunk is None or chunk.gnn_embedding is None:
                continue

            baseline_vec = chunk_qdrant.get(chunk.embedding_id)
            if baseline_vec is None:
                continue

            positives = sorted(set(row.positive_entity_ids) & common_entity_set)
            if not positives:
                continue

            eval_rows.append(
                EvalRow(
                    chunk_id=row.chunk_id,
                    positive_entity_ids=positives,
                    baseline_vector=np.asarray(baseline_vec, dtype=np.float32),
                    gnn_vector=np.asarray(chunk.gnn_embedding, dtype=np.float32),
                )
            )

        if not eval_rows:
            raise RuntimeError(
                "No valid eval rows after filtering. Check chunk embeddings and entity overlaps."
            )

        query_baseline_matrix = _normalize_rows(
            np.vstack([row.baseline_vector for row in eval_rows]).astype(np.float32), np
        )
        query_gnn_matrix = _normalize_rows(
            np.vstack([row.gnn_vector for row in eval_rows]).astype(np.float32), np
        )

        ks = sorted({int(k) for k in args.ks if int(k) > 0})
        if not ks:
            raise ValueError("ks must contain positive integers")

        positives_by_query = [set(row.positive_entity_ids) for row in eval_rows]
        chunk_ids = [row.chunk_id for row in eval_rows]

        baseline_summary, baseline_per_query = _evaluate_model(
            model_name="baseline",
            query_vectors=query_baseline_matrix,
            entity_matrix=entity_baseline_matrix,
            entity_ids=common_entity_ids,
            positives_by_query=positives_by_query,
            chunk_ids=chunk_ids,
            ks=ks,
            batch_size=args.score_batch_size,
            np=np,
        )
        gnn_summary, gnn_per_query = _evaluate_model(
            model_name="gnn",
            query_vectors=query_gnn_matrix,
            entity_matrix=entity_gnn_matrix,
            entity_ids=common_entity_ids,
            positives_by_query=positives_by_query,
            chunk_ids=chunk_ids,
            ks=ks,
            batch_size=args.score_batch_size,
            np=np,
        )

        delta: dict[str, float] = {}
        for key, value in gnn_summary.items():
            if key in baseline_summary and key != "num_queries":
                delta[key] = value - baseline_summary[key]

        result = {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "golden_path": str(Path(args.golden_path).resolve()),
            "counts": {
                "golden_rows": len(golden_rows),
                "eval_rows": len(eval_rows),
                "common_entities": len(common_entity_ids),
            },
            "baseline": baseline_summary,
            "gnn": gnn_summary,
            "delta_gnn_minus_baseline": delta,
            "ks": ks,
        }

        run_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_json = run_dir / "summary.json"
        summary_json.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

        summary_csv = run_dir / "summary.csv"
        metric_keys = sorted(
            [
                key
                for key in set(list(baseline_summary.keys()) + list(gnn_summary.keys()) + list(delta.keys()))
                if key != "num_queries"
            ]
        )
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "num_queries", *metric_keys],
            )
            writer.writeheader()
            baseline_row = {"model": "baseline", "num_queries": baseline_summary.get("num_queries", 0.0)}
            baseline_row.update({k: baseline_summary.get(k) for k in metric_keys})
            writer.writerow(baseline_row)

            gnn_row = {"model": "gnn", "num_queries": gnn_summary.get("num_queries", 0.0)}
            gnn_row.update({k: gnn_summary.get(k) for k in metric_keys})
            writer.writerow(gnn_row)

            delta_row = {"model": "delta_gnn_minus_baseline", "num_queries": baseline_summary.get("num_queries", 0.0)}
            delta_row.update({k: delta.get(k) for k in metric_keys})
            writer.writerow(delta_row)

        per_query_csv = run_dir / "per_query_metrics.csv"
        with per_query_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["model", "chunk_id", "num_positives"] + sorted(
                {k for row in baseline_per_query + gnn_per_query for k in row.keys() if k not in {"model", "chunk_id", "num_positives"}}
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in baseline_per_query + gnn_per_query:
                writer.writerow(row)

        result["artifacts"] = {
            "run_dir": str(run_dir.resolve()),
            "summary_json": str(summary_json.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "per_query_csv": str(per_query_csv.resolve()),
        }
        return result
    finally:
        runner.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval quality: baseline Qdrant vectors vs Neo4j gnn_embedding",
    )

    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j_password"))

    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument(
        "--qdrant-chunk-collection",
        default=os.getenv("QDRANT_CHUNK_COLLECTION", "chunks"),
    )
    parser.add_argument(
        "--qdrant-entity-collection",
        default=os.getenv("QDRANT_ENTITY_COLLECTION", "entities"),
    )
    parser.add_argument("--qdrant-id-field", default=os.getenv("QDRANT_ID_FIELD", "embedding_id"))

    parser.add_argument("--golden-path", default=os.getenv("GOLDEN_DATASET_PATH", "benchmarks/golden_dataset.jsonl"))
    parser.add_argument("--golden-mode", choices=["auto", "create", "reuse"], default="auto")
    parser.add_argument("--golden-limit", type=int, default=int(os.getenv("GOLDEN_LIMIT", "2000")))
    parser.add_argument(
        "--golden-min-positives",
        type=int,
        default=int(os.getenv("GOLDEN_MIN_POSITIVES", "1")),
    )

    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--qdrant-batch-size", type=int, default=int(os.getenv("QDRANT_BATCH_SIZE", "256")))
    parser.add_argument("--score-batch-size", type=int, default=int(os.getenv("SCORE_BATCH_SIZE", "128")))

    parser.add_argument("--output-dir", default=os.getenv("BENCHMARK_OUTPUT_DIR", "benchmarks/runs"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_benchmark(args)

    print("Benchmark completed")
    print(f"Golden dataset: {result['golden_path']}")
    print(f"Eval rows: {result['counts']['eval_rows']}")
    print(f"Common entities: {result['counts']['common_entities']}")
    print("Baseline vs GNN:")

    for key in sorted(result["delta_gnn_minus_baseline"].keys()):
        b = result["baseline"].get(key)
        g = result["gnn"].get(key)
        d = result["delta_gnn_minus_baseline"].get(key)
        if b is None or g is None or d is None:
            continue
        print(f"  {key}: baseline={b:.6f} gnn={g:.6f} delta={d:+.6f}")

    print(f"Artifacts: {result['artifacts']['run_dir']}")


if __name__ == "__main__":
    main()
