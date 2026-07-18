from __future__ import annotations

from dataclasses import dataclass

from fashion_retrieval.common import read_jsonl
from fashion_retrieval.indexer.config import IndexConfig
from fashion_retrieval.retriever.search import MultimodalRetriever


@dataclass
class QueryMetric:
    query: str
    recall_at_k: float
    mrr: float
    retrieved: list[str]
    relevant: list[str]


def load_ground_truth(path) -> dict[str, set[str]]:
    rows = read_jsonl(path)
    truth: dict[str, set[str]] = {}
    for row in rows:
        ids = row.get("relevant_image_ids") or row.get("image_ids") or []
        truth[row["query"]] = set(ids)
    return truth


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, image_id in enumerate(retrieved, start=1):
        if image_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_queries(
    queries: list[str],
    ground_truth_path,
    top_k: int = 5,
    config: IndexConfig = IndexConfig(),
) -> list[QueryMetric]:
    engine = MultimodalRetriever()
    truth = load_ground_truth(ground_truth_path)
    metrics: list[QueryMetric] = []
    for query in queries:
        results = engine.search_clip(query, k=top_k)
        retrieved = [row["image_id"] for row in results]
        relevant = truth.get(query, set())
        metrics.append(
            QueryMetric(
                query=query,
                recall_at_k=recall_at_k(retrieved, relevant, top_k),
                mrr=reciprocal_rank(retrieved, relevant),
                retrieved=retrieved,
                relevant=sorted(relevant),
            )
        )
    return metrics
