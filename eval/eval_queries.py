import argparse
from pathlib import Path

from fashion_retrieval.indexer.config import IndexConfig
from fashion_retrieval.eval.metrics import evaluate_queries
from fashion_retrieval.retriever.search import MultimodalRetriever


ASSIGNMENT_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=IndexConfig.device)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--use-itm", action="store_true")
    parser.add_argument("--itm-pool", type=int, default=20)
    args = parser.parse_args()

    if args.ground_truth:
        metrics = evaluate_queries(
            ASSIGNMENT_QUERIES,
            Path(args.ground_truth),
            top_k=args.top_k,
            config=IndexConfig(device=args.device),
        )
        for metric in metrics:
            print(
                f"{metric.query}\t"
                f"Recall@{args.top_k}={metric.recall_at_k:.3f}\t"
                f"MRR={metric.mrr:.3f}"
            )
        avg = sum(metric.recall_at_k for metric in metrics) / max(1, len(metrics))
        mrr = sum(metric.mrr for metric in metrics) / max(1, len(metrics))
        print(f"Average Recall@{args.top_k}={avg:.3f}")
        print(f"Mean MRR={mrr:.3f}")
        return

    engine = MultimodalRetriever()
    for query in ASSIGNMENT_QUERIES:
        print(f"\nQUERY: {query}")
        for rank, row in enumerate(
            engine.search(
                query,
                top_k=args.top_k,
                use_itm=args.use_itm,
                itm_pool=args.itm_pool,
            ),
            start=1,
        ):
            print(f"{rank}. {row['final_score']:.3f} {row['image_path']}")
            print(
                "   attrs: "
                f"colors={row.get('colors')} "
                f"garments={row.get('garments')} "
                f"contexts={row.get('contexts')} "
                f"objects={row.get('objects')} "
                f"styles={row.get('styles')} "
                f"pairs={row.get('color_garment_pairs')} "
                f"tags={row.get('query_tags')}"
            )


if __name__ == "__main__":
    main()
