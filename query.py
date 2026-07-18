import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

try:
    from retriever.search import MultimodalRetriever
    from retriever.rerank import rerank_top_k
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this script from the root project directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Glance ML - Fashion Retrieval CLI")
    parser.add_argument("query", type=str, help="Natural language description of the fashion image.")
    parser.add_argument("--k", type=int, default=5, help="Number of top results to return.")
    
    args = parser.parse_args()
    
    print(f"\nLoading models and FAISS indices...")
    retriever = MultimodalRetriever()
    
    print(f"\nSearching for: '{args.query}'")
    print("-" * 50)
    
    try:
        # Step 1: Candidate Generation (Dense Retrieval)
        candidate_pool_size = max(50, args.k * 2)
        clip_candidates = retriever.search_clip(args.query, k=candidate_pool_size)
        
        # Step 2: Attribute-Aware Metadata Reranking
        final_results = rerank_top_k(clip_candidates, args.query)[:args.k]
        
        print("\nTop Matching Images:")
        for i, res in enumerate(final_results):
            print(f"Rank {i+1}: {res['image_id']} (Score: {res['score']:.4f})")
            
    except Exception as e:
        print(f"\nError during retrieval: {e}")
        print("Please ensure you have run indexer/build_index.py to generate the FAISS indices.")

if __name__ == "__main__":
    main()
