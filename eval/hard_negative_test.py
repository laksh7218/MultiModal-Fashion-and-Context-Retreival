import sys
import csv
import itertools
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from retriever.search import MultimodalRetriever
from retriever.rerank import rerank_top_k

METADATA_PATH = BASE_DIR / "data" / "curated_context_metadata.csv"

def get_swapped_pairs(limit=80):
    images_by_garments = defaultdict(list)
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs = row.get("color_garment_pairs", "")
            if not pairs: continue
            
            pair_dict = {}
            for p in pairs.split(";"):
                if ":" in p:
                    c, g = p.split(":")
                    pair_dict[g.strip()] = c.strip()
            
            if len(pair_dict) >= 2:
                garments_key = tuple(sorted(pair_dict.keys()))
                images_by_garments[garments_key].append({
                    "id": row["image_id"],
                    "colors": {g: pair_dict[g] for g in garments_key},
                    "caption": row.get("caption", "")
                })

    swapped_pairs = []
    used_images = set()
    
    for garments, imgs in images_by_garments.items():
        for img1, img2 in itertools.combinations(imgs, 2):
            if img1["id"] in used_images or img2["id"] in used_images:
                continue
                
            colors1 = img1["colors"]
            colors2 = img2["colors"]
            
            for g1, g2 in itertools.combinations(garments, 2):
                if colors1[g1] == colors2[g2] and colors1[g2] == colors2[g1] and colors1[g1] != colors1[g2]:
                    # We found a color swapped pair!
                    # The query is exactly the caption of image 1. 
                    # Image 1 is the positive, Image 2 is the hard negative.
                    query = img1["caption"]
                    
                    if not query:
                        query = f"A {colors1[g1]} {g1} and a {colors1[g2]} {g2}"
                        
                    swapped_pairs.append({
                        "target": img1["id"],
                        "hard_negative": img2["id"],
                        "query": query
                    })
                    used_images.add(img1["id"])
                    used_images.add(img2["id"])
                    break
                    
            if len(swapped_pairs) >= limit:
                return swapped_pairs
                
    return swapped_pairs

def run_benchmark():
    print("="*60)
    print(" COMPOSITIONAL BINDING BENCHMARK (REAL CORPUS)")
    print("="*60)
    
    pairs = get_swapped_pairs(limit=80)
    print(f"Loaded {len(pairs)} color-swapped pairs from the existing dataset.\n")
    
    retriever = MultimodalRetriever()
    
    clip_correct = 0
    hybrid_correct = 0
    
    # We will only count pairs where the target was actually retrieved in top 150
    # to evaluate ranking accuracy rather than absolute recall
    valid_pairs = 0 
    
    for idx, pair in enumerate(tqdm(pairs, desc="Evaluating Queries")):
        query = pair["query"]
        target = pair["target"]
        hard_neg = pair["hard_negative"]
        
        # We fetch a large candidate pool to ensure both are typically retrieved
        clip_results = retriever.search_clip(query, k=150)
        hybrid_results = rerank_top_k(clip_results, query)
        
        def get_rank(results, img_id):
            for i, res in enumerate(results):
                if res["image_id"] == img_id:
                    return i
            return 9999
            
        # CLIP Evaluation
        clip_target_rank = get_rank(clip_results, target)
        clip_neg_rank = get_rank(clip_results, hard_neg)
        
        # If the target wasn't even in top 150, we skip this query because CLIP failed at basic semantic recall, 
        # so evaluating ranking swap is moot.
        if clip_target_rank == 9999:
            continue
            
        valid_pairs += 1
        
        if clip_target_rank < clip_neg_rank:
            clip_correct += 1
            
        # Hybrid Evaluation
        hybrid_target_rank = get_rank(hybrid_results, target)
        hybrid_neg_rank = get_rank(hybrid_results, hard_neg)
        if hybrid_target_rank < hybrid_neg_rank:
            hybrid_correct += 1

    print("\n" + "="*60)
    print(" RESULTS (Target Rank < Hard Negative Rank)")
    print(f" Valid semantic queries (Target in Top-150): {valid_pairs}")
    print("="*60)
    if valid_pairs > 0:
        print(f"Vanilla CLIP Accuracy:       {clip_correct/valid_pairs:.3f} ({clip_correct}/{valid_pairs})")
        print(f"Metadata Reranker Accuracy:  {hybrid_correct/valid_pairs:.3f} ({hybrid_correct}/{valid_pairs})")
    else:
        print("No valid queries found where target is retrieved.")
    print("="*60)
    
    print("\nNote: This benchmark runs against the real indexed corpus.")
    print("The Reranker accuracy reflects the true metadata quality of the dataset,")
    print("honestly capturing cases where missing or inconsistent annotations")
    print("prevent the reranker from distinguishing the pair.")

if __name__ == "__main__":
    run_benchmark()
