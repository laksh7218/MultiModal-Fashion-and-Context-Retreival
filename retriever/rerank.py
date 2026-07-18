import json
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).parent.parent
METADATA_PATH = BASE_DIR / "data" / "curated_context_metadata.csv"

import re
from .vocabulary import (
    COLOR_WORDS,
    GARMENT_WORDS,
    SCENE_WORDS,
    OBJECT_WORDS,
    STYLE_WORDS,
    GENDER_MAP,
    SYNONYMS
)

def parse_query_to_attributes(query: str) -> Dict[str, List[str]]:
    """
    Robust vocabulary-based parser using regex tokenization.
    """
    query = query.lower()
    
    # Normalize multi-word synonyms and edge cases BEFORE tokenization
    for phrase, replacement in SYNONYMS.items():
        query = query.replace(phrase, replacement)
        
    tokens = re.findall(r"\w+", query)
    
    attributes = {
        "garments": [],
        "colors": [],
        "scene": [],
        "objects": [],
        "styles": [],
        "gender": [],
        "color_garment_pairs": []
    }
    
    for token in tokens:
        
        if token in COLOR_WORDS:
            attributes["colors"].append(token)
        if token in GARMENT_WORDS:
            attributes["garments"].append(token)
        if token in SCENE_WORDS:
            attributes["scene"].append(token)
        if token in OBJECT_WORDS:
            attributes["objects"].append(token)
        if token in STYLE_WORDS:
            attributes["styles"].append(token)
        if token in GENDER_MAP:
            attributes["gender"].append(GENDER_MAP[token])
            
    # Extract dynamic color-garment pairs
    for i in range(len(tokens)-1):
        if tokens[i] in COLOR_WORDS and tokens[i+1] in GARMENT_WORDS:
            attributes["color_garment_pairs"].append(f"{tokens[i]}:{tokens[i+1]}")
            
    return attributes

def get_metadata_for_image(image_id: str) -> Dict:
    """
    Fetches the parsed metadata for an image from curated_context_metadata.csv
    """
    import csv
    if not METADATA_PATH.exists():
        return {}
        
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image_id"] == image_id.replace(".jpg", ""):
                return {
                    "garments": row["garments"].split(";") if row.get("garments") else [],
                    "colors": row["colors"].split(";") if row.get("colors") else [],
                    "scene": row["scene"].split(";") if row.get("scene") else [],
                    "objects": row["objects"].split(";") if row.get("objects") else [],
                    "color_garment_pairs": row["color_garment_pairs"].split(";") if row.get("color_garment_pairs") else [],
                    "styles": row["styles"].split(";") if row.get("styles") else [],
                    "gender": [row["gender"]] if row.get("gender") and row["gender"].strip() != "" else []
                }
    return {}

def calculate_rerank_score(semantic_score: float, img_meta: Dict, query_meta: Dict) -> float:
    """
    System C: Attribute-Aware Reranking Score Formula
    """
    def compute_match(q_list, img_list):
        if not q_list: return 1.0 # If query doesn't specify, perfect match
        matches = sum(1 for q in q_list if q in img_list)
        return matches / len(q_list)

    garment_score = compute_match(query_meta.get("garments", []), img_meta.get("garments", []))
    color_score = compute_match(query_meta.get("colors", []), img_meta.get("colors", []))
    scene_score = compute_match(query_meta.get("scene", []), img_meta.get("scene", []))
    object_score = compute_match(query_meta.get("objects", []), img_meta.get("objects", []))
    binding_score = compute_match(query_meta.get("color_garment_pairs", []), img_meta.get("color_garment_pairs", []))
    style_score = compute_match(query_meta.get("styles", []), img_meta.get("styles", []))
    
    # Handle gender specifically to penalize mismatches
    q_gender = query_meta.get("gender", [])
    if not q_gender:
        gender_score = 1.0
    else:
        i_gender = img_meta.get("gender", [])
        if any(g in i_gender for g in q_gender):
            gender_score = 1.0
        else:
            gender_score = 0.0 # Small penalty configured by user

    # FINAL RERANKING FORMULA (Normalized weights summing to 1.0)
    score = (
        (0.35 * semantic_score) +
        (0.15 * garment_score) +
        (0.10 * color_score) +
        (0.10 * style_score) +
        (0.08 * scene_score) +
        (0.07 * object_score) +
        (0.05 * gender_score) +
        (0.10 * binding_score)
    )

    # Active Penalties for Missing Required Attributes
    # (Forces deterministic compositional reasoning and rejection of false positives)
    penalty = 0.0
    if query_meta.get("color_garment_pairs") and binding_score == 0.0:
        penalty += 0.25  # Heavy penalty for missing a specific binding like "red:tie"
    elif query_meta.get("colors") and color_score == 0.0:
        penalty += 0.15
    elif query_meta.get("garments") and garment_score == 0.0:
        penalty += 0.15

    return max(0.0, score - penalty)

def rerank_top_k(clip_results: List[Dict], query: str) -> List[Dict]:
    """System C: Reranking Pipeline"""
    query_meta = parse_query_to_attributes(query)
    
    reranked = []
    for res in clip_results:
        img_id = res["image_id"]
        semantic_score = res["score"] # Original FAISS inner product score
        
        img_meta = get_metadata_for_image(img_id)
        if not img_meta:
            reranked.append(res)
            continue
            
        final_score = calculate_rerank_score(semantic_score, img_meta, query_meta)
        
        reranked.append({
            "image_id": img_id,
            "original_score": semantic_score,
            "score": final_score
        })
        
    # Sort by the new reranked score
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked

# if __name__ == "__main__":
#     # Note: This is a demonstration of how to call the reranker.
#     # In production, this is orchestrated by app.py to keep concerns separated.
#     from .search import MultimodalRetriever
#     retriever = MultimodalRetriever()
#     
#     query = "Red tie and white shirt in formal setting"
#     print(f"\nQuery: {query}")
#     
#     try:
#         top_20 = retriever.search_clip(query, k=20)
#         reranked_results = rerank_top_k(top_20, query)
#         print("\nSystem C (Attribute Reranker) Top 5:")
#         for r in reranked_results[:5]:
#             print(f"Image: {r['image_id']} | Final Score: {r['score']:.4f}")
#     except Exception as e:
#         print("Run build_index.py first!")
