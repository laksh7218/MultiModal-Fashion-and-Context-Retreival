import os
import json
import torch
import faiss
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
EMBEDDING_MODEL = "clip"
FAISS_TOP_K = 100
EVAL_TOP_K = 5

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

INDEX_FILE = DATA_DIR / f"{EMBEDDING_MODEL}_faiss_index.bin"
MAPPING_FILE = DATA_DIR / f"{EMBEDDING_MODEL}_image_mapping.json"
METADATA_FILE = DATA_DIR / "metadata.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Simple vocabularies for our naïve Query Parser
VOCAB_COLORS = {"red", "blue", "white", "black", "green", "yellow", "pink", "purple", "grey", "gray", "brown"}
VOCAB_GARMENTS = {"shirt", "tie", "jacket", "jeans", "pants", "dress", "skirt", "suit", "blazer", "t-shirt"}
VOCAB_SCENES = {"park", "meeting", "office", "street", "beach", "party"}

def load_system():
    print(f"Loading {EMBEDDING_MODEL.upper()} model on {device}...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    # Safetensors fallback for transformers v5+
    try:
        model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    except Exception:
        model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    
    print(f"Loading FAISS Index...")
    index = faiss.read_index(str(INDEX_FILE))
    
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        image_mapping = json.load(f)
        
    metadata_map = {}
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            metadata_map[record["image_id"]] = record
            
    return processor, model, index, image_mapping, metadata_map

def parse_query(query):
    """Naïve attribute extractor"""
    words = set(query.lower().split())
    return {
        "colors": list(words.intersection(VOCAB_COLORS)),
        "garments": list(words.intersection(VOCAB_GARMENTS)),
        "scenes": list(words.intersection(VOCAB_SCENES))
    }

def calculate_rerank_score(base_score, metadata, parsed_query):
    """
    Increases the cosine similarity score if explicit metadata attributes 
    match the parsed intents from the user's text query.
    """
    if not metadata:
        return base_score # Fallback if hash-name mismatch happens

    boost = 0.0
    
    # +0.1 for every matching color
    meta_colors = [c.lower() for c in metadata.get("colors", [])]
    for c in parsed_query["colors"]:
        if c in meta_colors:
            boost += 0.1
            
    # +0.1 for every matching garment
    meta_garments = [g.lower() for g in metadata.get("garments", [])]
    for g in parsed_query["garments"]:
        if g in meta_garments:
            boost += 0.1
            
    # +0.05 for scene match
    meta_scene = metadata.get("scene", "").lower()
    for s in parsed_query["scenes"]:
        if s in meta_scene:
            boost += 0.05
            
    # Advanced: Pair binding check (e.g. "red:tie")
    meta_pairs = [p.lower() for p in metadata.get("pairs", [])]
    for c in parsed_query["colors"]:
        for g in parsed_query["garments"]:
            if f"{c}:{g}" in meta_pairs:
                boost += 0.15 # Strong compositional boost
                
    return base_score + boost

@torch.no_grad()
def evaluate_query(query, processor, model, index, image_mapping, metadata_map):
    print(f"\n" + "="*60)
    print(f"QUERY: '{query}'")
    print("="*60)
    
    parsed_query = parse_query(query)
    print(f"Parsed Intents -> {parsed_query}\n")
    
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    outputs = model.get_text_features(**inputs)
    text_embeds = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0] if isinstance(outputs, tuple) else outputs
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    emb_np = text_embeds.cpu().numpy().astype('float32')
    
    # Baseline FAISS Search (Top 100 for reranking pool)
    scores, indices = index.search(emb_np, FAISS_TOP_K)
    
    results = []
    for i in range(FAISS_TOP_K):
        vector_idx = str(indices[0][i])
        base_score = float(scores[0][i])
        image_id = image_mapping.get(vector_idx, "UNKNOWN_ID")
        meta = metadata_map.get(image_id, {})
        
        # Attribute-Aware Rerank
        final_score = calculate_rerank_score(base_score, meta, parsed_query)
        
        results.append({
            "image_id": image_id,
            "base_score": base_score,
            "final_score": final_score,
            "metadata": meta
        })
        
    # --- PRINT BASELINE (Top K) ---
    print(f"--- BASELINE (Raw CLIP) TOP {EVAL_TOP_K} ---")
    baseline_results = sorted(results, key=lambda x: x["base_score"], reverse=True)[:EVAL_TOP_K]
    for i, r in enumerate(baseline_results):
        meta = r["metadata"]
        g = meta.get("garments", []) if meta else []
        c = meta.get("colors", []) if meta else []
        print(f"[{i+1}] Score: {r['base_score']:.4f} | ID: {r['image_id']} | Garments: {g} | Colors: {c}")
        
    # --- PRINT RERANKED (Top K) ---
    print(f"\n--- RERANKED (Attribute-Aware) TOP {EVAL_TOP_K} ---")
    reranked_results = sorted(results, key=lambda x: x["final_score"], reverse=True)[:EVAL_TOP_K]
    for i, r in enumerate(reranked_results):
        meta = r["metadata"]
        g = meta.get("garments", []) if meta else []
        c = meta.get("colors", []) if meta else []
        print(f"[{i+1}] Score: {r['final_score']:.4f} (Base: {r['base_score']:.4f}) | ID: {r['image_id']} | Garments: {g} | Colors: {c}")


def run_evaluation():
    processor, model, index, image_mapping, metadata_map = load_system()
    
    test_queries = [
        "a white shirt with red tie",
        "a person wearing jacket in a park",
        "blue jeans and black shirt"
    ]
    
    for query in test_queries:
        evaluate_query(query, processor, model, index, image_mapping, metadata_map)

if __name__ == "__main__":
    run_evaluation()
