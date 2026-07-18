import os
import json
import csv
import torch
import faiss
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
EMBEDDING_MODEL = "clip" 

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "curated_images"

# Metadata inputs/outputs
FASHION_CSV = DATA_DIR / "fashionpedia_metadata.csv"
SYNTH_CSV = DATA_DIR / "curated_context_metadata.csv"
METADATA_JSONL = DATA_DIR / "metadata.jsonl"

# Index outputs
INDEX_FILE = DATA_DIR / f"{EMBEDDING_MODEL}_faiss_index.bin"
MAPPING_FILE = DATA_DIR / f"{EMBEDDING_MODEL}_image_mapping.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_metadata_jsonl():
    """Merges both CSVs into the final ground-truth metadata.jsonl schema"""
    print(f"Building unified {METADATA_JSONL.name}...")
    
    metadata_records = []
    
    # Process both CSVs
    for csv_file in [FASHION_CSV, SYNTH_CSV]:
        if not csv_file.exists():
            print(f"Warning: {csv_file.name} not found.")
            continue
            
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle empty strings and convert to lists
                garments = [g.strip() for g in row.get("garments", "").split(";") if g.strip()]
                colors = [c.strip() for c in row.get("colors", "").split(";") if c.strip()]
                pairs = [p.strip() for p in row.get("color_garment_pairs", "").split(";") if p.strip()]
                
                record = {
                    "image_id": row["image_id"],
                    "garments": garments,
                    "colors": colors,
                    "pairs": pairs,
                    "scene": row.get("scene", ""),
                    "objects": row.get("objects", "")
                }
                metadata_records.append(record)
                
    # Write JSONL
    with open(METADATA_JSONL, "w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record) + "\n")
            
    print(f"Successfully generated {len(metadata_records)} metadata records!")
    return {rec["image_id"]: rec for rec in metadata_records}

def load_model(model_type="clip"):
    print(f"\nLoading {model_type.upper()} model on {device}...")
    if model_type == "clip":
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")
        
    model.eval()
    return processor, model

@torch.no_grad()
def build_index():
    # 1. Build Metadata
    metadata_map = build_metadata_jsonl()
    
    # 2. Load Model
    processor, model = load_model(EMBEDDING_MODEL)
    
    # 3. Gather images (deterministic sorting)
    image_paths = sorted(list(IMAGES_DIR.glob("*.jpg")))
    
    print(f"\nFound {len(image_paths)} images. Extracting embeddings...")
    if not image_paths:
        print("Error: No images found. Check the curated_images directory.")
        return
        
    image_mapping = {}
    embeddings_list = []
    
    batch_size = 16
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        
        valid_images = []
        valid_ids = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
                valid_ids.append(path.stem)
            except Exception as e:
                print(f"Error opening {path.name}: {e}")
                
        if not valid_images:
            continue
            
        try:
            inputs = processor(images=valid_images, return_tensors="pt").to(device)
            image_embeds = model.get_image_features(**inputs)
            
            # Normalize vector to L2 norm = 1 (Required for Cosine Similarity via Inner Product)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            emb_np = image_embeds.cpu().numpy().astype('float32')
            
            # Append safely to keep mapping aligned perfectly
            for j, emb in enumerate(emb_np):
                vector_idx = len(embeddings_list)
                embeddings_list.append(emb)
                image_mapping[vector_idx] = valid_ids[j]
                
        except Exception as e:
            print(f"Error extracting batch embeddings: {e}")
            
    if not embeddings_list:
        print("No embeddings were generated.")
        return
        
    # Stack all embeddings into shape (N, dim)
    all_embeddings = np.vstack(embeddings_list)
    
    # 4. Build FAISS Index dynamically based on embedding dim
    dim = all_embeddings.shape[1]
    print(f"\nBuilding FAISS IndexFlatIP (dim={dim})...")
    
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    
    # 5. Sanity Check
    assert index.ntotal == len(image_mapping), f"Mismatch: {index.ntotal} vectors vs {len(image_mapping)} mappings"
    
    print(f"\nIndex created.\nTotal vectors in index: {index.ntotal}")
    
    # 6. Save Artifacts
    faiss.write_index(index, str(INDEX_FILE))
    
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(image_mapping, f, indent=2)
        
    print(f"\nSaved index to: {INDEX_FILE.name}")
    print(f"Saved mapping to: {MAPPING_FILE.name}")
    print("FAISS Indexing Complete!")

if __name__ == "__main__":
    build_index()
