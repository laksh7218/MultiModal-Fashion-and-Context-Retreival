import faiss
import torch
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "indexer"
DATA_DIR = BASE_DIR / "data"

device = "cuda" if torch.cuda.is_available() else "cpu"

class MultimodalRetriever:
    def __init__(self):
        print("Loading models and FAISS indices...")
        
        # Load indices
        try:
            self.clip_index = faiss.read_index(str(DATA_DIR / "clip_faiss_index.bin"))
        except RuntimeError:
            print("Warning: FAISS indices not found in data/.")
            
        # Load mapping
        self.idx_to_image = {}
        mapping_path = DATA_DIR / "clip_image_mapping.json"
        if mapping_path.exists():
            import json
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
                for idx, img in mapping.items():
                    self.idx_to_image[int(idx)] = img

        # Lazy loading for models
        self.clip_model = None
        self.clip_processor = None

    def _init_clip(self):
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def search_clip(self, query: str, k: int = 10):
        """System A: CLIP + FAISS Baseline"""
        self._init_clip()
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype('float32')

        scores, indices = self.clip_index.search(text_features, k)
        return [{"image_id": self.idx_to_image[idx], "score": float(score)} 
                for score, idx in zip(scores[0], indices[0])]

if __name__ == "__main__":
    retriever = MultimodalRetriever()
    # Test query
    query = "Red tie and white shirt in formal setting"
    print(f"\nQuery: {query}")
    print("\nSystem A (CLIP Baseline) Top 5:")
    try:
        print(retriever.search_clip(query, k=5))
    except Exception as e:
        print("Run build_index.py first!")
