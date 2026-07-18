import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from fashion_retrieval.common import iter_images, l2_normalize
from fashion_retrieval.indexer.config import IndexConfig
from fashion_retrieval.indexer.models import resolve_device


class ClipBaseline:
    def __init__(self, model_name: str = IndexConfig.clip_model_name, device: str = "auto"):
        self.device = resolve_device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_image_features(**inputs)
            chunks.append(features.detach().cpu().numpy().astype("float32"))
        return l2_normalize(np.vstack(chunks))

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        return l2_normalize(features.detach().cpu().numpy().astype("float32"))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--image-dir", default=str(IndexConfig.image_dir))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    paths = iter_images(Path(args.image_dir))
    baseline = ClipBaseline(device=args.device)
    image_embeddings = baseline.encode_images(paths)
    query_embedding = baseline.encode_text(args.query)
    scores = image_embeddings @ query_embedding
    for rank, idx in enumerate(np.argsort(-scores)[: args.top_k], start=1):
        print(f"{rank}. {scores[idx]:.3f} {paths[idx]}")


if __name__ == "__main__":
    main()

