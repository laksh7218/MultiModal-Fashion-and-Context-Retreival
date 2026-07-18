from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexConfig:
    data_dir: Path = Path("fashion_retrieval/data")
    image_dir: Path = Path("fashion_retrieval/data/images")
    output_dir: Path = Path("fashion_retrieval/outputs")
    captions_path: Path = Path("fashion_retrieval/outputs/captions.jsonl")
    metadata_path: Path = Path("fashion_retrieval/outputs/metadata.jsonl")
    scene_scores_path: Path = Path("fashion_retrieval/outputs/scene_scores.jsonl")
    embeddings_path: Path = Path("fashion_retrieval/outputs/image_embeddings.npy")
    faiss_index_path: Path = Path("fashion_retrieval/outputs/faiss.index")
    numpy_index_path: Path = Path("fashion_retrieval/outputs/numpy_index.npz")
    image_manifest_path: Path = Path("fashion_retrieval/outputs/image_manifest.jsonl")
    blip_model_name: str = "Salesforce/blip-itm-base-coco"
    caption_model_name: str = "Salesforce/blip-image-captioning-base"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    batch_size: int = 8
    caption_prompt: str = "a photo of a person wearing"
    candidate_pool: int = 50


def ensure_output_dirs(config: IndexConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.image_dir.mkdir(parents=True, exist_ok=True)
