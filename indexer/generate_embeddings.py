import argparse
from pathlib import Path

import numpy as np

from fashion_retrieval.common import iter_images, write_jsonl
from fashion_retrieval.indexer.config import IndexConfig, ensure_output_dirs
from fashion_retrieval.indexer.models import BlipFashionEncoder


def generate_embeddings(config: IndexConfig) -> np.ndarray:
    ensure_output_dirs(config)
    image_paths = iter_images(config.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {config.image_dir}")

    encoder = BlipFashionEncoder(config.blip_model_name, config.device)
    chunks: list[np.ndarray] = []
    manifest: list[dict] = []
    for start in range(0, len(image_paths), config.batch_size):
        batch_paths = image_paths[start : start + config.batch_size]
        chunks.append(encoder.encode_images(batch_paths))
        for path in batch_paths:
            manifest.append({"image_id": path.stem, "image_path": str(path)})
        print(f"embedded {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}")

    embeddings = np.vstack(chunks).astype("float32")
    np.save(config.embeddings_path, embeddings)
    write_jsonl(config.image_manifest_path, manifest)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(IndexConfig.image_dir))
    parser.add_argument("--output", default=str(IndexConfig.embeddings_path))
    parser.add_argument("--manifest", default=str(IndexConfig.image_manifest_path))
    parser.add_argument("--batch-size", type=int, default=IndexConfig.batch_size)
    parser.add_argument("--device", default=IndexConfig.device)
    args = parser.parse_args()

    config = IndexConfig(
        image_dir=Path(args.image_dir),
        embeddings_path=Path(args.output),
        image_manifest_path=Path(args.manifest),
        batch_size=args.batch_size,
        device=args.device,
    )
    generate_embeddings(config)


if __name__ == "__main__":
    main()

