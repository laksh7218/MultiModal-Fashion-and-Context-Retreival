import argparse
from pathlib import Path

import numpy as np

from fashion_retrieval.common import iter_images, write_jsonl
from fashion_retrieval.indexer.config import IndexConfig, ensure_output_dirs
from fashion_retrieval.indexer.models import BlipFashionEncoder


SCENE_ANCHORS = {
    "office": [
        "a person wearing professional clothing inside a modern office",
        "business attire in an office interior",
        "a conference room or meeting room",
        "a workspace with a desk and laptop",
        "a person in formal clothes at work",
    ],
    "park": [
        "a person outdoors in a park",
        "a person sitting on a park bench",
        "a green park with trees",
        "a person walking in a garden",
        "casual clothing in a park",
    ],
    "street": [
        "a person walking on an urban street",
        "a city sidewalk with buildings",
        "a casual outfit for a city walk",
        "street style fashion outdoors",
        "a person on a road or crosswalk",
    ],
    "home": [
        "a person indoors at home",
        "a living room interior",
        "casual outfit in a home setting",
        "a sofa in a home interior",
        "a bedroom or home interior",
    ],
}


def _aggregate_anchor_scores(scores: np.ndarray, anchor_counts: dict[str, int]) -> dict[str, float]:
    result: dict[str, float] = {}
    start = 0
    for environment, count in anchor_counts.items():
        env_scores = scores[start : start + count]
        result[environment] = float(np.max(env_scores))
        start += count
    return result


def generate_scene_scores(config: IndexConfig) -> list[dict]:
    ensure_output_dirs(config)
    image_paths = iter_images(config.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {config.image_dir}")

    encoder = BlipFashionEncoder(config.blip_model_name, config.device)
    anchors = [anchor for values in SCENE_ANCHORS.values() for anchor in values]
    anchor_counts = {environment: len(values) for environment, values in SCENE_ANCHORS.items()}
    text_embeddings = encoder.encode_texts(anchors)

    rows: list[dict] = []
    for start in range(0, len(image_paths), config.batch_size):
        batch_paths = image_paths[start : start + config.batch_size]
        image_embeddings = encoder.encode_images(batch_paths)
        similarities = image_embeddings @ text_embeddings.T
        for path, scores in zip(batch_paths, similarities):
            environment_scores = _aggregate_anchor_scores(scores, anchor_counts)
            best_environment = max(environment_scores, key=environment_scores.get)
            rows.append(
                {
                    "image_id": path.stem,
                    "image_path": str(path),
                    "environment_scores": environment_scores,
                    "predicted_environment": best_environment,
                }
            )
        print(f"scene-scored {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}")

    write_jsonl(config.scene_scores_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(IndexConfig.image_dir))
    parser.add_argument("--output", default=str(IndexConfig.scene_scores_path))
    parser.add_argument("--batch-size", type=int, default=IndexConfig.batch_size)
    parser.add_argument("--device", default=IndexConfig.device)
    args = parser.parse_args()

    config = IndexConfig(
        image_dir=Path(args.image_dir),
        scene_scores_path=Path(args.output),
        batch_size=args.batch_size,
        device=args.device,
    )
    generate_scene_scores(config)


if __name__ == "__main__":
    main()
