import argparse
from pathlib import Path

from fashion_retrieval.common import iter_images, write_jsonl
from fashion_retrieval.indexer.config import IndexConfig, ensure_output_dirs
from fashion_retrieval.indexer.models import BlipCaptioner


def generate_captions(config: IndexConfig) -> list[dict]:
    ensure_output_dirs(config)
    image_paths = iter_images(config.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {config.image_dir}")

    captioner = BlipCaptioner(config.caption_model_name, config.device)
    rows: list[dict] = []
    for start in range(0, len(image_paths), config.batch_size):
        batch_paths = image_paths[start : start + config.batch_size]
        captions = captioner.caption(batch_paths, config.caption_prompt)
        for path, caption in zip(batch_paths, captions):
            rows.append({"image_id": path.stem, "image_path": str(path), "caption": caption})
        print(f"captioned {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}")

    write_jsonl(config.captions_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(IndexConfig.image_dir))
    parser.add_argument("--output", default=str(IndexConfig.captions_path))
    parser.add_argument("--batch-size", type=int, default=IndexConfig.batch_size)
    parser.add_argument("--device", default=IndexConfig.device)
    args = parser.parse_args()

    config = IndexConfig(
        image_dir=Path(args.image_dir),
        captions_path=Path(args.output),
        batch_size=args.batch_size,
        device=args.device,
    )
    generate_captions(config)


if __name__ == "__main__":
    main()
