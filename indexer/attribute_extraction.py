from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image


COLOR_RGB = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (130, 130, 130),
    "red": (210, 45, 45),
    "blue": (55, 95, 200),
    "yellow": (235, 205, 40),
    "green": (45, 150, 80),
    "pink": (225, 90, 160),
    "purple": (125, 70, 165),
    "brown": (130, 85, 45),
    "orange": (230, 120, 35),
    "beige": (205, 180, 135),
}

COLOR_TERMS = {
    "bright yellow": "yellow",
    "mustard": "yellow",
    "navy": "blue",
    "sky blue": "blue",
    "denim": "blue",
    "crimson": "red",
    "maroon": "red",
    "cream": "beige",
    "tan": "beige",
    "grey": "gray",
    **{color: color for color in COLOR_RGB},
}

GARMENT_TERMS = {
    "raincoat": "raincoat",
    "trench coat": "coat",
    "coat": "coat",
    "jacket": "jacket",
    "blazer": "blazer",
    "suit jacket": "blazer",
    "suit": "suit",
    "shirt": "shirt",
    "button down": "shirt",
    "button-down": "shirt",
    "t-shirt": "t-shirt",
    "tee": "t-shirt",
    "hoodie": "hoodie",
    "sweater": "sweater",
    "dress": "dress",
    "skirt": "skirt",
    "jeans": "jeans",
    "pants": "pants",
    "trousers": "pants",
    "tie": "tie",
}

CONTEXT_TERMS = {
    "office": "office",
    "workplace": "office",
    "conference": "office",
    "meeting room": "office",
    "street": "street",
    "city": "street",
    "urban": "street",
    "sidewalk": "street",
    "park": "park",
    "bench": "park",
    "garden": "park",
    "home": "home",
    "living room": "home",
    "bedroom": "home",
    "indoor": "indoor",
    "inside": "indoor",
    "outdoor": "outdoor",
    "outside": "outdoor",
}

OBJECT_TERMS = {
    "bench": "bench",
    "park bench": "bench",
    "chair": "chair",
    "desk": "desk",
    "table": "table",
    "laptop": "laptop",
    "computer": "laptop",
    "tree": "tree",
    "sofa": "sofa",
    "couch": "sofa",
    "road": "road",
    "sidewalk": "sidewalk",
    "building": "building",
    "bag": "bag",
    "umbrella": "umbrella",
}

STYLE_TERMS = {
    "formal": "formal",
    "professional": "formal",
    "business": "formal",
    "office wear": "formal",
    "casual": "casual",
    "weekend": "casual",
    "relaxed": "casual",
    "streetwear": "casual",
    "sporty": "sporty",
    "elegant": "formal",
}

QUERY_TAG_TERMS = {
    "business attire": "business_attire",
    "business": "business_attire",
    "professional": "business_attire",
    "weekend": "weekend_outfit",
    "city walk": "city_walk",
    "walking": "walking",
    "sitting": "sitting",
    "formal setting": "formal_setting",
}


def _find_terms(text: str, vocabulary: dict[str, str]) -> list[str]:
    text = text.lower()
    matches: list[str] = []
    for phrase, label in vocabulary.items():
        pattern = r"(?<![a-z])" + re.escape(phrase.lower()) + r"(?![a-z])"
        if re.search(pattern, text):
            matches.append(label)
    return sorted(set(matches))


def _find_color_garment_pairs(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    pairs: set[str] = set()
    for color_phrase, color in COLOR_TERMS.items():
        color_pattern = re.escape(color_phrase.lower().replace("-", " "))
        for garment_phrase, garment in GARMENT_TERMS.items():
            garment_pattern = re.escape(garment_phrase.lower().replace("-", " "))
            pattern = (
                r"(?<![a-z])"
                + color_pattern
                + r"(?:\s+\w+){0,2}\s+"
                + garment_pattern
                + r"(?![a-z])"
            )
            if re.search(pattern, text):
                pairs.add(f"{color}:{garment}")
    return sorted(pairs)


def estimate_center_color(image_path: Path) -> str | None:
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return None

    width, height = image.size
    left = int(width * 0.2)
    top = int(height * 0.15)
    right = int(width * 0.8)
    bottom = int(height * 0.85)
    crop = image.crop((left, top, right, bottom)).resize((96, 96))
    pixels = np.asarray(crop, dtype=np.float32).reshape(-1, 3)

    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)
    keep = (brightness > 25) & (brightness < 245) & (saturation > 12)
    if keep.any():
        pixels = pixels[keep]

    palette = np.asarray(list(COLOR_RGB.values()), dtype=np.float32)
    labels = list(COLOR_RGB.keys())
    distances = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    return Counter(labels[idx] for idx in nearest).most_common(1)[0][0]


def extract_attributes_from_text(text: str) -> dict[str, list[str]]:
    attrs = {
        "colors": _find_terms(text, COLOR_TERMS),
        "garments": _find_terms(text, GARMENT_TERMS),
        "contexts": _find_terms(text, CONTEXT_TERMS),
        "objects": _find_terms(text, OBJECT_TERMS),
        "styles": _find_terms(text, STYLE_TERMS),
        "color_garment_pairs": _find_color_garment_pairs(text),
        "query_tags": _find_terms(text, QUERY_TAG_TERMS),
    }
    attrs["query_tags"] = sorted(
        set(
            attrs["query_tags"]
            + attrs["colors"]
            + attrs["garments"]
            + attrs["contexts"]
            + attrs["objects"]
            + attrs["styles"]
        )
    )
    return attrs


def extract_image_attributes(
    caption: str,
    image_path: Path | None = None,
    external_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs = extract_attributes_from_text(caption)
    external_metadata = external_metadata or {}

    if external_metadata.get("garments"):
        attrs["garments"] = sorted(set(attrs["garments"] + external_metadata["garments"]))
    if external_metadata.get("garment"):
        attrs["garments"] = sorted(set(attrs["garments"] + [external_metadata["garment"]]))
    if external_metadata.get("contexts"):
        attrs["contexts"] = sorted(set(attrs["contexts"] + external_metadata["contexts"]))
    if external_metadata.get("context"):
        attrs["contexts"] = sorted(set(attrs["contexts"] + [external_metadata["context"]]))
    if external_metadata.get("environment"):
        attrs["contexts"] = sorted(set(attrs["contexts"] + [external_metadata["environment"]]))
    if external_metadata.get("object"):
        attrs["objects"] = sorted(set(attrs["objects"] + [external_metadata["object"]]))
    if external_metadata.get("objects"):
        attrs["objects"] = sorted(set(attrs["objects"] + external_metadata["objects"]))
    if external_metadata.get("color_garment_pairs"):
        attrs["color_garment_pairs"] = sorted(
            set(attrs["color_garment_pairs"] + external_metadata["color_garment_pairs"])
        )
    if external_metadata.get("styles"):
        attrs["styles"] = sorted(set(attrs["styles"] + external_metadata["styles"]))
    if external_metadata.get("style"):
        attrs["styles"] = sorted(set(attrs["styles"] + [external_metadata["style"]]))
    if external_metadata.get("query_tags"):
        attrs["query_tags"] = sorted(set(attrs["query_tags"] + external_metadata["query_tags"]))

    image_color = estimate_center_color(image_path) if image_path else None
    colors = set(attrs["colors"])
    if external_metadata.get("colors"):
        colors.update(external_metadata["colors"])
    if external_metadata.get("color"):
        colors.add(external_metadata["color"])
    if image_color:
        colors.add(image_color)
    attrs["colors"] = sorted(colors)
    attrs["query_tags"] = sorted(
        set(
            attrs["query_tags"]
            + attrs["colors"]
            + attrs["garments"]
            + attrs["contexts"]
            + attrs["objects"]
            + attrs["styles"]
        )
    )
    return attrs
