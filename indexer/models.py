from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipModel, BlipProcessor

from fashion_retrieval.common import l2_normalize


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class BlipFashionEncoder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
            self.device
        )
        features = self.model.get_image_features(**inputs)
        return l2_normalize(features.detach().cpu().numpy().astype("float32"))

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            self.device
        )
        features = self.model.get_text_features(**inputs)
        return l2_normalize(features.detach().cpu().numpy().astype("float32"))


class BlipCaptioner:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def caption(self, image_paths: list[Path], prompt: str = "") -> list[str]:
        captions: list[str] = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            kwargs = {"images": image, "return_tensors": "pt"}
            if prompt:
                kwargs["text"] = prompt
            inputs = self.processor(**kwargs).to(self.device)
            generated = self.model.generate(**inputs, max_new_tokens=35)
            captions.append(self.processor.decode(generated[0], skip_special_tokens=True))
        return captions

