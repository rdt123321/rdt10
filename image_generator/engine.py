from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sd-turbo")
OUTPUT_DIR = Path("outputs")


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 1
    guidance_scale: float = 0.0
    seed: int = -1


def sanitize_prompt(value: str) -> str:
    prompt = value.strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    return prompt


def create_outputs_zip(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(path for path in output_dir.glob("*.png") if path.is_file())

    if not image_files:
        raise ValueError("No generated PNG files found in outputs/. Generate an image first.")

    zip_name = f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = output_dir / zip_name

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for image_file in image_files:
            archive.write(image_file, arcname=image_file.name)

    return zip_path


class ImageGenerator:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._pipeline: Optional[Any] = None

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            self._pipeline = self._pipeline.to(self.device)
            if self.device == "cuda":
                self._pipeline.enable_attention_slicing()
        return self._pipeline

    def generate(self, config: GenerationConfig) -> Path:
        import torch

        pipe = self._load_pipeline()
        prompt = sanitize_prompt(config.prompt)

        generator: Optional[torch.Generator] = None
        if config.seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )

        image = result.images[0]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = OUTPUT_DIR / f"{stamp}.png"
        image.save(file_path)
        return file_path
