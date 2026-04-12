from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List

import torch
from PIL import Image
from minisgl.utils import init_logger, maybe_log_perf

from .image_io import load_image

logger = init_logger(__name__)


@dataclass
class ProcessorOutput:
    input_ids: torch.Tensor  # (seq_len,) int32
    pixel_values: torch.Tensor | None  # (num_patches, patch_dim) float
    image_grid_thw: torch.Tensor | None  # (num_images, 3) int


class Qwen3VLProcessor:
    """Thin wrapper around HuggingFace AutoProcessor for Qwen3-VL."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def process_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> ProcessorOutput:
        """Parse OpenAI-style messages with images and produce model inputs.

        Args:
            messages: List of {"role": ..., "content": ...} dicts. Content can be
                a string or a list of {"type": "text"/"image_url", ...} parts.

        Returns:
            ProcessorOutput with input_ids, pixel_values, image_grid_thw.
        """
        total_start = time.perf_counter()
        images: List[Image.Image] = []
        # Build a messages list compatible with processor.apply_chat_template
        processed_messages: List[Dict[str, Any]] = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            if isinstance(content, str):
                processed_messages.append({"role": role, "content": content})
                continue

            # content is a list of parts
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif part["type"] == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                    img = load_image(url)
                    images.append(img)
                    parts.append({"type": "image"})
            processed_messages.append({"role": role, "content": parts})

        # Use processor to generate text from chat template
        template_start = time.perf_counter()
        text = self.processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        maybe_log_perf(
            logger,
            f"qwen3_vl.apply_chat_template messages={len(processed_messages)} images={len(images)}",
            template_start,
        )

        # Call processor with images + text
        processor_start = time.perf_counter()
        if images:
            inputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
            )
        maybe_log_perf(
            logger,
            f"qwen3_vl.hf_processor images={len(images)} prompt_chars={len(text)}",
            processor_start,
        )

        input_ids = inputs["input_ids"].squeeze(0).to(torch.int32)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0) if pixel_values.dim() > 2 else pixel_values
        if image_grid_thw is not None:
            image_grid_thw = (
                image_grid_thw.squeeze(0) if image_grid_thw.dim() > 2 else image_grid_thw
            )

        maybe_log_perf(
            logger,
            f"qwen3_vl.process_messages images={len(images)} tokens={input_ids.numel()}",
            total_start,
        )
        return ProcessorOutput(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
