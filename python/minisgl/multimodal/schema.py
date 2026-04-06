from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultimodalData:
    """Minimal multimodal data container for Qwen3-VL image inputs."""

    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    mrope_positions: torch.Tensor | None = None
    mrope_position_delta: int | None = None
