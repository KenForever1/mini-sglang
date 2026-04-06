from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import ParallelLMHead, VocabParallelEmbedding
from minisgl.layers.vision import Qwen3VLVisionModel

from .base import BaseLLMModel
from .qwen3 import Qwen3Model

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3VLForConditionalGeneration(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        assert config.vision_config is not None, "vision_config is required for Qwen3-VL"

        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

        self.image_token_id = config.image_token_id
        self.hidden_size = config.hidden_size

        # Deepstack: decoder layers 0..N-1 each get one slice of deepstack embeddings
        ds_indexes = config.vision_config.get("deepstack_visual_indexes", [])
        self.num_deepstack = len(ds_indexes)
        self.deepstack_inject_layers = range(self.num_deepstack) if self.num_deepstack > 0 else None

        super().__init__()

    def _separate_deepstack_embeds(
        self, image_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split vision encoder output into input embeds and deepstack embeds.

        Vision encoder returns (merged_tokens, hidden_size * (1 + num_deepstack)).
        First hidden_size columns are the main input embeds; the rest are deepstack.
        """
        if self.num_deepstack == 0:
            return image_embeds, None
        input_embeds = image_embeds[:, : self.hidden_size]
        deepstack_embeds = image_embeds[:, self.hidden_size :]
        return input_embeds, deepstack_embeds

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        input_ids = batch.input_ids

        if batch.has_multimodal and batch.is_prefill:
            # Get text embeddings for all tokens in the extend portion
            safe_ids = input_ids.clamp(min=0, max=self.model.embed_tokens.num_embeddings - 1)
            hidden = self.model.embed_tokens.forward(safe_ids)

            # Check which tokens in the extend are image placeholders.
            # With prefix caching, image tokens may already be in the cached
            # prefix and absent from the current extend portion.
            mask = input_ids == self.image_token_id
            input_deepstack_embeds = None

            if mask.any() and batch.pixel_values is not None:
                # Run vision encoder only when there are image tokens to fill
                image_embeds = self.visual(batch.pixel_values, batch.image_grid_thw)
                vision_embeds, deepstack_embeds = self._separate_deepstack_embeds(image_embeds)

                hidden[mask] = vision_embeds.to(hidden.dtype)

                if deepstack_embeds is not None:
                    input_deepstack_embeds = torch.zeros(
                        hidden.shape[0],
                        self.hidden_size * self.num_deepstack,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    input_deepstack_embeds[mask] = deepstack_embeds.to(hidden.dtype)

            # Forward through decoder (MRoPE positions are still used correctly)
            output = self.model.forward(
                input_ids,
                inputs_embeds=hidden,
                deepstack_embeds=input_deepstack_embeds,
                deepstack_inject_layers=self.deepstack_inject_layers,
            )
        else:
            # Text-only or decode phase: standard path
            output = self.model.forward(input_ids)

        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3VLForConditionalGeneration"]
