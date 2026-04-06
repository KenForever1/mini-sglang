from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


@dataclass
class TokenizeResult:
    input_ids: torch.Tensor  # (seq_len,) int32
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    mrope_positions: torch.Tensor | None = None
    mrope_position_delta: int | None = None
    is_multimodal: bool = False


class TokenizeManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        processor: Any | None = None,
        model_config: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_config = model_config

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[TokenizeResult]:
        results: List[TokenizeResult] = []
        for msg in msgs:
            if msg.is_multimodal and self.processor is not None:
                result = self._tokenize_multimodal(msg)
            else:
                result = self._tokenize_text(msg)
            results.append(result)
        return results

    def _tokenize_text(self, msg: TokenizeMsg) -> TokenizeResult:
        if isinstance(msg.text, list):
            prompt = self.tokenizer.apply_chat_template(
                msg.text,
                tokenize=False,
                add_generation_prompt=True,
            )
            assert isinstance(prompt, str)
        else:
            prompt = msg.text
        input_ids: torch.Tensor = (  # type: ignore
            self.tokenizer.encode(prompt, return_tensors="pt")
        )
        return TokenizeResult(input_ids=input_ids.view(-1).to(torch.int32))

    def _tokenize_multimodal(self, msg: TokenizeMsg) -> TokenizeResult:
        from minisgl.multimodal.qwen3_vl_processor import Qwen3VLProcessor

        assert isinstance(msg.text, list), "Multimodal messages must be a list of dicts"
        vl_processor = Qwen3VLProcessor(self.processor)
        output = vl_processor.process_messages(msg.text)

        # Compute MRoPE positions
        mrope_positions = None
        mrope_position_delta = None
        if (
            self.model_config is not None
            and output.image_grid_thw is not None
            and self.model_config.image_token_id is not None
        ):
            from minisgl.multimodal.mrope import get_rope_index

            spatial_merge_size = 2  # default for Qwen3-VL
            if self.model_config.vision_config is not None:
                spatial_merge_size = self.model_config.vision_config.get(
                    "spatial_merge_size", 2
                )
            mrope_positions, mrope_position_delta = get_rope_index(
                input_ids=output.input_ids.to(torch.long),
                image_grid_thw=output.image_grid_thw,
                image_token_id=self.model_config.image_token_id,
                vision_start_token_id=self.model_config.vision_start_token_id,
                spatial_merge_size=spatial_merge_size,
            )

        return TokenizeResult(
            input_ids=output.input_ids,
            pixel_values=output.pixel_values,
            image_grid_thw=output.image_grid_thw,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
            is_multimodal=True,
        )
