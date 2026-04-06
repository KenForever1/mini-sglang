from __future__ import annotations

from typing import Tuple

import torch


def get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, int]:
    """Compute MRoPE (3D) position ids for Qwen3-VL image-only inputs.

    This is a simplified version of sglang's MRotaryEmbedding.get_rope_index(),
    supporting only image inputs (no video/audio).

    Args:
        input_ids: (seq_len,) token ids (int64).
        image_grid_thw: (num_images, 3) each row is (t, h, w) in patch-grid units.
        image_token_id: the token id used as image placeholder.
        vision_start_token_id: the token id marking the start of a vision segment.
        spatial_merge_size: spatial merge factor (typically 2).

    Returns:
        mrope_positions: (3, seq_len) int64 position ids for T/H/W dimensions.
        mrope_position_delta: int, equal to positions.max() + 1 - seq_len.
    """
    seq_len = input_ids.shape[0]

    if image_grid_thw is None or image_grid_thw.numel() == 0:
        # No images: all three dims get the same monotonic positions
        pos = torch.arange(seq_len, dtype=torch.long)
        return pos.unsqueeze(0).expand(3, -1).contiguous(), 0

    input_tokens = input_ids.tolist()

    # Find all vision_start positions and identify which are images
    vision_start_indices = []
    for idx, tok in enumerate(input_tokens):
        if tok == vision_start_token_id:
            vision_start_indices.append(idx)

    image_index = 0
    llm_pos_ids_list = []
    st = 0

    for vs_idx in vision_start_indices:
        # The token after vision_start should be image_token_id
        if vs_idx + 1 >= seq_len or input_tokens[vs_idx + 1] != image_token_id:
            continue

        t = image_grid_thw[image_index][0].item()
        h = image_grid_thw[image_index][1].item()
        w = image_grid_thw[image_index][2].item()
        image_index += 1

        llm_grid_t = t
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size

        # Find the actual position of image_token_id in the token list from st
        ed = input_tokens.index(image_token_id, st)
        text_len = ed - st

        st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0

        # Text tokens before this image: monotonic positions replicated across 3 dims
        if text_len > 0:
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        # Image tokens: 3D positions (t_index, h_index, w_index)
        t_index = (
            torch.arange(llm_grid_t)
            .view(-1, 1)
            .expand(-1, llm_grid_h * llm_grid_w)
            .flatten()
        )
        h_index = (
            torch.arange(llm_grid_h)
            .view(1, -1, 1)
            .expand(llm_grid_t, -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w)
            .view(1, 1, -1)
            .expand(llm_grid_t, llm_grid_h, -1)
            .flatten()
        )
        llm_pos_ids_list.append(
            torch.stack([t_index, h_index, w_index]) + text_len + st_idx
        )
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    # Remaining text tokens after the last image
    if st < seq_len:
        st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
        text_len = seq_len - st
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
        )

    mrope_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    mrope_position_delta = mrope_positions.max().item() + 1 - seq_len
    return mrope_positions, mrope_position_delta
