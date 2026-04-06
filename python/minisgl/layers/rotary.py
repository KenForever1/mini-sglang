from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, List, Tuple

import torch

from .base import StateLessOP


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (neox style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings using torch ops. x: (..., rotary_dim)."""
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


class MRotaryEmbedding(StateLessOP):
    """Rotary Embedding with Multimodal Sections (MRoPE) for Qwen3-VL.

    When positions is 1D, delegates to flashinfer fast path.
    When positions is (3, seq_len), applies torch-native MRoPE.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: List[int],
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.mrope_section = list(mrope_section)

        # Auto-correct mrope_section sum
        expected_sum = rotary_dim // 2
        actual_sum = sum(self.mrope_section)
        if actual_sum != expected_sum and actual_sum > 0:
            scale_factor = expected_sum / actual_sum
            self.mrope_section = [
                max(1, int(s * scale_factor)) for s in self.mrope_section
            ]
            current_sum = sum(self.mrope_section)
            if current_sum != expected_sum:
                self.mrope_section[-1] += expected_sum - current_sum

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self._apply_rope_flashinfer = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            # 1D text-only or decode: use flashinfer fast path
            self._apply_rope_flashinfer(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self._cos_sin_cache,
            )
            return query, key

        # 2D MRoPE path: positions shape (3, seq_len)
        assert positions.shape[0] == 3, f"Expected (3, seq_len), got {positions.shape}"
        return self._forward_mrope(positions, query, key)

    def _forward_mrope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # positions: (3, seq_len)
        # cos_sin_cache: (max_pos, rotary_dim) where first half is cos, second half is sin
        # Cast to query dtype to avoid float32 promotion from the cos/sin cache
        cos_sin = self._cos_sin_cache.to(device=query.device, dtype=query.dtype)[positions]  # (3, seq_len, rotary_dim)
        cos, sin = cos_sin.chunk(2, dim=-1)  # each (3, seq_len, rotary_dim/2)

        # Split by mrope_section and pick the correct dimension for each section
        cos = torch.cat(
            [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
            dim=-1,
        )  # (seq_len, rotary_dim/2)
        sin = torch.cat(
            [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
            dim=-1,
        )  # (seq_len, rotary_dim/2)

        # Double cos/sin for neox-style rotation
        cos = torch.cat([cos, cos], dim=-1)  # (seq_len, rotary_dim)
        sin = torch.cat([sin, sin], dim=-1)  # (seq_len, rotary_dim)

        # Apply to query
        seq_len = positions.shape[1]
        q_shape = query.shape
        query = query.view(seq_len, -1, self.head_size)
        q_rot = query[..., : self.rotary_dim]
        q_pass = query[..., self.rotary_dim :]
        cos_q = cos.unsqueeze(1)  # (seq_len, 1, rotary_dim)
        sin_q = sin.unsqueeze(1)
        q_rot = _apply_rotary_emb_torch(q_rot, cos_q, sin_q)
        query = torch.cat((q_rot, q_pass), dim=-1).reshape(q_shape)

        # Apply to key
        k_shape = key.shape
        key = key.view(seq_len, -1, self.head_size)
        k_rot = key[..., : self.rotary_dim]
        k_pass = key[..., self.rotary_dim :]
        k_rot = _apply_rotary_emb_torch(k_rot, cos_q, sin_q)
        key = torch.cat((k_rot, k_pass), dim=-1).reshape(k_shape)

        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    mrope_section: List[int] | None = None,
) -> RotaryEmbedding | MRotaryEmbedding:
    # Check if mrope_section is in rope_scaling
    if mrope_section is None and rope_scaling is not None:
        mrope_section = rope_scaling.get("mrope_section", None)

    if rope_scaling is None:
        if mrope_section:
            return MRotaryEmbedding(head_dim, rotary_dim, max_position, base, mrope_section)
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

    match rope_scaling["rope_type"]:
        case "default":
            if mrope_section:
                return MRotaryEmbedding(head_dim, rotary_dim, max_position, base, mrope_section)
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            if mrope_section:
                return MRotaryEmbedding(
                    head_dim, rotary_dim, max_position, base, mrope_section, post_process
                )
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            if mrope_section:
                return MRotaryEmbedding(
                    head_dim, rotary_dim, max_position, base, mrope_section, post_process
                )
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding | MRotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "MRotaryEmbedding", "set_rope_device"]
