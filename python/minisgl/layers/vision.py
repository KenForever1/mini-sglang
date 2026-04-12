"""Torch-native vision encoder components for Qwen3-VL.

All attention uses F.scaled_dot_product_attention (no custom kernels).
No tensor parallelism in the first phase (TP=1 only).
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minisgl.utils import init_logger, maybe_log_perf

logger = init_logger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb_vision(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class VisionRotaryEmbedding(nn.Module):
    """Simple rotary embedding for vision transformer (half-dim only)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, seqlen: int) -> torch.Tensor:
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        t = torch.arange(seqlen, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        return freqs  # (seqlen, dim/2)


class Qwen3VLVisionPatchEmbed(nn.Module):
    """3D convolution patch embedding for Qwen3-VL.

    When kernel_size == stride (the default for patch embedding), the Conv3d
    degenerates into a simple linear projection over flattened patches.  We
    store the weight/bias in Conv3d format for checkpoint compatibility but
    execute via F.linear to avoid cuDNN algorithm-selection overhead that can
    take tens of seconds on the first call with a new shape.
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        # Keep Conv3d for weight storage so existing checkpoints load directly.
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        # Flatten each patch into a vector: (N, C * T * H * W)
        flat_dim = (
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        hidden_states = hidden_states.view(-1, flat_dim).to(dtype=target_dtype)

        # Conv3d weight shape: (embed_dim, C, T, H, W) → reshape to (embed_dim, flat_dim)
        weight = self.proj.weight.view(self.embed_dim, flat_dim)
        hidden_states = F.linear(hidden_states, weight, self.proj.bias)
        return hidden_states


class Qwen3VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        act_fns = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
        }
        self.act = act_fns.get(hidden_act, nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VisionAttention(nn.Module):
    """Vision attention using F.scaled_dot_product_attention with
    segment-level processing via cu_seqlens."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: (total_tokens, embed_dim) — flat concatenation of all segments
            cu_seqlens: (num_segments + 1,) cumulative lengths
            position_embeddings: (cos, sin) each (total_tokens, head_dim)
        """
        S = x.shape[0]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        # cos/sin: (S, head_dim)
        cos_emb = cos.unsqueeze(1)  # (S, 1, head_dim)
        sin_emb = sin.unsqueeze(1)

        q = q.view(S, self.num_heads, self.head_dim)
        k = k.view(S, self.num_heads, self.head_dim)
        q = _apply_rotary_emb_vision(q, cos_emb, sin_emb)
        k = _apply_rotary_emb_vision(k, cos_emb, sin_emb)

        # Process each segment independently to respect attention boundaries
        v = v.view(S, self.num_heads, self.head_dim)
        # Ensure uniform dtype after rotary (cos/sin may upcast to float32)
        dtype = v.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        num_segments = cu_seqlens.shape[0] - 1
        output = torch.empty_like(q)

        for i in range(num_segments):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seg_q = q[start:end].transpose(0, 1).unsqueeze(0)  # (1, H, L, D)
            seg_k = k[start:end].transpose(0, 1).unsqueeze(0)
            seg_v = v[start:end].transpose(0, 1).unsqueeze(0)
            seg_out = F.scaled_dot_product_attention(seg_q, seg_k, seg_v)
            output[start:end] = seg_out.squeeze(0).transpose(0, 1)

        output = output.reshape(S, -1)
        return self.proj(output)


class Qwen3VisionBlock(nn.Module):
    """Pre-norm vision transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        hidden_act: str = "silu",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Qwen3VisionAttention(embed_dim=dim, num_heads=num_heads)
        self.mlp = Qwen3VisionMLP(
            dim, intermediate_dim, hidden_act=hidden_act, bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: (total_tokens, dim) flat sequence
        """
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchMerger(nn.Module):
    """Merges spatial patches after the vision transformer."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else context_dim,
            eps=norm_eps,
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x


def _rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
    """Compute 2D position ids for rotary embedding within a single frame.

    Returns (h*w, 2) tensor of (h_pos, w_pos) reordered so that adjacent
    spatial_merge_size x spatial_merge_size patches are grouped together.
    """
    m = spatial_merge_size
    hh, ww = h // m, w // m

    h_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
    w_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
    ids = torch.stack([h_ids, w_ids], dim=-1)  # (h, w, 2)
    ids = (
        ids.view(hh, m, ww, m, 2).permute(0, 2, 1, 3, 4).reshape(hh * ww, m * m, 2)
    )
    return ids.view(-1, 2)  # (h*w, 2)


def compute_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths from grid_thw.

    Each temporal frame is treated as a separate attention segment, matching
    the reference sglang implementation.
    """
    cu = [0]
    for t, h, w in grid_thw:
        frame_len = h.item() * w.item()
        for _ in range(t.item()):
            cu.append(cu[-1] + frame_len)
    return torch.tensor(cu, dtype=torch.int32)


class Qwen3VLVisionModel(nn.Module):
    """Complete Qwen3-VL vision encoder with deepstack support."""

    def __init__(self, vision_config: dict) -> None:
        super().__init__()
        self.hidden_size = vision_config["hidden_size"]
        self.num_heads = vision_config["num_heads"]
        self.patch_size = vision_config["patch_size"]
        self.spatial_merge_size = vision_config.get("spatial_merge_size", 2)
        self.temporal_patch_size = vision_config.get("temporal_patch_size", 2)
        self.num_position_embeddings = vision_config.get(
            "num_position_embeddings", 2304
        )
        depth = vision_config.get("depth", 27)
        intermediate_size = vision_config.get("intermediate_size", 4304)
        hidden_act = vision_config.get("hidden_act", "gelu_pytorch_tanh")
        in_channels = vision_config.get("in_channels", 3)
        out_hidden_size = vision_config.get("out_hidden_size", 3584)
        norm_eps = 1e-6

        self.deepstack_visual_indexes = vision_config.get(
            "deepstack_visual_indexes", [8, 16, 24]
        )
        self.out_hidden_size = out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=in_channels,
            embed_dim=self.hidden_size,
        )
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=intermediate_size,
                    hidden_act=hidden_act,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )

        self.merger = Qwen3VLVisionPatchMerger(
            dim=out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            norm_eps=norm_eps,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    dim=out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_eps=norm_eps,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings for all patches."""
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()
            base = _rot_pos_ids(h, w, self.spatial_merge_size)
            if t > 1:
                base = base.repeat(t, 1)
            pos_ids.append(base)

        pos_ids = torch.cat(pos_ids, dim=0)  # (total_patches, 2)
        max_grid_size = grid_thw[:, 1:].max().item()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        # Index into the rotary embedding table with h and w positions
        emb = rotary_pos_emb_full[pos_ids]  # (total_patches, 2, head_dim/4)
        emb = emb.flatten(1)  # (total_patches, head_dim/2)
        return emb

    def _fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of absolute 2D position embeddings."""
        num_grid = int(self.num_position_embeddings**0.5)
        m = self.spatial_merge_size

        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()
            h_idxs = np.linspace(0, num_grid - 1, h)
            w_idxs = np.linspace(0, num_grid - 1, w)

            hf = h_idxs.astype(int)
            wf = w_idxs.astype(int)
            hc = (hf + 1).clip(max=num_grid - 1)
            wc = (wf + 1).clip(max=num_grid - 1)

            dh = h_idxs - hf
            dw = w_idxs - wf

            idx_list[0].extend(
                ((hf * num_grid)[None].T + wf[None]).flatten().tolist() * t
            )
            idx_list[1].extend(
                ((hf * num_grid)[None].T + wc[None]).flatten().tolist() * t
            )
            idx_list[2].extend(
                ((hc * num_grid)[None].T + wf[None]).flatten().tolist() * t
            )
            idx_list[3].extend(
                ((hc * num_grid)[None].T + wc[None]).flatten().tolist() * t
            )

            weight_list[0].extend(
                ((1 - dh)[None].T * (1 - dw)[None]).flatten().tolist() * t
            )
            weight_list[1].extend(
                ((1 - dh)[None].T * dw[None]).flatten().tolist() * t
            )
            weight_list[2].extend(
                (dh[None].T * (1 - dw)[None]).flatten().tolist() * t
            )
            weight_list[3].extend(
                (dh[None].T * dw[None]).flatten().tolist() * t
            )

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        result = torch.zeros(
            len(idx_list[0]), self.hidden_size, device=device, dtype=dtype
        )
        for idxs, weights in zip(idx_list, weight_list):
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)
            w_t = torch.tensor(weights, dtype=dtype, device=device)[:, None]
            result += self.pos_embed(idx_t) * w_t

        # Permute for spatial merge
        patch_pos_embeds = result.split(
            [t * h * w for t, h, w in grid_thw.tolist()]
        )
        permuted = []
        for pos_emb, (t, h, w) in zip(patch_pos_embeds, grid_thw.tolist()):
            pos_emb = (
                pos_emb.view(t, h // m, m, w // m, m, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            permuted.append(pos_emb)
        return torch.cat(permuted)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: pixel values, flattened patches
            grid_thw: (num_images, 3) tensor of (t, h, w) in patch-grid units

        Returns:
            (total_merged_tokens, out_hidden_size * (1 + num_deepstack))
        """
        total_start = time.perf_counter()
        num_images = int(grid_thw.shape[0])
        x = x.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(device=self.device)

        # 1. Patch embedding
        patch_start = time.perf_counter()
        x = self.patch_embed(x)
        maybe_log_perf(
            logger,
            f"vision.patch_embed images={num_images} patches={x.shape[0]}",
            patch_start,
            rank0=True,
        )

        # 2. Absolute position embedding (bilinear interpolation)
        pos_start = time.perf_counter()
        x = x + self._fast_pos_embed_interpolate(grid_thw)
        maybe_log_perf(
            logger,
            f"vision.abs_pos_embed patches={x.shape[0]}",
            pos_start,
            rank0=True,
        )

        # 3. Rotary position embedding for vision attention
        prep_start = time.perf_counter()
        rot_emb = self._rot_pos_emb(grid_thw).to(x.device)
        emb = torch.cat((rot_emb, rot_emb), dim=-1)  # neox-style doubling
        position_embeddings = (emb.cos(), emb.sin())

        # 4. Compute cu_seqlens (each temporal frame is a separate segment)
        cu_seqlens = compute_cu_seqlens(grid_thw).to(x.device)
        maybe_log_perf(
            logger,
            f"vision.position_prep segments={cu_seqlens.numel() - 1}",
            prep_start,
            rank0=True,
        )

        # 5. Run through vision blocks, capture deepstack intermediates
        blocks_start = time.perf_counter()
        deepstack_features = []
        ds_captured = 0
        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                ds_feat = self.deepstack_merger_list[ds_captured](x)
                deepstack_features.append(ds_feat)
                ds_captured += 1
        maybe_log_perf(
            logger,
            f"vision.blocks layers={len(self.blocks)} patches={x.shape[0]}",
            blocks_start,
            rank0=True,
        )

        # 6. Final merger
        merger_start = time.perf_counter()
        x = self.merger(x)

        # 7. Concatenate with deepstack features along feature dimension
        hidden_states = torch.cat([x] + deepstack_features, dim=1)
        maybe_log_perf(
            logger,
            f"vision.merge merged_tokens={hidden_states.shape[0]}",
            merger_start,
            rank0=True,
        )
        maybe_log_perf(
            logger,
            f"vision.forward_total images={num_images} merged_tokens={hidden_states.shape[0]}",
            total_start,
            rank0=True,
        )
        return hidden_states
