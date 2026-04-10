"""Multi-tier KV cache pool: HBM (GPU) → CPU DRAM → SSD.

Implements ``BaseKVCachePool`` so that the rest of the system (attention
backend, scheduler, engine) can use it transparently.  New KV entries are
always written to the GPU tier; cold pages are evicted to CPU or SSD and
prefetched back on demand via ``PrefetchScheduler``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from .base import BaseKVCachePool

if TYPE_CHECKING:
    from .eviction import TieredEvictionPolicy
    from .prefetch import PrefetchScheduler

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

class Tier(IntEnum):
    GPU = 0
    CPU = 1
    SSD = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TieredCacheConfig:
    num_layers: int
    num_kv_heads: int          # global (pre-TP) kv heads
    head_dim: int
    dtype: torch.dtype
    gpu_pages: int             # pages on GPU
    cpu_pages: int             # pages on CPU (pinned)
    ssd_pages: int = 0         # pages on SSD (0 = disabled)
    ssd_path: str = "/tmp/minisgl_kv_cache"
    page_size: int = 1         # tokens per page (mini-sglang default)


# ---------------------------------------------------------------------------
# Location table  – maps logical page-id → (tier, offset)
# ---------------------------------------------------------------------------

class LocationTable:
    """Compact array-based page location index.

    Every logical page-id in ``[0, max_pages)`` has an entry.  The table is
    stored on *CPU* because it is primarily accessed by the Python-level
    prefetch / eviction logic, not GPU kernels.
    """

    def __init__(self, max_pages: int) -> None:
        self.max_pages = max_pages
        self.tiers = torch.full((max_pages,), -1, dtype=torch.int8)
        self.offsets = torch.zeros(max_pages, dtype=torch.int32)
        self.last_access = torch.zeros(max_pages, dtype=torch.float64)
        self.access_count = torch.zeros(max_pages, dtype=torch.int32)

    # -- batch helpers ------------------------------------------------------

    def lookup(self, page_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tiers, offsets)`` for *page_ids*."""
        ids = page_ids.long().cpu()
        return self.tiers[ids], self.offsets[ids]

    def update(
        self,
        page_ids: torch.Tensor,
        tier: Tier,
        offsets: torch.Tensor,
    ) -> None:
        ids = page_ids.long().cpu()
        self.tiers[ids] = tier.value
        self.offsets[ids] = offsets.int()

    def touch(self, page_ids: torch.Tensor) -> None:
        """Update last-access timestamp and bump access count."""
        ids = page_ids.long().cpu()
        now = time.monotonic()
        self.last_access[ids] = now
        self.access_count[ids] += 1

    def mark_free(self, page_ids: torch.Tensor) -> None:
        self.tiers[page_ids.long().cpu()] = -1


# ---------------------------------------------------------------------------
# Single-tier storage pool
# ---------------------------------------------------------------------------

class TierPool:
    """Manages a contiguous buffer on one tier (GPU / CPU / SSD)."""

    def __init__(
        self,
        tier: Tier,
        capacity_pages: int,
        num_layers: int,
        local_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        ssd_path: str = "",
    ) -> None:
        self.tier = tier
        self.capacity = capacity_pages
        self.num_layers = num_layers
        self.local_kv_heads = local_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size

        shape = (2, num_layers, capacity_pages, page_size, local_kv_heads, head_dim)

        if tier == Tier.GPU:
            self.buffer = torch.empty(shape, device="cuda", dtype=dtype)
        elif tier == Tier.CPU:
            self.buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            # SSD: memory-mapped file
            os.makedirs(os.path.dirname(ssd_path) or ".", exist_ok=True)
            np_dtype = {
                torch.float16: np.float16,
                torch.bfloat16: np.float16,   # numpy has no bfloat16, store raw
                torch.float32: np.float32,
            }.get(dtype, np.float16)
            self.buffer = np.memmap(
                ssd_path,
                dtype=np_dtype,
                mode="w+",
                shape=shape,
            )

        self._free_slots: List[int] = list(range(capacity_pages))

    # -- allocation ---------------------------------------------------------

    def allocate(self, n: int) -> torch.Tensor:
        if len(self._free_slots) < n:
            raise RuntimeError(
                f"{self.tier.name} pool exhausted: need {n}, have {len(self._free_slots)}"
            )
        offsets = [self._free_slots.pop() for _ in range(n)]
        return torch.tensor(offsets, dtype=torch.int32)

    def free(self, offsets: torch.Tensor) -> None:
        self._free_slots.extend(offsets.tolist())

    @property
    def free_count(self) -> int:
        return len(self._free_slots)

    @property
    def used_count(self) -> int:
        return self.capacity - self.free_count


# ---------------------------------------------------------------------------
# TieredKVCachePool – the main class
# ---------------------------------------------------------------------------

class TieredKVCachePool(BaseKVCachePool):
    """Multi-tier KV cache that is a drop-in replacement for ``MHAKVCache``."""

    def __init__(self, config: TieredCacheConfig) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(
            config.num_kv_heads, tp_info.size, allow_replicate=True
        )

        self._num_layers = config.num_layers
        self._local_kv_heads = local_kv_heads
        self._head_dim = config.head_dim
        self._page_size = config.page_size
        self._dtype = config.dtype

        pool_kwargs = dict(
            num_layers=config.num_layers,
            local_kv_heads=local_kv_heads,
            head_dim=config.head_dim,
            page_size=config.page_size,
            dtype=config.dtype,
        )

        self.gpu_pool = TierPool(Tier.GPU, config.gpu_pages, **pool_kwargs)
        self.cpu_pool = TierPool(Tier.CPU, config.cpu_pages, **pool_kwargs)
        self.ssd_pool: Optional[TierPool] = None
        if config.ssd_pages > 0:
            self.ssd_pool = TierPool(
                Tier.SSD, config.ssd_pages, ssd_path=config.ssd_path, **pool_kwargs
            )

        total_pages = config.gpu_pages + config.cpu_pages + config.ssd_pages
        self.location_table = LocationTable(total_pages)

        # Storage-shape used when passing a single layer to store_cache kernel
        self._storage_shape = (
            config.gpu_pages * config.page_size,
            local_kv_heads,
            config.head_dim,
        )

        # Lazily attached by Engine after construction
        self.prefetch_scheduler: Optional[PrefetchScheduler] = None
        self.eviction_policy: Optional[TieredEvictionPolicy] = None

        logger.info(
            f"TieredKVCachePool: GPU={config.gpu_pages} pages, "
            f"CPU={config.cpu_pages} pages, SSD={config.ssd_pages} pages"
        )

    # -- BaseKVCachePool interface ------------------------------------------

    def k_cache(self, index: int) -> torch.Tensor:
        return self.gpu_pool.buffer[0, index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self.gpu_pool.buffer[1, index]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self.gpu_pool.buffer[0, layer_id].view(self._storage_shape),
            v_cache=self.gpu_pool.buffer[1, layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return torch.device("cuda")

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    # -- tier helpers -------------------------------------------------------

    def get_pool(self, tier: Tier) -> TierPool:
        if tier == Tier.GPU:
            return self.gpu_pool
        if tier == Tier.CPU:
            return self.cpu_pool
        if tier == Tier.SSD:
            if self.ssd_pool is None:
                raise RuntimeError("SSD tier not configured")
            return self.ssd_pool
        raise ValueError(f"Unknown tier: {tier}")

    @property
    def total_pages(self) -> int:
        n = self.gpu_pool.capacity + self.cpu_pool.capacity
        if self.ssd_pool is not None:
            n += self.ssd_pool.capacity
        return n

    def debug_stats(self) -> str:
        """Return a one-line summary of per-tier page utilisation.

        Counts are derived from ``LocationTable.tiers`` which is the
        authoritative source of truth (``TierPool._free_slots`` only tracks
        cross-tier transfer allocations, not normal page usage).
        """
        lt = self.location_table
        gpu_used = int((lt.tiers == Tier.GPU.value).sum().item())
        cpu_used = int((lt.tiers == Tier.CPU.value).sum().item())
        parts = [
            f"GPU={gpu_used}/{self.gpu_pool.capacity}",
            f"CPU={cpu_used}/{self.cpu_pool.capacity}",
        ]
        if self.ssd_pool is not None:
            ssd_used = int((lt.tiers == Tier.SSD.value).sum().item())
            parts.append(f"SSD={ssd_used}/{self.ssd_pool.capacity}")
        active = gpu_used + cpu_used
        if self.ssd_pool is not None:
            active += int((lt.tiers == Tier.SSD.value).sum().item())
        parts.append(f"active={active}/{lt.max_pages}")
        return " | ".join(parts)


__all__ = [
    "Tier",
    "TieredCacheConfig",
    "LocationTable",
    "TierPool",
    "TieredKVCachePool",
]
