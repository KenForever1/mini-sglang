"""Multi-tier eviction policy for TieredKVCachePool.

Pages are *demoted* (never deleted) through the hierarchy:
    GPU → CPU → SSD → release

Eviction ordering is LRU based on ``LocationTable.last_access``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .tiered_pool import TieredKVCachePool

from .tiered_pool import Tier

logger = init_logger(__name__)


class TieredEvictionPolicy:
    def __init__(self, pool: TieredKVCachePool) -> None:
        self.pool = pool
        # page-ids that are currently locked (in-flight attention)
        self._locked_pages: set[int] = set()

    # -- public lock API (called by scheduler/cache manager) ----------------

    def lock_pages(self, page_ids: torch.Tensor) -> None:
        self._locked_pages.update(page_ids.tolist())

    def unlock_pages(self, page_ids: torch.Tensor) -> None:
        self._locked_pages.difference_update(page_ids.tolist())

    # -- eviction -----------------------------------------------------------

    def evict_from_gpu(self, count: int) -> List[int]:
        """Demote the *count* least-recently-used GPU pages to CPU.

        Returns the list of demoted page-ids.
        """
        return self._demote(count, src=Tier.GPU, dst=Tier.CPU)

    def evict_from_cpu(self, count: int) -> List[int]:
        """Demote the *count* least-recently-used CPU pages to SSD (or release).

        Returns the list of demoted/released page-ids.
        """
        if self.pool.ssd_pool is not None:
            return self._demote(count, src=Tier.CPU, dst=Tier.SSD)
        else:
            return self._release(count, tier=Tier.CPU)

    # -- internals ----------------------------------------------------------

    def _demote(self, count: int, *, src: Tier, dst: Tier) -> List[int]:
        """Move *count* pages from *src* to *dst* tier.  Returns demoted page-ids."""
        lt = self.pool.location_table
        src_pool = self.pool.get_pool(src)
        dst_pool = self.pool.get_pool(dst)

        # Ensure destination has space (recursive eviction)
        if dst_pool.free_count < count:
            shortfall = count - dst_pool.free_count
            if dst == Tier.CPU:
                self.evict_from_cpu(shortfall)
            elif dst == Tier.SSD:
                self._release(shortfall, tier=Tier.SSD)
            # After recursive eviction, re-check
            if dst_pool.free_count < count:
                raise RuntimeError(
                    f"Cannot free {count} pages on {dst.name}: "
                    f"only {dst_pool.free_count} available after recursive eviction"
                )

        evict_ids = self._select_lru_pages(src, count)
        if len(evict_ids) < count:
            raise RuntimeError(
                f"{src.name} eviction: need {count} pages but only "
                f"{len(evict_ids)} evictable (rest are locked)"
            )

        src_offsets = lt.offsets[evict_ids.long()]
        dst_offsets = dst_pool.allocate(len(evict_ids))

        self._copy_pages(src, dst, evict_ids, src_offsets, dst_offsets)

        # Bookkeeping: free the *source* physical slot – but NOT for GPU,
        # because page_id == GPU_offset is a permanent 1:1 mapping.
        if src != Tier.GPU:
            src_pool.free(src_offsets)
        lt.update(evict_ids, dst, dst_offsets)

        return evict_ids.tolist()

    def _release(self, count: int, tier: Tier) -> List[int]:
        """Permanently release pages from a tier (discard KV data).  Returns page-ids."""
        evict_ids = self._select_lru_pages(tier, count)
        if len(evict_ids) < count:
            raise RuntimeError(
                f"Cannot release {count} pages from {tier.name}: "
                f"only {len(evict_ids)} evictable"
            )
        pool = self.pool.get_pool(tier)
        offsets = self.pool.location_table.offsets[evict_ids.long()]
        pool.free(offsets)
        self.pool.location_table.mark_free(evict_ids)
        return evict_ids.tolist()

    def _select_lru_pages(self, tier: Tier, count: int) -> torch.Tensor:
        """Return up to *count* page-ids on *tier*, ordered by LRU."""
        lt = self.pool.location_table
        on_tier = (lt.tiers == tier.value).nonzero(as_tuple=True)[0]
        if len(on_tier) == 0:
            return torch.tensor([], dtype=torch.int64)

        # Filter out locked pages
        if self._locked_pages:
            locked_set = self._locked_pages
            mask = torch.tensor(
                [pid.item() not in locked_set for pid in on_tier], dtype=torch.bool
            )
            on_tier = on_tier[mask]

        if len(on_tier) == 0:
            return torch.tensor([], dtype=torch.int64)

        # Sort by last_access ascending (oldest first)
        access = lt.last_access[on_tier]
        _, sorted_idx = access.sort()
        selected = on_tier[sorted_idx[:count]]
        return selected

    def _copy_pages(
        self,
        src_tier: Tier,
        dst_tier: Tier,
        page_ids: torch.Tensor,
        src_offsets: torch.Tensor,
        dst_offsets: torch.Tensor,
    ) -> None:
        """Copy KV data for *page_ids* across all layers from src to dst."""
        src_buf = self.pool.get_pool(src_tier).buffer
        dst_buf = self.pool.get_pool(dst_tier).buffer

        for s_off, d_off in zip(src_offsets.tolist(), dst_offsets.tolist()):
            if isinstance(dst_buf, np.memmap):
                # → SSD: read from torch tensor, write to numpy
                if isinstance(src_buf, torch.Tensor):
                    data = src_buf[:, :, s_off].cpu().numpy()
                else:
                    data = np.array(src_buf[:, :, s_off])
                dst_buf[:, :, d_off] = data
            elif isinstance(src_buf, np.memmap):
                # SSD → torch: read from numpy, write to torch
                data = torch.from_numpy(np.array(src_buf[:, :, s_off]))
                dst_buf[:, :, d_off].copy_(data)
            else:
                # GPU ↔ CPU: torch copy
                dst_buf[:, :, d_off].copy_(
                    src_buf[:, :, s_off],
                    non_blocking=(src_tier == Tier.GPU),
                )


__all__ = ["TieredEvictionPolicy"]
