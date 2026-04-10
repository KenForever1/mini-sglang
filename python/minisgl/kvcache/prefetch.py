"""Layer-wise prefetch scheduler for TieredKVCachePool.

The key insight: decoder layers execute sequentially.  While Layer *N* is
computing attention, we can asynchronously transfer Layer *N+1*'s KV data
from CPU/SSD to GPU on a separate CUDA stream, hiding the PCIe latency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .tiered_pool import TieredKVCachePool

from .tiered_pool import Tier

logger = init_logger(__name__)


class PrefetchScheduler:
    """Asynchronous layer-wise KV cache prefetch."""

    def __init__(self, pool: TieredKVCachePool) -> None:
        self.pool = pool
        self._stream = torch.cuda.Stream()
        self._event = torch.cuda.Event()
        self._has_pending = False

    # -- public API ---------------------------------------------------------

    def prefetch_layer(self, layer_id: int, page_ids: torch.Tensor) -> None:
        """Asynchronously bring all *page_ids* for *layer_id* onto GPU.

        Pages already on GPU are skipped.  If GPU space is insufficient,
        the eviction policy is called to make room.

        Must be followed by a ``wait_prefetch()`` before the layer's
        attention actually reads the data.
        """
        lt = self.pool.location_table
        tiers, offsets = lt.lookup(page_ids)

        cpu_mask = tiers == Tier.CPU.value
        ssd_mask = tiers == Tier.SSD.value
        need_transfer = cpu_mask | ssd_mask

        if not need_transfer.any():
            self._has_pending = False
            return

        transfer_ids = page_ids[need_transfer]
        transfer_tiers = tiers[need_transfer]
        transfer_offsets = offsets[need_transfer]

        needed = int(need_transfer.sum().item())
        # page_id == GPU_offset: data goes back to the page's own GPU slot.
        # No gpu_pool allocation needed – the GPU offset is permanent.
        gpu_offsets = transfer_ids.int()

        with torch.cuda.stream(self._stream):
            gpu_buf = self.pool.gpu_pool.buffer

            for i in range(needed):
                src_tier = Tier(transfer_tiers[i].item())
                s_off = transfer_offsets[i].item()
                d_off = gpu_offsets[i].item()

                if src_tier == Tier.CPU:
                    # Pinned CPU → GPU (async)
                    gpu_buf[:, layer_id, d_off].copy_(
                        self.pool.cpu_pool.buffer[:, layer_id, s_off],
                        non_blocking=True,
                    )
                elif src_tier == Tier.SSD:
                    # SSD → GPU via CPU staging
                    self._ssd_to_gpu_single(layer_id, s_off, d_off)

            self._event.record()
        self._has_pending = True

        # Update location table — these pages are now on GPU
        # (for the prefetched layer; other layers may still be elsewhere)
        lt.update(transfer_ids, Tier.GPU, gpu_offsets)
        lt.touch(transfer_ids)

        # Free the source offsets on the *source* pool
        cpu_ids = transfer_ids[transfer_tiers == Tier.CPU.value]
        ssd_ids = transfer_ids[transfer_tiers == Tier.SSD.value]
        if len(cpu_ids) > 0:
            cpu_src = transfer_offsets[transfer_tiers == Tier.CPU.value]
            self.pool.cpu_pool.free(cpu_src)
        if len(ssd_ids) > 0 and self.pool.ssd_pool is not None:
            ssd_src = transfer_offsets[transfer_tiers == Tier.SSD.value]
            self.pool.ssd_pool.free(ssd_src)

    def wait_prefetch(self) -> None:
        """Block until the most recent ``prefetch_layer`` completes."""
        if self._has_pending:
            self._event.synchronize()
            self._has_pending = False

    # -- internals ----------------------------------------------------------

    def _ensure_gpu_space(self, needed: int) -> None:
        gpu = self.pool.gpu_pool
        if gpu.free_count >= needed:
            return
        shortfall = needed - gpu.free_count
        if self.pool.eviction_policy is None:
            raise RuntimeError(
                f"GPU KV cache full and no eviction policy attached "
                f"(need {shortfall} more pages)"
            )
        self.pool.eviction_policy.evict_from_gpu(shortfall)
        if gpu.free_count < needed:
            raise RuntimeError(
                f"After eviction, GPU still lacks {needed - gpu.free_count} pages"
            )

    def _ssd_to_gpu_single(
        self, layer_id: int, ssd_offset: int, gpu_offset: int
    ) -> None:
        """Transfer a single page for one layer: SSD → pinned CPU → GPU."""
        assert self.pool.ssd_pool is not None
        ssd_buf = self.pool.ssd_pool.buffer

        # SSD → numpy → torch (on CPU pinned staging)
        src_np = np.array(ssd_buf[:, layer_id, ssd_offset])
        src_cpu = torch.from_numpy(src_np).pin_memory()

        # CPU → GPU (async on self._stream)
        self.pool.gpu_pool.buffer[:, layer_id, gpu_offset].copy_(
            src_cpu, non_blocking=True
        )


__all__ = ["PrefetchScheduler"]
