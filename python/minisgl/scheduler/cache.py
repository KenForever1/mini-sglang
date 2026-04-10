from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.kvcache.tiered_pool import Tier, TieredKVCachePool
from minisgl.utils import div_ceil, init_logger

if TYPE_CHECKING:
    from .utils import PendingReq

logger = init_logger(__name__)


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        # Optionally linked to tiered KV cache pool for location tracking
        self._tiered_pool: Optional[TieredKVCachePool] = None

    def set_tiered_pool(self, pool: TieredKVCachePool) -> None:
        """Attach a tiered KV cache pool for location-table bookkeeping.

        Design invariant: ``page_id == GPU_buffer_offset`` (1:1 permanent
        mapping).  Cross-tier transfers move *data* but never reassign GPU
        offsets.  Free pages start with ``tier = -1`` in the LocationTable
        and are marked ``Tier.GPU`` only when actually allocated for a
        request (see ``_allocate``).
        """
        self._tiered_pool = pool

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            pages = self._allocate(needed_pages)
            allocated = self._page_to_token(pages)
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
            # Lock newly-allocated pages to prevent tiered eviction during attention
            if self._tiered_pool is not None and self._tiered_pool.eviction_policy is not None:
                self._tiered_pool.eviction_policy.lock_pages(
                    (pages // self.page_size).cpu()
                )

    def ensure_batch_on_gpu(self, reqs: List[Req]) -> None:
        """Ensure every page referenced by *reqs* is physically on GPU.

        When tiered KV-cache is enabled, some prefix-cached pages may have
        been demoted to CPU or SSD.  This method copies them back to their
        permanent GPU offset (page_id == GPU_offset invariant) so that the
        FlashInfer attention kernel receives valid data.
        """
        if self._tiered_pool is None:
            return

        lt = self._tiered_pool.location_table

        for req in reqs:
            device_len = req.device_len
            if device_len == 0:
                continue

            # page_table entries are token indices; with page_size=1, page_id == token_index
            pt_entries = self.page_table[req.table_idx, :device_len].cpu()
            page_ids = pt_entries.int() // self.page_size

            tiers, offsets = lt.lookup(page_ids)
            # -1 means free/untracked – skip those (they belong to freshly allocated pages)
            off_gpu_mask = (tiers != Tier.GPU.value) & (tiers >= 0)

            if not off_gpu_mask.any():
                continue

            off_gpu_positions = off_gpu_mask.nonzero(as_tuple=True)[0]
            off_gpu_page_ids = page_ids[off_gpu_positions]
            off_gpu_tiers = tiers[off_gpu_positions]
            off_gpu_offsets = offsets[off_gpu_positions]

            # page_id == GPU_offset: copy data back to the page's own GPU slot.
            gpu_buf = self._tiered_pool.gpu_pool.buffer

            for i in range(len(off_gpu_page_ids)):
                pid = off_gpu_page_ids[i].item()
                src_tier = Tier(off_gpu_tiers[i].item())
                src_off = off_gpu_offsets[i].item()
                src_pool = self._tiered_pool.get_pool(src_tier)

                # Copy all layers from source tier back to GPU at offset = page_id
                if isinstance(src_pool.buffer, np.memmap):
                    data = torch.from_numpy(np.array(src_pool.buffer[:, :, src_off]))
                    gpu_buf[:, :, pid].copy_(data)
                else:
                    gpu_buf[:, :, pid].copy_(src_pool.buffer[:, :, src_off])

                src_pool.free(torch.tensor([src_off], dtype=torch.int32))

            # Update location table: pages are back on GPU at their permanent offset
            lt.update(off_gpu_page_ids, Tier.GPU, off_gpu_page_ids.int())
            lt.touch(off_gpu_page_ids)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)
            if self._tiered_pool is not None and lazy_free_list:
                all_pages = torch.cat(lazy_free_list)
                page_ids = (all_pages // self.page_size).cpu()
                # Mark lazily-freed pages as untracked so eviction ignores them
                self._tiered_pool.location_table.mark_free(page_ids)
                if self._tiered_pool.eviction_policy is not None:
                    self._tiered_pool.eviction_policy.unlock_pages(page_ids)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            evicted_pages = evicted[:: self.page_size]

            # Evicted prefix entries may have been demoted to CPU/SSD.
            # Free their physical storage on the non-GPU tier.
            if self._tiered_pool is not None:
                lt = self._tiered_pool.location_table
                eids = (evicted_pages // self.page_size).cpu()
                tiers, offsets = lt.lookup(eids)
                for tier, pool in [
                    (Tier.CPU, self._tiered_pool.cpu_pool),
                    (Tier.SSD, self._tiered_pool.ssd_pool),
                ]:
                    if pool is None:
                        continue
                    mask = tiers == tier.value
                    if mask.any():
                        pool.free(offsets[mask])
                lt.mark_free(eids)

            self.free_slots = torch.cat([self.free_slots, evicted_pages])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        # Mark allocated pages as GPU-resident in the location table
        if self._tiered_pool is not None:
            page_ids = (allocated // self.page_size).cpu()
            gpu_offsets = page_ids.int()  # page_id == GPU buffer offset
            self._tiered_pool.location_table.update(page_ids, Tier.GPU, gpu_offsets)
            self._tiered_pool.location_table.touch(page_ids)
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            page_indices = indices[:: self.page_size]
            self.free_slots = torch.cat([self.free_slots, page_indices])
            if self._tiered_pool is not None:
                page_ids = (page_indices // self.page_size).cpu()
                # Mark freed pages as untracked so eviction ignores them
                self._tiered_pool.location_table.mark_free(page_ids)
                if self._tiered_pool.eviction_policy is not None:
                    self._tiered_pool.eviction_policy.unlock_pages(page_ids)

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
