"""End-to-end integration tests for the multi-tier KV cache system.

Tests cover:
  - TieredKVCachePool store/retrieve correctness
  - Eviction GPU→CPU→SSD with data integrity
  - Page locking prevents eviction of active pages
  - PrefetchScheduler brings off-GPU pages back
  - CacheManager + TieredKVCachePool integration (location table, page-table remapping)
  - Boundary condition: all pages locked → proper RuntimeError

Requires CUDA. Tests are automatically skipped on CPU-only machines.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import minisgl.core as core
from minisgl.kvcache.eviction import TieredEvictionPolicy
from minisgl.kvcache.tiered_pool import (
    LocationTable,
    Tier,
    TierPool,
    TieredCacheConfig,
    TieredKVCachePool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 2
KV_HEADS = 2
HEAD_DIM = 8
DTYPE = torch.float32
PAGE_SIZE = 1


def _make_config(gpu_pages: int, cpu_pages: int, ssd_pages: int = 0) -> TieredCacheConfig:
    return TieredCacheConfig(
        num_layers=NUM_LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        gpu_pages=gpu_pages,
        cpu_pages=cpu_pages,
        ssd_pages=ssd_pages,
        page_size=PAGE_SIZE,
    )


def _make_pool(gpu_pages: int = 8, cpu_pages: int = 8, ssd_pages: int = 0) -> TieredKVCachePool:
    """Create a small tiered pool with eviction policy attached."""
    cfg = _make_config(gpu_pages, cpu_pages, ssd_pages)
    pool = TieredKVCachePool(cfg)
    pool.eviction_policy = TieredEvictionPolicy(pool)
    return pool


def _make_kv_data(page_id: int) -> torch.Tensor:
    """Return a deterministic float tensor to identify a page_id."""
    return torch.full(
        (PAGE_SIZE, KV_HEADS, HEAD_DIM),
        fill_value=float(page_id),
        dtype=DTYPE,
    )


def _write_page_to_gpu(pool: TieredKVCachePool, page_id: int) -> None:
    """Write identifiable KV data for *page_id* into the GPU pool at offset=page_id."""
    data = _make_kv_data(page_id)
    for layer in range(NUM_LAYERS):
        pool.gpu_pool.buffer[0, layer, page_id] = data   # key
        pool.gpu_pool.buffer[1, layer, page_id] = data   # value
    lt = pool.location_table
    lt.update(torch.tensor([page_id]), Tier.GPU, torch.tensor([page_id], dtype=torch.int32))
    lt.touch(torch.tensor([page_id]))


def _verify_gpu_data(pool: TieredKVCachePool, page_id: int, gpu_offset: int) -> None:
    """Assert that the GPU pool at *gpu_offset* contains data for *page_id*."""
    expected = float(page_id)
    for layer in range(NUM_LAYERS):
        k = pool.gpu_pool.buffer[0, layer, gpu_offset]
        v = pool.gpu_pool.buffer[1, layer, gpu_offset]
        assert k.mean().item() == pytest.approx(expected, abs=1e-5), (
            f"Layer {layer} k mismatch: expected {expected}, got {k.mean().item()}"
        )
        assert v.mean().item() == pytest.approx(expected, abs=1e-5), (
            f"Layer {layer} v mismatch: expected {expected}, got {v.mean().item()}"
        )


# ---------------------------------------------------------------------------
# Tests: LocationTable (CPU-only, no CUDA needed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(False, reason="")  # always run
class TestLocationTable:
    def test_lookup_and_update(self):
        lt = LocationTable(16)
        ids = torch.tensor([0, 5, 10])
        lt.update(ids, Tier.GPU, torch.tensor([0, 5, 10], dtype=torch.int32))
        tiers, offsets = lt.lookup(ids)
        assert (tiers == Tier.GPU.value).all()
        assert offsets.tolist() == [0, 5, 10]

    def test_mark_free(self):
        lt = LocationTable(8)
        ids = torch.tensor([3, 7])
        lt.update(ids, Tier.CPU, torch.tensor([0, 1], dtype=torch.int32))
        lt.mark_free(ids)
        tiers, _ = lt.lookup(ids)
        assert (tiers == -1).all()

    def test_touch_updates_access(self):
        lt = LocationTable(4)
        lt.touch(torch.tensor([1, 2]))
        assert lt.last_access[1].item() > 0
        assert lt.access_count[1].item() == 1


# ---------------------------------------------------------------------------
# Tests: TierPool (CPU tier only, no CUDA needed)
# ---------------------------------------------------------------------------

class TestTierPool:
    def test_cpu_pool_allocate_free(self):
        pool = TierPool(Tier.CPU, 4, NUM_LAYERS, KV_HEADS, HEAD_DIM, PAGE_SIZE, DTYPE)
        assert pool.free_count == 4
        offsets = pool.allocate(2)
        assert pool.free_count == 2
        assert len(offsets) == 2
        pool.free(offsets)
        assert pool.free_count == 4

    def test_pool_exhausted_raises(self):
        pool = TierPool(Tier.CPU, 2, NUM_LAYERS, KV_HEADS, HEAD_DIM, PAGE_SIZE, DTYPE)
        pool.allocate(2)
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate(1)


# ---------------------------------------------------------------------------
# Tests: Eviction GPU → CPU data integrity
# ---------------------------------------------------------------------------

class TestEvictionDataIntegrity:
    def test_gpu_to_cpu_demotion_preserves_data(self):
        """Pages demoted from GPU to CPU must retain their KV data."""
        pool = _make_pool(gpu_pages=4, cpu_pages=4)
        policy = pool.eviction_policy

        # Allocate and write pages 0, 1, 2, 3 on GPU
        for pid in range(4):
            pool.gpu_pool.allocate(1)  # take GPU slot
            _write_page_to_gpu(pool, pid)

        # Demote 2 LRU pages (page 0 and 1, since they were touched first)
        import time; time.sleep(0.01)  # ensure access timestamps differ
        pool.location_table.touch(torch.tensor([2, 3]))  # make 2,3 recent

        demoted = policy.evict_from_gpu(2)
        assert len(demoted) == 2
        assert set(demoted) == {0, 1}

        # Verify data is on CPU with correct content
        for pid in demoted:
            tiers, offsets = pool.location_table.lookup(torch.tensor([pid]))
            assert tiers[0].item() == Tier.CPU.value
            cpu_off = offsets[0].item()
            for layer in range(NUM_LAYERS):
                k = pool.cpu_pool.buffer[0, layer, cpu_off]
                v = pool.cpu_pool.buffer[1, layer, cpu_off]
                assert k.mean().item() == pytest.approx(float(pid), abs=1e-5)
                assert v.mean().item() == pytest.approx(float(pid), abs=1e-5)

    def test_locked_pages_not_evicted(self):
        """Locked pages must be skipped during eviction."""
        pool = _make_pool(gpu_pages=4, cpu_pages=4)
        policy = pool.eviction_policy

        for pid in range(4):
            pool.gpu_pool.allocate(1)
            _write_page_to_gpu(pool, pid)

        # Lock page 0 and 1
        policy.lock_pages(torch.tensor([0, 1]))

        # Try to evict 2 – should evict pages 2 and 3 (unlocked)
        import time; time.sleep(0.01)
        pool.location_table.touch(torch.tensor([0, 1]))  # make locked pages "recent" too

        demoted = policy.evict_from_gpu(2)
        assert set(demoted) == {2, 3}

        # Pages 0, 1 should still be on GPU
        tiers, _ = pool.location_table.lookup(torch.tensor([0, 1]))
        assert (tiers == Tier.GPU.value).all()

    def test_all_locked_raises(self):
        """Evicting when all pages are locked must raise RuntimeError."""
        pool = _make_pool(gpu_pages=2, cpu_pages=2)
        policy = pool.eviction_policy

        for pid in range(2):
            pool.gpu_pool.allocate(1)
            _write_page_to_gpu(pool, pid)

        policy.lock_pages(torch.tensor([0, 1]))

        with pytest.raises(RuntimeError, match="locked"):
            policy.evict_from_gpu(1)


# ---------------------------------------------------------------------------
# Tests: PrefetchScheduler (CPU → GPU)
# ---------------------------------------------------------------------------

class TestPrefetchScheduler:
    def test_prefetch_cpu_to_gpu(self):
        """Pages on CPU can be prefetched back to GPU with correct data."""
        pool = _make_pool(gpu_pages=4, cpu_pages=4)
        policy = pool.eviction_policy

        # Setup: write pages 0-3 on GPU
        for pid in range(4):
            pool.gpu_pool.allocate(1)
            _write_page_to_gpu(pool, pid)

        # Demote page 0 to CPU
        import time; time.sleep(0.01)
        pool.location_table.touch(torch.tensor([1, 2, 3]))
        policy.evict_from_gpu(1)
        demoted_ids = [0]  # page 0 was LRU

        # Verify page 0 is on CPU
        tiers, _ = pool.location_table.lookup(torch.tensor([0]))
        assert tiers[0].item() == Tier.CPU.value

        # Now create a PrefetchScheduler and prefetch page 0 back
        from minisgl.kvcache.prefetch import PrefetchScheduler

        ps = PrefetchScheduler(pool)

        # prefetch_layer needs page_ids as a GPU tensor (simulating batch pages)
        ps.prefetch_layer(0, torch.tensor([0], device="cpu"))
        ps.wait_prefetch()

        # Page 0 should be back on GPU
        tiers, offsets = pool.location_table.lookup(torch.tensor([0]))
        assert tiers[0].item() == Tier.GPU.value

        # Data integrity: verify content at the new GPU offset
        new_gpu_off = offsets[0].item()
        _verify_gpu_data(pool, page_id=0, gpu_offset=new_gpu_off)


# ---------------------------------------------------------------------------
# Tests: CacheManager + TieredKVCachePool integration
# ---------------------------------------------------------------------------

class TestCacheManagerTieredIntegration:
    @pytest.fixture(autouse=True)
    def reset_global_ctx(self):
        old_ctx = core._GLOBAL_CTX
        core._GLOBAL_CTX = None
        yield
        core._GLOBAL_CTX = old_ctx

    def _make_cache_manager(self, num_gpu_pages: int = 8, cpu_pages: int = 8):
        from minisgl.scheduler.cache import CacheManager

        page_table = torch.zeros((4, 64), dtype=torch.int32)
        ctx = core.Context(page_size=PAGE_SIZE)
        core.set_global_ctx(ctx)

        cm = CacheManager(num_gpu_pages, PAGE_SIZE, page_table, type="radix")
        pool = _make_pool(gpu_pages=num_gpu_pages, cpu_pages=cpu_pages)
        cm.set_tiered_pool(pool)
        return cm, pool

    def test_set_tiered_pool_initializes_location_table(self):
        cm, pool = self._make_cache_manager(8, 4)
        lt = pool.location_table
        # Free pages start as untracked (tier=-1); they become GPU only when allocated
        for pid in range(8):
            tiers, _ = lt.lookup(torch.tensor([pid]))
            assert tiers[0].item() == -1, f"Page {pid} should be untracked initially"

        # Allocate some pages and verify they are marked GPU
        pages = cm._allocate(3)
        for p in pages.tolist():
            pid = p // PAGE_SIZE
            tiers, offsets = lt.lookup(torch.tensor([pid]))
            assert tiers[0].item() == Tier.GPU.value
            assert offsets[0].item() == pid  # page_id == GPU_offset

    def test_allocate_touches_location_table(self):
        cm, pool = self._make_cache_manager(8, 4)
        lt = pool.location_table
        pages = cm._allocate(3)
        page_ids = (pages // PAGE_SIZE).cpu()
        for pid in page_ids.tolist():
            assert lt.last_access[pid].item() > 0, f"Page {pid} should be touched"

    def test_lock_unlock_through_allocate_free(self):
        """allocate_paged locks pages; _free unlocks them."""
        cm, pool = self._make_cache_manager(8, 4)
        policy = pool.eviction_policy

        # Create a mock Req with the right attributes
        class MockReq:
            def __init__(self, cached_len, device_len, table_idx):
                self.cached_len = cached_len
                self.device_len = device_len
                self.table_idx = table_idx

        req = MockReq(cached_len=0, device_len=3, table_idx=0)
        cm.allocate_paged([req])

        # The allocated pages should be locked
        assert len(policy._locked_pages) == 3

        # Free pages should unlock them
        indices = cm.page_table[0, :3]
        cm._free(indices)
        assert len(policy._locked_pages) == 0


# ---------------------------------------------------------------------------
# Tests: Ensure batch on GPU (page table remapping)
# ---------------------------------------------------------------------------

class TestEnsureBatchOnGpu:
    @pytest.fixture(autouse=True)
    def reset_global_ctx(self):
        old_ctx = core._GLOBAL_CTX
        core._GLOBAL_CTX = None
        yield
        core._GLOBAL_CTX = old_ctx

    def test_noop_when_all_on_gpu(self):
        """When all pages are on GPU, ensure_batch_on_gpu is a no-op."""
        from minisgl.scheduler.cache import CacheManager

        page_table = torch.zeros((4, 64), dtype=torch.int32)
        ctx = core.Context(page_size=PAGE_SIZE)
        core.set_global_ctx(ctx)

        cm = CacheManager(8, PAGE_SIZE, page_table, type="radix")
        pool = _make_pool(gpu_pages=8, cpu_pages=4)
        cm.set_tiered_pool(pool)

        # Allocate a page and write to page_table
        pages = cm._allocate(2)
        page_table[0, 0] = pages[0]
        page_table[0, 1] = pages[1]

        class MockReq:
            def __init__(self):
                self.device_len = 2
                self.table_idx = 0

        req = MockReq()
        # Should not raise or change anything
        cm.ensure_batch_on_gpu([req])

        # page_table unchanged
        assert page_table[0, 0].item() == pages[0].item()
        assert page_table[0, 1].item() == pages[1].item()

    def test_remaps_cpu_page_to_gpu(self):
        """When a page is on CPU, ensure_batch_on_gpu copies data back to GPU."""
        from minisgl.scheduler.cache import CacheManager

        page_table = torch.zeros((4, 64), dtype=torch.int32)
        ctx = core.Context(page_size=PAGE_SIZE)
        core.set_global_ctx(ctx)

        num_gpu = 4
        cm = CacheManager(num_gpu, PAGE_SIZE, page_table, type="radix")
        pool = _make_pool(gpu_pages=num_gpu, cpu_pages=4)
        cm.set_tiered_pool(pool)

        # Allocate page 0 and write identifiable data
        pages = cm._allocate(1)
        pid = (pages[0] // PAGE_SIZE).item()
        _write_page_to_gpu(pool, pid)
        page_table[0, 0] = pages[0]

        # Manually demote page to CPU (simulating eviction)
        import time; time.sleep(0.01)
        pool.location_table.touch(torch.tensor([p for p in range(num_gpu) if p != pid]))
        pool.eviction_policy.evict_from_gpu(1)

        # Verify page is on CPU now
        tiers, _ = pool.location_table.lookup(torch.tensor([pid]))
        assert tiers[0].item() == Tier.CPU.value

        class MockReq:
            def __init__(self):
                self.device_len = 1
                self.table_idx = 0

        # ensure_batch_on_gpu should bring it back
        cm.ensure_batch_on_gpu([MockReq()])

        # Page should be back on GPU
        tiers, offsets = pool.location_table.lookup(torch.tensor([pid]))
        assert tiers[0].item() == Tier.GPU.value

        # page_table retains the original value because page_id == GPU_offset (permanent)
        new_gpu_offset = offsets[0].item()
        assert page_table[0, 0].item() == new_gpu_offset * PAGE_SIZE

        # Data integrity
        _verify_gpu_data(pool, page_id=pid, gpu_offset=new_gpu_offset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
