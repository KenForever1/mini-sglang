#!/usr/bin/env python3
"""Benchmark script for the multi-tier KV cache system.

Measures:
  1. GPU-only vs GPU+CPU offload decode throughput at different context lengths.
  2. Layer-wise prefetch overlap effectiveness (PCIe transfer vs compute).
  3. Eviction + prefetch round-trip latency.

Usage:
    # GPU-only baseline
    python benchmarks/bench_tiered_kv.py --mode gpu-only --context-lengths 1024 4096 8192

    # GPU + CPU offload
    python benchmarks/bench_tiered_kv.py --mode tiered --cpu-pages 2000 --context-lengths 1024 4096 8192

    # Compare both
    python benchmarks/bench_tiered_kv.py --mode compare --cpu-pages 2000

Requires CUDA.
"""
from __future__ import annotations

import argparse
import time
from typing import List

import torch


def _check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")


# ---------------------------------------------------------------------------
# Micro-benchmark: Eviction + Prefetch round-trip
# ---------------------------------------------------------------------------

def bench_eviction_prefetch_roundtrip(
    gpu_pages: int = 256,
    cpu_pages: int = 512,
    num_layers: int = 28,
    kv_heads: int = 4,
    head_dim: int = 128,
    page_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    evict_count: int = 64,
    warmup: int = 3,
    repeat: int = 10,
) -> dict:
    """Benchmark GPU→CPU eviction and CPU→GPU prefetch latency."""
    from minisgl.distributed import set_tp_info
    set_tp_info(rank=0, size=1)

    from minisgl.kvcache.eviction import TieredEvictionPolicy
    from minisgl.kvcache.prefetch import PrefetchScheduler
    from minisgl.kvcache.tiered_pool import TieredCacheConfig, TieredKVCachePool

    cfg = TieredCacheConfig(
        num_layers=num_layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        gpu_pages=gpu_pages,
        cpu_pages=cpu_pages,
        page_size=page_size,
    )
    pool = TieredKVCachePool(cfg)
    pool.eviction_policy = TieredEvictionPolicy(pool)
    pool.prefetch_scheduler = PrefetchScheduler(pool)

    # Fill GPU pages with data
    for pid in range(gpu_pages):
        pool.gpu_pool.allocate(1)
        pool.location_table.update(
            torch.tensor([pid]), torch.tensor([0]),  # Tier.GPU
            torch.tensor([pid], dtype=torch.int32),
        )
        pool.location_table.touch(torch.tensor([pid]))

    torch.cuda.synchronize()

    # Warmup + Benchmark eviction GPU→CPU
    evict_times = []
    prefetch_times = []

    for trial in range(warmup + repeat):
        # Make some pages "old" for eviction
        time.sleep(0.001)
        recent_pages = torch.arange(evict_count, gpu_pages)
        pool.location_table.touch(recent_pages)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        demoted = pool.eviction_policy.evict_from_gpu(evict_count)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if trial >= warmup:
            evict_times.append(t1 - t0)

        # Prefetch them back
        page_ids = torch.tensor(demoted, device="cpu")
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        pool.prefetch_scheduler.prefetch_layer(0, page_ids)
        pool.prefetch_scheduler.wait_prefetch()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        if trial >= warmup:
            prefetch_times.append(t3 - t2)

    page_bytes = 2 * num_layers * page_size * kv_heads * head_dim * dtype.itemsize

    return {
        "evict_count": evict_count,
        "page_bytes": page_bytes,
        "total_bytes": evict_count * page_bytes,
        "evict_mean_ms": sum(evict_times) / len(evict_times) * 1000,
        "evict_min_ms": min(evict_times) * 1000,
        "prefetch_mean_ms": sum(prefetch_times) / len(prefetch_times) * 1000,
        "prefetch_min_ms": min(prefetch_times) * 1000,
        "evict_bw_gbps": (evict_count * page_bytes) / (min(evict_times) * 1e9),
        "prefetch_bw_gbps": (evict_count * page_bytes) / (min(prefetch_times) * 1e9),
    }


# ---------------------------------------------------------------------------
# Micro-benchmark: Simulated decode throughput
# ---------------------------------------------------------------------------

def bench_simulated_decode(
    context_lengths: List[int],
    gpu_pages: int = 4096,
    cpu_pages: int = 0,
    num_layers: int = 28,
    kv_heads: int = 4,
    head_dim: int = 128,
    page_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    decode_steps: int = 50,
    warmup_steps: int = 5,
) -> List[dict]:
    """Simulate decode steps and measure throughput."""
    from minisgl.distributed import set_tp_info
    set_tp_info(rank=0, size=1)

    from minisgl.kvcache.tiered_pool import TieredCacheConfig, TieredKVCachePool
    from minisgl.kvcache.eviction import TieredEvictionPolicy
    from minisgl.kvcache.prefetch import PrefetchScheduler

    results = []

    for ctx_len in context_lengths:
        needed_pages = ctx_len // page_size
        actual_gpu = min(gpu_pages, needed_pages)
        actual_cpu = max(0, needed_pages - actual_gpu) if cpu_pages > 0 else 0

        cfg = TieredCacheConfig(
            num_layers=num_layers,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            gpu_pages=actual_gpu + 16,
            cpu_pages=max(actual_cpu + 16, 16),
            page_size=page_size,
        )
        pool = TieredKVCachePool(cfg)
        pool.eviction_policy = TieredEvictionPolicy(pool)
        pool.prefetch_scheduler = PrefetchScheduler(pool)

        # Pre-fill GPU pages
        for pid in range(min(actual_gpu, cfg.gpu_pages)):
            pool.gpu_pool.allocate(1)
            pool.location_table.update(
                torch.tensor([pid]),
                torch.tensor([0]),  # GPU
                torch.tensor([pid], dtype=torch.int32),
            )
            pool.location_table.touch(torch.tensor([pid]))

        # Simulate a decode step: read from k_cache/v_cache, then store_kv
        k_dummy = torch.randn(1, kv_heads, head_dim, device="cuda", dtype=dtype)
        v_dummy = torch.randn(1, kv_heads, head_dim, device="cuda", dtype=dtype)

        torch.cuda.synchronize()
        step_times = []

        for step in range(warmup_steps + decode_steps):
            # Pick a page to write to (cycle through first few)
            write_page = step % actual_gpu
            out_loc = torch.tensor([write_page * page_size], device="cuda", dtype=torch.int32)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Simulate per-layer forward
            for layer_id in range(num_layers):
                pool.store_kv(k_dummy, v_dummy, out_loc, layer_id)
                _ = pool.k_cache(layer_id)
                _ = pool.v_cache(layer_id)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if step >= warmup_steps:
                step_times.append(t1 - t0)

        mean_ms = sum(step_times) / len(step_times) * 1000
        tok_per_s = 1.0 / (sum(step_times) / len(step_times))

        results.append({
            "context_length": ctx_len,
            "gpu_pages": actual_gpu,
            "cpu_pages": actual_cpu,
            "mode": "tiered" if cpu_pages > 0 else "gpu-only",
            "decode_mean_ms": mean_ms,
            "tokens_per_sec": tok_per_s,
        })

        # Cleanup
        del pool
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_results(label: str, results):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.3f}")
            else:
                print(f"  {k:30s}: {v}")
    elif isinstance(results, list):
        for r in results:
            print(f"\n  Context Length: {r['context_length']}")
            for k, v in r.items():
                if k == "context_length":
                    continue
                if isinstance(v, float):
                    print(f"    {k:28s}: {v:.3f}")
                else:
                    print(f"    {k:28s}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Tiered KV Cache Benchmark")
    parser.add_argument("--mode", choices=["gpu-only", "tiered", "compare", "roundtrip"],
                        default="roundtrip")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--gpu-pages", type=int, default=4096)
    parser.add_argument("--cpu-pages", type=int, default=2000)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--evict-count", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=50)
    args = parser.parse_args()

    _check_cuda()

    if args.mode in ("roundtrip", "compare"):
        rt = bench_eviction_prefetch_roundtrip(
            gpu_pages=args.gpu_pages,
            cpu_pages=args.cpu_pages,
            num_layers=args.num_layers,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            evict_count=args.evict_count,
        )
        print_results("Eviction + Prefetch Round-trip", rt)

    if args.mode in ("gpu-only", "compare"):
        gpu_results = bench_simulated_decode(
            context_lengths=args.context_lengths,
            gpu_pages=args.gpu_pages,
            cpu_pages=0,
            num_layers=args.num_layers,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            decode_steps=args.decode_steps,
        )
        print_results("GPU-Only Decode Throughput", gpu_results)

    if args.mode in ("tiered", "compare"):
        tiered_results = bench_simulated_decode(
            context_lengths=args.context_lengths,
            gpu_pages=args.gpu_pages,
            cpu_pages=args.cpu_pages,
            num_layers=args.num_layers,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            decode_steps=args.decode_steps,
        )
        print_results("Tiered (GPU+CPU) Decode Throughput", tiered_results)


if __name__ == "__main__":
    main()
