# Tiered KV Cache 设计文档

## 概述

Tiered KV Cache 将 KV 缓存数据分布在 GPU (HBM)、CPU (pinned DRAM)、SSD (mmap) 三层存储上，通过 LRU 驱逐策略和异步逐层预取，在不增加 GPU 显存的前提下扩展有效 prefix 缓存容量。

## 核心不变量

**`page_id == GPU_offset`**：每个逻辑页永久拥有一个固定的 GPU buffer 槽位。跨层级数据迁移只移动数据，不重新分配 GPU 偏移。

这一设计消除了 GPU 偏移冲突和双重 free-list 同步问题。详见下文"设计决策"一节。

## 存储架构

```
GPU (HBM) ──驱逐──> CPU (pinned DRAM) ──驱逐──> SSD (mmap)
    <──预取/拷回──       <──预取/拷回──
```

- **TierPool**：管理单层物理 buffer 和空闲槽位（`_free_slots`）
- **LocationTable**：权威数据源，记录每个 page 的 `(tier, offset, last_access, access_count)`
- **TieredKVCachePool**：组合三个 TierPool + LocationTable，实现 `BaseKVCachePool` 接口

## 模块职责

- **CacheManager** (`scheduler/cache.py`)：管理所有 GPU 页的分配/释放，维护 `free_slots`，驱动 LocationTable 状态转换
- **LocationTable** (`kvcache/tiered_pool.py`)：`tiers[]` + `offsets[]` + `last_access[]`，所有驱逐/预取决策的数据源
- **TieredEvictionPolicy** (`kvcache/eviction.py`)：LRU 策略，将冷页从 GPU 降级到 CPU（或 CPU->SSD），不释放 GPU 偏移
- **PrefetchScheduler** (`kvcache/prefetch.py`)：异步流上逐层预取，Layer N 计算 attention 时并行搬运 Layer N+1 的数据

## 页面生命周期

```
free (tier=-1)
  |  CacheManager._allocate()
  v
GPU-active (tier=GPU, locked)     <-- 正在被 attention 使用
  |  cache_req() -> insert_prefix()
  v
GPU-cached (tier=GPU, unlocked)   <-- 在 prefix_cache 中，可被驱逐
  |  eviction._demote()
  v
CPU-cached (tier=CPU)             <-- 数据在 CPU，prefix_cache 条目保留
  |  ensure_batch_on_gpu() / prefetch_layer()
  v
GPU-cached (tier=GPU)             <-- 数据拷回原 GPU 槽位
  |  prefix_cache.evict() -> _allocate()
  v
free (tier=-1)                    <-- 若在 CPU/SSD 上则释放物理存储
```

## 驱逐策略（Eviction）

### 触发条件

当 `ensure_batch_on_gpu()` 发现 prefix-matched 的页不在 GPU 上，或 `prefix_cache.evict()` 回收 prefix 条目时触发相关逻辑。

### _demote 流程（GPU -> CPU 为例）

```python
# 1. 从 LocationTable 找 GPU tier 上未锁定的 LRU 页
evict_ids = _select_lru_pages(src=GPU, count=N)

# 2. 在 CPU tier 分配物理槽位
dst_offsets = cpu_pool.allocate(N)

# 3. 全层数据拷贝 GPU -> CPU
_copy_pages(GPU, CPU, evict_ids, src_offsets, dst_offsets)

# 4. 关键：GPU 偏移不归还给 gpu_pool（永久属于该 page_id）
if src != Tier.GPU:
    src_pool.free(src_offsets)

# 5. 更新 LocationTable: tier=CPU, offset=dst_offset
lt.update(evict_ids, CPU, dst_offsets)
```

### 锁机制

- `allocate_paged()` 时 lock 新分配的页，防止 attention 进行中被驱逐
- `_free()` / `lazy_free_region()` 时 unlock
- `_select_lru_pages()` 过滤掉所有 locked 页

### 递归驱逐

CPU 满时自动触发 CPU -> SSD 驱逐（或直接 release）以腾出空间。

## 预取策略（Prefetch）

### 逐层异步预取

在 `fi.py` 的 `forward()` 中，利用独立 CUDA stream 与 attention 计算重叠：

```
Layer 0: wait_prefetch() -> attention -> prefetch_layer(1)
Layer 1: wait_prefetch() -> attention -> prefetch_layer(2)
...
Layer N-1: wait_prefetch() -> attention
```

### prefetch_layer 流程

```python
# 1. 查 LocationTable，找出不在 GPU 上的页
tiers, offsets = lt.lookup(page_ids)
need_transfer = (tiers == CPU) | (tiers == SSD)

# 2. GPU 偏移 = page_id 本身（永久映射，无需分配）
gpu_offsets = transfer_ids.int()

# 3. 异步拷贝到 GPU 原位（non_blocking=True）
with torch.cuda.stream(prefetch_stream):
    gpu_buf[:, layer_id, d_off].copy_(
        cpu_buf[:, layer_id, s_off], non_blocking=True
    )

# 4. 释放 CPU/SSD 物理槽位
cpu_pool.free(cpu_src_offsets)
```

### CUDA Graph 兼容

CUDA graph capture 期间跳过 prefetch/wait 逻辑，因为 CPU<->GPU transfer 与 graph recording 不兼容。通过 `torch.cuda.is_current_stream_capturing()` 检测。

## 分配路径中的层级感知

当 `prefix_cache.evict()` 返回的页已被降级到 CPU/SSD 时，`_allocate()` 负责释放物理存储：

```python
tiers, offsets = lt.lookup(evicted_page_ids)
for tier, pool in [(CPU, cpu_pool), (SSD, ssd_pool)]:
    mask = tiers == tier.value
    if mask.any():
        pool.free(offsets[mask])   # 释放 CPU/SSD 物理槽位
lt.mark_free(evicted_page_ids)     # 标记为 free
# 后续重新标记为 GPU（复用原 GPU 槽位写入新 KV 数据）
```

## ensure_batch_on_gpu

在 `_prepare_batch` 中调用，保证 attention 之前所有 prefix-matched 页的数据在 GPU 上：

```python
# 发现 CPU/SSD 上的页 -> 拷回 GPU[page_id]（原位，不改 page_table）
gpu_buf[:, :, pid].copy_(src_pool.buffer[:, :, src_off])
src_pool.free(src_off)
lt.update(page_ids, GPU, page_ids.int())
```

因为 `page_id == GPU_offset`，page_table 已持有正确值，无需重写。

## 设计决策：为什么 page_id == GPU_offset 永久绑定

### 问题背景

如果跨层级迁移时重新分配 GPU 偏移，会导致：

1. **数据覆写**：页 X 降级后 GPU 偏移 X 被分配给页 Y，后续页 X 被 prefix_cache.evict 返回时 CacheManager 写入 GPU[X] 覆盖页 Y 的数据
2. **双重 free-list 不同步**：`CacheManager.free_slots` 和 `gpu_pool._free_slots` 各自独立管理同一批 GPU 偏移，`gpu_pool.free_count` 永远满，驱逐永远不触发
3. **page_table 频繁重写**：每次预取回 GPU 都可能拿到不同偏移

### 永久绑定的优势

- **无偏移冲突**：一个 GPU 槽位只属于一个 page_id，不同 page 不可能争抢
- **去掉 gpu_pool 分配**：核心路径不再使用 `gpu_pool.allocate/free`，同步问题从架构上消除
- **page_table 无需重写**：prefix_cache 存储的值就是 page_id = GPU 偏移，匹配后直接可用
- **并发安全**：page_id 在 free_slots / prefix_cache / 活跃请求 三种状态互斥，不可能同时有两个路径写入同一个 GPU 槽位

### 代价

GPU buffer 是启动时整块预分配的（`torch.empty(shape, device="cuda")`），页降级到 CPU 后 GPU 槽位中的过期数据占据的显存不会被释放。但这不是实际问题，因为无论数据是否有效，GPU KV cache buffer 的显存占用不变。

## 运行时可观测性

### debug_stats

`TieredKVCachePool.debug_stats()` 从 LocationTable 统计各 tier 的实际页数：

```
GPU=1234/288884 | CPU=5678/149796 | active=6912/438680
```

### 周期性日志

Scheduler 每 N 步打印 KV cache 状态：

```
[KV-step 100] batch=prefill(n=4) free_pages=1000 prefix_cache(evict=5000 prot=200 total=5200) tiered=[GPU=1234/288884 | CPU=5678/149796 | active=6912/438680]
```

## 配置

通过 Engine 启动参数控制：

- `--cpu-kv-cache-gb N`：启用 tiered KV cache，分配 N GB CPU pinned memory
- `--ssd-kv-cache-gb N`：额外分配 N GB SSD 存储（可选）
- `--ssd-kv-cache-path /path`：SSD mmap 文件路径
- `--kv-offload-strategy per-layer`：启用逐层预取策略

## 文件索引

| 文件 | 内容 |
|---|---|
| `kvcache/tiered_pool.py` | Tier 枚举、TieredCacheConfig、LocationTable、TierPool、TieredKVCachePool |
| `kvcache/eviction.py` | TieredEvictionPolicy（LRU 驱逐、锁、递归驱逐） |
| `kvcache/prefetch.py` | PrefetchScheduler（异步逐层预取） |
| `scheduler/cache.py` | CacheManager（分配/释放/ensure_batch_on_gpu/LocationTable 状态管理） |
| `attention/fi.py` | FlashInfer attention forward 中的 prefetch/wait 集成 |
| `engine/engine.py` | TieredKVCachePool 初始化和 Layer 0 预取触发 |
