# mini-sglang 支持 Qwen3-VL-2B

在 mini-sglang 中支持 `Qwen/Qwen3-VL-2B-Instruct`，优先实现“可正确跑通”的图片理解能力，而不是先追求性能。

约束如下：

- 第一阶段只支持图片，不支持视频。
- 复杂模块优先采用 torch native 实现，不依赖 Triton、自定义 CUDA kernel 或高性能 DP encoder 路径。
- 优先复用 mini-sglang 现有文本推理链路；仅在多模态必需处新增数据结构和逻辑。
- 优先复用 sglang 中已经验证过的模型结构、权重映射和 MRoPE 规则，但不整块搬运其完整多模态框架。
- 建议第一阶段先把多模态能力限制为 `TP=1`、eager 模式、`/v1/chat/completions` 图片输入，待功能稳定后再扩展。

## 当前差距

mini-sglang 当前已经具备 Qwen3 文本模型基础，但距离 Qwen3-VL-2B 仍有 5 个关键缺口：

1. 模型注册缺失：`python/minisgl/models/register.py` 还没有 `Qwen3VLForConditionalGeneration`。
2. 配置抽象不完整：`python/minisgl/models/config.py` 配置vision、多模态 token id、MRoPE section 等 VL 专属字段。
3. 权重加载视觉部分：`python/minisgl/models/weight.py`。
4. RoPE 只支持 1D 文本位置：`python/minisgl/layers/attention.py` 和 `python/minisgl/layers/rotary.py` 只处理一维 positions，不支持 Qwen3-VL 的 3 维 MRoPE。
5. 请求链路只有文本：`python/minisgl/server/api_server.py`、`python/minisgl/message/tokenizer.py`、`python/minisgl/tokenizer/tokenize.py`、`python/minisgl/scheduler/scheduler.py` 都只面向 text-only 输入。

## 总体迁移策略

总体策略是“复用现有 Qwen3 文本主干 + 新增一套最小多模态补丁层”。

- 文本解码层尽量复用 `python/minisgl/models/qwen3.py` 和 `python/minisgl/models/utils.py`。
- 视觉编码器单独新增，使用 torch native attention。
- tokenizer worker 侧引入 `AutoProcessor`，负责图片预处理、chat template、input_ids、`image_grid_thw` 生成。
- scheduler 只负责把多模态字段带到 batch 中；真正的图片 embedding 替换在模型 forward 内完成。
- MRoPE 用原生 PyTorch 实现，1D 文本沿用现有逻辑，2D/3D positions 走新分支。

## 耗时长问题

```bash
INFO:     127.0.0.1:14906 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-04-11|04:07:36] INFO     [profile] image_io.load_image source=remote_url took 1140.09 ms
[2026-04-11|04:07:36] INFO     [profile] qwen3_vl.hf_processor images=1 prompt_chars=103 took 157.78 ms
[2026-04-11|04:07:36] INFO     [profile] qwen3_vl.process_messages images=1 tokens=2767 took 1301.22 ms
[2026-04-11|04:07:36] INFO     [profile] tokenize.multimodal.processor tokens=2767 took 1301.32 ms
[2026-04-11|04:07:36] INFO     [profile] tokenize.multimodal.total images=1 tokens=2767 took 1302.52 ms
[2026-04-11|04:08:56|core|rank=0] INFO     [profile] vision.patch_embed images=1 patches=11008 took 78917.19 ms
[2026-04-11|04:08:56|core|rank=0] INFO     [profile] vision.abs_pos_embed patches=11008 took 211.42 ms
[2026-04-11|04:08:56|core|rank=0] INFO     [profile] vision.position_prep segments=1 took 15.86 ms
[2026-04-11|04:08:57|core|rank=0] INFO     [profile] vision.blocks layers=24 patches=11008 took 370.41 ms
[2026-04-11|04:08:57|core|rank=0] INFO     [profile] vision.forward_total images=1 merged_tokens=2752 took 79738.51 ms
[2026-04-11|04:08:57|core|rank=0] INFO     [profile] qwen3_vl.visual_encode image_count=1 image_tokens=2752 took 79750.53 ms
[2026-04-11|04:08:57|core|rank=0] INFO     [profile] qwen3_vl.decoder_prefill tokens=2764 took 23.31 ms
[2026-04-11|04:08:57|core|rank=0] INFO     [profile] qwen3_vl.forward_prefill tokens=2764 batch=1 took 79819.63 ms
```

根因：Qwen3VLVisionPatchEmbed 使用 nn.Conv3d，kernel_size == stride == (2, 16, 16)。首次调用时 cuDNN 对这个 3D 卷积形状做算法搜索，耗时 78.9 秒，占总 79.8 秒的 99%。

修复（python/minisgl/layers/vision.py:48）：当 kernel == stride 时，Conv3d 数学上等价于 reshape + 矩阵乘。改用 F.linear 执行，保留 nn.Conv3d 做权重存储（兼容现有 checkpoint）。这样：

完全绕过 cuDNN 算法选择，消除 78.9 秒的首次延迟
F.linear → cuBLAS GEMM，无 autotuning 开销
权重形状不变，现有 checkpoint 直接加载
后续调用也更快（GEMM 比 Conv3d 路径效率更高）
把 Conv3d 替换成等价的 F.linear。当 kernel == stride 时，Conv3d 等价于 reshape → 矩阵乘，完全绕过 cuDNN。

```bash
[2026-04-11|04:24:48] INFO     [profile] image_io.load_image source=remote_url took 1079.26 ms
[2026-04-11|04:24:48] INFO     [profile] qwen3_vl.apply_chat_template messages=1 images=1 took 0.40 ms
[2026-04-11|04:24:48] INFO     [profile] qwen3_vl.hf_processor images=1 prompt_chars=103 took 70.64 ms
[2026-04-11|04:24:48] INFO     [profile] qwen3_vl.process_messages images=1 tokens=2767 took 1154.00 ms
[2026-04-11|04:24:48] INFO     [profile] tokenize.multimodal.processor tokens=2767 took 1154.10 ms
[2026-04-11|04:24:48] INFO     [profile] tokenize.multimodal.mrope image_count=1 took 0.84 ms
[2026-04-11|04:24:48] INFO     [profile] tokenize.multimodal.total images=1 tokens=2767 took 1155.39 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] vision.patch_embed images=1 patches=11008 took 0.72 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] vision.abs_pos_embed patches=11008 took 14.38 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] vision.position_prep segments=1 took 10.25 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] vision.blocks layers=24 patches=11008 took 159.97 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] vision.merge merged_tokens=2752 took 0.24 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] vision.forward_total images=1 merged_tokens=2752 took 198.16 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] qwen3_vl.visual_encode image_count=1 image_tokens=2752 took 209.63 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] qwen3_vl.decoder_prefill tokens=2764 took 233.33 ms
[2026-04-11|04:24:49|core|rank=0] INFO     [profile] qwen3_vl.forward_prefill tokens=2764 batch=1 took 487.07 ms
[2026-04-11|04:24:56|core|rank=0] INFO     Scheduler is idle, waiting for new reqs...
```