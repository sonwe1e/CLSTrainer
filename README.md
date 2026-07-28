# CLSTrainer

双帧、多游戏二分类训练框架。输入为两张 `[B,3,448,208]` RGB 图像，模型输出
`[B,2]`，部署判定固定为第二通道 Softmax 概率严格大于 `0.99`。

## 安装

项目的基础安装不会安装或替换 PyTorch，避免破坏已经与 CANN 匹配的
`torch + torch_npu` 环境：

```bash
python -m pip install -e .
```

CUDA 开发环境可以安装：

```bash
python -m pip install -e ".[cuda]"
```

测试与 CPU CI 使用：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Ascend 环境应先按服务器 CANN 版本安装匹配的 PyTorch 和 TorchNPU，再执行基础安装。

正式 NPU 配置位于 `configs/npu_production.yaml`，不再继承 CUDA demo 配置。
其中的 `model.factory` 和 `model.checkpoint_path` 必须替换为真实模型；未替换、
checkpoint 不存在或非 `cls` 主干权重未完整加载时，训练会立即终止。

## 数据索引和严格审计

目录必须满足：

```text
<split>/<game>/<0|1>/<video_id><frame_id>.png
```

文件名满足 `^\d{2}\d{5}\.png$`。执行：

```bash
python tools/build_index.py \
  --train-root /data/train \
  --test-root /data/test \
  --output-dir indexes

python tools/audit_dataset.py \
  --index-dir indexes \
  --output-dir reports/data_audit \
  --strict
```

默认正式训练启用 `data.strict_audit`。尺寸、通道、文件名、类别完整性或
test `delta=2` pair 不符合要求时，训练会在创建 DataLoader 前终止。审计报告
同时列出 `game × label × delta` 的合法 pair 数；生产配置要求每个游戏、每个
标签至少存在一个 `delta=2` pair。

训练阶段不会物化数百万个 `PairSample`：内存中只保留视频级帧数组和各 delta
合法起点，sampler 在每个 step 懒生成 pair。测试 pair 使用 rank-local NumPy
紧凑数组，不补齐、不重复。

索引阶段还会生成 `train_video_entries.parquet` 和
`test_video_entries.parquet`，训练直接按视频行读取，不再让每个 rank 将百万帧
转换成 Python dict 和 `FrameRecord`。默认同时计算 SHA-256，严格审计会拒绝
train/test 中字节完全相同的图片。两位 `video_id` 默认按 split 内编号处理，
跨 split 重名只作为信息记录；只有确认它在整个项目中全局唯一后，才应启用
`require_unique_video_keys_across_splits` 强制检查。

## 训练

CPU/CUDA 合成数据 smoke test：

```bash
python tools/train.py \
  --config configs/cuda_debug.yaml \
  train.max_steps=2 \
  evaluation.quick_test_every_steps=1 \
  evaluation.full_test_every_steps=2
```

Ascend 单卡和八卡：

```bash
bash scripts/run_npu_1p.sh
bash scripts/run_npu_8p.sh
```

正式长跑前先执行固定验收入口：

```bash
bash scripts/smoke_npu_1p.sh
bash scripts/smoke_npu_8p.sh
```

它们分别运行 100 和 500 step，并确保触发 quick/full、报告合并和 checkpoint。
必须在真实 910B2 环境确认通过后，才能把 CPU/Gloo 测试结论扩展到 HCCL。

NPU 运行时会先导入 `torch_npu`、绑定设备，再初始化 HCCL。训练 batch 在 CPU
侧保持 `uint8`，一次传输到设备后再转换为 FP32/BF16/FP16 并归一化。

如 Profiler 确认 PNG 解码仍是瓶颈，可预解码为固定大小 uint8 分片：

```bash
python tools/pack_dataset.py \
  --frame-index indexes/train_frames.parquet \
  --output-dir /local_nvme/train_packed
```

分别打包 train/test，然后把生产配置的 `data.backend` 改为 `packed_uint8`，
并设置 frame/video 两组 packed index。也可直接以
`configs/npu_production_packed.yaml` 为模板。新版 packed 索引以连续整数定位
帧，不再为每个 rank 建立百万项路径字典；shard 路径相对 manifest 保存，运行时
仅维护最多 `packed_max_open_shards` 个 LRU memmap。该后端避免训练热路径中的
PNG 解压，数据仍应优先复制到本地 NVMe。

接入真实模型时，将 `model.factory` 设置为 `包名.模块名:函数名`。工厂函数必须
返回接受 `(image0, image1)` 并输出 `[B,2]` 的 `torch.nn.Module`。

## 评估契约

quick test 会从每个 `(game,label,video)` 的 `delta=2` pair 中按时间均匀选取固定
数量；full test 枚举全部合法 pair。分布式运行时，每个 rank 处理不重复分片，
混淆矩阵使用 int64、损失使用 float32 all-reduce，兼容 HCCL。full test 默认用
固定直方图分布式计算 ROC-AUC/PR-AUC，不再把百万 Python 分数集中到 rank 0；
错例在 batch 内流式写分片，再由 rank 0 流式合并。评估前向按
`evaluation.amp/amp_dtype` 使用与部署一致的 BF16/FP16；CE、Brier 和混淆矩阵
在设备上累计，结束时再统一归约，避免每个 batch 多次 `.item()` 同步。

full test 报告包含：

```text
metrics.json
metrics_by_game.csv
metrics_by_video.csv
metrics_by_game_label.csv
false_positive.parquet
false_negative.parquet
near_threshold.parquet
errors.html
```

quick test 仅写 `metrics.json`、受 `quick_save_error_limit` 全局限制的少量
FP/FN 与 near-threshold Parquet，不再生成 HTML 和分组 CSV，避免短周期评估
承担完整报告开销。同一步同时满足 quick/full 周期时只运行 full。

`near_threshold.parquet` 只保存 `0.98 <= p1 <= 0.995` 的样本；更高置信度只记录
区间计数。指标同时包含 Brier Score、20-bin ECE 和置信度直方图。由于训练采用
50/50 平衡采样并加入阈值损失，`0.99` 应解释为固定业务分数阈值，而不是天然
校准后的真实发生概率。

周期性 full test 的角色明确标记为 `observed_dev_test`。训练摘要同时记录 last
checkpoint 指标、best observed dev-test 指标，以及 quick/full test 的执行次数。
最佳模型由 `evaluation.selection_metric` 决定；生产配置使用 global、macro-game
和 worst-game F1 的组合分数，并可通过 `minimum_worst_game_f1` 阻止单个游戏
灾难性退化。单类 `by_game_label` 行不再展示无意义的 F1/AUC：正类报告 recall
和 FN rate，负类报告 specificity 和 FP rate。

## 精确恢复

完整 checkpoint 保存 epoch、`step_in_epoch`、global step、sampler 状态、优化器、
scheduler、scaler、CPU/CUDA/NPU RNG 和各 rank 独立 RNG。训练 pair 自带确定性增强
seed，因此恢复时可以直接从 epoch 内下一 batch 继续，不重新解码已经消费的 batch。

生产配置的周期性恢复 checkpoint 只保存可训练状态、优化器和基础权重哈希；
同时记录并严格核对预期 trainable state keys。完整快照按较低频率保存为
`model_<tag>_full.pth`，文件内明确记录 `global_step` 和 artifact role；周期恢复
文件保持为 `checkpoint_last.pth`。同一步同时触发 best 与 last 时只序列化一次，
再创建稳定别名，减少冻结主干的重复 I/O。

普通训练日志中的分段时间明确标记为 host enqueue 时间，不表示 NPU 实际算子
耗时；日志同时报告排除评估/保存的 `train_only_samples/s` 和包含全部停顿的
`wall_samples/s`。设备级瓶颈必须使用 NPU Event 或 TorchNPU Profiler 验证。
