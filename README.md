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
test `delta=2` pair 不符合要求时，训练会在创建 DataLoader 前终止。

训练阶段不会物化数百万个 `PairSample`：内存中只保留视频级帧数组和各 delta
合法起点，sampler 在每个 step 懒生成 pair。测试 pair 使用 rank-local NumPy
紧凑数组，不补齐、不重复。

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

NPU 运行时会先导入 `torch_npu`、绑定设备，再初始化 HCCL。训练 batch 在 CPU
侧保持 `uint8`，一次传输到设备后再转换为 FP32/BF16/FP16 并归一化。

接入真实模型时，将 `model.factory` 设置为 `包名.模块名:函数名`。工厂函数必须
返回接受 `(image0, image1)` 并输出 `[B,2]` 的 `torch.nn.Module`。

## 评估契约

quick test 会从每个 `(game,label,video)` 的 `delta=2` pair 中按时间均匀选取固定
数量；full test 枚举全部合法 pair。分布式运行时，每个 rank 处理不重复分片，
混淆矩阵和交叉熵通过 all-reduce 汇总，各 rank 写独立错例分片，由 rank 0 合并。

报告包含：

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

周期性 full test 的角色明确标记为 `observed_dev_test`。训练摘要同时记录 last
checkpoint 指标、best observed dev-test 指标，以及 quick/full test 的执行次数。

## 精确恢复

完整 checkpoint 保存 epoch、`step_in_epoch`、global step、sampler 状态、优化器、
scheduler、scaler、CPU/CUDA/NPU RNG 和各 rank 独立 RNG。训练 pair 自带确定性增强
seed，因此恢复时可以直接从 epoch 内下一 batch 继续，不重新解码已经消费的 batch。
