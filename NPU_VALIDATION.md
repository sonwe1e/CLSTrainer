# Ascend / HCCL validation checklist

CPU/Gloo 可以验证 DDP 编排，但不能证明 CANN、torch_npu、HCCL 和业务模型算子可用。最终必须在真实 Ascend 服务器验收。

## 1. 环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 按服务器实际路径
python tools/check_npu.py
```

预期至少看到 NPU available、device count、BF16 基础算子可用。

## 2. 安装

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

不要让 pip 自动替换当前与 CANN 匹配的 torch/torch_npu。

## 3. 单卡

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
clstrainer-lite check --config configs/example_npu_8p.yaml

ASCEND_RT_VISIBLE_DEVICES=0 \
clstrainer-lite train --config configs/example_npu_8p.yaml \
  train.epochs=1 train.batch_size=8 train.num_workers=0
```

确认输入 `[B,2,3,H,W]`、输出 `[B,2]`、BF16 forward/backward、checkpoint 与诊断图正常。

## 4. 两卡 HCCL

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m clstrainer_lite.cli train \
  --config configs/example_npu_8p.yaml \
  train.epochs=1 train.batch_size=8 train.num_workers=1
```

确认 `world_size=2 backend=hccl`，只有 rank0 写 run 产物。

## 5. 八卡 HCCL

```bash
bash scripts/run_npu_8p.sh configs/example_npu_8p_aug_focal.yaml
```

`train.batch_size` 是每 rank local batch。

## 6. 正确性检查

- val/test `samples` 与真实数据一致；
- TP+FP+FN+TN = samples；
- per-game samples 求和 = overall samples；
- `history.json` 与单独 `eval` 的同 checkpoint 指标一致；
- `best_model.pt` 可被业务模型严格 load；
- balanced sampler 开启时每 rank step 数一致；
- `find_unused_parameters=true` 下无 DDP reduction error。

## 7. 性能检查

重点记录首 batch 时间、samples/s、NPU utilization、CPU utilization。Image backend 与 Video backend 分别扫描：

- `num_workers`: 0/1/2/4/8；
- `prefetch_factor`: 2/4；
- Video `chunk_frames`: 64/128/256；
- Video `cache_chunks`: 1/2/4；
- Video `ffmpeg_threads`: 1/2/4。

不要只根据 CPU 核数堆 worker × ffmpeg thread；192 核也需要留出训练主进程和系统余量。
