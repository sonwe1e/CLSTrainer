# CLSTrainer Lite 0.4.0

一个保持“小、直观、可修改”的双帧二分类训练器，支持 PNG 与压缩视频两种数据后端、在线 delta、成对增强、CE/Focal Loss、周期 val/test、分游戏诊断、平衡采样，以及 CPU/CUDA/Ascend HCCL DDP。

## 核心约定

模型接口固定为：

```python
logits = model(image0, image1)  # [B, 2]
```

只配置 train/test 两个物理数据池；validation 始终从 train 内按完整 video 切出，避免相邻帧泄漏。

```text
train pool -> train videos + val videos
test pool  -> independent test videos
```

best 权重只由 validation 决定。test 只用于周期观察与最终评估，不参与模型选择。

## 安装

Ascend 环境建议保留服务器已有、与 CANN 匹配的 torch/torch_npu：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

CPU 开发环境可额外：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

快速验证：

```bash
bash scripts/demo_cpu.sh
python -m pytest -q
```

## 数据后端

### 1. Image backend

目录：

```text
train_root/
├── game_a/
│   ├── 0/
│   │   ├── 0000000.png
│   │   ├── 0000001.png
│   │   └── ...
│   └── 1/
└── game_b/

test_root/
└── 同样结构
```

PNG 文件名使用 `VVFFFFF.png`：2 位 video id + 5 位 frame id。

配置：

```yaml
data:
  backend: image
  image:
    train_root: /data/png/train
    test_root: /data/png/test
    strict_filenames: true
```

### 2. Video backend

压缩视频目录保持同样的 `game/label/video` 语义：

```text
compressed/train/
├── game_a/
│   ├── 0/
│   │   ├── 00.mp4
│   │   └── 01.mp4
│   └── 1/
└── game_b/

compressed/test/
└── 同样结构
```

配置：

```yaml
data:
  backend: video
  video:
    train_root: /data/compressed/train
    test_root: /data/compressed/test
    extensions: [".mp4"]
    chunk_frames: 128
    cache_chunks: 2
    ffmpeg_bin: ffmpeg
    ffprobe_bin: ffprobe
    ffmpeg_threads: 2
```

VideoPairDataset 采用 worker-local chunk cache。训练时不是“一 pair 一次打开视频”，而是 seek 后连续解码一个 chunk，再从缓存中取 `t` 与 `t+delta`。

## 在线 delta

训练 delta 是时序增强，不会把 `[1,2,3]` 预展开成三倍 PairPosition：

```yaml
data:
  train_delta_range: [1, 3]
  train_delta_probabilities: [0.15, 0.70, 0.15]
  eval_delta: 2
```

每个 train sample 在 `__getitem__` 中在线选择 delta。val/test 固定 `eval_delta`，保证不同 step 的指标可比较。

ImagePairDataset 与 VideoPairDataset 都遵循同一语义。

## 增量视频转码

`tools/transcode_videos.py` 把原始 train/test 视频镜像到新的压缩目录，并保存 `.transcode_manifest.json`。已经成功处理且源文件/参数未变化的视频会跳过；新增视频只处理新增项。

指定的 resize 流水线：

```text
source frame
 -> Torch bilinear 832x1792
 -> Torch bilinear 416x896
 -> DPID 2x, lambda=3
 -> 208x448
 -> H.264 training video
```

示例：

```bash
python tools/transcode_videos.py \
  --train-root /data/raw/TRAINDATA \
  --test-root /data/raw/TESTDATA \
  --output-root /data/STAIRDATA_COMPRESSED \
  --workers 4 \
  --batch-frames 2 \
  --torch-threads 8 \
  --decoder-threads 2 \
  --encoder-threads 4
```

默认编码偏向训练质量和随机访问：`libx264 / CRF 12 / yuv444p / GOP 12`。如源视频可能被同名覆盖且希望强内容校验，可加 `--fingerprint sha256`。

## 成对数据增强

增强只用于 train。几何变换永远同步到两帧，避免人为制造错误运动关系。

```yaml
augment:
  enabled: true
  horizontal_flip_p: 0.0
  crop_scale: [0.92, 1.0]
  brightness: 0.10
  contrast: 0.10
  saturation: 0.08
  gamma: [0.92, 1.08]
  color_shared: true
  noise_std: 0.01
  erase_p: 0.05
  erase_scale: [0.02, 0.06]
```

## CE / Focal Loss

```yaml
loss:
  type: focal          # cross_entropy | focal
  gamma: 1.5
  alpha: [1.0, 1.0]
```

统计实际 train 分布：

```bash
python tools/count_class_distribution.py \
  --config configs/example_npu_8p_aug_focal.yaml
```

输出 `class_distribution.csv`、`game_balance.csv` 与 `focal_loss_recommendation.json`。权重只基于 actual train，不读取 test 分布来调参。

## 统一 Game × Class × Video 平衡采样

图片和视频后端共用一个 `BalancedVideoSampler`，采样逻辑只有一份：

```text
game -> class -> video -> position -> dataset online delta
```

配置：

```yaml
sampler:
  enabled: true
  class_probability: [0.5, 0.5]
  game_balance_alpha: 0.5
  samples_per_epoch: null
```

`game_balance_alpha`：

- `0.0`：接近按原始 sampling position 数量分配游戏概率；
- `0.5`：sqrt 平衡，推荐初始值；
- `1.0`：所有游戏等概率。

同一 game/class 内按 video 均匀采样，避免长视频仅因帧更多而支配训练。

## 诊断系统

每次 val/test 都会进行 DDP-safe 聚合，不保存海量逐样本 logits，而是保存足够定位问题的统计：

- overall loss / accuracy / precision / recall / F1；
- class0/class1 precision / recall / F1 / support；
- 每个游戏的同类指标与 TP/FP/FN/TN；
- TP/TN/FP/FN 的 confidence mean / P10 / P50 / P90；
- true class0/class1 的 `P(class=1)` 直方图。

run 目录：

```text
runs/<timestamp>_<name>/
├── config.json
├── history.json
├── history.csv
├── metrics_detail.json
├── summary.json
├── loss_curve.png
├── f1_curve.png
├── class_metrics_curve.png
├── best_val_diagnostics.png
├── final_test_diagnostics.png
└── checkpoints/
    ├── best_model.pt
    ├── last_model.pt
    └── last_checkpoint.pt
```

`summary.json` 保存 best validation 的完整诊断以及 final val/test 诊断，但不会把 test 参与 best model 选择。

## 周期 val/test

```yaml
train:
  log_every_steps: 50
  n_val_step: 2000
  n_test_step: 5000
```

step 指 optimizer step。训练最后一步强制执行一次 val + test。

## DDP / HCCL

```yaml
runtime:
  accelerator: npu
  backend: hccl
  amp: true
  amp_dtype: bfloat16
  find_unused_parameters: true
```

`find_unused_parameters` 默认 `true`，适合存在动态分支/部分迭代参数未参与 loss 的模型；若确认所有 trainable 参数每步都参与 backward，可设为 `false` 减少 DDP graph traversal 开销。

8 卡：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m clstrainer_lite.cli train \
  --config configs/example_npu_8p_aug_focal.yaml
```

NPU DataLoader 在 worker>0 时使用 `spawn`。val/test 使用不 padding 的 `DistributedEvalSampler`，避免分布式评估重复样本。

## 模型接入

内置 tiny model 仅用于 smoke：

```yaml
model:
  factory: clstrainer_lite.models:build_tiny_model
```

业务模型：

```yaml
model:
  factory: your_package.models:build_model
  kwargs: {}
  init_weights: /path/to/base_model.pt
```

factory 必须是当前 Python 环境可 import 的 `module:function`。

## CLI

```bash
clstrainer-lite check --config configs/example_cpu.yaml
clstrainer-lite train --config configs/example_cpu.yaml
clstrainer-lite eval --config configs/example_cpu.yaml \
  --checkpoint runs/.../checkpoints/best_model.pt --split test
clstrainer-lite plot --history runs/.../history.json
```

配置支持命令行覆盖：

```bash
clstrainer-lite train --config configs/example_npu_8p.yaml \
  train.batch_size=64 sampler.enabled=true
```

## 目录结构

```text
src/clstrainer_lite/
├── augments.py       # 双帧同步增强
├── checkpoint.py     # best/last/full checkpoint
├── cli.py
├── config.py
├── data.py           # ImagePairDataset + train/val video split + backend dispatch
├── distributed.py    # CPU/CUDA/NPU + gloo/nccl/hccl
├── losses.py         # Focal Loss
├── metrics.py        # overall + per-class + per-game diagnostics
├── models.py         # factory loader + tiny model
├── plotting.py       # run 曲线与诊断图
├── sampler.py        # 统一 game/class/video balanced sampler
├── trainer.py
└── video_data.py     # VideoPairDataset + ffmpeg chunk cache

tools/
├── check_npu.py
├── count_class_distribution.py
├── make_demo_data.py
└── transcode_videos.py
```

## 测试

```bash
python -m compileall -q src tests tools
python -m pytest -q
```

真实 Ascend/HCCL 的最终验收步骤见 `NPU_VALIDATION.md`。
