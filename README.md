[![CPU CI](https://github.com/OWNER/clstrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/clstrainer/actions/workflows/ci.yml)
[![NPU gate](https://github.com/OWNER/clstrainer/actions/workflows/npu-ci.yml/badge.svg)](https://github.com/OWNER/clstrainer/actions/workflows/npu-ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](#安装)
[![License](https://img.shields.io/badge/license-proprietary-lightgrey)](#)

# CLSTrainer

双帧、多游戏二分类训练框架。输入两张 `[B,3,208,448]` RGB 帧，模型输出
`[B,2]`，部署判定以第二通道概率与业务阈值 `decision.threshold`（默认 `0.99`）
比较，训练损失与评估器共用同一阈值来源。

- [快速开始](#快速开始)
- [了解更多](#了解更多)
- [特性](#特性)

## About

*两帧进，一判出——CLSTrainer 让"低误报优先"的双帧分类训练开箱即用。*

框架面向"楼梯 vs 地板/木桥"这类需要在极低误报下工作的生产任务：自动按源视频划分
train/val/test、困难负样本闭环（逐视频 `negative_subtype` 元数据与 hard-negative
mixing）、FPR 受约束的模型选择、分阶段解冻，以及从数据准备到 NPU 验收的完整工具链。

它同时支持 **CUDA** 与 **Ascend NPU**（CANN / torch_npu，含 HCCL 八卡），并保证
**可复现**：同一数据、seed 与算法版本产生字节级一致的划分与训练轨迹。每个 Run
都是不可覆盖的目录，`--resume`/`--fork` 记录完整谱系。

我们欢迎 [贡献](https://github.com/OWNER/clstrainer)。如果你希望训练更快 🔨、
调试更有信心 📚、部署更可靠 💖，CLSTrainer 就是为此而建。

## 快速开始

下面的快速开始带你完成一次合成数据 smoke 训练。真实数据的完整操作路径见
[`tutorial.html`](tutorial.html)。

### 1. 安装

#### 环境要求

- 一个 CUDA 环境，**或** Ascend NPU（按服务器 CANN 版本安装匹配的
  `torch + torch_npu`）。基础安装**不会**安装或替换 PyTorch，避免破坏
  与 CANN 匹配的既有环境。

#### 创建环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

#### 安装 CLSTrainer

**基础安装**（最小依赖：numpy / Pillow / PyYAML / pyarrow，见
[`requirements.txt`](requirements.txt)）：

```bash
python -m pip install -e .
```

**OR** CUDA 开发环境（额外安装 torch / torchvision）：

```bash
python -m pip install -e ".[cuda]"
```

**OR** 测试 / CPU CI（含 pytest，见 [`requirements-dev.lock`](requirements-dev.lock)）：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

### 2. 第一次训练！

```bash
# 预检环境、数据、模型和配置：
cls-trainer doctor --config configs/recipes/example_debug.yaml

# 在合成数据上跑一个最小训练：
cls-trainer train --config configs/recipes/example_debug.yaml train.max_steps=2
```

如果一切正常，你会看到类似下面的进度输出，结束时的终端输出会给出
`summary.md` 与 `overview.html` 等关键文件路径：

```text
step=2/2 loss=1.14 ... interval_samples/s=... wall_samples/s=...
```

每次 `train` 启动都会在 `experiment.output_dir` 下分配一个**不可覆盖**的 Run 目录
（`manifest.json`、`status.json`、`summary.md`、`overview.html`、`checkpoints/` 等）。
重复执行同一条命令永远不会覆盖上一次实验。

### 3. 配置你的真实训练

配置分四层，普通用户只维护 **Recipe**：

```text
task_profile → profile → presets → recipe → 命令行覆写
```

```text
configs/
├── task_profiles/dual_frame_binary.yaml    # 业务默认值：阈值 0.99、test delta=2 等
├── profiles/                               # cpu_debug / cuda_1p / npu_1p / npu_8p
├── presets/
│   ├── augmentation/   none | light | standard
│   ├── dataloader/     stable | throughput
│   └── evaluation/     smoke | production
├── recipes/
│   ├── example_debug.yaml                  # 合成数据，开箱即跑
│   └── game_cls_production.yaml            # 生产模板
├── cuda_debug.yaml  npu_1p.yaml  npu_8p.yaml  npu_production.yaml  npu_production_packed.yaml
└── (旧式扁平配置，`base:` 继承链：npu_8p → npu_1p → npu_production)
```

`configs/profiles/` 下的同名文件（`cuda_1p` 等）是 Recipe 分层用的 profile 层，
**只含环境差异，不能独立训练**；根目录的扁平版（`cuda_debug.yaml` 等）才包含完整
data/model/train 配置，可直接 `cls-trainer train --config configs/cuda_debug.yaml`。
新项目推荐走 Recipe + profile/presets 分层。

任何配置都会先经过**严格 Schema 校验**：未知键报错并给出"你是否想写……"的候选，
从未被消费的字段（`optimizer.name` 等）给出迁移说明，`decision.threshold` 是唯一
业务阈值来源。完整键表见 `cls-trainer config reference` 或
[`docs/config_reference.md`](docs/config_reference.md)。

```bash
# 校验并查看合并后的配置：
cls-trainer config validate --config configs/recipes/game_cls_production.yaml
cls-trainer config show --config configs/recipes/game_cls_production.yaml --with-source

# 以某个 Recipe 训练，命令行覆写用于 A/B：
cls-trainer train --config configs/recipes/game_cls_production.yaml \
  presets.dataloader=throughput

# 先 dry-run 预览运行计划（不写任何文件）：
cls-trainer train --config configs/recipes/game_cls_production.yaml --dry-run
```

### 4. 准备自己的数据

目录必须满足 `<split>/<game>/<0|1>/<video_id><frame_id>.png`（文件名
`^\d{2}\d{5}\.png$`）。推荐用 `dataset prepare` 从单一 `train_all` 根目录自动做
**源视频级** train/val 划分，并用持久化 manifest 保证划分可复现、不跨 split：

```bash
cls-trainer dataset prepare \
  --config configs/recipes/game_cls_production.yaml \
  --train-root /data/train_all --test-root /data/test --val-ratio 0.10

cls-trainer dataset audit --config configs/recipes/game_cls_production.yaml
```

| 数据源 | 入口 | 说明 |
| --- | --- | --- |
| 📷 原始帧 | `dataset prepare` → `dataset audit` | 自动划分 + SHA-256 泄漏检查 |
| 📦 高吞吐 packed | `dataset pack` | uint8 分片，避免训练热路径 PNG 解码 |
| 🛠 逐视频元数据 | `dataset annotate` | `negative_subtype` / `sample_weight` 侧车 |

### 5. 高级选项

#### 训练后评估（test split 只评估一次，绝不参与选模）

```bash
cls-trainer evaluate --run <RUN_ID> --checkpoint best_selection --split test
```

没有独立 test 集（`data.val_index` 缺失、test 被当作 validation）时，`--split test`
会被拒绝并返回退出码 3。`--checkpoint` 支持 `last` / `best_selection` /
`best_val_loss` / `best_worst_game` 等稳定别名。

#### 恢复与派生

```bash
cls-trainer train --resume <run 目录>     # 按身份语义精确继续
cls-trainer train --fork <run>            # 以某 run 的配置为新起点
cls-trainer run list / show / compare / export-tensorboard
```

#### 困难负样本挖掘与 challenge 基准

```bash
# 用最佳 checkpoint 扫描训练侧负样本池，产出 hard_negatives.parquet：
cls-trainer benchmark scan-negatives --run <RUN_ID> --checkpoint best_selection
# 将挖掘结果标注为困难负样本并写入侧车：
cls-trainer dataset annotate --config configs/recipes/game_cls_production.yaml \
  --from-mining indexes/hard_negatives.parquet --subtype wooden_bridge
# 在固定 challenge 集上做不可漂移的验收基准：
cls-trainer benchmark evaluate --run <RUN_ID> --checkpoint best_selection
```

#### NPU 单卡 / 八卡

```bash
bash scripts/run_npu_1p.sh
bash scripts/run_npu_8p.sh
```

正式长跑前先跑固定验收入口 `scripts/smoke_npu_1p.sh` 与 `scripts/smoke_npu_8p.sh`。
真实生产配置（`configs/npu_production.yaml`）的 `model.factory` 与
`model.checkpoint_path` **必须**替换为真实模型，且非 `cls` 主干权重必须 100% 加载；
未替换或 checkpoint 不存在时 `doctor` 与训练都会立即失败——这是预期行为，不是 bug。

## 了解更多

| 章节 | 描述 |
| --- | --- |
| 🎒 **入门** | |
| [教程 `tutorial.html`](tutorial.html) | 自包含中文教程：目的 → 架构 → 数据 → 训练评估 → NPU 验收 |
| [配置参考](docs/config_reference.md) | 每个配置键的类型、默认值与含义 |
| 💻 **开发者** | |
| [CI 状态](https://github.com/OWNER/clstrainer/actions) | CPU：ruff + pytest（3.11/3.12/3.13）+ wheel；NPU：自托管 1P/8P 门禁 |
| NPU 算子检查 | `tools/check_npu_ops.py` 在 `doctor --config npu` 时探测 bincount/HCCL 等 |

## 特性

- 🔎 **不可覆盖的 Run** — 每个训练都是带 manifest / status / 一页摘要的版本化目录，重复执行永不覆盖，`--resume`/`--fork` 记录完整谱系。
- ✏️ **严格配置** — 未知键报错并提示候选；`decision.threshold` 是训练、评估与部署共用的唯一业务阈值。
- 📈 **低误报优先的评估** — train/val/test 三分协议、FPR 受约束的模型选择（`selection_mode: constrained`，max_global_fpr 0.01 / max_worst_game_fpr 0.02 / min_positive_recall 0.8）、Brier/ECE 与负样本尾部指标。
- 🛡 **数据闭环** — 源视频级自动划分、SHA-256 泄漏审计、逐视频 `negative_subtype` 元数据与困难负样本 mixing、固定 challenge 集验收。

## Built On

[![PyTorch](https://img.shields.io/badge/-PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
- 核心张量框架，CUDA / Ascend `torch_npu` 后端。

[![Ascend](https://img.shields.io/badge/-Ascend%20NPU-0062ff)](https://www.hiascend.com/)
- CANN / HCCL 运行时与 NPU 验收门禁。

## Citation

如果你在研究中使用了 CLSTrainer，请考虑引用：

```bibtex
@software{clstrainer,
  title  = {CLSTrainer: A Production Dual-Frame Binary Classification Training Framework},
  author = {CLSTrainer contributors},
  year   = {2026},
  url    = {https://github.com/OWNER/clstrainer},
}
```
