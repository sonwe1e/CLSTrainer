[![CPU CI](https://github.com/sonwe1e/CLSTrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/sonwe1e/CLSTrainer/actions/workflows/ci.yml)
[![NPU gate](https://github.com/sonwe1e/CLSTrainer/actions/workflows/npu-ci.yml/badge.svg)](https://github.com/sonwe1e/CLSTrainer/actions/workflows/npu-ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](#安装)
[![License](https://img.shields.io/badge/license-proprietary-lightgrey)](#)

# CLSTrainer

双帧、多游戏二分类训练框架。输入两张 `[B,3,208,448]` RGB 帧，模型输出
`[B,2]`，部署判定以第二通道概率与业务阈值 `decision.threshold`（默认 `0.99`）
比较——训练损失、评估器与导出 manifest 共用同一阈值来源，不会因环节不同而漂移。

- [快速开始](#快速开始)
- [特性](#特性)
- [了解更多](#了解更多)

## About

*两帧进，一判出——CLSTrainer 让"低误报优先"的双帧分类训练开箱即用。*

框架面向"楼梯 vs 地板/木桥"这类需要在极低误报下工作的生产任务，覆盖从数据准备到
NPU 验收的完整工具链：

| 能力 | 说明 |
| --- | --- |
| **数据闭环** | 源视频级 train/val 划分（`data.source_video_identity.mode` 显式建模源身份，混合标签源视频原子划分）、SHA-256 泄漏审计、packed uint8 分片、逐视频 `negative_subtype` 元数据、困难负样本挖掘与 mixing |
| **训练** | FPR 受约束的 `constrained` 模型选择、分阶段解冻（`trainable_rules`）、早停 selection 合约、top-k checkpoint 注册表 |
| **评估** | 全局/最差游戏/亚型三维 FPR-recall 矩阵、固定 challenge 集验收、Brier/ECE/负样本尾部分位数 |
| **导出** | TorchScript 权重 + ONNX 导出、部署 manifest（含阈值/输入形状）、release gate 自动拦截未达标模型 |
| **硬件** | CUDA 单/多卡 + Ascend NPU（CANN / HCCL 八卡），可复现：同一 seed 产生字节级一致划分 |

每个 Run 都是不可覆盖的目录，`--resume`/`--fork` 记录完整谱系。

---

## 快速开始

### 1. 安装

#### 环境要求

- CUDA 环境，**或** Ascend NPU（按服务器 CANN 版本安装匹配的 `torch + torch_npu`）。
  基础安装**不会**安装或替换 PyTorch，避免破坏与 CANN 匹配的既有环境。

#### 创建环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

#### 安装 CLSTrainer

**基础安装**（最小依赖：numpy / Pillow / PyYAML / pyarrow）：

```bash
python -m pip install -e .
```

**CUDA 开发环境**（额外安装 torch / torchvision）：

```bash
python -m pip install -e ".[cuda]"
```

**测试 / CPU CI**（含 pytest）：

```bash
python -m pip install -e ".[dev]"
python -m pytest   # 427+ 测试全部通过
```

### 2. 第一次训练

```bash
# 预检环境、数据、模型和配置：
cls-trainer doctor --config configs/recipes/example_debug.yaml

# 在合成数据上跑最小训练（2 步，< 10 秒）：
cls-trainer train --config configs/recipes/example_debug.yaml train.max_steps=2
```

结束时输出类似：

```text
step=2/2 loss=1.14 ... interval_samples/s=... wall_samples/s=...
── Run complete ──────────────────────────────
  run dir   : outputs/runs/20260806_143200_abc123/
  summary   : outputs/runs/.../summary.md
  best ckpt : outputs/runs/.../checkpoints/best_selection.pt
```

每次 `train` 启动都会在 `experiment.output_dir` 下分配一个**不可覆盖**的 Run 目录。
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
└── recipes/
    ├── example_debug.yaml                  # 合成数据，开箱即跑
    ├── npu_synthetic_smoke.yaml            # NPU 预检：合成数据，可在未上数据的 Ascend 主机运行
    ├── game_cls_production.yaml            # 生产模板（默认 8 卡）
    └── game_cls_release.yaml               # 带 release gate 的发布模板
```

生产模板默认 `profile: npu_8p`；单卡或 packed 数据后端通过命令行覆写切换，例如
`profile=npu_1p experiment.output_dir=runs/game_cls_1p` 或
`data.backend=packed_uint8`（完整覆写示例见 `game_cls_production.yaml` 头部注释）。

任何配置都会先经过**严格 Schema 校验**：未知键报错并给出候选，`decision.threshold`
是唯一业务阈值来源。完整键表见
[`docs/config_reference.md`](docs/config_reference.md) 或：

```bash
cls-trainer config reference
```

```bash
# 校验并查看合并后的配置（含来源标注）：
cls-trainer config validate --config configs/recipes/game_cls_production.yaml
cls-trainer config show     --config configs/recipes/game_cls_production.yaml --with-source

# dry-run 预览运行计划（不写任何文件）：
cls-trainer train --config configs/recipes/game_cls_production.yaml --dry-run

# 以某 Recipe 训练，命令行覆写用于 A/B：
cls-trainer train --config configs/recipes/game_cls_production.yaml \
  presets.dataloader=throughput
```

### 4. 准备自己的数据

目录结构须满足 `<split>/<game>/<0|1>/<video_id><frame_id>.png`（文件名
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
| 📷 原始帧 | `dataset prepare` → `dataset audit` | 源视频级划分 + SHA-256 泄漏检查 |
| 📦 高吞吐 packed | `dataset pack` | uint8 分片，消除训练热路径 PNG 解码开销 |
| 🛠 逐视频元数据 | `dataset annotate` | `negative_subtype` / `sample_weight` 侧车 |
| 🔎 困难负样本 | `benchmark scan-negatives` → `dataset annotate --from-mining` | 闭环挖掘 |

##### 源身份预检与混合标签视频

`dataset prepare` 会在划分流程中打印 **Source identity analysis** 预检报告，
统计单标签 / 混合标签源视频数量，让数据结构的任何异常尽早暴露（若划分失败，
报告会直接附在错误信息里）：

```text
Source identity analysis
──────────────────────────────────
Source videos:             184
Single-label videos:       162
Mixed-label videos:         22

Mixed-label examples:
  MC::01  labels=[0,1]
  MC::05  labels=[0,1]
```

**混合标签源视频**（同一原始视频既包含 label 0 也包含 label 1 的帧）现在被显式
支持：整个源视频是一个**原子单位**，整体进入 train 或整体进入 val，绝不跨 split
拆开，也不会跨 label 构造 pair。

源身份由可选配置 `data.source_video_identity.mode` 显式建模：

| 模式 | 源身份 `source_video_uid` | 适用场景 |
| --- | --- | --- |
| `game_video`（默认） | `game::video_id` | 一个源视频同时含两个 label 的帧；整个视频是一个原子单位，防泄漏最保守 |
| `game_label_video` | `game::label::video_id` | 0/01 与 1/01 是物理上无关的两个独立视频、只是各自从 01 编号 |

```yaml
data:
  source_video_identity:
    mode: game_video   # 或 game_label_video
```

**`game_label_video` 会打印显著的泄漏警告**——框架假设不同 label 下相同的
`video_id` 是物理无关的源视频；**仅当同一个物理视频不会同时包含两个 label 的帧
时才能使用**，否则同一真实视频会被拆进 train/val，造成严重数据泄漏。默认必须
保持 `game_video`。

本次划分 manifest 的 schema 升级到 **v3**：每行新增按 label 分列的有效 pair 计数、
`labels` 列与 `split_source_identity_mode` 字段。旧的 v2 manifest 会被明确拒绝并
提示 "Delete/rebuild the manifest"（删除后重建即可）。

**源池命名空间（source namespace）** 解决另一个独立问题：`train` 与 `test` 是
分别准备、互不重叠的两组原始视频，只是局部编号恰好相同（例如两边都有 `MC/0/01`），
此时 strict audit 会把它们误判为同一源视频跨 split 泄漏。用
`data.source_video_identity.namespaces` 显式声明测试池来自独立原始视频池后，跨
split 的源身份比较会带上来源池前缀，编号冲突不再触发泄漏：

```yaml
data:
  source_video_identity:
    mode: game_label_video      # 按数据需要
    namespaces:
      source: train_pool        # 简写：source 同时覆盖 train 与 val（from_train 划分）
      test:   heldout_pool      # 测试池必须存在且与 train 侧不同
  # 显式形态：{train: train_pool, val: train_pool, test: heldout_pool}
```

规则：

- `train` 与 `val` 必须共享同一 namespace（它们来自同一个物理池，自动划分）。
- `test` 必须存在且与 train 侧不同，否则配置校验直接报错。
- 不配置 `namespaces` 时行为与旧版完全一致（字节级不变）。
- namespace 只作用于 audit 的跨 split 身份比较；split manifest、dataset 指纹、
  逐 split parquet 的 uid、metadata sidecar key 均不受影响。三 root
  （`split.mode=off`）下 `train`/`val` 也必须共享 namespace；同 namespace 内
  train/val 的真重叠仍被强制判定为泄漏。
- SHA-256 内容层对 namespace **不可见**：即使声明了不同 namespace，任何跨 split
  的相同帧内容仍被强制判定为泄漏。
- `dataset audit --strict` 遇到源身份重叠时，会额外输出**内容冲突分类**诊断：对
  每个冲突源视频报告跨 split 共享内容帧数——`0` 说明大概率只是编号冲突，`>0`
  说明可能是同一原始视频的真泄漏。该诊断只是报告，不放松泄漏闸门。

### 5. 高级选项

#### 模型选择与早停

框架默认使用 **`constrained` 模型选择**：只有满足
`max_global_fpr ≤ 0.01 / max_worst_game_fpr ≤ 0.02 / min_positive_recall ≥ 0.8`
的 checkpoint 才视为合格，合格模型按
`(global_positive_recall, worst_game_positive_recall, -negative_score_p999)`
排序；不合格模型被录入 top-k 注册表但永远排在合格模型之后。
早停 monitor 设为 `selection_score` 时遵循相同合约，`mode: min` 与 selection
monitor 同时出现会在配置校验阶段报错。

#### 分阶段解冻

```yaml
# configs/recipes/game_cls_production.yaml 片段
model:
  trainable_rules:
    - pattern: "^cls\\."          # step 0 起训练线性头
      unfreeze_at_step: 0
    - pattern: "^backbone\\.stage4\\."   # step 500 起解冻 stage4
      unfreeze_at_step: 500
      lr_scale: 0.1
```

规则包含 fingerprint，resume 时若规则变动会立即报错，防止谱系污染。

#### 评估（test split 只评估一次，绝不参与选模）

```bash
cls-trainer evaluate --run <RUN_ID> --checkpoint best_selection --split test
```

没有独立 test 集时 `--split test` 会被拒绝并返回退出码 3。
`--checkpoint` 支持 `last` / `best_selection` / `best_val_loss` / `best_worst_game`
等稳定别名。

#### 困难负样本挖掘与 challenge 验收

```bash
# 用最佳 checkpoint 扫描训练侧负样本池：
cls-trainer benchmark scan-negatives \
  --run <RUN_ID> --checkpoint best_selection

# 将挖掘结果标注为 hard negative 并写入侧车：
cls-trainer dataset annotate \
  --config configs/recipes/game_cls_production.yaml \
  --from-mining indexes/hard_negatives.parquet \
  --subtype wooden_bridge

# 在固定 challenge 集上做不可漂移的验收基准：
cls-trainer benchmark evaluate --run <RUN_ID> --checkpoint best_selection
```

#### 导出与 Release Gate

```bash
# 导出 TorchScript 权重（含 input shape / 阈值 manifest）：
cls-trainer export weights --run <RUN_ID> --checkpoint best_selection \
  --output exports/model.pt

# 导出 ONNX：
cls-trainer export onnx --run <RUN_ID> --checkpoint best_selection \
  --output exports/model.onnx

# Release gate：校验指标是否满足 gate_metrics 阈值，不满足则以非零退出码拦截：
cls-trainer benchmark gate --run <RUN_ID> \
  --config configs/recipes/game_cls_release.yaml
```

导出 manifest（`export_manifest.json`）包含 `decision_threshold`、`input_shape`、
`selection_mode`、`selection_eligible` 等字段，供部署侧消费。

#### 恢复与派生

```bash
cls-trainer train --resume <run 目录>     # 按身份语义精确继续
cls-trainer train --fork   <run 目录>     # 以某 run 的配置为新起点
cls-trainer run list
cls-trainer run show    <RUN_ID>
cls-trainer run compare <RUN_ID_A> <RUN_ID_B>
cls-trainer run export-tensorboard --run <RUN_ID>
```

#### NPU 单卡 / 八卡

```bash
bash scripts/run_npu_1p.sh
bash scripts/run_npu_8p.sh
```

正式长跑前先跑固定验收入口：

```bash
bash scripts/smoke_npu_1p.sh
bash scripts/smoke_npu_8p.sh
```

真实生产配置的 `model.factory` 与 `model.checkpoint_path` **必须**替换为真实模型；
未替换或 checkpoint 不存在时 `doctor` 与训练都会立即失败——这是预期行为，不是 bug。

---

## 了解更多

| 章节 | 描述 |
| --- | --- |
| 🎒 **入门** | |
| [教程 `tutorial.html`](tutorial.html) | 自包含中文教程：目的 → 架构 → 数据 → 训练评估 → NPU 验收 |
| [配置参考](docs/config_reference.md) | 每个配置键的类型、默认值与含义（自动生成） |
| 💻 **开发者** | |
| [CI 状态](https://github.com/sonwe1e/CLSTrainer/actions) | CPU：ruff + mypy + pytest（3.11/3.12/3.13）+ wheel smoke；NPU：自托管 1P/8P 门禁 |
| [NPU 算子检查](tools/check_npu_ops.py) | `doctor --config npu` 时探测 bincount/HCCL/GradScaler 等算子可用性 |
| [数据集审计](tools/audit_dataset.py) | SHA-256 泄漏检查、帧对完整性、划分统计 |

---

## 特性

- 🔎 **不可覆盖的 Run** — 每次训练分配唯一版本化目录（manifest / status / summary），
  `--resume`/`--fork` 记录完整谱系，重复执行永不覆盖。

- ✏️ **严格配置** — 未知键报错并提示候选；`decision.threshold` 是训练、评估与导出
  manifest 共用的唯一业务阈值；`mode: min` 与 selection monitor 同时出现在配置
  校验阶段报错。

- 📈 **低误报优先的评估** — `constrained` 选择模式：先按 FPR/recall 合格性分层，
  再按 `(global_recall, worst_game_recall, -negative_score_p999)` 三元键排序；
  top-k 注册表与早停遵循完全相同的合约。

- 🛡 **数据闭环** — 源视频级自动划分，`data.source_video_identity.mode` 显式建模
  源身份并支持混合标签源视频原子划分、SHA-256 泄漏审计、逐视频 `negative_subtype`
  元数据与困难负样本 mixing、固定 challenge 集验收、packed uint8 高吞吐 loader。

- 🏗 **分阶段解冻** — `trainable_rules` 支持正则匹配 + `unfreeze_at_step` +
  `lr_scale`，rule fingerprint 防止 resume 谱系污染。

- 📦 **导出 & Release Gate** — TorchScript / ONNX 导出 + 部署 manifest，
  `benchmark gate` 自动拦截未达业务指标的候选模型。

---

## Built On

[![PyTorch](https://img.shields.io/badge/-PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
核心张量框架，CUDA / Ascend `torch_npu` 后端。

[![Ascend](https://img.shields.io/badge/-Ascend%20NPU-0062ff)](https://www.hiascend.com/)
CANN / HCCL 运行时与 NPU 验收门禁。

---

## Citation

如果你在研究中使用了 CLSTrainer，请考虑引用：

```bibtex
@software{clstrainer,
  title  = {CLSTrainer: A Production Dual-Frame Binary Classification Training Framework},
  author = {CLSTrainer contributors},
  year   = {2026},
  url    = {https://github.com/sonwe1e/CLSTrainer},
}
```
