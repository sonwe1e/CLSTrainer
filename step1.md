## 核心结论

截至 `main` 分支最新提交 `944eca1`，CLSTrainer 的**训练内核已经比较工程化，但使用接口仍停留在“开发者工具”阶段**。项目当前真正的问题不是“YAML 参数太多”，而是：

1. 配置没有区分“普通用户必须填写的内容”和“框架内部优化参数”；
2. 配置缺少严格模式、参数来源说明和生效检查；
3. 一次训练没有被建模为独立、不可覆盖、可查询的 Run；
4. 输出虽然丰富，但主要面向程序读取，没有形成面向人的运行摘要。

因此，不建议简单删除配置项，也不建议立刻引入复杂的 Hydra 体系。更合适的方向是：**在现有训练内核之上增加 Recipe、Profile、严格配置 Schema 和 Run Manager 四层产品化接口。**

---

## 当前问题的根因

### 1. 配置系统过于自由，导致“看得见但不一定生效”

当前配置加载器本质上是递归合并普通 `dict`，命令行覆写遇到不存在的路径时会通过 `setdefault()` 创建新字段。也就是说：

```bash
optimzier.learning_rate=0.0001
```

即使把 `optimizer` 拼成了 `optimzier`，程序也不会立即报错，而是生成一个无人读取的新配置项。

项目中已经存在实际的“配置与实现不一致”：

* YAML 中有 `optimizer.name: AdamW`，但训练器固定构造 `torch.optim.AdamW`；
* YAML 中有 `scheduler.name: cosine`，但调度器始终执行固定的 cosine 实现；
* `evaluation.save_all_errors` 出现在 CUDA 配置中，但代码搜索不到相应消费者；
* `experiment.name` 没有参与运行目录命名或运行索引，实际决定产物位置的是固定的 `experiment.output_dir`。

这类问题比“参数太多”更危险，因为用户会认为自己完成了实验控制，实际上参数可能没有生效。

此外，生产配置把数据扫描规则、重复数据策略、增强参数、NPU DataLoader 策略、损失设计、模型选择策略和 checkpoint 优化全部放在一个文件中。它适合作为完整的 resolved config，却不适合作为新用户直接修改的入口。

### 2. 训练入口缺少“理解项目”的中间层

当前训练入口只提供：

```bash
python tools/train.py --config xxx.yaml key=value
```

加载配置后立即调用 `run_training()`，没有以下能力：

* 配置校验但不启动训练；
* 查看最终合并配置；
* 查看某个参数来自哪个配置层；
* 检查数据、模型、checkpoint 和设备环境；
* 解释哪些参数属于高级选项；
* 估算 batch、总步数、评估频率和输出路径。

NPU smoke 脚本需要连续传入大量完整路径覆写，说明框架内部虽然可以精确控制，但用户必须先理解整个配置树才能正确操作。

### 3. 运行记录不是一个独立实体

当前训练开始时会：

* 在固定的 `experiment.output_dir` 写入 `resolved_config.json`；
* 非 resume 情况下清空 `train_metrics.jsonl`；
* 使用固定的 `checkpoints/model_last.pth` 等名称；
* 按 step 创建评估目录；
* 成功结束后写入 `training_summary.json`。

这会带来几个直接问题：

* 同一个配置再次运行，可能覆盖上一次的配置、指标、summary 和 checkpoint；
* stdout 没有自动保存，很多启动检查和错误只能在终端滚动记录中查找；
* 训练失败时通常没有最终 `training_summary.json`；
* 不知道某个目录对应哪个 Git commit、启动命令、主机、设备和依赖环境；
* resume checkpoint 与新输出目录之间没有明确的父子关系；
* 很难批量查询“哪些实验成功、哪个最好、两个实验改了什么”。

checkpoint 本身已经保存了 config、optimizer、scheduler、随机状态等信息，说明底层可复现基础不错；缺少的是 run 级别的组织和呈现。

### 4. 文档内容完整，但缺少按角色组织的使用路径

README 已经包含安装、索引、审计、packed backend、NPU、评估契约和性能分析，也提供了 `tutorial.html`。问题在于，这些内容被组织成了“项目完整知识”，而不是“第一次使用时该做什么”。

新用户首先需要的是：

```text
准备什么
→ 修改哪几个字段
→ 运行哪个命令
→ 去哪里查看结果
→ 如何判断训练是否正常
```

而不是先理解 packed shard、AUC histogram、HCCL 归约和 checkpoint state mode。

---

## 推荐的目标架构

### 1. 将配置拆成四种职责

不要让用户直接维护完整生产配置，而是建立以下分层：

```text
Contract      框架和业务必须遵守的固定契约
Profile       CPU、CUDA、NPU 1P、NPU 8P 等运行环境
Recipe        当前模型、数据集和训练目标
Override      临时实验覆写
```

建议目录：

```text
configs/
├── contracts/
│   └── dual_frame_binary.yaml
├── profiles/
│   ├── cpu_debug.yaml
│   ├── cuda_1p.yaml
│   ├── npu_1p.yaml
│   └── npu_8p.yaml
├── presets/
│   ├── augmentation/
│   │   ├── none.yaml
│   │   ├── light.yaml
│   │   └── standard.yaml
│   ├── dataloader/
│   │   ├── stable.yaml
│   │   └── throughput.yaml
│   └── evaluation/
│       ├── smoke.yaml
│       └── production.yaml
└── recipes/
    ├── example_debug.yaml
    └── game_cls_production.yaml
```

普通用户的 Recipe 应只保留约十几个真正需要决策的字段：

```yaml
profile: npu_8p

run:
  name: game_cls_v1

model:
  factory: my_project.models:build_model
  checkpoint: /models/base_model.pt

data:
  index_dir: /data/game_cls/indexes
  backend: png

train:
  max_steps: 10000
  local_batch_size: 64
  learning_rate: 0.001

presets:
  augmentation: standard
  evaluation: production
```

以下内容不应重复暴露给普通用户：

* NPU 强制 `spawn`；
* 输入必须是 `[B,2,3,208,448]`；
* 非 `cls` 权重必须完整加载；
* test delta 固定为 2；
* checkpoint 原子保存策略；
* histogram AUC 的默认 bin 数；
* worker 线程数等稳定性参数。

这些应属于 Contract 或 Profile。

### 2. 使用严格的类型化 Schema

建议使用 Pydantic v2 或标准 dataclass 建立配置模型，核心要求是：

```python
class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
```

这样可以做到：

* 拼错参数立即报错；
* 枚举值、范围和类型统一验证；
* 支持跨字段约束；
* 自动生成配置参考文档；
* IDE 能够补全配置模型；
* 废弃字段可以给出迁移提示。

例如：

```text
Unknown config key: optimzier.learning_rate
Did you mean: optimizer.learning_rate?
```

还应增加两个 CI 约束：

1. 所有公开配置字段必须有明确的代码消费者；
2. 所有配置字段必须有描述、默认值、适用场景和风险说明。

这可以从机制上防止再次出现 `optimizer.name` 这类“存在但不生效”的参数。

当前项目没有必要立即采用 Hydra。Hydra 的配置组合能力很强，但它会引入 defaults list、工作目录切换、插值规则和插件概念，可能进一步增加首次上手成本。现阶段使用“Pydantic Schema + 简单分层合并 + 参数来源追踪”更合适。

### 3. 合并重复的单一事实来源

目前 `loss.threshold` 和 `evaluation.threshold` 都是 `0.99`，但两者可以被分别修改，从而造成训练目标和评估判定不一致。

应改成：

```yaml
decision:
  threshold: 0.99
```

由训练损失、评估器和部署导出共同读取。

类似原则还应应用于：

* 图像规格；
* 类别数；
* test delta；
* 模型输出契约；
* best checkpoint 选择指标。

一个业务事实只能有一个配置来源。

### 4. 将优化技巧改为命名预设

随机仿射、颜色扰动、随机擦除、worker、prefetch、pin memory 等参数不是新用户的首要决策。

用户优先选择：

```yaml
presets:
  augmentation: standard
  dataloader: stable
```

专家用户仍可以覆写：

```yaml
advanced:
  augmentation:
    random_affine:
      degrees: 3.0
```

这样既不牺牲框架能力，也能显著降低阅读和决策成本。

---

## 训练入口应升级为完整 CLI

建议在 `pyproject.toml` 中注册：

```toml
[project.scripts]
cls-trainer = "game_cls.cli:main"
```

当前 `pyproject.toml` 没有命令行 entry point，用户必须记住 `tools/train.py` 的位置。

推荐提供以下工作流：

```bash
# 创建最小 Recipe
cls-trainer init --profile npu_8p

# 检查环境、模型、数据和 checkpoint
cls-trainer doctor --recipe configs/recipes/game_cls.yaml

# 只解析和验证，不初始化设备
cls-trainer config validate --recipe ...

# 查看最终配置及每个字段的来源
cls-trainer config show --recipe ... --with-source

# 展示预计运行计划
cls-trainer train --recipe ... --dry-run

# 正式运行
cls-trainer train --recipe ...

# 查看和比较实验
cls-trainer run show latest
cls-trainer run compare RUN_A RUN_B
```

`--dry-run` 至少应输出：

```text
设备：NPU × 8
模型工厂：...
基础 checkpoint：... SHA256=...
训练数据：...
全局 batch：512
总步数：10000
Quick test：每 1000 step
Full test：每 5000 step
预计输出目录：...
配置警告：0
```

这会成为用户理解项目最有效的入口。

---

## 将每次训练建模为不可覆盖的 Run

### 推荐目录结构

```text
runs/game_cls/
└── 20260804/
    └── 230712_game-cls-v1_944eca1_a13f/
        ├── manifest.json
        ├── status.json
        ├── command.txt
        ├── console.log
        ├── config/
        │   ├── recipe.yaml
        │   ├── resolved.yaml
        │   └── diff_from_defaults.yaml
        ├── environment.json
        ├── metrics/
        │   ├── train.jsonl
        │   └── events.jsonl
        ├── checkpoints/
        ├── reports/
        ├── summary.json
        ├── summary.md
        └── overview.html
```

运行目录应自动包含：

* run ID；
* Git commit 和 dirty 状态；
* 完整启动命令；
* hostname、设备类型和 world size；
* Python、PyTorch、TorchNPU、CANN 版本；
* 数据索引和审计报告指纹；
* 基础 checkpoint 路径及 SHA-256；
* seed；
* 父 run 和 resume checkpoint；
* 开始、结束时间和最终状态。

默认不允许覆盖已有 Run。只有显式 `--resume RUN_ID` 才能继续原 Run，或者通过 `--fork RUN_ID` 创建带父子关系的新实验。

`status.json` 应在运行过程中原子更新：

```json
{
  "state": "RUNNING",
  "step": 4200,
  "last_update": "2026-08-04T23:40:12+08:00"
}
```

异常退出时记录：

```json
{
  "state": "FAILED",
  "step": 4200,
  "error_type": "DataLoaderTimeout",
  "error_message": "...",
  "traceback_file": "failure.log"
}
```

这样即使训练没有正常结束，也不会只剩一堆无法判断状态的中间文件。

### 面向人的运行摘要

当前 `train_metrics.jsonl` 和评估 Parquet 很适合机器读取，但新用户需要一页式摘要。建议 `overview.html` 和 `summary.md` 固定展示：

* 本次训练是否成功；
* 模型、数据、设备和训练时长；
* 最关键的六至十个超参数；
* loss、吞吐、data wait、学习率和梯度范数曲线；
* best 与 last 的指标；
* global、macro-game、worst-game 指标；
* 最差游戏和主要错误类型；
* 最佳 checkpoint 的明确路径；
* 与基线 Run 的参数及指标差异；
* 所有 warning。

`training_summary.json` 仍可保留作为机器接口，但不应是用户查看结果的首要入口。

---

## 推荐实施优先级

### P0：先解决会误导实验的问题

优先完成以下改造：

1. 未知配置键必须报错，命令行覆写不得静默创建字段；
2. 修复或删除 `optimizer.name`、`scheduler.name`、`save_all_errors` 等不一致字段；
3. 合并重复的 threshold 等业务契约；
4. 自动创建唯一运行目录，禁止无意覆盖；
5. 自动保存 `console.log`、`manifest.json` 和 `status.json`；
6. 训练结束不再直接 `print()` 整个嵌套结果，而是输出简洁 summary 和关键文件路径。

### P1：降低首次使用门槛

随后建立：

* 类型化配置 Schema；
* Contract、Profile、Recipe 分层；
* `init`、`doctor`、`config show`、`dry-run` 命令；
* augmentation、dataloader、evaluation 命名预设；
* 自动生成的配置参考文档。

### P2：提高长期实验管理能力

最后增加：

* Run 索引，可先使用 `runs/index.jsonl` 或 SQLite；
* `run list/show/compare`；
* 静态 HTML 曲线和实验对比；
* 可选 TensorBoard 导出；
* resume 配置差异检查和实验 lineage；
* CI 中的配置消费覆盖检查。

---

## 最终判断

CLSTrainer 不需要推翻当前训练实现。其数据审计、严格 checkpoint 加载、分布式评估、NPU DataLoader 约束和运行安全性已经构成了较好的内核。当前最值得投入的是**把“内部能力”包装成清晰的使用路径**：

> 普通用户只维护 Recipe，机器环境由 Profile 决定，稳定性要求由 Contract 保证，所有底层参数经过 Schema 校验，每次训练自动形成独立 Run。

完成这一层后，新用户不需要先理解上百个配置字段，也能知道该修改什么、实际生效了什么、训练是否正常，以及最终应该使用哪个模型。
