# CLSTrainer 可扩展架构重构 USERPLAN

更新时间：2026-07-31  
目标分支建议：`agent/extensible-architecture-foundation`

---

## 1. 计划结论

本轮重构覆盖八项工作：

1. 将双帧二分类语义抽成 `TaskAdapter`；
2. 将固定的 `game + label + video_id + delta` 数据结构抽成通用逻辑 schema；
3. 将“名称包含 cls”升级为 `TrainablePolicy`；
4. 为配置增加严格 schema、版本迁移和未知字段检查；
5. 将单体 `run_training()` 拆成 `ExperimentRunner`；
6. 将二分类决策、指标、分组、错例与报告拆成 `EvaluatorSuite + DecisionPolicy`；
7. 将固定 game/label/delta 采样拆成 `SamplingPolicy`，并预留反馈接口；
8. 将 PNG/packed 与 CPU/CUDA/NPU/DDP 条件分支抽成 `DataBackendFactory + RuntimeStrategy`。

本轮目标不是把 CLSTrainer 做成万能框架，而是把当前可靠的双帧二分类实现封装成默认组件，使第二个任务可以通过新增组件扩展，而不必同时修改 trainer、dataset、sampler、evaluator 和 report。

### 必须保持的默认行为

- 输入仍为两张 `[B,3,208,448]` RGB 图像；
- 默认模型调用仍为 `model(image0, image1)`；
- 默认输出仍为 `[B,2]` logits；
- 默认判定仍为 `softmax(logits)[:,1] > 0.99`；
- 默认 loss 仍为 CE + threshold margin；
- 默认采样仍为 game/label/delta 平衡策略；
- 默认可训练参数仍为名称包含小写 `cls`；
- 默认冻结主干仍要求全部非 trainable state 100% 加载；
- 默认继续支持 PNG 和 packed_uint8；
- NPU 多 worker 仍强制 `spawn`；
- quick/full 报告字段、文件名和选模结果保持兼容；
- 现有 checkpoint 恢复语义不发生破坏性变化。

### 本轮不做

- 不实现真正的多分类、多标签、回归或分割任务；
- 不实现三帧/N 帧读取，仅建立接口；
- 不实现 LoRA/Adapter 算法，仅让策略能够表达；
- 不实现 hard mining，仅提供反馈协议和 No-op 实现；
- 不新增视频解码、WebDataset、LMDB 等 backend；
- 不实现 FSDP、ZeRO、elastic 或多节点；
- 不引入破坏性的 checkpoint v2；
- 不改变 quick/full 的当前业务语义。

---

## 2. 目标架构

```text
ExperimentRunner
│
├── TypedExperimentConfig
├── RuntimeStrategy
│   ├── AcceleratorAdapter
│   └── DistributedAdapter
├── DataModule
│   ├── LogicalDataSchema
│   ├── IndexCodec
│   ├── DataBackend
│   └── SamplingPolicy
├── TaskAdapter
│   ├── BatchContract
│   ├── ForwardContract
│   ├── Loss
│   └── PredictionContract
├── TrainablePolicy
├── EvaluatorSuite
│   ├── DecisionPolicy
│   ├── MetricAccumulator
│   ├── GroupAggregator
│   ├── ErrorExtractor
│   └── ReportWriter
└── Existing Checkpoint / Artifact Services
```

默认组件组合：

```text
DualFrameBinaryTask
GameVideoPairDataModule
LegacyGameBinaryIndexCodec
PngBackend / PackedUint8Backend
BalancedGameLabelDeltaPolicy
NameTokenTrainablePolicy(token="cls")
BinaryThresholdEvaluatorSuite
BinaryThresholdDecision(threshold=0.99)
NpuAccelerator + DdpDistributedAdapter
```

---

## 3. 总体迁移原则

### 3.1 渐进替换

每个新接口先包装旧实现，再让 trainer 切换到接口，最后删除重复旧代码。任何阶段都必须能单独合并和回归。

### 3.2 默认实现复用现有逻辑

例如 `DualFrameBinaryTask.compute_loss()` 必须调用现有 `combined_loss()`，不能复制出第二套 loss。

### 3.3 动态解析只发生在启动阶段

registry、factory、Pydantic schema 和插件解析不得进入每 step 或每 sample 热路径。

### 3.4 兼容优先

V1 配置、现有 Parquet index、现有 checkpoint 和现有报告在迁移期继续可用。

---

## 4. 建议目录结构

```text
src/game_cls/
├── config/
│   ├── loader.py
│   ├── migrations.py
│   ├── schema.py
│   └── plugin_validation.py
├── contracts/
│   ├── task.py
│   ├── data.py
│   ├── evaluation.py
│   ├── runtime.py
│   └── trainable.py
├── registry.py
├── tasks/
│   └── dual_frame_binary.py
├── data/
│   ├── module.py
│   ├── logical_schema.py
│   ├── index_codecs/
│   │   ├── base.py
│   │   └── legacy_game_binary.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── png.py
│   │   └── packed_uint8.py
│   └── sampling/
│       ├── base.py
│       ├── balanced_game_label_delta.py
│       └── feedback.py
├── trainable/
│   ├── base.py
│   ├── name_token.py
│   ├── regex.py
│   └── model_declared.py
├── evaluation/
│   ├── suite.py
│   ├── decision.py
│   ├── groups.py
│   ├── errors.py
│   └── reports.py
├── runtime/
│   ├── strategy.py
│   ├── accelerator.py
│   ├── distributed.py
│   └── factories.py
└── engine/
    ├── runner.py
    ├── state.py
    ├── builders.py
    └── trainer.py
```

目录迁移随职责切换逐步完成，不要求第一批 PR 一次移动所有文件。

---

# 5. TaskAdapter

## 5.1 要完成的事情

从 trainer 抽离：

- CPU batch 校验；
- batch 到设备的转换和归一化；
- 模型 forward；
- 输出 shape 校验；
- loss 计算；
- prediction batch 构造；
- 任务相关的日志组件。

## 5.2 核心接口

```python
@dataclass(frozen=True)
class StepContext:
    global_step: int
    total_steps: int
    epoch: int
    device: Any
    use_amp: bool
    amp_dtype: str


@dataclass
class TaskOutput:
    raw: Any
    extras: dict[str, Any]


@dataclass
class LossOutput:
    total: Any
    components: Mapping[str, Any]


@dataclass
class PredictionBatch:
    scores: Any
    predictions: Any
    targets: Any
    metadata: Any
    extras: dict[str, Any]


class TaskAdapter(Protocol):
    task_name: str
    contract_version: int

    def validate_model(self, model) -> None: ...
    def validate_cpu_batch(self, batch) -> None: ...
    def move_batch_to_device(self, batch, context: StepContext) -> Any: ...
    def forward(self, model, device_batch, context: StepContext) -> TaskOutput: ...
    def compute_loss(self, output, device_batch, context) -> LossOutput: ...
    def build_predictions(self, output, device_batch, context) -> PredictionBatch: ...
```

## 5.3 默认实现

新增 `DualFrameBinaryTask`：

- 调用现有 `ImageSpec.validate_pair_batch_shape()`；
- 负责 uint8 → device → compute dtype → `/255`；
- 调用 `model(images[:,0], images[:,1])`；
- 验证 `[B,2]`；
- 调用现有 `combined_loss()`；
- 输出 CE、threshold_loss 和 threshold_weight；
- 构造 logits、margin、class-1 score、labels 和 metadata；
- 最终 threshold 判定交给 `DecisionPolicy`。

配置：

```yaml
task:
  type: dual_frame_binary
  factory: game_cls.tasks.dual_frame_binary:build_task
  params:
    positive_class_index: 1
    num_classes: 2
```

## 5.4 trainer 迁移

```python
device_batch = task.move_batch_to_device(batch, context)
task_output = task.forward(model, device_batch, context)
loss_output = task.compute_loss(task_output, device_batch, context)
```

trainer 不再直接出现：

```text
images[:, 0]
images[:, 1]
logits.shape[1] == 2
combined_loss(...)
```

## 5.5 测试与验收

新增：

```text
tests/test_task_contract.py
tests/test_dual_frame_binary_task.py
tests/test_task_training_equivalence.py
```

必须验证：

- 正确/错误 shape；
- uint8、FP32、BF16；
- 错误模型输出；
- 新旧 loss 数值一致；
- 同 seed、同 batch、同模型下 10 step 参数更新一致；
- 当前 1P NPU smoke 无回归。

---

# 6. 通用逻辑数据 schema

## 6.1 要完成的事情

将通用数据层从以下业务字段中解耦：

```text
game
binary label
video_id
frame0/frame1
delta
```

但本轮不要求重建现有生产索引。

## 6.2 逻辑 schema

```python
@dataclass(frozen=True)
class TargetValue:
    kind: str
    value: Any


@dataclass(frozen=True)
class SequenceEntry:
    sequence_id: str
    frame_ids: np.ndarray
    frame_references: Any
    target: TargetValue
    groups: Mapping[str, Any]
    temporal_index: Any
    metadata: Mapping[str, Any]
```

## 6.3 IndexCodec

```python
class IndexCodec(Protocol):
    schema_name: str
    schema_version: int

    def read_sequences(self, path, temporal_requirements) -> list[SequenceEntry]: ...
    def write_sequences(self, entries, path) -> None: ...
```

第一轮实现 `LegacyGameBinaryIndexCodec`，读取当前 schema，并映射：

```python
SequenceEntry(
    sequence_id=f"{game}/{label}/{video_id}",
    target=TargetValue(kind="class_index", value=label),
    groups={"game": game, "label": label},
    ...
)
```

## 6.4 兼容策略

保留 `VideoEntry`，增加与 `SequenceEntry` 的双向转换。Dataset 返回 metadata 时新增：

```python
{
    "sequence_id": ...,
    "target": ...,
    "groups": {...},
    "frame_ids": (...),
    "frame_references": (...),
    "temporal_offsets": (...),
    # 兼容字段：
    "game": ...,
    "label": ...,
    "video_id": ...,
    "frame0_id": ...,
    "frame1_id": ...,
    "delta": ...,
}
```

兼容字段至少保留一个版本周期。

## 6.5 GroupKey

```python
@dataclass(frozen=True)
class GroupKey:
    fields: tuple[str, ...]
```

默认：

```yaml
evaluation:
  group_by:
    - [game]
    - [game, label]
    - [sequence_id]
```

## 6.6 测试与验收

```text
tests/test_logical_data_schema.py
tests/test_legacy_index_codec.py
tests/test_sequence_entry_compatibility.py
tests/test_metadata_compatibility.py
```

要求：

- 当前 index 无需重建；
- 当前 metrics_by_game 内容不变；
- SequenceEntry round-trip 不丢数据；
- 热路径无明显性能回归。

---

# 7. TrainablePolicy

## 7.1 要完成的事情

支持：

- 当前 cls-only；
- regex include/exclude；
- 多参数组；
- 学习率倍率；
- 显式 buffer/state 选择；
- 模型自行声明参数组。

## 7.2 核心结构

```python
@dataclass(frozen=True)
class ParameterGroupSpec:
    name: str
    parameter_names: tuple[str, ...]
    learning_rate_multiplier: float = 1.0
    weight_decay: float | None = None


@dataclass(frozen=True)
class StateSelection:
    parameter_keys: tuple[str, ...]
    buffer_keys: tuple[str, ...]


@dataclass(frozen=True)
class TrainableSelection:
    groups: tuple[ParameterGroupSpec, ...]
    frozen_parameter_names: tuple[str, ...]
    trainable_state: StateSelection
    frozen_state: StateSelection
```

接口：

```python
class TrainablePolicy(Protocol):
    policy_name: str

    def select(self, model) -> TrainableSelection: ...
    def configure_module_modes(self, model, selection) -> None: ...
    def validate_loaded_state(self, model, load_report, selection) -> float: ...
```

## 7.3 默认策略

```yaml
trainable:
  policy:
    type: name_token
    params:
      token: cls
      case_sensitive: true
      freeze_trainable_batchnorm_stats: true
      freeze_frozen_batchnorm_stats: true
```

`NameTokenTrainablePolicy` 必须与当前行为完全一致。

## 7.4 新策略

Regex：

```yaml
trainable:
  policy:
    type: regex
    params:
      include:
        - "(^|\\.)cls(\\.|$)"
        - "\\.lora_[AB]$"
      exclude:
        - "running_"
```

Model-declared：

```python
def trainable_parameter_groups(self) -> list[dict]:
    ...
```

## 7.5 optimizer 与 checkpoint 集成

参数组由 policy 生成，仍只使用现有 AdamW：

```text
head: lr × 1.0
adapter: lr × 0.1
bias/norm: weight_decay = 0
```

严格加载规则改为：

```text
所有 policy 判定为 frozen 的 parameter 和 persistent buffer 必须 100% 加载。
```

trainable-only checkpoint 使用显式 `StateSelection`，禁止继续从父模块路径推断全部 state。

## 7.6 测试与验收

```text
tests/test_trainable_policy_name_token.py
tests/test_trainable_policy_regex.py
tests/test_trainable_policy_model_declared.py
tests/test_trainable_policy_checkpoint_selection.py
tests/test_optimizer_parameter_groups.py
```

要求：

- 默认 trainable/frozen 名称完全一致；
- 冻结 state 缺失仍失败；
- regex 无匹配明确报错；
- 参数不得属于多个 group；
- trainable-only checkpoint 不误保存整个父模块；
- 恢复 key 集严格一致。

---

# 8. 严格配置 schema

## 8.1 要完成的事情

- `config_version`；
- unknown key 报错；
- 类型校验；
- 跨字段约束；
- V1 → V2 migration；
- 插件参数独立校验；
- dotted override 拼写建议；
- 废弃字段 warning。

## 8.2 技术选型

建议增加：

```toml
pydantic>=2.10,<3
```

若环境不允许新增依赖，再使用 dataclass + 自定义 schema，但不建议同时维护两套。

## 8.3 加载流程

```text
读取 YAML
→ 递归 base
→ deep merge
→ 识别 config_version
→ 运行 migration
→ 应用 CLI overrides
→ 顶层 schema
→ plugin params schema
→ 跨组件校验
→ normalized config
```

## 8.4 V2 结构

```yaml
config_version: 2

task:
  type: dual_frame_binary
  factory: game_cls.tasks.dual_frame_binary:build_task
  params: {}

trainable:
  policy:
    type: name_token
    factory: game_cls.trainable.name_token:build_policy
    params:
      token: cls

data:
  module_factory: game_cls.data.module:build_game_video_pair_data_module
  index_codec:
    type: legacy_game_binary
  backend:
    type: png
    params: {}

sampler:
  policy:
    type: balanced_game_label_delta
    params:
      game_alpha: 0.25
      class_probability: {0: 0.5, 1: 0.5}

runtime:
  accelerator:
    type: npu
  distributed:
    type: single_process

evaluation:
  suite:
    type: binary_threshold
  decision:
    type: threshold
    params:
      threshold: 0.99
```

为降低风险，原有 `model/loss/train/device/dataloader/checkpoint` 可继续保留，不要求一次大改键名。

## 8.5 V1 → V2 migration

`config_version` 缺失视为 V1：

- 添加默认 dual-frame task；
- `model.trainable_name_contains` → name-token policy；
- `data.backend` → backend config；
- `sampler` 原字段 → default sampling policy params；
- `device + distributed` → runtime；
- `evaluation.threshold` → decision policy。

输出：

```text
[CONFIG MIGRATION] Loaded v1 config and normalized to v2.
```

## 8.6 override 校验

误写：

```bash
dataloader.train.num_worker=4
```

必须报错并建议：

```text
Did you mean: dataloader.train.num_workers
```

## 8.7 跨字段校验

- NPU + workers > 0 → spawn；
- packed → index 必须存在；
- DDP → WORLD_SIZE > 1；
- threshold ∈ `(0,1)`；
- trainable-only → 基础 checkpoint 必须存在；
- plugin params 禁止 unknown key。

## 8.8 测试与验收

```text
tests/test_config_schema.py
tests/test_config_migrations.py
tests/test_config_unknown_keys.py
tests/test_config_plugin_params.py
tests/test_config_cross_validation.py
```

要求：

- 所有正式 YAML 可直接或迁移加载；
- unknown field 必须报错；
- `resolved_config.json` 保存规范化 V2；
- schema 可输出 JSON schema，供 tutorial 参数表生成。

---

# 9. ExperimentRunner 拆分

## 9.1 要完成的事情

将 `run_training()` 从所有功能的拥有者改成组件编排器。

## 9.2 核心结构

```python
@dataclass
class ExperimentState:
    global_step: int
    epoch: int
    step_in_epoch: int
    total_steps: int
    best_metrics: dict
    evaluation_state: dict
    processed_samples: int


@dataclass
class ExperimentComponents:
    config: TypedExperimentConfig
    runtime: RuntimeStrategy
    task: TaskAdapter
    data: DataModule
    trainable_policy: TrainablePolicy
    evaluator: EvaluatorSuite
    model: Any
    optimizer: Any
    scheduler: Any
    scaler: Any
```

```python
class ExperimentRunner:
    def setup(self) -> None: ...
    def restore_if_needed(self) -> None: ...
    def run(self) -> dict: ...
    def run_train_step(self, batch) -> StepResult: ...
    def run_evaluation(self, kind: str) -> Any: ...
    def save_checkpoint(self, tag: str) -> None: ...
    def close(self) -> None: ...
```

## 9.3 Builder

```python
def build_experiment(config) -> ExperimentComponents:
    runtime = build_runtime(config.runtime)
    task = build_task(config.task)
    backend = build_backend(...)
    sampling_policy = build_sampling_policy(...)
    data = build_data_module(...)
    model = build_model(...)
    trainable_policy = build_trainable_policy(...)
    evaluator = build_evaluator_suite(...)
    ...
```

## 9.4 兼容入口

继续保留：

```python
def run_training(config: dict[str, Any]) -> dict:
    typed = validate_and_normalize_config(config)
    runner = ExperimentRunner(build_experiment(typed))
    return runner.run()
```

`tools/train.py` 命令不改变。

## 9.5 Hook

第一轮只提供轻量 hook：

```text
on_run_start
on_batch_ready
on_step_end
on_evaluation_end
on_checkpoint_saved
on_run_end
```

默认 hook：

- console logger；
- JSONL metric logger；
- frozen verifier；
- optional profiler。

## 9.6 热路径

```python
batch = next(loader)
device_batch = task.move_batch_to_device(batch, context)
optimizer.zero_grad(set_to_none=True)

with runtime.autocast(...):
    output = task.forward(model, device_batch, context)
    loss_output = task.compute_loss(output, device_batch, context)

runtime.backward(loss_output.total, scaler)
runtime.clip_gradients(...)
runtime.optimizer_step(...)
scheduler.step()
```

## 9.7 测试与验收

```text
tests/test_experiment_builder.py
tests/test_experiment_runner_smoke.py
tests/test_runner_resume_equivalence.py
tests/test_runner_hook_order.py
tests/test_legacy_run_training_compatibility.py
```

要求：

- `run_training()` 外部签名不变；
- trainer 不再 import 具体双帧 loss/backend/evaluator；
- 固定 seed 下新旧路径 10 step 参数一致；
- quick/full/checkpoint/resume 全部通过。

---

# 10. EvaluatorSuite + DecisionPolicy

## 10.1 要完成的事情

拆分：

```text
模型输出 → 分数 → 决策 → 指标 → 分组 → 错例 → 报告 → 选模
```

## 10.2 DecisionPolicy

```python
@dataclass
class DecisionOutput:
    scores: Any
    predictions: Any
    auxiliary: dict[str, Any]


class DecisionPolicy(Protocol):
    policy_name: str
    def decide(self, prediction_batch: PredictionBatch) -> DecisionOutput: ...
```

默认 `BinaryThresholdDecision`：

```text
margin = logit1 - logit0
probability = sigmoid(margin)
prediction = probability > threshold
```

严格 `>` 行为不能变化。

## 10.3 MetricAccumulator

```python
class MetricAccumulator(Protocol):
    def update(self, prediction_batch, decision) -> None: ...
    def distributed_reduce(self, runtime) -> None: ...
    def compute(self) -> dict[str, Any]: ...
```

默认：

- BinaryConfusionAccumulator；
- BinaryLossAccumulator；
- HistogramAucAccumulator；
- CalibrationAccumulator；
- ConfidenceDistributionAccumulator。

## 10.4 GroupAggregator

内部从通用 metadata `groups` 读取 group key，默认仍输出：

```text
game
game + label
video/sequence
```

## 10.5 ErrorExtractor 与 ReportWriter

```python
class ErrorExtractor(Protocol):
    def extract(self, prediction_batch, decision, checkpoint_step) -> ErrorBatch: ...
```

默认产生当前 FP/FN/near-threshold rows。

ReportWriter 使用显式 ReportSchema，并保留当前：

```text
metrics.json
metrics_by_game.csv
metrics_by_game_label.csv
metrics_by_video.csv
false_positive.parquet
false_negative.parquet
near_threshold.parquet
errors.html
```

预览通过 `PreviewProvider`，不让 report writer 检查具体 decoder 类型。

## 10.6 EvaluatorSuite

```python
class EvaluatorSuite:
    def evaluate(self, model, dataloader, runtime, task, context): ...
```

执行顺序：

```text
Task.forward
→ Task.build_predictions
→ DecisionPolicy.decide
→ MetricAccumulator.update
→ GroupAggregator.update
→ ErrorExtractor.extract
→ ReportWriter.write
```

ModelSelector 移到独立模块，当前 composite 算法不变。

## 10.7 测试与验收

```text
tests/test_binary_decision_policy.py
tests/test_binary_metric_equivalence.py
tests/test_group_aggregator.py
tests/test_error_extractor.py
tests/test_report_schema.py
tests/test_evaluator_suite_equivalence.py
```

Golden 要求：

- metrics 数值一致；
- histogram AUC 容差一致；
- FP/FN schema 不变；
- quick/full 文件集合不变；
- HTML preview 行为不变；
- composite best selection 不变。

---

# 11. SamplingPolicy

## 11.1 要完成的事情

将“如何选样本”与“如何把请求交给 DataLoader”拆开。

## 11.2 SamplingCatalog

```python
class SamplingCatalog(Protocol):
    def available_temporal_keys(self): ...
    def available_groups(self, temporal_key): ...
    def available_targets(self, temporal_key, group): ...
    def sample_request(self, rng, temporal_key, group, target): ...
```

## 11.3 Policy

```python
@dataclass
class SamplingContext:
    epoch: int
    step: int
    global_batch_size: int
    world_size: int
    seed: int


class SamplingPolicy(Protocol):
    policy_name: str
    state_version: int

    def sample_global_batch(self, catalog, context) -> list[SampleRequest]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def update_feedback(self, feedback) -> None: ...
```

## 11.4 默认策略

`BalancedGameLabelDeltaPolicy` 必须复现当前算法：

1. delta 按配置概率；
2. game 权重为可用视频数 `** game_alpha`；
3. label 按 class_probability；
4. video 均匀；
5. valid start 均匀；
6. 全局 batch 内尽可能去重；
7. 先生成 global batch，再按 rank 切片；
8. seed/epoch 规则保持；
9. delta 与 game-label-delta 统计保持。

## 11.5 PolicyBatchSampler

保留：

```text
set_epoch(epoch, start_step)
state_dict(step_in_epoch)
```

保证现有 checkpoint resume。

## 11.6 Feedback

```python
@dataclass(frozen=True)
class SamplingFeedback:
    sample_id: str
    loss: float | None
    error_type: str | None
    score: float | None
    checkpoint_step: int
```

默认 `NoOpFeedbackStore`。本轮不实现 hard mining，但 future policy 可读取训练 loss、评估错例、不确定样本或人工权重。

## 11.7 测试与验收

```text
tests/test_sampling_policy_equivalence.py
tests/test_sampling_policy_determinism.py
tests/test_sampling_policy_rank_sharding.py
tests/test_sampling_policy_resume.py
tests/test_sampling_feedback_contract.py
```

要求：

- 固定 seed 下前 N 个 global batch 与当前一致；
- 1P/8P rank 切片一致；
- resume 后下一 batch 与不中断训练一致；
- 大样本 delta 分布符合配置；
- feedback disabled 时无行为变化。

---

# 12. DataBackendFactory + RuntimeStrategy

## 12.1 DataBackend 接口

```python
@dataclass(frozen=True)
class BackendCapabilities:
    batch_decode: bool
    random_access: bool
    supports_preview: bool
    spawn_safe: bool


class FrameBackend(Protocol):
    backend_name: str
    capabilities: BackendCapabilities

    def get(self, reference): ...
    def get_many(self, references): ...
    def preview(self, reference, output_path): ...
    def close(self) -> None: ...
```

`get_many` 提供默认 fallback。

## 12.2 Factory 与 registry

```python
class DataBackendFactory(Protocol):
    config_model: type
    def create(self, config, image_spec, split: str) -> FrameBackend: ...
```

默认 registry：

```text
png
packed_uint8
```

trainer 不再判断 backend 名称。

## 12.3 Worker-lazy 初始化

Dataset 保存 serializable `BackendSpec`，第一次在 worker 中访问时才构建 backend：

```python
@dataclass(frozen=True)
class BackendSpec:
    factory: str
    params: dict
    split: str
```

要求：

- parent 不打开 worker-owned FD/mmap；
- packed memmap 只在 worker 内打开；
- report preview 在 rank 0 单独构建只读 backend；
- spawn 序列化测试通过。

## 12.4 RuntimeStrategy

采用组合而不是大量组合子类：

```text
RuntimeStrategy
├── AcceleratorAdapter
│   ├── CpuAccelerator
│   ├── CudaAccelerator
│   └── NpuAccelerator
└── DistributedAdapter
    ├── SingleProcess
    └── DdpDistributed
```

Accelerator：

```python
class AcceleratorAdapter(Protocol):
    device: Any
    requires_spawn_workers: bool

    def setup(self, local_rank: int) -> None: ...
    def autocast(self, enabled: bool, dtype: str): ...
    def synchronize(self) -> None: ...
    def make_grad_scaler(self, enabled: bool, dtype: str): ...
```

Distributed：

```python
class DistributedAdapter(Protocol):
    rank: int
    world_size: int
    local_rank: int

    def setup(self, backend: str) -> None: ...
    def wrap_model(self, model, device): ...
    def barrier(self) -> None: ...
    def all_reduce(self, tensor, op="sum"): ...
    def gather_object(self, value, dst=0): ...
    def cleanup(self) -> None: ...
```

## 12.5 NPU 约束

```python
NpuAccelerator.requires_spawn_workers = True
```

配置交叉验证：

```text
requires_spawn_workers + num_workers > 0
→ multiprocessing_context 必须为 spawn
```

不再在 trainer 中写死 `accelerator == "npu"`。

## 12.6 DDP 配置

```yaml
runtime:
  accelerator:
    type: npu
  distributed:
    type: ddp
    params:
      backend: hccl
      find_unused_parameters: false
      broadcast_buffers: false
      gradient_as_bucket_view: true
```

## 12.7 测试与验收

```text
tests/test_backend_registry.py
tests/test_png_backend_contract.py
tests/test_packed_backend_contract.py
tests/test_backend_worker_lazy_init.py
tests/test_runtime_cpu.py
tests/test_runtime_cuda_optional.py
tests/test_runtime_npu_policy.py
tests/test_runtime_ddp_gloo.py
```

要求：

- trainer 不判断 backend 名称；
- trainer 不按 device type 手写 synchronize；
- evaluator 不直接调用 torch.distributed；
- NPU spawn 约束继续有效；
- PNG/packed 输出等价；
- CPU/Gloo、1P NPU、8P HCCL 通过。

---

# 13. 分阶段执行路线

## Phase 0：冻结基线

- 固定当前 commit；
- 建立 golden mini dataset；
- 保存 expected metrics、report schema、checkpoint 与训练 trace；
- 记录 PNG/packed 1P 性能基线。

## Phase 1：ConfigSchema + registry

- config_version=2；
- V1 migration；
- unknown-key validation；
- component registry；
- normalized config。

不改训练语义。

## Phase 2：RuntimeStrategy + DataBackendFactory

- backend adapters；
- worker-lazy 初始化；
- accelerator/distributed adapters；
- trainer/evaluator 使用 runtime facade。

## Phase 3：TrainablePolicy

- NameToken、Regex、ModelDeclared；
- policy-based frozen load；
- policy-based optimizer groups；
- policy-based checkpoint selection。

## Phase 4：TaskAdapter

- DualFrameBinaryTask；
- batch/forward/loss/prediction 迁移；
- 新旧 step 等价。

## Phase 5：LogicalSchema + SamplingPolicy

- SequenceEntry；
- Legacy codec；
- metadata compatibility；
- Balanced policy；
- PolicyBatchSampler；
- NoOp feedback。

## Phase 6：EvaluatorSuite + DecisionPolicy

- binary decision；
- metrics；
- groups；
- errors；
- report schema；
- model selector。

## Phase 7：ExperimentRunner

- state/components/builder/runner/hooks；
- trainer compatibility wrapper；
- 删除重复旧代码。

## Phase 8：文档和生产验收

- README；
- tutorial.html；
- V2 configs；
- 扩展开发文档；
- 完整 CI；
- 1P/8P smoke；
- 性能回归。

---

# 14. 兼容策略

## Python API

至少一个版本周期继续支持：

```python
from game_cls.config import load_config
from game_cls.engine.trainer import run_training
```

## 配置

- V1 自动迁移 V2；
- 迁移期 warning；
- V2 strict unknown-key；
- CI 同时加载 V1/V2 示例。

## 索引

- 当前 frame/video Parquet 继续支持；
- legacy codec 默认；
- 不要求立即重建生产 index。

## 报告

继续输出当前 metrics、CSV、Parquet 和 HTML 文件。

## checkpoint

- 文件名不变；
- 现有 checkpoint 可恢复；
- sampler state 可增加 policy_name/state_version；
- 缺失新字段时按 legacy 读取。

---

# 15. 测试矩阵

## 单元测试

| 组件 | 必测内容 |
|---|---|
| ConfigSchema | unknown field、migration、override、plugin params |
| TaskAdapter | batch、forward、loss、prediction |
| LogicalSchema | legacy 映射、metadata、round-trip |
| TrainablePolicy | selection、group、frozen load、state selection |
| SamplingPolicy | determinism、distribution、rank shard、resume |
| DataBackend | get/get_many、spawn、preview、close |
| Runtime | CPU/CUDA/NPU policy、DDP collectives |
| Evaluator | decision、metrics、groups、errors、reports |
| Runner | setup、step、eval、checkpoint、resume、hook order |

## 集成测试

```text
CPU synthetic 2-step
CPU synthetic quick/full
CPU Gloo 2-rank
PNG real mini dataset
packed real mini dataset
checkpoint interruption/resume
V1 config migration
V2 config direct load
```

## NPU

```text
1P workers=0
1P spawn workers=1
1P spawn workers=2 + augmentation
1P quick/full
1P checkpoint resume
8P HCCL 100-step
8P quick/full/report merge
```

## 性能门槛

- PNG 稳定吞吐下降不超过 5%；
- packed 稳定吞吐下降不超过 5%；
- first batch 不显著变慢；
- worker RSS 不显著增加；
- full evaluator wall time 不显著增加。

---

# 16. 推荐 PR 拆分

```text
PR1  Strict config schema and component registry
PR2  RuntimeStrategy and DataBackendFactory
PR3  TrainablePolicy
PR4  TaskAdapter
PR5  Logical data schema and legacy codec
PR6  SamplingPolicy
PR7  EvaluatorSuite and DecisionPolicy
PR8  ExperimentRunner migration
PR9  Remove deprecated internals and update docs
```

需要真实 NPU 验证：

```text
PR2
PR4
PR6
PR7
PR8
```

---

# 17. 完成定义

```text
[ ] TaskAdapter 控制 batch、forward、loss、prediction
[ ] SequenceEntry/IndexCodec 隔离具体物理 schema
[ ] TrainablePolicy 替代 trainer 中 cls 字符串判断
[ ] 配置具有版本、迁移和 strict unknown-key
[ ] ExperimentRunner 替代单体 run_training
[ ] EvaluatorSuite/DecisionPolicy 隔离二分类语义
[ ] SamplingPolicy 复现当前算法并支持 state/feedback
[ ] DataBackendFactory 替代 PNG/packed 条件分支
[ ] RuntimeStrategy 替代 CPU/CUDA/NPU/DDP 条件分支
[ ] V1 配置、现有 index、checkpoint 和报告继续兼容
[ ] CPU CI、1P NPU、8P NPU 通过
[ ] 默认指标和报告与重构前等价
[ ] 无超过 5% 的无解释性能回归
```

---

# 18. 最终技术判断

正确的重构结果不是新增大量抽象类，而是：

```text
现有双帧二分类仍然严格、清晰和高效；
新任务通过新增组件扩展；
核心 runner 不再因业务变化反复修改。
```

未来实现第二个任务时，主要新增：

```text
一个 TaskAdapter
一个 DataModule/IndexCodec（如需要）
一个 SamplingPolicy（如需要）
一个 EvaluatorSuite
一份严格配置
```

而不再修改现有双帧任务的 trainer、loss、evaluator 和报告代码。
