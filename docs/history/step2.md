## 核心判断

你的判断方向基本正确：**如果模型在约 10,000 step 后，训练侧仍继续改善，而独立验证侧的损失持续上升、业务指标持续下降，那么就是典型过拟合。**

但当前 CLSTrainer 还不能可靠证明这一点。它存在一个比“没有训练曲线”更严重的问题：**项目只有 train/test 两种数据角色，训练过程中反复在 test 上评估并用它选择最佳 checkpoint。**生产配置甚至将其称为 `observed dev-test`，full test 每 5,000 step 执行一次，并用 composite F1 选择最佳模型。严格来说，这个 test 已经承担了 validation 的职责，不再是独立测试集。

因此，接下来的修改不能只加 TensorBoard 曲线。正确顺序应当是：

> **先修正 train/validation/test 协议和指标统计口径，再增加可视化与自动过拟合诊断，最后引入 early stopping 和泛化训练策略。**

---

## 目前的核心缺陷

### 1. 缺少真正独立的 validation 和 test

当前配置只有：

```yaml
data:
  train_index: ...
  test_index: ...
  train_video_index: ...
  test_video_index: ...
```

训练过程中的 quick/full evaluation 都使用 test，最佳 checkpoint 也根据 test 指标选择。这样会产生两个问题：

* 频繁观察 test 并据此修改训练方案，会逐渐对 test 过拟合；
* 最终报告中的 test 分数实际上已经是“被用于调参的验证分数”，不能再代表真实泛化能力。

必须改成严格的三分法：

```text
train
用于梯度更新

validation
用于曲线、模型选择、early stopping、超参数比较

test
训练完成后仅评估一次，不能参与模型选择
```

此外，划分必须以完整视频为单位，而不是以帧或 pair 为单位。当前配置允许跨 split 的同标签重复内容只产生 warning，并且默认没有强制全局视频键唯一；这会削弱验证结果的可信度。

### 2. 当前训练 loss 曲线统计并不准确

现在 `train_metrics.jsonl` 中的 loss、CE 和 threshold loss 不是日志区间内的平均值，而是日志触发时**最后一个 batch**的值：

```python
loss_value = float(loss.detach().item())
ce_value = float(components["cross_entropy"].item())
```

这意味着曲线可能有较大随机噪声，并且不能准确代表最近 50 或 100 step 的训练状态。

应改成区间内按样本加权累计：

```text
sum_loss += batch_loss × batch_size
sum_ce += batch_ce × batch_size
sample_count += batch_size

interval_loss = distributed_sum(sum_loss) /
                distributed_sum(sample_count)
```

DDP/NPU 多卡环境还必须在所有 rank 间进行 `all_reduce`，否则 rank 0 记录的只是自己的局部 batch。

### 3. 验证损失已经计算，但没有形成连续曲线

评估器实际上已经计算：

* cross entropy；
* Brier Score；
* ECE；
* ROC-AUC、PR-AUC；
* global、macro-game、worst-game 指标。

其中 validation cross entropy 已经存在于每个 `reports/*/metrics.json` 中。

问题是这些数据：

* 没有汇总成统一的 evaluation history；
* overview 页面只读取 `train_metrics.jsonl`；
* overview 目前只画训练 loss、吞吐、data wait、学习率和梯度范数；
* TensorBoard 需要训练结束后再从报告目录导出。

所以数据并非完全没有，而是**没有形成可持续观察的训练—验证时间序列**。

### 4. 训练 loss 与 validation loss 不能直接硬画在一起

当前训练总 loss 是：

```text
CE + 动态权重 × threshold margin loss
```

而 evaluator 只报告 CE。threshold loss 的权重还会根据总训练进度动态变化。

另外：

* 训练 loss 在 `model.train()` 下计算；
* 使用数据增强；
* validation 在 `model.eval()` 下计算；
* validation 不使用增强。

所以仅仅把当前 `train/loss` 和 `validation/cross_entropy` 画在同一张图上，统计口径并不一致。

应同时保留两类曲线：

1. **Optimization curve**：真实训练过程中的 total loss、CE、threshold loss；
2. **Generalization curve**：固定 train probe 与 validation 在相同 `eval()`、无增强条件下计算的 CE、F1、Brier。

只有第二类曲线才能准确反映泛化间隙。

### 5. 没有自动控制过拟合的机制

当前只有固定的 `stop_after_steps`，它是验收或阶段运行功能，不是 early stopping。配置中也没有：

* patience；
* minimum delta；
* restore best；
* label smoothing；
* head dropout；
  -多参数组微调；
  -基于 validation plateau 的停止或降学习率策略。

因此即使模型在 10,000 step 达到最佳，框架仍会按照预设步数继续训练。

---

## 明确的训练修改计划

### P0：先修正数据角色和评估协议

这是最高优先级，不能被可视化功能替代。

#### 1. 配置增加 validation

修改 `config_schema.py`、生产配置和 Recipe：

```yaml
data:
  train_index: indexes/train_frames.parquet
  train_video_index: indexes/train_video_entries.parquet

  val_index: indexes/val_frames.parquet
  val_video_index: indexes/val_video_entries.parquet

  test_index: indexes/test_frames.parquet
  test_video_index: indexes/test_video_entries.parquet
```

同时修改 `build_index.py` 和 `audit_dataset.py`，支持 train/val/test 三个 split。

为了兼容旧配置，可以暂时提供迁移逻辑：

```text
旧 test_index
→ 作为 val_index 使用
→ 明确警告：当前没有独立 test set
```

但生产验收必须要求独立 test。

#### 2. 重构 DataLoader 角色

当前 `LoaderBundle` 应从：

```text
train
quick_test
full_test
```

改成：

```text
train
train_probe
val_quick
val_full
test_full
```

各自职责如下：

| DataLoader    | 数据来源             | 增强 | 用途                      |
| ------------- | ---------------- | -: | ----------------------- |
| `train`       | train            | 开启 | 梯度更新                    |
| `train_probe` | 固定 train 子集      | 关闭 | 与 validation 比较泛化差距     |
| `val_quick`   | 固定 validation 子集 | 关闭 | 高频趋势观察                  |
| `val_full`    | 全部 validation    | 关闭 | best 模型和 early stopping |
| `test_full`   | 全部 test          | 关闭 | 训练结束后最终评估               |

`test_full` 不应该在训练循环中周期执行。建议增加独立命令：

```bash
cls-trainer evaluate \
  --run <RUN_ID> \
  --checkpoint best_selection \
  --split test
```

#### 3. 强化 split 泄漏检查

审计阶段必须检查：

* 同一个源视频不能横跨 train/val/test；
* 相同 SHA-256 内容跨 split 必须是 error，而不是 warning；
* 同一视频的相邻帧和不同 delta pair 必须属于同一个 split；
* 每个 game、label 在 validation 和 test 中都要有最低样本数量。

不要直接依赖两位 `video_id`，应在索引阶段生成稳定的：

```text
source_video_uid
```

例如由游戏名、原始视频路径或数据集清单中的唯一 ID 组成。

---

### P1：建立可审查的训练与验证指标体系

#### 1. 修正训练指标统计

在 `trainer.py` 增加区间累计器：

```text
train/total_loss
train/cross_entropy
train/threshold_loss
train/threshold_weight
train/accuracy
train/positive_recall_tau099
train/negative_specificity_tau099
train/grad_norm
train/learning_rate
```

所有 loss 必须按样本数加权，并在分布式环境中归约。

现有指标保留原始值，不要只保存平滑值。EMA 或滑动平均只能用于显示。

#### 2. 增加统一 evaluation history

新增：

```text
<run>/metrics/evaluation.jsonl
```

每次评估写入一行：

```json
{
  "step": 10000,
  "split": "validation",
  "scope": "full",
  "cross_entropy": 0.134,
  "threshold_loss": 0.421,
  "objective_loss": 0.218,
  "brier_score": 0.037,
  "ece_20_bins": 0.061,
  "global_f1_tau099": 0.912,
  "macro_game_f1_tau099": 0.887,
  "worst_game_f1_tau099": 0.743,
  "selection_score": 0.865
}
```

评估器应额外计算：

```text
threshold_loss
positive_margin_pass_rate
negative_margin_pass_rate
positive_margin_p10/p50/p90
negative_margin_p10/p50/p90
```

这样可以判断模型是否因为 threshold loss 而变得过度自信。

#### 3. 增加 train probe

从 train 中构造固定且可复现的无增强子集：

```yaml
evaluation:
  train_probe_pairs_per_video: 32
  train_probe_every_steps: 1000
```

`train_probe` 与 validation 使用完全相同的 evaluator。由此产生：

```text
train_probe/cross_entropy
validation/cross_entropy

train_probe/selection_score
validation/selection_score
```

并计算：

```text
generalization_ce_gap
    = validation_ce - train_probe_ce

generalization_score_gap
    = train_probe_selection - validation_selection
```

这才是框架判断过拟合的核心依据。

#### 4. 可视化必须展示六类趋势

`overview.html` 和 TensorBoard 至少应包括：

1. 训练 total loss、CE、threshold loss；
2. train probe CE 与 validation CE；
3. train probe 与 validation selection score；
4. validation global、macro-game、worst-game F1；
5. validation Brier 和 ECE；
6. 学习率、梯度范数、吞吐和 data wait。

图中还要明确标记：

```text
最佳 validation step
最佳 validation loss step
开始出现泛化退化的 step
early-stop step
```

overview 页面应在每次 full validation 后原子更新，而不是只在训练成功结束时生成。TensorBoard 应支持训练时直接写入；目前的后处理导出功能可以继续保留，用于兼容旧 Run。

---

### P2：加入 early stopping 和多目标 checkpoint

新增配置：

```yaml
early_stopping:
  enabled: true
  monitor: validation.selection_score
  mode: max
  full_validation_only: true
  burn_in_steps: 6000
  patience_evaluations: 3
  min_delta: 0.001
  restore_best: true
```

逻辑是：

```text
burn-in 之前不停止
→ 每次 val_full 判断是否改善
→ 改善幅度低于 min_delta 不算新最佳
→ 连续 patience 次未改善则停止
→ 最终恢复最佳模型
```

不要使用 `val_quick` 决定停止，因为 quick 子集的统计波动更大。

建议保存四种稳定 checkpoint：

```text
model_last.pth
model_best_selection.pth
model_best_val_loss.pth
model_best_worst_game.pth
```

原因是这些 checkpoint 代表不同目标：

* `best_selection`：业务综合指标最好；
* `best_val_loss`：概率建模和泛化损失最好；
* `best_worst_game`：最差游戏表现最好；
* `last`：精确续训。

early stopping 的状态必须写入 checkpoint：

```text
best_value
best_step
bad_evaluation_count
stop_reason
```

恢复训练后不能重新计算 patience。

#### 推荐的初始评估频率

既然你观察到最佳点大约在 10,000 step，当前每 5,000 step 一次 full evaluation 太粗，只能看到 5k、10k、15k 这种稀疏点。

建议第一版使用：

```yaml
evaluation:
  train_probe_every_steps: 1000
  val_quick_every_steps: 500
  val_full_every_steps: 2000

early_stopping:
  burn_in_steps: 6000
  patience_evaluations: 3
  min_delta: 0.001
```

这样如果最佳点为 10k，并在 12k、14k、16k 连续没有改善，训练会在约 16k 停止。这里的 `max_steps` 只是安全上限，不再代表必须训练到该位置。

---

### P3：在可观测性完善后处理泛化问题

不要同时改变十几个训练策略，否则无法知道究竟是什么解决了问题。建议按以下顺序做单变量实验。

#### 第一组：threshold loss 消融

当前 threshold loss 权重最高为 `0.20`，warmup 和 ramp 都通过总训练进度比例决定。

这存在两个值得验证的问题：

1. 延长 `max_steps` 会改变 threshold loss 开始生效的绝对 step；
2. 强制输出跨过 0.99 决策边界，可能提升固定阈值 F1，但也可能导致概率过度自信和 calibration 恶化。

建议增加显式 step 配置：

```yaml
loss:
  threshold_loss_weight: 0.10
  threshold_warmup_steps: 2000
  threshold_ramp_steps: 3000
```

逐步比较：

```text
weight = 0
weight = 0.05
weight = 0.10
weight = 0.20
```

重点观察：

* validation composite；
* worst-game F1；
* validation CE；
* Brier；
* ECE；
* margin pass rate。

如果 PR-AUC 基本不变，但固定 0.99 阈值 F1 和 ECE 变差，那么更可能是**置信度漂移**，而不是表征能力真正退化。

#### 第二组：轻量正则化

在 `combined_loss` 中增加：

```yaml
loss:
  label_smoothing: 0.02
```

默认值保持 `0.0`，保证旧配置行为不变。建议先测试 `0.02` 和 `0.05`，不要一开始使用过大的 smoothing，因为部署阈值固定在 0.99。

模型配置增加：

```yaml
model:
  kwargs:
    cls_dropout: 0.20
```

当前严格 Schema 没有通用 `model.kwargs`，模型 factory 虽然接收 config，但很难添加项目特有的 dropout 或 head 结构参数。

对于只训练 `cls` 的当前模式，优先测试：

```text
head dropout：0.1、0.2、0.3
weight decay：1e-4、5e-4、1e-3
```

#### 第三组：面向实际分布的数据增强

当前只有轻量 affine、color jitter 和 random erasing。

建议补充更接近真实视频链路的增强：

* JPEG/H.264 类压缩退化；
* resize down/up；
  -轻微模糊；
  -传感器或编码噪声；
  -亮度和 gamma 偏移；
  -轻微锐化；
  -实际 false positive 的 hard-negative replay。

所有几何变换默认应对两帧保持一致。是否允许两帧独立的压缩或颜色扰动，应根据真实输入是否可能存在帧间编码差异决定。

不建议第一阶段直接加入 MixUp/CutMix。当前任务是双帧输入且部署使用极高的 0.99 阈值，软标签可能改变 calibration，必须作为独立实验验证。

#### 第四组：调整可训练层范围

如果最终曲线表现为：

```text
train probe 很好
validation 明显差
```

说明 head 过拟合，应减小 head 容量、增加 dropout 或提高正则。

如果表现为：

```text
train probe 和 validation 都较差
```

则不一定是过拟合，更可能是冻结主干导致表征不足。此时才考虑：

```text
cls head：较高学习率
backbone 最后一阶段：较低学习率
其余主干：冻结
```

这需要将当前单一的 `trainable_name_contains: cls` 升级为参数规则和分组学习率，而不是直接全部解冻。

---

## 验收标准与实验顺序

第一轮修改只改变观测能力，不改变模型训练行为。使用相同数据、seed 和超参数重跑一次，验证新旧最佳指标基本一致。这样可以确认监控系统没有改变训练语义。

随后按以下顺序执行：

1. **Baseline-observable**：只增加 train probe、validation history 和曲线；
2. **Early-stop-only**：只增加 early stopping；
3. **Threshold ablation**：分别测试 threshold weight；
4. **Regularization**：测试 label smoothing、dropout 和 weight decay；
5. **Domain augmentation**：加入实际退化与 hard negative；
6. **Final test**：只对选定方案和 checkpoint 执行一次独立 test。

完成后，框架应能自动给出类似结论：

```text
Best validation selection: step 10,000
Best validation CE: step 8,000
Overfitting warning:
  train-probe CE decreased by 12.4%
  validation CE increased by 9.8%
  validation selection declined for 3 full evaluations
Early stopped at step 16,000
Restored checkpoint: model_best_selection.pth
Final test has not been evaluated
```

---

## 综合结论

当前最优先的工作不是立即增加更多增强或修改 loss，而是先让框架能够**正确区分训练、验证和测试，并用同口径指标衡量泛化差距**。

最核心的实施链路是：

> **三分数据集 → train probe → 区间平均训练指标 → 连续 validation history → train/validation 对照曲线 → early stopping → threshold loss 和正则化消融。**

只有完成这条链路后，才能判断约 10,000 step 是真正的过拟合拐点、置信度校准开始恶化，还是当前稀疏评估和固定 0.99 阈值造成的表象。
