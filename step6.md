## 核心结论

本轮更新幅度很大，**困难负样本、负样本亚型、分阶段解冻、challenge benchmark、数据性能测试和模型导出等框架能力都已补齐**。最新提交 `5dda2ea` 的 CPU CI 已覆盖 Python 3.11、3.12、3.13、Mypy、完整单测、wheel 和训练 smoke，全部通过。

但当前仍不能认定所有问题已经关闭。我发现 **4 个 P0 级问题和若干契约缺口**，其中最严重的是：**分阶段解冻在 8 卡 DDP 下当前不正确**，以及 **benchmark/mining 在 8 卡和 subtype 场景下没有真正连通**。

| 审查项                 | 当前判断                                         |
| ------------------- | -------------------------------------------- |
| NPU float64、HCCL 归约 | 代码修复保留                                       |
| 自动 train/val 划分     | 主体可用，manifest 契约问题仍在                         |
| 困难负样本采样             | 已实现并进入训练数据流                                  |
| 负样本亚型评估             | 普通验证可用，challenge benchmark 未连通               |
| 分阶段解冻               | 单卡可用，DDP 路径存在严重问题                            |
| 低 FPR 最佳模型选择        | best checkpoint 正确，early stopping/top-k 仍不一致 |
| NPU 实机验收            | 尚未执行                                         |
| 实际楼梯误报改善            | 已具备实验框架，但尚无结果证明                              |

---

## 已经有效完成的改造

### 困难负样本闭环基本形成

当前已经具备：

* 以 `source_video_uid` 为键的 metadata sidecar；

* `negative_subtype`、`sample_weight` 等字段；

* sidecar 在 split 完成后挂载，不影响数据划分身份；

* 普通负样本与困难负样本 bucket 混合采样；

* 每个视频合法 pair 数量限制；

* subtype 分组指标；

* 负样本池扫描、top-K per video、pair 去重和 mining manifest；

* 独立 challenge set 与 benchmark gate。

这意味着此前“地板、木板桥等误报缺少针对性数据机制”的框架问题，已经从设计层进入实现层。

### 分阶段解冻能力已实现

新增的 `model.trainable_rules` 支持：

* 正则匹配参数；
* `unfreeze_at_step`；
* 不同参数组 `lr_scale`；
* optimizer state 迁移；
* rule fingerprint；
* resume 漂移保护。

单设备场景下，这套设计能够解决此前纯 `cls` 线性头适配能力不足的问题。

### 工程结构显著改善

原先超过三千行的 trainer 和大型 CLI 已拆分成多个职责清晰的模块，并增加了导入兼容、行为一致性、Mypy 和 Python 3.13 检查。最新 Actions 中 lint、Mypy 和三版本测试全部成功。

此外，新增了数据吞吐 benchmark、rank-0 checkpoint 读取、权重/ONNX 导出和配置 release gate，整体产品化程度明显提升。

---

## 仍需优先修复的问题

### P0-1：分阶段解冻在 DDP 下不正确

当前执行顺序是：

1. 在裸模型上应用 step 0 的 trainable rules；
2. 将模型包装成 `DistributedDataParallel`；
3. 到 unfreeze step 后，再对已经包装过的 `model` 调用 `apply_trainable_state`；
4. 重建 optimizer，但不重建 DDP。

这里有两个问题。

第一，示例规则使用：

```yaml
pattern: "^cls\\."
pattern: "^backbone\\.stage4\\."
```

但 DDP 包装后的参数名带有 wrapper 层级。当前 boundary 和 resume 路径将 DDP 对象直接传给按名称正则匹配的函数，没有先 `unwrap_model`。

第二，即使改用 `unwrap_model`，DDP 在构造时只基于当时参与训练的参数建立梯度归约结构。当前代码在 DDP 构造后开启新的 `requires_grad` 参数，却没有重建 DDP，因此新解冻参数缺少明确的跨 rank 同步路径。

现有测试只覆盖裸模型上的规则匹配、参数组和学习率，没有覆盖两进程 DDP 动态解冻。

**建议方案：**

* 在 DDP 构造前，让所有未来可能解冻的参数保持 `requires_grad=True`；
* 在冻结阶段通过梯度 hook、optimizer 参数组或 loss 后清梯度阻止更新；
* DDP 从一开始注册所有可能训练的参数；
* unfreeze 时只把参数加入 optimizer，不改变 DDP 参数集合。

备选方案是在每次解冻时所有 rank barrier 后销毁并重新包装 DDP，但复杂度和风险更高。

### P0-2：benchmark/mining 的 subtype 和 packed 数据流没有连通

`_build_pool_loader` 接受 `index` 和 `video_index`，但实际上只使用了 `video_index`。它没有：

* 使用 `pool_index/challenge_index`；
* 根据 `backend` 创建 packed decoder；
* 读取 `pool_metadata/challenge_metadata`；
* 应用 sidecar；
* 开启 `group_by_negative_subtype`。

因此当前行为是：

* PNG video index 可能正常扫描；
* packed mining/challenge 很可能无法解码整数 frame location；
* `subtype_before` 通常为空；
* challenge dataset 不生成 subtype catalog；
* `worst_subtype_fpr_at_decision_threshold` 无法按设计产出；
* `challenge_metadata` 最终只是被打印，并没有进入 loader。

这意味着普通训练和验证中的 subtype 路径已经接通，但**独立 challenge benchmark 的 subtype 验收尚未真正生效**。

### P0-3：benchmark 命令在 8P 配置下存在并发竞争

`scan-negatives` 和 `benchmark evaluate` 都调用 `initialize_runtime(config)`。使用 `npu_8p` 运行时会初始化 HCCL，但随后：

* 每个 rank 都构建完整数据集；
* 每个 rank 都扫描完整负样本池或 challenge set；
* benchmark evaluate 明确调用 `distributed=False`；
* 每个 rank 都写同一个 manifest 或 report 路径。

这可能导致重复计算、文件竞争和报告覆盖。

当前最简单可靠的策略是：**benchmark 和 mining 暂时强制 `WORLD_SIZE=1`**。后续确实需要分布式扫描时，再实现 rank-local dataset、指标归约、结果 gather 和 rank-0 原子写入。

### P0-4：NPU 验收仍未运行，GradScaler probe 仍有错误

NPU workflow 至今没有任何运行记录。

此外，GradScaler probe 仍然是：

```python
x = torch.randn(4, 4, device=device)
y = (x * x).sum()
scaler.scale(y).backward()
```

`x` 没有 `requires_grad=True`，因此该 probe 本身无法正常 backward。

应改为：

```python
x = torch.randn(4, 4, device=device, requires_grad=True)
```

NPU workflow 本身也仍是手动触发，并直接运行依赖真实模型、checkpoint 和 indexes 的生产配置。 在没有 runner 预置数据协议的情况下，它还不是可重复的验收门禁。

---

## 上一轮遗留的契约问题仍然存在

### 自动准备仍有多卡写入竞争

`cmd_train` 在 distributed runtime 初始化之前调用 `_maybe_prepare_split`；该函数没有 rank 判断、文件锁或原子目录提交，并且只检查 `val_index` 是否存在。

八卡首次运行时，仍可能有八个进程同时生成同一批索引。

应直接规定：

```text
WORLD_SIZE > 1 且完整索引缺失
→ 拒绝自动 prepare
→ 提示先执行 cls-trainer dataset prepare
```

同时检查完整产物集合，而不是只检查 `val_frames.parquet`。

### split manifest 契约未修正

当前 splitter 仍然存在：

* `on_new_groups=extend` 对外宣称支持，但实际抛出 `NotImplementedError`；
* fingerprint 相同时直接复用，不验证 seed、算法版本、val ratio、target delta；
* fingerprint 使用绝对路径且不使用内容 SHA；
* summary 固定计算并命名 `val_ratio_achieved_delta2`，即使 target delta 不是 2。

这部分与上一轮审查结论没有实质变化。

### constrained early stopping 和 top-k 仍未统一

`best_selection` 已正确使用 eligibility 和三层 rank key，但 constrained 模式仍把：

```text
selection_score = global positive recall
```

作为单一标量。

Early stopping 仍只读取配置中的一个数值 monitor。

Top-k 仍默认按 `selection_score` 排序，Run index 也只记录这个标量。

因此会出现：

* best checkpoint 选择正确；
* early stopping 判断错误；
* top-k 顺序错误；
* Run compare 无法体现 eligibility、worst-game recall 和 negative p99.9。

应让 early stopping、top-k 和 Run index 都复用 `_selection_eligible + _selection_rank_key`。

---

## 新增功能中的其他明显问题

### benchmark gate 的配置契约不一致

Schema 示例使用：

```yaml
benchmark:
  gate_metrics:
    max_global_fpr: 0.01
    min_positive_recall: 0.8
```

但 `check_gates` 会直接用配置 key 查询 metrics。真正指标名称是：

```text
global_fpr_at_decision_threshold
global_positive_recall_at_decision_threshold
```

所以文档示例键会被判定为 `metric absent`。

比较方向也通过字符串猜测：

```python
if "fpr" in name or "specificity" in name:
    value <= bound
else:
    value >= bound
```

这会错误处理：

* specificity：应越高越好，却被按上限处理；
* ECE：应越低越好，却被按下限处理；
* negative p99.9：通常应越低越好，却被按下限处理。

建议改为显式结构：

```yaml
gate_metrics:
  global_fpr_at_decision_threshold:
    op: "<="
    value: 0.01
  global_positive_recall_at_decision_threshold:
    op: ">="
    value: 0.80
```

当前 `config validate --release` 也只是检查 gate 字典非空，没有验证键名、比较符，也没有验证真实 benchmark report。

### `dataset annotate --from-mining` 的 CLI 无法按设计使用

Parser 把 `--metadata` 设置为必填，同时又提供 `--from-mining` 作为替代输入。结果是使用：

```bash
cls-trainer dataset annotate --from-mining ...
```

时仍必须提供一个无意义的 `--metadata`。

应改为 mutually exclusive required group：

```text
--metadata
或
--from-mining
```

此外，当 `--out` 未提供且 `data.metadata_sidecar` 为 null 时，当前代码先执行 `Path(None)`，无法到达后面的友好错误提示。

从 mining 导入时，同一个视频可能有多个 top-K pair，当前会为同一 UID 写多行 sidecar；但 sidecar 又声明 `source_video_uid` 是 primary key。应先按 UID 聚合去重。

### hard-negative 配置可能静默退化

Schema 只要求 `metadata_sidecar` 字段非空，没有要求文件实际存在，也没有要求 `hard_subtypes` 非空。sidecar 读取函数在文件不存在时返回空映射。

因此可能出现：

```yaml
hard_negative.enabled: true
metadata_sidecar: indexes/not_exists.parquet
```

配置校验通过，但训练悄悄退化为普通负样本采样。

hard-negative enabled 时应强制：

* sidecar 文件存在；
* sidecar 至少含一个 hard subtype video；
* `hard_subtypes` 非空；
* 启动日志输出每个 bucket 的视频数和合法 pair 数。

### 导出配置仍有语义问题

导出代码把输入形状硬编码为 `[1,2,3,208,448]`，没有读取 `data.height/width/channels`；如果 task profile 被覆写，manifest 和 ONNX 都会错误。

Parser 中 `--format` 默认值为 `weights`、`--out` 默认值为 `exports`，因此配置中的 `export.format` 和 `export.output_dir` 永远不会在缺省 CLI 调用时生效。

另外，文档声称 manifest 包含 base checkpoint SHA-256 和 metric summary，但实际 manifest 没有这两个字段；ONNX manifest 的 `run_id` 还是空字符串。

---

## 综合判断与修改顺序

本次版本已经从“通用分类训练器”明显进化为“具备困难负样本闭环和部署工具的生产框架”。**此前最关键的泛化机制缺失已经补上，框架覆盖度可以从约 70% 提升到约 85%。**

但当前仍不建议直接开始 8P 正式训练。优先顺序应是：

1. **修复 DDP staged unfreeze**，并增加两进程参数一致性测试；
2. **让 benchmark/mining 强制单进程，接通 sidecar、subtype 和 packed decoder；**
3. **修复 GradScaler probe，并实际执行一次 NPU 1P/8P workflow；**
4. **禁用多卡 `prepare_if_missing`，完善 split manifest 契约；**
5. **统一 constrained best/early-stop/top-k/Run 排序；**
6. **修复 benchmark gate、annotate 和 export 的配置契约。**

楼梯与木桥、地板的误识别问题现在已经具备正确的解决工具，但生产模板仍没有启用 metadata sidecar、hard-negative sampling 或 trainable rules。 因此最终是否解决，仍需要一轮真实的：

```text
baseline
→ mining
→ subtype 标注
→ hard-negative 训练
→ challenge benchmark
→ 独立 test
```

对比结果来确认，而不能仅依据单元测试判定。
