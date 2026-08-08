## 必须优先处理的问题

### P0：测试集同时承担验证集和最终测试集职责

训练器周期性运行 `quick_test` 和 `full_test`，并依据这些结果选择、保存最佳 checkpoint；训练结束时仍然使用同一个 test 数据集评估。这样 test 已经参与了模型选择，最终指标不再是无偏测试结果，而是“验证集最优结果”。代码自己将其称为 `observed_dev_test`，本质上也说明它承担的是 dev set 职责。

生产配置还将相同内容跨 train/test 的重复样本默认设为 `warning`，训练不会停止，这会进一步放大泄漏风险。

建议改成：

```text
train
  └─ 参数优化

validation
  ├─ 周期 quick/full 评估
  └─ 最佳 checkpoint 选择

test
  └─ 训练及模型选择全部结束后，只运行一次
```

有两个方面需要修改

1. 我应该不会额外采集验证集，所以验证集应该通过一些划分方式从训练集中取出
2. 目前权重只保留best和last，但可能最好的效果不是这俩，所以中间需要报春topk个权重

---

### P0：`--resume` 会破坏 Run 身份和索引一致性

CLI 在恢复时将 `experiment.output_dir` 指向原目录，并把 `run_mode` 改成 `fixed`。随后训练器重新写入：

- `resolved_config.json`
- `manifest.json`
- `status.json`

但 fixed 模式下新建的 `run_id` 是 `None`，因此原始 manifest 中的 run ID、创建时间和部分谱系信息会被覆盖。恢复成功后又不会向根目录 `index.jsonl` 写入新状态，所以 `run list` 可能继续显示恢复前的失败状态或旧步数。 fileciteturn28file0L2-L2 fileciteturn41file0L2-L2

正确方案应当是：

- `manifest.json` 创建后不可修改；
- 恢复时读取并保留原 `run_id`；
- 新增 `resume_events.jsonl`，记录每次恢复时间、checkpoint、配置差异和命令；
- `status.json` 更新 `resumed_at`，不要覆盖首次 `started`；
- Run index 应按 `run_id` 更新，或改成 append-only event index，再在读取时聚合最新状态；
- 保留 `resolved_config.initial.json`，每次恢复另存配置快照。

---

### P0：Contract 实际上只是可覆盖默认值

文档将宽高、二分类、test delta、阈值和 `cls-only` 描述为业务 Contract，并声称代码会保证这些不变量。但配置合并顺序允许 profile、preset、recipe 和命令行覆盖 contract；代码也没有普遍强制：

- `width/height/channels` 只要求大于零；
- `pair.test_delta` 可以被任意覆盖；
- `model.num_classes` 没有被强制为 2；
- `trainable_name_contains` 可以被覆盖；
- `require_pretrained_backbone` 可以关闭。 fileciteturn19file0L2-L2 fileciteturn23file0L2-L2 fileciteturn51file0L2-L2

因此现在的 Contract 更准确的名称应该是 `defaults`，而不是 contract。

后续修改为
- **可变任务规格**：取消“不可变 Contract”的表述，将它改名为 task profile，并让指标名称、模型输出、数据审计全部随配置变化。

不要保留目前这种“文档说不可变、代码允许变化”的状态。

---

### P0：分布式启动存在文件覆盖和崩溃风险

代码只检查：

```text
distributed.enabled=true 且 WORLD_SIZE<=1
```

却不检查反向情况：

```text
WORLD_SIZE>1 且 distributed.enabled=false
```

如果错误地使用 `torchrun` 启动单卡配置：

- 不会初始化 process group；
- unique 模式下非零 rank 无法获得 rank 0 分配的目录；
- fixed 模式下所有进程可能独立训练，并同时写入相同 checkpoint、status 和报告文件；
- 结果可能是直接崩溃，也可能是更危险的静默文件损坏。 fileciteturn48file0L2-L2 fileciteturn41file0L2-L2

还应补充 accelerator/backend 的合法组合校验：

```text
CPU  → gloo
CUDA → nccl
NPU  → hccl
```

此外，CPU DDP 不应使用 `device_ids=[local_rank]`；当前 DDP 创建逻辑没有区分 CPU。建议引入统一的 `DistributedRuntime`，集中验证环境变量、设备、backend、rank 和 world size，删除 `device.py` 与 `distributed.py` 中重复的设备初始化逻辑。


---

### P0：没有易观察的trainloss, validloss, trainacc, validacc等变化曲线




## 高优先级正确性与可靠性问题

### 1. Schema 只检查类型，不检查完整性和数值语义

当前 Schema 能拒绝未知键、错误基本类型和非法枚举，这是优点。但它没有 required 字段，也基本没有范围和跨字段校验。空字典可以经过 `finalize_config()`，随后 CLI 或训练器访问 `config["train"]`、`config["data"]` 时才出现 `KeyError`。 fileciteturn21file0L2-L2 fileciteturn22file0L2-L2

当前可能被接受的配置包括：

- `local_batch_size <= 0`
- `learning_rate < 0`
- `weight_decay < 0`
- `log_every_steps = 0`
- `decision.threshold >= 1`
- 负的概率或概率和不为 1
- `warmup_steps > total_steps`
- `threshold_temperature <= 0`
- `quick/full cadence < 0`
- composite selection 中存在负权重
- 关闭所有 full evaluation 后仍作为生产配置运行

应在最终配置阶段完成三层验证：

1. **结构完整性**：生产训练所需字段必须存在；
2. **字段约束**：范围、列表长度、字典键和值；
3. **跨字段约束**：warmup、总步数、分布式、backend、评估计划、Contract。

现有“配置消费测试”只搜索 leaf key 字符串是否在任意源码中出现，同名 leaf 很容易造成假阳性，不能证明字段真的以正确路径被消费。 fileciteturn63file0L2-L2

### 2. `doctor` 对 checkpoint 给出错误的成功结论

只要 `checkpoint_path` 是非空字符串，`doctor` 就显示：

```text
[OK] model.checkpoint_path exists
```

并没有调用 `Path(checkpoint_path).is_file()`。因此不存在的 checkpoint 也可能通过最关键的预检项。 fileciteturn30file0L2-L2

同时 doctor 目前只检查部分索引路径，并不完整验证：

- video-level index；
- packed video index；
- audit 文件是否真正通过；
- checkpoint 是否能解析；
- 模型能否完成一次 dummy forward；
- 输出是否严格为 `[B,2]`。

### 3. 恢复配置漂移检查不完整

`check_resume_drift()` 只比较新旧配置键的交集：

```python
for key in set(base_flat) & set(new_flat):
```

因此新增或删除关键字段不会被发现。关键字段列表还遗漏了采样概率、video index、packed index、world size、local batch、优化器和增强配置等。 fileciteturn27file0L2-L2

尤其是 `train.max_steps` 被视为正常变化，但 scheduler 的 Lambda 在恢复时由新的 `total_steps` 重新创建。改变 max steps 会改变后续余弦曲线，不能再称为精确恢复。应区分：

- `resume-exact`：所有影响轨迹的配置完全一致；
- `resume-extend`：允许延长训练，但明确重新规划 scheduler；
- `fork`：改变训练策略，创建新 Run。

### 4. 采样去重语义与配置名称不一致

配置写的是“避免一个全局 batch 中重复视频”，实际 identity 是：

```python
(video_index, delta, start_position)
```

它只避免完全相同的 pair，同一视频的其他起点仍可反复出现。尝试次数达到上限后，代码还会静默放弃去重。 fileciteturn54file0L2-L2

此外，当所有采样权重都小于等于零时， `_choice()` 会回退到均匀采样，而不是拒绝非法配置。这会造成“配置写了 0，实际仍在采样”的隐蔽行为。

应明确提供不同策略：

```yaml
deduplication:
  level: video        # none | pair | video
  on_exhaustion: error # error | warn_and_relax
```

同时记录每个 epoch 的去重失败次数。

### 5. 评估指标名称和阈值配置不一致

阈值允许配置，但以下内容固定为 0.99 语义：

- `global_f1_tau099`
- `macro_game_f1_tau099`
- `worst_game_f1_tau099`
- 0.980、0.990、0.995 的近阈值区间。 fileciteturn44file0L2-L2 fileciteturn46file0L2-L2

只要将业务阈值改成其他值，报告字段就会产生误导。应使用中性名称，例如 `global_f1_at_decision_threshold`，并在 payload 中单独保存阈值。

多卡评估方面，本地已经使用 NumPy 数组进行向量化统计，但随后又转换成大型 Python 字典，通过 `gather_object` 汇总到 rank 0。视频数量很大时，这会成为通信和 rank-0 内存瓶颈。相同 catalog 下应直接对统计数组执行 `all_reduce`。精确 AUC 模式也会把所有 score 以 Python list 聚集到 rank 0，应设置样本规模上限或采用分布式外部排序。

### 6. Checkpoint 的信任边界不清晰

基础模型和训练状态都使用：

```python
torch.load(..., weights_only=False)
```

加载不可信 `.pth` 时存在 pickle 代码执行风险。基础模型 checkpoint 通常只需要 tensor state dict，应优先使用：

- `weights_only=True`；
- 或 `safetensors`。

包含优化器、RNG 和 sampler 状态的内部 resume checkpoint 可以保留完整反序列化，但必须明确标注“只允许加载本项目生成且可信的文件”。 fileciteturn33file0L2-L2 fileciteturn34file0L2-L2

另外，checkpoint loader 默认假设字典中每个值都有 `.shape`；混入 epoch、版本号等 metadata 的裸字典可能直接报错。去除 `module.` 前缀时也应检测键冲突，不能静默覆盖。

## 性能与工程维护问题

生产启动时，每个 rank 都会独立从共享存储加载和反序列化基础 checkpoint。八卡环境下相当于重复执行八次大文件 I/O。更合理的策略是 rank 0 加载后由 DDP 广播参数，或先将 checkpoint 缓存到各节点本地 NVMe。 fileciteturn41file0L2-L2

Packed backend 会把完整帧索引数组随 dataset 复制到每个 spawn worker。百万级帧、多个 worker、八个 rank 时，索引内存会成倍增长；可以将索引改成共享内存、memory-mapped NumPy，或在 worker 内懒加载。打包过程也不是目录级原子事务，失败时可能留下半成品 shard 和 parquet。 fileciteturn60file0L2-L2

代码职责集中程度较高：

- `trainer.py` 同时负责运行管理、数据构建、优化器、评估、checkpoint、日志和训练状态机；
- `cli.py` 同时承担命令解析、恢复策略、doctor、Run 查询和 TensorBoard 导出；
- 同时保留 eager 与 lazy 两套 pair 数据集/采样器；
- 设备初始化在两个模块中重复实现。 fileciteturn7file0L2-L2 fileciteturn9file0L2-L2

建议至少拆分为：

```text
runtime/
  distributed_runtime.py
  run_lifecycle.py

training/
  loop.py
  optimizer.py
  scheduler.py
  evaluation_schedule.py

data/
  train_pipeline.py
  evaluation_pipeline.py

cli/
  train_command.py
  doctor_command.py
  run_command.py
```

依赖只有下界，没有 lock/constraints；CI 只在 Python 3.11 CPU 环境执行 pytest 和合成数据 smoke，没有 lint、类型检查、包构建、Python 版本矩阵或多进程分布式测试。 fileciteturn14file0L2-L2 fileciteturn56file0L2-L2

仓库还提交了 `src/game_cls.egg-info`，虽然 `.gitignore` 已经忽略 `*.egg-info`，说明它可能是在添加 ignore 规则前进入版本控制。应执行 `git rm -r --cached src/game_cls.egg-info`。 fileciteturn6file0L2-L2 fileciteturn57file0L2-L2

## 值得保留的设计

项目并非需要推倒重写。以下设计较好，应在整改时保留：

- 未知配置键立即报错并提供拼写建议；
- video-level 索引和懒生成 pair，避免物化海量 `PairSample`；
- 每个 pair 独立生成 `augmentation_seed`，使多 worker 重启后的随机增强可复现；
- packed backend 支持 batch 解码和 shard LRU；
- checkpoint 保存 rank RNG、sampler 和 evaluation state；
- 对多 worker 增强恢复有专门测试；
- 报告、审计、运行状态和错误样本导出较完整。 fileciteturn55file0L2-L2 fileciteturn61file0L2-L2 fileciteturn62file0L2-L2

## 推荐整改顺序

第一阶段先保证实验结论可信：增加 validation split、禁止跨 split 内容重复、修复 resume manifest/index、落实 Contract 和完整语义校验。

第二阶段保证训练不会静默跑错：替换 `cls` 子串匹配、增加分布式环境一致性校验、修正 doctor、重构 exact resume 与 extend/fork 的边界。

第三阶段再做规模优化：分布式评估改用 tensor/array reduction、基础 checkpoint 单次加载、packed index 共享、Run index 加锁或改用 SQLite。

最后再补工程门禁：依赖锁定、Ruff、Pyright/Mypy、构建测试、Python 3.10–3.13 矩阵、2-process Gloo CI 和可选的 NPU 自托管 smoke。

**当前最不应做的是直接继续扩展训练功能或损失函数。先修复数据集角色、Run 恢复语义、配置 Contract 和分类头选择边界，否则后续实验即使成功运行，其结果仍可能不可比较、不可复现或存在测试集选择偏差。**

---

## 整改进度（本轮完成项）

### 已修复（step3 实施，2026-08-06）

- **P0-1b — Top-K checkpoint**：`checkpoint.save_topk` / `checkpoint.topk_monitor`
  保留按 monitor 排序的 Top-N 全量验证权重（`model_topk_<step>.pth` +
  `topk_registry.json`），registry 随 evaluation_state 持久化，resume 继续维护；
  summary.md / `run show` 展示 topk 列表。
- **P0-2 — Resume Run 身份**：manifest 创建后不可变（保留 run_id/created/谱系）；
  `resolved_config.initial.json` 保存首次快照；`resume_events.jsonl` 记录每次恢复；
  `status.resumed_at` 不覆盖 `started`；index 改为 append-only 事件日志并按
  run_id 聚合；resume 时 index 写入原 runs root（cli 从 run 目录结构推断）。
- **P0-3 — Contract → Task Profile**：`configs/contracts/` 改名 `task_profiles/`，
  recipe 键 `contract:` 报迁移错误并改用 `task_profile:`；schema/文档措辞全部
  改为"默认值可覆写"；指标名中性化：`global_f1_at_decision_threshold` /
  `macro_game_f1_at_decision_threshold` / `worst_game_f1_at_decision_threshold`
  为规范字段，`_tau099` 保留为兼容别名，selection_metric/early_stopping.monitor
  新旧名都接受。
- **P0-4 — DistributedRuntime**：新增 `runtime/distributed_runtime.py` 统一
  校验：反向检查（WORLD_SIZE>1 且 distributed.enabled=false 拒绝）、
  accelerator→backend 合法组合（cpu→gloo / cuda→nccl / npu→hccl）、
  RANK/LOCAL_RANK 范围；CPU DDP 不再传 `device_ids`；`engine/distributed.py`
  改为薄封装；doctor 复用同一校验。
- **高优-1 — Schema 三层语义校验**：`semantic_validate` 覆盖 batch/lr/weight_decay/
  log_every 正负、threshold 开区间、温度与 margin 正值、cadence 非负、
  selection_weights 非负且和>0、概率向量非负且和≈1、warmup≤max_steps；
  生产配置禁止全部关闭 full evaluation。
- **高优-2 — doctor**：`model.checkpoint_path` 真正 `is_file()`；新增
  video-level/packed 索引检查、audit JSON 解析、checkpoint weights_only 解析、
  dummy forward 输出严格 `[B,2]`。
- **高优-3 — resume 漂移**：对称比较（新增/删除键也检测）；关键字段扩充
  （采样概率、video/packed 索引、local batch、优化器、分布式配置等）；
  `resume-exact` / `resume-extend`（仅 max_steps/stop_after，重规划 scheduler）/
  `fork`（其余策略变更拒绝并提示 --fork）三分。
- **高优-4 — 去重语义**：全零采样权重直接报错（不再静默均匀回退）；
  `data.deduplication.level: none|pair|video` + `on_exhaustion: error|warn_and_relax`；
  每 epoch 去重失败次数记录到指标与 evaluation_state。
- **高优-6 — checkpoint 信任边界**：基础模型加载与 evaluate/export/restore-best
  全部 `weights_only=True`；内部 resume checkpoint 增加
  `cls_training_checkpoint` 标记，无标记文件拒绝加载。
- **工程**：`git rm --cached src/game_cls.egg-info`；`docs/config_reference.md`
  重新生成；README 同步。

### 附带发现并处理的环境问题

- Windows 本机存在 torch/OpenMP/MKL + pyarrow 的原生线程竞争：CPU 训练后首次
  parquet 错误分片写入偶发 0xC0000005 崩溃（与基线时 `test_distributed_evaluator`
  同码崩溃同源，时序敏感、与任何单条测试无关）。`conftest.py` 在 Windows 上
  限制 OMP/MKL 线程数，使测试套件确定性通过。
- 修复 CLI `--runs-root` 被误用作运行放置目录的问题（fork/run list 解析才用它）；
  修复 fixed/resume 模式下 index 写入 run 目录导致 `run list` 看不到的问题。

### 本轮未做（按 step3 第三阶段/工程门禁）

- 分布式评估 `gather_object` → tensor all_reduce 重构（evaluator 已部分向量化）；
- 基础 checkpoint 单次加载 + DDP broadcast / 本地缓存；
- packed index 共享内存 / mmap；
- `trainer.py`（~2900 行）模块拆分；CI 门禁（Ruff/Pyright/版本矩阵/2 进程 Gloo）；
- 工程依赖锁定（lock/constraints）。

### 第三阶段（规模优化）追加完成项

- **分布式评估归约**：vectorized 分组统计（per-catalog counts/sum/min/max 数组）
  由 `gather_object` 大字典通信改为 tensor `all_reduce`（min/max 用 MIN/MAX op），
  只有无共享 catalog 的小字典路径保留 gather；exact-AUC 模式增加
  `EXACT_AUC_MAX_SAMPLES`（200 万）样本上限，超限报错建议 histogram 模式。
  新增 2 进程 gloo 测试 `test_distributed_group_reduction.py` 验证跨 rank 合并。
- **工程门禁（lint 部分）**：pyproject 配置 Ruff（E/F/W/I/UP/B/SIM），
  全库修复 111 处 lint 问题（未使用导入、zip 缺 strict、闭包绑定循环变量等）；
  `ruff check` 现为 0 错误。mypy 可运行（既有 76 个动态代码噪音作为已知基线，
  本轮引入的错误已清零）。CI 更新：独立 lint job + Python 3.11/3.12 测试矩阵 +
  `pip check` + wheel 构建冒烟 + 合成训练冒烟。
- **`git rm --cached src/game_cls.egg-info`** 完成（索引已清除）。

### 仍待办（需要多卡/NPU 环境验证，本机无法安全验证）

- 基础 checkpoint 单次加载 + DDP broadcast（依赖 `broadcast_buffers` 语义，
  需真实多卡验证）；packed index 共享内存/mmap；
- trainer.py 模块拆分（`training/`、`cli/` 分层，纯机械移动，留待独立改动）；
- mypy 0 错误收敛（既有噪音）；Python 3.13 CI（torch 支持跟进后）；
- 依赖 lock 文件（uv pip compile，随依赖升级节奏补充）。
