## 核心结论

step4 已经把 NPU 正确性、自动源视频划分、低 FPR 受约束选择、负样本尾部 OHEM/pairwise ranking 等核心能力落地。当前框架真正欠缺的不是更多常规功能，而是三块被 step4 明确延后的工作，它们正好构成 step5 的三条主线：

1. **建模轮（WS1）**：把"困难负样本"从一次性实验变成闭环——逐视频 `negative_subtype` 元数据 sidecar、hard-negative mining 工作流、challenge 验证集，以及用 `trainable_rules` 替代单一 `trainable_name_contains` 的分阶段解冻。
2. **效率与工程硬化（WS2）**：`trainer.py`（约 3145 行）和 `cli.py`（约 1963 行）的拆包重构、`cls-trainer benchmark data`、纯 head-only 的离线 feature cache、packed 索引共享内存、基座 checkpoint 单次加载 + DDP 广播，以及 mypy/lockfile/Python 3.13 工具链收敛。
3. **部署与基准（WS3）**：权重导出（ONNX 可选）、固定 challenge 集基准报告、`minimum_worst_game_f1` 生产基线门与 wheel/release 就绪。

两条贯穿原则：

- **默认值全中立**：所有新增配置默认 `null`/`false`/`0`，现有配置文件逐字节解析等价；`trainable_name_contains` 保留为无 `trainable_rules` 时的回退路径。
- **不可变 Run 与恢复语义不被破坏**：新命令只追加 `metrics/evaluation.jsonl`（标记 `kind=benchmark`/`kind=export`）、写入自己的 `benchmarks/`/`exports/` 目录，绝不修改 `manifest.json`；任何改变训练轨迹的配置变更都进入 resume 漂移判定。

建议按 **P1 工程地基 → P2 元数据与采样 → P3 挖掘闭环 → P4 分阶段解冻 → P5 效率硬化 → P6 部署基准** 的顺序实施，每阶段独立可测、可合并。P1 是纯机械重构，最先落地以降低后续 trainer/cli 密集改动的风险；P2→P3 前置数据协议工作，因为 WS1 与 WS3 都依赖 subtype 元数据；P4 是正确性风险最高的 trainer 手术，与数据工作解耦。

---

## 一、P1 工程地基：trainer/cli 拆包重构与工具链收敛

### 1. `trainer.py` 拆分

`src/game_cls/engine/trainer.py` 已膨胀到约 3145 行，混合了 Run 读写、dataloader 构建、优化器、选择、评估、早停与训练循环。拆成 `src/game_cls/engine/training/` 包，**按现有函数边界纯搬移，不修改逻辑**：

```text
engine/training/
├── run_io.py        # _write_run_manifest/_write_resolved_config/_record_run_failure/
│                    # _finalize_run_success/_read_status/_iso_now/_append_jsonl/
│                    # _append_training_metrics/_append_evaluation_history/
│                    # _evaluation_history_record/_maybe_save_topk
├── config.py        # validate_training_config/_validate_dataloader_config/
│                    # _dataloader_option/has_independent_test
├── optimizer.py     # build_optimizer_parameter_groups/_set_train_mode
├── loaders.py       # _loader_common/_build_real_data_components/
│                    # build_eval_loader_for_split/_make_dataloaders/LoaderBundle
├── state.py         # _distributed_sum_int/_build_scheduler/_broadcast_object/
│                    # _gather_random_states/_normalized_position/_save_all_ranks
├── selection.py     # _selection_mode/_selection_score/_selection_eligible/
│                    # _selection_rank_key/_annotate_selection/_is_better_model/
│                    # _save_best_enabled
├── evaluation.py    # _run_evaluation/_threshold_weight_for_eval/
│                    # _new_interval_accumulator/_reduce_interval_accumulator/_EVALUATION_ROLES
├── early_stopping.py# _early_stopping_defaults/_update_early_stopping
├── loop_util.py     # _tb_write_scalars/_seed_everything/
│                    # _synchronize_device_for_metrics/_initialize_data_worker
├── synthetic.py     # SyntheticPairDataset
└── loop.py          # run_training（import 上述所有模块，仅剩训练主循环）
```

`engine/trainer.py` 退化为 **re-export shim**：`from game_cls.engine.training.loop import run_training`，并 re-export 所有被 `cli.py` 与既有测试 import 的名字（`_annotate_selection`、`_append_evaluation_history`、`build_eval_loader_for_split`、`has_independent_test`、`LoaderBundle` 等）。shim 保留到 P6 切换内部 import，保证 `cmd_evaluate` 与现有测试零改动。

### 2. `cli.py` 拆分

`src/game_cls/cli.py`（约 1963 行）拆成 `src/game_cls/cli/` 包：

```text
cli/
├── parser.py        # build_parser
├── common.py        # argparse 助手/_StreamTee/_TeeContext/DEFAULT_RUNS_ROOT/
│                    # run 目录解析/check_resume_drift/classify_resume
├── train.py         # cmd_train
├── evaluate.py      # cmd_evaluate
├── config_tools.py  # config show/validate/reference
├── run_tools.py     # run list/show/compare/export-tensorboard
├── dataset.py       # dataset prepare/audit/pack
├── doctor.py        # cmd_doctor
└── init_cmd.py      # cmd_init
```

`cli.py` 保留为薄入口（`main` + `build_parser` + `train_command_main`），console script `cls-trainer = game_cls.cli:main` 不变。

### 3. 保真策略（先测后移）

- 新增 `tests/test_import_parity.py`：断言 `game_cls.engine.trainer.run_training is training.loop.run_training`，且每个被移动符号解析一致。
- 新增 `tests/test_trainer_split_parity.py`（golden 对比）：跑现有合成 smoke，快照产出的 `train_metrics.jsonl` metric 名集合、`metrics/evaluation.jsonl` 键、checkpoint dict 键，断言搬移后逐字节不变。
- **每移一个模块跑一次全量测试**；搬移期间严禁顺手改逻辑，任何疑似 bug 都按"搬移 bug"处理（diff 定位）。
- 必须保持全绿：`test_training_smoke`、`test_exact_training_resume`、`test_checkpoint_resume`、`test_distributed_evaluator`。

### 4. 工具链收敛

- **mypy 0 错误**：`[tool.mypy]` 保持 `files=["src/game_cls"]`，对 ~76 处已知动态代码噪音用精确 `# type: ignore[code]` 处理。
- **lockfile**：`uv pip compile pyproject.toml -o requirements.lock`、`uv pip compile --extra dev -o requirements-dev.lock`，提交进仓库。
- **CI**：`.github/workflows/ci.yml` pytest 矩阵加 `"3.13"`（`requires-python >=3.10,<3.14` 允许，torch 2.6+ 提供 3.13 wheel）；lint job 加 mypy 步骤。
- **清理**：`git rm` 废弃空目录 `src/game_cls/config/`、`contracts/`、`evaluation/`、`tasks/`、`trainable/`、`data/backends/`、`data/index_codecs/`、`data/sampling/`（只含陈旧 `__pycache__`）。

**验收标准**：全量测试绿；golden 对比一致；`ruff check .` 与 `mypy` 零错误；`python -m build` 与 wheel smoke 通过；无任何 config 键或数据格式变化。

---

## 二、P2 逐视频元数据 sidecar 与困难负样本采样/指标

### 1. 硬约束的重新审视

step4 明确遵守了"无 per-video 元数据 sidecar（分组/指标保持 game 粒度）"这一硬约束。step5 需要重新引入逐视频 `negative_subtype`，但必须**不破坏** game 粒度不变量。设计是：

- **sidecar 是可选、附加的**，按 `source_video_uid` 键控，绝不替代 game 粒度索引；
- **split 不受影响**：`splitter.py` 仍只按 `source_video_uid` + `game/label/legal_pair_count` 划分，sidecar 在 split **之后** join，因此不会跨 split 泄漏、不会漂移 manifest、无需 bump `SPLIT_ALGORITHM_VERSION`；
- **dedup 身份不变**：subtype 不作为 dedup 键，视频与 `(video,delta,start)` 身份不变；
- **指标是附加 catalog**：保留 game/game_label/video catalog，subtype 作为新增 catalog 走同一套向量化归约 `_all_reduce_group_arrays`。

### 2. sidecar 数据协议

新增可选 parquet `video_metadata.parquet`（路径由 `data.metadata_sidecar` 配置，默认 `None`）：

```text
source_video_uid   string, 主键,必须等于索引里的 game::video_id
negative_subtype   string|null
scene_type         string|null
capture_domain     string|null
difficulty         string|null
sample_weight      float, 默认 1.0
metadata_version   int
metadata_fingerprint  string（对行做 SHA-256）
```

新模块 `src/game_cls/data/sidecar.py`：`load_metadata_sidecar(path)`、`validate_sidecar_against_index(sidecar, entries)`（sidecar 中每个 uid 必须存在于索引，未知 uid 报错，缺行记 NULL）、`write_metadata_sidecar(rows, path)`（原子重写）。`data/video_index.py` 的 `VideoEntry` 增可空字段 `negative_subtype: str | None = None`、`sample_weight: float = 1.0`，读写 parquet 时兼容缺失列。

### 3. 采样：困难负样本混合

`VideoBalancedPairBatchSampler` 增加可选 `hard_negative_cfg`：

- **禁用时**（默认）：`_sample_for_delta` 逐字节不变，保证现有确定性训练无任何变化；
- **启用时**：选择负样本 label 后，按 `data.hard_negative.negative_mix`（`ordinary`/`hard` 权重）选择 subtype 桶，再在桶内选视频；桶内无可选视频时回退另一桶并记一次日志，训练不会停滞；
- `max_pairs_per_video` 可选上限：初始化时对每个视频的 `valid_start_positions` 只保留前 N 个起始位（确定性），避免模型记忆少量场景。

### 4. 指标

- `EvalPairDataset.group_catalogs` 增 `game_label_subtype` catalog（仅当有非空 subtype 且 `evaluation.group_by_negative_subtype=true`），batch 增 `game_label_subtype_id`；现有 catalog/键不变。
- `evaluate()` 向量化路径增 `subtype_counts_array`，`_all_reduce_group_arrays` 归约新数组（同样在加速器设备上做归约）。
- 新指标：`worst_subtype_fpr_at_decision_threshold`、`worst_subtype_recall_at_decision_threshold`、`subtype_negative_counts`（分组分母，避免小分组被误读为"零误报"）。
- 新报告 `metrics_by_game_label_subtype.csv`（仅启用分组时写）。
- `_selection_eligible` 增门 `evaluation.max_worst_subtype_fpr`（`null` 表示禁用），纳入受约束选择第一层。

### 5. 新增配置键（默认全中立）

```yaml
data:
  metadata_sidecar: null
  hard_negative:
    enabled: false
    subtype_field: negative_subtype
    hard_subtypes: []
    ordinary_subtypes: []        # 空 = 不属于 hard_subtypes 的全部
    negative_mix: { ordinary: 0.5, hard: 0.5 }
    max_pairs_per_video: null
    min_videos_per_subtype_bucket: 1
evaluation:
  group_by_negative_subtype: false
  max_worst_subtype_fpr: null
```

每个键同步：schema 注册 → 默认值 → `semantic_validate` 范围检查 → **代码消费方**（`tests/test_config_consumption.py` 会扫字符串断言）→ `docs/config_reference.md` 重新生成。

### 6. CLI

`cls-trainer dataset annotate --config <cfg> --metadata <video_metadata.csv/parquet> [--out indexes/video_metadata.parquet]`：把用户 CSV（`source_video_uid, negative_subtype, …`）导入规范 sidecar，对 train/val/test 索引逐一校验 uid。

**测试**：`test_metadata_sidecar.py`（加载/校验/应用、未知 uid 报错、缺失= NULL、原子重写、fingerprint 变化检测）、`test_hard_negative_sampler.py`（禁用时确定性、启用时混合比例、上限、空桶回退）、`test_negative_subtype_metrics.py`（向量化 + 非向量化 subtype 计数、`worst_subtype_fpr`、分组分母、扩展双进程 gloo 归约测试）。

**验收标准**：无 sidecar 时训练/评估与现状逐字节等价；有 sidecar 时 split/dedup 不变、subtype 指标正确、跨 rank 归约一致。

---

## 三、P3 困难负样本挖掘闭环与 challenge 集

### 1. mining 清单协议

新文件 `indexes/hard_negatives.parquet`（默认 `data.mining.output`）：

```text
source_video_uid, game, video_id, frame0_id, frame1_id,
delta, p_positive, subtype_before, rank_in_video, mining_version
```

挖掘流程（复用 `evaluate()`，不写报告目录）：

1. 加载当前最佳 checkpoint；
2. 对 `data.mining.pool_index`（训练侧负样本池索引，默认配置）做全量评估；
3. 负样本按 `p_positive` 从高到低排序；
4. 每个 `source_video_uid` 只保留 top-K（`top_k_per_video`），避免连续帧淹没；
5. 按 pair 身份去重；记录 `subtype_before`（挖掘前的 subtype）与 `mining_version`；
6. 可选 `score_threshold`（只保留 `p_positive >= 阈值`）与 `max_samples`。

### 2. challenge 集协议

challenge 集是**独立索引三元组**（`data.challenge_index` / `challenge_video_index` / `challenge_metadata`），固定不变：

- 它**不属于** train/val/test 协议，**永不参与模型选择**；
- 只被 `cls-trainer benchmark evaluate` 消费；
- 这同时强制"验证/测试集误报不混入训练"：mining 默认用训练侧 `pool_index`，challenge 仅供基准报告。

### 3. 新增配置键

```yaml
data:
  mining:
    enabled: false
    pool_index: null
    pool_video_index: null
    pool_metadata: null
    output: indexes/hard_negatives.parquet
    top_k_per_video: 8
    max_samples: null
    score_threshold: null
    version: 1
  challenge_index: null
  challenge_video_index: null
  challenge_metadata: null
benchmark:
  output_dir: benchmarks
  gate_metrics: {}     # 例如 {max_global_fpr: 0.01, min_positive_recall: 0.8, max_worst_subtype_fpr: 0.02}
```

### 4. CLI

- `cls-trainer benchmark scan-negatives --config <cfg> --checkpoint <run>:<alias|path>`：产出版本化 mining 清单，不改动任何 split。
- `cls-trainer dataset annotate --from-mining <hard_negatives.parquet> --subtype wooden_bridge [--out ...]`：把挖掘出的 subtype 标签应用进 sidecar（仍按 `source_video_uid`，附加式）。
- `cls-trainer benchmark evaluate`（命令在此落地，完整报告在 P6）。

**行为要点**：`data.hard_negative.enabled=true` + 带挖掘 subtype 的 sidecar 才是真正改变下一轮采样的开关；mining 命令本身只产产物。

**测试**：`test_mining_manifest.py`（top-K/视频、去重、版本化、确定性排序、旧版本拒绝）、`test_benchmark_scan.py`（合成索引挖掘产出合法清单且不改变任何 split）、扩展 `test_metadata_sidecar.py` 覆盖 `--from-mining`。

---

## 四、P4 分阶段解冻 `trainable_rules`

### 1. 规则与参数组

新模块 `src/game_cls/model/trainable_rules.py`（或扩展现有 `freeze_policy.py`）：

```yaml
model:
  trainable_rules:
    cls_head:
      pattern: ^cls\.
      lr_scale: 1.0
      unfreeze_at_step: 0
      priority: null        # 越大越先匹配
    backbone_stage4:
      pattern: ^backbone\.stage4\.
      lr_scale: 0.10
      unfreeze_at_step: 1000
```

- `resolve_trainable_parameters(model, rules, step)`：参数在 `step >= unfreeze_at_step` 且匹配规则（按 priority 降序、规则插入序首匹配）时方可训练；
- `apply_trainable_state(model, rules, step)`：设置 `requires_grad`；
- `build_optimizer_parameter_groups(model, rules, *, weight_decay, base_lr, step)`：每条规则一组（含 decay/no-decay 子组，沿用现有约定），组内 `lr = base_lr * lr_scale`；
- `freeze_policy.py` / `checkpoint_loader.py` 增加规则感知变体：冻结集覆盖率校验按"规则冻结集合"计算，而非单一 `name_contains` token。

### 2. 优化器与调度

`_build_scheduler` 已按 `base_lr = max(group["lr"])` 推导，因此每组 `lr_scale` 天然兼容；保持 `total_steps` 语义，cosine 不变。

### 3. 恢复与 checkpoint 身份（正确性风险最高的部分）

`requires_grad` 在训练中随 step 变化，恢复时若按保存时刻的掩码恢复会错位。设计：

- checkpoint 增存：`trainable_state`（有序可训练参数名快照）、每个 optimizer 组的 `param_names`、`trainable_rules_fingerprint`；
- 新增 `restore_optimizer_for_rules(optimizer, saved_state, current_trainable_names)`：按**当前 step** 的规则重建 optimizer 组，再按参数名拷贝已保存状态；**新解冻的参数用全新 AdamW 状态**（文档注明：规则不变时才承诺 exact-resume）；
- 恢复顺序：恢复保存的掩码 → `load_state_dict` → 按当前 step 重新应用掩码 → 重建 optimizer 组 → 按名恢复；
- `classify_resume`/`check_resume_drift`：`trainable_rules` 变化视为轨迹性变更，除非 fingerprint 全同否则 `resume-exact` 降级为 `--fork`，规则差异记入 `resume_events.jsonl`。

### 4. 新增配置键

`model.trainable_rules.<name>{pattern, lr_scale, unfreeze_at_step, priority}`（dict 键，兼容 `_NESTED_KEY_SCHEMAS` 与 `check_override_path`）。`trainable_name_contains` 保留：无 `trainable_rules` 时走旧路径，逐字节等价。

### 5. dry-run

`train --dry-run` 打印分组解冻计划：每条规则的 step 解冻点、参数数量、`lr_scale`。

**测试**：`test_trainable_rules.py`（pattern 匹配、priority、`unfreeze_at_step`、`lr_scale` 组、decay/no-decay 拆分、冻结断言）、`test_trainable_rules_resume.py`（规则不变 = exact；中途解冻恢复 = 新解冻参数全新状态；checkpoint 键集不匹配检测）、扩展 `test_freeze_policy/test_checkpoint_resume/test_exact_training_resume/test_production_safety`；CI 合成 smoke 增加 `trainable_rules` 变体。

**验收标准**：无 `trainable_rules` 时与现状逐字节等价；解冻过程中恢复的优化器状态与从头训练在相同轨迹下一致；任何规则变更不会静默沿用旧轨迹。

---

## 五、P5 效率硬化

### 1. `cls-trainer benchmark data`

纯 CLI 吞吐探针（无 schema 键），对 `[png, packed_uint8] × [num_workers 0,4] × [prefetch 1,2,4]` 组合，复用 `_loader_common`/`_build_real_data_components` 构建 loader，跑 N 步 `next(loader)`，测量 `samples/s`、`data_wait_ratio`、进程 RSS，输出 `benchmarks/data_<backend>_w<w>_p<p>.json`。worker/prefetch/packed-vs-png 结论由实测决定，不硬编码。

### 2. 离线 feature cache

- `FeatureCache`：按 `source_video_uid` + `(delta, start_position)` 把 frozen-backbone 特征写/读到 `feature_cache.path`；
- **启用前提**：仅 head 可训练（用 P4 规则感知检测：有效可训练集只含 head 规则）+ 无增强，否则清晰报错；
- 模型接口契约：factory 需暴露 `feature_dim` 与 `forward_features(image0, image1)`；`build_demo_model` 为测试扩展满足该契约；
- `LazyTrainingPairDataset` 外包一层 `FeatureCachedTrainingPairDataset`，训练循环前向变为 `head(features)`；
- CLI `cls-trainer cache features --config <cfg> [--split train|val]` 预计算缓存；
- 新增配置键：`feature_cache{enabled, path, head_only_required, no_augmentation_required}`（默认全中立）。

### 3. packed 索引共享内存

新键 `data.packed_index_shared_memory`（默认 `false`）。启用时帧索引数组（`shard_ids/offsets/lengths`）以文件背衬 `np.memmap` 载入（Linux `/dev/shm` 可用时；非 Linux 优雅回退内存数组并告警），让 DataLoader worker 共享页而非重复解析 parquet。收益用 `benchmark data` 实测。

### 4. 基座 checkpoint 单次加载 + DDP 广播

`distributed.enabled` 且 `world_size>1` 时：rank 0 加载 `model.checkpoint_path`，经 `dist.broadcast_object_list`（复用现有 `_broadcast_object`）广播 state dict，其余 rank 跳过文件 IO；保持 `weights_only=True`。

**测试**：`test_benchmark_data.py`（合成数据探针产出合法 JSON 报告）、`test_feature_cache.py`（有无缓存时纯 head 训练 loss 曲线一致；非 head-only/带增强被拒）、`test_packed_shared_memory.py`（两个"worker"索引数组一致；非 Linux 回退）、`test_base_checkpoint_broadcast.py`（双进程 gloo，文件仅打开一次、各 rank 状态一致）。

---

## 六、P6 部署与基准

### 1. 模型导出

`cls-trainer export --run <id> --checkpoint best_selection --config <cfg> [--format weights|onnx] [--out exports/]`

- **weights（默认）**：state dict + `export_manifest.json`：

  ```json
  {
    "run_id": "...",
    "checkpoint": "best_selection",
    "decision.threshold": 0.99,
    "base_checkpoint_sha256": "...",
    "model_factory": "...",
    "shape": [1, 2, 3, 208, 448],
    "metric_summary": {}
  }
  ```

  复用 `_resolve_checkpoint_state`，`decision.threshold` 是部署与训练共享的单一业务阈值。

- **onnx（可选）**：导出 logits `[B,2]`（输入 `[B,2,3,208,448]`）。**必须过验证 harness**：同一批 N 个样本上跑 ONNX Runtime 与 PyTorch 参考，最大绝对差 ≤ 1e-4；真实 factory 不可 trace（自定义算子等）时清晰失败并回退 `weights`（务实路径，文档注明）。

- 新增配置键：`export{format, output_dir, onnx_opset, verify_samples, include_threshold}`。

### 2. challenge 基准报告

`cls-trainer benchmark evaluate --config <cfg> --run <id> --checkpoint best_selection`：

- 对 `data.challenge_*`（或 `--split`/`--index-dir` 覆盖）计算 global/worst-game/worst-subtype FPR、recall、负样本 score p99/p99.9、分组计数；
- 跑 `benchmark.gate_metrics` 门，未达标返回非零退出码；
- 写 `benchmarks/<run>_<alias>/report.json` + `metrics_by_*.csv`；
- 以 `kind="benchmark"` 追加 `metrics/evaluation.jsonl`（经 `_append_evaluation_history`），**永不喂选择**；不碰 manifest。

### 3. release 就绪

- wheel 打包 configs/docs（`MANIFEST.in` 或 `[tool.setuptools.package-data]`）；lockfiles 纳入 sdist；
- `minimum_worst_game_f1` 保持模板 `null`（不编造基线），但新增 **release 门**：`cls-trainer config validate --release` 强制 release 配方设置非空 `minimum_worst_game_f1`（或新增 `configs/recipes/game_cls_release.yaml`）；
- CI wheel smoke 扩展为在合成 checkpoint 上跑 `export --format weights`。

**测试**：`test_export_verify.py`（weights 导出可 round-trip 进 `build_model`；ONNX 无 onnxruntime 时 skip）、`test_benchmark_evaluate.py`（报告 schema、门、追加 history 但永不选择）、`test_release_gate.py`（`--release` 校验强制非空）。

---

## 综合结论

step5 的三条主线回答的是三个不同层面的问题：

1. **WS1 建模轮**解决的是"楼梯误识别为地板/木桥"的业务问题——把困难负样本变成数据闭环（挖掘 → 标注 → sidecar → 采样 → challenge 验证），再用 `trainable_rules` 分阶段解冻让特征真正适配楼梯结构。
2. **WS2 工程硬化**解决的是可维护性与规模化问题——拆掉两个千行单体、用实测代替拍脑袋决定 worker/prefetch/packed、为 head-only 超参搜索缓存特征、避免 8 卡重复加载基座 checkpoint。
3. **WS3 部署基准**解决的是"训练完怎么交到生产"的问题——权重导出与单业务阈值对齐、固定 challenge 集给出不可漂移的验收口径、release 门防止空跑配置上线。

所有新增能力都遵守两条底线：**默认中立、不破坏既有逐字节行为**；**不可变 Run 与恢复身份不因新命令而失效**。分阶段实施、每阶段独立验收，确保任何一处改动都能单独回滚与归因。
