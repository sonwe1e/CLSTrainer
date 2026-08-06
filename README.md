# CLSTrainer

双帧、多游戏二分类训练框架。输入为两张 `[B,3,208,448]` RGB 图像，模型输出
`[B,2]`，部署判定固定为第二通道 Softmax 概率严格大于 `0.99`。

希望先了解项目全貌时，可直接在浏览器打开自包含的中文教程
[`tutorial.html`](tutorial.html)。教程按“项目目的 → 技术架构 → 数据准备 →
训练评估 → NPU 验收”的顺序提供完整操作路径。

## 安装

项目的基础安装不会安装或替换 PyTorch，避免破坏已经与 CANN 匹配的
`torch + torch_npu` 环境：

```bash
python -m pip install -e .
```

CUDA 开发环境可以安装：

```bash
python -m pip install -e ".[cuda]"
```

测试与 CPU CI 使用：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Ascend 环境应先按服务器 CANN 版本安装匹配的 PyTorch 和 TorchNPU，再执行基础安装。

正式 NPU 配置位于 `configs/npu_production.yaml`,不再继承 CUDA demo 配置。
其中的 `model.factory` 和 `model.checkpoint_path` 必须替换为真实模型;未替换、
checkpoint 不存在或非 `cls` 主干权重未完整加载时,训练会立即终止。

## 命令行接口(CLI)

安装后注册 `cls-trainer` 命令(等价于 `python -m game_cls.cli`;`python tools/train.py`
继续作为 train 入口保留):

```bash
# 预检环境、数据、模型和配置
cls-trainer doctor --config configs/npu_1p.yaml

# 只校验配置,不初始化任何设备
cls-trainer config validate --config configs/npu_1p.yaml

# 查看最终合并配置;--with-source 标注每个字段来自哪个文件或覆写
cls-trainer config show --config configs/npu_1p.yaml --with-source

# 列出全部配置键及含义(生成件见 docs/config_reference.md)
cls-trainer config reference

# 预览运行计划(设备、全局 batch、总步数、评估频率、输出位置),不写任何文件
cls-trainer train --config configs/npu_1p.yaml --dry-run

# 正式训练
cls-trainer train --config configs/npu_1p.yaml [key=value ...]

# 从模板创建最小 Recipe
cls-trainer init --profile npu_8p --name my_run

# 查询实验
cls-trainer run list
cls-trainer run show latest
cls-trainer run compare RUN_A RUN_B
cls-trainer run export-tensorboard RUN_ID
```

## 配置分层:Task Profile / Profile / Recipe / Presets

普通用户只维护 Recipe,机器环境由 Profile 决定,稳定性参数用命名 Preset 选择,
任务事实由 Task Profile 提供默认值。Task Profile 不是不可变 Contract——
其中每个字段都只是默认值,recipe、preset 和命令行都可以覆写,框架始终以
解析后的配置为准。合并顺序为
`task_profile → profile → presets → recipe 自身 → 命令行覆写`:

```text
configs/
├── task_profiles/dual_frame_binary.yaml   # 阈值 0.99、test delta=2、cls-only 等任务默认值(可覆写)
├── profiles/                          # cpu_debug / cuda_1p / npu_1p / npu_8p
├── presets/
│   ├── augmentation/  none | light | standard
│   ├── dataloader/    stable | throughput
│   └── evaluation/    smoke | production
└── recipes/
    ├── example_debug.yaml             # 合成数据,开箱即跑
    └── game_cls_production.yaml       # 生产模板,填 REPLACE_ME 即用
```

Recipe 示例(只列需要决策的十几个字段):

```yaml
profile: npu_8p
presets:
  augmentation: standard
  dataloader: stable
  evaluation: production
experiment:
  name: game_cls_v1
model:
  factory: my_project.models:build_model
  checkpoint_path: /models/base_model.pt
data:
  train_index: indexes/train_frames.parquet
  val_index: indexes/val_frames.parquet
  test_index: indexes/test_frames.parquet
  train_video_index: indexes/train_video_entries.parquet
  val_video_index: indexes/val_video_entries.parquet
  test_video_index: indexes/test_video_entries.parquet
train:
  max_steps: 10000
  local_batch_size: 64
optimizer:
  learning_rate: 0.001
```

`profile`、`task_profile`、`presets.<组>` 也可以作为命令行覆写,便于 A/B:

```bash
cls-trainer train --config configs/recipes/game_cls_production.yaml \
  presets.dataloader=throughput
```

旧的扁平配置(`base:` 继承,如 `configs/npu_production.yaml`)继续可用。

## 严格配置校验

配置对照严格 Schema 校验:

* 未知键立即报错并给出候选建议。`optimzier.learning_rate=...` 不再静默创建
  一个无人读取的字段;
* 从未被消费的已移除字段(`optimizer.name`、`scheduler.name`、
  `evaluation.save_all_errors`)在配置时报出明确迁移说明,而不是假装生效;
* 业务判定阈值只有一个来源 `decision.threshold`,训练阈值损失与评估器共同读取;
  分别配置不一致的 `loss.threshold` / `evaluation.threshold` 会被拒绝;
* 全部配置键的类型、枚举和含义见 `cls-trainer config reference`。

## Run 管理

默认每次 `train` 启动都创建一个**不可覆盖的 Run**:`experiment.output_dir` 被视为
runs 根目录,每次启动在其下分配一个新的带时间戳目录:

```text
runs/dual_frame_game_cls_npu/
└── 20260804/
    └── 230712_dual_frame_game_cls_npu_a13f/
        ├── manifest.json         # 命令、git commit、主机、环境版本、seed、基座 ckpt SHA-256
        ├── status.json           # RUNNING/SUCCEEDED/FAILED 原子更新,含 step 与错误信息
        ├── console.log           # rank-0 的 stdout/stderr
        ├── resolved_config.json
        ├── summary.md            # 一页式人类摘要
        ├── overview.html         # 自包含曲线/指标报告,浏览器直接打开
        ├── train_metrics.jsonl
        ├── reports/
        ├── checkpoints/
        └── training_summary.json
```

重复执行同一条命令永远不会覆盖上一次实验;`<runs 根>/index.jsonl` 为每个 run
记录一行,`cls-trainer run list/show` 据此查询;`cls-trainer run compare A B`
输出两个 run 的配置差异与最佳指标对比;`cls-trainer run export-tensorboard`
把曲线导出为 TensorBoard 事件文件。训练异常退出时 `status.json`
记录 `FAILED`、错误类型和 `failure.log`,不会只剩一堆无法判断状态的中间文件。
训练结束不再把整个嵌套结果打印到终端,而是输出简洁摘要和各关键文件路径。

实验谱系与恢复安全性:

* `cls-trainer train --resume <run 目录>` 在原目录内继续。恢复前会对比当前
  配置与该 run 的 `resolved_config.json`:改变 `decision.threshold`、图像规格、
  seed、模型工厂等关键字段会被直接拒绝(退出码 3),其余差异打印 warning;
* `cls-trainer train --fork <run>` 以某个 run 的配置为起点创建新 run,
  manifest 与索引中记录 `parent_run_id` 父子关系。

## 数据索引和严格审计

目录必须满足：

```text
<split>/<game>/<0|1>/<video_id><frame_id>.png
```

文件名满足 `^\d{2}\d{5}\.png$`。执行：

```bash
python tools/build_index.py \
  --config configs/npu_production.yaml \
  --train-root /data/train \
  --test-root /data/test \
  --output-dir indexes

python tools/audit_dataset.py \
  --config configs/npu_production.yaml \
  --index-dir indexes \
  --output-dir reports/data_audit \
  --strict
```

索引、审计、packed 和训练共同从 `data.width/height/channels` 创建
`ImageSpec`。当前生产规格是 `width=448`、`height=208`、`channels=3`，
因此单帧 tensor 为 `[3,208,448]`，训练输入为 `[B,2,3,208,448]`。系统不会
自动交换宽高或 resize；首个训练 batch 的形状不一致会立即终止。

扫描器只接受直接位于 `<game>/<0|1>` 下、扩展名符合配置的帧。MP4、JSON 等
非帧文件会被忽略并计数，`_` 或 `.` 开头及配置命中的缓存目录会被整目录剪枝；
直接出现的非法命名 PNG 仍是 error，非忽略嵌套目录默认为 warning。ignored
报告每类只保留有限示例，不会把百万个辅助文件路径写入 JSON。

默认正式训练启用 `data.strict_audit`。尺寸、通道、非法帧、类别完整性或 test
`delta=2` pair 不符合要求时，训练会在创建 DataLoader 前终止。审计报告同时
列出 `game × label × delta` 的合法 pair 数；生产配置要求每个游戏、每个标签
至少存在一个 `delta=2` pair。

训练阶段不会物化数百万个 `PairSample`：内存中只保留视频级帧数组和各 delta
合法起点，sampler 在每个 step 懒生成 pair。测试 pair 使用 rank-local NumPy
紧凑数组，不补齐、不重复。

索引阶段还会生成 `train_video_entries.parquet` 和
`test_video_entries.parquet`，训练直接按视频行读取，不再让每个 rank 将百万帧
转换成 Python dict 和 `FrameRecord`。默认同时计算 SHA-256：相同内容但标签
不同属于 fatal error；同标签的跨 split 或 split 内重复属于 warning；仅文件名
相同只做 info 汇总。两位 `video_id` 不是全局身份；索引阶段生成跨 label 的
`source_video_uid`（`game::video_id`）作为泄漏检查单位。同一个源视频横跨
train/val/test（video key 重叠或 source uid 重叠）以及相同 SHA-256 内容跨
split，在严格审计中一律是 error；`require_unique_video_keys_across_splits`
不再需要显式开启，旧的“跨 split 重名仅警告”语义已移除。

SHA-256 会完整读取每张图片，是一次性但明显的 I/O 成本。百万帧数据推荐按以下
顺序准备，避免在远程小文件链路上反复扫描：

```text
复制 train/val/test 到本地 NVMe
→ 建索引并计算 SHA-256（build_index.py --val-root 生成三份 split 索引）
→ 生成 packed 数据
→ 执行审计
→ 开始 smoke/正式训练
```

## 训练

CPU/CUDA 合成数据 smoke test:

```bash
python tools/train.py \
  --config configs/cuda_debug.yaml \
  train.max_steps=2 \
  evaluation.val_quick_every_steps=1 \
  evaluation.val_full_every_steps=2
```

产物位于 `runs/dual_frame_game_cls_debug/<日期>/<时间戳>_<名称>_<id>/` 下的新
run 目录;结束后的终端输出会给出 `summary.md`、`status.json` 等关键文件路径。
需要就地写入固定目录时(例如分阶段验收脚本)显式加 `--run-mode fixed`。

Ascend 单卡和八卡：

```bash
bash scripts/run_npu_1p.sh
bash scripts/run_npu_8p.sh
```

正式长跑前先执行固定验收入口：

```bash
bash scripts/smoke_npu_1p.sh
bash scripts/smoke_npu_8p.sh
```

单卡脚本按“无 worker 基线 → spawn 1 worker → spawn 2 workers + 增强 → quick
test → full test”的顺序逐级验收；任一阶段失败都会停止，不会把 DataLoader
问题误判为模型或算子问题。八卡脚本先固定每 rank 1 个 train worker 和 1 个
按需启动的 eval worker，运行 100 step 并触发 quick/full 与 checkpoint。必须在
真实 910B2 环境确认通过后，才能把 CPU/Gloo 测试结论扩展到 HCCL。

NPU 运行时会先导入 `torch_npu`、绑定设备，再初始化 HCCL。训练 batch 在 CPU
侧保持 `uint8`，一次传输到设备后再转换为 FP32/BF16/FP16 并归一化。

生产配置强制 NPU 多 worker DataLoader 使用 `spawn`，并设置每批最长等待时间，
避免 NPU runtime 已初始化后再 fork worker 导致静默挂起。train 与 eval 使用独立
参数：train worker 常驻，quick/full 的 eval worker 不常驻；默认先关闭 pin
memory，稳定后再通过 A/B 测试决定是否开启：

```yaml
dataloader:
  multiprocessing_context: spawn
  timeout_seconds: 180
  worker_num_threads: 1
  train:
    num_workers: 2
    persistent_workers: true
    prefetch_factor: 2
    pin_memory: false
  eval:
    num_workers: 1
    persistent_workers: false
    prefetch_factor: 2
    pin_memory: false
```

旧配置中的根级 `dataloader.num_workers` 等字段仍可作为 fallback 使用，但角色级
配置优先。NPU 下只要任一角色的 `num_workers>0`，显式配置 `fork` 或
`forkserver` 都会在设备初始化前报错。启动日志会打印解析后的 worker、context
和 timeout，并分别标记开始等待与首 batch 返回时间、shape、dtype；因此 worker
异常最长在 180 秒内转为明确的 DataLoader timeout，而不是无限等待。

如 Profiler 确认 PNG 解码仍是瓶颈，可预解码为固定大小 uint8 分片：

```bash
python tools/pack_dataset.py \
  --config configs/npu_production.yaml \
  --frame-index indexes/train_frames.parquet \
  --output-dir /local_nvme/train_packed

python tools/pack_dataset.py \
  --config configs/npu_production.yaml \
  --frame-index indexes/test_frames.parquet \
  --output-dir /local_nvme/test_packed
```

分别打包 train/test，然后把生产配置的 `data.backend` 改为 `packed_uint8`，
并设置 frame/video 两组 packed index。也可直接以
`configs/npu_production_packed.yaml` 为模板。新版 packed 索引以连续整数定位
帧，不再为每个 rank 建立百万项路径字典；shard 路径相对 manifest 保存，运行时
仅维护最多 `packed_max_open_shards` 个 LRU memmap。该后端避免训练热路径中的
PNG 解压，数据仍应优先复制到本地 NVMe。

尺寸、扫描策略或重复数据策略发生变化后，必须更换或删除旧 `indexes`，重新执行
build、audit 和 train/test packed 打包。审计格式、记录的图片规格或重复策略与
当前配置不一致时，训练会明确要求重建，而不会继续使用旧产物。

普通生产配置默认保留 `backend: png`，便于直接接入原始数据；高吞吐训练应显式
使用 `configs/npu_production_packed.yaml`。正式选择后端前，应保持模型、batch、
worker 和训练步数完全一致，分别运行 500～1000 step，对比：

- 同步后的 `interval_samples/s` 和 `data_wait_ratio`；
- CPU 使用率及主进程/worker RSS；
- NPU 利用率和 step P95；
- page cache 稳定后而非冷启动阶段的吞吐。

默认每 shard 4096 张图，约 1.07 GiB。随机均衡采样下建议实测
`images_per_shard=512/1024/2048/4096` 与
`packed_max_open_shards=8/16/32` 的组合；memmap 只建立映射，不等于一次读取
整个 shard，但 shard 大小和 LRU 数量会影响 page cache 命中率。

packed backend 会通过 `get_many()` 将一个 batch 的帧按 shard 分组，预分配单个
连续 uint8 输出并执行 `np.copyto`；训练和评估 Dataset 的 `__getitems__()` 会
把整批 pair 一次交给该接口。PNG 后端仍保持逐图片解码兼容路径。

接入真实模型时，将 `model.factory` 设置为 `包名.模块名:函数名`。工厂函数必须
返回接受 `(image0, image1)` 并输出 `[B,2]` 的 `torch.nn.Module`。
只训练 `cls` 时，所有非 `cls` 参数和 buffer 必须 100% 从基础 checkpoint 加载；
该严格规则固定生效，不提供容易产生误解的关闭开关。

## 评估契约

quick test 会从每个 `(game,label,video)` 的 `delta=2` pair 中按时间均匀选取固定
数量；full test 枚举全部合法 pair。分布式运行时，每个 rank 处理不重复分片，
混淆矩阵使用 int64、损失使用 float32 all-reduce，兼容 HCCL。full test 默认用
固定直方图分布式计算 ROC-AUC/PR-AUC，不再把百万 Python 分数集中到 rank 0；
错例在 batch 内流式写分片，再由 rank 0 流式合并。评估前向按
`evaluation.amp/amp_dtype` 使用与部署一致的 BF16/FP16；CE、Brier 和混淆矩阵
在设备上累计，结束时再统一归约，避免每个 batch 多次 `.item()` 同步。
真实视频评估使用连续的 game、game-label 和 video 整数 ID，通过 NumPy
`bincount` 按 batch 聚合分组混淆矩阵；Python 逐样本处理仅用于 FP/FN 和临界
样本。Parquet writer 默认累计 `parquet_row_group_size=4096` 条记录再写 row
group，避免每个 batch 产生一次小写入。

full test 报告包含：

```text
metrics.json
metrics_by_game.csv
metrics_by_video.csv
metrics_by_game_label.csv
false_positive.parquet
false_negative.parquet
near_threshold.parquet
errors.html
previews/*.png
```

quick test 仅写 `metrics.json`、受 `quick_save_error_limit` 全局限制的少量
FP/FN 与 near-threshold Parquet，不再生成 HTML 和分组 CSV，避免短周期评估
承担完整报告开销。同一步同时满足 quick/full 周期时只运行 full。

PNG 后端的 HTML 直接引用原图；packed 后端只为 HTML 上限内的错例解码并导出
`previews/*.png`，浏览器不会再尝试加载无效的 `packed://` 地址。完整 FP/FN
Parquet 仍保存紧凑的 packed frame index，不会为全部错例重复导出图片。

`near_threshold.parquet` 只保存 `0.98 <= p1 <= 0.995` 的样本；更高置信度只记录
区间计数。指标同时包含 Brier Score、20-bin ECE 和置信度直方图。由于训练采用
50/50 平衡采样并加入阈值损失，`0.99` 应解释为固定业务分数阈值，而不是天然
校准后的真实发生概率。

周期性 full validation 的角色明确标记为 `validation`。训练摘要同时记录 last
checkpoint 指标、best validation 指标，以及 train probe/quick/full validation
的执行次数。最佳模型由 `evaluation.selection_metric` 决定；生产配置使用 global、
macro-game 和 worst-game F1 的组合分数，并可通过 `minimum_worst_game_f1` 阻止
单个游戏灾难性退化。单类 `by_game_label` 行不再展示无意义的 F1/AUC：正类报告
recall 和 FN rate，负类报告 specificity 和 FP rate。

生产模板将 `minimum_worst_game_f1` 留为 `null`，因为没有可靠基线时不应猜测
门限。首轮稳定基线完成后，应根据各游戏结果设为非零值（例如基线明确支持时再
设为 `0.75`）。

### 训练/验证/测试三分协议

数据角色严格三分：`train` 只用于梯度更新；`validation` 用于曲线、模型选择和
early stopping；`test` 在训练结束后只评估一次，绝不参与选模。旧配置的
`quick_test_*`/`full_test_*` 键会在 `finalize_config` 中自动迁移为
`val_quick_*`/`val_full_*`；若未配置 `data.val_index`，`test_index` 会被临时
用作 validation 并打印明确警告——该 run 没有独立测试集。

训练循环新增四个评估角色：

| DataLoader | 数据 | 增强 | 用途 |
| --- | --- | :-: | --- |
| `train_probe` | 固定 train 子集 | 关 | 与 validation 同口径比较泛化差距 |
| `val_quick` | 固定 validation 子集 | 关 | 高频趋势观察 |
| `val_full` | 全部 validation | 关 | best 模型选择与 early stopping |
| `test_full` | 全部 test | 关 | 仅由 `cls-trainer evaluate` 执行 |

```bash
# 训练结束后，对从未参与选模的 test split 执行一次最终评估：
cls-trainer evaluate --run <RUN_ID> --checkpoint best_selection --split test
```

每次评估追加一行到 `<run>/metrics/evaluation.jsonl`，与训练指标一起驱动
`overview.html` 的六类曲线（训练损失、probe vs validation CE/selection、
validation F1 三分、Brier/ECE、学习率/梯度/吞吐）和关键 step 标记（best
selection、best val loss、泛化退化起点、early-stop）。训练指标改为区间内按样本
加权平均（`interval_loss` 等），多卡下在 rank 间 all-reduce；原始 last-batch 值
仍保留在 `loss`/`ce`/`threshold_loss` 字段中。

多卡评估的分组统计（按 game/video 的混淆矩阵、概率 min/max/count）在共享
catalog 下直接对 per-catalog 数组做 tensor `all_reduce`（min/max 用 MIN/MAX
归约），不再把大型 Python 字典 gather 到 rank 0；只有无共享 catalog 的小字典
路径保留 `gather_object`。精确 AUC 模式（`full_auc_mode: exact`）在分布式下
超过 200 万样本会报错，建议大测试集改用默认的 histogram 模式。

```yaml
early_stopping:
  enabled: true
  monitor: selection_score
  mode: max
  full_validation_only: true
  burn_in_steps: 6000
  patience_evaluations: 3
  min_delta: 0.001
  restore_best: true
```

early stopping 只在 full validation 上判断（quick 子集波动太大），burn-in 前
不停止，连续 `patience` 次未改善则停止并恢复最佳权重。状态（best_value、
best_step、bad_evaluation_count、stop_reason）写入 checkpoint，恢复训练不会
重算 patience。四种稳定 checkpoint：`model_last.pth`、`model_best_selection.pth`、
`model_best_val_loss.pth`、`model_best_worst_game.pth`（旧别名
`best_observed_dev_test_selection` 继续生成以兼容旧脚本）。

还可以用 `checkpoint.save_topk` 保留按 `checkpoint.topk_monitor`（默认
`selection_score`）排序的 Top-K 全量验证 checkpoint
（`model_topk_<step>.pth` 加 `checkpoints/topk_registry.json`），
避免“best 或 last 恰好都不是最优”的情况；registry 随训练状态持久化，
恢复后继续维护。

全局 batch 内采样去重用 `data.deduplication` 配置：

```yaml
data:
  deduplication:
    level: pair            # none | pair(默认) | video
    on_exhaustion: warn_and_relax   # error | warn_and_relax(默认)
```

`pair` 避免完全相同的 (video, delta, start) 三元组，`video` 禁止同一视频在
一个全局 batch 内出现两次；`on_exhaustion: error` 在无法填满 batch 时直接
失败，默认 `warn_and_relax` 会记录并放宽。每 epoch 的去重失败次数写入
训练指标与 `evaluation_state`。全零采样权重（如 `class_probability`）现在
直接报配置错误，不再静默回退到均匀采样。

## 精确恢复

推荐入口是 `cls-trainer train --resume <run 目录>`:自动读取该 run 的
`resolved_config.json` 与 `checkpoints/checkpoint_last.pth`,并在原 run 目录内
继续,不新建目录。

恢复是**按身份语义**进行的,与 Run 的不可变 manifest 保持一致:

- `manifest.json` 创建后不可修改;恢复保留原始 `run_id`、创建时间和谱系信息;
- 首次配置快照保存在 `resolved_config.initial.json`,每次恢复只更新
  `resolved_config.json`;
- 每次恢复追加一行 `resume_events.jsonl`(时间、checkpoint、命令、配置差异、
  resume 类型);
- `status.json` 保留首次 `started`,新增 `resumed_at`;
- Run 索引是 append-only 事件日志,读取时按 `run_id` 聚合到最新状态,恢复
  后 `run list` 不再显示旧的失败/中间状态。

恢复前会做**对称的配置漂移检查**(新增/删除/修改都检测):

- `resume-exact`:无任何轨迹差异,精确继续;
- `resume-extend`:仅 `max_steps`/`stop_after_steps` 变化,允许继续但明确
  重新规划 scheduler;
- 其余任何影响轨迹的变更(优化器、数据、模型、采样概率、分布式配置、
  augmentation 等)属于 **critical / fork**,恢复被拒绝,改用
  `--fork` 创建新 Run。

完整 checkpoint 保存 epoch、`step_in_epoch`、global step、sampler 状态、优化器、
scheduler、scaler、CPU/CUDA/NPU RNG 和各 rank 独立 RNG。训练 pair 自带确定性增强
seed，因此恢复时可以直接从 epoch 内下一 batch 继续，不重新解码已经消费的 batch。
内部恢复 checkpoint 带有 `cls_training_checkpoint` 可信标记,无标记的文件会被拒绝;
基础模型 checkpoint 与所有导出路径一律使用 `weights_only=True` 加载。

生产配置的周期性恢复 checkpoint 只保存可训练状态、优化器和基础权重哈希；
同时记录并严格核对预期 trainable state keys。产物契约为：

```text
model_<tag>.pth                 纯 state_dict，可直接 load_state_dict
model_<tag>.metadata.json       global_step、epoch、artifact role
checkpoint_<tag>.pth            完整训练与恢复状态
```

周期恢复文件为 `checkpoint_last.pth`。完整纯权重按较低频率保存；同一步同时触发
best 与 last 时只序列化一次，再创建稳定别名，减少冻结主干的重复 I/O。部署端可
继续直接执行：

```python
model.load_state_dict(torch.load("model_last.pth", map_location="cpu"))
```

普通训练日志中的细分时间仍明确标记为 host enqueue 时间，不表示 NPU 实际算子
耗时。每个日志区间结束时会同步 NPU，再报告可比较的
`interval_samples/s`、`interval_step_time`、`data_wait_ratio`、学习率、梯度
范数以及评估和 checkpoint 耗时。相同内容同时以逐行 JSON 写入
`<experiment.output_dir>/train_metrics.jsonl`，可直接用于曲线、TensorBoard
转换或实验对比。设备级算子瓶颈仍须使用 NPU Event 或 TorchNPU Profiler 验证。
