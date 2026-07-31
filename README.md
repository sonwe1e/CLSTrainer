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

正式 NPU 配置位于 `configs/npu_production.yaml`，不再继承 CUDA demo 配置。
其中的 `model.factory` 和 `model.checkpoint_path` 必须替换为真实模型；未替换、
checkpoint 不存在或非 `cls` 主干权重未完整加载时，训练会立即终止。

## 可扩展架构

CLSTrainer 当前的实现是**双帧二分类**，但训练主循环已经被拆成可替换的组件。
新增一个任务时，不需要同时修改 trainer、dataset、sampler、evaluator 和 report，
只需要新增对应组件并注册即可。

```text
ExperimentRunner
│
├── TypedExperimentConfig       严格 Pydantic schema，V1 自动迁移到 V2
├── RuntimeStrategy             AcceleratorAdapter + DistributedAdapter
├── DataModule                  LogicalDataSchema + IndexCodec + DataBackend + SamplingPolicy
├── TaskAdapter                 batch / forward / loss / prediction
├── TrainablePolicy             可训练参数选择（name_token / regex / model_declared）
└── EvaluatorSuite              DecisionPolicy + MetricAccumulator + GroupAggregator + ...
```

默认组件组合：

```text
DualFrameBinaryTask
GameVideoPairDataModule
LegacyGameBinaryIndexCodec
PngBackend / PackedUint8Backend
BalancedGameLabelDeltaPolicy
NameTokenTrainablePolicy(token="cls")
BinaryThresholdEvaluatorSuite + BinaryThresholdDecision(threshold=0.99)
NpuAccelerator + DdpDistributedAdapter
```

### 关键目录

```text
src/game_cls/
├── config/        严格 schema、V1→V2 migration、override 校验
├── contracts/     所有组件的 Protocol 接口（task / data / evaluation / runtime / trainable）
├── registry/       组件注册与解析
├── tasks/          TaskAdapter 实现（默认 dual_frame_binary）
├── data/           backends/（png、packed_uint8）、index_codecs/、sampling/
├── trainable/      TrainablePolicy 实现
├── evaluation/     EvaluatorSuite 组件
├── runtime/        AcceleratorAdapter / DistributedAdapter / RuntimeStrategy
└── engine/         ExperimentRunner + builders + state
```

### 配置版本

- **V1**（当前生产配置）：扁平的 `experiment/device/data/pair/sampler/model/...` 结构，
  加载时自动迁移到 V2 并打印 `[CONFIG MIGRATION]` 提示。
- **V2**：组件选择器结构，顶层 `task/trainable/data/sampler/runtime/evaluation`，
  未知的选择器键会报错，legacy 顶层键在迁移期继续容忍。

```yaml
config_version: 2
task:
  type: dual_frame_binary
  factory: game_cls.tasks.dual_frame_binary:build_task
trainable:
  policy:
    type: name_token
    params: {token: cls}
runtime:
  accelerator: {type: npu}
  distributed: {type: single_process}
```

### 添加第二个任务

未来实现第二个任务时，主要新增：

- 一个 `TaskAdapter`（`tasks/`）
- 一个 `DataModule/IndexCodec`（如需要）
- 一个 `SamplingPolicy`（如需要）
- 一个 `EvaluatorSuite`
- 一份严格配置

而不再修改现有双帧任务的 trainer、loss、evaluator 和报告代码。

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
相同只做 info 汇总。两位 `video_id` 默认按 split 内编号处理，跨 split 重名只
作为信息记录；只有确认它在整个项目中全局唯一后，才应启用
`require_unique_video_keys_across_splits` 强制检查。

SHA-256 会完整读取每张图片，是一次性但明显的 I/O 成本。百万帧数据推荐按以下
顺序准备，避免在远程小文件链路上反复扫描：

```text
复制 train/test 到本地 NVMe
→ 建索引并计算 SHA-256
→ 生成 packed 数据
→ 执行审计
→ 开始 smoke/正式训练
```

## 训练

CPU/CUDA 合成数据 smoke test：

```bash
python tools/train.py \
  --config configs/cuda_debug.yaml \
  train.max_steps=2 \
  evaluation.quick_test_every_steps=1 \
  evaluation.full_test_every_steps=2
```

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

周期性 full test 的角色明确标记为 `observed_dev_test`。训练摘要同时记录 last
checkpoint 指标、best observed dev-test 指标，以及 quick/full test 的执行次数。
最佳模型由 `evaluation.selection_metric` 决定；生产配置使用 global、macro-game
和 worst-game F1 的组合分数，并可通过 `minimum_worst_game_f1` 阻止单个游戏
灾难性退化。单类 `by_game_label` 行不再展示无意义的 F1/AUC：正类报告 recall
和 FN rate，负类报告 specificity 和 FP rate。

生产模板将 `minimum_worst_game_f1` 留为 `null`，因为没有可靠基线时不应猜测
门限。首轮稳定基线完成后，应根据各游戏结果设为非零值（例如基线明确支持时再
设为 `0.75`）。周期 full test 参与选模，因此其角色是 observed dev-test；对外
报告无偏结果时，还需要一个从未参与选模的独立 final test。若分数需要跨游戏和
版本解释为概率，还应在自然分布 calibration 集上拟合 temperature 和 bias。

## 精确恢复

完整 checkpoint 保存 epoch、`step_in_epoch`、global step、sampler 状态、优化器、
scheduler、scaler、CPU/CUDA/NPU RNG 和各 rank 独立 RNG。训练 pair 自带确定性增强
seed，因此恢复时可以直接从 epoch 内下一 batch 继续，不重新解码已经消费的 batch。

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
