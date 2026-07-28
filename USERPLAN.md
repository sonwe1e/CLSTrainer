# 结论

这次修改是一次**实质性升级**。上次提出的大部分结构性问题已经解决：训练入口可导入、CPU CI 已通过、NPU 设备初始化顺序正确、quick/full test 分离、评估支持多 rank 分片、训练 pair 改为懒生成、严格数据审计和精确恢复框架也已加入。该分支的提交 `b992446...` 已通过 CPU checks。

但目前仍然**不建议直接进行长时间八卡训练**。存在两个优先级最高的问题：

1. 八卡第一次分布式评估很可能因 `float64 HCCL AllReduce` 失败。
2. NPU 正式配置仍继承 CUDA 调试配置中的 demo 模型和空 checkpoint，直接运行脚本可能训练错模型。

整体成熟度可以从上次的约 **3～4/10 提升到 6/10**。修复下面的 P0 问题并完成一次真实模型的八卡 smoke test 后，才能进入长跑性能调优。

---

# 一、上次问题的解决情况

| 上次问题                           | 当前状态                                             | 判断                           |
| ------------------------------ | ------------------------------------------------ | ---------------------------- |
| `reports` 模块缺失，训练入口可能直接导入失败    | 已增加 `reports/__init__.py` 和完整的 `error_writer.py` | **已解决**                      |
| HCCL 初始化早于 `torch_npu` 导入和设备绑定 | 现在先导入 `torch_npu`、调用 `set_device`，再初始化进程组        | **已解决**                      |
| 没有 CI，无法发现主链路错误                | 已增加 pytest 和 2-step 训练 smoke test，CI 已成功         | **已解决，但仅覆盖 CPU**             |
| quick test 实际跑完整 test          | 已创建独立 quick/full dataset 和独立触发频率                 | **已解决**                      |
| 只有 rank 0 测试，其余 7 卡等待          | 每个 rank 处理不重复的测试分片                               | **基本解决**                     |
| 数百万 `PairSample` 全量物化          | 训练改为视频索引和 `PairRequest` 懒采样                      | **主要问题已解决**                  |
| 游戏、视频指标缺失                      | 已增加 by-game、by-video、by-game-label               | **实现了，但指标定义有新问题**            |
| checkpoint 不能恢复 epoch 内位置      | 已保存 `step_in_epoch`、sampler 和各 rank RNG          | **机制基本解决**                   |
| 数据审计只报告、不阻止训练                  | 错误尺寸和非法文件不再写入索引，正式训练默认严格检查                       | **基本解决**                     |
| 项目安装可能替换 Ascend PyTorch        | PyTorch 已移到 `cuda/dev` 可选依赖                      | **已解决**                      |
| CPU 将图像变成 FP32 后再传 NPU         | 增强后保持 uint8，在设备端转换和归一化                           | **已解决，但引入 RandomErasing 问题** |
| 游戏视频数会因不同类别相同 video ID 而少计     | 新 sampler 使用不同 label 下的视频索引数量求和                  | **已解决**                      |
| delta 实际比例偏离 15/70/15          | 新 sampler 改为 delta-first，并记录实际比例                 | **已解决**                      |

相关实现可见运行时初始化、懒数据集和 sampler。

---

# 二、当前新增或仍未解决的 P0 问题

## 1. 分布式评估的 `float64 AllReduce` 会阻断 A3 评估

评估代码创建了一个位于 NPU 上的 `torch.float64` 张量：

```python
numeric = torch.tensor(
    [tp, fp, fn, tn, cross_entropy_sum, sample_count],
    dtype=torch.float64,
    device=device,
)
dist.all_reduce(numeric)
```

Atlas A3 的 HCCL AllReduce 支持 `int8/int16/int32/int64/float16/float32/bfloat16`，不支持 `float64`。因此八卡训练可能正常运行，直到第一次 quick/full test 才报错。([hiascend.com][1])

建议立即拆成两个统计张量：

```python
counts = torch.tensor(
    [tp, fp, fn, tn, sample_count],
    dtype=torch.int64,
    device=device,
)
loss_sum = torch.tensor(
    [cross_entropy_sum],
    dtype=torch.float32,
    device=device,
)

dist.all_reduce(counts)
dist.all_reduce(loss_sum)
```

只有修复这一项后，分布式评估链路才具备基本可运行性。

---

## 2. NPU 配置仍然会默认构建 demo 模型

`npu_1p.yaml` 继承：

```yaml
base: cuda_debug.yaml
```

但没有覆盖 `model.factory` 和 `checkpoint_path`。

而基础配置仍然是：

```yaml
model:
  factory: game_cls.model.builder:build_demo_model
  checkpoint_path: null
```

这个 demo 模型只是一个 6 通道深度卷积、全局池化和 Linear。

因此直接执行：

```bash
bash scripts/run_npu_8p.sh
```

并不会训练你的真实模型。

正式配置应该与 debug 配置分离，并增加强制校验：

```python
if not data_cfg["synthetic"]:
    if model_cfg["factory"].endswith(":build_demo_model"):
        raise RuntimeError("Production training cannot use build_demo_model")
    if not model_cfg.get("checkpoint_path"):
        raise RuntimeError("Production training requires checkpoint_path")
```

同时应覆盖输出目录和 scheduler 等调试默认值。当前 NPU 配置还继承了 CUDA debug 的 `warmup_steps: 10`，这对于 10,000 step 的正式训练通常过短。

---

# 三、会影响训练效率的主要问题

## 1. 完整评估仍然具有 O(N) 的 Python 内存和中心化通信

每个 rank 在完整测试期间保存：

```python
margins: list[float]
targets: list[int]
errors: list[dict]
near_threshold: list[dict]
```

随后通过：

```python
dist.gather_object((margins, targets), ...)
```

把所有预测分数和标签集中到 rank 0。

在百万级测试 pair 下，这会产生：

* 每个 rank 的大量 Python float、int 和 dict 对象；
* rank 0 同时保存所有 rank 的列表；
* Python pickle 序列化和反序列化；
* rank 0 对全量分数进行排序，计算 ROC-AUC 和 PR-AUC；
* 其他 NPU 在 rank 0 聚合和写报告期间等待。

这虽然比“rank 0 单卡完成全部前向”更好，但仍可能成为完整测试时的主要内存和时间瓶颈。

### 推荐调整

固定阈值指标只需要：

```text
TP / FP / FN / TN / CE sum / count
```

直接 AllReduce 即可，不应汇聚全部样本。

ROC-AUC 和 PR-AUC 可以采用两种模式：

* quick test：继续精确汇总，数据量小；
* full test：各 rank 把 `float32 margin + uint8 label` 写入 Arrow/NumPy 分片，由 rank 0 流式合并，或者用固定直方图区间近似计算。

错误样本也应在评估 batch 内流式写 Parquet，而不是全部放进 `errors` 列表后才落盘。

---

## 2. `near_threshold` 可能保存几乎全部高置信度正样本

当前 `_threshold_band()` 将：

```python
probability >= 0.999
```

也定义为 near-threshold。

如果模型训练良好，大量类别 1 样本都会高于 0.999，于是 `near_threshold` 文件可能比真正的错误文件还大。

更合理的设计是：

```text
样本级 near-threshold：
0.98 <= p <= 0.995

只统计数量、不保存逐样本：
p >= 0.995
p >= 0.999
```

置信度分布应保存计数直方图，而不是保存所有高置信度样本路径。

另外，恰好等于 `0.990` 的样本当前不属于任何区间，因为第二段使用了 `0.990 < probability`，应改为 `0.990 <= probability`。

---

## 3. 视频级索引仍会在每个 rank 瞬间物化百万 Python 行

训练 pair 已不再物化，这是明显进步。但 `read_video_entries_parquet()` 仍执行：

```python
table = pq.read_table(...)
table.to_pylist()
FrameRecord(**row)
```

对于一百万帧，每个 rank 都会暂时创建：

```text
100万 Python dict
100万 FrameRecord
100万路径字符串引用
```

然后再转换为 `VideoEntry`。八个 rank 同时启动时，这可能形成很高的 CPU 内存峰值。

`video_index_memory_bytes()` 只统计数组字节和字符串 UTF-8 长度，没有统计：

* Python string、tuple、dict、list 对象头；
* NumPy 对象自身；
* DataLoader worker 的数据集副本；
* PyArrow `to_pylist()` 的瞬时内存。

所以日志中的 `video_index_bytes_per_rank` 会显著低估真实 RSS。

更适合生产环境的是在索引阶段直接生成一个真正的视频级文件：

```text
train_video_entries.parquet
test_video_entries.parquet
```

每行包含：

```text
game_id
label
video_id
frame_ids: list<int32>
frame_paths: list<string>
valid_starts_delta1: list<int32>
valid_starts_delta2: list<int32>
valid_starts_delta3: list<int32>
```

训练进程直接读取该文件，不再从 frame 表重新分组。

---

## 4. PNG 解码仍可能是核心吞吐瓶颈

每个训练样本仍然执行两次：

```python
Image.open(...)
image.convert("RGB")
```

在你约 300 MB/s 的远程链路以及百万小文件场景下，即使消除了 pair 对象，仍然存在：

```text
远程随机小文件访问
+ PNG 解压
+ 两次文件打开
+ CPU 数据增强
```

当前分支还没有 packed dataset、tar shard、LMDB 或预解码 uint8 后端。因此它解决的是**索引内存问题**，还没有解决主要的数据吞吐问题。

执行顺序仍应是：

```text
本地 NVMe
→ 调 worker/batch
→ PNG tar 分片
→ 仍慢时预解码 uint8 分片
```

---

## 5. quick 和 full 在同一步会重复执行

NPU 配置为：

```yaml
quick_test_every_steps: 1000
full_test_every_steps: 5000
```

在 step 5000、10000 时，代码会先运行 quick，再立即运行 full。

应改为：

```python
run_full = full_every and step % full_every == 0
run_quick = quick_every and step % quick_every == 0 and not run_full
```

---

## 6. checkpoint 会产生严重重复 I/O

每次保存会连续写：

```text
model_<tag>.pth                 完整模型
checkpoint_<tag>.pth            再包含一次完整模型
```

如果某个 step 同时是：

```text
full test 最佳
+ save_last_every_steps
```

则会保存：

```text
best model-only
best full checkpoint
last model-only
last full checkpoint
```

相当于连续序列化四次完整模型。只有 `cls` 在变化，但冻结主干仍然被重复保存。

这是典型的利用率锯齿来源。

建议：

* 周期性恢复 checkpoint 只保存 `cls` 参数、优化器和基础权重哈希；
* `model_last.pth` 和 `model_best.pth` 保存完整模型；
* 同一步的 best 和 last 复用同一个临时 state dict；
* 不要让 quick test 写完整错误报告和 checkpoint；
* 降低完整模型保存频率。

---

## 7. 当前分段计时在 NPU 上不准确

代码使用 `time.perf_counter()` 包围 H2D、forward、backward 和 optimizer。

NPU 操作是异步下发的，因此这些时间主要是 Python enqueue 时间，实际设备耗时可能在后续 `.item()`、评估、保存或其他同步点才体现出来。当前打印的 `forward/backward/h2d` 不能可靠定位锯齿根因。

建议：

* 正常训练不加同步；
* 专门的 profile 模式使用 NPU Event 或 Profiler；
* 同时报告 `train_only_samples/s` 和包含评估、checkpoint 的 `wall_samples/s`。

---

# 四、会影响训练精度和指标可信度的问题

## 1. 当前“每视频 F1”定义不正确

视频分组键为：

```python
(game, video_id)
```

没有包含 label。

如果类别 0 和类别 1 文件夹中都存在 `video_id="01"`，两组完全不同的视频会被合并。

即使不存在重号，每个视频通常只有一个真实类别：

* 类别 0 视频没有正样本，它的 positive-class F1 永远为 0；
* 类别 1 视频没有负样本，无法评价 specificity。

因此：

```python
macro_video_f1 = mean(每个视频的pair级F1)
```

在统计上没有合理含义，会让所有负类视频贡献 0 分。

### 正确做法

分组键至少改为：

```python
(game, label, video_id)
```

然后每视频报告：

```text
label=1：recall、FN rate、平均/最小置信度
label=0：specificity、FP rate、最大置信度
```

如果确实需要“视频级 F1”，应先把一个视频的多个 pair 聚合为一个视频预测，再在所有视频之间计算一次 F1，而不是平均每个单类视频的 F1。

`by_game_label` 同样不应把单类分组的 F1 作为主要指标。

---

## 2. uint8 优化改变了 RandomErasing 的实际语义

当前增强链保持 uint8，并直接执行：

```python
v2.RandomErasing(value="random")
```

Torchvision 的 `value="random"` 实现生成的是均值为 0、标准差为 1 的 `float32` 正态噪声，并直接赋值到输入张量。([PyTorch Documentation][2])

在 uint8 图像上，这并不是预期的 `[0,255]` 随机 RGB 噪声，而会被转换为少数接近 0 或发生整数转换后的值。它可能不会报错，但增强语义已经发生变化。

建议选择以下一种：

```yaml
random_erasing:
  value: 0
```

或者：

```yaml
random_erasing:
  value: [127, 127, 127]
```

若确实需要随机 RGB 噪声，应实现 uint8 专用版本：

```python
torch.randint(0, 256, size, dtype=torch.uint8)
```

或者将 RandomErasing 移到转换为 `[0,1]` 浮点之后。

---

## 3. 平衡采样会改变概率先验，0.99 不再天然是“99%概率”

训练阶段强制每个游戏内部类别约 50/50，但真实部署数据中的类别 1 比例可能远低于 50%。同时阈值辅助损失主动推动正样本 margin 超过 `log(99)`。

这能帮助提高固定阈值召回，但会改变输出概率的校准性。换言之：

> 训练后的 `Softmax=0.99` 更接近业务分数 0.99，而不一定代表真实发生概率为 99%。

建议增加：

```text
Brier Score
ECE
正负样本置信度直方图
自然分布下的校准偏置
```

最佳方案仍然是从训练视频中保留一个自然分布的 calibration split，仅拟合一个：

[
d_{calibrated}=d/T+b
]

若坚持不设 calibration，则不要把 0.99 解释为概率，只将其定义为固定业务阈值。

---

## 4. 权重加载仍然过于宽松

当前权重加载会静默跳过形状不匹配参数，并使用 `strict=False` 继续训练。

训练代码只是打印：

```text
Missing
Unexpected
Shape mismatch
```

不会中止。

因为你只训练 `cls`，如果 backbone 大量权重未正确加载，模型几乎没有机会修复，最终精度会灾难性下降。

应设置生产模式规则：

```text
所有非 cls 权重必须成功加载
只允许最后分类层因输出维度不同而 shape mismatch
基础模型加载覆盖率必须接近 100%
checkpoint 不允许为空
```

---

## 5. 数据审计还没有检查 train/test 泄漏

当前严格审计覆盖：

* 路径结构；
* 文件名；
* 尺寸和通道；
* 每个游戏是否有两个类别；
* test 是否存在合法 delta=2 pair。

但没有检查：

```text
同一个视频是否同时存在于 train 和 test
同一图片内容是否跨集合重复
近重复帧是否被复制到不同集合
```

视频泄漏会让 test F1 虚高，尤其相邻帧高度相似。

建议至少检查：

```text
(game, label, video_id) 跨 split 重复
文件哈希重复
每个视频抽样帧的感知哈希重复
```

---

## 6. 所有 BatchNorm 都被强制冻结，包括 cls 内部的 BatchNorm

当前代码先把 `cls` 模块设为 train，然后又把整个模型中的所有 BatchNorm 设为 eval。

如果你的多个 `cls` 卷积中包含 BatchNorm：

* affine weight/bias 可以训练；
* running mean/variance 不会更新。

这可能是你想要的，也可能限制新游戏数据的适应能力。

建议拆成两个配置：

```yaml
freeze_backbone_batchnorm_stats: true
freeze_cls_batchnorm_stats: true/false
```

分别做消融实验，而不是全局一刀切。

---

## 7. 最终 full test 不参与最佳模型更新

训练末尾如果额外运行 `full_final`，代码只更新：

```python
evaluation_state["last_full_metrics"]
```

没有与 `best_metrics` 比较，也没有保存 final-best checkpoint。

因此最后一个 checkpoint 即使取得最高 F1，也不会成为：

```text
best_observed_dev_test_f1_tau099
```

应把周期性 full 和 final full 的最佳模型判断抽成同一个函数。

---

# 五、当前测试覆盖仍缺少什么

CPU CI 的建立是非常正确的，且当前 workflow 已完成 pytest 和训练 smoke test。

但以下关键路径尚未真正验证：

```text
2进程或8进程 DDP
分布式 evaluator
HCCL dtype
百万级索引内存
多 worker + augmentation 精确恢复
真实 NPU BF16
真实模型 checkpoint 加载覆盖率
```

精确恢复测试目前只覆盖：

```text
CPU
synthetic dataset
num_workers=0
无真实数据增强
单进程
4 steps
```

所以“恢复位置和 sampler 序列正确”已经得到证明，但“八卡 NPU bitwise 精确恢复”目前还不能作出这一结论。

---

# 六、推荐修改顺序

## P0：八卡运行前必须完成

1. 将评估 AllReduce 从 `float64` 改成 `int64 + float32`。
2. 新建独立 production 配置，禁止 demo model 和空 checkpoint。
3. 启动时强制校验非 `cls` 权重加载完整。
4. 八卡运行 100 step，并确保至少触发一次 quick 和一次 full test。

## P1：避免完整测试 OOM 和指标错误

1. full test 不再 `gather_object` 全部 Python 分数。
2. 错误样本和 near-threshold 改为 batch 流式写入。
3. `p>=0.999` 只做计数，不保存所有样本。
4. 修正视频分组键和视频指标定义。
5. final full test 参与 best checkpoint 选择。
6. full test step 跳过重复 quick test。

## P2：优化训练吞吐

1. 构建真正的视频级 Parquet，避免每 rank `to_pylist()` 百万行。
2. 数据复制到本地 NVMe。
3. 增加 tar shard 或 packed uint8 backend。
4. 减少重复完整模型 checkpoint。
5. 用 NPU Profiler/Event 代替当前异步 `perf_counter` 分段计时。
6. quick test 只保存指标和少量错例，不生成完整报告。

## P3：提高精度可信度

1. 修正 RandomErasing。
2. 增加自然分布校准集或 logit bias/temperature 校准。
3. 增加 ECE、Brier Score 和置信度分布。
4. 检测 train/test 视频和内容泄漏。
5. 对 cls BatchNorm 冻结策略做消融。
6. 为 bias 和 BatchNorm 参数设置 `weight_decay=0`，不要对所有 cls 参数统一 AdamW 衰减。

---

# 最终判断

这次改动已经把项目从“CUDA 原型骨架”推进到了：

> **具备生产化结构，但尚未通过真实八卡 NPU 评估链路验证的候选版本。**

最值得肯定的是，训练数据的懒采样、分布式评估框架、严格审计、报告模块、CI 和恢复状态都已经补齐。当前真正阻止它进入长跑训练的不是整体设计，而是几个局部但关键的问题：

```text
float64 HCCL
production配置仍用demo模型
完整评估中心化汇总
视频指标定义错误
uint8 RandomErasing
重复checkpoint和PNG I/O
```

先修复前两项，再进行一次“8 卡、100 step、触发一次 full test”的验证；在这个验证成功前，不建议启动正式的 10 epoch 长跑。

[1]: https://www.hiascend.com/document/detail/en/canncommercial/850/API/hcclapiref/hcclcpp_07_0021.html?utm_source=chatgpt.com "HcclAllReduce-Collective Communication-Communication Operators-HCCL API-HCCL API-API-CANN Commercial Edition8.5.0开发文档-昇腾社区"
[2]: https://docs.pytorch.org/vision/main/_modules/torchvision/transforms/v2/_augment.html "https://docs.pytorch.org/vision/main/_modules/torchvision/transforms/v2/_augment.html"
