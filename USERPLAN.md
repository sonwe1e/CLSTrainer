# 结论

我基于当前分支最新提交 `9e1b1787` 重新检查了完整训练链路。该提交的 CPU CI 已通过，包括 pytest、分布式 Gloo 评估和训练 smoke test。

**上次指出的确定性阻断问题已经基本全部解决。** 当前版本已经从“生产化候选框架”推进到“可以接入真实模型进行 NPU 单卡和八卡短跑验证”的阶段，但还不能仅凭 CPU CI 认定八卡 A3 长跑已经可靠。

当前综合判断：

| 维度           |         评价 |
| ------------ | ---------: |
| 代码架构         | **8.5/10** |
| CUDA/CPU 完整性 | **8.5/10** |
| 百万帧数据适应性     |   **7/10** |
| 八卡 A3 代码就绪度  |   **7/10** |
| 八卡 A3 实测就绪度  | **4.5/10** |
| 指标与精度可信度     |   **7/10** |

---

# 一、上次问题的解决情况

| 上次问题                             | 当前状态                                              | 结论                |
| -------------------------------- | ------------------------------------------------- | ----------------- |
| HCCL 对 `float64` AllReduce 不支持   | 混淆矩阵改为 `int64`，CE/Brier 改为 `float32`              | **已解决**           |
| NPU 配置继承 demo 模型                 | 新增独立 `npu_production.yaml`，占位工厂和空 checkpoint 会被拒绝 | **已解决**           |
| full test 汇总百万 Python 分数到 rank 0 | full 默认使用 4096-bin 直方图计算 AUC                      | **已解决**           |
| 错例全部存入内存                         | 改为 batch 级 Parquet 流式写入                           | **已解决**           |
| `p>=0.999` 全部保存到 near-threshold  | 只保存 `0.98～0.995`，高置信度仅计数                          | **已解决**           |
| 视频指标把不同标签的同名视频合并                 | 键改成 `(game,label,video_id)`                       | **已解决**           |
| 负类视频被纳入 macro-video F1           | 删除 macro-video F1，按标签报告 recall 或 specificity      | **已解决**           |
| uint8 RandomErasing 语义不正确        | 实现 uint8 专用擦除，生产配置默认填 0                           | **已解决**           |
| quick/full 同一步重复执行               | full 优先，同一步跳过 quick                               | **已解决**           |
| final full 不参与最佳模型选择             | final full 也会比较并更新 best                           | **已解决**           |
| checkpoint 重复序列化完整主干             | 周期 checkpoint 可只保存可训练状态，best 通过硬链接或复制形成别名         | **主要解决**          |
| 每个 rank 从百万帧表创建百万个 `FrameRecord` | 增加视频级 Parquet，按视频行读取                              | **主要解决**          |
| 主干权重加载过于宽松                       | 正式训练检查非 `cls` 权重覆盖率和形状                            | **已解决但策略过严**      |
| train/test 泄漏不检查                 | 增加视频键和 SHA-256 重复检测                               | **已增加，但视频键规则需修正** |
| cls 与 backbone BN 无法分别控制         | 已拆成两个配置项                                          | **已解决**           |
| AdamW 对 bias/BN 施加衰减             | 一维参数和 bias 放入无衰减组                                 | **已解决**           |

分布式评估现在确实使用 `int64 + float32` 归约。

独立生产配置也已建立，真实模型工厂、checkpoint 和主干加载覆盖率都有启动门禁。

---

# 二、现在最需要处理的问题

## 1. 测试阶段仍然强制使用 FP32，与 BF16 训练和部署路径不一致

生产配置启用了：

```yaml
amp: true
amp_dtype: bfloat16
```

但 evaluator 会把 uint8 输入直接转换成 FP32，并且模型前向不在 autocast 中：

```python
images = images.to(torch.float32).div_(255.0)
logits = model(images[:, 0], images[:, 1])
```

这会产生两类影响：

* **效率影响**：完整测试走 FP32，NPU 测试吞吐可能显著低于训练吞吐。
* **精度影响**：模型选择依据 FP32 下的 `F1@0.99`，但实际部署若使用 BF16、FP16 或其他低精度，阈值附近样本可能发生翻转。

建议在配置中增加：

```yaml
evaluation:
  amp: true
  amp_dtype: bfloat16
```

评估模型前向与部署精度保持一致。可以另外低频运行一次 FP32 诊断测试，但不应让 FP32 指标决定最终部署模型。

这是当前对训练精度和测试效率影响最大的问题之一。

---

## 2. CPU/Gloo CI 仍不能证明 A3 上的 evaluator 算子链可运行

当前分布式测试使用：

```python
dist.init_process_group("gloo")
device = cpu
```

而实际 evaluator 在 NPU 上使用了：

```text
torch.bincount
scatter_add_
int64 HCCL AllReduce
float32 HCCL AllReduce
多个直方图 AllReduce
```

当前代码选择的 dtype 已合理，但还没有真实验证：

* TorchNPU 2.10 对这些输入 dtype 的支持；
* 4096-bin `bincount` 和多个 AllReduce 是否产生异常；
* BF16 模型前向与 FP32统计组合是否正常；
* 八卡 gather-object、报告合并和 checkpoint 是否会死锁。

因此现在不存在已知的确定性 HCCL 阻断，但仍存在**未经过实际设备验证的兼容性风险**。

---

## 3. 最佳模型仍然只按全局 F1 选择

代码已经计算：

```text
global_f1_tau099
macro_game_f1_tau099
worst_game_f1_tau099
```

但保存最佳模型时仍然只比较：

```python
metrics["f1"]
```

这与“每个游戏都有较好识别效果”的目标并不完全一致。一个大游戏的样本可能显著提高 global F1，同时某个小游戏性能下降，但该模型仍会被选为最佳。

建议支持可配置的模型选择分数，例如：

```yaml
evaluation:
  selection_metric: composite
  selection_weights:
    global_f1: 0.40
    macro_game_f1: 0.40
    worst_game_f1: 0.20
```

或者第一版直接使用：

```yaml
selection_metric: macro_game_f1_tau099
```

至少同时设置 `worst_game_f1` 下限，防止某个游戏出现灾难性退化。

---

## 4. 视频泄漏检查可能错误拒绝合法数据

当前将以下键在 train/test 中重复视为泄漏：

```python
(game, label, video_id)
```

但你的 `video_id` 只有两位。假如 train 和 test 各自都从 `01` 开始编号，即使它们是完全不同的原始视频，也会被判定为泄漏。正式配置又要求严格审计，因此可能直接阻止训练。

需要先明确视频 ID 是否在整个项目中全局唯一：

* 若全局唯一，现有规则正确。
* 若每个 split 或每次采集重新编号，应删除该强制条件，或者引入真正的 `recording_uid`。
* SHA-256 重复检查可以继续保留。

此外，SHA-256 只能检测字节完全相同的图片，不能识别重新编码、轻微裁剪或颜色变化后的近重复帧。

---

# 三、仍会影响训练效率的问题

## 1. evaluator 每个 batch 发生多次 NPU 同步

当前每个测试 batch 都执行：

```python
cross_entropy(...).item()
brier.sum().item()
tp.sum().item()
fp.sum().item()
fn.sum().item()
tn.sum().item()
```

每次 `.item()` 都可能迫使主机等待 NPU 计算完成。随后又分别执行：

```python
logits.cpu()
margins.cpu()
labels.cpu()
probabilities.cpu()
```

因此虽然 full test 已不再把所有分数保存在内存里，但设备流水仍然被频繁同步。

建议：

1. CE、Brier 和四个计数都保留为设备 tensor。
2. 整个评估结束后再统一 AllReduce 和 `.cpu()`。
3. 每个 batch 只拷贝一次紧凑的统计 tensor。
4. 只有错误或临界样本才复制完整 logits。

这会明显提升完整 test 的 NPU 利用率。

---

## 2. packed backend 仍可能占用大量 CPU 内存

新的 packed backend 确实消除了 PNG 解码，但初始化时仍然：

```python
rows = pq.read_table(...).to_pylist()
self.locations = {
    path: (shard_path, offset, length)
    for row in rows
}
```

一百万帧就意味着：

* 每个 rank 一个约百万项的 Python 字典；
* 每项含原始绝对路径字符串、shard 路径、offset 和 length；
* train 和 test 分别一份；
* 视频索引中又保存一份每帧路径字符串；
* 八个 rank 各自重复。

这可能重新把“pair 对象内存问题”变成“路径字典内存问题”。

更合适的结构是：

```text
VideoEntry:
  frame_ids
  packed_shard_ids
  packed_offsets
```

Dataset 直接通过整数索引读取，不再以完整路径作为 packed backend 的主键。

---

## 3. packed backend 会长期打开所有访问过的 shard

每个 shard 第一次访问后都会保存在：

```python
self._memory_maps
```

直到 Dataset 销毁。

按默认每 shard 4096 张图片，一个百万帧数据集大约有 245 个 shard。随机均衡采样经过足够长时间后，每个 worker 可能打开大量 shard。

建议实现 8～32 个 shard 的 LRU memmap 缓存，并关闭被淘汰 shard，避免文件描述符和虚拟地址空间持续增长。

packed index 中的 shard 路径还是绝对路径，整体移动 packed 目录后索引会失效，也建议改为相对于 index 文件的路径。

---

## 4. 正式配置默认仍使用 PNG

生产配置当前为：

```yaml
backend: png
```

因此这次新增的 packed backend **不会自动提升正式训练吞吐**。除非手动完成 train/test 打包并修改配置，热路径仍是：

```text
Image.open
→ PNG 解码
→ RGB 转换
→ 两次文件读取
```

若目前约 200 samples/s 的瓶颈来自数据读取和 PNG 解码，当前默认配置不会改变这一点。

建议先在本地 NVMe 上分别测试：

```text
PNG backend
packed_uint8 backend
```

保持模型、batch 和 worker 数完全一致，比较纯训练吞吐。

---

## 5. 视频索引仍保存每一帧的完整路径

视频级 Parquet 已避免百万个 `FrameRecord`，这是正确优化。但读取后仍保留：

```python
frame_paths: tuple[str, ...]
```

所以每个 rank 仍有约一百万个 Python 字符串。当前日志中的内存估算只统计字符串 UTF-8 内容和 NumPy 数组，没有统计 Python 对象、tuple 和 dict 的额外开销。

更紧凑的方式是存储：

```text
video_directory
frame_ids
```

运行时按命名规则生成文件路径；packed 模式则完全使用整数 offset。

---

## 6. checkpoint 的 `model_last.pth` 可能不是最新 step

生产配置：

```yaml
save_last_every_steps: 1000
periodic_state_mode: trainable_only
full_model_every_steps: 5000
```

因此：

* `checkpoint_last.pth` 每 1000 step 更新；
* `model_last.pth` 可能只在第 5000、10000 step 或最佳模型时更新。

如果训练在第 9000 step 中断：

```text
checkpoint_last.pth → step 9000
model_last.pth      → 可能还是 step 5000
```

最终训练正常结束时会写最新完整模型，因此最终产物没有问题；但训练中途查看或部署 `model_last.pth` 可能拿到旧权重。

建议改名为：

```text
model_last_full.pth
checkpoint_last.pth
```

并在 metadata 中明确各自的 global step。也可以每 1000 step 额外保存很小的：

```text
cls_last.pth
```

---

# 四、仍会影响训练精度和指标可信度的问题

## 1. 平衡采样导致的概率校准问题仍然存在

这次已经增加：

```text
Brier Score
20-bin ECE
置信度直方图
threshold_is_business_score
```

这解决了“无法观察校准状态”的问题，但没有解决校准本身。

由于训练采用：

```text
类别 0/1 约 50/50
+ 阈值间隔损失
```

而真实部署的类别 1 先验可能远低于 50%，Softmax 的 `0.99` 仍不能解释为真实概率 99%。

现阶段将其定义为“业务分数阈值”是合理的，但若希望该分数在不同游戏和后续新增数据上稳定，仍建议预留一个自然分布 calibration 集，拟合一个轻量的：

[
d_{\text{calibrated}}=d/T+b
]

---

## 2. `by_game_label` 中的 F1 仍不适合作为解释指标

视频行已经增加了正确的主指标：

* 正类视频：recall、FN rate；
* 负类视频：specificity、FP rate。

但 `by_game_label` 仍调用普通二分类指标。负类分组不含任何正样本，所以该分组的 F1 固定没有实际解释价值。

建议：

* 正类 game-label 行只突出 recall/FN rate；
* 负类 game-label 行只突出 specificity/FP rate；
* 单类分组中的 F1、ROC-AUC、PR-AUC 留空，而不是显示 0。

这不会影响模型训练，但会影响人工判断和实验结论。

---

## 3. 数据审计缺少 game × label × delta 覆盖检查

当前审计只检查：

* 每个游戏是否同时有类别 0 和 1；
* 全局是否有合法 delta pair；
* test 是否有 delta=2。

它没有检查：

```text
某个游戏的类别1是否有delta=2 pair
某个游戏的类别0是否只有delta=1 pair
```

sampler 会自动只从存在合法 pair 的组合中采样，因此训练不会报错，但实际采样分布可能偏离业务目标。

建议审计输出：

```text
game × label × delta 的合法pair数量
```

并设置最低要求，尤其确保每个游戏、每个标签都有足够的 `delta=2` 样本。

当前训练只记录全局 delta 分布，还应记录：

```text
game × label × delta 的实际采样分布
```

---

## 4. RandomAffine 应明确插值方法

当前没有显式指定：

```python
interpolation
fill
```

为了避免不同 torchvision 版本的默认行为差异，并减少 RGB 游戏画面旋转、缩放后的锯齿，建议明确使用双线性插值和固定边界填充值。

例如：

```python
interpolation=InterpolationMode.BILINEAR
fill=0
```

同时应做一次无增强、轻增强、当前增强的消融实验。你的训练视频数量相对有限，增强过强可能比增强不足更容易损害精度。

---

## 5. 主干加载策略实际上是 100% key 严格，而不是 99%

配置写的是：

```yaml
minimum_non_cls_coverage: 0.99
```

但代码只要有任意非 `cls` missing 或 shape mismatch 就直接报错。

因此当前真实语义是：

```text
非cls key必须100%加载
```

这对于主干参数是安全的，但可能因为无关 buffer，例如某些 BatchNorm 追踪计数不同，而错误拒绝有效 checkpoint。

建议二选一：

* 要求真正的 100%，删除 `minimum_non_cls_coverage`，逻辑更明确；
* 允许显式白名单 buffer 缺失，并按参数元素数或字节数计算覆盖率，而不是按 key 数量。

---

## 6. 解冻 cls BatchNorm 时，八卡统计量可能不一致

虽然现在可以分别配置 cls BN，但 DDP 仍固定：

```python
broadcast_buffers=False
```

生产默认 `freeze_cls_batchnorm_stats: true`，因此默认路径安全。

但未来若设置：

```yaml
freeze_cls_batchnorm_stats: false
```

各 rank 的 running mean/variance 会分别更新且不再同步。此时需要：

* 禁止该组合；
* 或使用 SyncBatchNorm；
* 或启用适当的 buffer 同步。

---

# 五、新 checkpoint 方案的一个完整性问题

trainable-only checkpoint 恢复时使用：

```python
load_state_dict(..., strict=False)
```

随后只检查 `unexpected_keys`，没有检查 checkpoint 是否缺少某个预期 `cls` 参数。

如果 checkpoint 文件损坏或旧版本缺少某个 `cls` tensor，该 tensor会继续保留基础 checkpoint 中的值，恢复过程可能静默成功。

建议保存时记录：

```text
expected_trainable_state_keys
```

恢复时要求：

```text
checkpoint keys == 当前期望的可训练参数和相关buffer keys
```

同时检查 missing 和 unexpected。

---

# 六、建议的执行优先级

## 八卡正式长跑前必须完成

1. 让 evaluator 支持与部署一致的 BF16/FP16 autocast。
2. 在 910B2 单卡上跑 100 step，并触发一次 quick 和 full test。
3. 在八卡上跑 100～500 step，再触发一次 full test。
4. 验证 NPU 上 `bincount`、`scatter_add_`、直方图 AllReduce 和报告合并。
5. 将最佳模型选择指标改为 macro-game 或组合指标。
6. 确认 train/test 的两位 video ID 是否全局唯一。

## 百万帧高吞吐训练前完成

1. 用实际数据对比 PNG 和 packed backend。
2. packed 索引改为整数索引，移除百万项路径字典。
3. memmap 增加 LRU。
4. evaluator 消除 batch 内多次 `.item()` 同步。
5. 使用本地 NVMe，而不是通过约 300 MB/s 的远程随机小文件链路长跑。
6. 记录各 rank 的 CPU RSS、worker RSS、NPU 利用率和 step P95。

## 精度基线阶段完成

1. 增加 game × label × delta 数据审计。
2. 使用可配置的最佳模型选择分数。
3. 比较 FP32 与部署精度下的 `F1@0.99` 差异。
4. 做增强强度和 cls BatchNorm 策略的消融。
5. 保留自然分布的 calibration 数据，至少用于验证置信度稳定性。

---

# 最终判断

当前分支已经解决了上一轮几乎全部明确代码缺陷，尤其是：

```text
HCCL dtype
生产模型门禁
full评估内存
流式错例
视频指标
RandomErasing
final-best
checkpoint重复I/O
视频级索引
packed backend
```

最新版本已经适合进入**真实 NPU 单卡和八卡 smoke test**，不再需要继续大规模重构后才上机。

现在剩余问题中，最值得优先解决的是：

> **评估精度路径与部署不一致、最佳模型仍按 global F1 选择、packed backend 的百万级 Python 字典，以及缺少真实 HCCL/NPU 验证。**

这四项处理完后，框架才适合进行正式的百万帧八卡长时间训练。
