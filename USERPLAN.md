# 总体评价

这版仓库的**技术方向正确，模块边界也比较清晰**，尤其是双帧合法配对、分层采样、`cls` 冻结、固定 `0.99` 阈值损失和双 checkpoint 的设计，已经形成了不错的 CUDA 训练框架骨架。

但当前 `main` 分支还不能视为“可直接在 8 卡 A3 上训练”的版本。我的评估是：

| 维度                |         评价 |
| ----------------- | ---------: |
| 总体架构设计            | **7.5/10** |
| CUDA 最小闭环完整度      |   **5/10** |
| 百万图片规模适应性         |   **3/10** |
| 8 卡 Ascend A3 就绪度 |   **2/10** |
| 指标与实验可信度          |   **4/10** |

最先需要解决的不是继续调整损失函数，而是几个会阻断运行、放大内存或使测试结果不可信的问题。

---

# 一、目前实现得比较好的部分

## 1. 双帧配对逻辑是正确的

代码按照 `game + label + video_id` 分组，再检查 `frame_id + delta` 是否真实存在，因此不会跨视频、跨类别配对，也不会假设帧号一定连续。这个实现符合你的数据定义。

训练和测试的 delta 设计也已落地：

* 训练：`1/2/3 = 0.15/0.70/0.15`
* 测试：固定 `delta=2`

配置与原始需求一致。

## 2. 双帧一致增强处理正确

两帧先堆叠为 `[2,C,H,W]`，再统一经过 `RandomAffine`、`ColorJitter` 和 `RandomErasing`。这种实现可以避免两帧被施加不同的几何变化，整体思路正确。

尤其是把空间增强应用于整个双帧张量，而不是分别调用两次随机变换，这是必要的。

## 3. `cls` 冻结逻辑基本符合需求

代码使用大小写敏感的：

```python
parameter.requires_grad = name_contains in name
```

默认只让名称包含小写 `cls` 的参数参与训练，并在没有匹配参数时立即报错。

模型模式处理也比较稳妥：

* 整体先 `eval()`
* 名称包含 `cls` 的模块切换回 `train()`
* 所有 BatchNorm 统计量保持冻结

这可以避免冻结主干的 BatchNorm running statistics 继续变化。

## 4. 固定 0.99 阈值损失的数学实现正确

代码正确地将概率阈值转换为：

[
\log \frac{0.99}{1-0.99}=\log 99
]

并围绕这个 logit margin 构造正负样本辅助损失。

同时保留交叉熵，并让阈值损失经过 warmup 和线性增权，而不是从第一个 step 就施加强约束，这个设计合理。

评估使用严格的 `margin > cutoff`，也与你定义的“第二通道概率大于 0.99”一致。

## 5. 双 checkpoint 结构已经实现

代码同时保存：

```text
model_<tag>.pth
checkpoint_<tag>.pth
```

其中完整 checkpoint 包含模型、优化器、scheduler、scaler、epoch、global step、随机状态和配置，并使用临时文件加原子替换。

这与需求方向一致。

---

# 二、P0：必须立即修复的阻断问题

## 1. 当前训练入口很可能无法启动

`trainer.py` 中明确存在：

```python
from game_cls.reports.error_writer import write_evaluation_report
```

但我分别检查了：

```text
main
962ea3a5cba4f9b4ded56ad80a17132ef62ebe67
```

在这两个 ref 下，GitHub 都对以下文件返回了 `404 Not Found`：

```text
src/game_cls/reports/error_writer.py
src/game_cls/reports/__init__.py
```

因此执行：

```bash
python tools/train.py ...
```

时，`tools/train.py` 导入 `game_cls.engine.trainer` 后，大概率直接出现：

```text
ModuleNotFoundError: No module named 'game_cls.reports'
```

训练入口确实会立即导入 trainer。

### 修复要求

至少补齐：

```text
src/game_cls/reports/__init__.py
src/game_cls/reports/error_writer.py
```

并增加一个真正导入训练主链路的测试：

```python
def test_training_entrypoint_imports():
    from game_cls.engine.trainer import run_training
    assert callable(run_training)
```

目前测试主要覆盖独立组件，没有覆盖 `run_training()` 的完整导入和执行，因此这个遗漏没有被测试发现。

仓库也没有任何 GitHub Actions workflow 运行记录。

---

## 2. 八卡 HCCL 初始化顺序存在高风险

当前主流程先执行：

```python
rank, world_size, local_rank = distributed_context(...)
```

然后才执行：

```python
device = initialize_device(...)
```

但是 `distributed_context()` 内已经调用：

```python
dist.init_process_group(backend="hccl")
```

而 `torch_npu` 是在之后的 `initialize_device()` 中才导入。

这意味着 HCCL 初始化发生时，`torch_npu` 可能还没有完成后端注册，可能出现：

```text
Unknown c10d backend type HCCL
```

或者分布式初始化异常。

Ascend 当前官方 DDP 示例会在初始化 HCCL 前先导入 `torch_npu`；官方完整示例也明确在模块顶部导入该扩展。([hiascend.com][1])

### 推荐结构

不要把设备初始化和分布式初始化拆成现在的顺序，改成统一运行时初始化：

```python
def initialize_runtime(config):
    import os
    import torch
    import torch.distributed as dist

    distributed = config.get("distributed", {}).get("enabled", False)
    accelerator = config["device"]["accelerator"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if accelerator == "npu":
        import torch_npu
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif accelerator == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group(
            backend=config["distributed"]["backend"],
            init_method="env://",
        )

    return rank, world_size, local_rank, device
```

---

# 三、P1：会明显影响训练正确性和性能的问题

## 1. “快速测试”实际上每次都执行完整测试

配置中定义了：

```yaml
quick_test_every_steps: 50
quick_test_pairs_per_video: 16
full_test_every_steps: 100
```

但训练代码只读取：

```python
quick_test_every_steps
```

然后直接对完整的 `test_loader` 调用 `evaluate()`。

当前以下配置实际上没有被使用：

```text
quick_test_pairs_per_video
full_test_every_steps
```

在百万帧数据下，这会造成：

```text
到达 quick_test 间隔
→ rank 0 完整遍历 test
→ 其他 7 个 rank 全部在 barrier 等待
→ 完整生成错误记录
→ 可能再保存 checkpoint
```

这很容易制造你之前提到的 NPU 利用率锯齿。

### 应拆成两个 DataLoader

```text
quick_test_loader
    每个 game-label-video 均匀抽取固定数量 delta=2 pair

full_test_loader
    枚举全部合法 delta=2 pair
```

训练逻辑分别使用：

```python
if step % quick_test_every_steps == 0:
    evaluate(quick_test_loader)

if step % full_test_every_steps == 0:
    evaluate(full_test_loader)
```

---

## 2. 八卡评估没有并行

当前只有 rank 0 执行测试，其他 rank 在 barrier 中等待。

这虽然不会重复计算指标，但在 8 卡环境中效率很低。正确方案应为：

```text
每个 rank 处理 test 的不重复分片
→ 本地累计 TP/FP/FN/TN 和 CE
→ all_reduce 数值指标
→ 每个 rank 单独写错误样本分片
→ rank 0 合并报告
```

特别要避免使用会补齐重复样本的普通 `DistributedSampler`；测试 sampler 应不补齐、不重复。

---

## 3. 当前 test 已经被当作 validation 使用

代码在周期性 test 后比较：

```python
if metrics["f1"] > best_f1:
    save best_f1 checkpoint
```

因此这个 test 实际上参与了：

* checkpoint 选择；
* 模型版本选择；
* 训练过程判断。

这在统计上已经是 validation/dev set，而不是独立 final test。

如果你坚持只保留 train 和 test，也可以继续这样做，但最终报告必须准确描述为：

```text
best observed dev-test F1
```

不能把反复查看并选择过的最高值当作完全独立的最终泛化指标。至少应同时报告：

```text
last checkpoint F1
best checkpoint F1
模型选择过程中总共评估 test 的次数
```

---

## 4. 百万级数据下会生成数百万 Python 对象

当前实现会：

```python
train_frames = read_frame_parquet(...)
train_pairs = enumerate_pairs(train_frames, [1, 2, 3])
```

`enumerate_pairs()` 会针对三个 delta 分别构造完整 `PairSample` 列表。

随后：

* `PairDataset` 再复制一次 pair 列表；

* sampler 再复制一次 pair 列表；

* sampler 再为所有 pair 建立多层索引和 Python 整数列表。

对于约 100 万帧：

```text
约 300 万个 train pair
× 8 个独立训练进程
```

每个 rank 都读取完整 Parquet、创建完整 pair 对象和完整 test pair。启动时间、CPU 内存和 Python GC 压力都会很高，可能直接导致内存不足。

### 应改成视频级懒采样

训练阶段不要物化全部 pair，只保留紧凑的视频索引：

```python
VideoEntry(
    game_id,
    label,
    video_id,
    frame_ids: np.ndarray,
    frame_paths: np.ndarray,
    valid_starts_delta1: np.ndarray,
    valid_starts_delta2: np.ndarray,
    valid_starts_delta3: np.ndarray,
)
```

sampler 直接返回：

```text
video_index
delta
start_position
```

Dataset 再解析出两张图片路径。

这样内存规模从“所有 pair 数量”降为“所有 frame 数量 + 视频索引”。

---

## 5. 需求中的分游戏、分视频指标尚未实现

当前 `BinaryMetrics` 只提供一组全局指标。

`evaluate()` 也只调用一次全局 `confusion_from_margins()`，没有计算：

```text
每游戏 F1
macro_game_f1
worst_game_f1
每视频 F1
macro_video_f1
```

这会导致大游戏继续主导最终指标，无法验证“每个游戏都具有较好识别效果”的核心需求。

此外，目前错误记录缺少：

```text
logit0
logit1
阈值附近样本
按游戏汇总
按视频汇总
HTML双帧展示
```

并且错误报告模块本身还没有提交。

---

## 6. 完整 checkpoint 不能精确续训

虽然 checkpoint 保存了 optimizer、scheduler、scaler 和 CPU RNG，但只保存了：

```text
epoch
global_step
sampler_epoch
```

没有保存：

```text
当前 epoch 内已经消费的 batch 数量
CUDA RNG
NPU RNG
各 rank 独立 RNG
DataLoader worker RNG
```

恢复后代码重新设置 sampler epoch，然后从该 epoch 的第一个 batch 开始迭代。

例如在 epoch 3 的第 400 step 保存，恢复时会重新消费 epoch 3 的前 400 个 batch。因此它是“可以继续训练”，但不是“精确恢复全部训练状态”。

建议保存：

```text
epoch
step_in_epoch
global_step
sampler state
torch CPU RNG
CUDA/NPU RNG
每个 rank 的 RNG state
```

恢复时跳过已经消费的 batch，或者让 sampler 接受 `start_step`。

---

# 四、P2：需要在性能阶段优化的问题

## 1. 游戏视频数量统计存在一个小错误

当前游戏权重使用：

```python
{
    video
    for by_video in self.groups[game].values()
    for video in by_video
}
```

如果类别 0 和类别 1 中都存在 `video_id="01"`，它们会被当成同一个视频，只计数一次。

应改成：

```python
{
    (label, video)
    for label, by_video in self.groups[game].items()
    for video in by_video
}
```

否则游戏采样权重并不是真正的 game-label-video 数量。

## 2. 实际 delta 分布不一定是 15/70/15

当前流程是：

```text
先选视频
→ 再在该视频可用的 delta 中抽样
```

如果部分视频缺少合法的 `delta=2` pair，最终全局 delta 分布会偏离 70%。

更严格的实现应先选择 delta，再从支持该 delta 的视频中选择视频，或者至少每个 epoch 输出实际 delta 分布。

## 3. 数据增强后以 FP32 传输到 NPU

增强链中明确执行：

```python
v2.ToDtype(dtype=torch.float32, scale=True)
```

因此训练 batch 从 DataLoader 输出时已经是 FP32；训练循环发现不是 uint8 后会直接传输。

相对于 uint8，H2D 数据量放大四倍。对于 `[B,2,3,448,208]` 的输入，这会显著增加内存带宽、共享内存和 pin memory 压力。

后续高性能实现应考虑：

```text
CPU执行PNG解码和必要空间增强
→ 尽量保持uint8
→ 一次H2D
→ NPU上转换BF16/FP16并除以255
→ 适合设备执行的增强放在设备端
```

## 4. 数据审计只报告，不阻止错误数据进入训练

索引代码会统计不符合 `208×448×3` 的样本，但仍把这些帧写入 Parquet。

训练主流程也不会读取 `audit.json` 或检查尺寸。遇到异常图片时，可能直到 DataLoader collate 才因张量尺寸不同而报错。

建议正式训练默认：

```yaml
data:
  strict_audit: true
```

出现以下任一问题就拒绝启动：

```text
尺寸错误
通道错误
损坏PNG
非法文件名
某游戏缺少一个类别
test中没有合法delta=2 pair
```

## 5. Python 包依赖可能破坏 Ascend 环境

`pyproject.toml` 把：

```text
torch>=2.2
torchvision>=0.17
```

设为普通安装依赖。

在 910B2 环境执行：

```bash
pip install -e .
```

可能让 pip 尝试安装或替换普通 PyTorch wheel，从而破坏已经匹配好的 `torch + torch_npu + CANN` 环境。

更安全的是：

```toml
dependencies = [
  "numpy>=1.26",
  "Pillow>=10.0",
  "PyYAML>=6.0",
  "pyarrow>=15.0",
]

[project.optional-dependencies]
cuda = ["torch", "torchvision"]
dev = ["pytest>=8.0"]
```

Ascend 环境安装时使用预先匹配好的 PyTorch/TorchNPU，不由该项目自动修改。

---

# 五、建议的修改顺序

## 第一批：让仓库真正可运行

1. 补齐 `reports` 包和 `write_evaluation_report()`。
2. 增加 `run_training()` 导入及 2-step smoke test。
3. 调整 NPU/HCCL 初始化，确保先导入 `torch_npu`。
4. 添加 CPU GitHub Actions，至少执行：

   ```bash
   python -m pip install -e .[dev]
   python -m pytest
   python tools/train.py --config configs/cuda_debug.yaml train.max_steps=2
   ```

## 第二批：修复评估契约

1. 真正实现 quick test 子集。
2. 真正实现 `full_test_every_steps`。
3. 增加按游戏和按视频指标。
4. 增加 near-threshold、FP、FN 和 HTML 报告。
5. 八卡分布式评估，不让 7 张卡等待 rank 0。

## 第三批：解决百万数据规模问题

1. 删除完整 `train_pairs` 物化。
2. 改为视频级紧凑索引和懒采样。
3. test pair 使用 NumPy/Arrow 紧凑数组。
4. 每个 rank 不重复创建数百万 Python 对象。
5. 再加入本地 NVMe、PNG shard 或 packed uint8 backend。

## 第四批：完善训练状态和 A3 性能

1. 支持精确的 `step_in_epoch` 恢复。
2. 保存 CUDA/NPU 和各 rank RNG。
3. 启用 BF16 AMP 并验证算子兼容性。
4. 增加 DataLoader、H2D、forward、backward、HCCL 分段计时。
5. 最后再考虑 `torch.compile` 或图模式。

---

# 最终判断

这版不是一个“思路错误、需要推倒重写”的方案。相反，它已经具备了一个良好的训练框架核心：

```text
合法双帧配对
+ 多层均衡采样
+ 一致增强
+ cls冻结
+ 0.99阈值目标
+ 双checkpoint
```

但当前更准确的定位是：

> **结构合理的 CUDA 原型代码，而不是已经完成的百万数据、8 卡 Ascend 生产训练框架。**

最关键的下一步是先修复缺失的报告模块和 HCCL 初始化，再重构评估与 pair 索引。否则直接放到 8 卡服务器上，最可能遇到的不是模型精度问题，而是导入失败、HCCL 初始化失败、启动内存过大，以及周期性完整测试导致的严重锯齿停顿。

[1]: https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/trainingmigrguide/PT_LMTMOG_0022.html "https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/trainingmigrguide/PT_LMTMOG_0022.html"
