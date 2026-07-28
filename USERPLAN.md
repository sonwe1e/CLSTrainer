# 双帧多游戏二分类训练框架技术方案 v1.0

## 核心结论

现有需求已经澄清，可以直接进入实现。框架采用以下固定定义：

* 开发环境：单卡 CUDA。
* 正式训练：单机 8 卡 Ascend 910B2，HCCL + DDP。
* 输入：两张 RGB 图像，按“宽 × 高”解释为 `208 × 448`，因此张量形状为 `[B, 3, 448, 208]`。
* 部署输入：固定 `delta=2`，即第 1 帧和第 3 帧。
* 训练增强：`delta=2` 占多数，`delta=1/3` 作为时间跨度增强，初始比例设为 `15%/70%/15%`。
* 模型输出：`[B, 2]`。
* 训练参数：名称包含小写 `cls` 的参数可训练，其他参数全部冻结。
* 判定规则：第二通道 Softmax 概率严格大于 `0.99` 判为类别 1，否则判为类别 0。
* 数据平衡：游戏、类别、视频三级分层采样。
* 评估：训练期间周期性执行 test，保存固定 `0.99` 阈值指标和错分类样本。
* 模型保存：每个 checkpoint 同时保存纯模型参数文件和完整训练状态文件。

你提到的 `random find`，本方案按常见的 `RandomAffine` 理解。Torchvision 原生提供 `RandomAffine`、`ColorJitter` 和 `RandomErasing`，并支持带有额外前导维度的图像或视频张量，因此可以对两帧应用一致的随机参数。([PyTorch Docs][1])

---

# 一、项目目录和模块边界

建议从一开始就把设备差异和业务逻辑分开：

```text
game_cls/
├── configs/
│   ├── cuda_debug.yaml
│   ├── npu_1p.yaml
│   └── npu_8p.yaml
├── tools/
│   ├── build_index.py
│   ├── audit_dataset.py
│   ├── pack_dataset.py
│   ├── train.py
│   ├── test.py
│   └── profile_train.py
├── src/
│   ├── data/
│   │   ├── records.py
│   │   ├── pair_dataset.py
│   │   ├── pair_sampler.py
│   │   ├── augment.py
│   │   ├── png_backend.py
│   │   ├── packed_backend.py
│   │   └── collate.py
│   ├── model/
│   │   ├── builder.py
│   │   ├── checkpoint_loader.py
│   │   ├── freeze_policy.py
│   │   └── model_wrapper.py
│   ├── losses/
│   │   └── threshold_loss.py
│   ├── engine/
│   │   ├── device.py
│   │   ├── distributed.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── metrics/
│   │   ├── binary_metrics.py
│   │   └── grouped_metrics.py
│   └── reports/
│       ├── error_writer.py
│       └── html_report.py
├── scripts/
│   ├── run_cuda_debug.sh
│   ├── run_npu_1p.sh
│   └── run_npu_8p.sh
└── tests/
    ├── test_filename_parser.py
    ├── test_pair_sampler.py
    ├── test_freeze_policy.py
    ├── test_threshold_loss.py
    └── test_checkpoint_resume.py
```

CUDA 和 NPU 的区别只应存在于：

```text
src/engine/device.py
src/engine/distributed.py
AMP上下文
Profiler初始化
启动脚本
```

数据、损失、指标和模型冻结逻辑完全复用。

---

# 二、数据索引方案

## 2.1 原始目录约定

```text
train/
├── game_A/
│   ├── 0/
│   │   ├── 0100001.png
│   │   ├── 0100002.png
│   │   └── 0100003.png
│   └── 1/
├── game_B/
│   ├── 0/
│   └── 1/
└── ...

test/
├── game_A/
│   ├── 0/
│   └── 1/
└── ...
```

文件名解析规则：

```regex
^(?P<video_id>\d{2})(?P<frame_id>\d{5})\.png$
```

例如：

```text
0100001.png
│ │
│ └── frame_id = 00001
└──── video_id = 01
```

## 2.2 建立帧索引

`build_index.py` 扫描一次目录，输出：

```text
indexes/
├── train_frames.parquet
├── train_videos.parquet
├── test_frames.parquet
├── test_videos.parquet
└── audit.json
```

每帧至少记录：

```text
sample_id
split
game
label
video_id
frame_id
path
width
height
channels
file_size
```

每个视频记录：

```text
game
label
video_id
frame_count
min_frame_id
max_frame_id
valid_pair_count_delta1
valid_pair_count_delta2
valid_pair_count_delta3
```

必须根据“目标帧是否真实存在”判断是否合法，不能默认帧号连续。

例如视频中存在：

```text
00001
00002
00004
```

则：

```text
00001 → 00003，delta=2：非法
00002 → 00004，delta=2：合法
```

建议在索引阶段直接建立：

```python
frame_id_to_path: dict[int, str]
valid_starts: dict[int, list[int]]  # key为delta
```

---

# 三、训练样本生成

一个训练样本定义为：

```text
PairSample(
    game,
    label,
    video_id,
    frame0_id,
    frame1_id,
    delta,
    image0_path,
    image1_path
)
```

## 3.1 delta 分布

部署固定使用 `delta=2`，训练初始采用：

```yaml
pair:
  train_delta_prob:
    1: 0.15
    2: 0.70
    3: 0.15
  test_delta: 2
```

这意味着训练的大多数样本与部署一致，同时通过 `delta=1/3` 增加对时间变化速度的鲁棒性。

测试集只枚举：

```text
frame_t + frame_t+2
```

不把 `delta=1/3` 混入核心测试指标。可以额外生成鲁棒性报告，但不参与主指标。

## 3.2 三级平衡采样

每次采样按以下顺序执行：

```text
选择游戏
→ 选择类别
→ 选择视频
→ 选择delta
→ 选择合法起始帧
```

推荐默认配置：

```yaml
sampler:
  game_alpha: 0.25
  class_probability:
    0: 0.5
    1: 0.5
  uniform_video: true
  deduplicate_within_batch: true
```

游戏概率为：

[
P(g)\propto N_g^{0.25}
]

这里的 `N_g` 建议使用游戏的视频数量，而不是帧数量。

这样可以避免：

* 帧多的游戏统治训练；
* 长视频统治短视频；
* 类别数量不均衡导致模型倾向多数类别；
* 极小游戏被完全均匀采样时发生过度重复。

## 3.3 分布式采样方案

不要直接使用普通 `DistributedSampler` 完成全部逻辑，因为它只能对已有索引切片，不能自然表达游戏—类别—视频的三级均衡。

实现一个：

```text
BalancedDistributedPairBatchSampler
```

每个训练 step：

1. 根据固定随机种子生成完整 global batch。
2. global batch 大小为：

```text
local_batch_size × world_size
```

3. 所有 rank 使用相同确定性算法生成同一 global batch。
4. 每个 rank 只取得自己的切片。
5. 不需要每 step 广播样本索引。

例如八卡、每卡 64：

```text
global batch = 512

rank0: [0:64]
rank1: [64:128]
...
rank7: [448:512]
```

PyTorch DDP 只负责同步模型梯度，不会自动切分输入，因此必须由 sampler 或 dataset 负责每个进程的数据划分。([PyTorch Docs][2])

---

# 四、双帧一致的数据增强

## 4.1 基本原则

两帧来自同一个视频，其空间坐标系和光照环境是关联的。因此：

* 几何变换必须对两帧使用完全相同的参数。
* 颜色变换第一版也使用相同参数。
* 不能分别随机旋转两张图。
* 不能让一张水平翻转、另一张不翻转。
* 不能让两帧采用不同的随机裁剪位置。

将输入先堆叠为：

```text
pair: [2, 3, 448, 208]
```

然后把它当作两帧视频执行增强。

Torchvision v2 的 `RandomAffine` 和 `ColorJitter` 支持具有任意前导维度的图像输入，`ColorJitter` 也明确支持图像或视频。([PyTorch Docs][1])

## 4.2 推荐初始增强配置

```yaml
augmentation:
  enabled: true

  random_affine:
    enabled: true
    probability: 0.50
    degrees: 2.0
    translate: [0.02, 0.02]
    scale: [0.98, 1.02]
    shear: [-1.0, 1.0]

  color_jitter:
    enabled: true
    probability: 0.80
    brightness: 0.15
    contrast: 0.15
    saturation: 0.10
    hue: 0.02

  random_erasing:
    enabled: true
    probability: 0.10
    scale: [0.005, 0.03]
    ratio: [0.3, 3.3]
    value: random

  horizontal_flip:
    enabled: false

  vertical_flip:
    enabled: false
```

`RandomErasing` 会随机擦除矩形区域，适合模拟小范围遮挡，但第一版概率和面积都应保持较小。([PyTorch Docs][3])

## 4.3 为什么默认不打开水平翻转

游戏画面可能包含：

* 固定方向的 UI；
* 文字；
* 小地图；
* 左右方向具有业务语义的动作；
* 方向性场景元素。

所以水平翻转不能默认开启。只有确认左右翻转不改变类别含义后，再设置：

```yaml
horizontal_flip:
  enabled: true
  probability: 0.20
```

## 4.4 暂时不使用的增强

第一阶段不使用：

```text
MixUp
CutMix
Label Smoothing
大幅RandomResizedCrop
大角度旋转
强Perspective
垂直翻转
```

原因是你的部署目标要求类别 1 概率稳定超过 `0.99`。软标签和过强的形变可能让模型刻意降低置信度，或者破坏两帧之间的真实运动关系。

---

# 五、模型加载、冻结和训练模式

## 5.1 参数冻结规则

按照你的原始目标，本方案解释为：

> 名称包含小写 `cls` 的参数解冻并训练，其他参数全部冻结。

实现：

```python
def configure_trainable_parameters(model):
    trainable_names = []

    for name, parameter in model.named_parameters():
        parameter.requires_grad = "cls" in name

        if parameter.requires_grad:
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError(
            "No trainable parameter contains lowercase 'cls'."
        )

    return trainable_names
```

启动日志必须输出：

```text
Trainable parameters:
  xxx.cls.conv1.weight
  xxx.cls.conv1.bias
  xxx.cls.fc.weight
  xxx.cls.fc.bias

Trainable parameter count: ...
Frozen parameter count: ...
Trainable ratio: ...
```

优化器只接收：

```python
trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]
```

## 5.2 冻结主干的运行模式

训练开始时：

```python
model.eval()
```

然后只将包含 `cls` 参数的模块切换为训练模式。

需要特别处理 BatchNorm：

* 冻结主干中的 BatchNorm 必须保持 `eval`。
* `cls` 中若存在 BatchNorm，第一版建议同样冻结运行均值和方差。
* BatchNorm 的 `weight`、`bias` 可以训练，但 running mean/variance 不更新。
* `cls` 中的卷积、激活、Dropout 和 Linear 正常训练。

这是为了避免八卡上各 rank 的 BatchNorm 统计不一致。

## 5.3 拆分 backbone 和 cls

最好把模型包装成：

```python
class TrainModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = ...
        self.cls = ...

    def forward(self, image0, image1):
        with torch.no_grad():
            features = self.backbone(image0, image1)

        logits = self.cls(features)
        return logits
```

`torch.no_grad()` 只包裹冻结主干，不能包裹 `cls`。

如果原模型无法直接拆开，第一版可以只依靠 `requires_grad=False`，但性能优化阶段应尽量明确划分 backbone 和 `cls`。

---

# 六、权重加载方案

兼容以下两类 PyTorch checkpoint：

```python
checkpoint = torch.load(path, map_location="cpu")

if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint
```

清理 DDP 前缀：

```python
state_dict = {
    key.removeprefix("module."): value
    for key, value in state_dict.items()
}
```

加载时输出：

```text
成功加载参数
缺失参数
多余参数
形状不匹配参数
cls参数加载结果
```

如果最后分类层原来不是两通道：

```text
跳过原分类层参数
重新初始化输出维度为2的分类层
```

如果已经是两通道，则直接加载。

---

# 七、固定 0.99 阈值优化

## 7.1 部署判定的等价形式

输出为：

```text
logits[:, 0]
logits[:, 1]
```

定义：

[
d=z_1-z_0
]

则第二通道 Softmax 概率为：

[
p_1=\operatorname{sigmoid}(d)
]

部署规则：

[
p_1>0.99
]

等价于：

[
d>\log(99)\approx4.59511985
]

测试和部署建议使用 FP32 margin：

```python
margin = logits[:, 1].float() - logits[:, 0].float()

prediction = margin > 4.59511985
```

需要展示置信度时再计算：

```python
probability = torch.softmax(logits.float(), dim=1)[:, 1]
```

## 7.2 训练损失

第一部分使用普通交叉熵：

```python
ce_loss = F.cross_entropy(logits.float(), target)
```

第二部分加入固定阈值辅助损失。

```python
def threshold_margin_loss(
    logits,
    target,
    threshold_margin=4.59511985,
    safety_margin=0.20,
    temperature=0.50,
):
    margin = logits[:, 1].float() - logits[:, 0].float()
    target = target.float()

    positive_loss = temperature * F.softplus(
        (
            threshold_margin
            + safety_margin
            - margin
        ) / temperature
    )

    negative_loss = temperature * F.softplus(
        (
            margin
            - threshold_margin
            + safety_margin
        ) / temperature
    )

    return torch.where(
        target > 0.5,
        positive_loss,
        negative_loss,
    ).mean()
```

总损失：

```python
loss = ce_loss + lambda_threshold * threshold_loss
```

建议初始值：

```yaml
loss:
  threshold: 0.99
  safety_margin: 0.20
  temperature: 0.50
  threshold_loss_weight: 0.20
```

## 7.3 损失权重调度

不要从第一个 step 就施加强阈值约束。

推荐：

```text
前10%训练step：
lambda_threshold = 0

接下来20%训练step：
lambda_threshold从0线性增加到0.2

剩余70%：
lambda_threshold = 0.2
```

这样先学会区分类别，再推动类别 1 样本越过高置信度边界。

## 7.4 类别平衡方式

第一版只使用分层采样，不同时增加较大的 `class_weight`。

否则会形成：

```text
类别均衡采样
+
类别加权交叉熵
+
阈值辅助损失
```

三重补偿，可能导致类别 1 置信度过高、假阳性增加。

---

# 八、测试系统

按照你的要求，本版只建立：

```text
train
test
```

不单独建立 validation。

但要明确：如果根据周期性 test 的结果选择最佳 epoch，那么该 test 在统计意义上已经参与了模型开发。因此框架应同时报告：

```text
last checkpoint结果
best observed test F1 checkpoint结果
```

并且：

* 不在 test 上搜索阈值，阈值始终固定为 `0.99`。
* 不根据 test 自动修改采样比例。
* 默认不根据 test 自动提前终止训练。
* test 主要用于观察和保存中间结果。

## 8.1 快速测试

每隔固定 step 执行确定性子集测试：

```yaml
evaluation:
  quick_test_every_steps: 1000
  quick_test_pairs_per_video: 128
```

从每个 `(game, label, video)` 的全部 `delta=2` pair 中，按时间均匀抽取最多 128 个。

这样小型游戏和短视频不会在快速测试中消失。

## 8.2 完整测试

```yaml
evaluation:
  full_test_every_steps: 5000
  full_test_at_end: true
```

完整测试枚举 test 中所有合法 `delta=2` pair。

## 8.3 固定阈值指标

全局保存：

```text
TP
FP
FN
TN
Precision@0.99
Recall@0.99
F1@0.99
Accuracy
Specificity
Balanced Accuracy
PR-AUC
ROC-AUC
Cross Entropy
```

同时按以下维度分组：

```text
每个游戏
每个类别
每个视频
游戏 × 类别
```

核心聚合指标：

```text
global_f1_tau099
macro_game_f1_tau099
worst_game_f1_tau099
macro_video_f1_tau099
```

建议监控重点不是只有 global F1，而是：

```text
global F1
+
macro game F1
+
worst game F1
```

## 8.4 错分类样本

所有 FP 和 FN 记录：

```text
game
label
video_id
frame0_id
frame1_id
delta
image0_path
image1_path
logit0
logit1
margin
probability_class1
prediction
error_type
checkpoint_step
```

输出目录：

```text
reports/test_step_00005000/
├── metrics.json
├── metrics_by_game.csv
├── metrics_by_video.csv
├── false_positive.parquet
├── false_negative.parquet
├── near_threshold.parquet
└── errors.html
```

HTML 中两张图片并排展示：

```text
image0 | image1
真实标签
预测标签
类别1概率
margin
game/video/frame信息
```

阈值附近样本额外分组：

```text
0.980 ≤ p1 < 0.990
0.990 < p1 < 0.995
0.995 ≤ p1 < 0.999
p1 ≥ 0.999
```

---

# 九、checkpoint 保存设计

每个需要保存的 checkpoint 标签都产生两个文件。

例如 `last`：

```text
checkpoints/
├── model_last.pth
└── checkpoint_last.pth
```

例如固定阈值 F1 最佳：

```text
checkpoints/
├── model_best_f1_tau099.pth
└── checkpoint_best_f1_tau099.pth
```

## 9.1 纯模型参数

```python
torch.save(
    unwrap_model(model).state_dict(),
    "model_last.pth",
)
```

只包含模型参数，可直接部署或重新加载。

## 9.2 完整训练状态

```python
torch.save(
    {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": best_metrics,
        "sampler_epoch": sampler_epoch,
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "config": resolved_config,
    },
    "checkpoint_last.pth",
)
```

八卡环境下只有 rank 0 写公共 checkpoint，并使用：

```text
先写临时文件
→ fsync
→ 原子rename
```

避免进程中断留下半个 checkpoint。

---

# 十、训练配置初始版本

```yaml
experiment:
  name: dual_frame_game_cls
  seed: 20260728
  output_dir: runs/dual_frame_game_cls

device:
  accelerator: cuda
  amp: true
  amp_dtype: float16
  compile: false

data:
  train_root: /data/train
  test_root: /data/test
  width: 208
  height: 448
  channels: 3
  normalization: zero_one
  filename_pattern: '^(?P<video_id>\d{2})(?P<frame_id>\d{5})\.png$'

pair:
  train_delta_probability:
    1: 0.15
    2: 0.70
    3: 0.15
  test_delta: 2

sampler:
  game_alpha: 0.25
  class0_probability: 0.50
  class1_probability: 0.50
  uniform_video: true
  deduplicate_within_global_batch: true

augmentation:
  random_affine:
    enabled: true
    probability: 0.50
    degrees: 2.0
    translate: [0.02, 0.02]
    scale: [0.98, 1.02]
    shear: [-1.0, 1.0]

  color_jitter:
    enabled: true
    probability: 0.80
    brightness: 0.15
    contrast: 0.15
    saturation: 0.10
    hue: 0.02

  random_erasing:
    enabled: true
    probability: 0.10
    scale: [0.005, 0.03]
    ratio: [0.3, 3.3]
    value: random

  horizontal_flip:
    enabled: false

model:
  checkpoint_path: /models/base_model.pth
  trainable_name_contains: cls
  num_classes: 2
  freeze_batchnorm_stats: true

loss:
  cross_entropy_weight: 1.0
  threshold: 0.99
  threshold_loss_weight: 0.20
  threshold_safety_margin: 0.20
  threshold_temperature: 0.50
  threshold_warmup_ratio: 0.10
  threshold_ramp_ratio: 0.20

optimizer:
  name: AdamW
  learning_rate: 0.001
  weight_decay: 0.0001

scheduler:
  name: cosine
  warmup_steps: 500
  min_learning_rate: 0.00001

train:
  epochs: 10
  steps_per_epoch: 1000
  local_batch_size: 64
  gradient_clip_norm: 5.0
  log_every_steps: 50

dataloader:
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 4
  pin_memory: true
  drop_last: true

evaluation:
  threshold: 0.99
  quick_test_every_steps: 1000
  quick_test_pairs_per_video: 128
  full_test_every_steps: 5000
  full_test_at_end: true
  save_all_errors: true
  html_max_errors_per_group: 200

checkpoint:
  save_last_every_steps: 1000
  save_best_test_f1: true
  save_model_only: true
  save_full_state: true
```

这里的 epoch 不是强制遍历全部帧，而是固定 `steps_per_epoch`。这样数据量继续增加时，训练时长仍然可控。

---

# 十一、CUDA 单卡开发路线

## 阶段 1：数据正确性

执行：

```bash
python tools/build_index.py \
  --train-root /data/train \
  --test-root /data/test \
  --output-dir indexes

python tools/audit_dataset.py \
  --index-dir indexes \
  --output-dir reports/data_audit
```

验收条件：

```text
全部图片尺寸为448×208
全部图片为3通道
全部文件名可解析
没有跨视频pair
没有跨类别pair
随机可视化100组delta=1/2/3 pair正确
test仅生成delta=2 pair
```

## 阶段 2：模型冻结正确性

先跑 100 step：

```bash
python tools/train.py \
  --config configs/cuda_debug.yaml \
  train.max_steps=100
```

验收：

```text
所有包含cls的参数requires_grad=True
其他参数requires_grad=False
cls参数有非零梯度
非cls参数没有梯度
训练100步后非cls参数逐元素完全不变
loss能够下降
输出形状为[B,2]
```

## 阶段 3：采样分布正确性

运行一个不训练的 sampler 检查：

```text
采样100万次
统计game比例
统计每游戏类别比例
统计视频比例
统计delta比例
```

期望：

```text
delta1 ≈ 15%
delta2 ≈ 70%
delta3 ≈ 15%

每游戏内部：
label0 ≈ 50%
label1 ≈ 50%
```

## 阶段 4：损失正确性

构造人工 logit：

```text
正样本margin很小：threshold loss大
正样本margin>4.8：threshold loss小
负样本margin接近4.6：threshold loss大
负样本margin<0：threshold loss小
```

验证损失无 NaN、梯度有限。

## 阶段 5：完整 CUDA 闭环

运行：

```bash
bash scripts/run_cuda_debug.sh
```

必须得到：

```text
训练日志
quick test
full test
错误样本报告
model_last.pth
checkpoint_last.pth
断点恢复成功
```

---

# 十二、迁移到 Ascend 单卡

设备抽象：

```python
def initialize_device(accelerator, local_rank):
    if accelerator == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    if accelerator == "npu":
        import torch_npu
        torch.npu.set_device(local_rank)
        return torch.device(f"npu:{local_rank}")

    raise ValueError(accelerator)
```

先检查实际环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python - <<'PY'
import sys
import torch
import torch_npu

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("TorchNPU:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
PY

npu-smi info
```

当前 TorchNPU 官方仓库给出的安装示例包含 CANN 9.0.0、PyTorch 2.10.0 和 TorchNPU 2.10.0.post2；官方发布信息也列出了 PyTorch 2.10 和 Python 3.13 支持。实际训练前仍要把服务器的精确版本写入实验日志。([GitHub][4])

单卡启动：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/train.py \
  --config configs/npu_1p.yaml
```

验收顺序：

```text
固定输入FP32前向正常
单batch反向正常
cls参数更新
checkpoint可保存
checkpoint可恢复
100步无NaN
测试指标可生成
```

随后再打开 BF16 或 FP16 AMP。PyTorch AMP 会让适合低精度的卷积、线性等运算使用 FP16/BF16，同时保留部分需要 FP32 数值范围的运算；本方案仍强制把阈值 margin 和指标计算转换到 FP32。([PyTorch Docs][5])

---

# 十三、八卡训练方案

Ascend 侧使用：

```text
一个进程绑定一张NPU
8个训练进程
HCCL后端
DistributedDataParallel
```

Ascend 官方迁移文档推荐使用 DDP，并通过 `backend="hccl"` 初始化进程组；单机多卡也支持使用 `torchrun` 拉起。([Hiascend][6])

初始化：

```python
dist.init_process_group(
    backend="hccl",
    init_method="env://",
)

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
```

包装模型：

```python
model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False,
    broadcast_buffers=False,
    gradient_as_bucket_view=True,
)
```

`broadcast_buffers=False` 的前提是所有 BatchNorm running stats 已被冻结。否则不能直接关闭。

启动脚本：

```bash
#!/bin/bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  tools/train.py \
  --config configs/npu_8p.yaml
```

八卡验收：

```text
8个rank全部启动
每个rank绑定不同NPU
每个rank的local batch不同
所有rank step数量一致
没有重复写checkpoint
测试样本不重复、不补齐
指标all_reduce正确
单卡和八卡loss趋势一致
```

---

# 十四、当前每秒 200 samples 的性能路线

## 14.1 首先判断数据瓶颈

一张未压缩 RGB 图像大小为：

```text
208 × 448 × 3 = 279,552 bytes
```

一个双帧样本约为：

```text
559,104 bytes
≈ 0.533 MiB
```

一百万张原始 RGB 图像约为：

```text
260 GiB
```

如果训练数据通过你提到的约 `300 MB/s` 链路读取，则不考虑任何开销时，未压缩双帧样本的理论上限约为：

```text
536 samples/s
```

当前 `200 samples/s` 对应约 `106.6 MiB/s` 的原始像素吞吐。由于实际还存在 PNG 解码、小文件打开、随机访问、数据增强和 CPU 到 NPU 传输，因此“远程存储 + PNG”很可能是锯齿利用率的重要来源，但需要 Profiler 进一步确认。

## 14.2 数据存储优化分三步

### 第一步：复制到本机 NVMe

正式性能测试不要直接跨服务器随机读取百万小文件。

优先：

```text
远程数据
→ 一次性复制到A3服务器本地NVMe
→ 本地建立索引
→ 本地训练
```

### 第二步：PNG 分片

如果本地仍受到小文件 metadata 开销影响，将 PNG 打包为大分片：

```text
shard_000.tar
shard_001.tar
...
```

保留 PNG 压缩，减少随机打开大量小文件的成本。

### 第三步：预解码 uint8 分片

如果 Profiler 显示 CPU PNG 解码仍是瓶颈，转换为固定大小原始数据：

```text
packed/
├── shard_000.bin
├── shard_001.bin
├── ...
└── index.parquet
```

每张图固定存储：

```text
uint8 [3, 448, 208]
```

运行时直接按 offset 读取，不再执行 PNG 解码。

代价是每百万张图片约需要 `260 GiB` 本地空间。

## 14.3 DataLoader 参数扫描

PyTorch DataLoader 原生支持 `num_workers`、`prefetch_factor` 和 `persistent_workers` 等配置。([PyTorch Docs][7])

不要直接固定一个参数，执行短时间网格测试：

```text
num_workers_per_rank = 2 / 4 / 8
prefetch_factor = 2 / 4
pin_memory = true / false
```

八卡时总 worker 数为：

```text
8 × num_workers_per_rank
```

因此 worker 太多也可能造成 CPU 争用。

初始值：

```yaml
dataloader:
  num_workers: 4
  prefetch_factor: 4
  persistent_workers: true
  pin_memory: true
```

## 14.4 一次传输完整双帧 batch

CPU DataLoader 返回：

```text
images: uint8 [B, 2, 3, 448, 208]
labels: int64 [B]
```

一次传到设备：

```python
images = images.to(device, non_blocking=True)
images = images.to(compute_dtype).div_(255.0)

image0 = images[:, 0]
image1 = images[:, 1]
```

不要分别对 `image0`、`image1` 执行多次零散 H2D。

## 14.5 batch size 扫描

在单卡 NPU 和八卡 NPU 上分别测试：

```text
local_batch = 32
local_batch = 64
local_batch = 128
local_batch = 256
```

每组固定运行 500～1000 step，记录：

```text
平均samples/s
P50/P95 step时间
data wait时间
forward时间
backward时间
optimizer时间
NPU峰值显存
```

选择吞吐最高且稳定的 batch，而不是只选择显存刚好不溢出的 batch。

## 14.6 减少同步

热路径中禁止每一步执行：

```python
loss.item()
tensor.cpu()
torch.npu.synchronize()
保存checkpoint
写CSV
```

建议每 50 step 统一记录一次日志。

---

# 十五、Profiler 执行方式

Ascend PyTorch Profiler 支持设置 `skip_first`、`warmup` 和 `active` step；官方文档建议通过短采集窗口避免全程 Profiling。([Hiascend][8])

建议：

```python
schedule = torch_npu.profiler.schedule(
    skip_first=10,
    wait=0,
    warmup=1,
    active=5,
    repeat=1,
)
```

第一轮不要打开 `with_stack=True`。官方案例指出调用栈采集会显著增加 Profiling 膨胀。([Hiascend][9])

Profiler 需要回答四个问题：

```text
1. DataLoader是否出现明显空洞？
2. CPU增强和PNG解码占用多少时间？
3. H2D是否与NPU计算重叠？
4. forward、backward、HCCL分别占多少时间？
```

根据结果决策：

```text
DataLoader空洞高
→ 本地缓存、加worker、打包数据、预解码

forward占比高
→ 增大batch、AMP、后续测试图模式

HCCL占比高
→ 检查是否误同步了冻结参数或buffer

测试/保存占比高
→ 降低完整测试和checkpoint频率
```

TorchNPU 当前官方仓库也提供图模式能力，但应在 eager 模式精度和数据流水线稳定后再开启。([GitHub][4])

---

# 十六、最终执行顺序

## 第一阶段：CUDA 正确性

```text
1. 建立索引
2. 数据审计
3. 双帧配对
4. 一致数据增强
5. 模型加载
6. cls冻结策略
7. 普通CE训练
8. 0.99固定阈值测试
9. checkpoint保存与恢复
```

## 第二阶段：算法增强

```text
1. 三级平衡采样
2. delta=15/70/15
3. 阈值辅助损失
4. 每游戏指标
5. 每视频指标
6. 错分类HTML
```

## 第三阶段：NPU 单卡

```text
1. FP32前向
2. 单步反向
3. 100步训练
4. AMP
5. 测试和checkpoint
```

## 第四阶段：八卡 DDP

```text
1. HCCL初始化
2. 分布式pair sampler
3. 八卡训练
4. 分布式测试
5. rank0 checkpoint
```

## 第五阶段：性能优化

严格按以下顺序：

```text
本地化数据
→ 调DataLoader
→ 调batch size
→ AMP
→ 减少同步
→ PNG分片
→ 预解码uint8
→ Profiler复测
→ 最后再测试图模式
```

---

## 最终验收标准

框架完成时应满足：

```text
输入固定为两张[B,3,448,208]
部署和主测试固定delta=2
训练delta比例可配置且统计正确
只有名称包含cls的参数发生变化
每个游戏和类别都被均衡采样
0.99阈值比较与部署严格一致
周期性test可复现
所有FP/FN可定位到两张原图
模型参数和完整训练状态分别保存
CUDA、NPU单卡、NPU八卡使用同一套业务代码
八卡训练不存在持续的DataLoader空洞
Profiler能够明确解释剩余性能瓶颈
```

实际实施从 `build_index.py → pair_dataset.py → freeze_policy.py → CUDA 100 step smoke test` 开始；在这四项通过前，不应先迁移八卡或打开图编译。

[1]: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html "https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html "https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"
[3]: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html "https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html"
[4]: https://github.com/Ascend/pytorch/blob/master/README.md "https://github.com/Ascend/pytorch/blob/master/README.md"
[5]: https://docs.pytorch.org/docs/stable/amp.html "https://docs.pytorch.org/docs/stable/amp.html"
[6]: https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000202.html "https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000202.html"
[7]: https://docs.pytorch.org/docs/stable/data.html "https://docs.pytorch.org/docs/stable/data.html"
[8]: https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0033.html "https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0033.html"
[9]: https://www.hiascend.com/document/caselibrary/detail/profilingcase_007 "https://www.hiascend.com/document/caselibrary/detail/profilingcase_007"
