



## 核心结论

这套框架的基础工程能力已经比较完整，真正需要补强的不是“再加几个常规数据增强”，而是三条主线：

1. **把数据划分改成基于源视频的自动、可复现逻辑划分**，训练集和验证集可以共用一个物理目录，但必须通过持久化 manifest 保证源视频不跨 split、验证集不会随数据变化而漂移。
2. **把训练目标从普通二分类准确性，升级为 0.99 阈值下的低误报工程**。核心手段应是困难负样本闭环、负样本细分类、受约束的模型选择指标和适度解冻，而不是单纯加强正则化。
3. **先修复两个 NPU 高优先级正确性问题**：最新的 float64 修复仍然不完整；八卡评估中还存在 CPU tensor 通过 HCCL 做归约的高风险路径。

建议按照 **NPU 正确性 → 自动划分 → 困难负样本与评估协议 → 模型和损失改造 → 性能与易用性** 的顺序实施。不要同时修改所有训练策略，否则很难判断泛化提升来自哪里。

---

## 一、当前框架审查结果

### 1. 已经做得比较好的部分

当前仓库不是一个简单训练脚本，已经具备不少生产框架应有的能力：

- 严格配置 Schema、配置来源追踪、运行预检和 dry-run；
- train/val/test 角色区分、内容哈希与源视频泄漏检测；
- 视频级索引、懒生成 pair、packed uint8 后端；
- 确定性 sampler、断点恢复、Run 管理和错误报告；
- 按游戏统计 macro/worst-game 指标；
- 训练阈值损失与生产阈值共用 `decision.threshold`。

生产配置目前固定使用 0.99 阈值、类别平衡采样、轻量增强、仅训练名称包含 `cls` 的参数，并用固定阈值下的 global/macro/worst-game F1 组合选择模型。fileciteturn13file0L1-L7

数据侧也已经定义了 `source_video_uid = game::video_id`，并明确要求同一源视频不能横跨 train/val/test。这为自动划分提供了正确的基础。fileciteturn11file0L1-L7

所以，不建议推翻现有框架；应该在现有索引、审计、sampler 和 evaluator 上扩展。

### 2. 必须先修复的正确性问题

#### P0-1：最新 float64 修复不完整

最新提交把 interval accumulator 从 `float64` 改成了 `float32`，提交说明也明确指出 NPU 上创建 double tensor 会触发 `k::double`。fileciteturn15file0L3-L12

但是训练循环当前仍然执行：

```python
loss.detach().double()
components["cross_entropy"].double()
components["threshold_loss"].double()
```

然后才累加到 float32 accumulator。也就是说，**NPU 上仍然会先创建 float64 device tensor**，最新修复只改了目标 accumulator，没有消除 float64 中间张量。fileciteturn31file0L1-L2

应改为：

```python
interval_accum["loss_sum"].add_(
    loss.detach().to(interval_accum["loss_sum"].dtype) * batch_samples
)
```

其他两项同理。更简单也可以统一 `.float()`，但绑定到 accumulator dtype 更不容易再次回归。

同时增加三层保护：

- AST 或 grep 测试：NPU device 路径禁止 `.double()` 和 `torch.float64`；
- 单元测试：检查 interval 更新过程中所有 device tensor dtype；
- NPU 单卡真实 smoke，而不是仅使用 mock。

#### P0-2：八卡分组指标归约存在 HCCL 风险

Evaluator 的 `_all_reduce_group_arrays` 先把 NumPy 数组转换成 **CPU tensor**，再直接调用默认 process group 的 `dist.all_reduce`。fileciteturn34file0L1-L2

但运行时将 NPU 固定映射到 HCCL。fileciteturn36file0L1-L7 当前对应测试只在 CPU/Gloo 下运行，没有覆盖 HCCL。fileciteturn53file0L1-L7

这是一个高风险路径：PyTorch 的设备后端设计要求 collective 与后端支持的设备类型匹配；当 CPU 与加速器通信需要共存时，应显式配置多后端或将 tensor 放到对应加速器设备。citeturn533517search0turn533517search2

建议将接口改为：

```python
def _all_reduce_group_arrays(dist, arrays, device):
    tensor = torch.from_numpy(array).to(device)
    dist.all_reduce(tensor)
    array[...] = tensor.cpu().numpy()
```

MIN/MAX 数组也采用相同方式。数组可能较大时，可以分块归约，避免一次占用过多 NPU 内存。

这项结论是静态代码审查得到的高风险判断，仍需在真实八卡 HCCL 环境中确认。

#### P0-3：DDP 下记录的 threshold weight 不正确

`threshold_weight_sum` 是 Python float，没有被加入分布式 all-reduce；但分母 `samples` 被全局归约了。因此在 8 卡时，日志中的 `interval_threshold_weight` 大约会变成真实值的 `1/8`。fileciteturn27file0L1-L2

这不直接改变梯度，但会误导训练诊断。应把它作为第四个 float32 accumulator 一起 all-reduce。

#### P0-4：NPU 自动化门禁不足

当前 GitHub Actions 只执行 CPU lint、CPU 测试、Gloo 分布式测试和合成训练。fileciteturn16file0L1-L7 仓库有 1P/8P NPU smoke 脚本，但不是自动验收门禁；NPU runtime 测试也主要通过 fake module 验证初始化顺序。fileciteturn54file0L1-L7 fileciteturn58file0L1-L7

建议在内部 NPU 服务器增加独立 CI：

- 1P：forward、backward、BF16 autocast、完整评估、保存与恢复；
- 8P：HCCL DDP、quick/full evaluation、分组指标、checkpoint；
- 设备算子检查：`bincount`、`scatter_add_`、`nonzero`、`index_select`、GradScaler 路径；
- world size 为 1 和 8 时，混淆矩阵计数必须完全一致；
- loss 允许小幅 BF16 误差，但不能出现 dtype 或 device fallback。

---

## 二、训练目录内自动生成验证集

### 1. 正确的设计不是随机抽帧，而是源视频级划分

目前 `build_index.py` 要求用户准备独立的 train、val、test root；不提供 val 时，还会把 test 当作 validation，从而失去独立测试集。fileciteturn10file0L1-L7 README 也要求物理准备三个目录。fileciteturn42file0L1-L2

新的流程应允许：

```text
/data/train_all/
  game_a/0/...
  game_a/1/...
  game_b/0/...
  game_b/1/...

/data/test/...
```

框架扫描一次 `train_all`，然后按 `source_video_uid` 生成 train/val 两套逻辑索引。图片本身不移动、不复制。

绝对不能按 frame 或 pair 随机划分，否则相邻帧、同一视频中的高度相似画面会同时进入训练和验证，产生严重的信息泄漏。

### 2. 建议的数据协议

增加下面这类配置。字段名可以调整，但语义应保持：

```yaml
data:
  source_root: /data/train_all
  test_root: /data/test

  split:
    mode: from_train
    val_ratio: 0.10
    seed: 20260728
    group_key: source_video_uid
    stratify_by: [game, label]
    balance_by: legal_pair_count
    target_delta: 2
    manifest: indexes/split_manifest.parquet
    on_new_groups: error
    small_stratum_policy: error
```

划分过程应为：

1. 扫描所有帧，构造视频级统计；
2. 以 `game::video_id` 为不可分割 group；
3. 统计每个 group 的 label、帧数以及 delta=1/2/3 的合法 pair 数；
4. 在每个 `(game, label)` stratum 内做确定性 group split；
5. 优先让验证集的 **合法 delta=2 pair 比例** 接近目标比例，而不是只平衡视频数或帧数；
6. 生成 train/val/test frame、video 和 video-entry Parquet；
7. 执行现有内容哈希、源视频 UID 和类别完整性审计。

建议自己实现一个无外部依赖的确定性 greedy splitter：

```text
目标函数 =
  验证集 pair 比例误差
  + λ1 × 各游戏比例误差
  + λ2 × 各标签比例误差
```

先用稳定哈希 `hash(seed, source_uid)` 打乱同一 stratum 内的 group，再逐个选择使目标函数最小的 group。

### 3. manifest 必须成为长期契约

建议输出：

```text
indexes/
├── split_manifest.parquet
├── split_summary.json
├── train_frames.parquet
├── val_frames.parquet
├── test_frames.parquet
└── ...
```

manifest 至少记录：

```text
source_video_uid
game
label
split
frame_count
valid_pair_count_delta1/2/3
dataset_fingerprint
split_seed
split_algorithm_version
```

重要行为：

- 相同数据、seed 和算法版本必须产生完全相同的划分；
- 再次运行默认复用 manifest；
- 数据增加后不能静默重新洗牌已有验证集；
- 默认 `on_new_groups=error`，要求用户显式执行 extend；
- extend 时只给新 group 分配 split，不改变已有 group；
- 某个 `(game,label)` 只有一个源视频时，应直接报出无法无泄漏划分，而不是偷偷按帧拆分。

### 4. 易用性入口

将低层级的 Python tools 收口为 CLI：

```bash
cls-trainer dataset prepare \
  --config configs/recipes/game_cls_production.yaml \
  --train-root /data/train_all \
  --test-root /data/test \
  --val-ratio 0.10

cls-trainer dataset audit --config ...
cls-trainer dataset pack --config ...
```

训练命令可以支持：

```yaml
data:
  prepare_if_missing: true
```

但生产环境更推荐 prepare 与 train 分离，避免八卡进程同时尝试建索引。

---

## 三、针对楼梯误识别为地板、木桥的泛化方案

### 1. 根因判断

这种错误很像模型学习了“重复横向纹理、透视线、木质边缘”等统计捷径，而没有真正学习楼梯的三维层级和结构关系。

当前框架有四个限制：

- 只训练名称包含 `cls` 的参数，主干完全冻结；如果预训练特征不能线性地区分楼梯和木桥，分类头只能继续依赖表面纹理。fileciteturn22file0L1-L7
- sampler 只按游戏、类别和帧间 delta 平衡，不知道“地板”“木桥”“栅栏”等负样本亚型。fileciteturn44file0L1-L7
- 当前增强主要是轻微 affine、color jitter 和 random erasing，无法系统覆盖结构相似的困难负样本。fileciteturn43file0L1-L7
- 模型选择仍以固定阈值下的 F1 为主；F1 并不直接表达“误报必须极低”的业务约束。fileciteturn29file0L1-L7

因此，误识别首先是**数据与验收协议问题**，其次才是损失函数问题。

### 2. 建立困难负样本闭环

增加每个视频的可选 metadata：

```text
source_video_uid
scene_type
negative_subtype
capture_domain
difficulty
sample_weight
```

楼梯任务可以先定义：

```text
negative_subtype:
- flat_floor
- wooden_bridge
- railing
- ladder_like_texture
- roof_tiles
- repeated_stripes
- perspective_lines
- ambiguous
```

然后增加 `mine-hard-negatives` 工作流：

1. 使用当前最佳模型扫描训练侧负样本池或独立 mining pool；
2. 按 `p(stair)` 从高到低排序；
3. 每个源视频只保留有限 top-K，避免连续帧淹没结果；
4. 按 embedding、图像哈希或视频来源去重；
5. 人工确认标签；
6. 写入版本化 hard-negative manifest；
7. 下一轮 sampler 固定混入一定比例困难负样本。

建议初始 batch 构成：

```text
正样本             50%
普通负样本         25%
困难负样本         25%
```

比例不应固定死，应根据验证集的 recall 和 FPR 调整。困难负样本还要限制单个视频的采样上限，避免模型记忆少量场景。

**不要直接把验证集或测试集中的误报样本加入训练。** 应在训练侧同分布数据中寻找同类错误，否则会破坏验证协议。

### 3. 重构模型选择指标

生产阈值为 0.99 时，建议把模型选择改为受约束的分层规则：

```text
第一层：模型是否合格
- global FPR <= 配置上限
- worst-game FPR <= 配置上限
- worst-negative-subtype FPR <= 配置上限
- positive recall >= 最低要求

第二层：在合格模型中排序
- 优先最大化 recall
- 再最大化 worst-group recall
- 再最小化 negative score p99.9
```

新增指标：

- 固定 0.99 阈值下的 FP、FN、FPR、specificity、recall；
- 按 game、negative subtype、domain 分组；
- `recall_at_max_fpr`；
- low-FPR partial AUC；
- 负样本 score 的 p99、p99.9、最大值；
- 接近阈值区间的样本数量；
- 每组负样本总量，避免在样本太少时误判“零误报”。

现有 evaluator 已经计算 Brier score、ECE 和阈值附近样本，但主要是全局统计。fileciteturn35file0L1-L2 建议增加只覆盖 `[0.95, 1.0]` 的 tail calibration，因为普通全局 ECE 很容易被大量低分负样本稀释。

### 4. 损失函数的合理升级

现有 threshold loss 本质是一个决策边界附近的 margin loss。0.99 对应 logit margin 约为 4.595；当前 safety margin 0.2 大致要求：

- 正样本达到约 0.9918 以上；
- 负样本保持在约 0.9878 以下。

这适合让输出越过业务边界，但无法单独解决“楼梯与木桥的语义区分”。fileciteturn23file0L1-L7

建议扩展为逐样本 loss，再加入两项可选能力：

**负样本尾部 OHEM**

只对 batch 中分数最高的部分负样本额外加权：

```python
negative_tail_loss = topk(
    softplus((negative_margin - target_margin) / temperature),
    k=hard_negative_k,
).mean()
```

重点压制接近或越过 0.99 的负样本，而不是平均处理所有容易负样本。

**正负 pairwise ranking**

在同一 batch 中要求正样本 margin 高于困难负样本：

```text
L_rank = relu(rank_margin - positive_margin + hard_negative_margin)
```

推荐最终形式：

```text
L =
  CE
  + λ_threshold × threshold_margin_loss
  + λ_tail × negative_tail_OHEM
  + λ_rank × pairwise_ranking
```

不建议第一轮直接默认使用 Focal Loss、MixUp 或 CutMix。它们不是不能用，但会改变输出概率形态和场景语义，必须单独验证 0.99 阈值下的校准和误报。

### 5. 从纯分类头训练升级为分阶段解冻

当前 `trainable_name_contains="cls"` 过于依赖参数命名，而且只能表达“训或不训”。建议改成正则规则和参数组：

```yaml
model:
  trainable_rules:
    - pattern: "^cls\\."
      lr_scale: 1.0
      unfreeze_at_step: 0

    - pattern: "^backbone.stage4\\."
      lr_scale: 0.10
      unfreeze_at_step: 1000

    - pattern: "^backbone.adapters\\."
      lr_scale: 0.30
      unfreeze_at_step: 0
```

训练分为：

1. **Head warmup**：只训练分类头；
2. **Partial unfreeze**：解冻主干最后一个 stage 或轻量 adapter；
3. **低学习率联合收敛**：主干最后层学习率为 head 的 0.05～0.1 倍。

这通常比直接全量解冻更适合当前用途：既允许特征适配楼梯结构，又降低小数据集把整个主干训坏的风险。

### 6. 数据增强只做有针对性的扩展

保留“两帧共享几何变换”的现有设计，这是正确的。新增增强应围绕真实混淆来源：

- 轻微 perspective；
- random resized crop；
- gamma、曝光和阴影变化；
- blur、噪声、压缩退化；
- 局部遮挡；
- 木纹、条纹和重复结构相关的背景覆盖。

几何变换必须对两帧一致。光照类增强可以采用“共享主变化 + 很小的逐帧扰动”，模拟两帧曝光差，但不能让两帧完全独立随机，否则可能破坏任务时序语义。

---

## 四、训练效率与使用体验改造顺序

### 第一阶段：正确性门禁

优先修改：

1. 清除训练路径中的 `.double()`；
2. 把 `threshold_weight_sum` 加入 all-reduce；
3. 分组指标使用 NPU tensor 做 HCCL reduction；
4. 增加真实 NPU 1P/8P smoke；
5. 在 `doctor` 中检查 torch、torch_npu、CANN、BF16、HCCL 和关键算子。

验收标准：

- NPU 路径不产生任何 float64 device tensor；
- 1P 和 8P 均完成 train、quick eval、full eval、save、resume；
- 1P/8P 混淆矩阵完全一致；
- 日志中的 threshold weight 不随 world size 改变。

### 第二阶段：自动划分与数据协议

修改 `indexing.py` 和 CLI，增加 split manifest、视频级 stratified split 和增量策略。

验收标准：

- 无须物理 val 文件夹；
- 相同输入与 seed 产生字节级一致的 manifest；
- `source_video_uid` 在 split 间零重叠；
- 验证集按合法 pair 数接近目标比例；
- 新增数据不会静默改变原验证集；
- test 始终保持独立。

### 第三阶段：困难负样本与低误报评估

增加：

- metadata sidecar；
- `negative_subtype`；
- hard-negative mining；
- challenge validation；
- FPR、tail score 和受约束模型选择。

验收标准不应写成“F1 提升”，而应写成：

```text
固定 challenge set：
- global FPR 下降
- wooden_bridge FPR 下降
- flat_floor FPR 下降
- positive recall 不低于业务下限
- 独立 test 上结论一致
```

### 第四阶段：模型与损失消融

至少分开做以下实验：

| 实验 | 变化 |
|---|---|
| A | 当前基线 |
| B | 基线 + 困难负样本 |
| C | B + partial unfreeze |
| D | C + negative-tail OHEM |
| E | D + 后处理校准 |

通常 B 的信息价值最大。只有 B 仍不足时，再判断 C、D 是否值得进入默认配置。

### 第五阶段：效率和易用性

现有视频级懒采样与 packed uint8 后端已经是正确方向。fileciteturn45file0L1-L7 fileciteturn52file0L1-L7 后续重点应放在：

- `val_quick` 默认只计算指标，不创建 Parquet/HTML/report 目录；
- full evaluation 才写完整错误报告；
- 避免每个 eval batch 都同步 `.cpu()` 全量数据，只回传聚合统计和有限 top-K 错误；
- 增加 `cls-trainer benchmark data`，自动比较 worker、prefetch 和 packed/png；
- 对纯 head-only、无增强的超参搜索支持离线 feature cache；
- checkpoint improvement 不必每次强制写完整大模型，可把训练状态与轻量模型快照分开；
- 生产配置默认使用本地 NVMe 上的 packed 数据，`pin_memory` 和 worker 数由实测决定，不硬编码认为越大越好。

---

## 综合结论

最优先的改造不是更换 backbone 或堆复杂 loss，而是：

1. **修完 NPU dtype、分布式归约和监控正确性问题；**
2. **实现以源视频为 group 的自动 train/val 划分和不可漂移 manifest；**
3. **建立地板、木桥等困难负样本的采集、标注、采样和 challenge 验证闭环；**
4. **用 FPR 约束和负样本尾部指标替代 F1-only 模型选择；**
5. **最后再通过 partial unfreeze、tail OHEM 和校准进一步提高泛化。**

其中，自动划分解决的是操作成本和验证可信度；困难负样本闭环解决的才是楼梯误识别问题；NPU 门禁则决定这套框架能否稳定进入生产。
