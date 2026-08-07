## 核心结论

这个报错已经可以明确定位：**最新 `main` 的自动 train/val splitter 对“源视频”的定义与真实数据集结构发生了冲突。**

当前代码把源视频定义成：

```text
source_video_uid = game::video_id
```

因此 `MC/0/01xxxxx.png` 和 `MC/1/01xxxxx.png` 都会被认为属于同一个 `MC::01`。但 `_group_frames()` 随后又强制规定一个 `source_video_uid` 只能对应一个 `(game, label)`，所以只要 `MC::01` 同时在 label 0 和 label 1 中出现，就必然抛出你看到的异常。

**我认为这是当前 splitter 的设计缺陷，而不应该简单归咎于你的数据。** 更准确地说，框架没有覆盖真实数据中两种都很合理的情况：

1. 同一原始视频中可能同时存在 label 0 和 label 1 的片段；
2. label 0、label 1 各自独立编号，恰好都存在 `video_id=01`，但其实是两个不同的视频。

这两种情况的修法完全不同，所以**现在最不应该做的是直接把 UID 改成 `game::label::video_id` 后继续训练**。如果第一种情况成立，那样会把同一真实视频拆进 train/val，制造严重的数据泄漏。

---

# 一、为什么这个问题没有在测试中被发现

这里还有一个比较明显的测试漏洞。

生产文件名协议明确规定：

```python
DEFAULT_FILENAME_PATTERN =
    r"^(?P<video_id>\d{2})(?P<frame_id>\d{5})\.png$"
```

也就是说真实数据里的 `video_id` **只有两位，00–99**。

但 `test_splitter.py` 构造测试数据时使用了：

```python
f"{game_index:02d}{label}{video:02d}"
```

作为 `video_id`。

这实际上是 **5 位 video_id**，而且特意把 `label` 编进了 video ID：

```text
game_a label=0 video=0 → 00000
game_a label=1 video=0 → 00100
```

所以测试数据天然不可能出现：

```text
game_a / label 0 / video 01
game_a / label 1 / video 01
```

这种真实数据碰撞。

换句话说：

> **splitter 的单元测试使用了不符合生产 filename contract 的 source video ID，并因此绕过了当前这个真实问题。**

这也是这次应该一起修掉的地方。

---

# 二、先判断你当前 `MC::01` 属于哪一种情况

在改代码前，我建议先对数据做一次非常简单的 source identity 检查。

你可以在服务器上运行：

```python
from pathlib import Path
from collections import defaultdict

root = Path("/你的/train_all")

groups = defaultdict(lambda: defaultdict(list))

for game_dir in root.iterdir():
    if not game_dir.is_dir():
        continue

    for label in ("0", "1"):
        label_dir = game_dir / label
        if not label_dir.is_dir():
            continue

        for p in label_dir.glob("*.png"):
            name = p.stem

            if len(name) != 7:
                continue

            video_id = name[:2]
            frame_id = int(name[2:])

            groups[(game_dir.name, video_id)][label].append(frame_id)

for (game, video_id), labels in sorted(groups.items()):
    if len(labels) <= 1:
        continue

    print(f"\n{game}::{video_id}")

    for label, frame_ids in labels.items():
        print(
            f"  label={label}: "
            f"frames={len(frame_ids)}, "
            f"range={min(frame_ids)}..{max(frame_ids)}"
        )
```

你大概率会看到：

```text
MC::01
  label=0: frames=....
  label=1: frames=....
```

然后需要按照实际数据来源判断。

### 情况 A：`MC::01` 确实是同一个原始视频

例如原视频 `01.mp4` 中：

```text
frame 00000~01200 → label 0
frame 01201~02000 → label 1
```

那么当前：

```text
source_video_uid = MC::01
```

**是正确的。**

错误的是 splitter 假设：

> 一个 source video 必须只有一个 label。

这种情况下应该修改 splitter，使一个 source video 可以贡献多个 label。

### 情况 B：它其实是两个完全不同的视频

例如：

```text
MC/0/0100000.png → negative_video_01.mp4
MC/1/0100000.png → positive_video_01.mp4
```

只是 0/1 两个目录分别从 `01` 开始编号。

那么当前：

```text
source_video_uid = MC::01
```

就是错误的。

这种情况下真正的源身份实际上应该类似：

```text
MC::0::01
MC::1::01
```

或者更理想地由原始视频 ID 明确定义。

---

# 三、我推荐的框架修复方案

我不建议只针对这一个报错打一行补丁。应该让 CLSTrainer **显式建模 source identity**。

## 第一层：立即修复 splitter，使 mixed-label source video 合法

这是我认为当前 `main` 必须修的部分。

目前 `_group_frames()` 大致是：

```python
uid = source_video_uid(frame.game, frame.video_id)

group = {
    "game": frame.game,
    "label": frame.label,
    ...
}

if group["label"] != frame.label:
    raise ValueError(...)
```

应该改成：

```text
source video
├── game
├── label 0
│   ├── frame_ids
│   └── pair_count delta 1/2/3
│
└── label 1
    ├── frame_ids
    └── pair_count delta 1/2/3
```

也就是类似：

```python
group = {
    "game": frame.game,
    "labels": set(),
    "frame_ids_by_label": {
        0: set(),
        1: set(),
    },
}
```

最后计算：

```python
pair_counts_by_label = {
    0: {1: ..., 2: ..., 3: ...},
    1: {1: ..., 2: ..., 3: ...},
}
```

这里有一个非常重要的正确性条件：

**不能跨 label 构造 pair。**

而当前真正训练 pair 的实现本来就是按：

```python
(game, label, video_id)
```

分组的，所以 mixed-label source video 并不会导致 label 0 和 label 1 的帧互相组成 pair。

因此这项修改在训练语义上是完全兼容的。

---

# 四、split 算法也必须一起修改

这才是核心。

当前 splitter 的算法实际上是：

```text
先把 source video 放入一个 (game,label) stratum

例如：

(MC, 0):
    video 01
    video 02
    video 03

(MC, 1):
    video 04
    video 05
    video 06

然后每个 stratum 独立挑 10%~20% 到 val
```

这依赖：

> 一个 source video 只能属于一个 label。

所以 `_group_frames()` 即便不报错，后面的算法也不能直接继续用。

应该改成 **group-aware multi-stratum split**。

假设：

```text
MC::01

label 0:
    delta2 pairs = 300

label 1:
    delta2 pairs = 100
```

这个 source video 的贡献向量就是：

```text
(MC,0) → 300
(MC,1) → 100
```

整个 `MC::01` 只能做一个原子决策：

```text
全部 → train

或者

全部 → val
```

不能拆。

### 新的优化目标

对于每个 stratum：

```text
s = (game, label)
```

计算：

```text
total_pairs[s]
target_val_pairs[s] = val_ratio × total_pairs[s]
```

选择一批完整 source videos，使：

```text
val_pairs[s]
```

尽量接近：

```text
target_val_pairs[s]
```

可以使用归一化误差作为 objective：

```text
cost =
Σ_s
|val_pairs[s] - target_val_pairs[s]|
------------------------------------
       max(total_pairs[s], 1)
```

然后：

1. 所有 source video 按现有 `_stable_rank(seed, uid)` 固定排序；
2. 一个个尝试加入 val；
3. 如果加入后全局 cost 更低，则加入；
4. 必须保证有足够 source video 的 stratum 同时保留 train 和 val；
5. 用 stable rank 做 tie-break，保证同 seed 完全可复现。

这比目前“每个 label 独立 greedy”更正确。

---

# 五、manifest 也必须升级

现在 `split_manifest.parquet` 每个 source video 只保存：

```text
source_video_uid
game
label          ← 单个 label
frame_count
pair_count_delta1
pair_count_delta2
pair_count_delta3
```

因此 manifest schema 本身也假设：

> source video = 单标签。

建议直接把：

```python
SPLIT_ALGORITHM_VERSION = 2
```

升级到：

```python
SPLIT_ALGORITHM_VERSION = 3
```

每行改为：

```text
source_video_uid
game
labels

frame_count_label0
frame_count_label1

valid_pair_count_label0_delta1
valid_pair_count_label0_delta2
valid_pair_count_label0_delta3

valid_pair_count_label1_delta1
valid_pair_count_label1_delta2
valid_pair_count_label1_delta3

split
...
```

或者保存一个结构化 contribution 字段。

我更推荐前一种，Parquet 后续分析比较方便。

升级后旧的 v2 manifest 应明确拒绝：

```text
split manifest algorithm version mismatch:
manifest=2 current=3

Delete/rebuild the manifest.
```

不要尝试静默迁移，因为分割算法已经改变，旧指标和新指标不是严格可比的。

---

# 六、还需要支持“label 下 video_id 独立编号”的数据集

前面的修改解决的是“一个真实视频包含两个 label”。

但还存在情况 B。

因此我建议增加一个显式配置：

```yaml
data:
  source_video_identity:
    mode: game_video
```

支持至少两种模式：

```text
game_video
    source_video_uid = game::video_id

game_label_video
    source_video_uid = game::label::video_id
```

其中默认必须保持：

```yaml
mode: game_video
```

因为它在防泄漏方面更保守。

只有用户明确知道：

> 0/01 和 1/01 是两个独立原始视频

才允许：

```yaml
mode: game_label_video
```

并在 `doctor` / `dataset prepare` 中打印一个明显提示：

```text
WARNING:
source_video_identity=game_label_video

The framework assumes identical video_id values under different
labels are physically unrelated source videos.

If this assumption is false, train/validation leakage may occur.
```

更长期的理想方案甚至不是这两个模式，而是允许显式 source ID sidecar，例如：

```text
path / logical video
→ true_source_video_uid
```

这样不会依赖两位数字猜测真实视频身份。

---

# 七、当前 strict audit 不应该被削弱

这一部分现在的设计反而是对的。

当前 audit 明确认为：

> label-independent `source_video_uid` 如果横跨 train/val/test，就是 leakage。

这个检查应该继续保留。

因此我不建议为了“让程序跑起来”直接：

```python
source_video_uid = f"{game}::{label}::{video_id}"
```

然后把所有 audit 一起改成 label-dependent。

那相当于绕过问题。

正确做法应该是：

```text
先明确 source identity
        ↓
source-video-level atomic split
        ↓
(game,label) 只是 balance dimension
        ↓
label 不再决定 source identity
```

这几个概念必须分开。

---

# 八、这次应该顺便增加一个数据身份预检

实际上用户不应该等到：

```text
splitter.py line 125
ValueError
```

才知道数据有这种结构。

建议在：

```bash
cls-trainer dataset prepare
```

真正 split 前增加：

```text
Source identity analysis
──────────────────────────────────
Source videos:             184
Single-label videos:       162
Mixed-label videos:         22

Mixed-label examples:
  MC::01  labels=[0,1]
  MC::05  labels=[0,1]
  XY::03  labels=[0,1]

label0 pairs: ...
label1 pairs: ...
```

如果当前 splitter 不支持 mixed label，则应该在这里给出清晰解释，而不是底层 traceback。

未来修好之后，它甚至可以正常显示：

```text
Mixed-label source videos: 22
Supported: yes
Atomic split enforced: yes
```

这会显著改善数据准备体验。

---

# 九、对应测试必须补齐

这次至少应该增加以下测试。

第一组一定使用**真实的两位 video ID**：

```python
_frames("MC", 0, "01", ...)
_frames("MC", 1, "01", ...)
```

不能再使用现在测试里的 5 位伪 video ID。

然后覆盖：

```text
1. 同 source video 同时拥有 label 0/1，不报错
2. MC::01 的所有 frame 永远位于同一 split
3. mixed-label video 不会跨 label 构造 pair
4. train/val 都保持合理的 game × label pair 比例
5. 同 seed 得到完全相同 assignment
6. manifest roundtrip 后 assignment 不变
7. v2 manifest 被 v3 splitter 明确拒绝
8. extend 模式新增 mixed-label source video 时旧 assignment 不变化
9. strict audit 能发现 source UID 跨 split
```

还要增加一个生产契约测试：

```python
assert re.fullmatch(DEFAULT_FILENAME_PATTERN, ...)
```

确保 splitter fixture 使用的文件命名与真实生产协议一致。

---

# 我建议这次的修改优先级

**P0 必修：**

1. `_group_frames()` 支持一个 source video 下多个 label；
2. splitter 改为 source-video 原子、多 stratum 联合平衡；
3. manifest schema 升级 v3；
4. 修复 splitter 测试，使其真正使用两位 video ID；
5. 增加 mixed-label regression test。

**P1 强烈建议：**

6. `dataset prepare` 增加 source identity 预检报告；
7. 加 `source_video_identity.mode`，显式支持 label 目录独立编号的数据集；
8. `doctor` 输出当前 source identity contract。

---

## 你现在应该怎么处理

**先不要改文件名，也不要把 `source_video_uid()` 直接改成带 label。**

如果你的 `MC::01` 是同一个实际录屏中分别抽出的正负帧，那么这次就是 **CLSTrainer splitter 本身需要修复**，正确方案是支持 mixed-label source video。

如果 `MC/0/01` 和 `MC/1/01` 实际来自两个完全无关的视频，那么说明当前两位 `video_id` 只在 label 内唯一，需要改 source identity contract；短期可以重新编号使 `(game, video_id)` 全局唯一，长期则应该让框架显式支持这种身份模式。

**从当前代码结构和测试来看，我建议按前面的 P0 方案修改 `splitter.py`，而不是绕过这个异常。** 这样既能解决你当前 `MC::01` 的问题，也能继续保证我们前面为了泛化评估特别强调的 train/validation 源视频隔离。
