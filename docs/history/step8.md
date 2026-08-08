## 核心结论

这个报错说明：**最新 `main` 已经把 source identity 做得更严格了，但你当前 train/test 的两位 `video_id` 很可能只是各 split 内部编号，并不是全数据集唯一 ID。**

例如现在框架看到：

```text
train/MC/0/0100001.png  → MC::01
test /MC/0/0100001.png  → MC::01
```

于是认为这是同一个源视频跨到了 train/test，按数据泄漏直接终止。`main` 当前的 strict audit 确实会无条件拒绝 `source_video_uid` 跨 split；同时生产 Recipe 还额外启用了 `require_unique_video_keys_across_splits: true`，所以你同时看到了两条错误。

**如果你能够确认 train/test 中这些同编号视频实际上是不同的原始视频，那么这是“ID namespace 冲突”，不是实际数据泄漏。当前代码还没有完整支持这种情况，短期应重新编号；长期应给 source identity 增加数据源 namespace。**

---

## 先说明：哪些修改不能真正解决

### 1. 只把这个设成 false 不够

```yaml
data:
  require_unique_video_keys_across_splits: false
```

它只能消掉：

```text
train/test share video keys
```

但是下面这一条仍然会报：

```text
source videos span the train/test splits: ['MC::01', ...]
```

因为 `source_video_uid_overlap` 是 strict audit 的无条件 fatal 检查。

所以**不要只关这个开关**。

### 2. 改成 `game_label_video` 也不一定解决

最新 Step7 已经支持：

```yaml
data:
  source_video_identity:
    mode: game_video
```

以及：

```yaml
data:
  source_video_identity:
    mode: game_label_video
```

两者对应：

```text
game_video:
MC::01

game_label_video:
MC::0::01
MC::1::01
```

但你现在出现的是：

```text
train: MC / label=0 / video=01
test : MC / label=0 / video=01
```

即便切成 `game_label_video`，仍然都是：

```text
MC::0::01
```

所以依然冲突。

---

# 你现在最应该做的

先确认一个事实：

> `train/MC/0/01` 和 `test/MC/0/01` 到底是不是同一个原始录屏？

这是唯一关键判断。

### 情况 A：确实来自同一个原始视频

例如同一个 `01.mp4` 的不同时间段分别被放进 train/test。

那**框架报错完全正确**。

不能通过改配置绕过。应该把整个源视频放进同一个 split：

```text
01.mp4 的所有帧
        ↓
全部 train

或者

全部 val

或者

全部 test
```

否则连续视频帧高度相关，validation/test 指标会虚高。

---

### 情况 B：实际上是两个不同视频，只是分别从 01 开始编号

结合你这次错误一次出现：

```text
MC label0: 01~09
MC label1: 01~04
```

我认为**这种可能性相当高**。

例如真实情况可能是：

```text
train:
    MC/0/01 → train_negative_video_A.mp4

test:
    MC/0/01 → test_negative_video_X.mp4
```

它们完全没有关系，只不过每个数据目录自己的编号都从 `01` 开始。

那么当前框架：

```text
(game, video_id)
```

作为全数据集 source identity 的假设就不适用于你的数据。

---

# 一、短期解决方案：重新编号 test 视频

这是当前 `main` 下最安全、最少改代码的办法。

如果 MC train 已经：

```text
01
02
...
09
```

那么 test 可以改为：

```text
10
11
12
13
...
```

例如：

```text
原：

test/MC/0/0100000.png
test/MC/0/0100001.png
...

改：

test/MC/0/1000000.png
test/MC/0/1000001.png
...
```

注意文件格式仍然是：

```text
2 位 video_id + 5 位 frame_id
```

所以：

```text
1000000.png
^^
video_id = 10

  ^^^^^
frame_id = 00000
```

框架的生产 filename contract 确实固定为两位 `video_id`。

### 如果 label 0 / label 1 本身也是独立编号

如果：

```text
MC/0/01
```

和：

```text
MC/1/01
```

本身也是两个不同原始视频，那么建议同时使用：

```yaml
data:
  source_video_identity:
    mode: game_label_video
```

此时只需要保证：

```text
(game, label, video_id)
```

跨 train/val/test 唯一。

例如：

```text
train:
MC / 0 / 01~09
MC / 1 / 01~04

test:
MC / 0 / 10~15
MC / 1 / 05~08
```

这是目前框架已经能够正确表达的结构。

---

# 二、修改完一定要重新构建 indexes

不要直接使用现在的：

```text
indexes/audit.json
train_frames.parquet
val_frames.parquet
test_frames.parquet
```

因为 audit 中已经保存了：

```text
source_identity_mode
source_video_uid_overlap
```

而训练时还会检查 audit 使用的 identity mode 是否与当前配置一致，不一致也会要求重建。

建议直接新建 index 目录，例如：

```bash
rm -rf indexes_v2
```

然后重新执行你的数据准备流程，例如自动 train/val 划分时：

```bash
cls-trainer dataset prepare \
    --config configs/recipes/game_cls_production.yaml \
    --train-root /data/train_all \
    --test-root /data/test \
    --val-ratio 0.10 \
    --output-dir indexes_v2
```

如果 CLI 参数与你现在实际使用的命令略有区别，核心原则是不复用旧的 index/audit/manifest。

如果你修改了 train_all 中的 video ID，那么旧的：

```text
split_manifest.parquet
```

也不要复用，因为 dataset fingerprint 已改变。

---

# 三、我更建议项目继续修改：增加 source namespace

重新编号虽然能运行，但我认为**这还不是 CLSTrainer 最终应该采用的设计**。

现在最新 Step7 提供了：

```text
game_video
game_label_video
```

解决了：

> video ID 是否在 label 间共享？

但是没有解决另一个独立问题：

> video ID 是否在不同原始数据池之间共享？

这正是你现在碰到的问题。

---

## 推荐增加第三个身份维度：source namespace

不要让：

```text
split
```

本身成为 identity。

更合理的是定义：

```text
source_namespace
```

例如：

```text
train_pool
holdout_pool
```

于是：

```text
train:
train_pool::MC::0::01

test:
holdout_pool::MC::0::01
```

它们就不会被错误认为是同一个视频。

### 为什么不是直接用 split

不能定义：

```python
source_video_uid = f"{split}::{game}::{video_id}"
```

因为 train/val 是从同一个 `train_all` 自动切出来的。

如果把 split 写进 identity：

```text
train::MC::01
val::MC::01
```

即使真的把同一个视频错误放进 train 和 val，audit 也发现不了。

这会直接破坏防泄漏机制。

---

## 正确结构应该是

```text
source collection
        │
        ├── train_pool
        │       │
        │       └── 自动 split
        │            ├── train
        │            └── val
        │
        └── heldout_pool
                │
                └── test
```

对应 UID：

```text
train/val:
train_pool::MC::01

test:
heldout_pool::MC::01
```

这样：

* train 与 val 仍然可以严格检查同源视频泄漏；
* test 可以合法重用本地编号；
* test 真正来自独立采集池这一事实得到显式记录。

---

# 四、建议的配置设计

我建议下一版增加：

```yaml
data:
  source_video_identity:
    mode: game_label_video

    namespaces:
      train: train_pool
      val: train_pool
      test: heldout_pool
```

如果使用自动 `from_train`：

```yaml
data:
  source_video_identity:
    mode: game_label_video

    namespaces:
      source: train_pool
      test: heldout_pool
```

内部得到：

```python
train/val uid =
    train_pool::MC::0::01

test uid =
    heldout_pool::MC::0::01
```

但必须增加安全检查：

```text
train namespace == val namespace
```

因为 train/val 必须共享 source identity 空间。

对于 test：

```text
test namespace != train namespace
```

只能表示：

> 用户明确声明它来自独立原始视频池。

---

# 五、即便有 namespace，SHA-256 检查仍然不能关闭

namespace 只解决：

```text
不同数据池恰好编号相同
```

不能成为绕过真实泄漏的工具。

当前代码已经规定：

> 相同内容跨 split 永远属于 leakage，即使 duplicate policy 只是 warning/info，也会在 strict audit 阶段强制失败。

这个设计应该保持。

所以未来可以形成两层保护：

```text
第一层：source UID
检测“同一原视频”

第二层：SHA-256
检测“同一帧内容”
```

而且如果有原始视频 manifest，未来最好再增加：

```text
source_file_sha256
```

这比根据两位 `video_id` 推测原视频身份更加可靠。

---

# 六、针对你当前数据，我的建议

根据你给出的错误：

```text
train/test share:
MC label0 01~09
MC label1 01~04

source overlap:
MC::01~09
```

**如果你确认 train/test 是分别准备的、互不重叠的两组原视频，我建议现在这样处理：**

1. 如果 0/1 目录里的相同 video ID 也是不同视频，Recipe 加：

```yaml
data:
  source_video_identity:
    mode: game_label_video
```

2. 将 **test 中每个 `(game,label)` 的 video_id 重新编号，使其与 train/val 不重叠**。

3. 保持：

```yaml
require_unique_video_keys_across_splits: true
require_content_hash_audit: true
```

不要为了通过检查把它们关闭。

4. 删除并完整重建：

```text
indexes/
split_manifest.parquet
audit.json
所有 train/val/test parquet
```

5. 先执行：

```bash
cls-trainer dataset audit \
    --config configs/recipes/game_cls_production.yaml \
    --strict
```

确认：

```text
source_video_uid_overlap = {}
```

之后再启动八卡训练。

---

## 长期修正

我建议 CLSTrainer 再补一个小版本：

> **Step8：source provenance namespace**

因为当前 Step7 已经解决了“label 内/跨 label 的 ID 语义”，但你这个真实数据暴露出还缺了**跨原始数据池的 ID namespace**。

因此这次并不是 strict audit 做得太严格；**严格检查本身是正确的，缺陷在于 source identity 还不足以表达你的数据来源结构。**在当前代码下，重新编号是最安全的立即解决方式；从框架设计上，则应该新增 `source namespace`，而不是关闭 leakage audit。
