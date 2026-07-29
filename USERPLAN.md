# CLSTrainer 数据规格、索引扫描与审计策略整改计划

更新时间：2026-07-29

## 结论

本轮整改将索引模块中混合的三类职责拆开：

1. `ImageSpec` 只描述图像宽、高、通道和 tensor 布局；
2. `ScanPolicy` 只描述候选帧发现、目录剪枝和辅助文件忽略规则；
3. `DuplicatePolicy` 只描述重复内容与同名文件的审计严重程度。

训练、索引、审计和 packed 数据必须共同读取同一份配置。任何生产函数不得再携带
`208`、`448` 或 `3` 这类业务尺寸默认值，也不得自动交换宽高或静默 resize。

当前真实规格固定为：

```yaml
data:
  width: 448
  height: 208
  channels: 3
```

对应张量契约为：

```text
单帧：[3, 208, 448]
训练 batch：[B, 2, 3, 208, 448]
模型输出：[B, 2]
```

当前实施状态：

- 代码、配置、工具入口、README 和教程修改已完成；
- 55 项 CPU 测试、Python 编译检查和差异格式检查已通过；
- 真实数据索引与 packed 产物必须在取得实际数据路径后重建；
- 910B2 单卡、八卡 smoke 仍须在真实 NPU 环境执行。

---

## 一、目标配置

生产配置统一维护以下内容：

```yaml
data:
  width: 448
  height: 208
  channels: 3

  frame_extensions:
    - ".png"
  ignore_directory_prefixes:
    - "_"
    - "."
  ignore_directory_names:
    - "__pycache__"
    - "cache"
    - "caches"
    - "tmp"
    - "temp"
  ignore_file_globs:
    - "*.tmp"
    - "*.part"
    - "*.log"
  unexpected_nested_directory_severity: warning
  ignored_example_limit: 20

  duplicate_policy:
    same_label_cross_split: warning
    same_label_within_split: warning
    cross_label_same_content: error
    same_basename: info

  strict_audit: true
  require_content_hash_audit: true
```

配置通过现有 `load_config()` 加载，并继续支持 base 配置和 `key=value` 点号覆盖。

---

## 二、实施阶段

### 阶段 1：统一图像规格

- 新增 `src/game_cls/data/image_spec.py`。
- `build_index.py`、`pack_dataset.py` 和训练器从 `config["data"]` 创建同一个
  `ImageSpec`。
- `scan_split()`、`make_audit()`、`pack_frame_index()` 和
  `PackedUint8Backend` 必须显式接收图像规格。
- packed manifest 必须记录配置中的 `width=448`、`height=208` 和
  `channels=3`。
- packed manifest 与运行配置不一致时立即报错。
- 训练首个 batch 必须严格验证 `[B,2,3,208,448]`。

### 阶段 2：结构化扫描候选帧

- 删除无条件 `root.rglob("*")`。
- 按 `<game>/<0|1>/<frame>` 三层结构扫描。
- 非帧扩展名只计入 ignored，不进入错误列表。
- `_`、`.` 前缀和命名忽略目录执行整目录剪枝。
- 直接位于标签目录中的非法命名 PNG 保持 error。
- 含帧文件的非法标签目录保持 error。
- 非忽略的嵌套目录按配置记录 warning 或 error。
- ignored 只保存分类计数及每类有限路径示例。

### 阶段 3：按语义分析重复数据

- SHA-256 使用 `hash -> list[FrameRecord]` 分组，不再覆盖重复记录。
- 相同内容且标签不同：`error`，严格审计阻止训练。
- 相同内容、相同标签、跨 split：`warning`。
- 相同内容、相同标签、split 内重复：`warning`。
- 相同 basename 只输出 `info` 汇总，不作为样本身份依据。
- 两位 `video_id` 继续默认按 split 内编号；只有明确全局唯一时才开启跨 split
  video key 强制检查。

### 阶段 4：重构 severity 与严格审计

审计 findings 统一为：

```json
{
  "errors": [],
  "warnings": [],
  "info": [],
  "ignored": {
    "counts": {},
    "examples": {}
  }
}
```

`validate_audit()` 只因以下情况失败：

- 扫描或重复数据 findings 中存在 `errors`；
- 图片规格、审计格式或重复策略与当前配置不一致；
- 缺少必要标签或合法 pair；
- 配置要求内容 hash，但索引未执行 hash；
- 显式启用全局 video key 唯一检查且发现冲突。

warnings 只输出汇总，不阻止训练。

### 阶段 5：统一工具入口并迁移数据产物

推荐命令：

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

python tools/pack_dataset.py \
  --config configs/npu_production.yaml \
  --frame-index indexes/train_frames.parquet \
  --output-dir /local_nvme/train_packed

python tools/pack_dataset.py \
  --config configs/npu_production.yaml \
  --frame-index indexes/test_frames.parquet \
  --output-dir /local_nvme/test_packed
```

需要临时覆盖尺寸时只修改配置：

```bash
python tools/build_index.py \
  --config configs/npu_production.yaml \
  --train-root /data/train \
  --test-root /data/test \
  data.width=448 \
  data.height=208
```

旧 indexes 和 packed 数据不允许继续复用。尺寸、扫描策略、重复策略或审计格式变化
后，必须更换输出目录或删除旧产物，再依次重新 build、audit 和 pack。

---

## 三、验收矩阵

| 情况 | 预期结果 |
| --- | --- |
| 448×208 RGB | 进入索引 |
| 208×448 RGB | `unexpected_dimensions` error |
| 修改为其他合法配置尺寸 | 无需改代码即可通过 |
| `gameA/0/video01.mp4` | ignored |
| `gameA/0/meta.json` | ignored |
| `gameA/0/_cache/thumb.png` | ignored directory |
| `gameA/_cache/foo.png` | ignored directory |
| `gameA/0/bad_name.png` | error |
| `gameA/2/0100001.png` | error |
| `gameA/0/random_dir/file.png` | warning 或配置为 error |
| 相同 hash、相同 label、跨 split | warning，不阻断 |
| 相同 hash、不同 label | error，strict 阻断 |
| 相同文件名、不同内容 | info，不阻断 |
| packed manifest 为 448×208×3 | 后端输出 `[3,208,448]` |
| packed manifest 与配置不一致 | 初始化立即报错 |
| 首批训练输入宽高颠倒 | 训练立即报错 |

---

## 四、验证要求与边界

代码级验收必须包括：

- ImageSpec 和训练 batch 形状测试；
- 分层扫描、目录剪枝及 ignored 限量示例测试；
- 重复内容 severity 矩阵测试；
- strict audit 只拒绝 errors 的测试；
- packed manifest、输出 tensor 和配置不一致测试；
- 完整 CPU pytest 与训练 smoke test；
- README 和 `tutorial.html` 命令同步。

真实数据迁移不在本地合成测试中伪造完成。获得实际 train/test 路径后，还必须：

1. 在本地 NVMe 上重新生成索引；
2. 检查 audit warnings 和 errors；
3. 重新生成 train/test packed 数据；
4. 运行 CPU/CUDA smoke；
5. 在 910B2 上执行 `scripts/smoke_npu_1p.sh` 和
   `scripts/smoke_npu_8p.sh`。

只有真实数据重建和 NPU smoke 通过后，才能启动正式八卡长跑。
