# 核心方案

根因已经确认：当前 NPU runtime 在 DataLoader worker 启动前已经初始化，而 DataLoader 未显式指定启动方式，最终导致多进程 worker 与父进程设备状态冲突并长期等待。

修复不能只增加一个 `spawn` 参数。生产化方案应同时完成：

> **NPU 强制 spawn、DataLoader 超时、训练/评估 worker 分离、可序列化检查、首 batch 日志，以及 1P→8P 分阶段验收。**

当前 `_loader_common()` 只设置了 worker 数、pin memory、persistent 和 prefetch，没有 `multiprocessing_context` 与 `timeout`；同一份参数又被 train、quick test 和 full test 共用。 当前生产配置是 4 个 persistent worker 和 `pin_memory=true`。

---

# 第一阶段：立即恢复可运行状态

先使用单进程 DataLoader 验证模型训练本身正常：

```bash
pkill -f "tools/train.py" || true

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=1

python -u tools/train.py \
  --config configs/npu_production.yaml \
  train.max_steps=5 \
  train.steps_per_epoch=5 \
  train.local_batch_size=2 \
  train.log_every_steps=1 \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  dataloader.num_workers=0 \
  dataloader.pin_memory=false \
  augmentation.enabled=false \
  experiment.output_dir=runs/npu_single_process_baseline
```

这个阶段必须满足：

```text
step=1/5
...
step=5/5
```

并确认：

* 输入为 `[B,2,3,208,448]`；
* forward、backward、optimizer 均完成；
* 没有 NaN/Inf；
* 只有一张 NPU 有当前进程；
* AI Core 在训练阶段有活动。

若这个基线仍失败，说明除了 DataLoader 外还有模型或 NPU 算子问题；若成功，则继续代码改造。

---

# 第二阶段：改造 DataLoader 配置结构

## 1. 不再让三个 DataLoader 共用同一组 worker 参数

当前 train、quick test、full test 都使用同一个 `common`。这会带来新问题：在八卡训练中，如果三个 DataLoader 都是 4 个 persistent worker，理论上最多可能保留：

```text
8 ranks × 3 DataLoaders × 4 workers = 96 workers
```

PyTorch 在 `num_workers>0` 时会为 DataLoader iterator 创建 worker，而 `persistent_workers=true` 会在一次数据集遍历完成后继续保留 worker。([PyTorch Docs][1])

建议改成分角色配置：

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

设计逻辑：

* train worker 长期使用，所以 persistent；
* quick/full 间隔较长，不应长期保留 eval worker；
* 8 卡时，2 个 train worker/rank 等于 16 个常驻 worker；
* 评估时额外启动 1 个 eval worker/rank，总量约 24 个；
* 初始阶段关闭 pin memory，减少设备相关变量；
* worker timeout 防止再次静默等待 20 分钟。

---

## 2. 保留旧配置兼容性

配置读取逻辑应支持：

```yaml
dataloader:
  num_workers: 4
```

作为旧配置 fallback，同时优先读取：

```yaml
dataloader:
  train:
    num_workers: 2
```

这样 CUDA debug 和旧实验配置不会立即失效。

---

# 第三阶段：修改 `_loader_common()`

在 `src/game_cls/engine/trainer.py` 顶部增加：

```python
from functools import partial
```

增加模块级 worker 初始化函数。必须放在模块顶层，不能使用 lambda 或局部函数，因为 spawn 需要 pickle：

```python
def _initialize_data_worker(
    worker_id: int,
    *,
    num_threads: int,
) -> None:
    import torch

    # 防止每个 worker 又创建大量 OpenMP/PyTorch CPU 线程。
    torch.set_num_threads(max(1, int(num_threads)))
```

PyTorch 官方建议限制每个子进程内部线程数，避免 worker 数与 CPU 线程数相乘造成过度并发。([PyTorch Docs][2])

将 `_loader_common()` 改成：

```python
def _loader_common(
    config: dict,
    role: str,
) -> dict:
    dataloader_cfg = config["dataloader"]
    role_cfg = dataloader_cfg.get(role, {})

    def get_option(name: str, default):
        return role_cfg.get(
            name,
            dataloader_cfg.get(name, default),
        )

    workers = int(get_option("num_workers", 0))
    if workers < 0:
        raise ValueError(
            f"dataloader.{role}.num_workers must be non-negative"
        )

    pin_memory = bool(get_option("pin_memory", False))

    common = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": pair_collate,
    }

    # 这些参数只能在 num_workers > 0 时传给 DataLoader。
    if workers == 0:
        return common

    accelerator = str(config["device"]["accelerator"])
    context = get_option("multiprocessing_context", None)

    # NPU 默认强制使用 spawn。
    if context is None and accelerator == "npu":
        context = "spawn"

    if context is not None:
        context = str(context)

    allowed_contexts = {"spawn", "fork", "forkserver"}
    if context is not None and context not in allowed_contexts:
        raise ValueError(
            "dataloader multiprocessing_context must be one of "
            f"{sorted(allowed_contexts)}, got {context!r}"
        )

    # 当前项目中 NPU 明确禁止退回 fork。
    if accelerator == "npu" and context != "spawn":
        raise RuntimeError(
            "NPU DataLoader with num_workers > 0 must use "
            "multiprocessing_context=spawn"
        )

    timeout = float(get_option("timeout_seconds", 180))
    if timeout <= 0:
        raise ValueError(
            f"dataloader.{role}.timeout_seconds must be positive "
            "when num_workers > 0"
        )

    prefetch_factor = int(get_option("prefetch_factor", 2))
    if prefetch_factor <= 0:
        raise ValueError(
            f"dataloader.{role}.prefetch_factor must be positive"
        )

    worker_num_threads = int(
        get_option("worker_num_threads", 1)
    )
    if worker_num_threads <= 0:
        raise ValueError(
            "dataloader.worker_num_threads must be positive"
        )

    common.update(
        {
            "persistent_workers": bool(
                get_option(
                    "persistent_workers",
                    role == "train",
                )
            ),
            "prefetch_factor": prefetch_factor,
            "timeout": timeout,
            "multiprocessing_context": context,
            "worker_init_fn": partial(
                _initialize_data_worker,
                num_threads=worker_num_threads,
            ),
        }
    )

    return common
```

`DataLoader` 原生支持 `multiprocessing_context`、`timeout`、`prefetch_factor` 和 `persistent_workers`；其中 timeout 表示等待 worker 返回一个 batch 的最长时间。([PyTorch Docs][1])

---

# 第四阶段：分别构建 train 和 eval DataLoader

在 `_make_dataloaders()` 中，将：

```python
common = _loader_common(config)
```

改为：

```python
train_common = _loader_common(
    config,
    role="train",
)

eval_common = _loader_common(
    config,
    role="eval",
)
```

所有训练 DataLoader 使用：

```python
train=DataLoader(
    train_dataset,
    batch_sampler=sampler,
    **train_common,
)
```

quick/full 使用：

```python
quick_test=DataLoader(
    quick_dataset,
    batch_size=batch_size,
    **eval_common,
)

full_test=DataLoader(
    full_dataset,
    batch_size=batch_size,
    **eval_common,
)
```

合成数据分支也要同步修改，不能只修改真实数据分支。

---

# 第五阶段：增加配置安全校验

在 `validate_training_config()` 中增加 DataLoader 校验，避免生产配置再次回退到危险组合。

```python
def _validate_dataloader_config(config: dict) -> None:
    accelerator = str(config["device"]["accelerator"])
    root = config["dataloader"]

    for role in ("train", "eval"):
        scoped = root.get(role, {})

        def get_option(name, default):
            return scoped.get(name, root.get(name, default))

        workers = int(get_option("num_workers", 0))
        if workers < 0:
            raise ValueError(
                f"dataloader.{role}.num_workers must be non-negative"
            )

        if workers == 0:
            continue

        context = get_option(
            "multiprocessing_context",
            "spawn" if accelerator == "npu" else None,
        )

        if accelerator == "npu" and context != "spawn":
            raise RuntimeError(
                f"NPU dataloader.{role} must use spawn, got {context!r}"
            )

        if float(get_option("timeout_seconds", 180)) <= 0:
            raise ValueError(
                f"dataloader.{role}.timeout_seconds must be positive"
            )

        if int(get_option("prefetch_factor", 2)) <= 0:
            raise ValueError(
                f"dataloader.{role}.prefetch_factor must be positive"
            )
```

然后在 `validate_training_config()` 开头调用：

```python
_validate_dataloader_config(config)
```

这样即使有人后来写回：

```yaml
multiprocessing_context: fork
```

训练也会在初始化设备之前明确终止。

PyTorch 官方也明确提醒，加速器初始化与后续 fork 容易发生冲突；多 worker 与 DDP 一起使用时应采用 `spawn` 或 `forkserver`，而当前已确认你的 NPU 环境应固定为 `spawn`。([PyTorch Docs][3])

---

# 第六阶段：增加启动阶段日志

当前程序打印 `Data pipeline` 后就进入首 batch 等待，没有任何进度提示。需要增加明确日志。

在进入训练循环前打印：

```python
if rank == 0:
    train_loader_cfg = config["dataloader"].get(
        "train",
        config["dataloader"],
    )
    eval_loader_cfg = config["dataloader"].get(
        "eval",
        config["dataloader"],
    )

    print(
        "[DATALOADER] "
        f"train_workers={train_loader_cfg.get('num_workers', 0)} "
        f"eval_workers={eval_loader_cfg.get('num_workers', 0)} "
        f"context={config['dataloader'].get('multiprocessing_context')} "
        f"timeout={config['dataloader'].get('timeout_seconds')}",
        flush=True,
    )
```

在第一次进入训练 iterator 前：

```python
first_batch_wait_started = time.perf_counter()

if rank == 0:
    print(
        "[DATALOADER] starting train workers and waiting for first batch",
        flush=True,
    )
```

循环拿到第一个 batch 后：

```python
if global_step == 0 and rank == 0:
    first_batch_wait = (
        time.perf_counter() - first_batch_wait_started
    )
    print(
        "[DATALOADER] first batch ready: "
        f"wait={first_batch_wait:.3f}s "
        f"shape={tuple(batch['images'].shape)} "
        f"dtype={batch['images'].dtype}",
        flush=True,
    )
```

预期日志：

```text
[DATALOADER] train_workers=2 eval_workers=1 context=spawn timeout=180
[DATALOADER] starting train workers and waiting for first batch
[DATALOADER] first batch ready: wait=8.421s shape=(8,2,3,208,448) dtype=torch.uint8
```

若 worker 再次异常，最多 180 秒后会抛出 DataLoader timeout，而不是无限等待。

---

# 第七阶段：检查 spawn 可序列化约束

`spawn` 会启动一个新的 Python 解释器，只继承必要资源，不会复制父进程全部运行时状态；代价是启动较慢，并要求传给 worker 的对象可以被 pickle。([Python documentation][4])

必须检查以下内容。

## 可以保留

这些当前结构通常是安全的：

```text
PairRequest dataclass
VideoEntry dataclass
NumPy frame arrays
模块顶层的 _decode_image_png
模块顶层的 pair_collate
torchvision v2 transform 对象
PackedUint8Backend 类实例
```

## 必须禁止

模型和数据模块中不能出现：

```python
decoder = lambda path: ...
```

不能把局部函数放进 Dataset：

```python
def build_dataset():
    def decoder(path):
        ...
```

不能在模块 import 阶段执行：

```python
torch.npu.set_device(0)
model = Model().npu()
run_training(...)
```

模型工厂应保持为：

```python
def build_model(config):
    return Model(...)
```

`tools/train.py` 必须继续保留：

```python
if __name__ == "__main__":
    main()
```

PyTorch 官方也明确指出，spawn 模式下 `worker_init_fn` 等对象必须可 pickle，lambda 不可用。([PyTorch Docs][1])

---

# 第八阶段：增加自动化测试

## 1. 配置单元测试

新增：

```text
tests/test_dataloader_policy.py
```

测试：

```python
def test_npu_loader_defaults_to_spawn():
    config = {
        "device": {"accelerator": "npu"},
        "dataloader": {
            "train": {
                "num_workers": 2,
                "persistent_workers": True,
            }
        },
    }

    options = _loader_common(config, "train")

    assert options["multiprocessing_context"] == "spawn"
    assert options["timeout"] == 180
    assert options["persistent_workers"] is True
```

以及：

```python
def test_npu_loader_rejects_fork():
    config = {
        "device": {"accelerator": "npu"},
        "dataloader": {
            "multiprocessing_context": "fork",
            "train": {"num_workers": 1},
        },
    }

    with pytest.raises(RuntimeError):
        _loader_common(config, "train")
```

## 2. CPU spawn 集成测试

Dataset 和 collate 必须定义在测试文件模块顶层：

```python
class SpawnDataset:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        import torch

        return {
            "images": torch.zeros(
                2, 3, 8, 8,
                dtype=torch.uint8,
            ),
            "label": index % 2,
            "meta": {"index": index},
        }
```

测试：

```python
def test_spawn_loader_returns_batch():
    from torch.utils.data import DataLoader

    loader = DataLoader(
        SpawnDataset(),
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        timeout=20,
        collate_fn=pair_collate,
    )

    batch = next(iter(loader))

    assert tuple(batch["images"].shape) == (2, 2, 3, 8, 8)
```

## 3. 真实 PNG Dataset spawn 测试

使用临时目录生成合法帧：

```text
gameA/0/0100000.png
gameA/0/0100001.png
gameA/0/0100002.png
```

然后测试：

* `LazyTrainingPairDataset`；
* `VideoBalancedPairBatchSampler`；
* `num_workers=1`；
* `multiprocessing_context=spawn`；
* augmentation 开启；
* 首 batch 在 20 秒内返回。

## 4. packed backend 测试

再增加：

* pack 临时 PNG；
* packed DataLoader 使用 spawn；
* 每个 worker 独立建立 memmap；
* 首 batch shape 正确；
* worker 退出后没有异常。

---

# 第九阶段：单卡验收顺序

## 1. spawn + 1 worker，关闭增强和评估

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
unset ASCEND_LAUNCH_BLOCKING

python -u tools/train.py \
  --config configs/npu_production.yaml \
  train.max_steps=10 \
  train.steps_per_epoch=10 \
  train.local_batch_size=2 \
  train.log_every_steps=1 \
  dataloader.train.num_workers=1 \
  dataloader.eval.num_workers=0 \
  augmentation.enabled=false \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  experiment.output_dir=runs/npu_spawn_1worker
```

验收：

```text
首batch返回
10 step完成
没有timeout
AI Core有活动
```

## 2. spawn + 2 worker，开启增强

```bash
python -u tools/train.py \
  --config configs/npu_production.yaml \
  train.max_steps=50 \
  train.steps_per_epoch=50 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=0 \
  augmentation.enabled=true \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  experiment.output_dir=runs/npu_spawn_2workers
```

记录：

```text
first_batch_wait
host_data_wait
train_only_samples/s
wall_samples/s
主进程RSS
worker RSS
```

## 3. 加回 quick test

```bash
python -u tools/train.py \
  --config configs/npu_production.yaml \
  train.max_steps=20 \
  train.steps_per_epoch=20 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=1 \
  evaluation.quick_test_every_steps=10 \
  evaluation.quick_test_pairs_per_video=2 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  experiment.output_dir=runs/npu_spawn_quick_eval
```

## 4. 最后才执行 full test

```bash
python -u tools/train.py \
  --config configs/npu_production.yaml \
  train.max_steps=20 \
  train.steps_per_epoch=20 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=1 \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=20 \
  evaluation.full_test_at_end=false \
  experiment.output_dir=runs/npu_spawn_full_eval
```

---

# 第十阶段：八卡放量

八卡第一轮不要使用 2～4 个 worker/rank，先从 1 个开始：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  tools/train.py \
  --config configs/npu_8p.yaml \
  train.max_steps=100 \
  train.steps_per_epoch=100 \
  train.local_batch_size=8 \
  train.log_every_steps=10 \
  dataloader.train.num_workers=1 \
  dataloader.eval.num_workers=1 \
  evaluation.quick_test_every_steps=50 \
  evaluation.quick_test_pairs_per_video=2 \
  evaluation.full_test_every_steps=100 \
  evaluation.full_test_at_end=false \
  experiment.output_dir=runs/npu_8p_spawn_smoke
```

验收以下内容：

```text
8个rank都启动成功
每个rank只有自己的NPU
每个rank启动1个train worker
没有DataLoader timeout
step一致
quick/full无死锁
checkpoint正常
```

通过后再测试：

```yaml
dataloader:
  train:
    num_workers: 2
```

不要直接使用 4 worker/rank。PyTorch 多进程 DataLoader 会复制 worker 访问到的父进程 Python 对象，worker 越多，CPU 内存越高。([PyTorch Docs][1])

---

# 第十一阶段：最后做性能调优

稳定后再逐项 A/B，不要同时修改。

建议顺序：

```text
train workers：1 → 2 → 4
prefetch_factor：2 → 4
pin_memory：false → true
PNG → packed_uint8
local_batch_size逐步增加
```

每组至少跑 500 step，忽略冷启动阶段，记录：

```text
first_batch_wait
train_only_samples/s
wall_samples/s
host_data_wait
CPU利用率
每worker RSS
NPU利用率
step P50/P95
```

`pin_memory=true` 不应默认恢复。只有在 NPU 环境下实测 H2D 和吞吐确实改善、且没有稳定性问题时再开启。

---

# 最终修改范围

建议此次提交至少包含：

```text
configs/npu_production.yaml
src/game_cls/engine/trainer.py
tests/test_dataloader_policy.py
tests/test_spawn_dataloader.py
tests/test_packed_backend.py
scripts/smoke_npu_1p.sh
scripts/smoke_npu_8p.sh
README.md
```

完成标准是：

```text
NPU + num_workers>0 必须使用spawn
worker最长等待时间有限
train/eval worker参数分离
eval worker不长期驻留
首batch有明确日志
CPU spawn测试通过
1P train/quick/full通过
8P train/quick/full通过
```

这次修复的重点不只是消除当前死锁，而是确保未来任何配置都无法再次无意中回退到 NPU 初始化后 fork DataLoader worker 的危险路径。

[1]: https://docs.pytorch.org/docs/stable/data.html?utm_source=chatgpt.com "torch.utils.data — PyTorch 2.13 documentation"
[2]: https://docs.pytorch.org/docs/stable/notes/multiprocessing.html?utm_source=chatgpt.com "Multiprocessing best practices — PyTorch 2.13 documentation"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?utm_source=chatgpt.com "DistributedDataParallel — PyTorch 2.12 documentation"
[4]: https://docs.python.org/3/library/multiprocessing.html?utm_source=chatgpt.com "multiprocessing — Process-based parallelism — Python 3.14.6 documentation"
