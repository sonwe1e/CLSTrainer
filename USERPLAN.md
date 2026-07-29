- 修正性能统计

当前：

train_only_samples/s=3264
wall_samples/s≈163

二者差距过大，是因为 NPU 异步执行。现有 train_only_samples/s 更接近主机提交速度，不是设备真实吞吐。训练器是在 batch 返回后开始累计 active time，但没有在统计前同步 NPU。

建议增加区间吞吐

每个日志区间同步一次，而不是每个 step 同步：

log_interval_start = time.perf_counter()
log_interval_samples = 0

...

log_interval_samples += local_batch_size * world_size

if global_step % log_every == 0:
    if device.type == "npu":
        torch.npu.synchronize()

    now = time.perf_counter()
    interval_seconds = now - log_interval_start
    interval_samples_per_second = (
        log_interval_samples / interval_seconds
    )

    print(
        f"interval_samples/s={interval_samples_per_second:.2f}"
    )

    log_interval_start = now
    log_interval_samples = 0

同时增加：

interval_step_time
data_wait_ratio
learning_rate
grad_norm
evaluation_seconds
checkpoint_seconds
输出 JSONL

增加：

runs/.../train_metrics.jsonl

每一行：

{
  "step": 1200,
  "loss": 0.13,
  "ce": 0.13,
  "threshold_loss": 3.2,
  "threshold_weight": 0.0,
  "interval_samples_per_second": 211.4,
  "data_wait_seconds": 0.28,
  "learning_rate": 0.00099
}

这可以直接用于后续曲线页面、TensorBoard 或训练实验对比。

- 优化 packed backend 的内存复制

当前 packed backend 每读取一帧都会：

array = memmap_slice.reshape(...)
return torch.from_numpy(array.copy())

即每帧额外执行一次完整 CPU copy。

batch 64 每步读取 128 帧，原始数据量约为：

128 × 3 × 208 × 448 ≈ 35.8 MB

可以尝试批量读取，避免每帧 Python 调用和临时 copy。

推荐接口

在 backend 中新增：

def get_many(self, frame_indices):
    output = np.empty(
        (
            len(frame_indices),
            self.channels,
            self.height,
            self.width,
        ),
        dtype=np.uint8,
    )

    # group by shard
    by_shard = defaultdict(list)

    for output_index, frame_index in enumerate(frame_indices):
        shard_id = int(self.shard_ids[frame_index])
        by_shard[shard_id].append(
            (output_index, frame_index)
        )

    for shard_id, items in by_shard.items():
        memory_map = self._map(shard_id)

        for output_index, frame_index in items:
            offset = int(self.offsets[frame_index])
            np.copyto(
                output[output_index].reshape(-1),
                memory_map[
                    offset : offset + self.image_bytes
                ],
            )

    return torch.from_numpy(output)

进一步在 Dataset 实现 __getitems__()，一次处理一个 batch 的 PairRequest，而不是执行 128 次独立 Python 解码。

验收目标不是某个固定数字，而是：

packed backend CPU时间显著下降
host_data_wait降低
吞吐提升且worker内存稳定
