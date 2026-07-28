# CLSTrainer

双帧、多游戏二分类训练框架。当前实现覆盖 `USERPLAN.md` 的首个 CUDA
正确性闭环：数据索引、合法配对、一致增强、三级平衡采样、`cls` 参数冻结、
固定 `0.99` 阈值损失、测试指标以及双文件 checkpoint。

## 快速开始

推荐使用 Python 3.10～3.12，并安装与本机 CUDA 匹配的 PyTorch，然后执行：

```bash
python -m pip install -e .[dev]
python -m pytest
python tools/train.py --config configs/cuda_debug.yaml train.max_steps=100
```

默认 CUDA 调试配置使用合成样本和内置小模型，用于验证训练闭环。接入真实模型时，
将 `model.factory` 设置为 `包名.模块名:函数名`；工厂函数必须返回接受
`(image0, image1)` 并输出 `[B, 2]` 的 `torch.nn.Module`。

真实数据索引：

```bash
python tools/build_index.py \
  --train-root /data/train \
  --test-root /data/test \
  --output-dir indexes
```

目录必须是 `<split>/<game>/<0|1>/<video_id><frame_id>.png`，其中文件名满足
`^\d{2}\d{5}\.png$`。索引不会假定帧号连续，只有目标帧真实存在时才形成 pair。

