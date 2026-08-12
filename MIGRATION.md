# Migration to CLSTrainer Lite 0.4.0

0.4.0 是当前整理后的完整版本，核心变化如下。

## 数据根目录

推荐新结构：

```yaml
data:
  backend: image
  image:
    train_root: /data/train
    test_root: /data/test
```

0.3 的 `data.train_root / data.test_root / data.strict_filenames` 仍有兼容迁移，但新配置不要继续使用旧写法。

## Delta

训练 delta 已改为在线采样：

```yaml
data:
  train_delta_range: [1, 3]
  train_delta_probabilities: [0.15, 0.70, 0.15]
  eval_delta: 2
```

不再为每个 delta 预展开 PairPosition。旧 `data.delta: N` 仍会自动迁移为固定 delta。

## 双后端

- PNG：`ImagePairDataset`
- 压缩视频：`VideoPairDataset`

通过 `data.backend: image|video` 切换。图片后端的旧 `PairDataset` 名称保留临时 alias，但新代码应使用 `ImagePairDataset`。

## 新诊断

新增 `metrics_detail.json`、`class_metrics_curve.png`、`best_val_diagnostics.png`、`final_test_diagnostics.png`，包含 per-game/per-class confusion、F1/Recall/Precision 与置信度统计。

## 新平衡采样

```yaml
sampler:
  enabled: true
  class_probability: [0.5, 0.5]
  game_balance_alpha: 0.5
```

Image/Video 共用一个 sampler。

## DDP

新增：

```yaml
runtime:
  find_unused_parameters: true
```

默认 true；确认模型每步所有 trainable 参数都参与 loss 后可关闭。
