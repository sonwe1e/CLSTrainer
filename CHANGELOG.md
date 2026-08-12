# Changelog

## 0.4.0

- 完整合并 PNG / compressed-video 双后端。
- train delta 改为在线时序增强，val/test 固定 eval delta。
- 新增增量 `transcode_videos.py` 与 DPID lambda=3 流水线。
- 新增 paired augmentation 与 Focal Loss。
- 新增统一 Game × Class × Video balanced sampler。
- 新增 per-game/per-class diagnostics、confidence 与 score histogram。
- runs 新增 `metrics_detail.json`、class curve、best-val/final-test diagnostics。
- `runtime.find_unused_parameters` 默认 true。
- 保留周期 val/test、best/last/full checkpoint 与 HCCL DDP。
