# Validation record — CLSTrainer Lite 0.4.0

本文件记录发布包在当前 CPU 环境可复现的验证范围。真实 Ascend/HCCL 仍需在用户服务器完成。

发布前要求：

```bash
python -m compileall -q src tests tools
python -m pytest -q
bash scripts/demo_cpu.sh
```

覆盖范围包括：

- train 内按完整 video 划 val，train/val video 无重叠；
- ImagePairDataset 在线 delta；
- VideoPairDataset 元数据/统一 sampler 接口；
- paired augmentation；
- CE/Focal；
- 周期 val/test；
- best/last/full checkpoint；
- overall 与 per-class/per-game diagnostics；
- confidence/score histogram 聚合；
- Image/Video 共用 BalancedVideoSampler；
- 真实 2-process Gloo DDP smoke；
- 分布式 val/test 不 padding 重复样本；
- run 曲线/诊断图生成；
- wheel 构建与安装后 CLI smoke。

离线视频转码工具另外验证：

- Torch bilinear -> bilinear -> DPID(lambda=3)；
- 输出 208x448 视频；
- `.transcode_manifest.json`；
- 第二次相同运行 skip；
- 新增源视频仅新增处理；
- 临时文件完成后原子替换。

未声称在本环境完成：真实 Ascend NPU、CANN、torch_npu 与 2P/8P HCCL collective。
