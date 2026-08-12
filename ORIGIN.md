# Origin

CLSTrainer Lite 是基于 `sonwe1e/CLSTrainer` 的架构分析重新实现的精简训练器，不是对原仓库机械删除文件后的残片。

前序分析基准：

- Repository: `sonwe1e/CLSTrainer`
- Branch: `main`
- Reviewed release line: CLSTrainer 5.0.0 / contract 5
- Reviewed commit: `92cd3011753393e10d8f71597053dcb6c0552f26`

保留的核心思想：

- 双帧输入与 `[B,2]` logits；
- train 内按完整 video 切 validation；
- CPU/CUDA/NPU DDP，Ascend 使用 HCCL；
- rank0 负责 checkpoint / history / plot；
- val/test 分布式评估不重复 padding 样本。

Lite 额外演进了 Image/Video 双后端、在线 delta、增量视频转码、Focal、paired augmentation、分游戏诊断和平衡采样，同时刻意不重新引入原项目的 contract/release/audit/packed/mining 等大型生产框架。
