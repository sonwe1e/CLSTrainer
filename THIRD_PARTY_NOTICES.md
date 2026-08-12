# Third-party notices

## DPID

`tools/transcode_videos.py` 中的 DPID 2x 实现是根据 Rapid, Detail-Preserving Image Downscaling (DPID) 的公开算法/参考实现重新向量化实现，用于最后一级 416x896 -> 208x448 下采样。

参考项目：`mergian/dpid`，其参考实现声明为 BSD 3-Clause License。

本仓库没有包含该项目的原始 MATLAB 源文件；这里保留此说明以便后续维护者理解算法来源和实现边界。
