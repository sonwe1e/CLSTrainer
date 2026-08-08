# CLSTrainer 5.0.0 发布说明

CLSTrainer 5.0.0 是当前唯一受支持的发布合同。项目生成和读取的 JSON、
checkpoint 与 Parquet 产物必须声明 `contract_version=5`。

## 发布合同

- JSON 与 checkpoint 顶层字段为 `contract_version: 5`。
- Parquet schema metadata 包含 `game_cls.contract_version=5`。
- 缺失、类型错误或值不为 `5` 的持久化产物会立即被拒绝。
- 数据索引、sidecar、packed 数据、mining 结果、benchmark 报告、
  checkpoint、Run 与导出产物必须由本版本重新生成。
- 配置只接受配置参考中列出的当前键；未知键、已移除命令与非规范别名不会被转换。

## 主要变化

- 统一数据、训练、评估、benchmark、release 与导出的身份合同。
- 新训练始终创建唯一 Run 目录；精确 resume 只写回所选 Run。
- validation 与 test 完全分离，test 不参与选模或 early stopping。
- `model.trainable_rules` 成为唯一的可训练参数规则，并绑定 checkpoint identity。
- benchmark gate 必须使用显式 `{op, value}` 结构。
- export 产物按 `<run>/<checkpoint_sha>/<format>` 布局，失败时清理空目录。
- 移除数据收口和元数据转换命令；数据必须通过当前 prepare/annotate/pack 流程生成。

## 发布前验证

```bash
python -m pytest -q
ruff check src tools tests
python -m compileall -q src tools
python -m pip wheel . -w dist --no-deps
cls-trainer --version
```

生产发布还应使用真实配置依次运行 `config validate`、`doctor`、训练、
独立 test 评估、challenge benchmark、`release check` 与目标格式导出。
