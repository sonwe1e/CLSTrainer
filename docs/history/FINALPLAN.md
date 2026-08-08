## 核心结论

我审查的是当前 `main` 最新提交 **`b0fa3cf3` — `fix: close release and packed identity contracts`**。结论是：**这个提交的设计方向是正确的，但当前 `main` 还不适合标记为 release-ready。** 我确认了 **3 个项目级 P0、1 个功能级 P0，以及一批 P1/P2 契约和 UX 问题**。尤其需要优先处理 release identity、packed identity、subtype evaluation 和 resume identity 四条链。最新 CI 的 CPU checks 是绿的，因此这些问题主要属于**测试没有覆盖真实集成路径**，而不是已有 CI 已经暴露出来的失败。

最危险的共同模式和之前 `step1–8` 很一致：**Schema/文件格式已经“声明支持”，但真实热路径没有消费；身份字段已经“记录”，但没有验证当前字节真的属于该身份；单元测试验证了局部函数，却绕过 CLI entrypoint、collate、真实 pack CLI 等 glue code。**

### 审查结果

1. **[P0｜本次提交直接回归] `cls-trainer release check` 实际不可达。** `parser.py` 已增加顶层 `release` 子命令，但 `cli/init.py::main()` 用另一份手写 known-command 集合做 implicit-train 路由，其中漏了 `"release"`。所以 README 推荐的 `cls-trainer release check ...` 会先被改写成 `cls-trainer train release check ...`，随后 argparse 失败。新 release gate 的显式 CLI 等于发布后不可用。修复不应只加一个字符串，建议直接消灭“双命令注册表”，让 implicit-train 根据 parser 或首参数形态判断。

2. **[P0｜本次提交核心契约未闭合] challenge fingerprint 并没有验证当前 PNG 内容。** `challenge_bundle_fingerprint()` 的注释明确声称 frame index 中存在 `content_sha256` 后，“hashing the index file detects an edited PNG”，但这是不成立的：代码只 SHA-256 parquet 文件本身，再检查 `content_sha256` 列是否存在；**没有重新读取当前 PNG 并和 row 中的 content hash 比较**。所以即使 `covers_content=True`，在 indexing 后原地替换 PNG，parquet 完全不变，release identity 也完全不变。PNG challenge loader 同样没有做这个验证。也就是说最新提交试图关闭的 P0-4 仍然可以被最直接的 TOCTOU 绕过。

3. **[P0｜本次提交核心契约未闭合] packed v4 可以把“新像素”包装成“旧 `content_version_id`”。** packer 实际打开 `row["path"]` 读取当前 PNG 并写 shard，但 `packed_backend.py` 根本没有消费 frame row 的 `content_sha256`。因此 indexing/seal 完成后，如果某张 PNG 被替换，`verify_split_bundle()` 仍只验证索引产物自身的 hash，packer 随后会把修改后的像素写入 shard，却从旧 source video index 继承旧 `content_version_id`。最终得到一个 shard SHA、manifest SHA 都自洽，但语义 identity 是假的 packed dataset。这对最新提交的 stable/content 双身份设计属于根本性漏洞。

4. **[P0-Feature / P1-Project｜历史问题，本次复查确认] 开启 `evaluation.group_by_negative_subtype=true` 后真实评估会 `KeyError`。** `EvalPairDataset` 会生成 `game_label_subtype_id`，但通用 `pair_collate()` 只保留 `game_id / game_label_id / video_group_id`，把 subtype id 丢掉；evaluator 发现 subtype catalog 后却会读取 `batch["game_label_subtype_id"]`。所以真实 DataLoader 第一批就可能崩。现有 subtype end-to-end 测试手工构造了带这个字段的 batch，正好绕开了生产 collate。

5. **[P1｜本次提交] mining v2 的 `source_version_id` 被写进文件，却没有真正参与 annotate 安全判断。** `scan_negative_pool()` 和 mining v2 manifest 都正确保存 `stable_source_id + source_version_id`；但 `dataset annotate --from-mining` 的 `_mining_rows()` 按 `stable_source_id` 聚合时直接丢掉 `source_version_id`，后续只验证 stable ID 是否仍存在。因此视频像素更新、重新生成 content version 后，**旧画面上的 hard-negative mining 结果仍然可以贴到新画面上**。新增 version 字段变成了审计信息，而不是保护条件。

6. **[P1｜本次提交] v2 strict identity 可以通过 frame-index fallback 绕开。** loader 在 `train/val/test_video_index` 未配置时明确回退到 frame index；而 `read_video_entries_parquet()` 对 frame parquet 又会调用 `build_video_entries()`，后者默认 `identity_mode="game_label_video"`、`namespace=None`、`require_content_hash=False`。这不仅允许重新生成空 `content_version_id`，还与当前配置默认的 `source_video_identity.mode="game_video"` 不一致。换句话说，**同一份配置只因为 video index 文件有没有配置，身份语义就会发生变化**。

7. **[P1｜本次提交] `tools/build_index.py --skip-content-hash` 已与 v2 reader 自相矛盾。** CLI 仍公开这个选项；三 root 模式能写出带 v2 schema、但 `content_version_id=""` 的 video parquet，之后新版 reader 又明确拒绝任何空 content version；`from_train` 路径则更早就因为 `require_content_hash=True` 直接失败。也就是工具仍然允许用户请求一个框架自身已经不接受的输出。建议不要继续兼容这个开关：如果 v2 identity 要求 content version，就应该在入口明确宣布 content hashing mandatory。

8. **[P1｜本次提交] packed source video index 没有绑定到刚验证的 bundle。** `dataset pack` 只执行 `verify_split_bundle(Path(frame_index).parent)`，然后允许 `--source-video-index` 指向其他任意文件。packer 只验证 `(game,label,video_id)` 和 `frame_ids` 相同，然后就继承对方的 `stable_source_id/content_version_id`，没有证明这个 video index 是 bundle manifest 承诺的那个 artifact。因此两个数据集只要帧编号结构相同，就可能交叉继承 identity。

9. **[P1｜本次提交] packed challenge/mining 是“Schema 支持、loader 支持、官方生成链不闭环”。** 最新 v4 provenance 强制 source bundle、audit、split manifest；但公共 `dataset pack` 又要求 `frame_index.parent` 是完整 sealed train/val/test split bundle。challenge/mining 外部池没有对应的 prepare/seal bundle 类型，因此官方 CLI 没有自然路径生成 loader 所要求的 packed external pool。测试通过直接调用 pack 函数并人工伪造 provenance 文件绕过了这个问题。这正是之前 step1/step6 的典型“声明支持但用户走不通”。

10. **[P1] 外部 challenge/mining 后端错误地绑定到训练 `data.backend`。** benchmark CLI 明确允许 plain 或 packed challenge/pool 任一种，但 `build_external_pool_loader()` 不根据外部池实际配置选择后端，而是直接看全局 `data.backend`。于是“训练 PNG + challenge packed”会通过 CLI 的存在性检查，然后进入 loader 后又要求 plain challenge video index；“训练 packed + challenge PNG”反向同样失败。外部池应该有自己的 backend，或根据互斥配置自动推导。

11. **[P1] exact resume 没有绑定实际 dataset/sidecar identity。** run manifest 保存 environment/git、checkpoint 等信息，但没有保存每个 split 的 bundle ID/index SHA、metadata sidecar fingerprint、packed manifest fingerprint；checkpoint 同样主要保存配置路径。用户如果在**同一路径**重新 annotate sidecar 或替换一代合法 dataset，config drift 为 0，resume 仍会继续，但 hard-negative buckets/subtype/data content 已经改变。这不是 exact resume。

12. **[P1] direct `train.resume_path` 仍能绕过 CLI 的完整 drift 策略，而且 critical key 有三处写错。** `RESUME_CRITICAL_DIFFS` 写的是 `data.class_probability / data.delta_probability / data.game_alpha`，实际消费者分别在 `sampler.*` 和 `pair.train_delta_probability`。标准 `--resume` 路径比较安全，因为普通 warning 最终也会分类为 fork；但 checkpoint restore 层只阻止 critical drift，因此直接配置 `train.resume_path` 或程序调用时，sampler、augmentation、loss 等策略改变仍可能恢复旧 optimizer/RNG/sampler 状态。所谓“保护所有 resume 入口”的目标还没有实现。

13. **[P1｜release UX] `benchmark evaluate` 在 `gate_metrics={}` 时会打印 `gate: PASSED` 并返回 0，而 `release check` 随后明确拒绝空 gates。** 根因是 `all([]) == True`。同一个报告在 `benchmark gate-show` 中又会因为 `bool(gates)==False` 被显示为未通过。一个命令说 PASSED、另一个说未通过、release 再说不可发布，是很强的用户误导。应该在 benchmark 开始前就拒绝空 gate，或者明确标成 `UNGATED/INFORMATIONAL`。

14. **[P1｜release provenance] release identity 没有绑定 benchmark/evaluator 的代码版本。** run manifest 的 environment 其实已经保存 git commit、dirty 状态和关键包版本，但 release identity 只包含 run id、checkpoint SHA、完整 config SHA、challenge fingerprint、gate-spec fingerprint。今天修复 evaluator 的 FPR/subtype 算法之后，昨天用旧 evaluator 得到的 PASS 仍可以被新 `release check` 接受。最低限度应加入独立的 `benchmark_contract_version`；更严格可以加入 code revision/evaluator schema version。

15. **[P1｜UX] benchmark/release report 定位依赖当前 CWD。** `benchmark.output_dir` 默认只是 `"benchmarks"`；benchmark 写报告和 release 搜索报告时都重新 `Path(...)`，没有锚定 run/config 所在目录。因此同一 run 在目录 A 做 benchmark，切到目录 B 再做 export/release，就可能“没有 PASS”。报告位置应该在 benchmark 时持久化进 run/release artifact，或统一 resolve 成绝对路径。

16. **[P1] `export.verify_samples <= 0` 可以静默关闭 ONNX 数值验证。** Schema 只要求 int；验证代码 `max_diff=0` 后直接 `for _ in range(verify_samples)`。设成 0 或负数时一次 ORT/PyTorch 对比都不做，最终却会保留 `max_diff=0` 并认为通过。必须在 config semantic validation 和运行入口双重保证 `>=1`。

17. **[P1/P2] export manifest 的 `selection_mode` 和 `selection_eligible` 永远得不到值。** `_export_manifest()` 从 `_metric_summary()` 中读取两个字段，但 `_metric_summary()` 根本不写这两个字段，因此 manifest 固定为 `null`。部署侧无法判断 checkpoint 是什么 selection contract 选出来的、是否满足 constrained eligibility。

18. **[P1/P2｜step1 同型] `sample_weight` 是完整公开的数据字段，但训练热路径没有消费者。** `VideoEntry`、sidecar、packed video index 都持久化它，但 sampler 没使用它改变视频选择概率，collate/loss 也不携带/消费它。若这是预留字段，名称和用户 API 会造成明显误导；若预期作为 sampling/loss weight，则现在属于失效配置。应该明确选一种语义并建立行为测试，或者暂时从公开接口移除。

19. **[P1] sampler 的 `deduplication.on_exhaustion=error` 在最后一次重采样仍重复时可以静默 append 重复样本。** 控制流只在进入 resample 前判断 error/relax，内部 while 把 attempts 消耗完以后不会重新执行 error 分支；如果此时恰好填满 batch，后面也没有机会再抛错。这使 `"error"` 不能提供它名字承诺的 hard guarantee。建议把“尝试重采样 → 最终仍冲突 → 执行 exhaustion policy”拆成明确的三阶段状态机。

低一级但也建议一并处理的 UX/契约问题包括：`data.mining.version` 当前只被读取和打印，writer 实际始终写常量 v2，因此可以出现“文件是 v2、CLI 声称 version=99”的结果；旧 v1 mining 文件交给 `dataset annotate --from-mining` 会从 reader 抛 `ValueError` 而不是给迁移提示；`dataset metadata-migrate` 的 config load 同样缺少统一的友好错误包装；sidecar API 把空 `negative_subtype` 解释为“保留旧值”，没有官方方式清掉错误 annotation；`dataset seal` 先直接改写 `audit.json/split_summary.json` 再写 bundle manifest，不是 transaction-safe，seal 中途失败可能把原本可用目录留成 mixed generation；export 的 immutable 目录只以 checkpoint SHA 为 key，同一 checkpoint 先导出 weights 后无法再导 ONNX；hard-negative mix、mining top-k/max-samples 等几个数值配置也缺少足够严格的范围约束。

## 为什么 CI 全绿仍然漏掉这些问题

这里不是简单的“测试数量不够”，而是**测试边界选错了**。最新提交对应 CPU CI 已经通过，但几个关键测试都有共同特征：测试生产函数的局部行为，而不是用户真正经过的完整路径。

release 测试没有从 console entrypoint `main(["release", ...])` 经过 implicit-train 路由，所以漏掉了最明显的 P0；challenge identity 测试验证 index/hash 元数据改变，却没有做最关键的 **index 不变、直接修改 PNG 字节** mutation test；packed contract 测试会直接构造 `content_version_id` 和 provenance 文件，没有从真实 `dataset prepare → 修改 PNG → dataset pack` 跑一遍；subtype evaluator 测试手工给 batch 填 `game_label_subtype_id`，没有走 `EvalPairDataset → pair_collate → evaluate`；`--skip-content-hash` 也没有完整 write→read regression test。

因此我建议以后把 step 中反复出现的问题正式转成一组 **contract mutation tests**：不是再增加十几个函数级单测，而是专门攻击“不变路径、同路径换内容、缺省 fallback、跨 bundle、真实 CLI dispatch、真实 collate、resume 同路径数据变化”这些边界。

## 推荐修复顺序

第一批应立即封住 **release CLI、challenge 实际字节验证、packer 实际字节验证、subtype collate**。其中 identity 最好不要分别在 benchmark 和 packer 里继续补 if，而是增加一个统一的 `verify_frame_content_integrity(frame_index)`：读取每个实际文件、重新计算 SHA-256、与 frame parquet 比较；release PNG challenge 和 dataset pack 都调用它。packed release 则应显式执行 `verify_packed_shards()`，不能把“manifest 里记录了预期 shard hash”等价成“当前 shard 已验证”。

第二批把 identity 链真正闭合：强制 v2 video index、删除或严格限制 frame-index legacy fallback，移除 `--skip-content-hash`；pack 时证明 frame/video index 都来自同一个 bundle；mining annotate 比较 `source_version_id`；为 external challenge/mining 定义独立 `external_bundle` contract，而不是硬借 train/val/test split bundle。

第三批再统一 resume/release provenance：checkpoint/run manifest 存 `bundle_id + index SHA + sidecar fingerprint + packed manifest SHA`，exact resume 对这些做强等价；release identity 加 `benchmark_contract_version`/evaluator revision。同时把所有直接 `train.resume_path` 入口路由到和 `--resume` 完全相同的 drift classifier。

如果按“最少修复次数、最大风险下降”排序，我会先动大约 **8–10 个核心文件**：`cli/init.py`、`release.py`、`reports/benchmark.py`、`data/indexing.py`、`data/packed_backend.py`、`data/video_index.py`、`data/collate.py`、`cli/dataset.py`、`engine/training/loaders.py`、`engine/checkpoint.py/common.py`。修完这批再补上述 integration/mutation tests，项目的 release/identity 可信度会比单纯继续增加局部 guard 提升得多。

本次审查通过连接的 GitHub 直接读取了当前 `main`、最新 commit diff、生产代码和相关测试，并核对了最新 Actions 状态；本地容器到 GitHub 的网络解析不可用，所以我没有重新执行一份本地 pytest/NPU runtime。上面 P0/P1 的核心结论都是可以从当前代码控制流直接证明的，不依赖运行时猜测。
