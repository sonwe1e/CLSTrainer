# CLSTrainer 配置参考

本文件由 `cls-trainer config reference` 自动生成,列出 Schema 认识的全部配置键。
未在表中出现的键会被严格拒绝(未知键直接报错,不再静默忽略)。

业务判定阈值只有一个来源:`decision.threshold`;训练阈值损失与评估器都从它读取。

```
augmentation.color_jitter.brightness  [float]
    Brightness jitter magnitude.
augmentation.color_jitter.contrast  [float]
    Contrast jitter magnitude.
augmentation.color_jitter.enabled  [bool]
    Enable color jitter.
augmentation.color_jitter.hue  [float]
    Hue jitter magnitude.
augmentation.color_jitter.probability  [float]
    Per-pair probability.
augmentation.color_jitter.saturation  [float]
    Saturation jitter magnitude.
augmentation.enabled  [bool]
    Master switch for deterministic pair augmentation.
augmentation.random_affine.degrees  [float]
    Rotation range in degrees.
augmentation.random_affine.enabled  [bool]
    Enable random affine perturbations.
augmentation.random_affine.fill  [any]
    Fill value for uncovered pixels.
augmentation.random_affine.interpolation  [str]
    Resampling filter name.
augmentation.random_affine.probability  [float]
    Per-pair probability.
augmentation.random_affine.scale  [list]
    Scale range [min, max].
augmentation.random_affine.shear  [list]
    Shear range in degrees.
augmentation.random_affine.translate  [list]
    Relative translation range [x, y].
augmentation.random_erasing.enabled  [bool]
    Enable random erasing.
augmentation.random_erasing.probability  [float]
    Per-pair probability.
augmentation.random_erasing.ratio  [list]
    Erased aspect ratio range.
augmentation.random_erasing.scale  [list]
    Erased area fraction range.
augmentation.random_erasing.value  [any]
    Fill value; 'random' for noise.
checkpoint.full_model_every_steps  [int]
    Cadence for full-weight periodic saves; 0 disables.
checkpoint.periodic_state_mode  [str] (choices: full|trainable_only)
    Periodic checkpoint content: full state or trainable-only.
checkpoint.save_best_selection  [bool]
    Clone the best observed dev-test checkpoint.
checkpoint.save_best_test_f1  [bool] (legacy)
    Legacy alias of save_best_selection.
checkpoint.save_last_every_steps  [int]
    Periodic resume checkpoint cadence; 0 disables.
data.audit_path  [str]
    Path to audit.json produced by audit_dataset.
data.backend  [str] (choices: png|packed_uint8)
    Frame storage backend.
data.channels  [int]
    Frame channels (contract: 3).
data.duplicate_policy.cross_label_same_content  [str] (choices: info|warning|error)
    Severity when identical content carries different labels.
data.duplicate_policy.same_basename  [str] (choices: info|warning|error)
    Severity for filename-only collisions.
data.duplicate_policy.same_label_cross_split  [str] (choices: info|warning|error)
    Severity for same-content duplicates across splits.
data.duplicate_policy.same_label_within_split  [str] (choices: info|warning|error)
    Severity for same-content duplicates within a split.
data.frame_extensions  [list]
    File extensions accepted as frames during scanning.
data.height  [int]
    Frame height in pixels (contract: 208).
data.ignore_directory_names  [list]
    Exact directory names pruned during scanning.
data.ignore_directory_prefixes  [list]
    Directory name prefixes pruned during scanning.
data.ignore_file_globs  [list]
    File globs ignored during scanning.
data.ignored_example_limit  [int]
    Max example paths kept per ignored-file category.
data.minimum_pairs_per_game_label_delta  [dict]
    Minimum legal pairs per (game,label) for each delta, e.g. {2: 1}.
data.packed_max_open_shards  [int]
    LRU limit of simultaneously memmapped packed shards.
data.require_content_hash_audit  [bool]
    Audit must include SHA-256 content hashes.
data.require_unique_video_keys_across_splits  [bool]
    Treat cross-split duplicate two-digit video ids as fatal.
data.strict_audit  [bool]
    Require a passing dataset audit before creating DataLoaders.
data.synthetic  [bool]
    Use the built-in synthetic dataset (smoke tests only).
data.test_index  [str]
    Test frame index parquet.
data.test_packed_index  [str|null]
    Packed test shard index (packed_uint8 backend).
data.test_packed_video_index  [str|null]
    Packed test integer video index.
data.test_video_index  [str]
    Test video-level entries parquet (row-per-video).
data.train_index  [str]
    Train frame index parquet.
data.train_packed_index  [str|null]
    Packed train shard index (packed_uint8 backend).
data.train_packed_video_index  [str|null]
    Packed train integer video index.
data.train_video_index  [str]
    Train video-level entries parquet (row-per-video).
data.unexpected_nested_directory_severity  [str] (choices: info|warning|error)
    Severity for unexpected nested directories.
data.width  [int]
    Frame width in pixels (contract: 448).
dataloader.eval.num_workers  [int]
    DataLoader worker processes for this role.
dataloader.eval.persistent_workers  [bool]
    Keep workers alive between epochs/evaluations.
dataloader.eval.pin_memory  [bool]
    Pin host memory before device transfer. Keep false until an A/B test proves it helps.
dataloader.eval.prefetch_factor  [int]
    Batches prefetched per worker.
dataloader.multiprocessing_context  [str|null] (choices: spawn|fork|forkserver)
    Worker start method; NPU forces spawn.
dataloader.num_workers  [int] (legacy)
    Fallback worker count; role-specific dataloader.train/eval values win when present.
dataloader.persistent_workers  [bool] (legacy)
    Fallback persistent_workers.
dataloader.pin_memory  [bool] (legacy)
    Fallback pin_memory.
dataloader.prefetch_factor  [int] (legacy)
    Fallback prefetch_factor.
dataloader.timeout_seconds  [float]
    Max wait per batch before a DataLoader timeout.
dataloader.train.num_workers  [int]
    DataLoader worker processes for this role.
dataloader.train.persistent_workers  [bool]
    Keep workers alive between epochs/evaluations.
dataloader.train.pin_memory  [bool]
    Pin host memory before device transfer. Keep false until an A/B test proves it helps.
dataloader.train.prefetch_factor  [int]
    Batches prefetched per worker.
dataloader.worker_num_threads  [int]
    CPU threads allowed inside each worker process.
decision.threshold  [float]
    THE business decision threshold. Single source of truth: the training threshold loss, the evaluator and deployment all read this value. Strictly-greater-than this softmax probability means positive.
device.accelerator  [str] (choices: auto|cpu|cuda|npu)
    Training device family.
device.amp  [bool]
    Enable mixed precision training.
device.amp_dtype  [str] (choices: float16|bfloat16)
    Mixed precision dtype.
distributed.backend  [str]
    Process group backend, e.g. gloo / nccl / hccl.
distributed.enabled  [bool]
    Initialize c10d process groups.
evaluation.amp  [bool]
    Mixed precision for evaluation forward passes.
evaluation.amp_dtype  [str] (choices: float16|bfloat16)
    Evaluation AMP dtype (deployment parity).
evaluation.auc_histogram_bins  [int]
    Histogram bins for AUC estimation.
evaluation.full_auc_mode  [str] (choices: histogram|exact)
    Distributed AUC strategy: fixed histogram or exact gather.
evaluation.full_test_at_end  [bool]
    Run a final full test after training.
evaluation.full_test_every_steps  [int]
    Full-test cadence (observed dev-test); 0 disables.
evaluation.html_max_errors_per_group  [int]
    Per-group error cap in the HTML report.
evaluation.minimum_worst_game_f1  [float|null]
    Reject best-checkpoint candidates whose worst game F1 falls below this gate.
evaluation.parquet_row_group_size  [int]
    Records accumulated before writing a parquet row group.
evaluation.quick_save_error_limit  [int]
    Global cap on quick-test error exports.
evaluation.quick_test_every_steps  [int]
    Quick-test cadence; 0 disables.
evaluation.quick_test_pairs_per_video  [int]
    Quick-test pairs sampled per video.
evaluation.selection_metric  [str] (choices: global_f1_tau099|macro_game_f1_tau099|worst_game_f1_tau099|composite)
    Metric used to pick the best checkpoint.
evaluation.selection_weights  [dict]
    Component weights for composite selection, e.g. {global_f1: 0.4, macro_game_f1: 0.4, worst_game_f1: 0.2}.
evaluation.threshold  [float] (legacy)
    Legacy copy of decision.threshold. Prefer decision.threshold; conflicting values are rejected.
experiment.name  [str]
    Human readable run name; used in the unique run directory name.
experiment.output_dir  [str]
    Run output location. With run_mode=unique it is the runs ROOT: every start creates a fresh dated subdirectory underneath it.
experiment.run_mode  [str] (choices: fixed|unique)
    fixed: write directly into output_dir (legacy). unique: allocate an immutable timestamped run directory under output_dir.
experiment.seed  [int]
    Base RNG seed; each rank adds its rank id.
loss.cross_entropy_weight  [float]
    Weight of the CE component.
loss.threshold  [float] (legacy)
    Legacy copy of decision.threshold. Prefer decision.threshold; conflicting values are rejected.
loss.threshold_loss_weight  [float]
    Maximum weight of the threshold margin loss.
loss.threshold_ramp_ratio  [float]
    Fraction of training used to ramp the margin weight.
loss.threshold_safety_margin  [float]
    Extra logit margin pushed beyond the decision boundary.
loss.threshold_temperature  [float]
    Softplus temperature of the margin loss.
loss.threshold_warmup_ratio  [float]
    Fraction of training before the margin loss activates.
model.checkpoint_path  [str|null]
    Base checkpoint; all non-cls weights must load from it.
model.factory  [str]
    Model factory 'package.module:function' returning a module that maps (image0, image1) to [B,2].
model.freeze_backbone_batchnorm_stats  [bool]
    Keep backbone BatchNorm statistics frozen.
model.freeze_batchnorm_stats  [bool] (legacy)
    Legacy global BatchNorm freeze switch; prefer the two role-specific keys above.
model.freeze_cls_batchnorm_stats  [bool]
    Keep cls-head BatchNorm statistics frozen (required for distributed training without SyncBatchNorm).
model.num_classes  [int]
    Output classes (contract: 2).
model.require_pretrained_backbone  [bool]
    Fail unless every non-cls weight is fully loaded from the base checkpoint.
model.trainable_name_contains  [str]
    Substring selecting trainable parameters (contract: cls).
optimizer.learning_rate  [float]
    Peak AdamW learning rate.
optimizer.weight_decay  [float]
    AdamW weight decay.
pair.test_delta  [int]
    Evaluation pair delta (contract: 2).
pair.train_delta_probability  [dict]
    Training frame-delta sampling distribution, e.g. {2: 0.7}.
sampler.class_probability  [dict]
    Label sampling probability, e.g. {0: 0.5, 1: 0.5}.
sampler.deduplicate_within_global_batch  [bool]
    Avoid repeating a video inside one global batch.
sampler.game_alpha  [float]
    Dirichlet smoothing for per-game balancing.
scheduler.min_learning_rate  [float]
    Cosine schedule floor.
scheduler.warmup_steps  [int]
    Linear warmup steps for the cosine schedule.
train.epochs  [int]
    Epoch count (used when max_steps is null).
train.gradient_clip_norm  [float]
    Global gradient clip norm.
train.local_batch_size  [int]
    Per-rank batch size.
train.log_every_steps  [int]
    Metric/log cadence in steps.
train.max_steps  [int|null]
    Global step budget; overrides epochs when set.
train.resume_path  [str|null]
    Checkpoint used for exact resume.
train.steps_per_epoch  [int]
    Sampler steps per epoch.
train.stop_after_steps  [int|null]
    Optional early stop for staged acceptance runs.
train.verify_frozen_parameters  [bool]
    Assert frozen weights stay bitwise unchanged.
```
