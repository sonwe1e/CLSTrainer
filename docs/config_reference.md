# CLSTrainer 5.0.0 配置参考

``cls-trainer config reference`` 根据当前 Schema 自动生成。

```text
augmentation.blur.enabled  [bool]
    Enable Gaussian blur.
augmentation.blur.kernel_size  [int]
    Odd Gaussian blur kernel size.
augmentation.blur.probability  [float]
    Per-pair probability.
augmentation.blur.sigma_range  [list]
    Gaussian sigma range [min, max].
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
augmentation.exposure.enabled  [bool]
    Enable exposure adjustment.
augmentation.exposure.factor_range  [list]
    Exposure factor range [min, max].
augmentation.exposure.probability  [float]
    Per-pair probability.
augmentation.gamma.enabled  [bool]
    Enable gamma correction.
augmentation.gamma.gamma_range  [list]
    Gamma range [min, max].
augmentation.gamma.probability  [float]
    Per-pair probability.
augmentation.jpeg_compression.enabled  [bool]
    Enable JPEG re-encode artifact injection.
augmentation.jpeg_compression.probability  [float]
    Per-pair probability.
augmentation.jpeg_compression.quality_range  [list]
    JPEG quality range [min, max].
augmentation.noise.enabled  [bool]
    Enable Gaussian noise injection.
augmentation.noise.noise_std  [float]
    Additive noise standard deviation.
augmentation.noise.probability  [float]
    Per-pair probability.
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
augmentation.random_perspective.distortion_scale  [float]
    Perspective distortion strength, typically <= 1.
augmentation.random_perspective.enabled  [bool]
    Enable random perspective warp.
augmentation.random_perspective.probability  [float]
    Per-pair probability.
augmentation.random_resized_crop.enabled  [bool]
    Enable random resized crop.
augmentation.random_resized_crop.probability  [float]
    Per-pair probability.
augmentation.random_resized_crop.ratio  [list]
    Crop aspect-ratio range [min, max].
augmentation.random_resized_crop.scale  [list]
    Crop area fraction range [min, max].
augmentation.random_resized_crop.size  [list]
    Output size [height, width].
benchmark.gate_metrics  [dict]
    Release/benchmark gates keyed by the metric name the evaluator emits, each {op: "<="|"<"|">="|">", value: <number>}, e.g. {global_fpr_at_decision_threshold: {op: "<=", value: 0.01}, global_positive_recall_at_decision_threshold: {op: ">=", value: 0.8}}. Every gate must use the explicit form; bare scalar bounds, unknown metric names and unknown operators are config errors. Unmet gates fail the command.
benchmark.output_dir  [str]
    Directory for benchmark reports and data probes.
checkpoint.full_model_every_steps  [int]
    Cadence for full-weight periodic saves; 0 disables.
checkpoint.periodic_state_mode  [str] (choices: full|trainable_only)
    Periodic checkpoint content: full state or trainable-only.
checkpoint.save_best_selection  [bool]
    Clone the best validation selection-score checkpoint (model_best_selection.pth).
checkpoint.save_best_val_loss  [bool]
    Clone the best validation loss checkpoint (model_best_val_loss.pth).
checkpoint.save_best_worst_game  [bool]
    Clone the best worst-game-F1 checkpoint (model_best_worst_game.pth).
checkpoint.save_last_every_steps  [int]
    Periodic resume checkpoint cadence; 0 disables.
checkpoint.save_topk  [int]
    Keep the top-N full-validation checkpoints ranked by checkpoint.topk_monitor (saved as model_topk_<step>.pth); 0 disables. Enabling it forces a last-checkpoint save after every full validation so the topk snapshot always matches the evaluated weights.
checkpoint.topk_monitor  [str] (choices: selection_score|cross_entropy|worst_game_f1_at_decision_threshold)
    `selection_score` ranks topk checkpoints by the unified selection contract, the same ordering as best-checkpoint selection; ineligible checkpoints are admitted but ranked strictly below every eligible one. Any other value names one numeric metric: lower is better for `cross_entropy`, higher is better otherwise.
data.audit_path  [str]
    Path to audit.json produced by audit_dataset.
data.backend  [str] (choices: png|packed_uint8)
    Frame storage backend.
data.challenge_index  [str|null]
    Fixed challenge-set frame index (never part of train/val/test; only consumed by benchmark evaluate).
data.challenge_metadata  [str|null]
    Optional challenge-set metadata sidecar.
data.challenge_packed_index  [str|null]
    packed_uint8 shard index of the challenge set; when set, the challenge uses packed storage independently of data.backend.
data.challenge_packed_video_index  [str|null]
    Video-level index of the packed challenge set; required with challenge_packed_index.
data.challenge_video_index  [str|null]
    Challenge-set video-level index.
data.channels  [int]
    Frame channels (task profile default: 3).
data.deduplication.level  [str] (choices: none|pair|video)
    Within one global batch: 'pair' avoids identical (video, delta, start) triples; 'video' avoids repeating the same video at all; 'none' samples freely.
data.deduplication.on_exhaustion  [str] (choices: error|warn_and_relax)
    What to do when a global batch cannot be filled without repeating the dedup identity: 'error' fails the run, 'warn_and_relax' logs a warning and relaxes the constraint for that batch.
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
data.hard_negative.enabled  [bool]
    Mix ordinary and hard negative videos by subtype bucket during sampling (requires metadata_sidecar).
data.hard_negative.hard_subtypes  [list]
    Subtype values treated as hard negatives.
data.hard_negative.max_pairs_per_video  [int|null]
    Optional cap on how many start positions each video contributes (deterministic first N).
data.hard_negative.min_videos_per_subtype_bucket  [int]
    Minimum eligible videos required in a bucket before it is used; smaller buckets fall back to the other bucket.
data.hard_negative.negative_mix  [dict]
    Sampling weights per bucket, e.g. {ordinary: 0.5, hard: 0.5}.
data.hard_negative.ordinary_subtypes  [list]
    Subtype values treated as ordinary negatives; empty means every subtype not listed in hard_subtypes.
data.hard_negative.subtype_field  [str]
    VideoEntry attribute holding the subtype label.
data.height  [int]
    Frame height in pixels (task profile default: 208).
data.ignore_directory_names  [list]
    Exact directory names pruned during scanning.
data.ignore_directory_prefixes  [list]
    Directory name prefixes pruned during scanning.
data.ignore_file_globs  [list]
    File globs ignored during scanning.
data.ignored_example_limit  [int]
    Max example paths kept per ignored-file category.
data.metadata_sidecar  [str|null]
    Optional contract-5 metadata parquet keyed by stable_source_id (negative_subtype, sample_weight). Joined AFTER the split; never part of split/dedup identity (step5 P2).
data.minimum_pairs_per_game_label_delta  [dict]
    Minimum legal pairs per (game,label) for each delta, e.g. {2: 1}.
data.mining.enabled  [bool]
    Enable the hard-negative mining workflow (scan-negatives and --from-mining annotation).
data.mining.max_samples  [int|null]
    Optional global cap on mined samples.
data.mining.output  [str]
    Output hard_negatives.parquet mining manifest path.
data.mining.pool_index  [str|null]
    Frame index of the training-side negative pool to scan.
data.mining.pool_metadata  [str|null]
    Optional sidecar of the mining pool (subtype_before).
data.mining.pool_packed_index  [str|null]
    packed_uint8 shard index of the mining pool; when set, this external pool uses packed storage independently of data.backend.
data.mining.pool_packed_video_index  [str|null]
    Video-level index of the packed mining pool; required with pool_packed_index.
data.mining.pool_video_index  [str|null]
    Video-level index of the mining pool.
data.mining.score_threshold  [float|null]
    Optional p_positive floor; only negatives at or above it are kept.
data.mining.top_k_per_video  [int]
    Max negatives kept per source video (avoids continuous frames drowning the manifest).
data.packed_max_open_shards  [int]
    LRU limit of simultaneously memmapped packed shards.
data.prepare_if_missing  [bool]
    Auto-run dataset prepare before training when split indexes are missing.
data.require_content_hash_audit  [bool]
    Audit must include SHA-256 content hashes.
data.require_independent_test  [bool]
    Production acceptance gate: require a dedicated test split distinct from validation.
data.require_unique_video_keys_across_splits  [bool]
    Treat cross-split duplicate two-digit video ids as fatal.
data.source_root  [str|null]
    Single root scanned for both train and validation (split.mode=from_train).
data.source_video_identity  [dict]
    Source video identity contract (step7): how a source video uid is derived. game_video (default) treats video_id as unique per game; game_label_video treats video_id as unique per (game,label) and prints a leakage warning.
data.source_video_identity.mode  [str] (choices: game_video|game_label_video)
    game_video: uid = game::video_id (default, conservative). game_label_video: uid = game::label::video_id -- assumes identical video_id under different labels are unrelated videos.
data.source_video_identity.namespaces.source  [str]
    Train-side pool namespace under the {source, test} shorthand (use with data.split.mode=from_train).
data.source_video_identity.namespaces.test  [str]
    Source-pool namespace for the test split; must differ from the train-side namespace (a declaration that test is an independent raw-video pool).
data.source_video_identity.namespaces.train  [str]
    Source-pool namespace for the train split (explicit form). Train and val are split from one physical pool, so train must equal val.
data.source_video_identity.namespaces.val  [str]
    Source-pool namespace for the validation split; must equal train.
data.split  [dict]
    Source-video-level train/validation split configuration (step4 ���).
data.split.balance_by  [str]
    Balancing statistic; must be 'legal_pair_count'.
data.split.group_key  [str]
    Split unit identity; must be 'stable_source_id'.
data.split.manifest  [str]
    Split manifest parquet path, relative to the index output dir (default: split_manifest.parquet).
data.split.mode  [str] (choices: off|from_train)
    Split derivation mode: 'off' consumes prepared indexes; 'from_train' scans source_root once and derives train/val by source video.
data.split.on_new_groups  [str] (choices: error|extend)
    Behavior when the dataset fingerprint changes: 'error' refuses to silently re-shuffle, 'extend' keeps every existing assignment and places only the new source videos. A change to seed, val_ratio or target_delta is always an error regardless of this setting.
data.split.seed  [int]
    Deterministic split seed; the same seed and data produce a byte-identical manifest.
data.split.small_stratum_policy  [str] (choices: error|warn)
    Behavior for strata with fewer than two source videos: 'error' rejects, 'warn' keeps the lone video in train.
data.split.stratify_by  [list]
    Stratum fields for balancing, e.g. ['game', 'label'].
data.split.target_delta  [int]
    Frame delta whose pair count drives balancing (1, 2 or 3).
data.split.val_ratio  [float]
    Fraction of source-video legal pairs moved to validation; strictly between 0 and 1 when mode is from_train.
data.strict_audit  [bool]
    Require a passing dataset audit before creating DataLoaders.
data.synthetic  [bool]
    Use the built-in synthetic dataset (smoke tests only).
data.test_index  [str|null]
    Optional independent test frame index parquet.
data.test_packed_index  [str|null]
    Packed test shard index (packed_uint8 backend).
data.test_packed_video_index  [str|null]
    Packed test integer video index.
data.test_root  [str|null]
    Independent test root used with split.mode=from_train.
data.test_video_index  [str|null]
    Optional independent test video-level entries parquet.
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
data.val_index  [str|null]
    Validation frame index parquet; required for real-data training.
data.val_packed_index  [str|null]
    Packed validation shard index (packed_uint8 backend).
data.val_packed_video_index  [str|null]
    Packed validation integer video index.
data.val_video_index  [str|null]
    Validation video-level entries parquet; required for real-data training.
data.width  [int]
    Frame width in pixels (task profile default: 448).
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
early_stopping.burn_in_steps  [int]
    Never stop before this global step.
early_stopping.enabled  [bool]
    Stop training when the monitored validation metric plateaus. train.max_steps remains a safety upper bound.
early_stopping.full_validation_only  [bool]
    Only full-validation evaluations may update patience; quick subsets are too noisy for stop decisions.
early_stopping.min_delta  [float]
    Minimum improvement that counts as an improvement; smaller deltas increment the patience counter.
early_stopping.mode  [str] (choices: max|min)
    max: higher monitor values are better; min: lower values. Applies only to non-selection monitors (any `monitor` other than `selection_score`); `mode: min` combined with a selection monitor is a config error because the selection rank key is always bigger-is-better.
early_stopping.monitor  [str] (choices: selection_score|cross_entropy|objective_loss|worst_game_f1_at_decision_threshold)
    Metric or selection contract watched for improvement. `selection_score` follows the unified selection contract: improvement is judged by the same ordering as best-checkpoint selection; in constrained mode that is (global_positive_recall, worst_game_positive_recall, -negative_score_p999), with ineligible evaluations counting toward patience rather than resetting it. Any other value names one numeric metric and uses the plain `mode` comparison.
early_stopping.patience_evaluations  [int]
    Consecutive non-improving full validations tolerated before stopping.
early_stopping.restore_best  [bool]
    Reload the best-selection checkpoint weights before the run finishes.
evaluation.amp  [bool]
    Mixed precision for evaluation forward passes.
evaluation.amp_dtype  [str] (choices: float16|bfloat16)
    Evaluation AMP dtype (deployment parity).
evaluation.auc_histogram_bins  [int]
    Histogram bins for AUC estimation.
evaluation.full_auc_mode  [str] (choices: histogram|exact)
    Distributed AUC strategy: fixed histogram or exact gather.
evaluation.group_by_negative_subtype  [bool]
    Add a game_label_subtype group catalog to evaluation and compute worst-subtype FPR/recall metrics (requires sidecar metadata with non-null negative_subtype).
evaluation.html_max_errors_per_group  [int]
    Per-group error cap in the HTML report.
evaluation.max_fpr_for_recall  [float]
    FPR bound for recall_at_max_fpr and low-FPR partial AUC.
evaluation.max_global_fpr  [float|null]
    Constrained selection: reject candidates whose global FPR at the decision threshold exceeds this (null disables the gate).
evaluation.max_worst_game_fpr  [float|null]
    Constrained selection: reject candidates whose worst-game FPR exceeds this (null disables).
evaluation.max_worst_subtype_fpr  [float|null]
    Constrained selection: reject candidates whose worst negative-subtype FPR exceeds this (null disables).
evaluation.min_positive_recall  [float|null]
    Constrained selection: require global positive recall at or above this (null disables).
evaluation.minimum_worst_game_f1  [float|null]
    Reject best-checkpoint candidates whose worst game F1 falls below this gate.
evaluation.parquet_row_group_size  [int]
    Records accumulated before writing a parquet row group.
evaluation.quick_save_error_limit  [int]
    Global cap on quick-test error exports.
evaluation.selection_metric  [str] (choices: global_f1_at_decision_threshold|macro_game_f1_at_decision_threshold|worst_game_f1_at_decision_threshold|composite)
    Metric used to pick the best checkpoint.
evaluation.selection_mode  [str] (choices: metric|composite|constrained)
    Model-selection strategy: metric (single metric), composite (weighted F1), or constrained (FPR/recall gates then recall/worst-recall/p99.9 ranking).
evaluation.selection_weights  [dict]
    Component weights for composite selection, e.g. {global_f1: 0.4, macro_game_f1: 0.4, worst_game_f1: 0.2}.
evaluation.tail_calibration_enabled  [bool]
    Compute ece_tail_95_100 in evaluation.
evaluation.tensorboard_live  [bool]
    Write TensorBoard scalars during training when the tensorboard package is importable; 0-cost when absent.
evaluation.train_probe_every_steps  [int]
    Train-probe cadence (augmentation-free train subset evaluated like validation); 0 disables.
evaluation.train_probe_pairs_per_video  [int]
    Train-probe pairs sampled per train video.
evaluation.val_full_at_end  [bool]
    Run a final full validation after training.
evaluation.val_full_every_steps  [int]
    Full validation cadence (drives model selection and early stopping); 0 disables.
evaluation.val_quick_every_steps  [int]
    Quick validation cadence (fixed validation subset, high frequency trend watching); 0 disables.
evaluation.val_quick_pairs_per_video  [int]
    Quick validation pairs sampled per validation video.
experiment.name  [str]
    Human readable run name; used in the unique run directory name.
experiment.output_dir  [str]
    Runs root; every fresh start creates a dated immutable subdirectory underneath it.
experiment.seed  [int]
    Base RNG seed; each rank adds its rank id.
experiment.smoke_mode  [bool]
    Mark a run as a smoke test so the production safety policy is relaxed (e.g. the require-a-full-validation-source gate). Real training must never set this; the NPU smoke stages disable full validation to probe forward/spawn/augmentation/eval separately (audit PR-E: test-env policy and production safety policy must not fight each other).
export.format  [str] (choices: weights|onnx)
    Default export format (weights|onnx).
export.include_threshold  [bool]
    Embed decision.threshold in the exported manifest.
export.onnx_opset  [int]
    ONNX opset version for --format onnx.
export.output_dir  [str]
    Directory for exported artifacts.
export.verify_samples  [int]
    Random sample tensors used to verify ONNX vs PyTorch.
loss.cross_entropy_weight  [float]
    Weight of the CE component.
loss.label_smoothing  [float]
    Cross-entropy label smoothing; keep small while the deployment threshold is fixed at 0.99.
loss.negative_tail_hard_negative_k  [int|null]
    Top-k hardest negatives for the tail OHEM; null uses all negatives.
loss.negative_tail_loss_weight  [float]
    Weight of the negative-tail OHEM component; 0 disables.
loss.rank_loss_weight  [float]
    Weight of the positive-vs-hard-negative pairwise ranking component; 0 disables.
loss.rank_margin  [float]
    Required logit margin between a positive and a hard negative in the ranking loss.
loss.threshold_loss_weight  [float]
    Maximum weight of the threshold margin loss.
loss.threshold_ramp_ratio  [float]
    Fraction of training used to ramp the margin weight.
loss.threshold_ramp_steps  [int|null]
    Explicit step count used to ramp the margin weight. Overrides threshold_ramp_ratio when set.
loss.threshold_safety_margin  [float]
    Extra logit margin pushed beyond the decision boundary.
loss.threshold_temperature  [float]
    Softplus temperature of the margin loss.
loss.threshold_warmup_ratio  [float]
    Fraction of training before the margin loss activates.
loss.threshold_warmup_steps  [int|null]
    Explicit step count before the margin loss activates. Overrides threshold_warmup_ratio when set; keeps the schedule independent of the total step budget.
model.checkpoint_path  [str|null]
    Base checkpoint; all non-cls weights must load from it.
model.factory  [str]
    Model factory 'package.module:function' returning a module that maps (image0, image1) to [B,2].
model.freeze_backbone_batchnorm_stats  [bool]
    Keep backbone BatchNorm statistics frozen.
model.freeze_cls_batchnorm_stats  [bool]
    Keep cls-head BatchNorm statistics frozen (required for distributed training without SyncBatchNorm).
model.kwargs  [dict]
    Free-form project-specific factory arguments (e.g. cls_dropout). Passed to the model factory inside the model config mapping.
model.num_classes  [int]
    Output classes (task profile default: 2).
model.require_pretrained_backbone  [bool]
    Fail unless every non-cls weight is fully loaded from the base checkpoint.
model.trainable_rules  [dict]
    Staged partial unfreeze: dict keyed by rule name, each rule {pattern, lr_scale, unfreeze_at_step, priority}.
optimizer.learning_rate  [float]
    Peak AdamW learning rate.
optimizer.weight_decay  [float]
    AdamW weight decay.
pair.test_delta  [int]
    Evaluation pair delta (task profile default: 2).
pair.train_delta_probability  [dict]
    Training frame-delta sampling distribution, e.g. {2: 0.7}.
sampler.class_probability  [dict]
    Label sampling probability, e.g. {0: 0.5, 1: 0.5}.
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
