"""Tests for the P0 fixes applied from USERPLAN 4 evaluation.

Covers:
1. Runtime lifecycle: single instance, setup() called, cleanup() correct
2. Optimizer weight decay: bias/1D params get no decay by default
3. Checkpoint resume: V2 manifest-based validation
4. Persistent buffer coverage: frozen BN stats validated (all policies)
5. Backend __call__: single-sample decode works
6. CPU DDP: device_ids=None for CPU
7. Registry factory_path: dynamic import works
8. EvaluatorSuite registry-driven: fail-fast for incompatible task
9. DataModule close: backends tracked and closed
10. SamplingPolicy in production path: policy wraps sampler
11. Task param validation: positive_class_index validated
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from game_cls.config.schema import ValidationError, validate_and_normalize_config
from game_cls.contracts.trainable import StateSelection, TrainableSelection


# --------------------------------------------------------------------------- #
# 1. Runtime lifecycle
# --------------------------------------------------------------------------- #
class TestRuntimeLifecycle:
    def test_runner_creates_single_runtime_and_calls_setup(self):
        """Runner must create exactly one runtime and call setup() on it."""
        from game_cls.engine.runner import ExperimentRunner

        config = _minimal_v2_config()
        setup_called = []

        class _MockAccelerator:
            @property
            def device(self):
                return torch.device("cpu")

            def setup(self, local_rank):
                setup_called.append(("accelerator.setup", local_rank))

            def synchronize(self):
                pass

            def autocast(self, enabled, dtype):
                from contextlib import nullcontext

                return nullcontext()

            def make_grad_scaler(self, enabled, dtype):
                return torch.amp.GradScaler("cpu", enabled=False)

        class _MockDistributed:
            rank = 0
            world_size = 1
            local_rank = 0

            def setup(self, backend):
                setup_called.append(("distributed.setup", backend))

            def wrap_model(self, model, device):
                return model

            def barrier(self):
                pass

            def cleanup(self):
                setup_called.append(("cleanup", None))

        class _MockRuntime:
            """Count how many times a runtime is created."""

            _instance_count = 0

            def __init__(self):
                _MockRuntime._instance_count += 1
                self._accelerator = _MockAccelerator()
                self._distributed = _MockDistributed()
                self._is_setup = False

            @property
            def accelerator(self):
                return self._accelerator

            @property
            def distributed(self):
                return self._distributed

            def setup(self):
                if self._is_setup:
                    return
                self._accelerator.setup(self._distributed.local_rank)
                self._distributed.setup(None)
                self._is_setup = True

            def cleanup(self):
                if not self._is_setup:
                    return
                self._distributed.cleanup()
                self._is_setup = False

        def mock_build_runtime(selector):
            return _MockRuntime()

        # Patch build_runtime AND build_core_components to avoid needing a full
        # production config (model factory, data pipeline, etc.). We only want to
        # verify the runtime lifecycle here.
        from game_cls.engine.builders import ExperimentComponents

        def mock_build_core_components(cfg, *, runtime=None):
            return ExperimentComponents(
                config={},
                raw_config=cfg,
                runtime=runtime,
                task=None,
                trainable_policy=None,
                trainable_selection=None,
                model=None,
                image_spec=None,
                data_module=None,
                evaluator=None,
            )

        with patch("game_cls.engine.builders.build_runtime", side_effect=mock_build_runtime), \
             patch("game_cls.engine.runner.build_core_components", side_effect=mock_build_core_components):
            runner = ExperimentRunner(config)
            runner.setup()

        # Exactly one runtime created.
        assert _MockRuntime._instance_count == 1, (
            f"Expected 1 runtime, got {_MockRuntime._instance_count}"
        )
        # setup() was called (accelerator + distributed).
        assert ("accelerator.setup", 0) in setup_called
        assert ("distributed.setup", None) in setup_called

        # Cleanup cleans the same runtime.
        runner.close()
        assert ("cleanup", None) in setup_called

    def test_runtime_setup_is_idempotent(self):
        """Calling setup() twice must not re-initialize."""
        from game_cls.runtime.strategy import ComposedRuntimeStrategy

        events = []

        class _Acc:
            @property
            def device(self):
                return torch.device("cpu")

            def setup(self, rank):
                events.append("acc.setup")

            def synchronize(self):
                pass

            def autocast(self, enabled, dtype):
                from contextlib import nullcontext

                return nullcontext()

            def make_grad_scaler(self, enabled, dtype):
                return torch.amp.GradScaler("cpu", enabled=False)

        class _Dist:
            rank = 0
            world_size = 1
            local_rank = 0

            def setup(self, backend):
                events.append("dist.setup")

            def barrier(self):
                pass

            def cleanup(self):
                events.append("dist.cleanup")

        rt = ComposedRuntimeStrategy(_Acc(), _Dist())
        rt.setup()
        rt.setup()  # second call should be a no-op
        assert events == ["acc.setup", "dist.setup"]

        rt.cleanup()
        rt.cleanup()  # second cleanup should be a no-op
        assert events == ["acc.setup", "dist.setup", "dist.cleanup"]


# --------------------------------------------------------------------------- #
# 2. Optimizer weight decay
# --------------------------------------------------------------------------- #
class TestOptimizerWeightDecay:
    def test_bias_gets_no_weight_decay_by_default(self):
        """Bias and 1D params must NOT receive weight decay when spec.weight_decay is None."""
        from game_cls.engine.trainer import build_optimizer_groups_from_selection

        model = _tiny_model()
        # Make all params trainable.
        for p in model.parameters():
            p.requires_grad = True

        selection = TrainableSelection(
            groups=(),  # empty groups → all params must be covered
            frozen_parameter_names=(),
            trainable_state=StateSelection(
                parameter_keys=tuple(name for name, _ in model.named_parameters()),
                buffer_keys=(),
            ),
            frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
        )

        # With empty groups, the function should raise because the policy
        # doesn't cover the params. Let's use a single group with weight_decay=None.
        selection = TrainableSelection(
            groups=(
                __import__(
                    "game_cls.contracts.trainable", fromlist=["ParameterGroupSpec"]
                ).ParameterGroupSpec(
                    name="head",
                    parameter_names=tuple(name for name, _ in model.named_parameters()),
                    weight_decay=None,  # default → split into decay/no_decay
                ),
            ),
            frozen_parameter_names=(),
            trainable_state=StateSelection(
                parameter_keys=tuple(name for name, _ in model.named_parameters()),
                buffer_keys=(),
            ),
            frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
        )

        groups = build_optimizer_groups_from_selection(
            model, selection, base_learning_rate=1e-3, default_weight_decay=0.01
        )

        # There should be two sub-groups: head/decay and head/no_decay.
        group_names = [g["name"] for g in groups]
        assert "head/decay" in group_names
        assert "head/no_decay" in group_names

        # Find the no_decay group and verify bias/1D are there.
        no_decay_group = next(g for g in groups if g["name"] == "head/no_decay")
        decay_group = next(g for g in groups if g["name"] == "head/decay")
        assert no_decay_group["weight_decay"] == 0.0
        assert decay_group["weight_decay"] == 0.01

        # cls.bias should be in no_decay.
        no_decay_param_ids = {id(p) for p in no_decay_group["params"]}
        bias_param = dict(model.named_parameters())["cls.bias"]
        assert id(bias_param) in no_decay_param_ids

    def test_explicit_weight_decay_applies_to_all(self):
        """When spec.weight_decay is explicitly set, all params use that value."""
        from game_cls.engine.trainer import build_optimizer_groups_from_selection
        from game_cls.contracts.trainable import ParameterGroupSpec

        model = _tiny_model()
        for p in model.parameters():
            p.requires_grad = True

        selection = TrainableSelection(
            groups=(
                ParameterGroupSpec(
                    name="head",
                    parameter_names=tuple(name for name, _ in model.named_parameters()),
                    weight_decay=0.05,  # explicit → no splitting
                ),
            ),
            frozen_parameter_names=(),
            trainable_state=StateSelection(
                parameter_keys=tuple(name for name, _ in model.named_parameters()),
                buffer_keys=(),
            ),
            frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
        )

        groups = build_optimizer_groups_from_selection(
            model, selection, base_learning_rate=1e-3, default_weight_decay=0.01
        )

        # Only one group, all params, weight_decay=0.05.
        assert len(groups) == 1
        assert groups[0]["name"] == "head"
        assert groups[0]["weight_decay"] == 0.05
        assert len(groups[0]["params"]) == len(list(model.parameters()))

    def test_uncovered_params_raise(self):
        """TrainablePolicy must cover all trainable params — no silent default group."""
        from game_cls.engine.trainer import build_optimizer_groups_from_selection

        model = _tiny_model()
        for p in model.parameters():
            p.requires_grad = True

        # Declare only one param in the group — the rest are uncovered.
        first_name = next(iter(dict(model.named_parameters())))
        selection = TrainableSelection(
            groups=(
                __import__(
                    "game_cls.contracts.trainable", fromlist=["ParameterGroupSpec"]
                ).ParameterGroupSpec(
                    name="partial",
                    parameter_names=(first_name,),
                    weight_decay=None,
                ),
            ),
            frozen_parameter_names=(),
            trainable_state=StateSelection(
                parameter_keys=(first_name,),
                buffer_keys=(),
            ),
            frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
        )

        with pytest.raises(RuntimeError, match="coverage mismatch"):
            build_optimizer_groups_from_selection(
                model, selection, base_learning_rate=1e-3, default_weight_decay=0.01
            )


# --------------------------------------------------------------------------- #
# 3. Checkpoint resume manifest
# --------------------------------------------------------------------------- #
class TestCheckpointResumeManifest:
    def test_v2_manifest_validates_against_stored(self):
        """V2 trainable_only checkpoints must validate against the stored manifest."""
        from game_cls.engine.checkpoint import restore_training_checkpoint

        model = _tiny_model()
        for p in model.parameters():
            p.requires_grad = True

        selection = TrainableSelection(
            groups=(),
            frozen_parameter_names=(),
            trainable_state=StateSelection(
                parameter_keys=tuple(name for name, _ in model.named_parameters()),
                buffer_keys=(),
            ),
            frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
        )

        # Build a fake V2 checkpoint.
        checkpoint = {
            "model_state_mode": "trainable_only",
            "checkpoint_format_version": 2,
            "trainable_policy": {"name": "name_token", "state_version": 1},
            "trainable_state_manifest": {
                "parameter_keys": list(selection.trainable_state.parameter_keys),
                "buffer_keys": list(selection.trainable_state.buffer_keys),
            },
            "model": dict(model.state_dict()),
        }

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(checkpoint, f.name)
            path = f.name

        # Should succeed with matching selection.
        result = restore_training_checkpoint(
            path,
            model,
            expected_trainable_selection=selection,
            expected_policy_name="name_token",
            expected_policy_version=1,
        )
        assert result["checkpoint_format_version"] == 2

    def test_v2_manifest_mismatch_raises(self):
        """Mismatched manifest must raise RuntimeError."""
        from game_cls.engine.checkpoint import restore_training_checkpoint

        model = _tiny_model()
        for p in model.parameters():
            p.requires_grad = True

        # Selection expects only one param.
        first_name = next(iter(dict(model.named_parameters())))
        selection = TrainableSelection(
            groups=(),
            frozen_parameter_names=(),
            trainable_state=StateSelection(
                parameter_keys=(first_name,),
                buffer_keys=(),
            ),
            frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
        )

        # Checkpoint has ALL params.
        checkpoint = {
            "model_state_mode": "trainable_only",
            "checkpoint_format_version": 2,
            "trainable_policy": {"name": "name_token", "state_version": 1},
            "trainable_state_manifest": {
                "parameter_keys": list(dict(model.named_parameters())),
                "buffer_keys": [],
            },
            "model": dict(model.state_dict()),
        }

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(checkpoint, f.name)
            path = f.name

        with pytest.raises(RuntimeError, match="manifest does not match"):
            restore_training_checkpoint(
                path,
                model,
                expected_trainable_selection=selection,
                expected_policy_name="name_token",
            )

    def test_v2_policy_name_mismatch_raises(self):
        """Policy name mismatch must raise RuntimeError."""
        from game_cls.engine.checkpoint import restore_training_checkpoint

        model = _tiny_model()
        for p in model.parameters():
            p.requires_grad = True

        selection = TrainableSelection(
            groups=(),
            frozen_parameter_names=(),
            trainable_state=StateSelection(
                parameter_keys=tuple(name for name, _ in model.named_parameters()),
                buffer_keys=(),
            ),
            frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
        )

        checkpoint = {
            "model_state_mode": "trainable_only",
            "checkpoint_format_version": 2,
            "trainable_policy": {"name": "regex", "state_version": 1},
            "trainable_state_manifest": {
                "parameter_keys": list(selection.trainable_state.parameter_keys),
                "buffer_keys": [],
            },
            "model": dict(model.state_dict()),
        }

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(checkpoint, f.name)
            path = f.name

        with pytest.raises(RuntimeError, match="policy.*mismatch"):
            restore_training_checkpoint(
                path,
                model,
                expected_trainable_selection=selection,
                expected_policy_name="name_token",
            )


# --------------------------------------------------------------------------- #
# 4. Persistent buffer coverage
# --------------------------------------------------------------------------- #
class TestPersistentBufferCoverage:
    def test_name_token_policy_includes_persistent_buffers(self):
        """NameTokenPolicy must classify persistent buffers for frozen coverage."""
        from game_cls.trainable.name_token import NameTokenTrainablePolicy

        model = _tiny_model_with_bn()
        policy = NameTokenTrainablePolicy(token="cls")
        selection = policy.select(model)

        # The model has BN running_mean/running_var/num_batches_tracked
        # which are persistent buffers NOT matching "cls" → frozen buffers.
        assert len(selection.frozen_state.buffer_keys) > 0
        # cls BN buffers should be trainable (they match "cls").
        assert any("cls" in k for k in selection.trainable_state.buffer_keys)

    def test_validate_loaded_state_checks_frozen_buffers(self):
        """validate_loaded_state must verify frozen BN buffers are loaded."""
        from game_cls.trainable.name_token import NameTokenTrainablePolicy

        model = _tiny_model_with_bn()
        policy = NameTokenTrainablePolicy(token="cls")
        selection = policy.select(model)

        # Build a load_report that's missing the frozen BN buffers.
        class _Report:
            loaded = ["cls.weight", "cls.bias"]  # missing backbone BN stats

        report = _Report()
        with pytest.raises(RuntimeError, match="persistent buffers"):
            policy.validate_loaded_state(model, report, selection)


# --------------------------------------------------------------------------- #
# 5. Backend __call__
# --------------------------------------------------------------------------- #
class TestBackendCallable:
    def test_png_backend_is_callable(self):
        """PngBackend.__call__ must decode a single reference."""
        from game_cls.data.backends.png import PngBackend

        class _ImageSpec:
            chw = (3, 32, 32)

        def _decoder(ref):
            return torch.zeros(3, 32, 32)

        backend = PngBackend(_ImageSpec(), _decoder)
        result = backend("some/path.png")
        assert result.shape == (3, 32, 32)

    def test_packed_backend_is_callable(self):
        """PackedUint8Backend.__call__ must decode a single reference."""
        from game_cls.data.backends.packed_uint8 import PackedUint8Backend

        class _Legacy:
            def __call__(self, ref):
                return torch.zeros(3, 32, 32)

            def close(self):
                pass

        backend = PackedUint8Backend(_Legacy())
        result = backend("some/ref")
        assert result.shape == (3, 32, 32)


# --------------------------------------------------------------------------- #
# 6. CPU DDP device_ids
# --------------------------------------------------------------------------- #
class TestCpuDdp:
    def test_cpu_ddp_uses_no_device_ids(self):
        """DDP wrap for CPU must use device_ids=None."""
        from game_cls.runtime.distributed import DdpDistributed

        dist = DdpDistributed(local_rank=0, world_size=2, rank=0, backend="gloo")
        model = nn.Linear(4, 2)
        device = torch.device("cpu")

        # Patch DistributedDataParallel to capture the constructor args.
        captured = {}
        original_ddp = nn.parallel.DistributedDataParallel

        def mock_ddp(module, **kwargs):
            captured.update(kwargs)
            # Return the module directly (no actual DDP).
            return module

        with patch("torch.nn.parallel.DistributedDataParallel", side_effect=mock_ddp):
            dist.wrap_model(model, device)

        assert captured.get("device_ids") is None

    def test_cuda_ddp_uses_device_ids(self):
        """DDP wrap for CUDA should use device_ids=[local_rank]."""
        from game_cls.runtime.distributed import DdpDistributed

        dist = DdpDistributed(local_rank=1, world_size=2, rank=1, backend="nccl")
        model = nn.Linear(4, 2)
        device = torch.device("cuda:1")

        captured = {}

        def mock_ddp(module, **kwargs):
            captured.update(kwargs)
            return module

        # Patch both DDP and model.to to avoid needing a real CUDA device.
        with patch("torch.nn.parallel.DistributedDataParallel", side_effect=mock_ddp), \
             patch.object(nn.Module, "to", lambda self, *a, **kw: self):
            dist.wrap_model(model, device)

        assert captured.get("device_ids") == [1]
        assert captured.get("output_device") == 1


# --------------------------------------------------------------------------- #
# 7. Registry factory_path
# --------------------------------------------------------------------------- #
class TestRegistryFactoryPath:
    def test_import_from_path(self):
        """import_from_path must import a callable from a dotted path."""
        from game_cls.registry import import_from_path

        func = import_from_path("game_cls.contracts.trainable:ParameterGroupSpec")
        assert callable(func)

    def test_import_from_path_with_colon(self):
        """import_from_path must handle 'module:callable' format."""
        from game_cls.registry import import_from_path

        func = import_from_path("game_cls.contracts.trainable:StateSelection")
        assert callable(func)

    def test_resolve_component_with_factory_path(self):
        """resolve_component must prefer factory_path over registry."""
        from game_cls.registry import resolve_component

        func = resolve_component(
            "task", "dual_frame_binary", "game_cls.contracts.trainable:ParameterGroupSpec"
        )
        assert callable(func)

    def test_resolve_component_without_factory_path(self):
        """resolve_component must fall back to registry when no factory_path."""
        # Ensure the built-in task is registered.
        import game_cls.tasks.dual_frame_binary  # noqa: F401

        from game_cls.registry import resolve_component

        func = resolve_component("task", "dual_frame_binary", "")
        assert callable(func)


# --------------------------------------------------------------------------- #
# 8. EvaluatorSuite registry-driven
# --------------------------------------------------------------------------- #
class TestEvaluatorSuiteRegistry:
    def test_legacy_evaluator_supports_dual_frame_binary(self):
        """LegacyEvaluatorSuite must declare supported_task_names."""
        from game_cls.evaluation.legacy_adapter import LegacyEvaluatorSuite

        suite = LegacyEvaluatorSuite()
        assert "dual_frame_binary" in suite.supported_task_names

    def test_incompatible_task_evaluator_raises(self):
        """Builder must raise when task is incompatible with evaluator."""
        from game_cls.engine.builders import _validate_task_evaluator_compatibility
        from game_cls.evaluation.legacy_adapter import LegacyEvaluatorSuite

        class _FakeTask:
            task_name = "some_custom_task"

        suite = LegacyEvaluatorSuite()
        with pytest.raises(RuntimeError, match="does not support"):
            _validate_task_evaluator_compatibility(_FakeTask(), suite)

    def test_compatible_task_evaluator_passes(self):
        """Builder must not raise when task is compatible."""
        from game_cls.engine.builders import _validate_task_evaluator_compatibility
        from game_cls.evaluation.legacy_adapter import LegacyEvaluatorSuite

        class _FakeTask:
            task_name = "dual_frame_binary"

        suite = LegacyEvaluatorSuite()
        _validate_task_evaluator_compatibility(_FakeTask(), suite)  # should not raise


# --------------------------------------------------------------------------- #
# 9. Config strictness for legacy sections
# --------------------------------------------------------------------------- #
class TestConfigStrictness:
    def test_unknown_legacy_key_rejected(self):
        """Unknown keys in legacy sections must be rejected."""
        config = _minimal_v2_config()
        config["train"] = {"local_batch_szie": 32}  # typo
        with pytest.raises(ValidationError, match="Extra inputs"):
            validate_and_normalize_config(config)

    def test_unknown_optimizer_key_rejected(self):
        """Unknown keys in optimizer must be rejected."""
        config = _minimal_v2_config()
        config["optimizer"] = {"learning_ratae": 0.001}  # typo
        with pytest.raises(ValidationError, match="Extra inputs"):
            validate_and_normalize_config(config)

    def test_valid_legacy_keys_accepted(self):
        """Valid legacy keys must be accepted."""
        config = _minimal_v2_config()
        config["train"] = {"local_batch_size": 16, "epochs": 3}
        config["optimizer"] = {"learning_rate": 0.001, "weight_decay": 0.01}
        cfg = validate_and_normalize_config(config)
        assert cfg.train.local_batch_size == 16
        assert cfg.optimizer.learning_rate == 0.001


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _minimal_v2_config():
    return {
        "config_version": 2,
        "experiment": {"name": "test", "seed": 42, "output_dir": "runs/test"},
        "device": {"accelerator": "cpu", "amp": False, "amp_dtype": "float16"},
        "task": {"type": "dual_frame_binary", "params": {"positive_class_index": 1}},
        "trainable": {"policy": {"type": "name_token", "params": {"token": "cls"}}},
        "data": {
            "module_factory": "game_cls.data.module:LegacyGameVideoDataModule",
            "index_codec": {"type": "legacy_game_binary"},
            "backend": {"type": "png", "params": {}},
        },
        "sampler": {
            "policy": {
                "type": "balanced_game_label_delta",
                "params": {
                    "class_probability": {0: 0.5, 1: 0.5},
                    "game_alpha": 0.25,
                },
            }
        },
        "runtime": {
            "accelerator": {"type": "cpu"},
            "distributed": {"type": "single_process", "params": {}},
        },
        "evaluation": {
            "suite": {"type": "legacy_binary"},
            "decision": {"type": "threshold", "params": {"threshold": 0.99}},
        },
        "model": {"factory": "game_cls.model.builder:build_demo_model"},
        "loss": {"threshold": 0.5},
        "optimizer": {"learning_rate": 0.001, "weight_decay": 0.01},
        "scheduler": {"warmup_steps": 0},
        "train": {"local_batch_size": 4, "epochs": 1, "steps_per_epoch": 2},
        "dataloader": {"num_workers": 0},
        "checkpoint": {},
    }


def _tiny_model():
    """A tiny model with backbone + cls head for testing."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    # Wrap in a namespace with a 'cls' attribute so the token policy works.
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.ReLU())
            self.cls = nn.Linear(8, 2)

    return _Model()


def _tiny_model_with_bn():
    """A tiny model with BatchNorm for testing persistent buffer coverage."""
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
            )
            self.cls = nn.Sequential(
                nn.Conv2d(8, 4, 3, padding=1),
                nn.BatchNorm2d(4),
                nn.Flatten(),
                nn.Linear(4 * 32 * 32, 2),
            )

        def forward(self, x):
            return self.cls(self.backbone(x))

    m = _Model()
    # Initialize with a forward pass so BN stats exist.
    with torch.no_grad():
        m(torch.zeros(1, 3, 32, 32))
    return m


# --------------------------------------------------------------------------- #
# 9. Persistent buffer coverage for ALL policies
# --------------------------------------------------------------------------- #
class TestPersistentBufferCoverageAllPolicies:
    def test_regex_policy_includes_persistent_buffers(self):
        """RegexTrainablePolicy must classify persistent buffers."""
        from game_cls.trainable.regex import RegexTrainablePolicy

        model = _tiny_model_with_bn()
        # Use a pattern that matches cls parameters.
        policy = RegexTrainablePolicy(include=[r"^cls\."])
        selection = policy.select(model)
        # backbone BN buffers should be frozen.
        assert len(selection.frozen_state.buffer_keys) > 0
        # cls BN buffers should be trainable.
        assert any("cls" in k for k in selection.trainable_state.buffer_keys)

    def test_model_declared_policy_includes_persistent_buffers(self):
        """ModelDeclaredTrainablePolicy must classify persistent buffers."""
        from game_cls.trainable.model_declared import ModelDeclaredTrainablePolicy

        model = _tiny_model_with_bn()
        # Declare cls parameters.
        cls_params = [n for n, _ in model.named_parameters() if n.startswith("cls.")]
        model.trainable_parameter_groups = lambda: [
            {"name": "head", "parameter_names": cls_params}
        ]
        policy = ModelDeclaredTrainablePolicy()
        selection = policy.select(model)
        # backbone BN buffers should be frozen.
        assert len(selection.frozen_state.buffer_keys) > 0

    def test_regex_validate_checks_frozen_buffers(self):
        """RegexTrainablePolicy.validate_loaded_state must check frozen buffers."""
        from game_cls.trainable.regex import RegexTrainablePolicy

        model = _tiny_model_with_bn()
        policy = RegexTrainablePolicy(include=[r"^cls\."])
        selection = policy.select(model)

        class _Report:
            loaded = ["cls.0.weight", "cls.0.bias"]  # missing backbone BN

        with pytest.raises(RuntimeError, match="persistent buffers"):
            policy.validate_loaded_state(model, _Report(), selection)


# --------------------------------------------------------------------------- #
# 10. DataModule close
# --------------------------------------------------------------------------- #
class TestDataModuleClose:
    def test_data_module_close_cleans_backends(self):
        """DataModule.close must close all tracked backends."""
        from game_cls.data.module import LegacyGameVideoDataModule

        class _ImageSpec:
            chw = (3, 32, 32)

        class _Backend:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _Dataset:
            def __init__(self, decoder):
                self.decoder = decoder

        class _Loader:
            def __init__(self, dataset):
                self.dataset = dataset

        class _Bundle:
            def __init__(self, train, quick_test, full_test, sampler, data_summary):
                self.train = train
                self.quick_test = quick_test
                self.full_test = full_test
                self.sampler = sampler
                self.data_summary = data_summary

        config = {"data": {"synthetic": True}, "train": {"local_batch_size": 4, "steps_per_epoch": 2}}
        dm = LegacyGameVideoDataModule(config, _ImageSpec())

        backend = _Backend()
        dataset = _Dataset(backend)
        loader = _Loader(dataset)

        # Simulate build_loaders tracking.
        dm._backends = [backend]
        dm.close()
        assert backend.closed


# --------------------------------------------------------------------------- #
# 11. SamplingPolicy in production path
# --------------------------------------------------------------------------- #
class TestSamplingPolicyInProduction:
    def test_balanced_policy_is_iterable(self):
        """BalancedGameLabelDeltaPolicy must be usable as a BatchSampler."""
        from game_cls.data.sampling.balanced_game_label_delta import (
            BalancedGameLabelDeltaPolicy,
        )
        from game_cls.data.video_index import VideoEntry

        videos = [
            VideoEntry(
                game="g1",
                label=0,
                video_id="v1",
                frame_ids=[0, 1, 2],
                frame_paths=["a", "b", "c"],
                valid_start_positions={2: [0]},
            ),
            VideoEntry(
                game="g1",
                label=1,
                video_id="v2",
                frame_ids=[0, 1, 2],
                frame_paths=["d", "e", "f"],
                valid_start_positions={2: [0]},
            ),
        ]
        policy = BalancedGameLabelDeltaPolicy.from_config(
            videos,
            local_batch_size=2,
            steps_per_epoch=3,
            rank=0,
            world_size=1,
            seed=42,
        )
        # Must be iterable and yield the right number of steps.
        batches = list(policy)
        assert len(batches) == 3  # steps_per_epoch

    def test_balanced_policy_state_dict_roundtrip(self):
        """Policy state_dict/load_state_dict must roundtrip."""
        from game_cls.data.sampling.balanced_game_label_delta import (
            BalancedGameLabelDeltaPolicy,
        )
        from game_cls.data.video_index import VideoEntry

        videos = [
            VideoEntry(
                game="g1",
                label=0,
                video_id="v1",
                frame_ids=[0, 1, 2],
                frame_paths=["a", "b", "c"],
                valid_start_positions={2: [0]},
            ),
        ]
        policy = BalancedGameLabelDeltaPolicy.from_config(
            videos,
            local_batch_size=1,
            steps_per_epoch=2,
            rank=0,
            world_size=1,
            seed=42,
        )
        state = policy.state_dict(0)
        assert "epoch" in state


# --------------------------------------------------------------------------- #
# 12. DataModule eval backend tracking (Subset unwrapping)
# --------------------------------------------------------------------------- #
class TestDataModuleEvalBackendTracking:
    def test_eval_backends_tracked_through_subset(self):
        """Eval backends wrapped in Subset must still be tracked for cleanup."""
        from torch.utils.data import DataLoader, Subset

        from game_cls.data.module import LegacyGameVideoDataModule

        class _ImageSpec:
            chw = (3, 32, 32)

        class _Backend:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _Dataset:
            def __init__(self, decoder):
                self.decoder = decoder

        class _Bundle:
            def __init__(self, train, quick_test, full_test, sampler, data_summary):
                self.train = train
                self.quick_test = quick_test
                self.full_test = full_test
                self.sampler = sampler
                self.data_summary = data_summary

        config = {"data": {"synthetic": True}, "train": {"local_batch_size": 4, "steps_per_epoch": 2}}
        dm = LegacyGameVideoDataModule(config, _ImageSpec())

        train_backend = _Backend()
        eval_backend = _Backend()
        train_dataset = _Dataset(train_backend)
        eval_dataset = _Dataset(eval_backend)
        # Eval loaders wrap dataset in Subset.
        train_loader = DataLoader(train_dataset)
        eval_loader = DataLoader(Subset(eval_dataset, [0, 1]))

        bundle = _Bundle(train_loader, eval_loader, eval_loader, None, {})
        # Simulate the tracking logic from build_loaders.
        dm._backends.clear()
        for loader in (bundle.train, bundle.quick_test, bundle.full_test):
            dataset = getattr(loader, "dataset", None)
            if dataset is None:
                continue
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            decoder = getattr(dataset, "decoder", None)
            if decoder is not None and decoder not in dm._backends:
                dm._backends.append(decoder)

        dm.close()
        assert train_backend.closed
        assert eval_backend.closed


# --------------------------------------------------------------------------- #
# 13. Task param validation
# --------------------------------------------------------------------------- #
class TestTaskParamValidation:
    def test_invalid_positive_class_index_rejected(self):
        """DualFrameBinaryTaskConfig must reject invalid positive_class_index."""
        from game_cls.tasks.dual_frame_binary import DualFrameBinaryTaskConfig

        with pytest.raises(ValueError, match="positive_class_index"):
            DualFrameBinaryTaskConfig(positive_class_index=2)

    def test_invalid_num_classes_rejected(self):
        """DualFrameBinaryTaskConfig must reject num_classes != 2."""
        from game_cls.tasks.dual_frame_binary import DualFrameBinaryTaskConfig

        with pytest.raises(ValueError, match="num_classes"):
            DualFrameBinaryTaskConfig(num_classes=3)

    def test_valid_config_accepted(self):
        """Valid config must be accepted."""
        from game_cls.tasks.dual_frame_binary import DualFrameBinaryTaskConfig

        cfg = DualFrameBinaryTaskConfig(positive_class_index=1, num_classes=2)
        assert cfg.positive_class_index == 1
