"""DDP + staged partial unfreeze (step6 P0-1).

``model.trainable_rules`` flips ``requires_grad`` mid-run, but DDP binds its
Reducer buckets to the trainable set that exists when the wrapper is built.
Without rebuilding the wrapper at the unfreeze boundary the newly unfrozen
gradients are never all-reduced and every rank silently trains its own copy
of those tensors.

The two-process tests below deliberately feed each rank DIFFERENT data, so an
implementation that skips the re-wrap ends up with rank-dependent weights and
fails the equality assertion. ``test_without_rewrap_diverges`` is the negative
control that proves the assertion has teeth.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ImportError:
    torch = None
    dist = None
    mp = None

# cls is trainable from step 0; the backbone conv joins at step 4.
RULES = {
    "cls_head": {"pattern": r"^cls\.", "lr_scale": 1.0, "unfreeze_at_step": 0},
    "backbone_late": {
        "pattern": r"^backbone\.0\.",
        "lr_scale": 1.0,
        "unfreeze_at_step": 4,
    },
}
UNFREEZE_STEP = 4
TOTAL_STEPS = 8


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_dir: str,
    rewrap: int,
) -> None:
    from game_cls.engine.training.loop import (
        _transfer_optimizer_state,
        _unfreeze_boundary,
        _wrap_distributed,
    )
    from game_cls.model.builder import build_demo_model
    from game_cls.model.trainable_rules import (
        apply_trainable_state,
        build_optimizer_parameter_groups,
        parse_rules,
        resolve_trainable_names,
    )

    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(0)  # identical initial weights on every rank
        device = torch.device("cpu")
        rules = parse_rules(RULES)
        bare_model = build_demo_model({})
        apply_trainable_state(bare_model, rules, 0)
        bare_model.to(device)
        model = _wrap_distributed(bare_model, device, rank)
        optimizer = torch.optim.AdamW(
            build_optimizer_parameter_groups(
                bare_model, rules, step=0, weight_decay=1e-4, base_lr=0.05
            )
        )
        current_trainable = {
            name
            for name, parameter in bare_model.named_parameters()
            if parameter.requires_grad
        }
        rewraps = 0
        for global_step in range(TOTAL_STEPS):
            if _unfreeze_boundary(rules, global_step):
                after, _ = resolve_trainable_names(bare_model, rules, global_step)
                if set(after) != current_trainable:
                    current_trainable = set(after)
                    apply_trainable_state(bare_model, rules, global_step)
                    if rewrap:
                        optimizer.zero_grad(set_to_none=True)
                        model = _wrap_distributed(bare_model, device, rank)
                        rewraps += 1
                    old_optimizer = optimizer
                    optimizer = torch.optim.AdamW(
                        build_optimizer_parameter_groups(
                            bare_model,
                            rules,
                            step=global_step,
                            weight_decay=1e-4,
                            base_lr=0.05,
                        )
                    )
                    _transfer_optimizer_state(old_optimizer, optimizer)
            # Rank-dependent inputs and labels: without gradient all-reduce
            # the two ranks pull the weights in different directions.
            images = torch.full((2, 2, 3, 4, 4), 0.25 * (rank + 1))
            labels = torch.tensor([rank % 2, (rank + 1) % 2])
            optimizer.zero_grad(set_to_none=True)
            logits = model(images[:, 0], images[:, 1])
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        payload = {
            "rewraps": rewraps,
            "params": {
                name: parameter.detach().flatten().tolist()
                for name, parameter in bare_model.named_parameters()
            },
            "trainable": sorted(
                name
                for name, parameter in bare_model.named_parameters()
                if parameter.requires_grad
            ),
        }
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _end_to_end_worker(
    rank: int,
    world_size: int,
    port: int,
    output_dir: str,
    max_steps: int = TOTAL_STEPS,
    resume_from: str = "",
) -> None:
    """Run the real training loop under gloo and dump the final weights."""
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    from game_cls.config import load_config
    from game_cls.engine.trainer import run_training
    from game_cls.engine.training import loop as loop_module

    # Only rank 0 writes checkpoints, so capture each rank's own live module to
    # compare weights across ranks. The wrapper call count also proves the
    # re-wrap actually happened at the unfreeze boundary.
    captured: dict[str, object] = {"model": None, "wraps": 0}
    original_wrap = loop_module._wrap_distributed

    def _spy_wrap(bare_model, device, local_rank):
        captured["model"] = bare_model
        captured["wraps"] = int(captured["wraps"]) + 1
        return original_wrap(bare_model, device, local_rank)

    loop_module._wrap_distributed = _spy_wrap

    config = load_config("configs/cuda_debug.yaml")
    config["experiment"]["output_dir"] = str(Path(output_dir) / "run")
    config["device"]["accelerator"] = "cpu"
    config["distributed"] = {"enabled": True, "backend": "gloo"}
    config["model"]["trainable_rules"] = RULES
    config["train"].update(
        {
            "max_steps": max_steps,
            "steps_per_epoch": max_steps,
            "local_batch_size": 2,
            "log_every_steps": max_steps,
            # The frozen-parameter verifier must survive a staged unfreeze.
            "verify_frozen_parameters": True,
        }
    )
    if resume_from:
        config["train"]["resume_path"] = resume_from
    config["evaluation"].update(
        {
            "train_probe_every_steps": 0,
            "val_quick_every_steps": 0,
            "val_full_every_steps": 0,
            "val_full_at_end": False,
        }
    )
    config["checkpoint"].update(
        {"save_last_every_steps": 0, "save_topk": 0, "save_best_selection": False}
    )
    config.pop("early_stopping", None)
    suffix = "resume" if resume_from else ""
    try:
        run_training(config)
    finally:
        loop_module._wrap_distributed = original_wrap
    bare_model = captured["model"]
    assert bare_model is not None
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"e2e{suffix}_rank{rank}.json").write_text(
        json.dumps(
            {
                "wraps": captured["wraps"],
                "params": {
                    name: parameter.detach().flatten().tolist()
                    for name, parameter in bare_model.named_parameters()  # type: ignore[union-attr]
                },
                "trainable": sorted(
                    name
                    for name, parameter in bare_model.named_parameters()  # type: ignore[union-attr]
                    if parameter.requires_grad
                ),
            }
        ),
        encoding="utf-8",
    )


@unittest.skipIf(
    torch is None or dist is None or not dist.is_available(),
    "torch distributed is not available",
)
class DdpStagedUnfreezeTests(unittest.TestCase):
    def _run(self, root: Path, rewrap: int) -> list[dict]:
        rendezvous = root / f"rendezvous{rewrap}"
        reports = root / f"reports{rewrap}"
        mp.spawn(
            _worker,
            args=(2, rendezvous.as_uri(), str(reports), rewrap),
            nprocs=2,
            join=True,
        )
        return [
            json.loads((reports / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in (0, 1)
        ]

    def test_rewrap_keeps_unfrozen_parameters_in_sync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rank0, rank1 = self._run(Path(directory), rewrap=1)
            self.assertEqual(rank0["rewraps"], 1)
            self.assertIn("backbone.0.weight", rank0["trainable"])
            # Every parameter -- including the one unfrozen mid-run -- must be
            # bitwise identical across ranks.
            for name, values in rank0["params"].items():
                self.assertEqual(values, rank1["params"][name], f"{name} diverged")
            # And the newly unfrozen tensor actually moved off its init (1.0),
            # so the equality above is not vacuous.
            self.assertTrue(
                any(
                    abs(value - 1.0) > 1e-6
                    for value in rank0["params"]["backbone.0.weight"]
                ),
                "backbone.0.weight never trained; the sync assertion is vacuous",
            )

    def test_without_rewrap_diverges(self) -> None:
        """Negative control: skipping the re-wrap desynchronizes the ranks."""
        with tempfile.TemporaryDirectory() as directory:
            rank0, rank1 = self._run(Path(directory), rewrap=0)
            self.assertEqual(rank0["rewraps"], 0)
            self.assertNotEqual(
                rank0["params"]["backbone.0.weight"],
                rank1["params"]["backbone.0.weight"],
            )

    def test_end_to_end_training_loop_stays_in_sync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(
                _end_to_end_worker,
                args=(2, 29517, str(root / "e2e")),
                nprocs=2,
                join=True,
            )
            payloads = [
                json.loads(
                    (root / "e2e" / f"e2e_rank{rank}.json").read_text(encoding="utf-8")
                )
                for rank in (0, 1)
            ]
            self.assertIn("backbone.0.weight", payloads[0]["trainable"])
            for key, values in payloads[0]["params"].items():
                self.assertEqual(values, payloads[1]["params"][key], f"{key} diverged")
            # Wrapped once at startup and once at the unfreeze boundary.
            self.assertEqual(payloads[0]["wraps"], 2)
            self.assertTrue(
                any(
                    abs(value - 1.0) > 1e-6
                    for value in payloads[0]["params"]["backbone.0.weight"]
                ),
                "backbone.0.weight never trained after the unfreeze boundary",
            )

    def test_end_to_end_resume_past_unfreeze_boundary(self) -> None:
        """Resume must restore the saved mask using UNPREFIXED names.

        ``trainable_state`` is written from ``unwrap_model(...)``, so comparing
        it against the DDP-wrapped ``named_parameters()`` yields an empty
        intersection: every parameter is frozen and the restore key-set check
        raises. This exercises the path for a resume taken past the boundary.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "e2e"
            mp.spawn(
                _end_to_end_worker,
                args=(2, 29519, str(out), TOTAL_STEPS, ""),
                nprocs=2,
                join=True,
            )
            checkpoint = out / "run" / "checkpoints" / "checkpoint_last.pth"
            self.assertTrue(checkpoint.is_file())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            # The mask stored past the boundary includes the backbone tensor.
            self.assertIn("backbone.0.weight", saved["trainable_state"])
            # Resume with a longer budget; must not raise and must stay synced.
            mp.spawn(
                _end_to_end_worker,
                args=(2, 29521, str(out), TOTAL_STEPS + 4, str(checkpoint)),
                nprocs=2,
                join=True,
            )
            payloads = [
                json.loads(
                    (out / f"e2eresume_rank{rank}.json").read_text(encoding="utf-8")
                )
                for rank in (0, 1)
            ]
            self.assertIn("backbone.0.weight", payloads[0]["trainable"])
            for key, values in payloads[0]["params"].items():
                self.assertEqual(values, payloads[1]["params"][key], f"{key} diverged")


@unittest.skipIf(torch is None, "torch is not installed")
class SingleProcessStagedUnfreezeTests(unittest.TestCase):
    def test_anchored_pattern_matches_unwrapped_names(self) -> None:
        """Anchored rules must resolve against unprefixed parameter names."""
        from game_cls.model.builder import build_demo_model
        from game_cls.model.trainable_rules import apply_trainable_state, parse_rules

        model = build_demo_model({})
        apply_trainable_state(model, parse_rules(RULES), 0)
        by_name = dict(model.named_parameters())
        self.assertTrue(by_name["cls.weight"].requires_grad)
        self.assertFalse(by_name["backbone.0.weight"].requires_grad)

    def test_prune_unfrozen_from_snapshot(self) -> None:
        from game_cls.engine.training.loop import _prune_unfrozen_from_snapshot
        from game_cls.model.builder import build_demo_model
        from game_cls.model.freeze_policy import snapshot_frozen_parameters
        from game_cls.model.trainable_rules import apply_trainable_state, parse_rules

        rules = parse_rules(RULES)
        model = build_demo_model({})
        apply_trainable_state(model, rules, 0)
        snapshot = snapshot_frozen_parameters(model)
        self.assertIn("backbone.0.weight", snapshot)
        apply_trainable_state(model, rules, UNFREEZE_STEP)
        pruned = _prune_unfrozen_from_snapshot(snapshot, model)
        assert pruned is not None
        self.assertNotIn("backbone.0.weight", pruned)
        self.assertIsNone(_prune_unfrozen_from_snapshot(None, model))

    def test_boundary_at_step_zero_is_not_a_change(self) -> None:
        """A rule with unfreeze_at_step == 0 must not rebuild on batch 1.

        ``_unfreeze_boundary`` fires at step 0, so the loop's real guard is the
        comparison against the trainable set the optimizer was built from.
        """
        from game_cls.engine.training.loop import _unfreeze_boundary
        from game_cls.model.builder import build_demo_model
        from game_cls.model.trainable_rules import (
            apply_trainable_state,
            parse_rules,
            resolve_trainable_names,
        )

        rules = parse_rules(RULES)
        model = build_demo_model({})
        apply_trainable_state(model, rules, 0)
        current_trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(_unfreeze_boundary(rules, 0))
        at_zero, _ = resolve_trainable_names(model, rules, 0)
        # Step 0 is a no-op for the optimizer: the set already matches.
        self.assertEqual(set(at_zero), current_trainable)
        # The real boundary adds exactly the backbone tensor.
        at_unfreeze, _ = resolve_trainable_names(model, rules, UNFREEZE_STEP)
        self.assertEqual(set(at_unfreeze) - current_trainable, {"backbone.0.weight"})


if __name__ == "__main__":
    unittest.main()
