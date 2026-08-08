"""NPU correctness stage-1 tests (step4 P0-1/P0-3).

1. No float64 device tensors may be created on the training path: an AST
   scan of the shipped package bans ``.double()`` and ``torch.float64``
   (the interval accumulators are float32 because NPU/CANN rejects
   float64 device tensors with ``k::double``).
2. ``_reduce_interval_accumulator`` derives every mean from float32 device
   accumulators (including ``threshold_weight_sum``).
3. ``threshold_weight_sum`` now participates in the distributed all-reduce,
   so a two-rank gloo run reports the *global* sample-weighted threshold
   weight instead of rank-local / world_size.
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Float64AstVisitor(ast.NodeVisitor):
    """Collect AST locations that create float64 device tensors."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        # ``x.double()`` — creates a float64 device tensor on the caller's
        # device. Also ``torch.float64`` / ``torch.float64(...)``.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "double":
            self.hits.append((node.lineno, ast.unparse(node)))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # ``torch.float64`` used as a dtype (host-side ``np.float64`` is
        # fine and NOT a device tensor).
        if (
            node.attr == "float64"
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
        ):
            self.hits.append((node.lineno, ast.unparse(node)))
        self.generic_visit(node)


class NoFloat64DeviceTensorTests(unittest.TestCase):
    def test_no_double_or_float64_in_shipped_package(self) -> None:
        offenders: list[tuple[str, int, str]] = []
        for path in sorted((REPO_ROOT / "src" / "game_cls").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            visitor = _Float64AstVisitor()
            visitor.visit(tree)
            for lineno, snippet in visitor.hits:
                offenders.append((str(path.relative_to(REPO_ROOT)), lineno, snippet))
        self.assertEqual(
            offenders,
            [],
            "NPU forbids float64 device tensors (k::double). Found: "
            + "; ".join(
                f"{path}:{line} {snippet}" for path, line, snippet in offenders
            ),
        )


class IntervalAccumulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch optional
            self.skipTest(f"torch unavailable: {exc}")
        self.torch = torch

    def test_accumulators_are_float32_and_reduce_to_float(self) -> None:
        from game_cls.engine.training.evaluation import (
            _new_interval_accumulator,
            _reduce_interval_accumulator,
        )

        device = self.torch.device("cpu")
        accum = _new_interval_accumulator(device)
        self.assertEqual(accum["loss_sum"].dtype, self.torch.float32)
        self.assertEqual(accum["ce_sum"].dtype, self.torch.float32)
        self.assertEqual(accum["threshold_sum"].dtype, self.torch.float32)
        self.assertEqual(accum["threshold_weight_sum"].dtype, self.torch.float32)
        self.assertEqual(accum["counts"].dtype, self.torch.int64)

        # Simulate one batch of 8 samples.
        accum["loss_sum"].add_(self.torch.tensor(2.5) * 8)
        accum["ce_sum"].add_(self.torch.tensor(1.5) * 8)
        accum["threshold_sum"].add_(self.torch.tensor(0.5) * 8)
        accum["threshold_weight_sum"].add_(self.torch.tensor(0.2) * 8)
        accum["samples"] += 8
        accum["counts"].add_(self.torch.tensor([4, 1, 1, 2], dtype=self.torch.int64))

        reduced = _reduce_interval_accumulator(accum, device)
        self.assertAlmostEqual(reduced["interval_loss"], 2.5)
        self.assertAlmostEqual(reduced["interval_ce"], 1.5)
        self.assertAlmostEqual(reduced["interval_threshold_loss"], 0.5)
        self.assertAlmostEqual(reduced["interval_threshold_weight"], 0.2)
        self.assertEqual(reduced["interval_samples"], 8)
        # All reduced means are plain Python floats (JSON-serializable).
        for key in (
            "interval_loss",
            "interval_ce",
            "interval_threshold_loss",
            "interval_threshold_weight",
        ):
            self.assertIsInstance(reduced[key], float)

    def test_threshold_weight_sum_reduces_across_ranks(self) -> None:
        try:
            import torch.distributed as dist
            import torch.multiprocessing as mp
        except ImportError as exc:  # pragma: no cover - torch optional
            self.skipTest(f"torch distributed unavailable: {exc}")
        if not dist.is_available():
            self.skipTest("torch.distributed not available")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendezvous = root / "rendezvous"
            mp.spawn(
                _threshold_weight_worker,
                args=(2, rendezvous.as_uri(), str(root)),
                nprocs=2,
                join=True,
            )
            payload = json.loads((root / "reduced.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["threshold_weight"], 0.3)
            self.assertEqual(payload["samples"], 16)
            self.assertAlmostEqual(payload["loss"], 1.0)


def _threshold_weight_worker(
    rank: int, world_size: int, init_method: str, out_dir: str
) -> None:
    import torch
    import torch.distributed as dist

    from game_cls.engine.training.evaluation import (
        _new_interval_accumulator,
        _reduce_interval_accumulator,
    )

    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    try:
        device = torch.device("cpu")
        accum = _new_interval_accumulator(device)
        # Rank-local: rank 0 weight 0.2, rank 1 weight 0.4; each sees
        # 8 samples. Global mean must be (0.2+0.4)/2 = 0.3.
        local_weight = 0.2 if rank == 0 else 0.4
        accum["loss_sum"].add_(torch.tensor(1.0) * 8)
        accum["ce_sum"].add_(torch.tensor(0.5) * 8)
        accum["threshold_sum"].add_(torch.tensor(0.1) * 8)
        accum["threshold_weight_sum"].add_(local_weight * 8)
        accum["samples"] += 8
        accum["counts"].add_(torch.tensor([4, 1, 1, 2], dtype=torch.int64))
        reduced = _reduce_interval_accumulator(accum, device)
        if rank == 0:
            payload = {
                "threshold_weight": reduced["interval_threshold_weight"],
                "loss": reduced["interval_loss"],
                "samples": reduced["interval_samples"],
            }
            (Path(out_dir) / "reduced.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
