"""End-to-end negative-subtype reduction through evaluate() (step6 P0).

The helpers in test_negative_subtype_metrics.py are covered in isolation, so a
reduction that builds the subtype counters and then throws them away stayed
invisible. These tests drive the real ``evaluate()`` reduction and assert the
counters reach both the metrics dict and the grouped rows.
"""

from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is required for these tests
    torch = None
    nn = None


CATALOGS = {
    "game": ["A"],
    "game_label": [("A", 0), ("A", 1)],
    "video": [("A", 0, "n1"), ("A", 0, "n2"), ("A", 1, "p1")],
    "game_label_subtype": [
        ("A", 0, "flat_floor"),
        ("A", 0, "wooden_bridge"),
        ("A", 1, "_untyped"),
    ],
}


class _ConstantMargin(nn.Module if nn is not None else object):
    """Returns a fixed margin per sample, driven by the batch's subtype id.

    Subtype 0 always scores below the cutoff, subtype 1 always above -- so the
    two negative subtypes land at FPR 0.0 and 1.0 and the worst-subtype
    reduction has an unambiguous answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._margins: torch.Tensor | None = None

    def forward(self, first, second):
        assert self._margins is not None
        half = self._margins / 2.0
        return torch.stack((-half, half), dim=1)


def _batch(subtype_ids: list[int], labels: list[int], video_ids: list[int]) -> dict:
    count = len(labels)
    return {
        "images": torch.zeros((count, 2, 3, 4, 4), dtype=torch.uint8),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "game_id": torch.zeros(count, dtype=torch.int64),
        "game_label_id": torch.tensor(labels, dtype=torch.int64),
        "video_group_id": torch.tensor(video_ids, dtype=torch.int64),
        "game_label_subtype_id": torch.tensor(subtype_ids, dtype=torch.int64),
        "meta": [{"game": "A", "video_id": f"v{index}"} for index in range(count)],
    }


def _distributed_subtype_worker(
    rank: int, world_size: int, init_method: str, output_dir: str
) -> None:
    """Each rank scores one negative of each subtype plus one positive."""
    import json
    from pathlib import Path

    import torch.distributed as dist

    from game_cls.engine.evaluator import evaluate

    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    try:
        batch = _batch(subtype_ids=[0, 1, 2], labels=[0, 0, 1], video_ids=[0, 1, 2])
        model = _ConstantMargin()
        model._margins = torch.tensor([-4.0, 4.0, 4.0])
        result = evaluate(
            model,
            _Loader([batch], CATALOGS),
            torch.device("cpu"),
            0.5,
            group_catalogs=CATALOGS,
            distributed=True,
            rank=rank,
            world_size=world_size,
            evaluation_kind="full",
            full_auc_mode="histogram",
            auc_histogram_bins=128,
        )
        if rank == 0:
            payload = {
                "metrics": result.metrics,
                "rows": (result.grouped_metrics or {}).get("by_game_label_subtype"),
            }
            Path(output_dir).write_text(json.dumps(payload), encoding="utf-8")
    finally:
        dist.destroy_process_group()


class _Loader:
    """Minimal iterable with the ``.dataset.group_catalogs`` attribute chain."""

    def __init__(self, batches: list[dict], catalogs: dict | None) -> None:
        self._batches = batches
        self.dataset = type("_DS", (), {"group_catalogs": catalogs})()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluatorSubtypeReductionTests(unittest.TestCase):
    def test_production_collate_preserves_subtype_ids(self) -> None:
        from game_cls.data.collate import pair_collate

        sample = {
            "images": torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
            "label": 0,
            "meta": {},
            "game_label_subtype_id": 7,
        }
        batch = pair_collate([sample, dict(sample, game_label_subtype_id=9)])
        self.assertEqual(batch["game_label_subtype_id"].tolist(), [7, 9])

    def _evaluate(self, catalogs: dict | None):
        from game_cls.engine.evaluator import evaluate

        # Negatives: subtype 0 scores -4 (correct), subtype 1 scores +4 (false
        # positive). Positive: subtype 2 scores +4 (correct).
        batch = _batch(
            subtype_ids=[0, 0, 1, 1, 2, 2],
            labels=[0, 0, 0, 0, 1, 1],
            video_ids=[0, 0, 1, 1, 2, 2],
        )
        model = _ConstantMargin()
        model._margins = torch.tensor([-4.0, -4.0, 4.0, 4.0, 4.0, 4.0])
        return evaluate(
            model,
            _Loader([batch], catalogs),
            torch.device("cpu"),
            0.5,
            group_catalogs=catalogs,
            evaluation_kind="full",
            full_auc_mode="histogram",
        )

    def test_subtype_counters_reach_metrics_and_rows(self) -> None:
        """Non-distributed vectorized path: the counters must survive."""
        result = self._evaluate(CATALOGS)
        metrics = result.metrics
        assert metrics is not None
        self.assertTrue(metrics["subtype_grouping_available"])
        # subtype 1 has fp=2, tn=0 -> FPR 1.0; subtype 0 has fp=0, tn=2 -> 0.0.
        self.assertAlmostEqual(metrics["worst_subtype_fpr_at_decision_threshold"], 1.0)
        self.assertAlmostEqual(
            metrics["worst_subtype_recall_at_decision_threshold"], 1.0
        )
        # Every populated group appears, including the positive-label group
        # whose negative denominator is 0.
        self.assertEqual(
            metrics["subtype_negative_counts"],
            {"A::0::flat_floor": 2, "A::0::wooden_bridge": 2, "A::1::_untyped": 0},
        )
        rows = (result.grouped_metrics or {}).get("by_game_label_subtype") or []
        self.assertEqual(len(rows), 3)
        by_key = {
            (row["game"], row["label"], row["negative_subtype"]): row for row in rows
        }
        self.assertIn(("A", 0, "wooden_bridge"), by_key)
        self.assertIn(("A", 1, "_untyped"), by_key)

    def test_metrics_survive_the_report_writer(self) -> None:
        """write_evaluation_report json.dumps() the metrics dict, so every key
        in it has to be JSON-safe -- tuple keys raise TypeError."""
        import json
        import tempfile
        from pathlib import Path

        from game_cls.reports.error_writer import write_evaluation_report

        result = self._evaluate(CATALOGS)
        metrics = dict(result.metrics or {})
        json.dumps(metrics)  # must not raise
        with tempfile.TemporaryDirectory() as directory:
            write_evaluation_report(
                directory,
                metrics,
                result.grouped_metrics,
                lightweight=False,
            )
            written = json.loads(
                (Path(directory) / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                written["subtype_negative_counts"]["A::0::wooden_bridge"], 2
            )
            csv_text = (
                Path(directory) / "metrics_by_game_label_subtype.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("wooden_bridge", csv_text)

    def test_absent_catalog_reports_unavailable(self) -> None:
        """No catalog -> metadata path, subtype metrics null but flagged."""
        result = self._evaluate(None)
        metrics = result.metrics
        assert metrics is not None
        self.assertFalse(metrics["subtype_grouping_available"])
        self.assertIsNone(metrics["worst_subtype_fpr_at_decision_threshold"])
        self.assertEqual((result.grouped_metrics or {})["by_game_label_subtype"], [])

    def test_dropped_counters_raise(self) -> None:
        """A catalog plus scored samples but no counters is an internal bug."""
        from game_cls.engine import evaluator

        original = evaluator._build_subtype_group
        evaluator._build_subtype_group = lambda catalogs, array: {}
        try:
            with self.assertRaises(RuntimeError) as caught:
                self._evaluate(CATALOGS)
        finally:
            evaluator._build_subtype_group = original
        self.assertIn("lost the negative-subtype counters", str(caught.exception))

    def test_constrained_selection_accepts_the_restored_metric(self) -> None:
        """The gate that silently rejected everything now sees a real value."""
        from game_cls.engine.training.selection import _selection_eligible

        result = self._evaluate(CATALOGS)
        metrics = dict(result.metrics or {})
        config = {"selection_mode": "constrained", "max_worst_subtype_fpr": 0.5}
        # worst FPR is 1.0 > 0.5 -> genuinely ineligible.
        self.assertFalse(_selection_eligible(metrics, config))
        config["max_worst_subtype_fpr"] = 1.0
        self.assertTrue(_selection_eligible(metrics, config))


@unittest.skipIf(
    torch is None or not torch.distributed.is_available(),
    "torch distributed is not available",
)
class DistributedSubtypeReductionTests(unittest.TestCase):
    def test_subtype_counts_aggregate_across_ranks(self) -> None:
        """The subtype array rides the same all-reduce as the other catalogs.

        Each rank holds one negative per subtype, so a reduction that skipped
        the subtype array would report half the counts.
        """
        import json
        import tempfile
        from pathlib import Path

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "metrics.json"
            mp.spawn(
                _distributed_subtype_worker,
                args=(2, (root / "rendezvous").as_uri(), str(output)),
                nprocs=2,
                join=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            self.assertEqual(metrics["sample_count"], 6)
            self.assertTrue(metrics["subtype_grouping_available"])
            self.assertAlmostEqual(
                metrics["worst_subtype_fpr_at_decision_threshold"], 1.0
            )
            # Both ranks contributed: each negative subtype has 2 negatives.
            counts = {
                tuple(key) if isinstance(key, list) else key: value
                for key, value in (metrics["subtype_negative_counts"] or {}).items()
            }
            self.assertEqual(sorted(counts.values()), [0, 2, 2])
            self.assertEqual(len(payload["rows"]), 3)


@unittest.skipIf(torch is None, "torch is not installed")
class SubtypeGroupingWarningTests(unittest.TestCase):
    def test_warning_mentions_selection_when_gate_is_set(self) -> None:
        import io
        from contextlib import redirect_stdout

        from game_cls.engine.training.loaders import (
            _warn_subtype_grouping_unavailable,
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _warn_subtype_grouping_unavailable(0, True, 0.01, "synthetic data")
        text = buffer.getvalue()
        self.assertIn("group_by_negative_subtype is enabled", text)
        self.assertIn("reject EVERY checkpoint", text)

    def test_silent_when_disabled_or_off_rank(self) -> None:
        import io
        from contextlib import redirect_stdout

        from game_cls.engine.training.loaders import (
            _warn_subtype_grouping_unavailable,
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _warn_subtype_grouping_unavailable(0, False, 0.01, "x")
            _warn_subtype_grouping_unavailable(1, True, 0.01, "x")
        self.assertEqual(buffer.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
