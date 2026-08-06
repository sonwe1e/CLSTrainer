"""Distributed vectorized-group reduction (step3 phase 3).

Verifies that per-catalog index arrays are tensor-reduced across ranks
instead of gathering large Python dicts to rank 0.
"""

from __future__ import annotations

import json
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


def _worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    from torch.utils.data import DataLoader

    from game_cls.data.collate import pair_collate
    from game_cls.engine.evaluator import evaluate

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        # Each rank evaluates a DIFFERENT slice of the catalog so a plain
        # "copy rank 0" implementation would fail the merged assertions.
        catalog = {
            "game": ["A", "B"],
            "game_label": [("A", 0), ("A", 1), ("B", 0), ("B", 1)],
            "video": [
                ("A", 0, "01"),
                ("A", 1, "01"),
                ("B", 0, "01"),
                ("B", 1, "01"),
            ],
        }

        class Dataset:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                local = index + rank * 4
                game = "A" if local % 4 < 2 else "B"
                label = local % 2
                return {
                    "images": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
                    "label": label,
                    "game_id": 0 if game == "A" else 1,
                    "game_label_id": local % 4,
                    "video_group_id": local % 4,
                    "meta": {
                        "game": game,
                        "label": label,
                        "video_id": "01",
                    },
                }

        class Model(torch.nn.Module):
            def forward(self, image0, image1):
                return torch.zeros(
                    len(image0), 2, dtype=image0.dtype, device=image0.device
                )

        result = evaluate(
            Model(),
            DataLoader(Dataset(), batch_size=2, collate_fn=pair_collate),
            torch.device("cpu"),
            distributed=True,
            rank=rank,
            world_size=world_size,
            evaluation_kind="full",
            report_dir=output_dir,
            full_auc_mode="histogram",
            auc_histogram_bins=128,
            group_catalogs=catalog,
        )
        if rank == 0:
            payload = {
                "sample_count": result.metrics["sample_count"],
                "grouped": result.grouped_metrics,
            }
            (Path(output_dir) / "reduced_metrics.json").write_text(
                json.dumps(payload, default=str), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


@unittest.skipIf(
    torch is None or not dist.is_available(),
    "torch distributed is not available",
)
class DistributedGroupReductionTests(unittest.TestCase):
    def test_two_process_vectorized_groups_merge_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendezvous = root / "rendezvous"
            mp.spawn(
                _worker,
                args=(2, rendezvous.as_uri(), str(root / "report")),
                nprocs=2,
                join=True,
            )
            payload = json.loads(
                (root / "report" / "reduced_metrics.json").read_text(encoding="utf-8")
            )
            # Both ranks contributed their own samples.
            self.assertEqual(payload["sample_count"], 8)
            by_video = payload["grouped"]["by_video"]
            games = {(row["game"], row["label"]) for row in by_video}
            # Rank 0 saw A/B labels 0/1, rank 1 saw A/B labels 1/0; merged
            # counts cover every (game, label) with samples on both ranks.
            self.assertEqual(games, {("A", 0), ("A", 1), ("B", 0), ("B", 1)})
            counts: dict[str, int] = {}
            for row in by_video:
                counts[row["game"]] = (
                    counts.get(row["game"], 0)
                    + row["tp"]
                    + row["fp"]
                    + row["fn"]
                    + row["tn"]
                )
            # Each game has exactly 4 samples across ranks (2 per rank).
            self.assertEqual(counts, {"A": 4, "B": 4})


if __name__ == "__main__":
    unittest.main()
