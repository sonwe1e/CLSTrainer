from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ImportError:
    torch = None
    dist = None
    mp = None


def _distributed_eval_worker(
    rank: int, world_size: int, init_method: str, output_dir: str
) -> None:
    from torch.utils.data import DataLoader, Subset

    from game_cls.data.collate import pair_collate
    from game_cls.data.image_spec import ImageSpec
    from game_cls.engine.evaluator import evaluate
    from game_cls.engine.trainer import SyntheticPairDataset
    from game_cls.reports.error_writer import prepare_evaluation_directory

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        dataset = SyntheticPairDataset(
            16, ImageSpec(width=8, height=16, channels=3), 91
        )
        local = Subset(dataset, list(range(rank, len(dataset), world_size)))
        loader = DataLoader(local, batch_size=2, collate_fn=pair_collate)

        class Model(torch.nn.Module):
            def forward(self, image0, image1):
                score = image0[:, 0].float().mean(dim=(1, 2)) / 255 * 10 - 2
                return torch.stack((-score, score), dim=1)

        prepare_evaluation_directory(output_dir, rank)
        dist.barrier()
        result = evaluate(
            Model(),
            loader,
            torch.device("cpu"),
            distributed=True,
            rank=rank,
            world_size=world_size,
            evaluation_kind="full",
            report_dir=output_dir,
            full_auc_mode="histogram",
            auc_histogram_bins=128,
        )
        if rank == 0:
            (Path(output_dir) / "distributed_metrics.json").write_text(
                json.dumps(result.metrics), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


@unittest.skipIf(
    torch is None or not dist.is_available(),
    "torch distributed is not available",
)
class DistributedEvaluatorTests(unittest.TestCase):
    def test_two_process_evaluator_uses_non_overlapping_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendezvous = root / "rendezvous"
            mp.spawn(
                _distributed_eval_worker,
                args=(2, rendezvous.as_uri(), str(root / "report")),
                nprocs=2,
                join=True,
            )
            metrics = json.loads(
                (root / "report" / "distributed_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metrics["sample_count"], 16)
            self.assertEqual(metrics["auc_method"], "histogram_128_bins")
            shards = list(
                (root / "report" / "shards").glob("errors_rank_*.parquet")
            )
            self.assertEqual(len(shards), 2)


if __name__ == "__main__":
    unittest.main()
