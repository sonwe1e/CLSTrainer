from __future__ import annotations

from collections import Counter
import unittest

from game_cls.data.lazy_pair_dataset import build_eval_dataset
from game_cls.data.records import FrameRecord
from game_cls.data.video_index import build_video_entries
from game_cls.data.video_sampler import VideoBalancedPairBatchSampler


def frame(
    game: str, label: int, video_id: str, frame_id: int
) -> FrameRecord:
    return FrameRecord(
        sample_id=f"{game}:{label}:{video_id}:{frame_id}",
        split="train",
        game=game,
        label=label,
        video_id=video_id,
        frame_id=frame_id,
        path=f"{game}/{label}/{video_id}{frame_id:05d}.png",
        width=208,
        height=448,
        channels=3,
        file_size=1,
    )


def videos():
    frames = []
    for game in ("A", "B"):
        for label in (0, 1):
            for video_id, ids in (("01", (1, 2, 3, 4, 5)), ("02", (1, 3, 5))):
                frames.extend(frame(game, label, video_id, item) for item in ids)
    return build_video_entries(frames)


class VideoIndexSamplerTests(unittest.TestCase):
    def test_delta_first_sampling_tracks_requested_distribution(self) -> None:
        entries = videos()
        sampler = VideoBalancedPairBatchSampler(
            entries,
            local_batch_size=64,
            steps_per_epoch=300,
            seed=9,
            delta_probability={1: 0.15, 2: 0.70, 3: 0.15},
        )
        requests = [request for batch in sampler for request in batch]
        counts = Counter(request.delta for request in requests)
        self.assertAlmostEqual(counts[2] / len(requests), 0.70, delta=0.03)

    def test_resume_start_step_matches_uninterrupted_sequence(self) -> None:
        entries = videos()
        kwargs = dict(
            videos=entries,
            local_batch_size=4,
            steps_per_epoch=6,
            seed=33,
        )
        full = list(VideoBalancedPairBatchSampler(**kwargs))
        resumed = VideoBalancedPairBatchSampler(**kwargs)
        resumed.set_epoch(0, start_step=2)
        self.assertEqual(list(resumed), full[2:])

    def test_eval_index_is_uniform_limited_and_rank_disjoint(self) -> None:
        entries = videos()
        complete = build_eval_dataset(entries, 2)
        rank0 = build_eval_dataset(
            entries, 2, rank=0, world_size=2, max_pairs_per_video=2
        )
        rank1 = build_eval_dataset(
            entries, 2, rank=1, world_size=2, max_pairs_per_video=2
        )
        keys0 = set(
            zip(
                rank0.video_indices.tolist(),
                rank0.start_positions.tolist(),
            )
        )
        keys1 = set(
            zip(
                rank1.video_indices.tolist(),
                rank1.start_positions.tolist(),
            )
        )
        self.assertTrue(keys0.isdisjoint(keys1))
        self.assertLessEqual(len(rank0) + len(rank1), len(entries) * 2)
        self.assertLess(complete.index_nbytes, max(1, len(complete)) * 16)


if __name__ == "__main__":
    unittest.main()

