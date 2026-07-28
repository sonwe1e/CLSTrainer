from __future__ import annotations

import unittest

from game_cls.data.records import FrameRecord, build_pairs, summarize_videos


def frame(frame_id: int, *, video_id: str = "01", label: int = 0) -> FrameRecord:
    return FrameRecord(
        sample_id=f"x:{frame_id}",
        split="train",
        game="game_A",
        label=label,
        video_id=video_id,
        frame_id=frame_id,
        path=f"{video_id}{frame_id:05d}.png",
        width=208,
        height=448,
        channels=3,
        file_size=1,
    )


class PairDatasetTests(unittest.TestCase):
    def test_pairs_require_real_target_frame(self) -> None:
        frames = [frame(1), frame(2), frame(4)]
        pairs = build_pairs(frames, delta=2)
        self.assertEqual([(p.frame0_id, p.frame1_id) for p in pairs], [(2, 4)])

    def test_never_pairs_across_video_or_class(self) -> None:
        frames = [
            frame(1, video_id="01", label=0),
            frame(3, video_id="02", label=0),
            frame(3, video_id="01", label=1),
        ]
        self.assertEqual(build_pairs(frames, delta=2), [])

    def test_video_summary_counts_each_delta(self) -> None:
        summary = summarize_videos([frame(1), frame(2), frame(3), frame(4)])[0]
        self.assertEqual(summary.valid_pair_count_delta1, 3)
        self.assertEqual(summary.valid_pair_count_delta2, 2)
        self.assertEqual(summary.valid_pair_count_delta3, 1)


if __name__ == "__main__":
    unittest.main()

