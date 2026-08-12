from pathlib import Path

from clstrainer_lite.data import (
    DistributedEvalSampler,
    ImagePairDataset,
    scan_image_root,
    split_train_val_videos,
)
from tests.helpers import make_pair_dataset


def test_video_level_train_val_split_is_deterministic_and_disjoint(tmp_path: Path):
    make_pair_dataset(tmp_path, frames=5, size=(20, 30))
    videos = scan_image_root(tmp_path / "train")
    train_a, val_a = split_train_val_videos(videos, val_ratio=0.25, seed=123)
    train_b, val_b = split_train_val_videos(videos, val_ratio=0.25, seed=123)

    assert [video.key for video in train_a] == [video.key for video in train_b]
    assert [video.key for video in val_a] == [video.key for video in val_b]
    assert {video.key for video in train_a}.isdisjoint({video.key for video in val_a})
    assert len(train_a) == 6
    assert len(val_a) == 2

    dataset = ImagePairDataset(
        val_a,
        source_root=tmp_path / "train",
        split_name="val",
        delta=2,
        image_size=(16, 24),
    )
    assert len(dataset) == 6
    sample = dataset[0]
    assert tuple(sample["images"].shape) == (2, 3, 16, 24)

    shards = [list(DistributedEvalSampler(dataset, rank=rank, world_size=5)) for rank in range(5)]
    flattened = [index for shard in shards for index in shard]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(len(dataset)))


def test_split_keeps_both_labels_in_validation_on_small_balanced_data(tmp_path: Path):
    make_pair_dataset(
        tmp_path,
        frames=4,
        train_videos_per_label=2,
        test_videos_per_label=1,
    )
    videos = scan_image_root(tmp_path / "train")
    train_videos, val_videos = split_train_val_videos(videos, val_ratio=0.10, seed=9)
    assert {video.label for video in train_videos} == {0, 1}
    assert {video.label for video in val_videos} == {0, 1}


def test_train_delta_is_online_and_does_not_expand_index(tmp_path: Path):
    from clstrainer_lite.data import build_datasets

    make_pair_dataset(tmp_path, frames=5, size=(20, 30))
    config = {
        "experiment": {"seed": 123},
        "data": {
            "train_root": str(tmp_path / "train"),
            "test_root": str(tmp_path / "test"),
            "val_ratio": 0.25,
            "train_delta_range": [1, 3],
            # Force delta=2 so the test is deterministic while still using the
            # online-delta code path.
            "train_delta_probabilities": [0.0, 1.0, 0.0],
            "eval_delta": 2,
            "image_size": [16, 24],
            "strict_filenames": True,
        },
        "augment": {"enabled": False},
    }
    train, val, test = build_datasets(config)
    assert train.deltas == (1, 2, 3)
    assert val.deltas == (2,)
    assert test.deltas == (2,)

    # Six train videos remain after the 25% video-level split.  With five
    # contiguous frames and max_delta=3, each video contributes only two start
    # positions.  The old pre-expanded implementation would have contributed
    # 9 pairs/video; online delta keeps the compact 2 positions/video.
    assert len(train) == 12
    assert train.summary()["delta_mode"] == "online"
    assert train.summary()["sampling_positions"] == 12
    assert len(val) == 6
    assert len(test) == 12
    assert {train[index]["delta"] for index in range(len(train))} == {2}
