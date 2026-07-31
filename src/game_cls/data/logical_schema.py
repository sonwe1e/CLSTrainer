from __future__ import annotations

from typing import Any

from ..contracts.data import GroupKey, SequenceEntry, TargetValue


def video_entry_to_sequence(video: Any) -> SequenceEntry:
    """Map a ``VideoEntry`` to a logical :class:`SequenceEntry`.

    The default dual-frame codec maps one video to one sequence (USERPLAN §6.3
    correct design). Each sequence stores the video's valid start positions per
    delta so that ``SampleRequest`` objects can be generated at batch time
    without materializing all pairs up front. This keeps index memory proportional
    to the number of videos rather than the number of pairs.

    The original flat fields are preserved in ``metadata`` for backward
    compatibility (USERPLAN §6.4).
    """
    target = TargetValue(kind="class_index", value=int(video.label))
    groups = {"game": video.game, "label": int(video.label)}
    # valid_start_positions: Mapping[delta, np.ndarray of start positions]
    valid_starts = {
        int(delta): positions.copy()
        for delta, positions in video.valid_start_positions.items()
        if len(positions) > 0
    }
    return SequenceEntry(
        sequence_id=f"{video.game}/{video.label}/{video.video_id}",
        frame_ids=tuple(int(fid) for fid in video.frame_ids),
        frame_references=tuple(video.frame_paths) if video.frame_paths else (),
        target=target,
        groups=groups,
        temporal_index=None,  # Set per SampleRequest at batch time
        metadata={
            "sequence_id": f"{video.game}/{video.label}/{video.video_id}",
            "target": target,
            "groups": dict(groups),
            "valid_start_positions": valid_starts,
            "frame_ids": tuple(int(fid) for fid in video.frame_ids),
            "frame_paths": tuple(video.frame_paths),
            "video_directory": video.video_directory,
            # Compatibility fields (USERPLAN §6.4):
            "game": video.game,
            "label": int(video.label),
            "video_id": video.video_id,
        },
    )


def pair_entry_to_sequence(
    video: Any,
    delta: int,
    start_position: int,
    frame0_id: int,
    frame1_id: int,
    frame0_reference: Any,
    frame1_reference: Any,
) -> SequenceEntry:
    """Map a single video pair to a logical :class:`SequenceEntry` (legacy).

    This is the per-pair materialization path used by
    ``LegacyGameBinaryIndexCodec.read_sequences``. It is kept for backward
    compatibility but the preferred path is :func:`video_entry_to_sequence`
    which does not materialize all pairs.
    """
    target = TargetValue(kind="class_index", value=int(video.label))
    groups = {"game": video.game, "label": int(video.label)}
    return SequenceEntry(
        sequence_id=f"{video.game}/{video.label}/{video.video_id}/{delta}/{start_position}",
        frame_ids=(frame0_id, frame1_id),
        frame_references=(frame0_reference, frame1_reference),
        target=target,
        groups=groups,
        temporal_index=delta,
        metadata={
            "sequence_id": f"{video.game}/{video.label}/{video.video_id}",
            "target": target,
            "groups": dict(groups),
            "frame_ids": (frame0_id, frame1_id),
            "frame_references": (frame0_reference, frame1_reference),
            "temporal_offsets": (0, delta),
            "game": video.game,
            "label": int(video.label),
            "video_id": video.video_id,
            "frame0_id": frame0_id,
            "frame1_id": frame1_id,
            "delta": delta,
        },
    )


# Default evaluation group keys (USERPLAN §6.5).
DEFAULT_GROUP_KEYS: tuple[GroupKey, ...] = (
    GroupKey(fields=("game",)),
    GroupKey(fields=("game", "label")),
    GroupKey(fields=("sequence_id",)),
)
