from __future__ import annotations

from typing import Any

from ..contracts.data import GroupKey, SequenceEntry, TargetValue


def video_entry_to_sequence(
    video: Any,
    delta: int,
    start_position: int,
    frame0_id: int,
    frame1_id: int,
    frame0_reference: Any,
    frame1_reference: Any,
) -> SequenceEntry:
    """Map a ``VideoEntry`` pair to a logical :class:`SequenceEntry`.

    The default dual-frame codec maps one video pair to one sequence. The
    original flat fields are preserved in ``metadata`` for backward compatibility
    (USERPLAN §6.4).
    """
    target = TargetValue(kind="class_index", value=int(video.label))
    groups = {"game": video.game, "label": int(video.label)}
    return SequenceEntry(
        sequence_id=f"{video.game}/{video.label}/{video.video_id}",
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
            # Compatibility fields (USERPLAN §6.4):
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
