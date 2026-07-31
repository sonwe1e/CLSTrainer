from __future__ import annotations

from typing import Any

from ...contracts.data import SequenceEntry
from ..logical_schema import video_entry_to_sequence
from .base import IndexCodecBase


class LegacyGameBinaryIndexCodec(IndexCodecBase):
    """Reads the current production Parquet video index into ``SequenceEntry``.

    The codec does **not** require rebuilding the existing index: it reads the
    same ``VideoEntry`` rows the trainer already consumes and maps each legal
    (video, delta, start_position) triple to a ``SequenceEntry``
    (USERPLAN §6.3). Round-tripping preserves every original field in
    ``metadata``.
    """

    schema_name = "legacy_game_binary"
    schema_version = 1

    def read_sequences(
        self, path: Any, temporal_requirements: Any
    ) -> list[SequenceEntry]:
        from ..video_index import read_video_entries_parquet

        videos = read_video_entries_parquet(path, temporal_requirements)
        entries: list[SequenceEntry] = []
        for video in videos:
            for delta in temporal_requirements:
                starts = video.valid_start_positions.get(delta)
                if not starts:
                    continue
                for start_position in starts:
                    frame0_id, frame1_id, ref0, ref1 = video.pair_paths(delta, int(start_position))
                    entries.append(
                        video_entry_to_sequence(
                            video,
                            delta,
                            int(start_position),
                            frame0_id,
                            frame1_id,
                            ref0,
                            ref1,
                        )
                    )
        return entries

    def write_sequences(self, entries: list[SequenceEntry], path: Any) -> None:
        # The legacy codec is read-only over the production index; writing a
        # sequence index is deferred to a future dedicated tool.
        del entries, path
        raise NotImplementedError(
            "LegacyGameBinaryIndexCodec does not write indexes; use the existing "
            "video_index tools to regenerate the Parquet index."
        )
