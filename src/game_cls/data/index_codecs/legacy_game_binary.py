from __future__ import annotations

from typing import Any

from ...contracts.data import SequenceEntry
from ..logical_schema import pair_entry_to_sequence, video_entry_to_sequence
from .base import IndexCodecBase


class LegacyGameBinaryIndexCodec(IndexCodecBase):
    """Reads the current production Parquet video index into ``SequenceEntry``.

    The codec does **not** require rebuilding the existing index: it reads the
    same ``VideoEntry`` rows the trainer already consumes and maps each video
    to a ``SequenceEntry`` (USERPLAN §6.3 correct design). Each sequence
    stores the video's valid start positions per delta so that ``SampleRequest``
    objects can be generated at batch time without materializing all pairs up
    front. This keeps index memory proportional to the number of videos rather
    than the number of pairs.

    Round-tripping preserves every original field in ``metadata``.
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
            # Verify the video has at least one valid start for some delta.
            has_valid = any(
                len(video.valid_start_positions.get(delta, [])) > 0
                for delta in temporal_requirements
            )
            if not has_valid:
                continue
            # Map the whole video to one SequenceEntry (no pair materialization).
            entries.append(video_entry_to_sequence(video))
        return entries

    def read_pairs(
        self, path: Any, temporal_requirements: Any
    ) -> list[SequenceEntry]:
        """Read all pairs (legacy per-pair materialization path).

        This is kept for backward compatibility and tests. For production use,
        prefer :meth:`read_sequences` which returns one SequenceEntry per video.
        """
        from ..video_index import read_video_entries_parquet

        videos = read_video_entries_parquet(path, temporal_requirements)
        entries: list[SequenceEntry] = []
        for video in videos:
            for delta in temporal_requirements:
                starts = video.valid_start_positions.get(delta)
                if starts is None or len(starts) == 0:
                    continue
                for start_position in starts:
                    frame0_id, frame1_id, ref0, ref1 = video.pair_paths(delta, int(start_position))
                    entries.append(
                        pair_entry_to_sequence(
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
