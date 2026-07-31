from __future__ import annotations

from typing import Any

from ...contracts.data import IndexCodec, SequenceEntry


class IndexCodecBase:
    """Convenience base exposing the protocol's required attributes."""

    schema_name = "base"
    schema_version = 1

    def read_sequences(
        self, path: Any, temporal_requirements: Any
    ) -> list[SequenceEntry]:
        raise NotImplementedError

    def write_sequences(self, entries: list[SequenceEntry], path: Any) -> None:
        raise NotImplementedError
