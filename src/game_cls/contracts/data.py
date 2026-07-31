from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class TargetValue:
    kind: str
    value: Any


@dataclass(frozen=True)
class SequenceEntry:
    """A logical training or evaluation unit, independent of physical storage.

    The default dual-frame codec maps one video pair to one SequenceEntry, but
    the logical schema is intentionally general (arbitrary frame counts, groups,
    temporal index) so future tasks need not reshape the data pipeline.
    """

    sequence_id: str
    frame_ids: Any
    frame_references: Any
    target: TargetValue
    groups: Mapping[str, Any]
    temporal_index: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupKey:
    fields: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Backend contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BackendCapabilities:
    batch_decode: bool
    random_access: bool
    supports_preview: bool
    spawn_safe: bool


@dataclass(frozen=True)
class BackendSpec:
    """Serializable description of how to build a backend inside a worker."""

    factory: str
    params: dict
    split: str


class FrameBackend(Protocol):
    @property
    def backend_name(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def get(self, reference: Any) -> Any: ...

    def get_many(self, references: Any) -> Any: ...

    def preview(self, reference: Any, output_path: Any) -> None: ...

    def close(self) -> None: ...

    def __call__(self, reference: Any) -> Any:
        """Decode a single frame reference (convenience alias for ``get``).

        The legacy single-sample dataset path calls ``decoder(reference)``.
        Implementations should return the same result as ``get(reference)``.
        """
        ...


class DataBackendFactory(Protocol):
    @property
    def config_model(self) -> type: ...

    def create(self, config: Any, image_spec: Any, split: str) -> FrameBackend: ...


# --------------------------------------------------------------------------- #
# Index codec
# --------------------------------------------------------------------------- #
class IndexCodec(Protocol):
    @property
    def schema_name(self) -> str: ...

    @property
    def schema_version(self) -> int: ...

    def read_sequences(
        self, path: Any, temporal_requirements: Any
    ) -> list[SequenceEntry]: ...

    def write_sequences(self, entries: list[SequenceEntry], path: Any) -> None: ...


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
class SamplingCatalog(Protocol):
    def available_temporal_keys(self) -> Any: ...

    def available_groups(self, temporal_key: Any) -> Any: ...

    def available_targets(self, temporal_key: Any, group: Any) -> Any: ...

    def sample_request(
        self, rng: Any, temporal_key: Any, group: Any, target: Any
    ) -> Any: ...


@dataclass(frozen=True)
class SamplingContext:
    epoch: int
    step: int
    global_batch_size: int
    world_size: int
    seed: int


@dataclass(frozen=True)
class SamplingFeedback:
    sample_id: str
    loss: float | None = None
    error_type: str | None = None
    score: float | None = None
    checkpoint_step: int = 0


class SamplingPolicy(Protocol):
    @property
    def policy_name(self) -> str: ...

    @property
    def state_version(self) -> int: ...

    def sample_rank_batch(
        self, catalog: SamplingCatalog, context: SamplingContext
    ) -> list[Any]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...

    def update_feedback(self, feedback: list[SamplingFeedback]) -> None: ...


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #
@dataclass
class LoaderBundle:
    """The dataloaders and sampler for a training run."""
    train: Any
    quick_test: Any
    full_test: Any
    sampler: Any
    data_summary: dict


class DataModule(Protocol):
    """Owns the data pipeline: indexing, backend, sampling, DataLoader.

    The trainer talks to this facade and never branches on backend name or
    sampling algorithm directly.
    """

    @property
    def config(self) -> Any: ...

    @property
    def image_spec(self) -> Any: ...

    def build_loaders(
        self,
        *,
        runtime: Any,
        state_mode: str = "full",
    ) -> LoaderBundle:
        """Build the dataloaders for a training run."""
        ...

    def close(self) -> None:
        """Close any resources held by the data module (backends, memmaps, ...)."""
        ...
