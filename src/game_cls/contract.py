"""The single persistent-artifact contract for CLSTrainer 5.0.0."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 5
PARQUET_CONTRACT_KEY = b"game_cls.contract_version"


class ContractError(ValueError):
    """Raised when an artifact is not written under the current contract."""


def require_contract(payload: Mapping[str, Any], artifact: str) -> None:
    """Require the one supported JSON/checkpoint contract."""
    value = payload.get("contract_version")
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ContractError(
            f"{artifact} requires contract_version={CONTRACT_VERSION}; "
            "rebuild it with CLSTrainer 5.0.0."
        )


def stamp_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying the current contract marker."""
    stamped = dict(payload)
    stamped["contract_version"] = CONTRACT_VERSION
    return stamped


def stamp_parquet_table(table):
    """Attach the contract marker to Arrow schema metadata."""
    metadata = dict(table.schema.metadata or {})
    metadata[PARQUET_CONTRACT_KEY] = str(CONTRACT_VERSION).encode("ascii")
    return table.replace_schema_metadata(metadata)


def require_parquet_contract(path: str | Path) -> None:
    """Validate contract metadata before reading parquet rows."""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    if metadata.get(PARQUET_CONTRACT_KEY) != str(CONTRACT_VERSION).encode("ascii"):
        raise ContractError(
            f"Parquet artifact {path} requires contract_version={CONTRACT_VERSION}; "
            "rebuild it with CLSTrainer 5.0.0."
        )
