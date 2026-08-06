from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

SEVERITIES = frozenset({"error", "warning", "info"})


def _severity(value: Any, *, field_name: str) -> str:
    result = str(value).lower()
    if result not in SEVERITIES:
        raise ValueError(
            f"{field_name} must be one of {sorted(SEVERITIES)}, got {value!r}"
        )
    return result


@dataclass(frozen=True)
class ScanPolicy:
    frame_extensions: frozenset[str]
    ignore_directory_prefixes: tuple[str, ...]
    ignore_directory_names: frozenset[str]
    ignore_file_globs: tuple[str, ...]
    unexpected_nested_directory_severity: str = "warning"
    ignored_example_limit: int = 20

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> ScanPolicy:
        extensions = []
        for item in data_config.get("frame_extensions", [".png"]):
            extension = str(item).lower()
            extensions.append(
                extension if extension.startswith(".") else f".{extension}"
            )
        policy = cls(
            frame_extensions=frozenset(extensions),
            ignore_directory_prefixes=tuple(
                str(item)
                for item in data_config.get(
                    "ignore_directory_prefixes", ["_", "."]
                )
            ),
            ignore_directory_names=frozenset(
                str(item).casefold()
                for item in data_config.get(
                    "ignore_directory_names",
                    ["__pycache__", "cache", "caches", "tmp", "temp"],
                )
            ),
            ignore_file_globs=tuple(
                str(item)
                for item in data_config.get("ignore_file_globs", [])
            ),
            unexpected_nested_directory_severity=_severity(
                data_config.get(
                    "unexpected_nested_directory_severity", "warning"
                ),
                field_name="unexpected_nested_directory_severity",
            ),
            ignored_example_limit=int(
                data_config.get("ignored_example_limit", 20)
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.frame_extensions:
            raise ValueError("frame_extensions must not be empty")
        if self.ignored_example_limit < 0:
            raise ValueError("ignored_example_limit must be non-negative")

    def ignore_directory(self, path: Path) -> bool:
        return (
            path.name.casefold() in self.ignore_directory_names
            or any(
                path.name.startswith(prefix)
                for prefix in self.ignore_directory_prefixes
            )
        )

    def ignore_file(self, path: Path) -> bool:
        name = path.name.casefold()
        return any(
            fnmatch(name, pattern.casefold())
            for pattern in self.ignore_file_globs
        )

    def is_frame_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.frame_extensions

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_extensions": sorted(self.frame_extensions),
            "ignore_directory_prefixes": list(
                self.ignore_directory_prefixes
            ),
            "ignore_directory_names": sorted(
                self.ignore_directory_names
            ),
            "ignore_file_globs": list(self.ignore_file_globs),
            "unexpected_nested_directory_severity": (
                self.unexpected_nested_directory_severity
            ),
            "ignored_example_limit": self.ignored_example_limit,
        }


@dataclass(frozen=True)
class DuplicatePolicy:
    same_label_cross_split: str = "warning"
    same_label_within_split: str = "warning"
    cross_label_same_content: str = "error"
    same_basename: str = "info"

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> DuplicatePolicy:
        raw = data_config.get("duplicate_policy", {})
        return cls(
            same_label_cross_split=_severity(
                raw.get("same_label_cross_split", "warning"),
                field_name="duplicate_policy.same_label_cross_split",
            ),
            same_label_within_split=_severity(
                raw.get("same_label_within_split", "warning"),
                field_name="duplicate_policy.same_label_within_split",
            ),
            cross_label_same_content=_severity(
                raw.get("cross_label_same_content", "error"),
                field_name="duplicate_policy.cross_label_same_content",
            ),
            same_basename=_severity(
                raw.get("same_basename", "info"),
                field_name="duplicate_policy.same_basename",
            ),
        )


@dataclass
class ScanFindings:
    ignored_example_limit: int = 20
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)
    ignored_counts: dict[str, int] = field(default_factory=dict)
    ignored_examples: dict[str, list[str]] = field(default_factory=dict)

    def add(
        self,
        severity: str,
        kind: str,
        path: str | Path,
        error: str,
    ) -> None:
        severity = _severity(severity, field_name="finding severity")
        getattr(self, f"{severity}s").append(
            {
                "severity": severity,
                "kind": kind,
                "path": str(path),
                "error": error,
            }
        )

    def add_ignored(self, kind: str, path: str | Path) -> None:
        self.ignored_counts[kind] = self.ignored_counts.get(kind, 0) + 1
        examples = self.ignored_examples.setdefault(kind, [])
        if len(examples) < self.ignored_example_limit:
            examples.append(str(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "ignored": {
                "counts": dict(sorted(self.ignored_counts.items())),
                "examples": {
                    key: value
                    for key, value in sorted(self.ignored_examples.items())
                },
            },
        }
