"""Single source of truth for deployment release verification.

A benchmark PASS is valid only for one complete release identity.  Every
consumer (``release check`` and ``export``) routes through this module so a
cached boolean or a checkpoint-only comparison cannot weaken the contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from game_cls.contract import CONTRACT_VERSION, require_contract


class ReleaseVerificationError(RuntimeError):
    """The requested artifact has no verifiable benchmark PASS."""


@dataclass(frozen=True)
class ReleaseVerification:
    run_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    report_path: Path
    release_identity_sha256: str
    identity: dict[str, Any]


def release_identity(
    *, run_id: str, checkpoint_sha256: str, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from game_cls.reports.benchmark import (
        benchmark_source_fingerprint,
        canonical_config_sha256,
        challenge_bundle_fingerprint,
        gate_spec_fingerprint,
    )

    gate_metrics = (config.get("benchmark") or {}).get("gate_metrics") or {}
    challenge_sha, components = challenge_bundle_fingerprint(config)
    identity = {
        "run_id": str(run_id),
        "checkpoint_sha256": str(checkpoint_sha256),
        "resolved_config_sha256": canonical_config_sha256(config),
        "challenge_dataset_fingerprint": challenge_sha,
        "gate_spec_fingerprint": gate_spec_fingerprint(gate_metrics),
        "contract_version": CONTRACT_VERSION,
        "benchmark_source_sha256": benchmark_source_fingerprint(),
    }
    return identity, components


def release_identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_checkpoint_path(run_dir: Path, name: str) -> Path | None:
    direct = Path(name)
    if direct.is_file():
        return direct.resolve()
    checkpoints = run_dir / "checkpoints"
    for candidate in (f"model_{name}.pth", f"checkpoint_{name}.pth"):
        path = checkpoints / candidate
        if path.is_file():
            return path.resolve()
    return None


def _read_report(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        require_contract(payload, f"Benchmark report {path}")
        return payload
    except (OSError, ValueError):
        return None


def _identity_problems(
    payload: dict[str, Any],
    expected: dict[str, Any],
    current_components: dict[str, Any],
) -> list[str]:
    from game_cls.reports.benchmark import challenge_component_differences

    problems: list[str] = []
    for field, current in expected.items():
        recorded = payload.get(field)
        if not recorded:
            problems.append(f"report records no {field}")
        elif not current:
            problems.append(f"current {field} cannot be recomputed")
        elif str(recorded) != str(current):
            detail = f"{field} differs: report={recorded} current={current}"
            if field == "challenge_dataset_fingerprint":
                differences = challenge_component_differences(
                    payload.get("challenge_bundle_components"), current_components
                )
                if differences:
                    detail += "; changed: " + "; ".join(differences)
            problems.append(detail)
    return problems


def _gate_problems(
    report: Path, payload: dict[str, Any], gate_metrics: dict[str, Any]
) -> list[str]:
    from game_cls.reports.benchmark import check_gates

    scores = payload.get("scores") or {}
    recorded = {str(entry.get("name")) for entry in (payload.get("gates") or [])}
    expected_names = set(map(str, gate_metrics))
    problems: list[str] = []
    missing = sorted(expected_names - recorded)
    if missing:
        problems.append(f"report never measured gates {missing}")
    measurable = {
        name: spec for name, spec in gate_metrics.items() if str(name) in scores
    }
    for name, passed, detail in check_gates(scores, measurable):
        if not passed:
            problems.append(f"gate {name} fails: {detail}")
    unmeasured = sorted(expected_names - set(map(str, measurable)) - set(missing))
    if unmeasured:
        problems.append(f"report records no values for gates {unmeasured}")
    return [f"benchmark report {report}: {problem}" for problem in problems]


def verify_release_artifact(
    run_dir: str | Path,
    checkpoint: str,
    config: dict[str, Any],
) -> ReleaseVerification:
    """Verify and return the exact PASS bound to one deployment artifact."""
    from game_cls.reports.benchmark import file_sha256, validate_gate_metrics

    run_dir = Path(run_dir).resolve()
    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint)
    if checkpoint_path is None:
        raise ReleaseVerificationError(
            f"checkpoint {checkpoint!r} was not found under {run_dir / 'checkpoints'}"
        )
    gate_metrics = (config.get("benchmark") or {}).get("gate_metrics") or {}
    gate_config_problems = validate_gate_metrics(gate_metrics)
    if not gate_metrics:
        gate_config_problems.append("benchmark.gate_metrics is empty")
    if gate_config_problems:
        raise ReleaseVerificationError("; ".join(gate_config_problems))

    checkpoint_sha = file_sha256(checkpoint_path)
    try:
        expected, components = release_identity(
            run_id=run_dir.name,
            checkpoint_sha256=checkpoint_sha,
            config=config,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ReleaseVerificationError(
            f"challenge identity cannot be verified: {exc}"
        ) from exc
    if not expected["challenge_dataset_fingerprint"]:
        raise ReleaseVerificationError(
            "challenge bundle fingerprint cannot be recomputed; restore every "
            "configured challenge artifact"
        )

    from game_cls.reports.benchmark import benchmark_output_dir

    output_dir = benchmark_output_dir(config, run_dir)
    candidates = sorted(
        output_dir.rglob("report.json") if output_dir.is_dir() else (),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    checkpoint_reports: list[tuple[Path, dict[str, Any]]] = []
    for report in candidates:
        payload = _read_report(report)
        if payload and payload.get("checkpoint_sha256") == checkpoint_sha:
            checkpoint_reports.append((report, payload))

    if not checkpoint_reports:
        raise ReleaseVerificationError(
            "no benchmark report is bound to checkpoint sha256=" + checkpoint_sha
        )

    mismatch_details: list[str] = []
    for report, payload in checkpoint_reports:
        identity_problems = _identity_problems(payload, expected, components)
        if identity_problems:
            mismatch_details.extend(f"{report}: {item}" for item in identity_problems)
            continue
        gate_problems = _gate_problems(report, payload, gate_metrics)
        if gate_problems:
            mismatch_details.extend(gate_problems)
            continue
        identity_sha = release_identity_sha256(expected)
        recorded_identity_sha = payload.get("release_identity_sha256")
        if recorded_identity_sha != identity_sha:
            mismatch_details.append(
                f"{report}: release_identity_sha256 is missing or differs"
            )
            continue
        return ReleaseVerification(
            run_id=run_dir.name,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            report_path=report,
            release_identity_sha256=identity_sha,
            identity=expected,
        )

    preview = "\n  - ".join(mismatch_details[:12])
    raise ReleaseVerificationError(
        "no benchmark PASS matches the complete current release identity"
        + (f":\n  - {preview}" if preview else "")
    )
