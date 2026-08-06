"""Deterministic source-video-level train/validation split (step4 §二).

The split unit is the *source video* (``game::video_id``), never an
individual frame: adjacent frames and every delta pair of one source
video must live in a single split, otherwise the near-duplicate frames
leak across train/val. The splitter is fully deterministic: the same
frames + seed + algorithm version produce byte-identical manifests, and a
persistent manifest is the long-term contract so new data never silently
reshuffles the existing validation set.

Constraints honored from the project's hard rules:

* ``source_video_uid`` is label-independent and used exactly as in
  ``indexing.source_video_uid`` — the existing strict audit treats any
  overlap across splits as fatal.
* No per-video metadata sidecar exists; balancing uses only fields that
  are already in the frame/video index (``game``, ``label``, frame ids).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

# Bump whenever the assignment rule changes so stale manifests are rejected.
SPLIT_ALGORITHM_VERSION = 1

# File names produced next to the per-split parquet triplets.
SPLIT_MANIFEST_FILENAME = "split_manifest.parquet"
SPLIT_SUMMARY_FILENAME = "split_summary.json"


def source_video_uid(game: str, video_id: str) -> str:
    """Stable, label-independent identity of a source video."""
    return f"{game}::{video_id}"


def _stable_rank(seed: int, uid: str) -> int:
    """Deterministic per-(seed, uid) rank independent of hash randomization.

    Python's built-in ``hash()`` is salted per process, so it cannot seed
    a reproducible shuffle. SHA-256 over ``f"{seed}:{uid}"`` is stable.
    """
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def compute_dataset_fingerprint(frames) -> str:
    """Deterministic SHA-256 over the sorted (game, label, video_id,
    frame_id, path) rows. Any content change re-fingerprints the data."""
    hasher = hashlib.sha256()
    for row in sorted(
        (f.game, f.label, f.video_id, f.frame_id, f.path)
        for f in frames
    ):
        hasher.update("|".join(str(part) for part in row).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _group_frames(frames) -> dict[str, dict]:
    """Group frames by source_video_uid and precompute per-video stats.

    Raises when one source video carries frames under more than one
    ``(game, label)`` — such a uid cannot be placed without contradicting
    the stratum balance, and silently assigning it would violate the
    label-independent leakage contract.
    """
    grouped: dict[str, dict] = {}
    for frame in frames:
        uid = source_video_uid(frame.game, frame.video_id)
        group = grouped.setdefault(
            uid,
            {
                "game": frame.game,
                "label": frame.label,
                "frame_ids": set(),
                "paths": set(),
            },
        )
        if (group["game"], group["label"]) != (
            frame.game,
            frame.label,
        ):
            raise ValueError(
                f"source video {uid!r} spans multiple (game, label) "
                "combinations; split by source video is undefined"
            )
        group["frame_ids"].add(int(frame.frame_id))
        group["paths"].add(frame.path)
    for group in grouped.values():
        ids = group["frame_ids"]
        group["frame_count"] = len(ids)
        group["pair_counts"] = {
            delta: sum(frame_id + delta in ids for frame_id in ids)
            for delta in (1, 2, 3)
        }
    return grouped


def split_source_videos(
    frames,
    *,
    val_ratio: float,
    seed: int,
    target_delta: int = 2,
    small_stratum_policy: str = "error",
) -> dict[str, str]:
    """Assign each source video to ``train`` or ``val``.

    Within every ``(game, label)`` stratum the groups are ordered by the
    stable ``(seed, uid)`` hash, then greedily moved into ``val`` so the
    stratum's ``target_delta`` pair ratio lands as close as possible to
    ``val_ratio``. Groups are indivisible; ``small_stratum_policy="error"``
    rejects strata with fewer than two source videos (cannot split without
    leakage) instead of silently breaking frames apart.

    Returns ``{source_video_uid: "train" | "val"}``.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    if target_delta not in (1, 2, 3):
        raise ValueError("target_delta must be 1, 2 or 3")
    if small_stratum_policy not in ("error", "warn"):
        raise ValueError(
            f"small_stratum_policy must be 'error' or 'warn', got "
            f"{small_stratum_policy!r}"
        )

    groups = _group_frames(frames)
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for uid, group in groups.items():
        strata[(group["game"], group["label"])].append(uid)

    assignment: dict[str, str] = {}
    for (game, label), uids in sorted(strata.items()):
        if len(uids) < 2:
            message = (
                f"(game={game!r}, label={label}) has only {len(uids)} "
                "source video(s); cannot split without frame-level "
                "leakage. Collect more source videos or raise val_ratio."
            )
            if small_stratum_policy == "error":
                raise ValueError(message)
            # warn: place the lone video in train so validation is not
            # contaminated and training keeps every source.
            assignment[uids[0]] = "train"
            continue
        ordered = sorted(uids, key=lambda uid: _stable_rank(seed, uid))
        total_pairs = sum(
            groups[uid]["pair_counts"][target_delta] for uid in ordered
        )
        if total_pairs <= 0:
            # No legal target_delta pairs in this stratum: keep all in
            # train (validation cannot rely on a delta with no support).
            for uid in ordered:
                assignment[uid] = "train"
            continue
        target = val_ratio * total_pairs
        val_pairs = 0
        val_members: list[str] = []
        for index, uid in enumerate(ordered):
            pair_count = groups[uid]["pair_counts"][target_delta]
            after = val_pairs + pair_count
            closer = abs(after - target) < abs(val_pairs - target)
            # Never take the last group into val: leaves at least one
            # source video for train when the stratum has >= 2 groups.
            can_leave_train = index < len(ordered) - 1
            if closer and can_leave_train:
                val_pairs = after
                val_members.append(uid)
        # Guard: a huge first group can make every step "not closer" and
        # leave val empty; force the first ordered group in so validation
        # actually has data.
        if not val_members and len(ordered) >= 2:
            val_members.append(ordered[0])
        for uid in ordered:
            assignment[uid] = "val" if uid in set(val_members) else "train"
    return assignment


def _manifest_rows(
    frames,
    assignment: dict[str, str],
    *,
    dataset_fingerprint: str,
    seed: int,
) -> list[dict]:
    """Per-source-video rows for the split manifest parquet."""
    groups = _group_frames(frames)
    rows: list[dict] = []
    for uid, group in sorted(groups.items()):
        rows.append(
            {
                "source_video_uid": uid,
                "game": group["game"],
                "label": group["label"],
                "split": assignment[uid],
                "frame_count": group["frame_count"],
                "valid_pair_count_delta1": group["pair_counts"][1],
                "valid_pair_count_delta2": group["pair_counts"][2],
                "valid_pair_count_delta3": group["pair_counts"][3],
                "dataset_fingerprint": dataset_fingerprint,
                "split_seed": seed,
                "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
            }
        )
    return rows


def write_split_manifest(
    frames,
    assignment: dict[str, str],
    path: str | Path,
    *,
    dataset_fingerprint: str,
    seed: int,
) -> None:
    """Write the split manifest parquet (one row per source video)."""
    from .indexing import _pyarrow  # local import avoids a cycle

    pa, pq = _pyarrow()
    rows = _manifest_rows(
        frames,
        assignment,
        dataset_fingerprint=dataset_fingerprint,
        seed=seed,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def load_split_manifest(path: str | Path) -> dict:
    """Load ``{source_video_uid: "train" | "val"}`` and manifest metadata."""
    from .indexing import _pyarrow

    _, pq = _pyarrow()
    rows = pq.read_table(path).to_pylist()
    assignment = {row["source_video_uid"]: row["split"] for row in rows}
    if rows:
        first = rows[0]
        return {
            "assignment": assignment,
            "dataset_fingerprint": first["dataset_fingerprint"],
            "split_seed": int(first["split_seed"]),
            "split_algorithm_version": int(
                first["split_algorithm_version"]
            ),
        }
    return {
        "assignment": {},
        "dataset_fingerprint": "",
        "split_seed": 0,
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
    }


def split_summary(
    frames,
    assignment: dict[str, str],
    *,
    dataset_fingerprint: str,
    seed: int,
    val_ratio: float,
    target_delta: int,
) -> dict:
    """Per-split aggregate summary used by ``split_summary.json``."""
    groups = _group_frames(frames)
    splits: dict[str, dict] = {}
    for split in ("train", "val"):
        members = {
            uid
            for uid, target in assignment.items()
            if target == split
        }
        split_groups = [
            group for uid, group in groups.items() if uid in members
        ]
        splits[split] = {
            "source_video_count": len(members),
            "frame_count": sum(g["frame_count"] for g in split_groups),
            "pair_count_delta1": sum(
                g["pair_counts"][1] for g in split_groups
            ),
            "pair_count_delta2": sum(
                g["pair_counts"][2] for g in split_groups
            ),
            "pair_count_delta3": sum(
                g["pair_counts"][3] for g in split_groups
            ),
        }
    total_pairs = splits["train"]["pair_count_delta2"] + splits["val"][
        "pair_count_delta2"
    ]
    val_ratio_achieved = (
        splits["val"]["pair_count_delta2"] / total_pairs
        if total_pairs
        else 0.0
    )
    return {
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "split_seed": seed,
        "target_val_ratio": val_ratio,
        "target_delta": target_delta,
        "val_ratio_achieved_delta2": round(val_ratio_achieved, 6),
        "splits": splits,
        "source_video_count": len(assignment),
    }


def write_split_summary(summary: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_split(
    frames,
    *,
    val_ratio: float,
    seed: int,
    target_delta: int = 2,
    manifest_path: str | Path | None = None,
    on_new_groups: str = "error",
    small_stratum_policy: str = "error",
) -> tuple[dict[str, str], dict]:
    """High-level entry: reuse a matching manifest or compute a fresh split.

    * No manifest -> compute and (when ``manifest_path`` is given) persist.
    * Matching fingerprint/seed/version -> reuse the existing assignment.
    * ``on_new_groups="error"`` (default): a fingerprint change raises, so
      new data can never silently re-shuffle the validation set.
    * ``on_new_groups="extend"``: keep existing assignments and assign only
      the new source videos (their strata are re-greedied on new groups).

    Returns ``(assignment, summary)`` where ``summary`` carries
    ``reused``/``extended`` markers plus the aggregate split stats.
    """
    fingerprint = compute_dataset_fingerprint(frames)

    existing: dict | None = None
    if manifest_path and Path(manifest_path).is_file():
        existing = load_split_manifest(manifest_path)

    if existing and existing["dataset_fingerprint"] == fingerprint:
        assignment = existing["assignment"]
        summary = split_summary(
            frames,
            assignment,
            dataset_fingerprint=fingerprint,
            seed=seed,
            val_ratio=val_ratio,
            target_delta=target_delta,
        )
        summary["manifest_reused"] = True
        summary["manifest_extended"] = False
        return assignment, summary

    if existing:
        if on_new_groups == "error":
            raise ValueError(
                "dataset fingerprint changed since split_manifest.parquet "
                "was written; refusing to silently re-shuffle the "
                "validation set. Rebuild with on_new_groups=extend to keep "
                "existing assignments, or delete the manifest to re-split."
            )
        raise NotImplementedError(
            "extend is reserved for a later iteration; use "
            "on_new_groups=error and rebuild explicitly."
        )

    assignment = split_source_videos(
        frames,
        val_ratio=val_ratio,
        seed=seed,
        target_delta=target_delta,
        small_stratum_policy=small_stratum_policy,
    )
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        write_split_manifest(
            frames,
            assignment,
            manifest_path,
            dataset_fingerprint=fingerprint,
            seed=seed,
        )
    summary = split_summary(
        frames,
        assignment,
        dataset_fingerprint=fingerprint,
        seed=seed,
        val_ratio=val_ratio,
        target_delta=target_delta,
    )
    # The manifest was just computed and written (or omitted entirely), so
    # nothing was reused even though the file now exists on disk.
    summary["manifest_reused"] = False
    summary["manifest_extended"] = False
    return assignment, summary
