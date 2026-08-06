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

# Bump whenever the assignment rule, the fingerprint definition or the
# manifest schema changes so stale manifests are rejected instead of reused
# under a contract they were not written for.
SPLIT_ALGORITHM_VERSION = 2

# val_ratio round-trips through parquet float64; compare with a tolerance
# rather than by identity so 0.2 never reads as "changed".
_RATIO_TOLERANCE = 1e-9

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


def _frame_identity(frame) -> tuple[str, int, str, int, int, str]:
    """Machine-independent identity of one frame.

    The absolute ``path`` is deliberately excluded: it encodes the mount
    point, so the same dataset staged under a different root would
    re-fingerprint and refuse to reuse a perfectly valid manifest. What
    remains is the dataset-relative identity plus, when the scan computed
    it, the content SHA -- so an *edited* frame still changes the
    fingerprint even though its name did not. Without a content hash the
    file size stands in as a weak content witness.
    """
    return (
        frame.game,
        int(frame.label),
        frame.video_id,
        int(frame.frame_id),
        int(frame.file_size),
        frame.content_sha256 or "",
    )


def compute_dataset_fingerprint(frames) -> str:
    """Deterministic SHA-256 over the sorted per-frame identity rows.

    Identity is ``(game, label, video_id, frame_id, file_size,
    content_sha256)`` -- see :func:`_frame_identity`. Adding, removing or
    editing a frame re-fingerprints the dataset; relocating it does not.
    """
    hasher = hashlib.sha256()
    for row in sorted(_frame_identity(frame) for frame in frames):
        hasher.update("|".join(str(part) for part in row).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def fingerprint_covers_content(frames) -> bool:
    """Whether every frame carried a content hash into the fingerprint.

    ``compute_content_hash=False`` scans fall back to the file size, which
    misses a same-size edit. Callers that need the stronger guarantee can
    surface this in the audit instead of assuming it.
    """
    return all(frame.content_sha256 for frame in frames)


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


def _validate_split_params(
    *,
    val_ratio: float,
    target_delta: int,
    small_stratum_policy: str,
) -> None:
    """Shared argument contract of the fresh-split and extend paths."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    if target_delta not in (1, 2, 3):
        raise ValueError("target_delta must be 1, 2 or 3")
    if small_stratum_policy not in ("error", "warn"):
        raise ValueError(
            f"small_stratum_policy must be 'error' or 'warn', got "
            f"{small_stratum_policy!r}"
        )


def _greedy_val_members(
    candidates: list[str],
    groups: dict[str, dict],
    *,
    target_delta: int,
    target_pairs: float,
    start_pairs: int,
    reserve_last_for_train: bool,
) -> list[str]:
    """Greedily pick candidates for ``val`` toward ``target_pairs``.

    ``candidates`` must already be in stable rank order. ``start_pairs`` is
    the val pair count already committed for this stratum (nonzero only when
    extending an existing manifest), so the greedy walk continues from where
    the previous split left off instead of restarting at zero.
    """
    val_pairs = start_pairs
    members: list[str] = []
    for index, uid in enumerate(candidates):
        pair_count = groups[uid]["pair_counts"][target_delta]
        after = val_pairs + pair_count
        closer = abs(after - target_pairs) < abs(val_pairs - target_pairs)
        # Never take the last group into val: leaves at least one source
        # video for train when the stratum has >= 2 groups.
        can_leave_train = not reserve_last_for_train or index < len(candidates) - 1
        if closer and can_leave_train:
            val_pairs = after
            members.append(uid)
    return members


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
    _validate_split_params(
        val_ratio=val_ratio,
        target_delta=target_delta,
        small_stratum_policy=small_stratum_policy,
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
        total_pairs = sum(groups[uid]["pair_counts"][target_delta] for uid in ordered)
        if total_pairs <= 0:
            # No legal target_delta pairs in this stratum: keep all in
            # train (validation cannot rely on a delta with no support).
            for uid in ordered:
                assignment[uid] = "train"
            continue
        val_members = _greedy_val_members(
            ordered,
            groups,
            target_delta=target_delta,
            target_pairs=val_ratio * total_pairs,
            start_pairs=0,
            reserve_last_for_train=True,
        )
        # Guard: a huge first group can make every step "not closer" and
        # leave val empty; force the first ordered group in so validation
        # actually has data.
        if not val_members and len(ordered) >= 2:
            val_members.append(ordered[0])
        for uid in ordered:
            assignment[uid] = "val" if uid in set(val_members) else "train"
    return assignment


def extend_split(
    frames,
    existing: dict[str, str],
    *,
    val_ratio: float,
    seed: int,
    target_delta: int = 2,
    small_stratum_policy: str = "error",
) -> tuple[dict[str, str], dict]:
    """Place only the source videos absent from ``existing``.

    Every uid already in ``existing`` keeps its split, so the validation set
    a model was selected against never reshuffles. New uids are ordered by
    the same stable ``(seed, uid)`` rank and greedily added to ``val``, but
    the greedy walk starts from the pair count the stratum *already* has in
    val, so the stratum converges on ``val_ratio`` over the union rather
    than over the new groups alone.

    Uids in the manifest that no longer appear in ``frames`` are dropped:
    they cannot be indexed, and keeping them would inflate the summary
    counts. Returns ``(assignment, stats)``.
    """
    _validate_split_params(
        val_ratio=val_ratio,
        target_delta=target_delta,
        small_stratum_policy=small_stratum_policy,
    )

    groups = _group_frames(frames)
    dropped = sorted(set(existing) - set(groups))
    assignment = {uid: existing[uid] for uid in groups if uid in existing}

    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for uid, group in groups.items():
        strata[(group["game"], group["label"])].append(uid)

    added: list[str] = []
    for (game, label), uids in sorted(strata.items()):
        fresh = sorted(
            (uid for uid in uids if uid not in assignment),
            key=lambda uid: _stable_rank(seed, uid),
        )
        if not fresh:
            continue
        added.extend(fresh)
        known = [uid for uid in uids if uid in assignment]
        if not known and len(uids) < 2:
            # A brand-new single-video stratum is the same unsplittable
            # case the fresh path rejects.
            message = (
                f"(game={game!r}, label={label}) has only {len(uids)} "
                "source video(s); cannot split without frame-level "
                "leakage. Collect more source videos or raise val_ratio."
            )
            if small_stratum_policy == "error":
                raise ValueError(message)
            assignment[fresh[0]] = "train"
            continue
        total_pairs = sum(groups[uid]["pair_counts"][target_delta] for uid in uids)
        if total_pairs <= 0:
            for uid in fresh:
                assignment[uid] = "train"
            continue
        val_pairs_held = sum(
            groups[uid]["pair_counts"][target_delta]
            for uid in known
            if assignment[uid] == "val"
        )
        # Only reserve a train slot when the stratum has no train member
        # yet; otherwise an existing train video already covers the
        # constraint and every new group may legitimately go to val.
        reserve_last = not any(assignment[uid] == "train" for uid in known)
        val_members = _greedy_val_members(
            fresh,
            groups,
            target_delta=target_delta,
            target_pairs=val_ratio * total_pairs,
            start_pairs=val_pairs_held,
            reserve_last_for_train=reserve_last,
        )
        if not known and not val_members and len(fresh) >= 2:
            val_members.append(fresh[0])
        chosen = set(val_members)
        for uid in fresh:
            assignment[uid] = "val" if uid in chosen else "train"

    stats = {
        "added_source_videos": len(added),
        "dropped_source_videos": len(dropped),
        "added_source_video_uids": sorted(added),
        "dropped_source_video_uids": dropped,
    }
    return assignment, stats


def _manifest_rows(
    frames,
    assignment: dict[str, str],
    *,
    dataset_fingerprint: str,
    seed: int,
    val_ratio: float,
    target_delta: int,
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
                # Recorded so reuse can verify the manifest was written
                # under the same balancing contract, not just the same data.
                "split_val_ratio": float(val_ratio),
                "split_target_delta": int(target_delta),
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
    val_ratio: float,
    target_delta: int,
) -> None:
    """Write the split manifest parquet (one row per source video)."""
    from .indexing import _pyarrow  # local import avoids a cycle

    pa, pq = _pyarrow()
    rows = _manifest_rows(
        frames,
        assignment,
        dataset_fingerprint=dataset_fingerprint,
        seed=seed,
        val_ratio=val_ratio,
        target_delta=target_delta,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def load_split_manifest(path: str | Path) -> dict:
    """Load ``{source_video_uid: "train" | "val"}`` and manifest metadata.

    ``split_val_ratio``/``split_target_delta`` are absent from manifests
    written by algorithm version 1 and come back as ``None``; the version
    check rejects those before the values are ever compared.
    """
    from .indexing import _pyarrow

    _, pq = _pyarrow()
    # Use pq.read_table (not pq.ParquetFile) so pyarrow opens the file via
    # ReadableFile, which on Windows sets FILE_SHARE_DELETE.  pq.ParquetFile
    # omits that flag, so any live reference blocks os.replace on the same
    # path with WinError 32 — exactly what resolve_split does when it
    # atomically rewrites a stale manifest.
    rows = pq.read_table(path).to_pylist()
    assignment = {row["source_video_uid"]: row["split"] for row in rows}
    if rows:
        first = rows[0]
        ratio = first.get("split_val_ratio")
        delta = first.get("split_target_delta")
        return {
            "assignment": assignment,
            "dataset_fingerprint": first["dataset_fingerprint"],
            "split_seed": int(first["split_seed"]),
            "split_algorithm_version": int(first["split_algorithm_version"]),
            "split_val_ratio": None if ratio is None else float(ratio),
            "split_target_delta": None if delta is None else int(delta),
        }
    return {
        "assignment": {},
        "dataset_fingerprint": "",
        "split_seed": 0,
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "split_val_ratio": None,
        "split_target_delta": None,
    }


def _manifest_mismatches(
    existing: dict,
    *,
    dataset_fingerprint: str,
    seed: int,
    val_ratio: float,
    target_delta: int,
) -> dict[str, str]:
    """Which of the five reuse fields disagree with the manifest.

    Keys are field names; values are printable ``manifest=... config=...``
    descriptions. An empty dict means the manifest may be reused verbatim.
    """
    mismatches: dict[str, str] = {}

    def note(field: str, found, wanted) -> None:
        mismatches[field] = f"{field}: manifest={found!r} config={wanted!r}"

    if existing["split_algorithm_version"] != SPLIT_ALGORITHM_VERSION:
        note(
            "split_algorithm_version",
            existing["split_algorithm_version"],
            SPLIT_ALGORITHM_VERSION,
        )
    if int(existing["split_seed"]) != int(seed):
        note("split_seed", existing["split_seed"], seed)
    found_ratio = existing.get("split_val_ratio")
    if found_ratio is None or abs(float(found_ratio) - val_ratio) > _RATIO_TOLERANCE:
        note("split_val_ratio", found_ratio, val_ratio)
    found_delta = existing.get("split_target_delta")
    if found_delta is None or int(found_delta) != int(target_delta):
        note("split_target_delta", found_delta, target_delta)
    if existing["dataset_fingerprint"] != dataset_fingerprint:
        note(
            "dataset_fingerprint",
            existing["dataset_fingerprint"],
            dataset_fingerprint,
        )
    return mismatches


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
        members = {uid for uid, target in assignment.items() if target == split}
        split_groups = [group for uid, group in groups.items() if uid in members]
        splits[split] = {
            "source_video_count": len(members),
            "frame_count": sum(g["frame_count"] for g in split_groups),
            "pair_count_delta1": sum(g["pair_counts"][1] for g in split_groups),
            "pair_count_delta2": sum(g["pair_counts"][2] for g in split_groups),
            "pair_count_delta3": sum(g["pair_counts"][3] for g in split_groups),
        }
    # Report the ratio for the delta that actually drove the balancing, not
    # a hardcoded delta=2 that would silently misreport a delta=1/3 split.
    achieved: dict[str, float] = {}
    for delta in (1, 2, 3):
        key = f"pair_count_delta{delta}"
        total = splits["train"][key] + splits["val"][key]
        achieved[f"val_ratio_achieved_delta{delta}"] = round(
            splits["val"][key] / total if total else 0.0, 6
        )
    return {
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "split_seed": seed,
        "target_val_ratio": val_ratio,
        "target_delta": target_delta,
        # Ratio on the balancing delta, under a delta-independent name.
        "val_ratio_achieved": achieved[f"val_ratio_achieved_delta{target_delta}"],
        **achieved,
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
    * All five reuse fields matching (algorithm version, seed, ``val_ratio``,
      ``target_delta``, dataset fingerprint) -> reuse the assignment verbatim.
    * Any field *other than* the fingerprint differing -> always raise. The
      manifest was written under a different balancing contract, so neither
      reusing nor extending it would mean what the config asks for.
    * Fingerprint-only change with ``on_new_groups="error"`` (default) ->
      raise, so new data can never silently re-shuffle the validation set.
    * Fingerprint-only change with ``on_new_groups="extend"`` -> keep every
      existing assignment and place only the new source videos, then rewrite
      the manifest so the next run reuses instead of re-extending.

    Returns ``(assignment, summary)`` where ``summary`` carries
    ``manifest_reused``/``manifest_extended`` markers plus the aggregate
    split stats.
    """
    if on_new_groups not in ("error", "extend"):
        raise ValueError(
            f"on_new_groups must be 'error' or 'extend', got {on_new_groups!r}"
        )
    fingerprint = compute_dataset_fingerprint(frames)

    existing: dict | None = None
    if manifest_path and Path(manifest_path).is_file():
        existing = load_split_manifest(manifest_path)

    def finish(
        assignment: dict[str, str],
        *,
        reused: bool,
        extended: bool,
        extend_stats: dict | None = None,
    ) -> tuple[dict[str, str], dict]:
        summary = split_summary(
            frames,
            assignment,
            dataset_fingerprint=fingerprint,
            seed=seed,
            val_ratio=val_ratio,
            target_delta=target_delta,
        )
        summary["manifest_reused"] = reused
        summary["manifest_extended"] = extended
        summary["fingerprint_covers_content"] = fingerprint_covers_content(frames)
        summary["added_source_videos"] = 0
        summary["dropped_source_videos"] = 0
        if extend_stats:
            summary.update(extend_stats)
        return assignment, summary

    if existing is not None:
        mismatches = _manifest_mismatches(
            existing,
            dataset_fingerprint=fingerprint,
            seed=seed,
            val_ratio=val_ratio,
            target_delta=target_delta,
        )
        if not mismatches:
            return finish(existing["assignment"], reused=True, extended=False)

        parameter_mismatches = [
            text for field, text in mismatches.items() if field != "dataset_fingerprint"
        ]
        if parameter_mismatches:
            # Not a data change: the split policy itself moved. Extending
            # would blend two balancing contracts in one validation set.
            raise ValueError(
                "split_manifest.parquet was written under different split "
                "parameters, so it cannot be reused or extended: "
                + "; ".join(parameter_mismatches)
                + ". Restore the original parameters, or delete the manifest "
                "to re-split from scratch (this changes the validation set, "
                "so previously reported metrics are no longer comparable)."
            )

        if on_new_groups == "error":
            raise ValueError(
                "dataset fingerprint changed since split_manifest.parquet "
                "was written; refusing to silently re-shuffle the "
                "validation set. Set data.split.on_new_groups=extend to keep "
                "existing assignments and place only the new source videos, "
                "or delete the manifest to re-split."
            )

        assignment, extend_stats = extend_split(
            frames,
            existing["assignment"],
            val_ratio=val_ratio,
            seed=seed,
            target_delta=target_delta,
            small_stratum_policy=small_stratum_policy,
        )
        if manifest_path:
            # Persist the extended assignment under the new fingerprint so
            # the next run reuses it instead of extending the stale base
            # again (which would re-place the same "new" groups).
            write_split_manifest(
                frames,
                assignment,
                manifest_path,
                dataset_fingerprint=fingerprint,
                seed=seed,
                val_ratio=val_ratio,
                target_delta=target_delta,
            )
        return finish(
            assignment,
            reused=False,
            extended=True,
            extend_stats=extend_stats,
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
            val_ratio=val_ratio,
            target_delta=target_delta,
        )
    # The manifest was just computed and written (or omitted entirely), so
    # nothing was reused even though the file now exists on disk.
    return finish(assignment, reused=False, extended=False)
