"""Deterministic source-video-level train/validation split (step4 §二).

The split unit is the *source video*, never an individual frame: adjacent
frames and every delta pair of one source video must live in a single
split, otherwise the near-duplicate frames leak across train/val. The
splitter is fully deterministic: the same frames + seed + algorithm
version produce byte-identical manifests, and a persistent manifest is the
long-term contract so new data never silently reshuffles the existing
validation set.

Source identity is pluggable through ``identity_mode``:

* ``"game_video"`` (default) — ``source_video_uid`` is label-independent
  and used exactly as in ``indexing.source_video_uid``; a video carrying
  frames under both labels stays a single uid, so the strict audit treats
  any overlap across splits as fatal.
* ``"game_label_video"`` — the label is embedded in the uid, so each label
  of a video becomes an independent identity.

Under both modes per-label pair counts stay isolated: a ``target_delta``
pair is counted only within one label's frame ids, never across labels. No
per-video metadata sidecar exists; balancing uses only fields that are
already in the frame/video index (``game``, ``label``, frame ids).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

# Bump whenever the assignment rule, the fingerprint definition or the
# manifest schema changes so stale manifests are rejected instead of reused
# under a contract they were not written for.
SPLIT_ALGORITHM_VERSION = 3

# Source identity contract: how a (game, video_id) (plus label) is reduced
# to a single source_video_uid. Both modes keep pair counts per label.
SOURCE_IDENTITY_MODES = ("game_video", "game_label_video")
SOURCE_IDENTITY_MODE_DEFAULT = "game_video"

# val_ratio round-trips through parquet float64; compare with a tolerance
# rather than by identity so 0.2 never reads as "changed".
_RATIO_TOLERANCE = 1e-9

# File names produced next to the per-split parquet triplets.
SPLIT_MANIFEST_FILENAME = "split_manifest.parquet"
SPLIT_SUMMARY_FILENAME = "split_summary.json"


def source_video_uid(
    game: str,
    video_id: str,
    label=None,
    *,
    mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> str:
    """Stable identity of a source video under ``mode``.

    The default ``"game_video"`` identity is label-independent: a source
    video keeps a single uid regardless of its labels, exactly the leakage
    unit ``indexing.source_video_uid`` audits against. Under
    ``"game_label_video"`` the label is embedded in the uid, so a video
    that carries frames under both labels becomes two independent
    identities and ``label`` is required.
    """
    if mode not in SOURCE_IDENTITY_MODES:
        raise ValueError(
            f"unknown source identity mode {mode!r}; expected one of "
            f"{SOURCE_IDENTITY_MODES}"
        )
    if mode == "game_label_video":
        if label is None:
            raise ValueError("source_video_uid mode=game_label_video requires label")
        return f"{game}::{int(label)}::{video_id}"
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


def _group_frames(
    frames,
    *,
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> dict[str, dict]:
    """Group frames by source_video_uid and precompute per-label stats.

    Under ``"game_label_video"`` the uid embeds the label, so every group
    has exactly one label. Under the default ``"game_video"`` a source
    video that carries frames under both label 0 and label 1 lands in a
    single group whose ``labels`` holds both: per-label frame ids and pair
    counts stay isolated so pair counts never leak across labels.
    """
    grouped: dict[str, dict] = {}
    for frame in frames:
        uid = source_video_uid(
            frame.game,
            frame.video_id,
            label=frame.label,
            mode=identity_mode,
        )
        group = grouped.setdefault(
            uid,
            {
                "game": frame.game,
                "labels": set(),
                "frame_ids_by_label": defaultdict(set),
            },
        )
        label = int(frame.label)
        group["labels"].add(label)
        group["frame_ids_by_label"][label].add(int(frame.frame_id))
    for group in grouped.values():
        group["labels"] = tuple(sorted(group["labels"]))
        frame_ids_by_label = dict(group["frame_ids_by_label"])
        group["frame_ids_by_label"] = frame_ids_by_label
        pair_counts_by_label = {
            label: {
                delta: sum(frame_id + delta in ids for frame_id in ids)
                for delta in (1, 2, 3)
            }
            for label, ids in frame_ids_by_label.items()
        }
        group["pair_counts_by_label"] = pair_counts_by_label
        # Source-level pairs: the sum over labels, never computed across
        # labels (a delta pair with one frame in label 0 and the other in
        # label 1 is not a valid pair).
        group["pair_counts"] = {
            delta: sum(pair_counts_by_label[label][delta] for label in group["labels"])
            for delta in (1, 2, 3)
        }
        group["frame_count"] = sum(len(ids) for ids in frame_ids_by_label.values())
        group["frame_count_label0"] = len(frame_ids_by_label.get(0, ()))
        group["frame_count_label1"] = len(frame_ids_by_label.get(1, ()))
    return grouped


def _validate_split_params(
    *,
    val_ratio: float,
    target_delta: int,
    small_stratum_policy: str,
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
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
    if identity_mode not in SOURCE_IDENTITY_MODES:
        raise ValueError(
            f"identity_mode must be one of {SOURCE_IDENTITY_MODES}, got "
            f"{identity_mode!r}"
        )


def _build_strata(groups, target_delta) -> dict[tuple[str, int], dict]:
    """Per-``(game, label)`` strata: member uids and total target pairs.

    A multi-label uid contributes to one stratum per label. ``total_pairs``
    is the sum of that label's ``target_delta`` pair counts, so a stratum's
    cost target never mixes labels.
    """
    strata: dict[tuple[str, int], dict] = {}
    for uid, group in groups.items():
        for label in group["labels"]:
            stratum = (group["game"], label)
            entry = strata.setdefault(stratum, {"uids": [], "total_pairs": 0})
            entry["uids"].append(uid)
            entry["total_pairs"] += group["pair_counts_by_label"][label][target_delta]
    for entry in strata.values():
        entry["uids"].sort()
    return strata


def _contribution(uid, groups, target_delta) -> dict[tuple[str, int], int]:
    """Val-pair contribution of ``uid``: per stratum its label pair count."""
    group = groups[uid]
    return {
        (group["game"], label): group["pair_counts_by_label"][label][target_delta]
        for label in group["labels"]
    }


def _global_cost(val_pairs, strata, *, val_ratio) -> float:
    """How far the current val-pair counts sit from ``val_ratio`` per stratum.

    Each stratum contributes ``abs(held - val_ratio * total)`` scaled by
    its total, so a stratum with more pairs weighs more but zero-pair
    strata still resolve deterministically.
    """
    cost = 0.0
    for stratum, entry in strata.items():
        total = entry["total_pairs"]
        cost += abs(val_pairs.get(stratum, 0) - val_ratio * total) / max(total, 1)
    return cost


def _add_pairs(val_pairs: dict, contribution: dict) -> dict:
    """Return ``val_pairs`` with ``contribution`` merged in."""
    merged = dict(val_pairs)
    for stratum, count in contribution.items():
        merged[stratum] = merged.get(stratum, 0) + count
    return merged


def _sub_pairs(val_pairs: dict, contribution: dict) -> dict:
    """Return ``val_pairs`` with ``contribution`` subtracted."""
    merged = dict(val_pairs)
    for stratum, count in contribution.items():
        merged[stratum] = merged.get(stratum, 0) - count
    return merged


def _enforce_stratum_presence(
    strata: dict,
    groups: dict,
    seed: int,
    target_delta: int,
    val_members: set,
    val_pairs: dict,
    *,
    forced_train: set,
    val_ratio: float,
    immutable: set,
) -> tuple[set, dict]:
    """Guarantee every eligible stratum keeps at least one train and one val.

    A stratum with at least two eligible uids (eligible = not forced into
    train) must not end up entirely in one split. Only non-immutable uids
    move; deterministic picks use the stable ``(seed, uid)`` rank.

    Mixed-label uids (present in several strata) make a naive local fix
    oscillate: ejecting a shared uid to give one stratum train presence can
    empty another stratum's val, whose local fix then pulls the same uid
    back -- the bounded loop exits mid-cycle with a stratum still in one
    split (step7 regression). Two guards prevent that:

    * *Safe moves* -- a pull/eject that would strip the last train (resp.
      last val) member from any *other* stratum is deferred in favour of a
      candidate that does not.
    * *A move ledger* -- a uid that has already moved is only used again
      when no other candidate exists, so a repair cannot bounce one uid
      between two strata pass after pass.

    The loop is bounded at ``2 * len(strata) + len(groups) + 1`` passes
    (each move consumes at least one fresh uid, so the ledger bounds the
    total) and stops early when a full pass changes nothing. Returns
    ``(val_members, val_pairs)``.
    """
    val_members = set(val_members)
    val_pairs = dict(val_pairs)
    moved: set[str] = set()

    def other_strata(uid: str, skip: tuple[str, int]):
        for stratum, entry in strata.items():
            if stratum != skip and uid in entry["uids"]:
                yield stratum, entry

    def eject_safe(uid: str, skip: tuple[str, int]) -> bool:
        # Removing uid from val must not leave another stratum val-less.
        for _, entry in other_strata(uid, skip):
            if not any(
                other != uid and other in val_members
                for other in entry["uids"]
                if other not in forced_train
            ):
                return False
        return True

    def pull_safe(uid: str, skip: tuple[str, int]) -> bool:
        # Adding uid to val must not strip another stratum of its last train.
        for _, entry in other_strata(uid, skip):
            if not any(
                other != uid and other not in val_members
                for other in entry["uids"]
                if other not in forced_train
            ):
                return False
        return True

    for _ in range(2 * len(strata) + len(groups) + 1):
        changed = False
        for stratum in sorted(strata):
            eligible = [
                uid for uid in strata[stratum]["uids"] if uid not in forced_train
            ]
            if len(eligible) < 2:
                continue
            present = [uid for uid in eligible if uid in val_members]
            if not present:
                # Val presence: pull a train uid into val, preferring an
                # unmoved uid whose pull does not starve another stratum.
                candidates = [
                    uid
                    for uid in eligible
                    if uid not in val_members and uid not in immutable
                ]
                candidates.sort(key=lambda uid: _stable_rank(seed, uid))
                pick = _presence_pick(candidates, stratum, pull_safe, moved)
                if pick is None:
                    continue
                val_members.add(pick)
                val_pairs = _add_pairs(
                    val_pairs, _contribution(pick, groups, target_delta)
                )
                moved.add(pick)
                changed = True
            elif len(present) == len(eligible):
                # Train presence: all eligible uids are in val; eject the
                # val uid whose removal minimizes the global cost (tie-break:
                # min stable rank) without starving another stratum of val.
                candidates = [
                    uid
                    for uid in eligible
                    if uid in val_members and uid not in immutable
                ]
                candidates.sort(
                    key=lambda uid: (
                        _global_cost(
                            _sub_pairs(
                                val_pairs,
                                _contribution(uid, groups, target_delta),
                            ),
                            strata,
                            val_ratio=val_ratio,
                        ),
                        _stable_rank(seed, uid),
                    )
                )
                pick = _presence_pick(candidates, stratum, eject_safe, moved)
                if pick is None:
                    continue
                val_members.remove(pick)
                val_pairs = _sub_pairs(
                    val_pairs, _contribution(pick, groups, target_delta)
                )
                moved.add(pick)
                changed = True
        if not changed:
            break
    return val_members, val_pairs


def _presence_pick(
    candidates: list[str],
    stratum: tuple[str, int],
    safe,
    moved: set[str],
) -> str | None:
    """Deterministic move candidate for :func:`_enforce_stratum_presence`.

    Order of preference: an unmoved uid whose move is safe for other strata,
    then a safe uid (even if already moved), then an unmoved uid, then the
    first candidate. The moved ledger prevents a shared mixed-label uid from
    being pulled and ejected across passes, which is what made the old loop
    oscillate.
    """
    for uid in candidates:
        if uid not in moved and safe(uid, stratum):
            return uid
    for uid in candidates:
        if safe(uid, stratum):
            return uid
    for uid in candidates:
        if uid not in moved:
            return uid
    return candidates[0] if candidates else None


def split_source_videos(
    frames,
    *,
    val_ratio: float,
    seed: int,
    target_delta: int = 2,
    small_stratum_policy: str = "error",
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> dict[str, str]:
    """Assign each source video to ``train`` or ``val``.

    The unit of assignment is the source video under ``identity_mode``:
    ``"game_video"`` (default) keeps a mixed-label video as one indivisible
    uid, ``"game_label_video"`` treats each label of a video as a separate
    uid. Uids are walked in stable ``(seed, uid)`` rank order and greedily
    moved into ``val`` while that strictly lowers the global stratum cost
    toward ``val_ratio`` at ``target_delta``; every ``(game, label)``
    stratum with at least two eligible videos is then forced to hold at
    least one train and one val video. ``small_stratum_policy="error"``
    rejects strata with fewer than two source videos (cannot split without
    leakage); ``"warn"`` puts those lone videos in train.

    Returns ``{source_video_uid: "train" | "val"}``.
    """
    _validate_split_params(
        val_ratio=val_ratio,
        target_delta=target_delta,
        small_stratum_policy=small_stratum_policy,
        identity_mode=identity_mode,
    )

    groups = _group_frames(frames, identity_mode=identity_mode)
    strata = _build_strata(groups, target_delta)

    forced_train: set[str] = set()
    for (game, label), entry in sorted(strata.items()):
        if len(entry["uids"]) < 2:
            message = (
                f"(game={game!r}, label={label}) has only {len(entry['uids'])} "
                "source video(s); cannot split without frame-level "
                "leakage. Collect more source videos or raise val_ratio."
            )
            if small_stratum_policy == "error":
                raise ValueError(message)
            forced_train.update(entry["uids"])

    val_members: set[str] = set()
    val_pairs: dict[tuple[str, int], int] = {}
    for uid in sorted(groups, key=lambda u: _stable_rank(seed, u)):
        if uid in forced_train:
            continue
        contribution = _contribution(uid, groups, target_delta)
        if all(count <= 0 for count in contribution.values()):
            # No legal target_delta pairs anywhere: adding this video to val
            # cannot move any stratum toward the target, so it stays train.
            continue
        after = _add_pairs(val_pairs, contribution)
        if _global_cost(after, strata, val_ratio=val_ratio) < _global_cost(
            val_pairs, strata, val_ratio=val_ratio
        ):
            val_members.add(uid)
            val_pairs = after

    val_members, val_pairs = _enforce_stratum_presence(
        strata,
        groups,
        seed,
        target_delta,
        val_members,
        val_pairs,
        forced_train=forced_train,
        val_ratio=val_ratio,
        immutable=set(),
    )
    return {uid: "val" if uid in val_members else "train" for uid in groups}


def extend_split(
    frames,
    existing: dict[str, str],
    *,
    val_ratio: float,
    seed: int,
    target_delta: int = 2,
    small_stratum_policy: str = "error",
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> tuple[dict[str, str], dict]:
    """Place only the source videos absent from ``existing``.

    Every uid already in ``existing`` keeps its split, so the validation
    set a model was selected against never reshuffles. Fresh uids are
    ordered by the same stable ``(seed, uid)`` rank and greedily added to
    ``val``, but the greedy walk starts from the pairs val *already* holds,
    so each stratum converges on ``val_ratio`` over the union rather than
    over the new groups alone. Only brand-new single-video strata trigger
    ``small_stratum_policy``; existing strata are untouched.

    Uids in the manifest that no longer appear in ``frames`` are dropped:
    they cannot be indexed, and keeping them would inflate the summary
    counts. Returns ``(assignment, stats)``.
    """
    _validate_split_params(
        val_ratio=val_ratio,
        target_delta=target_delta,
        small_stratum_policy=small_stratum_policy,
        identity_mode=identity_mode,
    )

    groups = _group_frames(frames, identity_mode=identity_mode)
    dropped = sorted(set(existing) - set(groups))
    assignment = {uid: existing[uid] for uid in groups if uid in existing}
    immutable = set(existing)

    strata = _build_strata(groups, target_delta)
    fresh = [uid for uid in groups if uid not in existing]

    forced_train: set[str] = set()
    for (game, label), entry in sorted(strata.items()):
        known = [uid for uid in entry["uids"] if uid in assignment]
        if not known and len(entry["uids"]) < 2:
            message = (
                f"(game={game!r}, label={label}) has only {len(entry['uids'])} "
                "source video(s); cannot split without frame-level "
                "leakage. Collect more source videos or raise val_ratio."
            )
            if small_stratum_policy == "error":
                raise ValueError(message)
            forced_train.update(entry["uids"])

    # Seed the walk from the pairs val already holds in each stratum.
    val_members: set[str] = set()
    val_pairs: dict[tuple[str, int], int] = {}
    for uid, target in assignment.items():
        if target == "val":
            val_members.add(uid)
            val_pairs = _add_pairs(val_pairs, _contribution(uid, groups, target_delta))

    for uid in sorted(fresh, key=lambda u: _stable_rank(seed, u)):
        if uid in forced_train:
            assignment[uid] = "train"
            continue
        contribution = _contribution(uid, groups, target_delta)
        if all(count <= 0 for count in contribution.values()):
            assignment[uid] = "train"
            continue
        after = _add_pairs(val_pairs, contribution)
        if _global_cost(after, strata, val_ratio=val_ratio) < _global_cost(
            val_pairs, strata, val_ratio=val_ratio
        ):
            val_members.add(uid)
            val_pairs = after
            assignment[uid] = "val"
        else:
            assignment[uid] = "train"

    val_members, val_pairs = _enforce_stratum_presence(
        strata,
        groups,
        seed,
        target_delta,
        val_members,
        val_pairs,
        forced_train=forced_train,
        val_ratio=val_ratio,
        immutable=immutable,
    )
    # Known uids were fixed at the start; apply the final membership only
    # to the fresh uids.
    for uid in fresh:
        assignment[uid] = "val" if uid in val_members else "train"

    stats = {
        "added_source_videos": len(fresh),
        "dropped_source_videos": len(dropped),
        "added_source_video_uids": sorted(fresh),
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
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> list[dict]:
    """Per-source-video rows for the split manifest parquet (v3 schema).

    Per-label stats are recorded per label so a mixed-label video's
    manifest row stays usable under either identity mode.
    """
    groups = _group_frames(frames, identity_mode=identity_mode)
    rows: list[dict] = []
    for uid, group in sorted(groups.items()):
        label0_pairs = group["pair_counts_by_label"].get(0, {})
        label1_pairs = group["pair_counts_by_label"].get(1, {})
        rows.append(
            {
                "source_video_uid": uid,
                "game": group["game"],
                "labels": list(group["labels"]),
                "frame_count_label0": group["frame_count_label0"],
                "frame_count_label1": group["frame_count_label1"],
                "valid_pair_count_label0_delta1": label0_pairs.get(1, 0),
                "valid_pair_count_label0_delta2": label0_pairs.get(2, 0),
                "valid_pair_count_label0_delta3": label0_pairs.get(3, 0),
                "valid_pair_count_label1_delta1": label1_pairs.get(1, 0),
                "valid_pair_count_label1_delta2": label1_pairs.get(2, 0),
                "valid_pair_count_label1_delta3": label1_pairs.get(3, 0),
                "split": assignment[uid],
                "dataset_fingerprint": dataset_fingerprint,
                "split_seed": seed,
                "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
                # Recorded so reuse can verify the manifest was written
                # under the same balancing contract, not just the same data.
                "split_val_ratio": float(val_ratio),
                "split_target_delta": int(target_delta),
                "split_source_identity_mode": identity_mode,
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
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
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
        identity_mode=identity_mode,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and atomically swap it in so an
    # interrupted rewrite (crash/power loss) can never leave the manifest
    # half-written. os.replace is atomic on the same filesystem; a live
    # reference from load_split_manifest does not block it because
    # pq.read_table opens with FILE_SHARE_DELETE on Windows.
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(pa.Table.from_pylist(rows), tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_split_manifest(path: str | Path) -> dict:
    """Load ``{source_video_uid: "train" | "val"}`` and manifest metadata.

    ``split_val_ratio``/``split_target_delta`` are absent from manifests
    written by algorithm version 1 and come back as ``None``; the version
    check rejects those before the values are ever compared. Manifests
    older than the identity-mode record default to ``"game_video"``.
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
            "split_source_identity_mode": str(
                first.get("split_source_identity_mode", "game_video")
            ),
        }
    return {
        "assignment": {},
        "dataset_fingerprint": "",
        "split_seed": 0,
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
        "split_val_ratio": None,
        "split_target_delta": None,
        "split_source_identity_mode": "game_video",
    }


def _manifest_mismatches(
    existing: dict,
    *,
    dataset_fingerprint: str,
    seed: int,
    val_ratio: float,
    target_delta: int,
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> dict[str, str]:
    """Which of the six reuse fields disagree with the manifest.

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
    found_mode = existing.get("split_source_identity_mode", "game_video")
    if found_mode != identity_mode:
        note("split_source_identity_mode", found_mode, identity_mode)
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
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> dict:
    """Per-split aggregate summary used by ``split_summary.json``."""
    groups = _group_frames(frames, identity_mode=identity_mode)
    splits: dict[str, dict] = {}
    for split in ("train", "val"):
        members = {uid for uid, target in assignment.items() if target == split}
        split_groups = [group for uid, group in groups.items() if uid in members]
        splits[split] = {
            "source_video_count": len(members),
            "frame_count": sum(g["frame_count"] for g in split_groups),
            "frame_count_label0": sum(g["frame_count_label0"] for g in split_groups),
            "frame_count_label1": sum(g["frame_count_label1"] for g in split_groups),
            "pair_count_delta1": sum(g["pair_counts"][1] for g in split_groups),
            "pair_count_delta2": sum(g["pair_counts"][2] for g in split_groups),
            "pair_count_delta3": sum(g["pair_counts"][3] for g in split_groups),
            "pair_count_label0_delta1": sum(
                g["pair_counts_by_label"].get(0, {}).get(1, 0) for g in split_groups
            ),
            "pair_count_label0_delta2": sum(
                g["pair_counts_by_label"].get(0, {}).get(2, 0) for g in split_groups
            ),
            "pair_count_label0_delta3": sum(
                g["pair_counts_by_label"].get(0, {}).get(3, 0) for g in split_groups
            ),
            "pair_count_label1_delta1": sum(
                g["pair_counts_by_label"].get(1, {}).get(1, 0) for g in split_groups
            ),
            "pair_count_label1_delta2": sum(
                g["pair_counts_by_label"].get(1, {}).get(2, 0) for g in split_groups
            ),
            "pair_count_label1_delta3": sum(
                g["pair_counts_by_label"].get(1, {}).get(3, 0) for g in split_groups
            ),
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
        "split_source_identity_mode": identity_mode,
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


def source_identity_precheck(
    frames,
    *,
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> dict:
    """Analyze how source videos map to labels under ``identity_mode``.

    Reports how many uids are single-label vs mixed-label, the per-label
    delta pair support, and whether assignment stays atomic per source
    video. Always ``supported: True``: the mixed-label case is handled by
    keeping per-label pair counts isolated and never summing across labels.
    """
    groups = _group_frames(frames, identity_mode=identity_mode)
    mixed = sorted(uid for uid, group in groups.items() if len(group["labels"]) > 1)
    pair_counts_label0 = {delta: 0 for delta in (1, 2, 3)}
    pair_counts_label1 = {delta: 0 for delta in (1, 2, 3)}
    for group in groups.values():
        for delta in (1, 2, 3):
            pair_counts_label0[delta] += (
                group["pair_counts_by_label"].get(0, {}).get(delta, 0)
            )
            pair_counts_label1[delta] += (
                group["pair_counts_by_label"].get(1, {}).get(delta, 0)
            )
    return {
        "identity_mode": identity_mode,
        "source_video_count": len(groups),
        "single_label_videos": len(groups) - len(mixed),
        "mixed_label_videos": len(mixed),
        "mixed_label_examples": [
            {"source_video_uid": uid, "labels": list(groups[uid]["labels"])}
            for uid in mixed
        ],
        "pair_counts_label0": pair_counts_label0,
        "pair_counts_label1": pair_counts_label1,
        "supported": True,
        "atomic_split_enforced": True,
    }


def format_source_identity_precheck(report: dict) -> str:
    """Render a :func:`source_identity_precheck` report as plain text."""
    lines = ["Source identity analysis"]
    lines.append(f"Source videos:           {report['source_video_count']}")
    lines.append(f"Single-label videos:     {report['single_label_videos']}")
    lines.append(f"Mixed-label videos:      {report['mixed_label_videos']}")
    if report["mixed_label_examples"]:
        lines.append("Mixed-label examples:")
        for example in report["mixed_label_examples"]:
            lines.append(
                f"  {example['source_video_uid']}  labels={example['labels']}"
            )
    label0 = report["pair_counts_label0"]
    label1 = report["pair_counts_label1"]
    lines.append(
        f"label0 pairs: delta1={label0[1]} delta2={label0[2]} delta3={label0[3]}"
    )
    lines.append(
        f"label1 pairs: delta1={label1[1]} delta2={label1[2]} delta3={label1[3]}"
    )
    supported = "yes" if report["supported"] else "no"
    enforced = "yes" if report["atomic_split_enforced"] else "no"
    lines.append(f"Supported: {supported}")
    lines.append(f"Atomic split enforced: {enforced}")
    return "\n".join(lines)


def resolve_split(
    frames,
    *,
    val_ratio: float,
    seed: int,
    target_delta: int = 2,
    manifest_path: str | Path | None = None,
    on_new_groups: str = "error",
    small_stratum_policy: str = "error",
    identity_mode: str = SOURCE_IDENTITY_MODE_DEFAULT,
) -> tuple[dict[str, str], dict]:
    """High-level entry: reuse a matching manifest or compute a fresh split.

    * No manifest -> compute and (when ``manifest_path`` is given) persist.
    * All six reuse fields matching (algorithm version, seed, ``val_ratio``,
      ``target_delta``, source identity mode, dataset fingerprint) -> reuse
      the assignment verbatim.
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
            identity_mode=identity_mode,
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
            identity_mode=identity_mode,
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
                + ". Restore the original parameters, or Delete/rebuild the "
                "manifest to re-split from scratch (this changes the "
                "validation set, so previously reported metrics are no "
                "longer comparable)."
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
            identity_mode=identity_mode,
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
                identity_mode=identity_mode,
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
        identity_mode=identity_mode,
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
            identity_mode=identity_mode,
        )
    # The manifest was just computed and written (or omitted entirely), so
    # nothing was reused even though the file now exists on disk.
    return finish(assignment, reused=False, extended=False)
