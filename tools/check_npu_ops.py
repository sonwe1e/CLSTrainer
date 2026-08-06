"""Standalone NPU device-operator probe (step4 P0-4).

Runs the same environment + operator checks that ``cls-trainer doctor``
performs for an ``npu`` config, as a self-contained script for quick
manual verification on an Ascend host:

    python tools/check_npu_ops.py

Exits non-zero if any required check fails (``[FAIL]``); a WARN (skipped)
result is reported but does not fail the run.
"""

from __future__ import annotations

import sys


def main() -> int:
    failures = 0
    try:
        from game_cls.runtime.npu_checks import probe_npu_environment
    except ImportError as exc:  # pragma: no cover - import path error
        print(f"[FAIL] game_cls not importable: {exc}")
        return 1

    for label, (ok, detail) in probe_npu_environment().items():
        mark = "[OK]" if ok else ("[WARN]" if ok is None else "[FAIL]")
        if ok is False:
            failures += 1
        suffix = f" — {detail}" if detail else ""
        print(f"{mark} {label}{suffix}")

    print("")
    print("FAIL" if failures else "PASS", f"({failures} failing checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
