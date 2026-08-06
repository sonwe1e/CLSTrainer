"""Test-environment configuration.

On Windows, the local dev stack (torch's OpenMP/MKL runtime, numpy's
OpenBLAS, pyarrow) has a pre-existing native-thread race: the first
parquet error-shard write after CPU training can crash with a raw access
violation (0xC0000005) in a thread with no Python frame. It is
timing-sensitive (never observed on the Linux CI, not reproducible when
the native thread pools are bounded) and is not caused by any single
test. Limiting the thread pools before torch/pyarrow initialize makes the
suite deterministic on Windows; Linux/macOS CI is unaffected.
"""

import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
