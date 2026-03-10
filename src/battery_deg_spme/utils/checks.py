from __future__ import annotations

import numpy as np


def assert_nonempty(name: str, obj):
    if obj is None:
        raise ValueError(f"{name} is None")
    if hasattr(obj, "__len__") and len(obj) == 0:
        raise ValueError(f"{name} is empty")


def assert_same_length(name_a: str, a, name_b: str, b):
    if len(a) != len(b):
        raise ValueError(
            f"Length mismatch: {name_a} has length {len(a)}, "
            f"but {name_b} has length {len(b)}."
        )


def assert_finite_array(name: str, arr):
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")