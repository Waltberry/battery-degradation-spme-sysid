from __future__ import annotations


def clamp_index(idx: int, n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return max(0, min(int(idx), n - 1))