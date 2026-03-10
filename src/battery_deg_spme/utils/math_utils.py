from __future__ import annotations

import numpy as np


def simple_line_fit_rmse(t, y):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if t.size != y.size:
        raise ValueError(f"t and y must have the same length, got {t.size} and {y.size}.")
    if t.size < 2:
        raise ValueError("Need at least 2 samples for a line fit.")

    p = np.polyfit(t, y, 1)
    ylin = np.polyval(p, t)
    rmse = float(np.sqrt(np.mean((y - ylin) ** 2)))
    return rmse, p