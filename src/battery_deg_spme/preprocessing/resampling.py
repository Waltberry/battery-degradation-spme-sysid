from __future__ import annotations

import numpy as np


def resample_uniform(t: np.ndarray, u: np.ndarray, y: np.ndarray, dt: float):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if len(t) < 2:
        raise ValueError("Need at least 2 time points to resample.")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")

    t0, t1 = float(t[0]), float(t[-1])
    tg = np.arange(t0, t1 + dt, dt, dtype=np.float64)

    u1 = np.interp(tg, t, u[:, 0]).reshape(-1, 1)
    y1 = np.interp(tg, t, y[:, 0]).reshape(-1, 1)

    return tg, u1, y1