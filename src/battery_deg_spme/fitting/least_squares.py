from __future__ import annotations

import numpy as np


def solve_ls_normal_eq(X: np.ndarray, y: np.ndarray, ridge: float = 0.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    XtX = X.T @ X
    Xty = X.T @ y

    if ridge > 0.0:
        XtX = XtX + ridge * np.eye(X.shape[1], dtype=np.float64)

    beta = np.linalg.solve(XtX, Xty)
    return beta


def solve_ls_qr(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    Q, R = np.linalg.qr(X, mode="reduced")
    beta = np.linalg.solve(R, Q.T @ y)
    return beta


def compare_ls_solvers(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float = 1e-12,
    label: str = "LS compare",
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    beta_ne = solve_ls_normal_eq(X, y, ridge=ridge)
    beta_qr = solve_ls_qr(X, y)
    beta_np, *_ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "label": label,
        "beta_normal_eq": beta_ne,
        "beta_qr": beta_qr,
        "beta_lstsq": beta_np,
        "diff_ne_np": float(np.linalg.norm(beta_ne - beta_np)),
        "diff_qr_np": float(np.linalg.norm(beta_qr - beta_np)),
        "diff_ne_qr": float(np.linalg.norm(beta_ne - beta_qr)),
    }