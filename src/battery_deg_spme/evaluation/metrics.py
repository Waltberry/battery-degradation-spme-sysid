# from __future__ import annotations

# import numpy as np


# def vec_reshape(y):
#     y = np.asarray(y, dtype=np.float64)
#     if y.ndim == 1:
#         y = y.reshape(-1, 1)
#     return y


# def compute_scores(Y, Yhat, fit: str = "R2"):
#     Y = vec_reshape(Y)
#     Yhat = vec_reshape(Yhat)

#     y = Y[:, 0]
#     yh = Yhat[:, 0]
#     fit = fit.lower()

#     if fit == "r2":
#         denom = np.sum((y - np.mean(y)) ** 2) + 1e-12
#         return 100.0 * (1.0 - np.sum((yh - y) ** 2) / denom)

#     if fit == "rmse":
#         return float(np.sqrt(np.mean((yh - y) ** 2)))

#     if fit == "bfr":
#         denom = np.sum((y - np.mean(y)) ** 2) + 1e-12
#         return 100.0 * (1.0 - np.linalg.norm(yh - y) / np.sqrt(denom))

#     raise ValueError("fit must be one of: R2 | BFR | RMSE")


# def summarize_err(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
#     y_true = vec_reshape(y_true)
#     y_pred = vec_reshape(y_pred)
#     err = y_pred[:, 0] - y_true[:, 0]

#     return {
#         "name": name,
#         "rmse": float(np.sqrt(np.mean(err ** 2))),
#         "mae": float(np.mean(np.abs(err))),
#         "p95": float(np.percentile(np.abs(err), 95)),
#         "p99": float(np.percentile(np.abs(err), 99)),
#         "max_abs": float(np.max(np.abs(err))),
#         "r2": float(compute_scores(y_true, y_pred, fit="R2")),
#         "bfr": float(compute_scores(y_true, y_pred, fit="BFR")),
#     }


# def report_fit(name, Y_true, Y_pred) -> dict:
#     Y_true = vec_reshape(Y_true)
#     Y_pred = vec_reshape(Y_pred)
#     err = Y_pred - Y_true

#     return {
#         "name": name,
#         "max_abs": float(np.max(np.abs(err))),
#         "rmse": float(np.sqrt(np.mean(err ** 2))),
#         "r2": float(compute_scores(Y_true, Y_pred, fit="R2")),
#         "bfr": float(compute_scores(Y_true, Y_pred, fit="BFR")),
#     }


# def signal_span(x) -> dict:
#     x = np.asarray(x, dtype=np.float64).reshape(-1)
#     return {
#         "min": float(np.min(x)),
#         "max": float(np.max(x)),
#         "span": float(np.max(x) - np.min(x)),
#         "std": float(np.std(x)),
#     }


# def regression_metrics(y_true, y_pred) -> dict:
#     y_true = vec_reshape(y_true)
#     y_pred = vec_reshape(y_pred)
#     err = y_pred[:, 0] - y_true[:, 0]

#     y = y_true[:, 0]
#     yh = y_pred[:, 0]
#     y_mean = float(np.mean(y))
#     ss_res = float(np.sum((y - yh) ** 2))
#     ss_tot = float(np.sum((y - y_mean) ** 2)) + 1e-12
#     r2 = 1.0 - ss_res / ss_tot

#     return {
#         "rmse": float(np.sqrt(np.mean(err ** 2))),
#         "mae": float(np.mean(np.abs(err))),
#         "p95": float(np.percentile(np.abs(err), 95)),
#         "p99": float(np.percentile(np.abs(err), 99)),
#         "max_abs": float(np.max(np.abs(err))),
#         "r2": float(r2),
#     }

from __future__ import annotations

import numpy as np


def vec_reshape(y):
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y


def compute_scores(Y, Yhat, fit="R2"):
    Y = vec_reshape(Y)
    Yhat = vec_reshape(Yhat)
    y = Y[:, 0]
    yh = Yhat[:, 0]
    fit = fit.lower()

    if fit == "r2":
        denom = np.sum((y - np.mean(y)) ** 2) + 1e-12
        return 100.0 * (1.0 - np.sum((yh - y) ** 2) / denom)

    if fit == "rmse":
        return float(np.sqrt(np.mean((yh - y) ** 2)))

    if fit == "bfr":
        denom = np.sum((y - np.mean(y)) ** 2) + 1e-12
        return 100.0 * (1.0 - np.linalg.norm(yh - y) / np.sqrt(denom))

    raise ValueError("fit must be one of: R2 | BFR | RMSE")


def summarize_err(y_true: np.ndarray, y_pred: np.ndarray, name=""):
    y_true = vec_reshape(y_true)
    y_pred = vec_reshape(y_pred)
    err = y_pred[:, 0] - y_true[:, 0]

    return dict(
        name=name,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        p95=float(np.percentile(np.abs(err), 95)),
        p99=float(np.percentile(np.abs(err), 99)),
        max_abs=float(np.max(np.abs(err))),
    )


def report_fit(name, Y_true, Y_pred):
    Y_true = vec_reshape(Y_true)
    Y_pred = vec_reshape(Y_pred)
    err = Y_pred - Y_true
    return dict(
        name=name,
        max_abs=float(np.max(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        r2=float(compute_scores(Y_true, Y_pred, fit="R2")),
        bfr=float(compute_scores(Y_true, Y_pred, fit="BFR")),
    )


def signal_span(x):
    x = np.asarray(x).reshape(-1)
    return dict(
        min=float(np.min(x)),
        max=float(np.max(x)),
        span=float(np.max(x) - np.min(x)),
        std=float(np.std(x)),
    )


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    p95 = float(np.percentile(np.abs(err), 95))
    p99 = float(np.percentile(np.abs(err), 99))
    max_abs = float(np.max(np.abs(err)))

    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    return {
        "rmse": rmse,
        "mae": mae,
        "p95": p95,
        "p99": p99,
        "max_abs": max_abs,
        "r2": r2,
    }