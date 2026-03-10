#src/battery_deg_spme/analysis/summaries.py
from __future__ import annotations

import numpy as np
import pandas as pd


def dicts_to_dataframe(rows):
    return pd.DataFrame(rows)


def compare_truth_and_estimated_parameters(theta_true, theta_hat, names):
    theta_true = np.asarray(theta_true, dtype=np.float64).reshape(-1)
    theta_hat = np.asarray(theta_hat, dtype=np.float64).reshape(-1)

    if theta_true.shape != theta_hat.shape:
        raise ValueError(
            f"Parameter size mismatch: {theta_true.shape} vs {theta_hat.shape}"
        )

    if len(names) != theta_true.size:
        raise ValueError(
            f"Length of names ({len(names)}) must match parameter count ({theta_true.size})."
        )

    abs_err = np.abs(theta_hat - theta_true)
    rel_err = abs_err / (np.abs(theta_true) + 1e-12)

    return pd.DataFrame(
        {
            "name": list(names),
            "theta_true": theta_true,
            "theta_hat": theta_hat,
            "abs_err": abs_err,
            "rel_err": rel_err,
        }
    )


def parameter_error_summary(theta_true, theta_hat):
    theta_true = np.asarray(theta_true, dtype=np.float64).reshape(-1)
    theta_hat = np.asarray(theta_hat, dtype=np.float64).reshape(-1)

    if theta_true.shape != theta_hat.shape:
        raise ValueError(
            f"Parameter size mismatch: {theta_true.shape} vs {theta_hat.shape}"
        )

    err = theta_hat - theta_true
    abs_err = np.abs(err)
    rel_err = abs_err / (np.abs(theta_true) + 1e-12)

    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "mean_rel_err": float(np.mean(rel_err)),
        "max_rel_err": float(np.max(rel_err)),
    }


def make_stage_summary_table(stage_results):
    rows = []
    for stage_name, result in stage_results.items():
        row = {"stage": stage_name}

        if isinstance(result, dict):
            metrics = result.get("metrics", {})
            if isinstance(metrics, dict):
                for key in ["rmse", "mae", "p95", "p99", "max_abs"]:
                    if key in metrics:
                        row[key] = float(metrics[key])

            if "R0_hat" in result:
                row["R0_hat"] = float(result["R0_hat"])

            if "thetaZ_hat" in result:
                thetaZ = np.asarray(result["thetaZ_hat"], dtype=np.float64).reshape(-1)
                if thetaZ.size > 0:
                    row["thetaZ_l2_norm"] = float(np.linalg.norm(thetaZ, 2))
                    row["thetaZ_max_abs"] = float(np.max(np.abs(thetaZ)))

        rows.append(row)

    return pd.DataFrame(rows)