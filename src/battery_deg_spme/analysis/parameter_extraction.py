from __future__ import annotations

from typing import Any

import numpy as np


def extract_monitorable_parameters(
    cfg,
    stage2_result: dict[str, Any] | None = None,
    stage3a_result: dict[str, Any] | None = None,
    stage3b_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # --------------------------------------------------
    # Stage 2 quantities
    # --------------------------------------------------
    if stage2_result is not None:
        thetaZ_hat = np.asarray(
            stage2_result.get("thetaZ_hat", []),
            dtype=np.float64,
        ).reshape(-1)

        if thetaZ_hat.size > 0:
            out["thetaZ_l2_norm"] = float(np.linalg.norm(thetaZ_hat, 2))
            out["thetaZ_max_abs"] = float(np.max(np.abs(thetaZ_hat)))
            out["thetaZ_c0"] = float(thetaZ_hat[0])

        if "R0_hat" in stage2_result:
            out["R0_stage2"] = float(stage2_result["R0_hat"])

        if "metrics" in stage2_result and isinstance(stage2_result["metrics"], dict):
            metrics = stage2_result["metrics"]
            for key in ["rmse", "mae", "p95", "p99", "max_abs"]:
                if key in metrics:
                    out[f"stage2_{key}"] = float(metrics[key])

    # --------------------------------------------------
    # Stage 3a metrics only (intermediate stage)
    # --------------------------------------------------
    if stage3a_result is not None and "metrics" in stage3a_result and isinstance(stage3a_result["metrics"], dict):
        metrics = stage3a_result["metrics"]
        for key in ["rmse", "mae", "p95", "p99", "max_abs"]:
            if key in metrics:
                out[f"stage3a_{key}"] = float(metrics[key])

    # --------------------------------------------------
    # Final parameter source:
    # prefer Stage 3b, otherwise fallback to Stage 3a
    # --------------------------------------------------
    thetaA_final = None
    thetaB_final = None

    if stage3b_result is not None:
        out["stage3b_available"] = True

        if "R0_hat" in stage3b_result:
            out["R0_stage3b"] = float(stage3b_result["R0_hat"])

        if "metrics" in stage3b_result and isinstance(stage3b_result["metrics"], dict):
            metrics = stage3b_result["metrics"]
            for key in ["rmse", "mae", "p95", "p99", "max_abs"]:
                if key in metrics:
                    out[f"stage3b_{key}"] = float(metrics[key])

        thetaZ_hat_stage3b = np.asarray(
            stage3b_result.get("thetaZ_hat", []),
            dtype=np.float64,
        ).reshape(-1)
        if thetaZ_hat_stage3b.size > 0:
            out["thetaZ_stage3b_l2_norm"] = float(np.linalg.norm(thetaZ_hat_stage3b, 2))
            out["thetaZ_stage3b_max_abs"] = float(np.max(np.abs(thetaZ_hat_stage3b)))

        thetaA_final = np.asarray(
            stage3b_result.get("thetaA_hat_stage3b", []),
            dtype=np.float64,
        ).reshape(-1)
        thetaB_final = np.asarray(
            stage3b_result.get("thetaB_hat_stage3b", []),
            dtype=np.float64,
        ).reshape(-1)

    elif stage3a_result is not None:
        rawA_hat = np.asarray(
            stage3a_result.get("rawA_hat_stage3a", []),
            dtype=np.float64,
        ).reshape(-1)
        rawB_hat = np.asarray(
            stage3a_result.get("rawB_hat_stage3a", []),
            dtype=np.float64,
        ).reshape(-1)

        if rawA_hat.size == 7:
            thetaA_final = np.exp(rawA_hat)
        if rawB_hat.size == 4:
            thetaB_final = rawB_hat.copy()

    # --------------------------------------------------
    # Final thetaA / thetaB monitoring values
    # --------------------------------------------------
    if thetaA_final is not None and thetaA_final.size == 7:
        for i, value in enumerate(thetaA_final, start=1):
            out[f"thetaA_hat_{i}"] = float(value)

        out["Dn_over_Rn2"] = float(thetaA_final[0])
        out["Dp_over_Rp2"] = float(thetaA_final[1])
        out["De_over_eps_L1_sq"] = float(thetaA_final[2])
        out["De_over_eps_L12_intf_sq"] = float(thetaA_final[3])
        out["De_over_eps_L2_sq"] = float(thetaA_final[4])
        out["De_over_eps_L23_intf_sq"] = float(thetaA_final[5])
        out["De_over_eps_L3_sq"] = float(thetaA_final[6])

    if thetaB_final is not None and thetaB_final.size == 4:
        out["thetaB_hat_8"] = float(thetaB_final[0])
        out["thetaB_hat_9"] = float(thetaB_final[1])
        out["thetaB_hat_10"] = float(thetaB_final[2])
        out["thetaB_hat_11"] = float(thetaB_final[3])

    return out