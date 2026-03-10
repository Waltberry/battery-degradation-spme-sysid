from __future__ import annotations

import pandas as pd


def build_degradation_story_table(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    preferred_cols = [
        c
        for c in [
            "cycle_idx",
            "stage2_rmse",
            "stage3b_rmse",
            "R0_stage2",
            "R0_stage3b",
            "thetaZ_l2_norm",
            "thetaZ_max_abs",
            "shape_drift_rmse",
            "shape_drift_mae",
            "shape_drift_max_abs",
            "Dn_over_Rn2",
            "Dp_over_Rp2",
            "De_over_eps_L1_sq",
            "De_over_eps_L12_intf_sq",
            "De_over_eps_L2_sq",
            "De_over_eps_L23_intf_sq",
            "De_over_eps_L3_sq",
        ]
        if c in df.columns
    ]

    if preferred_cols:
        return df[preferred_cols].copy()
    return df.copy()


def describe_trend(series: pd.Series, name: str) -> str:
    x = series.dropna().reset_index(drop=True)
    if len(x) < 2:
        return f"{name}: insufficient data"

    delta = float(x.iloc[-1] - x.iloc[0])
    rel = float(delta / (abs(x.iloc[0]) + 1e-12))

    if delta > 0:
        return f"{name}: upward trend (Δ={delta:.6g}, relative={rel:.3%})"
    if delta < 0:
        return f"{name}: downward trend (Δ={delta:.6g}, relative={rel:.3%})"
    return f"{name}: flat trend"


def build_degradation_story_text(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}

    if "R0_stage2" in df.columns:
        out["R0_stage2"] = describe_trend(df["R0_stage2"], "Stage-2 resistance")

    if "R0_stage3b" in df.columns:
        out["R0_stage3b"] = describe_trend(df["R0_stage3b"], "Stage-3b resistance")

    if "stage2_rmse" in df.columns:
        out["stage2_rmse"] = describe_trend(df["stage2_rmse"], "Stage-2 fit quality")

    if "stage3b_rmse" in df.columns:
        out["stage3b_rmse"] = describe_trend(df["stage3b_rmse"], "Stage-3b fit quality")

    if "shape_drift_rmse" in df.columns:
        out["shape_drift_rmse"] = describe_trend(
            df["shape_drift_rmse"],
            "Surrogate shape drift",
        )

    return out