from __future__ import annotations

import pandas as pd


def flag_unstable_cycles(df: pd.DataFrame, rmse_col: str, threshold: float):
    out = df.copy()
    out["is_unstable"] = out[rmse_col] > float(threshold)
    return out