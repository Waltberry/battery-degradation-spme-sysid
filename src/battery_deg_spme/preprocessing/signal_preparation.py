from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .resampling import resample_uniform
from .unit_handling import convert_current_to_amps, sanity_report_current_units


def prepare_cycle_data(
    cycle_df: pd.DataFrame,
    i_col: str,
    v_col: str,
    force_units: str = "A",
    v_ref: str = "none",
    resample: bool = True,
    enforce_discharge_only: bool = True,
    raw_discharge_sign: str = "negative",
    tmax: float = -1.0,
    drop_first_n: int = 0,
) -> dict[str, Any]:
    if i_col not in cycle_df.columns:
        raise KeyError(f"Missing current column: {i_col}")
    if v_col not in cycle_df.columns:
        raise KeyError(f"Missing voltage column: {v_col}")

    t = cycle_df.index.to_numpy(dtype=np.float64).reshape(-1)
    i_raw = cycle_df[i_col].to_numpy(dtype=np.float64).reshape(-1)
    v_raw = cycle_df[v_col].to_numpy(dtype=np.float64).reshape(-1)

    if len(t) < 5:
        raise RuntimeError("Cycle has too few samples.")

    unit_report = sanity_report_current_units(i_raw)
    i_a, unit_note = convert_current_to_amps(i_raw, force_units=force_units)

    keep = np.ones_like(t, dtype=bool)
    keep[1:] = t[1:] > t[:-1]
    t, i_a, v_raw = t[keep], i_a[keep], v_raw[keep]

    if len(t) < 5:
        raise RuntimeError("Not enough strictly increasing time samples after cleaning.")

    t = t - t[0]

    if enforce_discharge_only:
        tol = 1e-12
        if raw_discharge_sign == "negative":
            mask = i_a < -tol
        elif raw_discharge_sign == "positive":
            mask = i_a > +tol
        else:
            raise ValueError("raw_discharge_sign must be 'negative' or 'positive'")

        before = len(t)
        t, i_a, v_raw = t[mask], i_a[mask], v_raw[mask]

        if len(t) < 5:
            raise RuntimeError("After discharge-only filter, not enough points.")

        t = t - t[0]
        discharge_note = f"Discharge-only kept {len(t)}/{before} points"
    else:
        discharge_note = "Discharge-only filtering disabled"

    if drop_first_n > 0:
        if len(t) <= drop_first_n + 3:
            raise RuntimeError("drop_first_n is too large for this selected cycle.")
        t = t[drop_first_n:]
        i_a = i_a[drop_first_n:]
        v_raw = v_raw[drop_first_n:]
        t = t - t[0]

    if v_ref == "none":
        v = v_raw
    elif v_ref == "first":
        v = v_raw - float(v_raw[0])
    elif v_ref == "mean":
        v = v_raw - float(np.mean(v_raw))
    else:
        raise ValueError("v_ref must be none|first|mean")

    t_np = t.reshape(-1)
    u_np = i_a.reshape(-1, 1)
    y_np = v.reshape(-1, 1)

    resample_note = "RESAMPLE=False -> keeping original time grid"
    dt_med = None

    if resample and len(t_np) > 2:
        dt_med = float(np.median(np.diff(t_np)))
        if not (np.isfinite(dt_med) and dt_med > 0):
            raise RuntimeError("Bad dt_med; cannot resample.")

        t_np, u_np, y_np = resample_uniform(t_np, u_np, y_np, dt=dt_med)
        resample_note = f"Resampled: dt_med={dt_med:.12g}, len={len(t_np)}"

    if tmax > 0:
        m = t_np <= float(tmax)
        t_np, u_np, y_np = t_np[m], u_np[m], y_np[m]

    return {
        "t": t_np,
        "u": u_np,
        "y": y_np,
        "unit_report": unit_report,
        "unit_note": unit_note,
        "discharge_note": discharge_note,
        "resample_note": resample_note,
        "dt_med": dt_med,
    }