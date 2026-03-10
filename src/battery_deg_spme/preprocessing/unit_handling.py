from __future__ import annotations

from typing import Tuple

import numpy as np


def sanity_report_current_units(i_raw: np.ndarray) -> dict:
    i_raw = np.asarray(i_raw, dtype=np.float64).reshape(-1)
    i_as_a = i_raw
    i_as_ma = i_raw * 1e-3

    return {
        "if_A": {
            "min": float(np.min(i_as_a)),
            "max": float(np.max(i_as_a)),
            "median_abs": float(np.median(np.abs(i_as_a))),
        },
        "if_mA": {
            "min": float(np.min(i_as_ma)),
            "max": float(np.max(i_as_ma)),
            "median_abs": float(np.median(np.abs(i_as_ma))),
        },
    }


def guess_current_in_amps(i_raw: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Heuristic:
    - If median |I| is >= 1.0, very likely raw signal is in mA for battery data.
    - Otherwise treat as A.
    """
    i_raw = np.asarray(i_raw, dtype=np.float64).reshape(-1)
    med = float(np.nanmedian(np.abs(i_raw)))

    if med >= 1.0:
        return i_raw * 1e-3, f"auto units: mA->A (median|I|={med:.6g} mA)"
    return i_raw, f"auto units: treating as A (median|I|={med:.6g})"


def convert_current_to_amps(i_raw: np.ndarray, force_units: str) -> Tuple[np.ndarray, str]:
    mode = str(force_units).lower()

    if mode == "auto":
        return guess_current_in_amps(i_raw)
    if mode == "ma":
        return np.asarray(i_raw, dtype=np.float64) * 1e-3, "FORCE_UNITS=mA => converting to A"
    if mode == "a":
        return np.asarray(i_raw, dtype=np.float64), "FORCE_UNITS=A => using as A"

    raise ValueError("force_units must be 'auto'|'mA'|'A'")