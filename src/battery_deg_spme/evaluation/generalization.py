from __future__ import annotations

from typing import Callable

import pandas as pd


def summarize_generalization_rows(rows):
    df = pd.DataFrame(rows)
    summary = {}

    for col in [
        "stage2_rmse",
        "stage2_r2",
        "stage2_bfr",
        "stage3b_rmse",
        "stage3b_r2",
        "stage3b_bfr",
    ]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                summary[col] = {
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }

    return df, summary


def evaluate_all_cycles(
    cycles,
    fit_function: Callable,
    metric_function: Callable,
):
    """
    Fit model to each cycle and collect metrics.

    Expects each cycle item to unpack as:
        t, U, Y
    """
    results = []

    for cycle_id, cycle_data in enumerate(cycles):
        t, U, Y = cycle_data

        fit_result = fit_function(t, U, Y)

        if isinstance(fit_result, dict) and "yhat" in fit_result:
            Yhat = fit_result["yhat"]
        else:
            raise ValueError("fit_function must return a dict containing 'yhat'.")

        metrics = metric_function(Y, Yhat)

        results.append(
            {
                "cycle": cycle_id,
                **metrics,
            }
        )

    return results