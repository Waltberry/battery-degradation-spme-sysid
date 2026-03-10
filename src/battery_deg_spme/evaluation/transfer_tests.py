from __future__ import annotations

from typing import Any

import numpy as np

from battery_deg_spme.evaluation.metrics import report_fit, summarize_err
from battery_deg_spme.models.spme_proxy import build_proxy_signals
from battery_deg_spme.preprocessing.signal_preparation import prepare_cycle_data


def evaluate_stage2_transfer_on_cycle(
    cycle_df,
    cfg,
    settings,
    stage2_result: dict[str, Any],
):
    prep = prepare_cycle_data(
        cycle_df=cycle_df,
        i_col=settings.data.i_col,
        v_col=settings.data.v_col,
        force_units=settings.data.force_units,
        v_ref=settings.data.v_ref,
        resample=settings.data.resample,
        enforce_discharge_only=settings.data.enforce_discharge_only,
        raw_discharge_sign=settings.data.raw_discharge_sign,
        tmax=settings.data.tmax,
        drop_first_n=settings.data.drop_first_n,
    )

    proxy = build_proxy_signals(
        t_np=prep["t"],
        u_np=prep["u"],
        cfg=cfg,
        xn0=settings.initial_state.xn0,
        xp0=settings.initial_state.xp0,
        ce0_dev=settings.initial_state.ce0_dev,
    )

    model = stage2_result["model"]
    yhat, _ = model.predict(model.x0, prep["u"], prep["t"])

    fit = report_fit("Stage 2 transfer", prep["y"], yhat)
    err_summary = summarize_err(prep["y"], yhat, name="Stage 2 transfer")

    return {
        "prep": prep,
        "proxy": proxy,
        "fit": fit,
        "err_summary": err_summary,
        "yhat": yhat,
    }


def evaluate_stage3b_transfer_on_cycle(
    cycle_df,
    cfg,
    settings,
    stage3b_result: dict[str, Any],
):
    prep = prepare_cycle_data(
        cycle_df=cycle_df,
        i_col=settings.data.i_col,
        v_col=settings.data.v_col,
        force_units=settings.data.force_units,
        v_ref=settings.data.v_ref,
        resample=settings.data.resample,
        enforce_discharge_only=settings.data.enforce_discharge_only,
        raw_discharge_sign=settings.data.raw_discharge_sign,
        tmax=settings.data.tmax,
        drop_first_n=settings.data.drop_first_n,
    )

    proxy = build_proxy_signals(
        t_np=prep["t"],
        u_np=prep["u"],
        cfg=cfg,
        xn0=settings.initial_state.xn0,
        xp0=settings.initial_state.xp0,
        ce0_dev=settings.initial_state.ce0_dev,
    )

    model = stage3b_result["model"]
    yhat, _ = model.predict(model.x0, prep["u"], prep["t"])

    fit = report_fit("Stage 3b transfer", prep["y"], yhat)
    err_summary = summarize_err(prep["y"], yhat, name="Stage 3b transfer")

    return {
        "prep": prep,
        "proxy": proxy,
        "fit": fit,
        "err_summary": err_summary,
        "yhat": yhat,
    }


def run_transfer_suite_all_cycles(
    cycles,
    cfg,
    settings,
    stage2_result: dict[str, Any],
    stage3b_result: dict[str, Any] | None = None,
):
    rows = []

    for idx, cycle_df in enumerate(cycles):
        row = {"cycle_idx": idx}

        try:
            s2 = evaluate_stage2_transfer_on_cycle(cycle_df, cfg, settings, stage2_result)
            row.update(
                {
                    "stage2_rmse": s2["fit"]["rmse"],
                    "stage2_r2": s2["fit"]["r2"],
                    "stage2_bfr": s2["fit"]["bfr"],
                    "stage2_max_abs": s2["fit"]["max_abs"],
                }
            )
        except Exception as exc:
            row.update(
                {
                    "stage2_rmse": np.nan,
                    "stage2_r2": np.nan,
                    "stage2_bfr": np.nan,
                    "stage2_max_abs": np.nan,
                    "stage2_error": str(exc),
                }
            )

        if stage3b_result is not None:
            try:
                s3 = evaluate_stage3b_transfer_on_cycle(cycle_df, cfg, settings, stage3b_result)
                row.update(
                    {
                        "stage3b_rmse": s3["fit"]["rmse"],
                        "stage3b_r2": s3["fit"]["r2"],
                        "stage3b_bfr": s3["fit"]["bfr"],
                        "stage3b_max_abs": s3["fit"]["max_abs"],
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "stage3b_rmse": np.nan,
                        "stage3b_r2": np.nan,
                        "stage3b_bfr": np.nan,
                        "stage3b_max_abs": np.nan,
                        "stage3b_error": str(exc),
                    }
                )

        rows.append(row)

    return rows