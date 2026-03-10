from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from battery_deg_spme.io.data_io import load_mpr_as_dataframe
from battery_deg_spme.models.spme_proxy import Config, build_proxy_signals
from battery_deg_spme.preprocessing.cycle_detection import (
    find_discharging_cycles_with_meta,
    select_cycle,
    summarize_cycles,
)
from battery_deg_spme.preprocessing.signal_preparation import prepare_cycle_data
from battery_deg_spme.fitting.stage2 import fit_stage2_for_cycle
from battery_deg_spme.fitting.stage3 import fit_stage3a_for_cycle, fit_stage3b_for_cycle


def setup_runtime(settings):
    import os

    n_threads = int(
        os.environ.get(
            settings.runtime.slurm_cpus_env_var,
            str(settings.runtime.default_num_threads),
        )
    )

    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n_threads}"
    )
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    if settings.runtime.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)

    return n_threads, jnp.float64


def _load_all_cycles_from_settings(settings):
    df = load_mpr_as_dataframe(
        mpr_path=settings.data.mpr_path,
        time_col=settings.data.time_col,
        i_col=settings.data.i_col,
        v_col=settings.data.v_col,
    )

    min_len_for_search = (
        int(settings.cycle.min_cycle_len)
        if settings.cycle.use_min_cycle_len
        else None
    )

    cycles, cycle_meta = find_discharging_cycles_with_meta(
        df=df,
        i_col=settings.data.i_col,
        tol_i=1e-9,
        min_len=min_len_for_search,
        include_previous_segment=settings.cycle.include_previous_segment,
        n_prev_points=settings.cycle.n_prev_points,
    )

    cycle_summary = summarize_cycles(cycle_meta)

    return {
        "df": df,
        "cycles": cycles,
        "cycle_meta": cycle_meta,
        "cycle_summary": cycle_summary,
    }


def _build_cfg_from_settings(settings):
    cfg = Config(N_series=settings.experiment.n_series_real)
    cfg.discharge_positive = False
    return cfg


def _prepare_single_cycle_bundle(cycle_df, settings, cfg):
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

    return {
        "prep": prep,
        "proxy": proxy,
    }


def run_single_cycle_pipeline(settings) -> dict[str, Any]:
    _, dtype = setup_runtime(settings)

    loaded = _load_all_cycles_from_settings(settings)
    df = loaded["df"]
    cycles = loaded["cycles"]
    cycle_meta = loaded["cycle_meta"]
    cycle_summary = loaded["cycle_summary"]

    cycle_df, meta, chosen, note = select_cycle(
        cycles=cycles,
        cycle_meta=cycle_meta,
        cycle_mode=settings.cycle.cycle_mode,
        cycle_index=settings.cycle.cycle_index,
        random_seed=settings.cycle.random_seed,
    )

    cfg = _build_cfg_from_settings(settings)

    prepared = _prepare_single_cycle_bundle(cycle_df=cycle_df, settings=settings, cfg=cfg)
    prep = prepared["prep"]
    proxy = prepared["proxy"]

    stage2 = fit_stage2_for_cycle(
        t_np=prep["t"],
        u_np=prep["u"],
        y_np=prep["y"],
        proxy=proxy,
        cfg=cfg,
        settings=settings,
        dtype=dtype,
    )

    stage3a = None
    stage3b = None

    if settings.experiment.run_stage3 and settings.experiment.run_stage3a:
        stage3a = fit_stage3a_for_cycle(
            t_np=prep["t"],
            u_np=prep["u"],
            y_np=prep["y"],
            proxy=proxy,
            stage2_result=stage2,
            cfg=cfg,
            settings=settings,
            dtype=dtype,
        )

    if (
        settings.experiment.run_stage3
        and settings.experiment.run_stage3b
        and stage3a is not None
    ):
        stage3b = fit_stage3b_for_cycle(
            t_np=prep["t"],
            u_np=prep["u"],
            y_np=prep["y"],
            proxy=proxy,
            stage2_result=stage2,
            stage3a_result=stage3a,
            cfg=cfg,
            settings=settings,
            dtype=dtype,
        )

    final_result = stage3b if stage3b is not None else stage2
    final_stage_name = "stage3b" if stage3b is not None else "stage2"

    return {
        "df": df,
        "cycles": cycles,
        "cycle_meta": cycle_meta,
        "cycle_summary": cycle_summary,
        "cycle_df": cycle_df,
        "chosen_cycle_idx": chosen,
        "selection_note": note,
        "selected_cycle_meta": meta,
        "prep": prep,
        "cfg": cfg,
        "proxy": proxy,
        "stage2": stage2,
        "stage3a": stage3a,
        "stage3b": stage3b,
        "final_result": final_result,
        "final_stage_name": final_stage_name,
    }


def run_all_cycles_pipeline(settings) -> dict[str, Any]:
    _, dtype = setup_runtime(settings)

    loaded = _load_all_cycles_from_settings(settings)
    df = loaded["df"]
    cycles = loaded["cycles"]
    cycle_meta = loaded["cycle_meta"]
    cycle_summary = loaded["cycle_summary"]

    cfg = _build_cfg_from_settings(settings)

    rows = []
    per_cycle_results = []

    for cycle_idx, cycle_df in enumerate(cycles):
        try:
            prepared = _prepare_single_cycle_bundle(cycle_df=cycle_df, settings=settings, cfg=cfg)
            prep = prepared["prep"]
            proxy = prepared["proxy"]

            stage2 = fit_stage2_for_cycle(
                t_np=prep["t"],
                u_np=prep["u"],
                y_np=prep["y"],
                proxy=proxy,
                cfg=cfg,
                settings=settings,
                dtype=dtype,
            )

            stage3a = None
            stage3b = None

            if settings.experiment.run_stage3 and settings.experiment.run_stage3a:
                stage3a = fit_stage3a_for_cycle(
                    t_np=prep["t"],
                    u_np=prep["u"],
                    y_np=prep["y"],
                    proxy=proxy,
                    stage2_result=stage2,
                    cfg=cfg,
                    settings=settings,
                    dtype=dtype,
                )

            if (
                settings.experiment.run_stage3
                and settings.experiment.run_stage3b
                and stage3a is not None
            ):
                stage3b = fit_stage3b_for_cycle(
                    t_np=prep["t"],
                    u_np=prep["u"],
                    y_np=prep["y"],
                    proxy=proxy,
                    stage2_result=stage2,
                    stage3a_result=stage3a,
                    cfg=cfg,
                    settings=settings,
                    dtype=dtype,
                )

            final_result = stage3b if stage3b is not None else stage2
            final_stage_name = "stage3b" if stage3b is not None else "stage2"

            row = {
                "cycle_idx": cycle_idx,
                "n_samples": int(prep["t"].shape[0]),
                "stage2_rmse": float(stage2["metrics"]["rmse"]),
                "stage2_mae": float(stage2["metrics"]["mae"]),
                "stage2_p95": float(stage2["metrics"]["p95"]),
                "stage2_max_abs": float(stage2["metrics"]["max_abs"]),
                "R0_stage2": float(stage2["R0_hat"]),
                "final_stage_name": final_stage_name,
            }

            if stage3a is not None:
                row.update(
                    {
                        "stage3a_rmse": float(stage3a["metrics"]["rmse"]),
                        "stage3a_mae": float(stage3a["metrics"]["mae"]),
                        "stage3a_p95": float(stage3a["metrics"]["p95"]),
                        "stage3a_max_abs": float(stage3a["metrics"]["max_abs"]),
                    }
                )

            if stage3b is not None:
                row.update(
                    {
                        "stage3b_rmse": float(stage3b["metrics"]["rmse"]),
                        "stage3b_mae": float(stage3b["metrics"]["mae"]),
                        "stage3b_p95": float(stage3b["metrics"]["p95"]),
                        "stage3b_max_abs": float(stage3b["metrics"]["max_abs"]),
                        "R0_stage3": float(stage3b["R0_hat"]),
                    }
                )

            per_cycle_results.append(
                {
                    "cycle_idx": cycle_idx,
                    "cycle_df": cycle_df,
                    "prep": prep,
                    "proxy": proxy,
                    "stage2": stage2,
                    "stage3a": stage3a,
                    "stage3b": stage3b,
                    "final_result": final_result,
                    "final_stage_name": final_stage_name,
                }
            )
            rows.append(row)

        except Exception as exc:
            rows.append(
                {
                    "cycle_idx": cycle_idx,
                    "error": str(exc),
                }
            )

    return {
        "df": df,
        "cycles": cycles,
        "cycle_meta": cycle_meta,
        "cycle_summary": cycle_summary,
        "cfg": cfg,
        "rows": rows,
        "per_cycle_results": per_cycle_results,
        "final_stage_name": "stage3b",
    }