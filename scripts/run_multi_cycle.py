# scripts/run_multi_cycle.py
# real experimental data
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from battery_deg_spme.analysis.degradation_story import (
    build_degradation_story_table,
    build_degradation_story_text,
)
from battery_deg_spme.analysis.nonlinearity import (
    compute_shape_drift,
    evaluate_surface_on_grid,
    save_surface_visuals,
)
from battery_deg_spme.analysis.parameter_extraction import extract_monitorable_parameters
from battery_deg_spme.analysis.summaries import dicts_to_dataframe
from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.fitting.cycle_pipeline import run_all_cycles_pipeline
from battery_deg_spme.io.result_io import (
    save_cycle_metrics_table,
    save_cycle_parameter_table,
    save_json,
)
from battery_deg_spme.visualization.fit_plots import plot_residuals, plot_voltage
from battery_deg_spme.visualization.trend_plots import (
    plot_metric_vs_cycle,
    plot_thetaA_vs_cycle,
    plot_thetaB_vs_cycle,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _build_learned_surface_fn(
    thetaZ_hat: np.ndarray,
    zhat_from_thetaZ,
    cfg,
    state_template: np.ndarray,
):
    """
    Build a learned nonlinear voltage surface evaluator from a fitted
    thetaZ vector and a state template.

    This assumes the full_14 state layout used in this project:
      - x[3] = negative electrode surface concentration
      - x[7] = positive electrode surface concentration
    """
    thetaZ_hat = np.asarray(thetaZ_hat, dtype=np.float64).reshape(-1)
    state_template = np.asarray(state_template, dtype=np.float64).reshape(-1)

    def learned_surface_fn(xn: float, xp: float) -> float:
        x = state_template.copy()
        x[3] = float(xn) * float(cfg.csn_max)
        x[7] = float(xp) * float(cfg.csp_max)
        return float(zhat_from_thetaZ(x, thetaZ_hat))

    return learned_surface_fn


def _finalize_figure(save_path=None, show: bool = False):
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _should_skip_cycle(cycle_idx: int, prep: dict, min_points: int = 10) -> tuple[bool, str]:
    """
    Hard/soft skip rules for problematic cycles.
    """
    n_pts = len(prep["t"]) if ("t" in prep and prep["t"] is not None) else 0

    if cycle_idx == 269:
        return True, "explicit skip: problematic empty/short cycle 269"

    if n_pts < int(min_points):
        return True, f"too short after preparation: n_points={n_pts} < {min_points}"

    return False, ""


def _safe_float(x, default=np.nan) -> float:
    try:
        val = float(x)
        if np.isfinite(val):
            return val
        return float(default)
    except Exception:
        return float(default)


def _stage_rmse_from_result(stage: dict | None, y_true=None) -> float:
    """
    Robust RMSE getter for Stage 2 / Stage 3a / Stage 3b.

    Priority:
      1) stage["metrics"]["rmse"]
      2) stage["err_summary"]["rmse"]
      3) direct RMSE from y_true and stage["yhat"]
      4) inf if unavailable
    """
    if stage is None:
        return float("inf")

    metrics = stage.get("metrics", None)
    if isinstance(metrics, dict) and "rmse" in metrics:
        val = _safe_float(metrics["rmse"], default=np.inf)
        if np.isfinite(val):
            return val

    err_summary = stage.get("err_summary", None)
    if isinstance(err_summary, dict) and "rmse" in err_summary:
        val = _safe_float(err_summary["rmse"], default=np.inf)
        if np.isfinite(val):
            return val

    if y_true is not None and "yhat" in stage:
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        yh = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
        n = min(len(y), len(yh))
        if n > 0:
            return float(np.sqrt(np.mean((yh[:n] - y[:n]) ** 2)))

    return float("inf")


def _stage_metrics(stage: dict | None, y_true=None) -> dict[str, float]:
    """
    Return a consistent metrics dictionary for a stage.

    This keeps Stage 2, Stage 3a, and Stage 3b columns available
    even when some stages are missing.
    """
    if stage is None:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max_abs": np.nan,
            "R0_hat": np.nan,
        }

    metrics = stage.get("metrics", {})
    if metrics is None:
        metrics = {}

    rmse = _stage_rmse_from_result(stage, y_true=y_true)

    mae = _safe_float(metrics.get("mae", np.nan))
    p95 = _safe_float(metrics.get("p95", np.nan))
    p99 = _safe_float(metrics.get("p99", np.nan))
    max_abs = _safe_float(metrics.get("max_abs", np.nan))

    # Fallback direct residual metrics if needed.
    if y_true is not None and "yhat" in stage:
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        yh = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
        n = min(len(y), len(yh))
        if n > 0:
            err = yh[:n] - y[:n]
            if not np.isfinite(mae):
                mae = float(np.mean(np.abs(err)))
            if not np.isfinite(p95):
                p95 = float(np.percentile(np.abs(err), 95))
            if not np.isfinite(p99):
                p99 = float(np.percentile(np.abs(err), 99))
            if not np.isfinite(max_abs):
                max_abs = float(np.max(np.abs(err)))

    return {
        "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
        "mae": float(mae),
        "p95": float(p95),
        "p99": float(p99),
        "max_abs": float(max_abs),
        "R0_hat": _safe_float(stage.get("R0_hat", np.nan)),
    }


def _select_final_stage_by_rmse(
    *,
    prep: dict,
    stage2: dict,
    stage3a: dict | None,
    stage3b: dict | None,
) -> tuple[str, dict, dict[str, float]]:
    """
    Select final stage by actual RMSE.

    This keeps Stage 3 in the thesis workflow, but does not force
    Stage 3 to be the final model if it worsens the fit.
    """
    candidate_stages = {"stage2": stage2}

    if stage3a is not None:
        candidate_stages["stage3a"] = stage3a

    if stage3b is not None:
        candidate_stages["stage3b"] = stage3b

    y_true = prep["y"]

    stage_scores = {
        name: _stage_rmse_from_result(stage, y_true=y_true)
        for name, stage in candidate_stages.items()
    }

    final_stage_name = min(stage_scores, key=stage_scores.get)
    final_stage = candidate_stages[final_stage_name]

    return final_stage_name, final_stage, stage_scores


def _get_theta_and_state_for_surface(
    *,
    final_stage_name: str,
    final_stage: dict,
    stage2: dict,
    proxy: dict,
):
    """
    Get thetaZ/state template for learned surface visualization.

    zhat_from_thetaZ is normally created in Stage 2 and is reused by
    later stages. If a Stage 3 object does not carry xhat/thetaZ, this
    falls back safely to Stage 2.
    """
    if (
        final_stage is not None
        and "thetaZ_hat" in final_stage
        and "xhat" in final_stage
        and final_stage.get("xhat", None) is not None
    ):
        thetaZ_hat = final_stage["thetaZ_hat"]
        state_template = np.asarray(final_stage["xhat"][0], dtype=np.float64)
        return thetaZ_hat, state_template, final_stage_name

    if stage2 is not None and "thetaZ_hat" in stage2:
        thetaZ_hat = stage2["thetaZ_hat"]

        if stage2.get("xhat", None) is not None:
            state_template = np.asarray(stage2["xhat"][0], dtype=np.float64)
        else:
            state_template = np.asarray(proxy["X_proxy"][0], dtype=np.float64)

        return thetaZ_hat, state_template, "stage2_surface_fallback"

    raise RuntimeError("Could not find thetaZ/state template for surface evaluation.")


def _plot_cycle_grid(
    cycle_items: list[dict],
    title: str,
    mode: str = "measured_only",
    stage_key: str | None = None,
    ncols: int = 3,
    save_path=None,
    show: bool = False,
):
    """
    Plot cycle grids.

    mode:
      - measured_only
      - fit_compare
      - residual

    stage_key:
      - stage2
      - stage3a
      - stage3b
      - selected_final
    """
    if len(cycle_items) == 0:
        print(f"[INFO] No cycle items to plot for: {title}")
        return

    nrows = math.ceil(len(cycle_items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False)

    for idx, item in enumerate(cycle_items):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        cycle_idx = int(item["cycle_idx"])
        prep = item["prep"]
        t = np.asarray(prep["t"], dtype=np.float64).reshape(-1)
        y = np.asarray(prep["y"], dtype=np.float64).reshape(-1)

        if mode == "measured_only":
            ax.plot(t, y, label="Measured")
            ax.set_title(f"Cycle {cycle_idx}")
            ax.set_xlabel("relative time [s]")
            ax.set_ylabel("Voltage [V]")
            ax.grid(True)

        elif mode in ("fit_compare", "residual"):
            if stage_key is None:
                raise ValueError("stage_key must be provided for fit/residual mode.")

            if stage_key == "selected_final":
                stage = item.get("selected_final_stage_result", None)
                label = item.get("selected_final_stage_name", "selected")
            else:
                stage = item.get(stage_key, None)
                label = stage_key.upper()

            if stage is None:
                ax.axis("off")
                continue

            yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)

            if mode == "fit_compare":
                ax.plot(t, y, label="Measured")
                ax.plot(t, yhat, "--", label=label)
                ax.set_title(f"Cycle {cycle_idx}: {label}")
                ax.set_xlabel("relative time [s]")
                ax.set_ylabel("Voltage [V]")
                ax.grid(True)

            elif mode == "residual":
                n = min(len(y), len(yhat))
                err = yhat[:n] - y[:n]
                ax.plot(t[:n], err)
                ax.set_title(f"Cycle {cycle_idx}: {label} residual")
                ax.set_xlabel("relative time [s]")
                ax.set_ylabel("Pred - Meas [V]")
                ax.grid(True)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    for idx in range(len(cycle_items), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)

    if mode == "fit_compare" and len(cycle_items) > 0:
        axes[0][0].legend()

    _finalize_figure(save_path=save_path, show=show)


def _try_plot_metric(df, metric_col: str, title: str, ylabel: str, save_path, show: bool = False):
    """
    Plot a metric if the column exists and has at least one finite value.
    This prevents end-of-run crashes when a stage was not generated.
    """
    if metric_col not in df.columns:
        print(f"[WARN] Skipping plot; missing column: {metric_col}")
        return

    vals = np.asarray(df[metric_col], dtype=np.float64)
    if not np.isfinite(vals).any():
        print(f"[WARN] Skipping plot; no finite values in column: {metric_col}")
        return

    plot_metric_vs_cycle(
        df,
        metric_col=metric_col,
        title=title,
        ylabel=ylabel,
        save_path=save_path,
        show=show,
    )


def _try_plot_thetaA(df, save_path, show: bool = False):
    try:
        plot_thetaA_vs_cycle(df, save_path=save_path, show=show)
    except Exception as exc:
        print(f"[WARN] Skipping thetaA trend plot: {exc}")


def _try_plot_thetaB(df, save_path, show: bool = False):
    try:
        plot_thetaB_vs_cycle(df, save_path=save_path, show=show)
    except Exception as exc:
        print(f"[WARN] Skipping thetaB trend plot: {exc}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    settings = get_default_settings()

    cycle_start = int(os.environ.get("CYCLE_START", "1"))
    cycle_end_env = os.environ.get("CYCLE_END", None)
    cycle_end = int(cycle_end_env) if cycle_end_env is not None else None

    result = run_all_cycles_pipeline(
        settings,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )

    fig_dir = Path("results/figures/multi_cycle")
    nonlin_dir = fig_dir / "nonlinearity"
    metrics_dir = Path("results/metrics")
    tables_dir = Path("results/tables")

    fig_dir.mkdir(parents=True, exist_ok=True)
    nonlin_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    per_cycle_results = result["per_cycle_results"]
    cycle_start = int(result["cycle_start"])
    cycle_end = int(result["cycle_end"])
    total_available_cycles = int(result["total_available_cycles"])
    window_tag = f"{cycle_start:03d}_{cycle_end:03d}"

    print(
        f"[INFO] Pipeline returned cycle window {cycle_start} to {cycle_end} "
        f"(count={len(per_cycle_results)}) out of total {total_available_cycles} cycles"
    )

    all_cycle_rows = []
    valid_cycle_items = []
    skipped_cycles = []
    reference_surface = None

    total_cycles = len(per_cycle_results)

    for idx, item in enumerate(per_cycle_results, start=1):
        cycle_idx = int(item["cycle_idx"])
        prep = item["prep"]
        proxy = item.get("proxy", {})
        stage2 = item["stage2"]
        stage3a = item.get("stage3a", None)
        stage3b = item.get("stage3b", None)

        print(f"\n[INFO] Starting cycle {cycle_idx} ({idx}/{total_cycles})")

        skip_cycle, reason = _should_skip_cycle(
            cycle_idx=cycle_idx,
            prep=prep,
            min_points=10,
        )
        if skip_cycle:
            skipped_cycles.append({"cycle_idx": cycle_idx, "reason": reason})
            print(f"[SKIP] Cycle {cycle_idx} ({idx}/{total_cycles}): {reason}")
            continue

        final_stage_name, final_stage, stage_scores = _select_final_stage_by_rmse(
            prep=prep,
            stage2=stage2,
            stage3a=stage3a,
            stage3b=stage3b,
        )

        selected_final_rmse = float(stage_scores[final_stage_name])

        print(
            f"[INFO] Cycle {cycle_idx} ({idx}/{total_cycles}) "
            f"selected final stage: {final_stage_name} "
            f"with RMSE={selected_final_rmse:.6g}"
        )
        print(
            "[INFO] Stage scores: "
            + ", ".join(
                f"{name}={score:.6g}" if np.isfinite(score) else f"{name}=nan"
                for name, score in stage_scores.items()
            )
        )

        # Save selected stage back onto the item for grid plotting.
        item["selected_final_stage_name"] = final_stage_name
        item["selected_final_stage_result"] = final_stage

        # -------------------------------------------------------------
        # Per-cycle fit/residual plots for all stages that exist.
        # Stage 3 remains part of the thesis output even when not selected.
        # -------------------------------------------------------------
        plot_voltage(
            prep["t"],
            prep["y"],
            stage2["yhat"],
            title=f"Cycle {cycle_idx}: Stage 2 fit",
            save_path=fig_dir / f"stage2_fit_cycle_{cycle_idx:04d}.png",
            show=False,
        )
        plot_residuals(
            prep["t"],
            prep["y"],
            stage2["yhat"],
            title=f"Cycle {cycle_idx}: Stage 2 residuals",
            save_path=fig_dir / f"stage2_residuals_cycle_{cycle_idx:04d}.png",
            show=False,
        )

        if stage3a is not None:
            plot_voltage(
                prep["t"],
                prep["y"],
                stage3a["yhat"],
                title=f"Cycle {cycle_idx}: Stage 3a fit",
                save_path=fig_dir / f"stage3a_fit_cycle_{cycle_idx:04d}.png",
                show=False,
            )
            plot_residuals(
                prep["t"],
                prep["y"],
                stage3a["yhat"],
                title=f"Cycle {cycle_idx}: Stage 3a residuals",
                save_path=fig_dir / f"stage3a_residuals_cycle_{cycle_idx:04d}.png",
                show=False,
            )

        if stage3b is not None:
            plot_voltage(
                prep["t"],
                prep["y"],
                stage3b["yhat"],
                title=f"Cycle {cycle_idx}: Stage 3b fit",
                save_path=fig_dir / f"stage3b_fit_cycle_{cycle_idx:04d}.png",
                show=False,
            )
            plot_residuals(
                prep["t"],
                prep["y"],
                stage3b["yhat"],
                title=f"Cycle {cycle_idx}: Stage 3b residuals",
                save_path=fig_dir / f"stage3b_residuals_cycle_{cycle_idx:04d}.png",
                show=False,
            )

        plot_voltage(
            prep["t"],
            prep["y"],
            final_stage["yhat"],
            title=f"Cycle {cycle_idx}: Selected final fit ({final_stage_name})",
            save_path=fig_dir / f"selected_final_fit_cycle_{cycle_idx:04d}.png",
            show=False,
        )
        plot_residuals(
            prep["t"],
            prep["y"],
            final_stage["yhat"],
            title=f"Cycle {cycle_idx}: Selected final residuals ({final_stage_name})",
            save_path=fig_dir / f"selected_final_residuals_cycle_{cycle_idx:04d}.png",
            show=False,
        )

        # -------------------------------------------------------------
        # Learned nonlinearity surface for selected final stage.
        # This preserves the Stage 3 story but only uses Stage 3 surface
        # when Stage 3 is actually selected by RMSE.
        # -------------------------------------------------------------
        zhat_from_thetaZ = stage2["zhat_from_thetaZ"]

        thetaZ_hat, state_template, surface_source_name = _get_theta_and_state_for_surface(
            final_stage_name=final_stage_name,
            final_stage=final_stage,
            stage2=stage2,
            proxy=proxy,
        )

        learned_surface_fn = _build_learned_surface_fn(
            thetaZ_hat=thetaZ_hat,
            zhat_from_thetaZ=zhat_from_thetaZ,
            cfg=cfg,
            state_template=state_template,
        )

        try:
            surface_result = evaluate_surface_on_grid(
                surface_fn=learned_surface_fn,
                n_per_axis=settings.surrogate.nonlinearity_grid_n,
                guard=settings.surrogate.nonlinearity_guard,
            )

            save_surface_visuals(
                result=surface_result,
                output_dir=nonlin_dir,
                prefix=f"cycle_{cycle_idx:04d}_{final_stage_name}",
                show=False,
            )

            if reference_surface is None:
                reference_surface = surface_result
                drift = {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "max_abs": 0.0,
                    "mean_signed": 0.0,
                    "std_signed": 0.0,
                }
            else:
                drift = compute_shape_drift(reference_surface.Z, surface_result.Z)

            surface_eval_ok = True

        except Exception as exc:
            print(
                f"[WARN] Surface evaluation failed for cycle {cycle_idx} "
                f"using {final_stage_name}: {exc}"
            )
            drift = {
                "rmse": np.nan,
                "mae": np.nan,
                "max_abs": np.nan,
                "mean_signed": np.nan,
                "std_signed": np.nan,
            }
            surface_eval_ok = False
            surface_source_name = "surface_failed"

        # -------------------------------------------------------------
        # Extract monitorable parameters from all stages.
        # -------------------------------------------------------------
        monitor = extract_monitorable_parameters(
            cfg=cfg,
            stage2_result=stage2,
            stage3a_result=stage3a,
            stage3b_result=stage3b,
        )

        y_true = prep["y"]

        m2 = _stage_metrics(stage2, y_true=y_true)
        m3a = _stage_metrics(stage3a, y_true=y_true)
        m3b = _stage_metrics(stage3b, y_true=y_true)
        mf = _stage_metrics(final_stage, y_true=y_true)

        # -------------------------------------------------------------
        # Main row. Keep both old and new naming:
        #   final_stage_name       = selected final stage
        #   selected_final_stage   = selected final stage
        # -------------------------------------------------------------
        row = {
            "cycle_idx": cycle_idx,

            # Backward-compatible final-stage field.
            "final_stage_name": final_stage_name,

            # Explicit guarded-selection fields.
            "selected_final_stage": final_stage_name,
            "selected_final_rmse": float(stage_scores[final_stage_name]),
            "stage2_rmse_for_selection": float(stage_scores.get("stage2", np.nan)),
            "stage3a_rmse_for_selection": float(stage_scores.get("stage3a", np.nan)),
            "stage3b_rmse_for_selection": float(stage_scores.get("stage3b", np.nan)),

            # Selected final-stage metrics.
            "selected_final_mae": mf["mae"],
            "selected_final_p95": mf["p95"],
            "selected_final_p99": mf["p99"],
            "selected_final_max_abs": mf["max_abs"],
            "selected_final_R0": mf["R0_hat"],

            # Stage 2 metrics.
            "stage2_rmse": m2["rmse"],
            "stage2_mae": m2["mae"],
            "stage2_p95": m2["p95"],
            "stage2_p99": m2["p99"],
            "stage2_max_abs": m2["max_abs"],
            "R0_stage2": m2["R0_hat"],

            # Stage 3a metrics, kept for thesis diagnostics.
            "stage3a_rmse": m3a["rmse"],
            "stage3a_mae": m3a["mae"],
            "stage3a_p95": m3a["p95"],
            "stage3a_p99": m3a["p99"],
            "stage3a_max_abs": m3a["max_abs"],
            "R0_stage3a": m3a["R0_hat"],

            # Stage 3b metrics, kept for thesis diagnostics.
            "stage3b_rmse": m3b["rmse"],
            "stage3b_mae": m3b["mae"],
            "stage3b_p95": m3b["p95"],
            "stage3b_p99": m3b["p99"],
            "stage3b_max_abs": m3b["max_abs"],
            "R0_stage3b": m3b["R0_hat"],

            # Which surface was actually used.
            "surface_source_stage": surface_source_name,
            "surface_eval_ok": bool(surface_eval_ok),

            # Shape drift relative to first valid selected-final surface.
            "shape_drift_rmse": _safe_float(drift["rmse"]),
            "shape_drift_mae": _safe_float(drift["mae"]),
            "shape_drift_max_abs": _safe_float(drift["max_abs"]),
            "shape_drift_mean_signed": _safe_float(drift["mean_signed"]),
            "shape_drift_std_signed": _safe_float(drift["std_signed"]),
        }

        row.update(monitor)

        all_cycle_rows.append(row)
        valid_cycle_items.append(item)

        print(
            f"[DONE] Finished cycle {cycle_idx} "
            f"({idx}/{total_cycles}); valid processed so far: {len(valid_cycle_items)}"
        )

    df = dicts_to_dataframe(all_cycle_rows)

    save_cycle_metrics_table(
        df,
        metrics_dir / f"multi_cycle_metrics_{window_tag}.csv",
    )
    save_cycle_parameter_table(
        df,
        tables_dir / f"multi_cycle_parameter_table_{window_tag}.csv",
    )

    story_df = build_degradation_story_table(all_cycle_rows)
    save_cycle_parameter_table(
        story_df,
        tables_dir / f"multi_cycle_degradation_story_table_{window_tag}.csv",
    )

    story_text = build_degradation_story_text(df)
    save_json(
        story_text,
        tables_dir / f"multi_cycle_degradation_story_text_{window_tag}.json",
    )

    if skipped_cycles:
        save_json(
            skipped_cycles,
            tables_dir / f"multi_cycle_skipped_cycles_{window_tag}.json",
        )

    # -----------------------------------------------------------------
    # Summary plots over the cycle window.
    # -----------------------------------------------------------------
    _plot_cycle_grid(
        cycle_items=valid_cycle_items,
        title=f"Measured voltage for cycles {cycle_start} to {cycle_end}",
        mode="measured_only",
        ncols=3,
        save_path=fig_dir / f"all_cycles_measured_voltage_grid_{window_tag}.png",
        show=False,
    )

    _plot_cycle_grid(
        cycle_items=valid_cycle_items,
        title=f"Stage 2 fits for cycles {cycle_start} to {cycle_end}",
        mode="fit_compare",
        stage_key="stage2",
        ncols=3,
        save_path=fig_dir / f"all_cycles_stage2_fit_grid_{window_tag}.png",
        show=False,
    )

    _plot_cycle_grid(
        cycle_items=valid_cycle_items,
        title=f"Stage 2 residuals for cycles {cycle_start} to {cycle_end}",
        mode="residual",
        stage_key="stage2",
        ncols=3,
        save_path=fig_dir / f"all_cycles_stage2_residual_grid_{window_tag}.png",
        show=False,
    )

    has_stage3a = any(item.get("stage3a", None) is not None for item in valid_cycle_items)
    if has_stage3a:
        _plot_cycle_grid(
            cycle_items=valid_cycle_items,
            title=f"Stage 3a fits for cycles {cycle_start} to {cycle_end}",
            mode="fit_compare",
            stage_key="stage3a",
            ncols=3,
            save_path=fig_dir / f"all_cycles_stage3a_fit_grid_{window_tag}.png",
            show=False,
        )
        _plot_cycle_grid(
            cycle_items=valid_cycle_items,
            title=f"Stage 3a residuals for cycles {cycle_start} to {cycle_end}",
            mode="residual",
            stage_key="stage3a",
            ncols=3,
            save_path=fig_dir / f"all_cycles_stage3a_residual_grid_{window_tag}.png",
            show=False,
        )

    has_stage3b = any(item.get("stage3b", None) is not None for item in valid_cycle_items)
    if has_stage3b:
        _plot_cycle_grid(
            cycle_items=valid_cycle_items,
            title=f"Stage 3b fits for cycles {cycle_start} to {cycle_end}",
            mode="fit_compare",
            stage_key="stage3b",
            ncols=3,
            save_path=fig_dir / f"all_cycles_stage3b_fit_grid_{window_tag}.png",
            show=False,
        )
        _plot_cycle_grid(
            cycle_items=valid_cycle_items,
            title=f"Stage 3b residuals for cycles {cycle_start} to {cycle_end}",
            mode="residual",
            stage_key="stage3b",
            ncols=3,
            save_path=fig_dir / f"all_cycles_stage3b_residual_grid_{window_tag}.png",
            show=False,
        )

    _plot_cycle_grid(
        cycle_items=valid_cycle_items,
        title=f"Selected final fits for cycles {cycle_start} to {cycle_end}",
        mode="fit_compare",
        stage_key="selected_final",
        ncols=3,
        save_path=fig_dir / f"all_cycles_selected_final_fit_grid_{window_tag}.png",
        show=False,
    )

    _plot_cycle_grid(
        cycle_items=valid_cycle_items,
        title=f"Selected final residuals for cycles {cycle_start} to {cycle_end}",
        mode="residual",
        stage_key="selected_final",
        ncols=3,
        save_path=fig_dir / f"all_cycles_selected_final_residual_grid_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="stage2_rmse",
        title=f"Stage 2 RMSE vs cycle ({cycle_start}-{cycle_end})",
        ylabel="RMSE [V]",
        save_path=fig_dir / f"stage2_rmse_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="stage3a_rmse",
        title=f"Stage 3a RMSE vs cycle ({cycle_start}-{cycle_end})",
        ylabel="RMSE [V]",
        save_path=fig_dir / f"stage3a_rmse_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="stage3b_rmse",
        title=f"Stage 3b RMSE vs cycle ({cycle_start}-{cycle_end})",
        ylabel="RMSE [V]",
        save_path=fig_dir / f"stage3b_rmse_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="selected_final_rmse",
        title=f"Selected final RMSE vs cycle ({cycle_start}-{cycle_end})",
        ylabel="RMSE [V]",
        save_path=fig_dir / f"selected_final_rmse_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="R0_stage2",
        title=f"Stage 2 R0 vs cycle ({cycle_start}-{cycle_end})",
        ylabel="R0 [Ohm]",
        save_path=fig_dir / f"stage2_r0_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="R0_stage3b",
        title=f"Stage 3b R0 vs cycle ({cycle_start}-{cycle_end})",
        ylabel="R0 [Ohm]",
        save_path=fig_dir / f"stage3b_r0_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="selected_final_R0",
        title=f"Selected final R0 vs cycle ({cycle_start}-{cycle_end})",
        ylabel="R0 [Ohm]",
        save_path=fig_dir / f"selected_final_r0_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="shape_drift_rmse",
        title=f"Selected-surface shape drift RMSE vs cycle ({cycle_start}-{cycle_end})",
        ylabel="Surface drift RMSE",
        save_path=fig_dir / f"shape_drift_rmse_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_metric(
        df,
        metric_col="shape_drift_max_abs",
        title=f"Selected-surface shape drift max abs vs cycle ({cycle_start}-{cycle_end})",
        ylabel="Surface drift max abs",
        save_path=fig_dir / f"shape_drift_max_abs_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_thetaA(
        df,
        save_path=fig_dir / f"thetaA_vs_cycle_{window_tag}.png",
        show=False,
    )

    _try_plot_thetaB(
        df,
        save_path=fig_dir / f"thetaB_vs_cycle_{window_tag}.png",
        show=False,
    )

    # -----------------------------------------------------------------
    # Final summary.
    # -----------------------------------------------------------------
    if len(df) > 0 and "selected_final_stage" in df.columns:
        print("\n[SUMMARY] Selected final stage counts:")
        print(df["selected_final_stage"].value_counts(dropna=False))

    if len(df) > 0:
        print("\n[SUMMARY] Key RMSE medians:")
        for col in ["stage2_rmse", "stage3a_rmse", "stage3b_rmse", "selected_final_rmse"]:
            if col in df.columns:
                vals = np.asarray(df[col], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    print(f"  {col}: median={float(np.median(vals)):.6g}")

    print("\nSaved outputs:")
    print("  figures:", fig_dir.resolve())
    print("  nonlinearity figures:", nonlin_dir.resolve())
    print("  metrics:", metrics_dir.resolve())
    print("  tables:", tables_dir.resolve())


if __name__ == "__main__":
    main()