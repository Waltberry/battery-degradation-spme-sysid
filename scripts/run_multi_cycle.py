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
    n_pts = len(prep["t"]) if ("t" in prep and prep["t"] is not None) else 0

    if cycle_idx == 269:
        return True, "explicit skip: problematic empty/short cycle 269"

    if n_pts < int(min_points):
        return True, f"too short after preparation: n_points={n_pts} < {min_points}"

    return False, ""


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
    mode:
        - measured_only
        - fit_compare
        - residual
    stage_key:
        - "stage2"
        - "stage3a"
        - "stage3b"
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

        elif mode == "fit_compare":
            if stage_key is None:
                raise ValueError("stage_key must be provided for fit_compare mode.")
            stage = item.get(stage_key, None)
            if stage is None:
                ax.axis("off")
                continue

            yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
            ax.plot(t, y, label="Measured")
            ax.plot(t, yhat, "--", label=stage_key.upper())
            ax.set_title(f"Cycle {cycle_idx}")
            ax.set_xlabel("relative time [s]")
            ax.set_ylabel("Voltage [V]")
            ax.grid(True)

        elif mode == "residual":
            if stage_key is None:
                raise ValueError("stage_key must be provided for residual mode.")
            stage = item.get(stage_key, None)
            if stage is None:
                ax.axis("off")
                continue

            yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
            err = yhat - y
            ax.plot(t, err)
            ax.set_title(f"Cycle {cycle_idx}")
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
        stage2 = item["stage2"]
        stage3a = item["stage3a"]
        stage3b = item["stage3b"]

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

        final_stage = stage3b if stage3b is not None else stage2
        final_stage_name = "stage3b" if stage3b is not None else "stage2"

        print(
            f"[INFO] Cycle {cycle_idx} ({idx}/{total_cycles}) "
            f"using final stage: {final_stage_name}"
        )

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

        zhat_from_thetaZ = stage2["zhat_from_thetaZ"]

        if final_stage_name == "stage3b":
            thetaZ_hat = final_stage["thetaZ_hat"]
            state_template = np.asarray(final_stage["xhat"][0], dtype=np.float64)
        else:
            thetaZ_hat = stage2["thetaZ_hat"]
            state_template = np.asarray(stage2["xhat"][0], dtype=np.float64)

        learned_surface_fn = _build_learned_surface_fn(
            thetaZ_hat=thetaZ_hat,
            zhat_from_thetaZ=zhat_from_thetaZ,
            cfg=cfg,
            state_template=state_template,
        )

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

        monitor = extract_monitorable_parameters(
            cfg=cfg,
            stage2_result=stage2,
            stage3a_result=stage3a,
            stage3b_result=stage3b,
        )

        row = {
            "cycle_idx": cycle_idx,
            "final_stage_name": final_stage_name,
            "stage2_rmse": float(stage2["metrics"]["rmse"]),
            "stage2_mae": float(stage2["metrics"]["mae"]),
            "stage2_p95": float(stage2["metrics"]["p95"]),
            "stage2_p99": float(stage2["metrics"]["p99"]),
            "stage2_max_abs": float(stage2["metrics"]["max_abs"]),
            "shape_drift_rmse": float(drift["rmse"]),
            "shape_drift_mae": float(drift["mae"]),
            "shape_drift_max_abs": float(drift["max_abs"]),
            "shape_drift_mean_signed": float(drift["mean_signed"]),
            "shape_drift_std_signed": float(drift["std_signed"]),
        }

        if stage3b is not None:
            row.update(
                {
                    "stage3b_rmse": float(stage3b["metrics"]["rmse"]),
                    "stage3b_mae": float(stage3b["metrics"]["mae"]),
                    "stage3b_p95": float(stage3b["metrics"]["p95"]),
                    "stage3b_p99": float(stage3b["metrics"]["p99"]),
                    "stage3b_max_abs": float(stage3b["metrics"]["max_abs"]),
                    "R0_stage3b": float(stage3b["R0_hat"]),
                }
            )
        else:
            row.update(
                {
                    "stage3b_rmse": np.nan,
                    "stage3b_mae": np.nan,
                    "stage3b_p95": np.nan,
                    "stage3b_p99": np.nan,
                    "stage3b_max_abs": np.nan,
                    "R0_stage3b": np.nan,
                }
            )

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

    _plot_cycle_grid(
        cycle_items=valid_cycle_items,
        title=f"Initial measured voltage for cycles {cycle_start} to {cycle_end}",
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

    has_stage3a = any(item["stage3a"] is not None for item in valid_cycle_items)
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

    has_stage3b = any(item["stage3b"] is not None for item in valid_cycle_items)
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

    plot_metric_vs_cycle(
        df,
        metric_col="stage2_rmse",
        title=f"Stage 2 RMSE vs cycle ({cycle_start}-{cycle_end})",
        ylabel="RMSE [V]",
        save_path=fig_dir / f"stage2_rmse_vs_cycle_{window_tag}.png",
        show=False,
    )

    if "stage3b_rmse" in df.columns and df["stage3b_rmse"].notna().any():
        plot_metric_vs_cycle(
            df,
            metric_col="stage3b_rmse",
            title=f"Stage 3b RMSE vs cycle ({cycle_start}-{cycle_end})",
            ylabel="RMSE [V]",
            save_path=fig_dir / f"rmse_vs_cycle_{window_tag}.png",
            show=False,
        )

    if "R0_stage3b" in df.columns and df["R0_stage3b"].notna().any():
        plot_metric_vs_cycle(
            df,
            metric_col="R0_stage3b",
            title=f"R0 vs cycle ({cycle_start}-{cycle_end})",
            ylabel="R0 [Ohm]",
            save_path=fig_dir / f"r0_vs_cycle_{window_tag}.png",
            show=False,
        )

    plot_metric_vs_cycle(
        df,
        metric_col="shape_drift_rmse",
        title=f"Surrogate shape drift vs cycle ({cycle_start}-{cycle_end})",
        ylabel="Shape drift RMSE [V]",
        save_path=fig_dir / f"shape_drift_vs_cycle_{window_tag}.png",
        show=False,
    )

    plot_thetaA_vs_cycle(
        df,
        save_path=fig_dir / f"thetaA_vs_cycle_{window_tag}.png",
        show=False,
    )

    plot_thetaB_vs_cycle(
        df,
        save_path=fig_dir / f"thetaB_vs_cycle_{window_tag}.png",
        show=False,
    )

    print("Returned per-cycle results:", len(per_cycle_results))
    print("Cycle window:", f"{cycle_start} to {cycle_end}")
    print("Total available cycles:", total_available_cycles)
    print("Valid processed cycles:", len(valid_cycle_items))
    print("Skipped cycles:", len(skipped_cycles))
    if skipped_cycles:
        print("Skipped cycle details:", skipped_cycles[:10])
    print("Saved figures to:", fig_dir.resolve())
    print("Saved metrics to:", metrics_dir.resolve())
    print("Saved tables to:", tables_dir.resolve())


if __name__ == "__main__":
    main()