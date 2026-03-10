from __future__ import annotations

from pathlib import Path

import numpy as np

from battery_deg_spme.analysis.nonlinearity import evaluate_surface_on_grid, save_surface_visuals
from battery_deg_spme.analysis.summaries import make_stage_summary_table
from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.fitting.cycle_pipeline import run_single_cycle_pipeline
from battery_deg_spme.io.result_io import save_cycle_metrics_table
from battery_deg_spme.visualization.cycle_plots import plot_cycle_voltage_current, plot_selected_cycle
from battery_deg_spme.visualization.fit_plots import (
    plot_residuals,
    plot_stage_fit_comparison,
    plot_voltage,
)


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


def main():
    settings = get_default_settings()
    result = run_single_cycle_pipeline(settings)

    fig_dir = Path("results/figures/single_cycle")
    tables_dir = Path("results/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    prep = result["prep"]
    proxy = result["proxy"]
    stage2 = result["stage2"]
    stage3a = result["stage3a"]
    stage3b = result["stage3b"]

    plot_selected_cycle(
        cycle_df=result["cycle_df"],
        i_col=settings.data.i_col,
        chosen=result["chosen_cycle_idx"],
        mode=settings.cycle.cycle_mode,
        save_path=fig_dir / "selected_cycle_current.png",
        show=False,
    )

    plot_cycle_voltage_current(
        t=prep["t"],
        u=prep["u"],
        y=prep["y"],
        title="Selected cycle: current and voltage",
        save_path=fig_dir / "selected_cycle_voltage_current.png",
        show=False,
    )

    plot_voltage(
        prep["t"],
        prep["y"],
        stage2["yhat"],
        title="Stage 2 fit",
        save_path=fig_dir / "stage2_fit.png",
        show=False,
    )

    plot_residuals(
        prep["t"],
        prep["y"],
        stage2["yhat"],
        title="Stage 2 residuals",
        save_path=fig_dir / "stage2_residuals.png",
        show=False,
    )

    if stage3a is not None:
        plot_voltage(
            prep["t"],
            prep["y"],
            stage3a["yhat"],
            title="Stage 3a fit",
            save_path=fig_dir / "stage3a_fit.png",
            show=False,
        )

        plot_residuals(
            prep["t"],
            prep["y"],
            stage3a["yhat"],
            title="Stage 3a residuals",
            save_path=fig_dir / "stage3a_residuals.png",
            show=False,
        )

    if stage3b is not None:
        plot_voltage(
            prep["t"],
            prep["y"],
            stage3b["yhat"],
            title="Stage 3b fit",
            save_path=fig_dir / "stage3b_fit.png",
            show=False,
        )

        plot_residuals(
            prep["t"],
            prep["y"],
            stage3b["yhat"],
            title="Stage 3b residuals",
            save_path=fig_dir / "stage3b_residuals.png",
            show=False,
        )

    stage_curves = {"Stage 2": stage2["yhat"]}
    if stage3a is not None:
        stage_curves["Stage 3a"] = stage3a["yhat"]
    if stage3b is not None:
        stage_curves["Stage 3b"] = stage3b["yhat"]

    plot_stage_fit_comparison(
        prep["t"],
        prep["y"],
        stage_curves=stage_curves,
        title="Measured vs Stage fits",
        save_path=fig_dir / "stage_fit_comparison.png",
        show=False,
    )

    stage_results = {"stage2": stage2}
    if stage3a is not None:
        stage_results["stage3a"] = stage3a
    if stage3b is not None:
        stage_results["stage3b"] = stage3b

    stage_summary_df = make_stage_summary_table(stage_results)
    save_cycle_metrics_table(stage_summary_df, tables_dir / "single_cycle_stage_summary.csv")

    zhat_from_thetaZ = stage2["zhat_from_thetaZ"]

    stage2_state_template = (
        np.asarray(stage2["xhat"][0], dtype=np.float64)
        if stage2.get("xhat", None) is not None
        else np.asarray(proxy["X_proxy"][0], dtype=np.float64)
    )
    stage2_surface_fn = _build_learned_surface_fn(
        thetaZ_hat=stage2["thetaZ_hat"],
        zhat_from_thetaZ=zhat_from_thetaZ,
        cfg=cfg,
        state_template=stage2_state_template,
    )

    stage2_surface = evaluate_surface_on_grid(
        surface_fn=stage2_surface_fn,
        n_per_axis=settings.surrogate.nonlinearity_grid_n,
        guard=settings.surrogate.nonlinearity_guard,
    )
    save_surface_visuals(
        result=stage2_surface,
        output_dir=fig_dir,
        prefix="stage2_learned_nonlinearity",
        show=False,
    )

    if stage3b is not None:
        stage3b_state_template = np.asarray(stage3b["xhat"][0], dtype=np.float64)
        stage3b_surface_fn = _build_learned_surface_fn(
            thetaZ_hat=stage3b["thetaZ_hat"],
            zhat_from_thetaZ=zhat_from_thetaZ,
            cfg=cfg,
            state_template=stage3b_state_template,
        )

        stage3b_surface = evaluate_surface_on_grid(
            surface_fn=stage3b_surface_fn,
            n_per_axis=settings.surrogate.nonlinearity_grid_n,
            guard=settings.surrogate.nonlinearity_guard,
        )
        save_surface_visuals(
            result=stage3b_surface,
            output_dir=fig_dir,
            prefix="stage3b_learned_nonlinearity",
            show=False,
        )

    print("Chosen cycle:", result["chosen_cycle_idx"])
    print("Stage 2 RMSE:", stage2["metrics"]["rmse"])
    if stage3a is not None:
        print("Stage 3a RMSE:", stage3a["metrics"]["rmse"])
    if stage3b is not None:
        print("Stage 3b RMSE:", stage3b["metrics"]["rmse"])
    print("Saved figures to:", fig_dir.resolve())
    print("Saved tables to:", tables_dir.resolve())


if __name__ == "__main__":
    main()