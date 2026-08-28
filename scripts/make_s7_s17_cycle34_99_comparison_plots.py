#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_s7_s17_cycle34_99_comparison_plots.py

Purpose
-------
Compare S7_C4 and S17_C4 warm-continuation results after removing the
early bad region.

We remove cycles 0--33 and keep cycles 34--99.

This script makes:
    1. Completion checks for S7 and S17.
    2. Filtered CSV tables for cycles 34--99.
    3. Comparison plots for:
        - best RMSE
        - median RMSE
        - R2
        - BFR
        - success/failure counts
        - failure rate
        - rank(X)
        - rank(Phi)
        - every fitted parameter S7 vs S17 on the same plot
        - normalized parameter comparison relative to cycle 34
        - dashboard plots

Inputs
------
results/tables/real_warm_continuation_ctid/S7_C4/all_cycles_summary.csv
results/tables/real_warm_continuation_ctid/S7_C4/all_cycles_best_runs.csv

results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_summary.csv
results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_best_runs.csv

Outputs
-------
results/figures/s7_s17_cycle34_99_comparison/
results/tables/s7_s17_cycle34_99_comparison/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

START_CYCLE = 34
END_CYCLE = 99

MODEL_IDS = ["S7_C4", "S17_C4"]

FIG_DIR = PROJECT_DIR / "results/figures/s7_s17_cycle34_99_comparison"
TAB_DIR = PROJECT_DIR / "results/tables/s7_s17_cycle34_99_comparison"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-300, a / b, np.nan)


def load_model_tables(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_csv = (
        PROJECT_DIR
        / "results/tables/real_warm_continuation_ctid"
        / model_id
        / "all_cycles_summary.csv"
    )

    best_csv = (
        PROJECT_DIR
        / "results/tables/real_warm_continuation_ctid"
        / model_id
        / "all_cycles_best_runs.csv"
    )

    require_file(summary_csv)
    require_file(best_csv)

    summary = pd.read_csv(summary_csv)
    best = pd.read_csv(best_csv)

    summary["cycle_index"] = summary["cycle_index"].astype(int)
    best["cycle_index"] = best["cycle_index"].astype(int)

    summary = summary.sort_values("cycle_index").reset_index(drop=True)
    best = best.sort_values("cycle_index").reset_index(drop=True)

    # Standardize best-run column names.
    rename_map = {}
    if "rmse" in best.columns and "best_rmse" not in best.columns:
        rename_map["rmse"] = "best_rmse"
    if "mae" in best.columns and "best_mae" not in best.columns:
        rename_map["mae"] = "best_mae"
    if "r2_percent" in best.columns and "best_r2_percent" not in best.columns:
        rename_map["r2_percent"] = "best_r2_percent"
    if "bfr_percent" in best.columns and "best_bfr_percent" not in best.columns:
        rename_map["bfr_percent"] = "best_bfr_percent"

    if rename_map:
        best = best.rename(columns=rename_map)

    summary["model_id"] = model_id
    best["model_id"] = model_id

    # Derived diagnostics.
    total = summary["n_success"] + summary["n_fail"]
    summary["fail_rate"] = summary["n_fail"] / total.replace(0, np.nan)
    summary["median_to_best_rmse_ratio"] = safe_ratio(
        summary["median_rmse"],
        summary["best_rmse"],
    )

    return summary, best


def check_completion(summary: pd.DataFrame, model_id: str) -> dict:
    found = set(summary["cycle_index"].astype(int))
    expected = set(range(100))
    missing = sorted(expected - found)

    return {
        "model_id": model_id,
        "n_rows": int(len(summary)),
        "min_cycle": int(summary["cycle_index"].min()),
        "max_cycle": int(summary["cycle_index"].max()),
        "missing_cycles": missing,
        "is_complete_0_to_99": len(missing) == 0
        and int(summary["cycle_index"].min()) == 0
        and int(summary["cycle_index"].max()) == 99,
    }


def filter_cycles(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["cycle_index"] >= START_CYCLE)
        & (df["cycle_index"] <= END_CYCLE)
    ].copy()


def plot_compare_line(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    filename: str,
    logy: bool = False,
    marker: str = "o",
) -> None:
    plt.figure(figsize=(12, 6))

    for model_id in MODEL_IDS:
        d = df[df["model_id"] == model_id].sort_values("cycle_index")
        if y_col not in d.columns:
            continue

        if logy:
            y = pd.to_numeric(d[y_col], errors="coerce").to_numpy(dtype=float)
            y = np.where(y > 0, y, np.nan)
            plt.semilogy(d["cycle_index"], y, marker=marker, linewidth=2.2, label=model_id)
        else:
            plt.plot(d["cycle_index"], d[y_col], marker=marker, linewidth=2.2, label=model_id)

    plt.grid(True, which="both" if logy else "major", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    savefig(FIG_DIR / filename)


def plot_parameter_compare(df_best: pd.DataFrame, param: str, logy: bool = False) -> None:
    if param not in df_best.columns:
        print(f"[skip] parameter missing: {param}")
        return

    plt.figure(figsize=(12, 6))

    for model_id in MODEL_IDS:
        d = df_best[df_best["model_id"] == model_id].sort_values("cycle_index")
        y = pd.to_numeric(d[param], errors="coerce").to_numpy(dtype=float)

        if logy:
            y = np.where(y > 0, y, np.nan)
            plt.semilogy(d["cycle_index"], y, marker="o", linewidth=2.2, label=model_id)
        else:
            plt.plot(d["cycle_index"], y, marker="o", linewidth=2.2, label=model_id)

    plt.grid(True, which="both" if logy else "major", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel(param)
    plt.title(f"S7_C4 vs S17_C4: {param}, cycles {START_CYCLE}--{END_CYCLE}")
    plt.legend(loc="best")

    suffix = "log" if logy else "linear"
    savefig(FIG_DIR / f"param_{param}_{suffix}_compare.png")


def plot_parameter_normalized_compare(df_best: pd.DataFrame, param: str) -> None:
    if param not in df_best.columns:
        print(f"[skip] parameter missing: {param}")
        return

    plt.figure(figsize=(12, 6))

    for model_id in MODEL_IDS:
        d = df_best[df_best["model_id"] == model_id].sort_values("cycle_index").copy()
        y = pd.to_numeric(d[param], errors="coerce").to_numpy(dtype=float)

        if len(y) == 0:
            continue

        base = y[0]
        if not np.isfinite(base) or abs(base) < 1e-300:
            print(f"[skip normalized] {model_id} {param}: bad base value {base}")
            continue

        yn = y / base
        plt.plot(d["cycle_index"], yn, marker="o", linewidth=2.2, label=model_id)

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel(f"{param} / {param}(cycle {START_CYCLE})")
    plt.title(f"S7_C4 vs S17_C4: normalized {param}, cycles {START_CYCLE}--{END_CYCLE}")
    plt.legend(loc="best")
    savefig(FIG_DIR / f"param_{param}_normalized_to_cycle_{START_CYCLE}.png")


# ============================================================
# Main
# ============================================================
def main() -> None:
    all_summary = []
    all_best = []
    completion_rows = []

    for model_id in MODEL_IDS:
        summary, best = load_model_tables(model_id)

        completion = check_completion(summary, model_id)
        completion_rows.append(completion)

        all_summary.append(summary)
        all_best.append(best)

    summary_all = pd.concat(all_summary, ignore_index=True)
    best_all = pd.concat(all_best, ignore_index=True)

    summary_filtered = filter_cycles(summary_all)
    best_filtered = filter_cycles(best_all)

    # Save coverage/completion report.
    completion_txt = TAB_DIR / "completion_check.txt"
    with open(completion_txt, "w", encoding="utf-8") as f:
        f.write("S7/S17 completion check\n")
        f.write("=" * 80 + "\n\n")

        for row in completion_rows:
            f.write(f"Model: {row['model_id']}\n")
            f.write(f"Rows: {row['n_rows']}\n")
            f.write(f"Min cycle: {row['min_cycle']}\n")
            f.write(f"Max cycle: {row['max_cycle']}\n")
            f.write(f"Missing cycles: {row['missing_cycles']}\n")
            f.write(f"Complete 0--99: {row['is_complete_0_to_99']}\n")
            f.write("-" * 80 + "\n")

        f.write("\nFiltered plotting window:\n")
        f.write(f"Cycles kept: {START_CYCLE}--{END_CYCLE}\n")
        f.write("Cycles removed: 0--33\n")

    print(f"[saved] {completion_txt}")

    # Save filtered tables.
    summary_filtered.to_csv(TAB_DIR / "s7_s17_summary_cycles_34_99.csv", index=False)
    best_filtered.to_csv(TAB_DIR / "s7_s17_best_runs_cycles_34_99.csv", index=False)

    # Also save one model per file.
    for model_id in MODEL_IDS:
        summary_filtered[summary_filtered["model_id"] == model_id].to_csv(
            TAB_DIR / f"{model_id}_summary_cycles_34_99.csv",
            index=False,
        )

        best_filtered[best_filtered["model_id"] == model_id].to_csv(
            TAB_DIR / f"{model_id}_best_runs_cycles_34_99.csv",
            index=False,
        )

    # ------------------------------------------------------------
    # Diagnostics: RMSE, R2, BFR
    # ------------------------------------------------------------
    plot_compare_line(
        summary_filtered,
        y_col="best_rmse",
        ylabel="Best RMSE [V], log scale",
        title=f"S7_C4 vs S17_C4: best RMSE, cycles {START_CYCLE}--{END_CYCLE}",
        filename="01_best_rmse_log_compare.png",
        logy=True,
    )

    plot_compare_line(
        summary_filtered,
        y_col="median_rmse",
        ylabel="Median RMSE [V], log scale",
        title=f"S7_C4 vs S17_C4: median RMSE, cycles {START_CYCLE}--{END_CYCLE}",
        filename="02_median_rmse_log_compare.png",
        logy=True,
        marker="s",
    )

    plot_compare_line(
        summary_filtered,
        y_col="best_r2_percent",
        ylabel="Best R2 [%]",
        title=f"S7_C4 vs S17_C4: best R2, cycles {START_CYCLE}--{END_CYCLE}",
        filename="03_best_r2_compare.png",
        logy=False,
    )

    plot_compare_line(
        summary_filtered,
        y_col="best_bfr_percent",
        ylabel="Best BFR [%]",
        title=f"S7_C4 vs S17_C4: best BFR, cycles {START_CYCLE}--{END_CYCLE}",
        filename="04_best_bfr_compare.png",
        logy=False,
    )

    # ------------------------------------------------------------
    # Diagnostics: success/failure
    # ------------------------------------------------------------
    plot_compare_line(
        summary_filtered,
        y_col="n_success",
        ylabel="Successful starts out of 100",
        title=f"S7_C4 vs S17_C4: successful starts, cycles {START_CYCLE}--{END_CYCLE}",
        filename="05_success_count_compare.png",
        logy=False,
    )

    plot_compare_line(
        summary_filtered,
        y_col="n_fail",
        ylabel="Failed starts out of 100",
        title=f"S7_C4 vs S17_C4: failed starts, cycles {START_CYCLE}--{END_CYCLE}",
        filename="06_failure_count_compare.png",
        logy=False,
    )

    plot_compare_line(
        summary_filtered,
        y_col="fail_rate",
        ylabel="Failure rate",
        title=f"S7_C4 vs S17_C4: failure rate, cycles {START_CYCLE}--{END_CYCLE}",
        filename="07_failure_rate_compare.png",
        logy=False,
    )

    # ------------------------------------------------------------
    # Diagnostics: ranks
    # ------------------------------------------------------------
    plot_compare_line(
        summary_filtered,
        y_col="best_rank_X_raw",
        ylabel="rank(X)",
        title=f"S7_C4 vs S17_C4: raw state trajectory rank, cycles {START_CYCLE}--{END_CYCLE}",
        filename="08_rankX_compare.png",
        logy=False,
    )

    plot_compare_line(
        summary_filtered,
        y_col="best_rank_phi_raw",
        ylabel="rank(Phi)",
        title=f"S7_C4 vs S17_C4: raw output feature rank, cycles {START_CYCLE}--{END_CYCLE}",
        filename="09_rankPhi_compare.png",
        logy=False,
    )

    # ------------------------------------------------------------
    # Dashboard: diagnostics on same figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    axes = axes.reshape(-1)

    dashboard_specs = [
        ("best_rmse", "Best RMSE [V]", True),
        ("median_rmse", "Median RMSE [V]", True),
        ("best_bfr_percent", "Best BFR [%]", False),
        ("n_fail", "Failed starts", False),
        ("best_rank_X_raw", "rank(X)", False),
        ("best_rank_phi_raw", "rank(Phi)", False),
    ]

    for ax, (col, ylabel, logy) in zip(axes, dashboard_specs):
        for model_id in MODEL_IDS:
            d = summary_filtered[summary_filtered["model_id"] == model_id].sort_values("cycle_index")
            if col not in d.columns:
                continue

            y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)

            if logy:
                y = np.where(y > 0, y, np.nan)
                ax.semilogy(d["cycle_index"], y, marker="o", linewidth=2.0, label=model_id)
            else:
                ax.plot(d["cycle_index"], y, marker="o", linewidth=2.0, label=model_id)

        ax.grid(True, which="both" if logy else "major", alpha=0.35)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"S7_C4 vs S17_C4 diagnostics, cycles {START_CYCLE}--{END_CYCLE}", fontsize=16)
    savefig(FIG_DIR / "10_diagnostics_dashboard_compare.png")

    # ------------------------------------------------------------
    # Parameter comparisons
    # ------------------------------------------------------------
    param_cols = [
        "alpha_n_hat",
        "alpha_p_hat",
        "K_e_hat",
        "g_n_hat",
        "g_p_hat",
        "g_e_hat",
        "theta_n0_hat",
        "theta_p0_hat",
    ]

    for param in param_cols:
        if param not in best_filtered.columns:
            print(f"[skip] missing parameter column: {param}")
            continue

        # Linear comparison.
        plot_parameter_compare(best_filtered, param, logy=False)

        # Log comparison for strictly positive dynamic/gain parameters.
        if param not in ["theta_n0_hat", "theta_p0_hat"]:
            plot_parameter_compare(best_filtered, param, logy=True)

        # Normalized comparison relative to cycle 34.
        plot_parameter_normalized_compare(best_filtered, param)

    # ------------------------------------------------------------
    # Parameter dashboard: positive parameters log scale
    # ------------------------------------------------------------
    positive_params = [
        "alpha_n_hat",
        "alpha_p_hat",
        "K_e_hat",
        "g_n_hat",
        "g_p_hat",
        "g_e_hat",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    axes = axes.reshape(-1)

    for ax, param in zip(axes, positive_params):
        if param not in best_filtered.columns:
            continue

        for model_id in MODEL_IDS:
            d = best_filtered[best_filtered["model_id"] == model_id].sort_values("cycle_index")
            y = pd.to_numeric(d[param], errors="coerce").to_numpy(dtype=float)
            y = np.where(y > 0, y, np.nan)

            ax.semilogy(d["cycle_index"], y, marker="o", linewidth=2.0, label=model_id)

        ax.grid(True, which="both", alpha=0.35)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel(param)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"S7_C4 vs S17_C4 positive parameters, cycles {START_CYCLE}--{END_CYCLE}", fontsize=16)
    savefig(FIG_DIR / "11_positive_parameter_dashboard_log_compare.png")

    # ------------------------------------------------------------
    # Theta dashboard
    # ------------------------------------------------------------
    theta_params = ["theta_n0_hat", "theta_p0_hat"]

    plt.figure(figsize=(12, 6))

    for param in theta_params:
        if param not in best_filtered.columns:
            continue

        for model_id in MODEL_IDS:
            d = best_filtered[best_filtered["model_id"] == model_id].sort_values("cycle_index")
            plt.plot(
                d["cycle_index"],
                d[param],
                marker="o",
                linewidth=2.0,
                label=f"{model_id} {param}",
            )

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Theta estimate")
    plt.title(f"S7_C4 vs S17_C4 theta parameters, cycles {START_CYCLE}--{END_CYCLE}")
    plt.legend(loc="best", fontsize=8)
    savefig(FIG_DIR / "12_theta_parameter_compare.png")

    # ------------------------------------------------------------
    # Summary stats table
    # ------------------------------------------------------------
    stats_rows = []

    for model_id in MODEL_IDS:
        d = summary_filtered[summary_filtered["model_id"] == model_id]

        row = {
            "model_id": model_id,
            "cycle_start": START_CYCLE,
            "cycle_end": END_CYCLE,
            "n_cycles": len(d),
            "best_rmse_mean": d["best_rmse"].mean(),
            "best_rmse_median": d["best_rmse"].median(),
            "best_rmse_min": d["best_rmse"].min(),
            "best_rmse_max": d["best_rmse"].max(),
            "median_rmse_mean": d["median_rmse"].mean(),
            "median_rmse_median": d["median_rmse"].median(),
            "bfr_mean": d["best_bfr_percent"].mean(),
            "bfr_min": d["best_bfr_percent"].min(),
            "r2_mean": d["best_r2_percent"].mean(),
            "r2_min": d["best_r2_percent"].min(),
            "n_success_mean": d["n_success"].mean(),
            "n_fail_mean": d["n_fail"].mean(),
            "fail_rate_mean": d["fail_rate"].mean(),
            "rankX_median": d["best_rank_X_raw"].median(),
            "rankX_min": d["best_rank_X_raw"].min(),
            "rankX_max": d["best_rank_X_raw"].max(),
            "rankPhi_median": d["best_rank_phi_raw"].median(),
            "rankPhi_min": d["best_rank_phi_raw"].min(),
            "rankPhi_max": d["best_rank_phi_raw"].max(),
        }

        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(TAB_DIR / "comparison_summary_stats_cycles_34_99.csv", index=False)

    print()
    print("=" * 100)
    print("S7/S17 CYCLE 34--99 COMPARISON COMPLETE")
    print("=" * 100)
    print("Completion:")
    for row in completion_rows:
        print(row)
    print()
    print("Summary stats:")
    print(stats_df.to_string(index=False))
    print()
    print("Figures saved to:")
    print(FIG_DIR)
    print()
    print("Tables saved to:")
    print(TAB_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()