#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_s7_s17_c4k_parameter_plots_from_best_runs.py

This fixes the mismatch between the old thesis parameter plots and the new
parameter-comparison plots.

It uses the SAME source logic as the thesis S17 plot:
    results/tables/real_warm_continuation_ctid/<MODEL_ID>/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/<MODEL_ID>/all_cycles_best_runs.csv

Models:
    S7_C4K
    S17_C4K

Cycle window:
    original cycles 34--99

Optional filter:
    keep only good voltage fits using:
        RMSE <= 2 mV
        BFR >= 98 %
        R² >= 99.95 %

This script does not read parameter_long.csv.
"""

from pathlib import Path
import shutil
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

TABLE_ROOT = PROJECT / "results/tables/real_warm_continuation_ctid"

OUT_TABLE_DIR = PROJECT / "results/tables/chapter6_s7_s17_c4k_best_run_parameter_review"
OUT_FIG_DIR = PROJECT / "results/figures/chapter6_s7_s17_c4k_best_run_parameter_review"

THESIS_FIG_DIR = PROJECT / "figures/chapter6/s7_s17_c4k_best_run_parameter_review"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures/chapter6/s7_s17_c4k_best_run_parameter_review"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MODEL_IDS = ["S7_C4K", "S17_C4K"]

CYCLE_START = 34
CYCLE_END = 99

GOOD_RMSE_V = 0.002
GOOD_BFR_PERCENT = 98.0
GOOD_R2_PERCENT = 99.95

USE_GOOD_FIT_ONLY = True

CORE_COLS = [
    ("alpha_n_hat", r"$\alpha_n$"),
    ("alpha_p_hat", r"$\alpha_p$"),
    ("g_n_hat", r"$g_n$"),
    ("g_p_hat", r"$g_p$"),
    ("b_en_hat", r"$b_{e,n}$"),
    ("b_ep_hat", r"$b_{e,p}$"),
]

K_COLS = [
    ("k1_hat", r"$k_1$"),
    ("k2_hat", r"$k_2$"),
    ("k3_hat", r"$k_3$"),
    ("k4_hat", r"$k_4$"),
    ("k5_hat", r"$k_5$"),
]

FIG_DPI = 300
TITLE_SIZE = 15
AXIS_SIZE = 12
TICK_SIZE = 10
MARKER_SIZE = 4.2
LINE_WIDTH = 1.9

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("[saved]", path)

    for target in [THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / path.name)
        except Exception as exc:
            warnings.warn(f"Could not copy to {target}: {exc}")

def standardize_summary(df):
    df = df.copy()
    ren = {}
    if "rmse" in df.columns and "best_rmse" not in df.columns:
        ren["rmse"] = "best_rmse"
    if "bfr_percent" in df.columns and "best_bfr_percent" not in df.columns:
        ren["bfr_percent"] = "best_bfr_percent"
    if "r2_percent" in df.columns and "best_r2_percent" not in df.columns:
        ren["r2_percent"] = "best_r2_percent"
    if ren:
        df = df.rename(columns=ren)
    return df

def load_model_best_table(model_id: str):
    summary_path = TABLE_ROOT / model_id / "all_cycles_summary.csv"
    best_path = TABLE_ROOT / model_id / "all_cycles_best_runs.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    summary = standardize_summary(pd.read_csv(summary_path))
    best = pd.read_csv(best_path)

    summary["cycle_index"] = summary["cycle_index"].astype(int)
    best["cycle_index"] = best["cycle_index"].astype(int)

    summary = summary[summary["cycle_index"].between(CYCLE_START, CYCLE_END)].copy()
    best = best[best["cycle_index"].between(CYCLE_START, CYCLE_END)].copy()

    summary["model_id"] = model_id
    best["model_id"] = model_id

    summary["is_good_voltage_fit"] = (
        (summary["best_rmse"] <= GOOD_RMSE_V)
        & (summary["best_bfr_percent"] >= GOOD_BFR_PERCENT)
        & (summary["best_r2_percent"] >= GOOD_R2_PERCENT)
    )

    keep_cols = [
        "model_id",
        "cycle_index",
        "best_rmse",
        "best_bfr_percent",
        "best_r2_percent",
        "is_good_voltage_fit",
    ]

    keep_cols = [c for c in keep_cols if c in summary.columns]

    merged = best.merge(summary[keep_cols], on=["model_id", "cycle_index"], how="left")

    if USE_GOOD_FIT_ONLY:
        merged = merged[merged["is_good_voltage_fit"] == True].copy()

    merged["retained_cycle_index"] = merged["cycle_index"] - CYCLE_START
    merged = merged.sort_values("cycle_index").reset_index(drop=True)

    return merged, summary

def clean_axis(ax):
    ax.grid(True, alpha=0.30)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

def plot_dashboard(df, model_id, cols, title, filename):
    available = [(c, lab) for c, lab in cols if c in df.columns]

    if not available:
        print("[skip]", model_id, title, "no columns")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.5 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (col, label) in zip(axes, available):
        d = df.sort_values("retained_cycle_index")
        ax.plot(
            d["retained_cycle_index"],
            pd.to_numeric(d[col], errors="coerce"),
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
        )
        ax.set_title(label, fontsize=AXIS_SIZE)
        ax.set_ylabel("parameter value", fontsize=AXIS_SIZE)
        clean_axis(ax)

    for ax in axes[len(available):]:
        ax.axis("off")

    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)
    fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(OUT_FIG_DIR / filename)

def plot_s7_vs_s17(df_all, cols, title, filename):
    available = [(c, lab) for c, lab in cols if c in df_all.columns]

    if not available:
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.5 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (col, label) in zip(axes, available):
        for model_id in MODEL_IDS:
            d = df_all[df_all["model_id"] == model_id].sort_values("retained_cycle_index")
            if col not in d.columns or len(d) == 0:
                continue
            ax.plot(
                d["retained_cycle_index"],
                pd.to_numeric(d[col], errors="coerce"),
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=model_id,
            )
        ax.set_title(label, fontsize=AXIS_SIZE)
        ax.set_ylabel("parameter value", fontsize=AXIS_SIZE)
        clean_axis(ax)

    for ax in axes[len(available):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)
    fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(OUT_FIG_DIR / filename)

def make_summary(df_all):
    param_cols = [c for c, _ in CORE_COLS + K_COLS if c in df_all.columns]

    rows = []
    for model_id in MODEL_IDS:
        d = df_all[df_all["model_id"] == model_id].copy()

        for col in param_cols:
            y = pd.to_numeric(d[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(y) == 0:
                continue

            rows.append(
                {
                    "model_id": model_id,
                    "parameter_column": col,
                    "n_cycles": len(y),
                    "mean_value": float(np.mean(y)),
                    "median_value": float(np.median(y)),
                    "std_value": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
                    "min_value": float(np.min(y)),
                    "max_value": float(np.max(y)),
                    "first_value": float(y[0]),
                    "last_value": float(y[-1]),
                }
            )

    return pd.DataFrame(rows)

def main():
    print("=" * 100)
    print("S7/S17 C4K PARAMETER PLOTS FROM COMBINED BEST-RUN TABLES")
    print("=" * 100)
    print("USE_GOOD_FIT_ONLY:", USE_GOOD_FIT_ONLY)

    frames = []
    summary_frames = []

    for model_id in MODEL_IDS:
        best, summary = load_model_best_table(model_id)
        frames.append(best)
        summary_frames.append(summary)

        best.to_csv(OUT_TABLE_DIR / f"{model_id}_retained_best_runs_used_for_parameter_plots.csv", index=False)
        summary.to_csv(OUT_TABLE_DIR / f"{model_id}_retained_summary_with_good_fit_flags.csv", index=False)

        print()
        print(model_id)
        print("  parameter rows used:", len(best))
        print("  cycles used:", best["cycle_index"].min(), "to", best["cycle_index"].max())
        print("  available hat columns:", [c for c in best.columns if c.endswith("_hat")][:40])

        plot_dashboard(
            best,
            model_id,
            CORE_COLS,
            f"{model_id}: core/direct-input parameters from best-run table",
            f"fig_ch6_{model_id}_core_parameters_best_runs.png",
        )

        plot_dashboard(
            best,
            model_id,
            K_COLS,
            f"{model_id}: direct electrolyte k parameters from best-run table",
            f"fig_ch6_{model_id}_k_parameters_best_runs.png",
        )

    df_all = pd.concat(frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)

    df_all.to_csv(OUT_TABLE_DIR / "S7_S17_C4K_best_run_parameters_used_for_plots.csv", index=False)
    summary_all.to_csv(OUT_TABLE_DIR / "S7_S17_C4K_retained_summary_good_fit_flags.csv", index=False)

    param_summary = make_summary(df_all)
    param_summary.to_csv(OUT_TABLE_DIR / "S7_S17_C4K_parameter_summary_from_best_runs.csv", index=False)

    plot_s7_vs_s17(
        df_all,
        CORE_COLS,
        "S7_C4K vs S17_C4K: core/direct-input parameters from best-run tables",
        "fig_ch6_S7_vs_S17_C4K_core_parameters_best_runs.png",
    )

    plot_s7_vs_s17(
        df_all,
        K_COLS,
        "S7_C4K vs S17_C4K: direct electrolyte k parameters from best-run tables",
        "fig_ch6_S7_vs_S17_C4K_k_parameters_best_runs.png",
    )

    print()
    print("Saved tables to:")
    print(" ", OUT_TABLE_DIR)
    print("Saved figures to:")
    print(" ", OUT_FIG_DIR)
    print(" ", THESIS_FIG_DIR)
    print("=" * 100)

if __name__ == "__main__":
    main()
