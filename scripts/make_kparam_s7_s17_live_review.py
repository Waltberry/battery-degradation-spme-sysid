#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_kparam_s7_s17_live_review.py

Live visualization and diagnostic review for the new direct-K/B runs:

    S7_C4K
    S17_C4K

Purpose
-------
This script scans whatever has already finished, even if the full 5-cycle
Slurm chain is still running.

It reads from two possible places:

1. Final/combined tables, if available:

    results/tables/real_warm_continuation_ctid/S7_C4K/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/S7_C4K/all_cycles_best_runs.csv

    results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_best_runs.csv

2. Live chunk folders, even before combine jobs finish:

    results/real_warm_continuation_ctid/S7_C4K/warmseq_S7_C4K_cycle_...
    results/real_warm_continuation_ctid/S17_C4K/warmseq_S17_C4K_cycle_...

Outputs
-------
Figures:
    results/figures/kparam_s7_s17_live_review/

Tables:
    results/tables/kparam_s7_s17_live_review/

Main plots:
    01_available_cycles_best_rmse_log.png
    02_available_cycles_median_rmse_log.png
    03_available_cycles_r2_bfr.png
    04_success_failure_counts.png
    05_rank_diagnostics.png
    06_diagnostics_dashboard.png
    07_dynamic_gain_parameters_log.png
    08_k_parameters_log.png
    09_b_parameters_log.png
    10_beta_parameters_dashboard.png
    11_latest_best_fit_S7_C4K.png
    12_latest_best_fit_S17_C4K.png
    13_completion_status.png
"""

from __future__ import annotations

import os

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

MODEL_IDS = ["S7_C4K", "S17_C4K"]

RAW_ROOT = PROJECT_DIR / "results/real_warm_continuation_ctid"
TABLE_ROOT = PROJECT_DIR / "results/tables/real_warm_continuation_ctid"

FIG_DIR = PROJECT_DIR / "results/figures/kparam_s7_s17_live_review"
TAB_DIR = PROJECT_DIR / "results/tables/kparam_s7_s17_live_review"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

START_EXPECTED_CYCLE = int(os.environ.get("", "0"))
END_EXPECTED_CYCLE = int(os.environ.get("UN_REVIEW_END_CYCLE", "99"))
EXPECTED_TEST_CYCLES = list(range(START_EXPECTED_CYCLE, END_EXPECTED_CYCLE + 1))


# ============================================================
# Utility
# ============================================================
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}")
        return None


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-300, a / b, np.nan)


def parse_cycle_from_run_folder(name: str) -> int | None:
    """
    Parses folder names like:

        warmseq_S7_C4K_cycle_0_cold_chunk_0_20seeds_...
        warmseq_S17_C4K_cycle_3_warm_chunk_8_10seeds_...

    Returns integer cycle index.
    """
    m = re.search(r"_cycle_(\d+)_", name)
    if not m:
        return None
    return int(m.group(1))


def standardize_best_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}

    if "rmse" in df.columns and "best_rmse" not in df.columns:
        rename_map["rmse"] = "best_rmse"

    if "mae" in df.columns and "best_mae" not in df.columns:
        rename_map["mae"] = "best_mae"

    if "r2_percent" in df.columns and "best_r2_percent" not in df.columns:
        rename_map["r2_percent"] = "best_r2_percent"

    if "bfr_percent" in df.columns and "best_bfr_percent" not in df.columns:
        rename_map["bfr_percent"] = "best_bfr_percent"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def add_derived_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "n_success" in df.columns and "n_fail" in df.columns:
        total = df["n_success"] + df["n_fail"]
        df["fail_rate"] = df["n_fail"] / total.replace(0, np.nan)
    else:
        df["fail_rate"] = np.nan

    if "median_rmse" in df.columns and "best_rmse" in df.columns:
        df["median_to_best_rmse_ratio"] = safe_ratio(df["median_rmse"], df["best_rmse"])
    else:
        df["median_to_best_rmse_ratio"] = np.nan

    return df


# ============================================================
# Load combined tables if available
# ============================================================
def load_combined_model_tables(model_id: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    summary_csv = TABLE_ROOT / model_id / "all_cycles_summary.csv"
    best_csv = TABLE_ROOT / model_id / "all_cycles_best_runs.csv"

    summary = safe_read_csv(summary_csv)
    best = safe_read_csv(best_csv)

    if summary is not None and len(summary):
        summary["cycle_index"] = summary["cycle_index"].astype(int)
        summary["model_id"] = model_id
        summary["source_type"] = "combined_table"

    if best is not None and len(best):
        best = standardize_best_columns(best)
        best["cycle_index"] = best["cycle_index"].astype(int)
        best["model_id"] = model_id
        best["source_type"] = "combined_table"

    return summary, best


# ============================================================
# Load live chunk folders
# ============================================================
def load_live_chunk_tables(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_root = RAW_ROOT / model_id

    run_rows = []
    fail_rows = []

    if not model_root.exists():
        return pd.DataFrame(), pd.DataFrame()

    folders = sorted(
        p for p in model_root.glob(f"warmseq_{model_id}_cycle_*")
        if p.is_dir()
    )

    for folder in folders:
        cycle = parse_cycle_from_run_folder(folder.name)
        if cycle is None:
            continue

        all_runs_csv = folder / "all_runs.csv"
        failed_csv = folder / "failed_runs.csv"

        df = safe_read_csv(all_runs_csv)
        if df is not None and len(df):
            df = standardize_best_columns(df)
            df["cycle_index"] = df["cycle_index"].astype(int)
            df["cycle_index_from_folder"] = cycle
            df["model_id"] = model_id
            df["source_folder"] = str(folder)
            df["run_folder"] = folder.name
            df["source_type"] = "live_chunk"
            run_rows.append(df)

        dfail = safe_read_csv(failed_csv)
        if dfail is not None and len(dfail):
            dfail["cycle_index_from_folder"] = cycle
            dfail["model_id"] = model_id
            dfail["source_folder"] = str(folder)
            dfail["run_folder"] = folder.name
            dfail["source_type"] = "live_chunk"
            fail_rows.append(dfail)

    if run_rows:
        all_runs = pd.concat(run_rows, ignore_index=True)
    else:
        all_runs = pd.DataFrame()

    if fail_rows:
        all_fails = pd.concat(fail_rows, ignore_index=True)
    else:
        all_fails = pd.DataFrame()

    return all_runs, all_fails


def summarize_live_chunks(model_id: str, live_runs: pd.DataFrame, live_fails: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From all live chunk all_runs.csv files, build:
        summary_live: one row per cycle
        best_live: one best row per cycle
    """
    if live_runs is None or len(live_runs) == 0:
        return pd.DataFrame(), pd.DataFrame()

    live_runs = live_runs.copy()

    # Make sure these columns exist.
    needed = ["cycle_index", "seed", "best_rmse"]
    for c in needed:
        if c not in live_runs.columns:
            raise RuntimeError(f"Missing required column {c} in live_runs for {model_id}")

    if "best_mae" not in live_runs.columns and "mae" in live_runs.columns:
        live_runs["best_mae"] = live_runs["mae"]

    if "best_r2_percent" not in live_runs.columns and "r2_percent" in live_runs.columns:
        live_runs["best_r2_percent"] = live_runs["r2_percent"]

    if "best_bfr_percent" not in live_runs.columns and "bfr_percent" in live_runs.columns:
        live_runs["best_bfr_percent"] = live_runs["bfr_percent"]

    # Best row per cycle.
    best_rows = []
    summary_rows = []

    cycles = sorted(live_runs["cycle_index"].dropna().astype(int).unique())

    for cyc in cycles:
        d = live_runs[live_runs["cycle_index"].astype(int) == cyc].copy()
        d = d.sort_values("best_rmse").reset_index(drop=True)

        f = pd.DataFrame()
        if live_fails is not None and len(live_fails):
            if "cycle_index" in live_fails.columns:
                f = live_fails[live_fails["cycle_index"].astype(int) == cyc]
            elif "cycle_index_from_folder" in live_fails.columns:
                f = live_fails[live_fails["cycle_index_from_folder"].astype(int) == cyc]

        best = d.head(1).copy()
        best_rows.append(best)

        row = {
            "model_id": model_id,
            "cycle_index": int(cyc),
            "source_type": "live_chunk_aggregate",
            "n_success": int(len(d)),
            "n_fail": int(len(f)),
            "best_seed": int(best.iloc[0]["seed"]),
            "best_rmse": float(best.iloc[0]["best_rmse"]),
            "median_rmse": float(d["best_rmse"].median()),
            "mean_rmse": float(d["best_rmse"].mean()),
            "std_rmse": float(d["best_rmse"].std(ddof=1)) if len(d) > 1 else np.nan,
        }

        # Optional columns from best row.
        optional_cols = [
            "best_mae",
            "best_r2_percent",
            "best_bfr_percent",
            "rank_phi_raw",
            "ncols_phi_raw",
            "cond_phi_raw",
            "rank_X_raw",
            "ncols_X_raw",
            "cond_X_raw",
            "best_rank_phi_raw",
            "best_ncols_phi_raw",
            "best_cond_phi_raw",
            "best_rank_X_raw",
            "best_ncols_X_raw",
            "best_cond_X_raw",
            "ljung_box_Q",
            "ljung_box_p_value",
            "source_folder",
            "run_folder",
        ]

        for c in optional_cols:
            if c in best.columns:
                row[c] = best.iloc[0][c]

        # Normalize rank names for summary.
        if "rank_phi_raw" in row and "best_rank_phi_raw" not in row:
            row["best_rank_phi_raw"] = row["rank_phi_raw"]
        if "ncols_phi_raw" in row and "best_ncols_phi_raw" not in row:
            row["best_ncols_phi_raw"] = row["ncols_phi_raw"]
        if "cond_phi_raw" in row and "best_cond_phi_raw" not in row:
            row["best_cond_phi_raw"] = row["cond_phi_raw"]

        if "rank_X_raw" in row and "best_rank_X_raw" not in row:
            row["best_rank_X_raw"] = row["rank_X_raw"]
        if "ncols_X_raw" in row and "best_ncols_X_raw" not in row:
            row["best_ncols_X_raw"] = row["ncols_X_raw"]
        if "cond_X_raw" in row and "best_cond_X_raw" not in row:
            row["best_cond_X_raw"] = row["cond_X_raw"]

        summary_rows.append(row)

    best_live = pd.concat(best_rows, ignore_index=True)
    summary_live = pd.DataFrame(summary_rows)

    summary_live = add_derived_summary_columns(summary_live)

    return summary_live, best_live


# ============================================================
# Merge combined and live summaries
# ============================================================
def choose_best_available_summary(combined_summary: pd.DataFrame | None, live_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer combined summary for cycles that have already been combined.
    Use live chunk aggregate for cycles not yet combined.
    """
    frames = []

    if live_summary is not None and len(live_summary):
        frames.append(live_summary)

    if combined_summary is not None and len(combined_summary):
        c = combined_summary.copy()
        c = c.rename(
            columns={
                "best_rank_phi_raw": "best_rank_phi_raw",
                "best_ncols_phi_raw": "best_ncols_phi_raw",
                "best_cond_phi_raw": "best_cond_phi_raw",
                "best_rank_X_raw": "best_rank_X_raw",
                "best_ncols_X_raw": "best_ncols_X_raw",
                "best_cond_X_raw": "best_cond_X_raw",
            }
        )
        c["source_priority"] = 2
        frames.append(c)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    if "source_priority" not in df.columns:
        df["source_priority"] = np.where(df["source_type"].astype(str).str.contains("combined"), 2, 1)

    df = df.sort_values(["model_id", "cycle_index", "source_priority"])
    df = df.drop_duplicates(subset=["model_id", "cycle_index"], keep="last")
    df = df.sort_values(["model_id", "cycle_index"]).reset_index(drop=True)

    df = add_derived_summary_columns(df)

    return df


def choose_best_available_best(combined_best: pd.DataFrame | None, live_best: pd.DataFrame) -> pd.DataFrame:
    frames = []

    if live_best is not None and len(live_best):
        lb = live_best.copy()
        lb["source_priority"] = 1
        frames.append(lb)

    if combined_best is not None and len(combined_best):
        cb = combined_best.copy()
        cb = standardize_best_columns(cb)
        cb["source_priority"] = 2
        frames.append(cb)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = standardize_best_columns(df)

    df = df.sort_values(["model_id", "cycle_index", "source_priority", "best_rmse"])
    df = df.drop_duplicates(subset=["model_id", "cycle_index"], keep="last")
    df = df.sort_values(["model_id", "cycle_index"]).reset_index(drop=True)

    return df


# ============================================================
# Plot helpers
# ============================================================
def plot_compare_line(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    filename: str,
    logy: bool = False,
    marker: str = "o",
) -> None:
    if df is None or len(df) == 0:
        print(f"[skip] no data for {filename}")
        return

    if y_col not in df.columns:
        print(f"[skip] missing column {y_col}")
        return

    plt.figure(figsize=(12.5, 6.2))

    for model_id in MODEL_IDS:
        d = df[df["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        x = d["cycle_index"]
        y = pd.to_numeric(d[y_col], errors="coerce").to_numpy(dtype=float)

        if logy:
            y = np.where(y > 0, y, np.nan)
            plt.semilogy(x, y, marker=marker, linewidth=2.2, label=model_id)
        else:
            plt.plot(x, y, marker=marker, linewidth=2.2, label=model_id)

    plt.grid(True, which="both" if logy else "major", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    savefig(FIG_DIR / filename)


def plot_parameter_lines(
    best_df: pd.DataFrame,
    param_cols: list[str],
    title: str,
    filename: str,
    logy: bool = True,
) -> None:
    if best_df is None or len(best_df) == 0:
        print(f"[skip] no best data for {filename}")
        return

    available = [c for c in param_cols if c in best_df.columns]
    if not available:
        print(f"[skip] none of parameter columns found for {filename}")
        return

    plt.figure(figsize=(13, 7))

    for model_id in MODEL_IDS:
        d = best_df[best_df["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        for c in available:
            y = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
            if logy:
                y = np.where(y > 0, y, np.nan)
                plt.semilogy(d["cycle_index"], y, marker="o", linewidth=1.8, label=f"{model_id} {c}")
            else:
                plt.plot(d["cycle_index"], y, marker="o", linewidth=1.8, label=f"{model_id} {c}")

    plt.grid(True, which="both" if logy else "major", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Parameter value" + (" [log scale]" if logy else ""))
    plt.title(title)
    plt.legend(loc="best", fontsize=7, ncols=2)
    savefig(FIG_DIR / filename)

def plot_single_parameter_compare(
    best_df: pd.DataFrame,
    param: str,
    pretty_name: str | None = None,
    filename_prefix: str | None = None,
) -> None:
    """
    Make two plots for one parameter:
        1. Linear scale
        2. Log scale, only for positive values

    This compares S7_C4K and S17_C4K directly on the same axes.
    """
    if best_df is None or len(best_df) == 0:
        print(f"[skip] no best data for {param}")
        return

    if param not in best_df.columns:
        print(f"[skip] missing parameter column: {param}")
        return

    pretty = pretty_name or param
    prefix = filename_prefix or param

    # --------------------------
    # Linear plot
    # --------------------------
    plt.figure(figsize=(12.5, 6.2))

    plotted = False

    for model_id in MODEL_IDS:
        d = best_df[best_df["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        y = pd.to_numeric(d[param], errors="coerce").to_numpy(dtype=float)

        if np.all(~np.isfinite(y)):
            continue

        plt.plot(
            d["cycle_index"],
            y,
            marker="o",
            linewidth=2.2,
            label=model_id,
        )
        plotted = True

    if plotted:
        plt.grid(True, alpha=0.35)
        plt.xlabel("Cycle index")
        plt.ylabel(pretty)
        plt.title(f"S7_C4K vs S17_C4K: {pretty}, linear scale")
        plt.legend(loc="best")
        savefig(FIG_DIR / f"14_param_{prefix}_linear_compare.png")
    else:
        plt.close()
        print(f"[skip] no finite values for linear {param}")

    # --------------------------
    # Log plot
    # --------------------------
    plt.figure(figsize=(12.5, 6.2))

    plotted = False

    for model_id in MODEL_IDS:
        d = best_df[best_df["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        y = pd.to_numeric(d[param], errors="coerce").to_numpy(dtype=float)
        y = np.where(y > 0, y, np.nan)

        if np.all(~np.isfinite(y)):
            continue

        plt.semilogy(
            d["cycle_index"],
            y,
            marker="o",
            linewidth=2.2,
            label=model_id,
        )
        plotted = True

    if plotted:
        plt.grid(True, which="both", alpha=0.35)
        plt.xlabel("Cycle index")
        plt.ylabel(f"{pretty}, log scale")
        plt.title(f"S7_C4K vs S17_C4K: {pretty}, log scale")
        plt.legend(loc="best")
        savefig(FIG_DIR / f"15_param_{prefix}_log_compare.png")
    else:
        plt.close()
        print(f"[skip] no positive values for log {param}")


def plot_parameter_family_compare(
    best_df: pd.DataFrame,
    param_cols: list[str],
    title: str,
    filename_prefix: str,
    logy: bool = False,
) -> None:
    """
    Plot a family of parameters together.

    Example:
        k1_hat, k2_hat, ..., k5_hat

    It overlays both models and all available parameters.
    """
    if best_df is None or len(best_df) == 0:
        print(f"[skip] no best data for {filename_prefix}")
        return

    available = [c for c in param_cols if c in best_df.columns]

    if not available:
        print(f"[skip] no available columns for {filename_prefix}")
        return

    plt.figure(figsize=(14, 7.5))

    plotted = False

    for model_id in MODEL_IDS:
        d = best_df[best_df["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        for c in available:
            y = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)

            if logy:
                y = np.where(y > 0, y, np.nan)

            if np.all(~np.isfinite(y)):
                continue

            if logy:
                plt.semilogy(
                    d["cycle_index"],
                    y,
                    marker="o",
                    linewidth=1.8,
                    label=f"{model_id} {c}",
                )
            else:
                plt.plot(
                    d["cycle_index"],
                    y,
                    marker="o",
                    linewidth=1.8,
                    label=f"{model_id} {c}",
                )

            plotted = True

    if not plotted:
        plt.close()
        print(f"[skip] no finite values for {filename_prefix}")
        return

    plt.grid(True, which="both" if logy else "major", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Parameter value" + (" [log scale]" if logy else ""))
    plt.title(title + (" [log scale]" if logy else " [linear scale]"))
    plt.legend(loc="best", fontsize=7, ncols=2)

    suffix = "log" if logy else "linear"
    savefig(FIG_DIR / f"16_family_{filename_prefix}_{suffix}.png")


def plot_parameter_family_dashboard(
    best_df: pd.DataFrame,
    param_cols: list[str],
    title: str,
    filename: str,
    logy: bool = False,
) -> None:
    """
    Make one subplot per parameter, with S7_C4K and S17_C4K compared
    inside each subplot.

    This is useful for alpha_n to alpha_n, g_n to g_n, b_en to b_en, etc.
    """
    if best_df is None or len(best_df) == 0:
        print(f"[skip] no best data for dashboard {filename}")
        return

    available = [c for c in param_cols if c in best_df.columns]

    if not available:
        print(f"[skip] no dashboard columns for {filename}")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, param in zip(axes, available):
        plotted = False

        for model_id in MODEL_IDS:
            d = best_df[best_df["model_id"] == model_id].sort_values("cycle_index")
            if len(d) == 0:
                continue

            y = pd.to_numeric(d[param], errors="coerce").to_numpy(dtype=float)

            if logy:
                y = np.where(y > 0, y, np.nan)

            if np.all(~np.isfinite(y)):
                continue

            if logy:
                ax.semilogy(
                    d["cycle_index"],
                    y,
                    marker="o",
                    linewidth=2.0,
                    label=model_id,
                )
            else:
                ax.plot(
                    d["cycle_index"],
                    y,
                    marker="o",
                    linewidth=2.0,
                    label=model_id,
                )

            plotted = True

        ax.grid(True, which="both" if logy else "major", alpha=0.35)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel(param)
        ax.set_title(param)

        if plotted:
            ax.legend(loc="best", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title + (" [log scale]" if logy else " [linear scale]"), fontsize=16)
    savefig(FIG_DIR / filename)


def make_parameter_comparison_suite(best_df: pd.DataFrame) -> None:
    """
    Full parameter comparison suite for S7_C4K vs S17_C4K.

    Produces:
        - one linear and one log plot for each key parameter
        - grouped k plots
        - grouped k_edge plots
        - grouped B/electrolyte input gain plots
        - dashboards for major parameter families
    """
    if best_df is None or len(best_df) == 0:
        print("[skip] no best data for parameter comparison suite")
        return

    # ------------------------------------------------------------
    # Single parameter comparisons: S7 alpha_n vs S17 alpha_n, etc.
    # ------------------------------------------------------------
    single_params = [
        ("alpha_n_hat", "alpha_n_hat", "alpha_n_hat"),
        ("alpha_p_hat", "alpha_p_hat", "alpha_p_hat"),
        ("g_n_hat", "g_n_hat", "g_n_hat"),
        ("g_p_hat", "g_p_hat", "g_p_hat"),

        # Old-style electrolyte gain, only plotted if present.
        ("g_e_hat", "g_e_hat", "g_e_hat"),

        # New direct electrolyte input gains.
        ("b_en_hat", "b_en_hat", "b_en_hat"),
        ("b_ep_hat", "b_ep_hat", "b_ep_hat"),
    ]

    for param, pretty, prefix in single_params:
        plot_single_parameter_compare(
            best_df=best_df,
            param=param,
            pretty_name=pretty,
            filename_prefix=prefix,
        )

    # ------------------------------------------------------------
    # Independent electrolyte k parameters
    # S7 has k1,k2.
    # S17 has k1,k2,k3,k4,k5.
    # The plot automatically skips missing columns for each model.
    # ------------------------------------------------------------
    k_params = [
        "k1_hat",
        "k2_hat",
        "k3_hat",
        "k4_hat",
        "k5_hat",
    ]

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=k_params,
        title="Independent electrolyte coupling parameters k_i: S7_C4K vs S17_C4K",
        filename_prefix="k_independent_all",
        logy=False,
    )

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=k_params,
        title="Independent electrolyte coupling parameters k_i: S7_C4K vs S17_C4K",
        filename_prefix="k_independent_all",
        logy=True,
    )

    # ------------------------------------------------------------
    # Expanded electrolyte edge couplings.
    # This shows the actual edge couplings used inside A_e.
    # ------------------------------------------------------------
    k_edge_params = [
        "k_edge1_hat",
        "k_edge2_hat",
        "k_edge3_hat",
        "k_edge4_hat",
        "k_edge5_hat",
        "k_edge6_hat",
        "k_edge7_hat",
        "k_edge8_hat",
    ]

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=k_edge_params,
        title="Expanded electrolyte edge couplings k_edge: S7_C4K vs S17_C4K",
        filename_prefix="k_edges_all",
        logy=False,
    )

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=k_edge_params,
        title="Expanded electrolyte edge couplings k_edge: S7_C4K vs S17_C4K",
        filename_prefix="k_edges_all",
        logy=True,
    )

    # ------------------------------------------------------------
    # Solid input gains and old electrolyte gain if present.
    # ------------------------------------------------------------
    input_gain_params = [
        "g_n_hat",
        "g_p_hat",
        "g_e_hat",
    ]

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=input_gain_params,
        title="Input gain parameters: g_n, g_p, and g_e if available",
        filename_prefix="input_gains_all",
        logy=False,
    )

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=input_gain_params,
        title="Input gain parameters: g_n, g_p, and g_e if available",
        filename_prefix="input_gains_all",
        logy=True,
    )

    # ------------------------------------------------------------
    # New electrolyte B gains.
    # ------------------------------------------------------------
    b_params = [
        "b_en_hat",
        "b_ep_hat",
    ]

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=b_params,
        title="Direct electrolyte B gains: b_en and b_ep",
        filename_prefix="electrolyte_B_gains_all",
        logy=False,
    )

    plot_parameter_family_compare(
        best_df=best_df,
        param_cols=b_params,
        title="Direct electrolyte B gains: b_en and b_ep",
        filename_prefix="electrolyte_B_gains_all",
        logy=True,
    )

    # ------------------------------------------------------------
    # Dashboards: each parameter gets its own subplot.
    # ------------------------------------------------------------
    core_params = [
        "alpha_n_hat",
        "alpha_p_hat",
        "g_n_hat",
        "g_p_hat",
        "g_e_hat",
        "b_en_hat",
        "b_ep_hat",
    ]

    plot_parameter_family_dashboard(
        best_df=best_df,
        param_cols=core_params,
        title="Core dynamic and input-gain parameters: S7_C4K vs S17_C4K",
        filename="17_core_parameters_dashboard_linear.png",
        logy=False,
    )

    plot_parameter_family_dashboard(
        best_df=best_df,
        param_cols=core_params,
        title="Core dynamic and input-gain parameters: S7_C4K vs S17_C4K",
        filename="18_core_parameters_dashboard_log.png",
        logy=True,
    )

    plot_parameter_family_dashboard(
        best_df=best_df,
        param_cols=k_params,
        title="Independent k_i parameters: S7_C4K vs S17_C4K",
        filename="19_k_parameters_dashboard_linear.png",
        logy=False,
    )

    plot_parameter_family_dashboard(
        best_df=best_df,
        param_cols=k_params,
        title="Independent k_i parameters: S7_C4K vs S17_C4K",
        filename="20_k_parameters_dashboard_log.png",
        logy=True,
    )

    plot_parameter_family_dashboard(
        best_df=best_df,
        param_cols=k_edge_params,
        title="Expanded k_edge parameters: S7_C4K vs S17_C4K",
        filename="21_k_edge_parameters_dashboard_linear.png",
        logy=False,
    )

    plot_parameter_family_dashboard(
        best_df=best_df,
        param_cols=k_edge_params,
        title="Expanded k_edge parameters: S7_C4K vs S17_C4K",
        filename="22_k_edge_parameters_dashboard_log.png",
        logy=True,
    )


def plot_beta_dashboard(best_df: pd.DataFrame) -> None:
    beta_cols = [
        "beta_C",
        "beta_xppow2",
        "beta_xppow3",
        "beta_xppow4",
        "beta_minusxnpow2",
        "beta_minusxnpow3",
        "beta_minusxnpow4",
        "beta_I",
        "beta_zepow2",
        "beta_zepow3",
        "beta_zepow4",
    ]

    available = [c for c in beta_cols if c in best_df.columns]

    if not available:
        print("[skip] no beta columns found")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.0 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, c in zip(axes, available):
        for model_id in MODEL_IDS:
            d = best_df[best_df["model_id"] == model_id].sort_values("cycle_index")
            if len(d) == 0:
                continue
            y = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
            ax.plot(d["cycle_index"], y, marker="o", linewidth=2.0, label=model_id)

        ax.grid(True, alpha=0.35)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel(c)
        ax.set_title(c)
        ax.legend(loc="best", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("S7_C4K and S17_C4K beta coefficients from best runs", fontsize=16)
    savefig(FIG_DIR / "10_beta_parameters_dashboard.png")


def plot_dashboard(summary_df: pd.DataFrame) -> None:
    if summary_df is None or len(summary_df) == 0:
        return

    specs = [
        ("best_rmse", "Best RMSE [V]", True),
        ("median_rmse", "Median RMSE [V]", True),
        ("best_bfr_percent", "Best BFR [%]", False),
        ("n_success", "Successful starts", False),
        ("n_fail", "Failed starts", False),
        ("best_rank_X_raw", "rank(X)", False),
        ("best_rank_phi_raw", "rank(Phi)", False),
        ("fail_rate", "Failure rate", False),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.reshape(-1)

    for ax, (col, ylabel, logy) in zip(axes, specs):
        if col not in summary_df.columns:
            ax.axis("off")
            continue

        for model_id in MODEL_IDS:
            d = summary_df[summary_df["model_id"] == model_id].sort_values("cycle_index")
            if len(d) == 0:
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
        ax.set_title(col)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Live diagnostics for S7_C4K and S17_C4K", fontsize=16)
    savefig(FIG_DIR / "06_diagnostics_dashboard.png")


def find_response_csv_for_best_row(best_row: pd.Series) -> Path | None:
    """
    Try to find best_measured_estimated_response.csv for a best row.

    For live chunks, source_folder is usually available.
    For combined rows, best_source_folder may be available from combine summary,
    but not always. We therefore try source_folder first.
    """
    for col in ["source_folder", "best_source_folder"]:
        if col in best_row.index and pd.notna(best_row[col]):
            p = Path(str(best_row[col])) / "best_measured_estimated_response.csv"
            if p.exists():
                return p

    # Sometimes best_manifest has path, if source folder exists.
    for col in ["source_folder", "best_source_folder"]:
        if col in best_row.index and pd.notna(best_row[col]):
            manifest = Path(str(best_row[col])) / "best_manifest.csv"
            if manifest.exists():
                m = safe_read_csv(manifest)
                if m is not None and len(m) and "response_csv" in m.columns:
                    p = Path(str(m.iloc[0]["response_csv"]))
                    if p.exists():
                        return p

    return None


def plot_latest_best_response(best_df: pd.DataFrame, model_id: str, filename: str) -> None:
    if best_df is None or len(best_df) == 0:
        print(f"[skip] no best data for {model_id}")
        return

    d = best_df[best_df["model_id"] == model_id].sort_values("cycle_index")
    if len(d) == 0:
        print(f"[skip] no best rows for {model_id}")
        return

    # Pick latest available cycle.
    row = d.iloc[-1]
    response_csv = find_response_csv_for_best_row(row)

    if response_csv is None:
        print(f"[skip] could not locate response CSV for latest {model_id}")
        return

    resp = safe_read_csv(response_csv)
    if resp is None or len(resp) == 0:
        print(f"[skip] bad response CSV: {response_csv}")
        return

    needed = ["t_s", "measured_voltage_V", "estimated_voltage_V", "residual_V"]
    if not all(c in resp.columns for c in needed):
        print(f"[skip] response CSV missing needed columns: {response_csv}")
        return

    cycle = int(row["cycle_index"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(resp["t_s"], resp["measured_voltage_V"], linewidth=2.5, label="measured")
    axes[0].plot(resp["t_s"], resp["estimated_voltage_V"], "--", linewidth=2.2, label="estimated")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("Voltage [V]")
    axes[0].legend(loc="best")
    axes[0].set_title(f"{model_id}: latest available best fit, cycle {cycle}")

    axes[1].plot(resp["t_s"], resp["residual_V"], linewidth=1.7)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.2)
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Residual [V]")

    savefig(FIG_DIR / filename)


def plot_completion_status(summary_df: pd.DataFrame) -> None:
    rows = []

    for model_id in MODEL_IDS:
        d = summary_df[summary_df["model_id"] == model_id].copy()
        cycles_found = sorted(d["cycle_index"].dropna().astype(int).unique().tolist())
        found_set = set(cycles_found)

        for cyc in EXPECTED_TEST_CYCLES:
            rows.append(
                {
                    "model_id": model_id,
                    "cycle_index": cyc,
                    "available": cyc in found_set,
                }
            )

    status = pd.DataFrame(rows)
    status.to_csv(TAB_DIR / "completion_status_5cycle_test.csv", index=False)

    plt.figure(figsize=(10, 3.8))

    y_map = {"S7_C4K": 1, "S17_C4K": 2}

    for model_id in MODEL_IDS:
        d = status[status["model_id"] == model_id]
        available = d[d["available"]]
        missing = d[~d["available"]]

        plt.scatter(
            available["cycle_index"],
            [y_map[model_id]] * len(available),
            s=120,
            marker="s",
            label=f"{model_id} available",
        )

        plt.scatter(
            missing["cycle_index"],
            [y_map[model_id]] * len(missing),
            s=120,
            marker="x",
            label=f"{model_id} missing",
        )

    plt.yticks([1, 2], ["S7_C4K", "S17_C4K"])
    plt.xticks(EXPECTED_TEST_CYCLES)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.title("5-cycle test completion status")
    plt.legend(loc="best", fontsize=8, ncols=2)
    savefig(FIG_DIR / "13_completion_status.png")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 100)
    print("LIVE REVIEW: S7_C4K and S17_C4K")
    print("=" * 100)

    all_summary_rows = []
    all_best_rows = []
    all_live_runs = []
    all_live_fails = []

    coverage_rows = []

    for model_id in MODEL_IDS:
        print()
        print("-" * 100)
        print("Model:", model_id)
        print("-" * 100)

        combined_summary, combined_best = load_combined_model_tables(model_id)
        live_runs, live_fails = load_live_chunk_tables(model_id)
        live_summary, live_best = summarize_live_chunks(model_id, live_runs, live_fails)

        best_summary = choose_best_available_summary(combined_summary, live_summary)
        best_table = choose_best_available_best(combined_best, live_best)

        if len(best_summary):
            all_summary_rows.append(best_summary)

        if len(best_table):
            all_best_rows.append(best_table)

        if len(live_runs):
            all_live_runs.append(live_runs)

        if len(live_fails):
            all_live_fails.append(live_fails)

        found_cycles = []
        if len(best_summary):
            found_cycles = sorted(best_summary["cycle_index"].dropna().astype(int).unique().tolist())

        missing_test_cycles = sorted(set(EXPECTED_TEST_CYCLES) - set(found_cycles))

        coverage_rows.append(
            {
                "model_id": model_id,
                "n_cycles_available": len(found_cycles),
                "cycles_available": found_cycles,
                "missing_test_cycles_0_to_4": missing_test_cycles,
                "combined_summary_found": combined_summary is not None and len(combined_summary) > 0,
                "combined_best_found": combined_best is not None and len(combined_best) > 0,
                "n_live_success_rows": int(len(live_runs)),
                "n_live_fail_rows": int(len(live_fails)),
            }
        )

        print("Available cycles:", found_cycles)
        print("Missing cycles from 0--4:", missing_test_cycles)
        print("Live successful rows:", len(live_runs))
        print("Live failed rows:", len(live_fails))

    if not all_summary_rows:
        raise RuntimeError("No S7_C4K or S17_C4K data found yet.")

    summary_all = pd.concat(all_summary_rows, ignore_index=True, sort=False)
    summary_all = summary_all.sort_values(["model_id", "cycle_index"]).reset_index(drop=True)

    if all_best_rows:
        best_all = pd.concat(all_best_rows, ignore_index=True, sort=False)
        best_all = best_all.sort_values(["model_id", "cycle_index"]).reset_index(drop=True)
    else:
        best_all = pd.DataFrame()

    if all_live_runs:
        live_runs_all = pd.concat(all_live_runs, ignore_index=True, sort=False)
    else:
        live_runs_all = pd.DataFrame()

    if all_live_fails:
        live_fails_all = pd.concat(all_live_fails, ignore_index=True, sort=False)
    else:
        live_fails_all = pd.DataFrame()

    coverage_df = pd.DataFrame(coverage_rows)

    # Save tables.
    summary_all.to_csv(TAB_DIR / "s7_s17_kparam_live_summary.csv", index=False)
    best_all.to_csv(TAB_DIR / "s7_s17_kparam_live_best_runs.csv", index=False)
    live_runs_all.to_csv(TAB_DIR / "s7_s17_kparam_live_all_chunk_runs.csv", index=False)
    live_fails_all.to_csv(TAB_DIR / "s7_s17_kparam_live_failed_runs.csv", index=False)
    coverage_df.to_csv(TAB_DIR / "s7_s17_kparam_live_coverage.csv", index=False)

    # Text report.
    report_path = TAB_DIR / "live_review_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("S7_C4K / S17_C4K live review\n")
        f.write("=" * 100 + "\n\n")
        f.write("Coverage:\n")
        f.write(coverage_df.to_string(index=False))
        f.write("\n\nSummary table:\n")
        cols = [
            "model_id",
            "cycle_index",
            "source_type",
            "n_success",
            "n_fail",
            "best_seed",
            "best_rmse",
            "median_rmse",
            "best_r2_percent",
            "best_bfr_percent",
            "best_rank_X_raw",
            "best_ncols_X_raw",
            "best_rank_phi_raw",
            "best_ncols_phi_raw",
        ]
        cols = [c for c in cols if c in summary_all.columns]
        f.write(summary_all[cols].to_string(index=False))
        f.write("\n")

    print(f"[saved] {report_path}")

    # ------------------------------------------------------------
    # Main plots
    # ------------------------------------------------------------
    plot_compare_line(
        summary_all,
        y_col="best_rmse",
        ylabel="Best RMSE [V], log scale",
        title="S7_C4K vs S17_C4K: best RMSE, available cycles",
        filename="01_available_cycles_best_rmse_log.png",
        logy=True,
    )

    plot_compare_line(
        summary_all,
        y_col="median_rmse",
        ylabel="Median RMSE [V], log scale",
        title="S7_C4K vs S17_C4K: median RMSE, available cycles",
        filename="02_available_cycles_median_rmse_log.png",
        logy=True,
        marker="s",
    )

    # R2 and BFR together.
    if "best_r2_percent" in summary_all.columns or "best_bfr_percent" in summary_all.columns:
        plt.figure(figsize=(12.5, 6.2))

        for model_id in MODEL_IDS:
            d = summary_all[summary_all["model_id"] == model_id].sort_values("cycle_index")
            if len(d) == 0:
                continue

            if "best_r2_percent" in d.columns:
                plt.plot(
                    d["cycle_index"],
                    d["best_r2_percent"],
                    marker="o",
                    linewidth=2.2,
                    label=f"{model_id} R2",
                )

            if "best_bfr_percent" in d.columns:
                plt.plot(
                    d["cycle_index"],
                    d["best_bfr_percent"],
                    marker="s",
                    linewidth=2.2,
                    label=f"{model_id} BFR",
                )

        plt.grid(True, alpha=0.35)
        plt.xlabel("Cycle index")
        plt.ylabel("Percent [%]")
        plt.title("S7_C4K and S17_C4K: R2 and BFR, available cycles")
        plt.legend(loc="best")
        savefig(FIG_DIR / "03_available_cycles_r2_bfr.png")

    plot_compare_line(
        summary_all,
        y_col="n_success",
        ylabel="Successful starts",
        title="S7_C4K vs S17_C4K: successful local starts",
        filename="04a_success_counts.png",
        logy=False,
    )

    plot_compare_line(
        summary_all,
        y_col="n_fail",
        ylabel="Failed starts",
        title="S7_C4K vs S17_C4K: failed local starts",
        filename="04b_failure_counts.png",
        logy=False,
    )

    # Combined success/failure plot.
    plt.figure(figsize=(12.5, 6.2))
    for model_id in MODEL_IDS:
        d = summary_all[summary_all["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        if "n_success" in d.columns:
            plt.plot(d["cycle_index"], d["n_success"], marker="o", linewidth=2.2, label=f"{model_id} success")

        if "n_fail" in d.columns:
            plt.plot(d["cycle_index"], d["n_fail"], marker="s", linewidth=2.2, label=f"{model_id} fail")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Count")
    plt.title("S7_C4K and S17_C4K: success/failure counts")
    plt.legend(loc="best")
    savefig(FIG_DIR / "04_success_failure_counts.png")

    # Rank diagnostics.
    plt.figure(figsize=(12.5, 6.2))

    for model_id in MODEL_IDS:
        d = summary_all[summary_all["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        if "best_rank_X_raw" in d.columns:
            plt.plot(d["cycle_index"], d["best_rank_X_raw"], marker="o", linewidth=2.2, label=f"{model_id} rank(X)")

        if "best_rank_phi_raw" in d.columns:
            plt.plot(d["cycle_index"], d["best_rank_phi_raw"], marker="s", linewidth=2.2, label=f"{model_id} rank(Phi)")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Raw numerical rank")
    plt.title("S7_C4K and S17_C4K: raw rank diagnostics")
    plt.legend(loc="best")
    savefig(FIG_DIR / "05_rank_diagnostics.png")

    plot_dashboard(summary_all)

    # Parameter plots from best rows.
    if len(best_all):
        dynamic_gain_params = [
            "alpha_n_hat",
            "alpha_p_hat",
            "g_n_hat",
            "g_p_hat",
        ]

        b_params = [
            "b_en_hat",
            "b_ep_hat",
        ]

        k_params = [
            "k1_hat",
            "k2_hat",
            "k3_hat",
            "k4_hat",
            "k5_hat",
        ]

        k_edge_params = [
            "k_edge1_hat",
            "k_edge2_hat",
            "k_edge3_hat",
            "k_edge4_hat",
            "k_edge5_hat",
            "k_edge6_hat",
            "k_edge7_hat",
            "k_edge8_hat",
        ]

        plot_parameter_lines(
            best_all,
            param_cols=dynamic_gain_params,
            title="S7_C4K and S17_C4K: solid dynamic/input gains",
            filename="07_dynamic_gain_parameters_log.png",
            logy=True,
        )

        plot_parameter_lines(
            best_all,
            param_cols=k_params,
            title="S7_C4K and S17_C4K: independent electrolyte k parameters",
            filename="08a_k_parameters_log.png",
            logy=True,
        )

        plot_parameter_lines(
            best_all,
            param_cols=k_edge_params,
            title="S7_C4K and S17_C4K: expanded electrolyte edge couplings",
            filename="08b_k_edge_parameters_log.png",
            logy=True,
        )

        plot_parameter_lines(
            best_all,
            param_cols=b_params,
            title="S7_C4K and S17_C4K: direct electrolyte B gains",
            filename="09_b_parameters_log.png",
            logy=True,
        )

        make_parameter_comparison_suite(best_all)

        plot_beta_dashboard(best_all)

        plot_latest_best_response(
            best_all,
            model_id="S7_C4K",
            filename="11_latest_best_fit_S7_C4K.png",
        )

        plot_latest_best_response(
            best_all,
            model_id="S17_C4K",
            filename="12_latest_best_fit_S17_C4K.png",
        )

    plot_completion_status(summary_all)

    # ------------------------------------------------------------
    # Print concise console report
    # ------------------------------------------------------------
    print()
    print("=" * 100)
    print("LIVE REVIEW COMPLETE")
    print("=" * 100)
    print()
    print("Coverage:")
    print(coverage_df.to_string(index=False))
    print()
    print("Available cycle summary:")
    cols = [
        "model_id",
        "cycle_index",
        "source_type",
        "n_success",
        "n_fail",
        "best_seed",
        "best_rmse",
        "median_rmse",
        "best_r2_percent",
        "best_bfr_percent",
        "best_rank_X_raw",
        "best_ncols_X_raw",
        "best_rank_phi_raw",
        "best_ncols_phi_raw",
    ]
    cols = [c for c in cols if c in summary_all.columns]
    print(summary_all[cols].to_string(index=False))
    print()
    print("Figures saved to:")
    print(FIG_DIR)
    print()
    print("Tables saved to:")
    print(TAB_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()