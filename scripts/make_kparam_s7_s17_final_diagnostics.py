#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_kparam_s7_s17_final_diagnostics.py

Final readable diagnostics for completed S7_C4K and S17_C4K direct-K/B runs.

This is NOT a live script.

It reads the completed combined tables:

    results/tables/real_warm_continuation_ctid/S7_C4K/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/S7_C4K/all_cycles_best_runs.csv

    results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_best_runs.csv

It makes readable diagnostics:

    1. Linear RMSE in mV.
    2. Clipped linear RMSE in mV.
    3. RMSE quality bands.
    4. BFR and R2.
    5. Best vs median RMSE ratio.
    6. Success/failure diagnostics.
    7. Rank diagnostics.
    8. Bad-cycle and weak-cycle tables.
    9. Window statistics for cycles 0--4, 5--33, 34--99.
    10. Parameter dashboards, linear and log.
    11. Latest and worst voltage-fit response plots if response CSVs can be found.
    12. A text/markdown report.

Outputs:
    results/figures/kparam_s7_s17_final_diagnostics/
    results/tables/kparam_s7_s17_final_diagnostics/
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

MODEL_IDS = ["S7_C4K", "S17_C4K"]

TABLE_ROOT = PROJECT_DIR / "results/tables/real_warm_continuation_ctid"

FIG_DIR = PROJECT_DIR / "results/figures/kparam_s7_s17_final_diagnostics"
TAB_DIR = PROJECT_DIR / "results/tables/kparam_s7_s17_final_diagnostics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CYCLES = set(range(100))

# Readability thresholds.
GOOD_RMSE_V = 0.002       # 2 mV
WEAK_RMSE_V = 0.010       # 10 mV
BAD_RMSE_V = 0.050        # 50 mV

GOOD_BFR = 98.0
WEAK_BFR = 95.0

GOOD_R2 = 99.95
WEAK_R2 = 99.0

ROLLING_WINDOW = 5


# ============================================================
# Helpers
# ============================================================
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def safe_read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


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


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-300, a / b, np.nan)


def classify_fit(row: pd.Series) -> str:
    rmse = float(row.get("best_rmse", np.nan))
    bfr = float(row.get("best_bfr_percent", np.nan))
    r2 = float(row.get("best_r2_percent", np.nan))

    if (
        np.isfinite(rmse)
        and np.isfinite(bfr)
        and np.isfinite(r2)
        and rmse <= GOOD_RMSE_V
        and bfr >= GOOD_BFR
        and r2 >= GOOD_R2
    ):
        return "good"

    if (
        np.isfinite(rmse)
        and np.isfinite(bfr)
        and rmse <= WEAK_RMSE_V
        and bfr >= WEAK_BFR
    ):
        return "acceptable"

    return "bad_or_weak"


def add_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()

    df["cycle_index"] = df["cycle_index"].astype(int)

    df["best_rmse_mV"] = 1000.0 * df["best_rmse"]
    df["median_rmse_mV"] = 1000.0 * df["median_rmse"]

    if "best_mae" in df.columns:
        df["best_mae_mV"] = 1000.0 * df["best_mae"]

    total = df["n_success"] + df["n_fail"]
    df["fail_rate"] = df["n_fail"] / total.replace(0, np.nan)
    df["fail_rate_percent"] = 100.0 * df["fail_rate"]

    df["median_to_best_rmse_ratio"] = safe_ratio(df["median_rmse"], df["best_rmse"])

    df["fit_class"] = df.apply(classify_fit, axis=1)

    df["flag_good_fit"] = df["fit_class"].eq("good")
    df["flag_acceptable_fit"] = df["fit_class"].eq("acceptable")
    df["flag_bad_or_weak_fit"] = df["fit_class"].eq("bad_or_weak")

    df["flag_high_best_rmse"] = df["best_rmse"] > WEAK_RMSE_V
    df["flag_very_high_best_rmse"] = df["best_rmse"] > BAD_RMSE_V
    df["flag_low_bfr"] = df["best_bfr_percent"] < WEAK_BFR
    df["flag_low_r2"] = df["best_r2_percent"] < WEAK_R2

    if "best_rank_X_raw" in df.columns and "best_ncols_X_raw" in df.columns:
        df["rankX_fraction"] = df["best_rank_X_raw"] / df["best_ncols_X_raw"].replace(0, np.nan)
        df["flag_rankX_collapsed"] = df["best_rank_X_raw"] <= 2
        df["flag_rankX_weak"] = df["rankX_fraction"] < 0.50
    else:
        df["rankX_fraction"] = np.nan
        df["flag_rankX_collapsed"] = False
        df["flag_rankX_weak"] = False

    if "best_rank_phi_raw" in df.columns and "best_ncols_phi_raw" in df.columns:
        df["rankPhi_fraction"] = df["best_rank_phi_raw"] / df["best_ncols_phi_raw"].replace(0, np.nan)
        df["flag_rankPhi_deficient"] = df["best_rank_phi_raw"] < df["best_ncols_phi_raw"]
    else:
        df["rankPhi_fraction"] = np.nan
        df["flag_rankPhi_deficient"] = False

    df["best_rmse_mV_rollmed"] = (
        df.sort_values("cycle_index")
        .groupby("model_id")["best_rmse_mV"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, center=True, min_periods=1).median())
    )

    return df


def load_model_tables(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_csv = TABLE_ROOT / model_id / "all_cycles_summary.csv"
    best_csv = TABLE_ROOT / model_id / "all_cycles_best_runs.csv"

    summary = safe_read_csv(summary_csv)
    best = safe_read_csv(best_csv)

    summary = standardize_best_columns(summary)
    best = standardize_best_columns(best)

    summary["model_id"] = model_id
    best["model_id"] = model_id

    summary["cycle_index"] = summary["cycle_index"].astype(int)
    best["cycle_index"] = best["cycle_index"].astype(int)

    summary = summary.sort_values("cycle_index").reset_index(drop=True)
    best = best.sort_values("cycle_index").reset_index(drop=True)

    return summary, best


def get_window_label(cycle: int) -> str:
    if 0 <= cycle <= 4:
        return "0-4 smoke test"
    if 5 <= cycle <= 33:
        return "5-33 difficult region"
    if 34 <= cycle <= 99:
        return "34-99 later region"
    return "outside expected"


def find_response_csv_from_best_row(row: pd.Series) -> Path | None:
    """
    Try to locate best_measured_estimated_response.csv from best run row.

    This works if all_cycles_best_runs.csv kept source_folder from combine.
    """
    for col in ["source_folder", "best_source_folder"]:
        if col in row.index and pd.notna(row[col]):
            folder = Path(str(row[col]))
            p = folder / "best_measured_estimated_response.csv"
            if p.exists():
                return p

            manifest = folder / "best_manifest.csv"
            if manifest.exists():
                try:
                    m = pd.read_csv(manifest)
                    if len(m) and "response_csv" in m.columns:
                        p2 = Path(str(m.iloc[0]["response_csv"]))
                        if p2.exists():
                            return p2
                except Exception:
                    pass

    return None


# ============================================================
# Basic comparison plots
# ============================================================
def plot_rmse_linear(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 6.5))

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id].sort_values("cycle_index")
        plt.plot(
            d["cycle_index"],
            d["best_rmse_mV"],
            marker="o",
            linewidth=2.0,
            label=f"{model_id} best RMSE",
        )
        plt.plot(
            d["cycle_index"],
            d["best_rmse_mV_rollmed"],
            linestyle="--",
            linewidth=2.0,
            label=f"{model_id} {ROLLING_WINDOW}-cycle rolling median",
        )

    plt.axhline(1000 * GOOD_RMSE_V, linestyle=":", linewidth=1.5, label="2 mV good threshold")
    plt.axhline(1000 * WEAK_RMSE_V, linestyle=":", linewidth=1.5, label="10 mV weak threshold")
    plt.axhline(1000 * BAD_RMSE_V, linestyle=":", linewidth=1.5, label="50 mV bad threshold")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [mV]")
    plt.title("S7_C4K vs S17_C4K: best RMSE, readable linear scale")
    plt.legend(loc="best", fontsize=8, ncols=2)
    savefig(FIG_DIR / "01_best_rmse_mV_linear_full.png")


def plot_rmse_clipped(summary: pd.DataFrame) -> None:
    y = pd.to_numeric(summary["best_rmse_mV"], errors="coerce").to_numpy(dtype=float)
    ymax = np.nanpercentile(y, 90) * 1.25
    ymax = max(ymax, 5.0)

    plt.figure(figsize=(13, 6.5))

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id].sort_values("cycle_index")
        plt.plot(
            d["cycle_index"],
            d["best_rmse_mV"],
            marker="o",
            linewidth=2.0,
            label=model_id,
        )

    plt.axhline(1000 * GOOD_RMSE_V, linestyle=":", linewidth=1.5, label="2 mV")
    plt.axhline(1000 * WEAK_RMSE_V, linestyle=":", linewidth=1.5, label="10 mV")

    plt.ylim(0, ymax)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [mV]")
    plt.title("S7_C4K vs S17_C4K: best RMSE, clipped for readability")
    plt.legend(loc="best")
    savefig(FIG_DIR / "02_best_rmse_mV_linear_clipped.png")


def plot_rmse_quality_scatter(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 6.5))

    markers = {
        "good": "o",
        "acceptable": "s",
        "bad_or_weak": "x",
    }

    for model_id in MODEL_IDS:
        dmodel = summary[summary["model_id"] == model_id].sort_values("cycle_index")

        for klass, marker in markers.items():
            d = dmodel[dmodel["fit_class"] == klass]
            if len(d) == 0:
                continue
            plt.scatter(
                d["cycle_index"],
                d["best_rmse_mV"],
                marker=marker,
                s=55,
                label=f"{model_id} {klass}",
            )

    plt.axhline(1000 * GOOD_RMSE_V, linestyle=":", linewidth=1.5)
    plt.axhline(1000 * WEAK_RMSE_V, linestyle=":", linewidth=1.5)
    plt.axhline(1000 * BAD_RMSE_V, linestyle=":", linewidth=1.5)

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [mV]")
    plt.title("S7_C4K vs S17_C4K: fit-quality classification by RMSE")
    plt.legend(loc="best", fontsize=7, ncols=2)
    savefig(FIG_DIR / "03_rmse_quality_classification_linear.png")


def plot_bfr_r2(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id].sort_values("cycle_index")

        axes[0].plot(
            d["cycle_index"],
            d["best_bfr_percent"],
            marker="o",
            linewidth=2.0,
            label=model_id,
        )

        axes[1].plot(
            d["cycle_index"],
            d["best_r2_percent"],
            marker="o",
            linewidth=2.0,
            label=model_id,
        )

    axes[0].axhline(GOOD_BFR, linestyle=":", linewidth=1.5, label="98%")
    axes[0].axhline(WEAK_BFR, linestyle=":", linewidth=1.5, label="95%")
    axes[0].set_ylabel("Best BFR [%]")
    axes[0].set_title("Best BFR across cycles")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].axhline(GOOD_R2, linestyle=":", linewidth=1.5, label="99.95%")
    axes[1].axhline(WEAK_R2, linestyle=":", linewidth=1.5, label="99%")
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("Best R2 [%]")
    axes[1].set_title("Best R2 across cycles")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best")

    fig.suptitle("S7_C4K vs S17_C4K: readable fit-quality diagnostics", fontsize=16)
    savefig(FIG_DIR / "04_bfr_r2_readable.png")


def plot_best_vs_median(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id].sort_values("cycle_index")

        axes[0].plot(
            d["cycle_index"],
            d["best_rmse_mV"],
            marker="o",
            linewidth=2.0,
            label=f"{model_id} best",
        )
        axes[0].plot(
            d["cycle_index"],
            d["median_rmse_mV"],
            marker="s",
            linewidth=1.8,
            label=f"{model_id} median",
        )

        axes[1].plot(
            d["cycle_index"],
            d["median_to_best_rmse_ratio"],
            marker="o",
            linewidth=2.0,
            label=model_id,
        )

    axes[0].set_ylabel("RMSE [mV]")
    axes[0].set_title("Best RMSE vs median multistart RMSE")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=8, ncols=2)

    axes[1].axhline(2.0, linestyle=":", linewidth=1.5, label="2x")
    axes[1].axhline(10.0, linestyle=":", linewidth=1.5, label="10x")
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("Median RMSE / best RMSE")
    axes[1].set_title("Multistart stability: lower is better")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "05_best_vs_median_rmse_stability.png")


def plot_success_failure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id].sort_values("cycle_index")

        axes[0].plot(
            d["cycle_index"],
            d["n_success"],
            marker="o",
            linewidth=2.0,
            label=f"{model_id} success",
        )
        axes[0].plot(
            d["cycle_index"],
            d["n_fail"],
            marker="s",
            linewidth=2.0,
            label=f"{model_id} fail",
        )

        axes[1].plot(
            d["cycle_index"],
            d["fail_rate_percent"],
            marker="o",
            linewidth=2.0,
            label=model_id,
        )

    axes[0].set_ylabel("Count")
    axes[0].set_title("Successful and failed local starts")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=8, ncols=2)

    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("Failure rate [%]")
    axes[1].set_title("Failure rate per cycle")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "06_success_failure_readable.png")


def plot_rank_diagnostics(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id].sort_values("cycle_index")

        axes[0].plot(
            d["cycle_index"],
            d["best_rank_X_raw"],
            marker="o",
            linewidth=2.0,
            label=f"{model_id} rank(X)",
        )

        axes[1].plot(
            d["cycle_index"],
            d["best_rank_phi_raw"],
            marker="o",
            linewidth=2.0,
            label=f"{model_id} rank(Phi)",
        )

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        if "best_ncols_X_raw" in d.columns:
            axes[0].plot(
                d["cycle_index"],
                d["best_ncols_X_raw"],
                linestyle=":",
                linewidth=1.5,
                label=f"{model_id} max X",
            )

        if "best_ncols_phi_raw" in d.columns:
            axes[1].plot(
                d["cycle_index"],
                d["best_ncols_phi_raw"],
                linestyle=":",
                linewidth=1.5,
                label=f"{model_id} max Phi",
            )

    axes[0].set_ylabel("Raw rank")
    axes[0].set_title("State trajectory rank")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=8, ncols=2)

    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("Raw rank")
    axes[1].set_title("Output feature matrix rank")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best", fontsize=8, ncols=2)

    savefig(FIG_DIR / "07_rank_diagnostics_readable.png")


def plot_region_summary_bars(window_stats: pd.DataFrame) -> None:
    """
    Simple grouped bar-like line markers for window summaries.
    """
    metrics = [
        ("best_rmse_mV_median", "Median best RMSE [mV]"),
        ("best_bfr_percent_median", "Median best BFR [%]"),
        ("fail_rate_percent_mean", "Mean failure rate [%]"),
        ("rankX_fraction_median", "Median rankX fraction"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(-1)

    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in window_stats.columns:
            ax.axis("off")
            continue

        for model_id in MODEL_IDS:
            d = window_stats[window_stats["model_id"] == model_id]
            ax.plot(
                d["window"],
                d[metric],
                marker="o",
                linewidth=2.4,
                label=model_id,
            )

        ax.grid(True, alpha=0.35)
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="best")

    fig.suptitle("S7_C4K vs S17_C4K: region-level summary", fontsize=16)
    savefig(FIG_DIR / "08_region_summary_readable.png")


# ============================================================
# Parameter diagnostics
# ============================================================
def plot_parameter_dashboard(best: pd.DataFrame, cols: list[str], title: str, filename: str, logy: bool = False) -> None:
    available = [c for c in cols if c in best.columns]
    if not available:
        print(f"[skip] no parameter columns for {filename}")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, col in zip(axes, available):
        for model_id in MODEL_IDS:
            d = best[best["model_id"] == model_id].sort_values("cycle_index")
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
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.legend(loc="best", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title + (" [log]" if logy else " [linear]"), fontsize=16)
    savefig(FIG_DIR / filename)


def make_parameter_plots(best: pd.DataFrame) -> None:
    core_cols = [
        "alpha_n_hat",
        "alpha_p_hat",
        "g_n_hat",
        "g_p_hat",
        "b_en_hat",
        "b_ep_hat",
    ]

    k_cols = [
        "k1_hat",
        "k2_hat",
        "k3_hat",
        "k4_hat",
        "k5_hat",
    ]

    k_edge_cols = [
        "k_edge1_hat",
        "k_edge2_hat",
        "k_edge3_hat",
        "k_edge4_hat",
        "k_edge5_hat",
        "k_edge6_hat",
        "k_edge7_hat",
        "k_edge8_hat",
    ]

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

    plot_parameter_dashboard(
        best,
        core_cols,
        "Core dynamic and input gain parameters",
        "09_core_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        best,
        core_cols,
        "Core dynamic and input gain parameters",
        "10_core_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        best,
        k_cols,
        "Independent electrolyte k parameters",
        "11_independent_k_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        best,
        k_cols,
        "Independent electrolyte k parameters",
        "12_independent_k_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        best,
        k_edge_cols,
        "Expanded electrolyte edge k parameters",
        "13_k_edge_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        best,
        k_edge_cols,
        "Expanded electrolyte edge k parameters",
        "14_k_edge_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        best,
        beta_cols,
        "Voltage beta coefficients",
        "15_beta_parameters_linear.png",
        logy=False,
    )


# ============================================================
# Response plots for latest and worst cycles
# ============================================================
def plot_response_for_row(row: pd.Series, label: str, filename: str) -> None:
    response_csv = find_response_csv_from_best_row(row)
    if response_csv is None:
        print(f"[skip] cannot find response CSV for {label}")
        return

    try:
        df = pd.read_csv(response_csv)
    except Exception as exc:
        print(f"[skip] cannot read response CSV {response_csv}: {exc}")
        return

    needed = ["t_s", "measured_voltage_V", "estimated_voltage_V", "residual_V"]
    if not all(c in df.columns for c in needed):
        print(f"[skip] response CSV missing required columns: {response_csv}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(df["t_s"], df["measured_voltage_V"], linewidth=2.5, label="measured")
    axes[0].plot(df["t_s"], df["estimated_voltage_V"], "--", linewidth=2.2, label="estimated")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("Voltage [V]")
    axes[0].legend(loc="best")
    axes[0].set_title(label)

    axes[1].plot(df["t_s"], 1000.0 * df["residual_V"], linewidth=1.8)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.2)
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Residual [mV]")

    savefig(FIG_DIR / filename)


def make_response_plots(summary: pd.DataFrame, best: pd.DataFrame) -> None:
    for model_id in MODEL_IDS:
        dsum = summary[summary["model_id"] == model_id].sort_values("cycle_index")
        dbest = best[best["model_id"] == model_id].sort_values("cycle_index")

        if len(dsum) == 0 or len(dbest) == 0:
            continue

        # Latest cycle.
        latest_cycle = int(dsum["cycle_index"].max())
        latest_row = dbest[dbest["cycle_index"] == latest_cycle].tail(1)

        if len(latest_row):
            row = latest_row.iloc[0]
            label = f"{model_id}: latest cycle {latest_cycle}, RMSE={float(row['best_rmse']):.6e} V"
            plot_response_for_row(
                row,
                label,
                f"16_{model_id}_latest_cycle_response.png",
            )

        # Worst cycle by best RMSE.
        worst_summary = dsum.sort_values("best_rmse", ascending=False).head(1)
        if len(worst_summary):
            worst_cycle = int(worst_summary.iloc[0]["cycle_index"])
            worst_row = dbest[dbest["cycle_index"] == worst_cycle].tail(1)

            if len(worst_row):
                row = worst_row.iloc[0]
                label = f"{model_id}: worst cycle {worst_cycle}, RMSE={float(row['best_rmse']):.6e} V"
                plot_response_for_row(
                    row,
                    label,
                    f"17_{model_id}_worst_cycle_response.png",
                )


# ============================================================
# Tables and reports
# ============================================================
def make_window_stats(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["window"] = df["cycle_index"].apply(get_window_label)

    group_cols = ["model_id", "window"]

    stats = (
        df.groupby(group_cols)
        .agg(
            cycle_min=("cycle_index", "min"),
            cycle_max=("cycle_index", "max"),
            n_cycles=("cycle_index", "count"),
            best_rmse_mV_mean=("best_rmse_mV", "mean"),
            best_rmse_mV_median=("best_rmse_mV", "median"),
            best_rmse_mV_min=("best_rmse_mV", "min"),
            best_rmse_mV_max=("best_rmse_mV", "max"),
            median_rmse_mV_median=("median_rmse_mV", "median"),
            best_bfr_percent_mean=("best_bfr_percent", "mean"),
            best_bfr_percent_median=("best_bfr_percent", "median"),
            best_bfr_percent_min=("best_bfr_percent", "min"),
            best_r2_percent_mean=("best_r2_percent", "mean"),
            best_r2_percent_median=("best_r2_percent", "median"),
            best_r2_percent_min=("best_r2_percent", "min"),
            fail_rate_percent_mean=("fail_rate_percent", "mean"),
            fail_rate_percent_max=("fail_rate_percent", "max"),
            rankX_fraction_median=("rankX_fraction", "median"),
            rankPhi_fraction_median=("rankPhi_fraction", "median"),
            n_good=("flag_good_fit", "sum"),
            n_acceptable=("flag_acceptable_fit", "sum"),
            n_bad_or_weak=("flag_bad_or_weak_fit", "sum"),
        )
        .reset_index()
    )

    order = {
        "0-4 smoke test": 0,
        "5-33 difficult region": 1,
        "34-99 later region": 2,
        "outside expected": 3,
    }
    stats["window_order"] = stats["window"].map(order).fillna(99)
    stats = stats.sort_values(["model_id", "window_order"]).drop(columns=["window_order"])

    return stats


def make_suspicious_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    suspicious = summary[
        summary[
            [
                "flag_bad_or_weak_fit",
                "flag_high_best_rmse",
                "flag_very_high_best_rmse",
                "flag_low_bfr",
                "flag_low_r2",
                "flag_rankX_collapsed",
                "flag_rankX_weak",
                "flag_rankPhi_deficient",
            ]
        ].any(axis=1)
    ].copy()

    cols = [
        "model_id",
        "cycle_index",
        "fit_class",
        "best_seed",
        "best_rmse",
        "best_rmse_mV",
        "median_rmse_mV",
        "median_to_best_rmse_ratio",
        "best_bfr_percent",
        "best_r2_percent",
        "n_success",
        "n_fail",
        "fail_rate_percent",
        "best_rank_X_raw",
        "best_ncols_X_raw",
        "rankX_fraction",
        "best_rank_phi_raw",
        "best_ncols_phi_raw",
        "rankPhi_fraction",
        "flag_high_best_rmse",
        "flag_low_bfr",
        "flag_rankX_collapsed",
        "flag_rankX_weak",
        "flag_rankPhi_deficient",
    ]
    cols = [c for c in cols if c in suspicious.columns]

    suspicious = suspicious[cols].sort_values(["model_id", "cycle_index"])

    worst = summary.sort_values("best_rmse", ascending=False).copy()
    worst_cols = [
        "model_id",
        "cycle_index",
        "fit_class",
        "best_seed",
        "best_rmse",
        "best_rmse_mV",
        "median_rmse_mV",
        "best_bfr_percent",
        "best_r2_percent",
        "best_rank_X_raw",
        "best_ncols_X_raw",
        "best_rank_phi_raw",
        "best_ncols_phi_raw",
    ]
    worst_cols = [c for c in worst_cols if c in worst.columns]
    worst = worst[worst_cols].head(30)

    return suspicious, worst


def write_report(summary: pd.DataFrame, best: pd.DataFrame, window_stats: pd.DataFrame, suspicious: pd.DataFrame, worst: pd.DataFrame) -> None:
    report = TAB_DIR / "final_diagnostic_report.md"

    lines = []
    lines.append("# Final diagnostics for S7_C4K and S17_C4K")
    lines.append("")
    lines.append("This report is generated from the completed combined tables, not from live partial chunk scans.")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")

    for model_id in MODEL_IDS:
        d = summary[summary["model_id"] == model_id]
        found = set(d["cycle_index"].astype(int))
        missing = sorted(EXPECTED_CYCLES - found)

        lines.append(f"### {model_id}")
        lines.append("")
        lines.append(f"- Cycles found: {len(found)}")
        lines.append(f"- Min cycle: {int(d['cycle_index'].min()) if len(d) else 'NA'}")
        lines.append(f"- Max cycle: {int(d['cycle_index'].max()) if len(d) else 'NA'}")
        lines.append(f"- Missing cycles: {missing}")
        lines.append("")

    lines.append("## Window statistics")
    lines.append("")
    lines.append(window_stats.to_markdown(index=False))
    lines.append("")

    lines.append("## Worst cycles by best RMSE")
    lines.append("")
    lines.append(worst.to_markdown(index=False))
    lines.append("")

    lines.append("## Suspicious cycles")
    lines.append("")
    if len(suspicious):
        lines.append(suspicious.to_markdown(index=False))
    else:
        lines.append("No suspicious cycles detected by the current thresholds.")
    lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `best_rmse_mV` is the voltage-fit RMSE in millivolts.")
    lines.append("- `median_rmse_mV` tells whether most local starts worked or only the best one worked.")
    lines.append("- `median_to_best_rmse_ratio` much larger than 1 means the optimization is sensitive to initialization.")
    lines.append("- `rankX_fraction` near 1 is better; very low values mean the fitted states collapsed into a low-dimensional trajectory.")
    lines.append("- `rankPhi_fraction` near 1 means the voltage feature matrix is better excited.")
    lines.append("- A cycle can have low voltage RMSE but weak rank diagnostics. That means the voltage fit is good, but parameter interpretation should be cautious.")
    lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 100)
    print("FINAL DIAGNOSTICS: S7_C4K and S17_C4K")
    print("=" * 100)

    summaries = []
    bests = []

    for model_id in MODEL_IDS:
        summary, best = load_model_tables(model_id)
        summaries.append(summary)
        bests.append(best)

        found = set(summary["cycle_index"].astype(int))
        missing = sorted(EXPECTED_CYCLES - found)

        print()
        print(model_id)
        print("  rows:", len(summary))
        print("  min cycle:", summary["cycle_index"].min())
        print("  max cycle:", summary["cycle_index"].max())
        print("  missing:", missing)

    summary_all = pd.concat(summaries, ignore_index=True, sort=False)
    best_all = pd.concat(bests, ignore_index=True, sort=False)

    summary_all = standardize_best_columns(summary_all)
    best_all = standardize_best_columns(best_all)

    summary_all = add_diagnostics(summary_all)

    # Add diagnostic columns to best table if possible.
    best_all["cycle_index"] = best_all["cycle_index"].astype(int)

    # Save clean combined data.
    summary_all.to_csv(TAB_DIR / "final_s7_s17_summary_with_diagnostics.csv", index=False)
    best_all.to_csv(TAB_DIR / "final_s7_s17_best_runs.csv", index=False)

    # Tables.
    window_stats = make_window_stats(summary_all)
    suspicious, worst = make_suspicious_tables(summary_all)

    window_stats.to_csv(TAB_DIR / "window_stats.csv", index=False)
    suspicious.to_csv(TAB_DIR / "suspicious_cycles.csv", index=False)
    worst.to_csv(TAB_DIR / "worst_cycles_by_best_rmse.csv", index=False)

    # Plots.
    plot_rmse_linear(summary_all)
    plot_rmse_clipped(summary_all)
    plot_rmse_quality_scatter(summary_all)
    plot_bfr_r2(summary_all)
    plot_best_vs_median(summary_all)
    plot_success_failure(summary_all)
    plot_rank_diagnostics(summary_all)
    plot_region_summary_bars(window_stats)
    make_parameter_plots(best_all)
    make_response_plots(summary_all, best_all)

    # Report.
    write_report(summary_all, best_all, window_stats, suspicious, worst)

    print()
    print("=" * 100)
    print("FINAL DIAGNOSTICS COMPLETE")
    print("=" * 100)
    print("Figures saved to:")
    print(FIG_DIR)
    print()
    print("Tables saved to:")
    print(TAB_DIR)
    print()
    print("Most useful files:")
    print("  ", FIG_DIR / "01_best_rmse_mV_linear_full.png")
    print("  ", FIG_DIR / "02_best_rmse_mV_linear_clipped.png")
    print("  ", FIG_DIR / "05_best_vs_median_rmse_stability.png")
    print("  ", FIG_DIR / "07_rank_diagnostics_readable.png")
    print("  ", TAB_DIR / "final_diagnostic_report.md")
    print("=" * 100)


if __name__ == "__main__":
    main()