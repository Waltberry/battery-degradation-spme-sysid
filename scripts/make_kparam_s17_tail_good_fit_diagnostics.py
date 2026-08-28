#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_kparam_s17_tail_good_fit_diagnostics.py

Tail-only diagnostics for S17_C4K direct-K/B warm-continuation runs.

Purpose
-------
This script analyzes only the new tail region, default cycles 100 to the
last available cycle in the completed S17_C4K combined table.

It does NOT include cycles 0--99 unless you explicitly set TAIL_START_CYCLE=99.

Inputs
------
results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_summary.csv
results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_best_runs.csv

Outputs
-------
results/figures/kparam_s17_tail_good_fit_diagnostics/
results/tables/kparam_s17_tail_good_fit_diagnostics/

Default good-fit rule
---------------------
    best_rmse <= 0.002 V
    best_bfr_percent >= 98 %
    best_r2_percent >= 99.95 %

Because the discharge tail may be ugly, the script saves both:
    1. all tail cycles
    2. good tail cycles only
    3. excluded weak/bad tail cycles
"""

from __future__ import annotations

import os
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

MODEL_ID = "S17_C4K"

TABLE_ROOT = PROJECT_DIR / "results/tables/real_warm_continuation_ctid" / MODEL_ID

FIG_DIR = PROJECT_DIR / "results/figures/kparam_s17_tail_good_fit_diagnostics"
TAB_DIR = PROJECT_DIR / "results/tables/kparam_s17_tail_good_fit_diagnostics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

TAIL_START_CYCLE = int(os.environ.get("TAIL_START_CYCLE", "100"))
TAIL_END_CYCLE_ENV = os.environ.get("TAIL_END_CYCLE", "").strip()

GOOD_RMSE_V = float(os.environ.get("GOOD_RMSE_V", "0.002"))
GOOD_BFR_PERCENT = float(os.environ.get("GOOD_BFR_PERCENT", "98.0"))
GOOD_R2_PERCENT = float(os.environ.get("GOOD_R2_PERCENT", "99.95"))

REQUIRE_RANK_FILTER = os.environ.get("REQUIRE_RANK_FILTER", "False").lower() in ["1", "true", "yes", "y"]
MIN_RANKX_FRACTION = float(os.environ.get("MIN_RANKX_FRACTION", "0.50"))
MIN_RANKPHI_FRACTION = float(os.environ.get("MIN_RANKPHI_FRACTION", "0.90"))


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


def read_csv(path: Path) -> pd.DataFrame:
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


def simple_text_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or len(df) == 0:
        return "(empty)"

    d = df.copy()

    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows).copy()

    return d.to_string(index=False)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_csv = TABLE_ROOT / "all_cycles_summary.csv"
    best_csv = TABLE_ROOT / "all_cycles_best_runs.csv"

    summary = standardize_best_columns(read_csv(summary_csv))
    best = standardize_best_columns(read_csv(best_csv))

    summary["model_id"] = MODEL_ID
    best["model_id"] = MODEL_ID

    summary["cycle_index"] = summary["cycle_index"].astype(int)
    best["cycle_index"] = best["cycle_index"].astype(int)

    summary = summary.sort_values("cycle_index").reset_index(drop=True)
    best = best.sort_values("cycle_index").reset_index(drop=True)

    return summary, best


def add_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()

    df["best_rmse_mV"] = 1000.0 * df["best_rmse"]
    df["median_rmse_mV"] = 1000.0 * df["median_rmse"]

    if "best_mae" in df.columns:
        df["best_mae_mV"] = 1000.0 * df["best_mae"]

    if "n_success" in df.columns and "n_fail" in df.columns:
        total = df["n_success"] + df["n_fail"]
        df["fail_rate"] = df["n_fail"] / total.replace(0, np.nan)
        df["fail_rate_percent"] = 100.0 * df["fail_rate"]
    else:
        df["fail_rate"] = np.nan
        df["fail_rate_percent"] = np.nan

    df["median_to_best_rmse_ratio"] = safe_ratio(df["median_rmse"], df["best_rmse"])

    if "best_rank_X_raw" in df.columns and "best_ncols_X_raw" in df.columns:
        df["rankX_fraction"] = df["best_rank_X_raw"] / df["best_ncols_X_raw"].replace(0, np.nan)
    else:
        df["rankX_fraction"] = np.nan

    if "best_rank_phi_raw" in df.columns and "best_ncols_phi_raw" in df.columns:
        df["rankPhi_fraction"] = df["best_rank_phi_raw"] / df["best_ncols_phi_raw"].replace(0, np.nan)
    else:
        df["rankPhi_fraction"] = np.nan

    voltage_good = (
        (df["best_rmse"] <= GOOD_RMSE_V)
        & (df["best_bfr_percent"] >= GOOD_BFR_PERCENT)
        & (df["best_r2_percent"] >= GOOD_R2_PERCENT)
    )

    if REQUIRE_RANK_FILTER:
        rank_good = (
            (df["rankX_fraction"] >= MIN_RANKX_FRACTION)
            & (df["rankPhi_fraction"] >= MIN_RANKPHI_FRACTION)
        )
    else:
        rank_good = pd.Series(True, index=df.index)

    df["is_good_voltage_fit"] = voltage_good
    df["passes_rank_filter"] = rank_good
    df["is_good_fit_for_analysis"] = voltage_good & rank_good

    df["excluded_reason"] = ""

    df.loc[df["best_rmse"] > GOOD_RMSE_V, "excluded_reason"] += "high_rmse;"
    df.loc[df["best_bfr_percent"] < GOOD_BFR_PERCENT, "excluded_reason"] += "low_bfr;"
    df.loc[df["best_r2_percent"] < GOOD_R2_PERCENT, "excluded_reason"] += "low_r2;"

    if REQUIRE_RANK_FILTER:
        df.loc[df["rankX_fraction"] < MIN_RANKX_FRACTION, "excluded_reason"] += "weak_rankX;"
        df.loc[df["rankPhi_fraction"] < MIN_RANKPHI_FRACTION, "excluded_reason"] += "weak_rankPhi;"

    df.loc[df["excluded_reason"].eq(""), "excluded_reason"] = "kept"

    return df


def merge_summary_best(summary_subset: pd.DataFrame, best_all: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "cycle_index"]
    keep = summary_subset[keys].drop_duplicates()
    out = best_all.merge(keep, on=keys, how="inner")
    out = out.sort_values(["cycle_index"]).reset_index(drop=True)
    return out


# ============================================================
# Plots
# ============================================================
def plot_tail_rmse(tail: pd.DataFrame, good: pd.DataFrame, excluded: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 6.5))

    plt.plot(
        tail["cycle_index"],
        tail["best_rmse_mV"],
        marker="o",
        linewidth=1.8,
        label="all tail cycles",
    )

    if len(good):
        plt.scatter(
            good["cycle_index"],
            good["best_rmse_mV"],
            s=80,
            marker="s",
            label="good cycles",
        )

    if len(excluded):
        plt.scatter(
            excluded["cycle_index"],
            excluded["best_rmse_mV"],
            s=60,
            marker="x",
            label="excluded cycles",
        )

    plt.axhline(1000.0 * GOOD_RMSE_V, linestyle=":", linewidth=1.5, label=f"{1000*GOOD_RMSE_V:g} mV threshold")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [mV]")
    plt.title(f"{MODEL_ID}: tail region RMSE, cycles {tail['cycle_index'].min()}--{tail['cycle_index'].max()}")
    plt.legend(loc="best")
    savefig(FIG_DIR / "01_tail_rmse_mV_all_good_excluded.png")


def plot_tail_bfr_r2(tail: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(
        tail["cycle_index"],
        tail["best_bfr_percent"],
        marker="o",
        linewidth=2.0,
    )
    axes[0].axhline(GOOD_BFR_PERCENT, linestyle=":", linewidth=1.5, label=f"{GOOD_BFR_PERCENT:g}% threshold")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("BFR [%]")
    axes[0].set_title("Tail region BFR")
    axes[0].legend(loc="best")

    axes[1].plot(
        tail["cycle_index"],
        tail["best_r2_percent"],
        marker="o",
        linewidth=2.0,
    )
    axes[1].axhline(GOOD_R2_PERCENT, linestyle=":", linewidth=1.5, label=f"{GOOD_R2_PERCENT:g}% threshold")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("R2 [%]")
    axes[1].set_title("Tail region R2")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "02_tail_bfr_r2.png")


def plot_tail_multistart_stability(tail: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(
        tail["cycle_index"],
        tail["best_rmse_mV"],
        marker="o",
        linewidth=2.0,
        label="best RMSE",
    )
    axes[0].plot(
        tail["cycle_index"],
        tail["median_rmse_mV"],
        marker="s",
        linewidth=1.8,
        label="median RMSE",
    )
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("RMSE [mV]")
    axes[0].set_title("Tail region: best vs median multistart RMSE")
    axes[0].legend(loc="best")

    axes[1].plot(
        tail["cycle_index"],
        tail["median_to_best_rmse_ratio"],
        marker="o",
        linewidth=2.0,
    )
    axes[1].axhline(2.0, linestyle=":", linewidth=1.5, label="2x")
    axes[1].axhline(10.0, linestyle=":", linewidth=1.5, label="10x")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("Median / best RMSE")
    axes[1].set_title("Tail region: multistart stability")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "03_tail_multistart_stability.png")


def plot_tail_rank(tail: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(
        tail["cycle_index"],
        tail["rankX_fraction"],
        marker="o",
        linewidth=2.0,
    )
    axes[0].axhline(MIN_RANKX_FRACTION, linestyle=":", linewidth=1.5, label="rankX reference")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("rank(X) / n_x")
    axes[0].set_title("Tail region: state trajectory rank fraction")
    axes[0].legend(loc="best")

    axes[1].plot(
        tail["cycle_index"],
        tail["rankPhi_fraction"],
        marker="o",
        linewidth=2.0,
    )
    axes[1].axhline(MIN_RANKPHI_FRACTION, linestyle=":", linewidth=1.5, label="rankPhi reference")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("rank(Phi) / n_cols")
    axes[1].set_title("Tail region: output feature rank fraction")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "04_tail_rank_fractions.png")


def plot_parameter_dashboard(df: pd.DataFrame, cols: list[str], title: str, filename: str, logy: bool = False) -> None:
    available = [c for c in cols if c in df.columns]

    if not available:
        print(f"[skip] no available parameter columns for {filename}")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, col in zip(axes, available):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if logy:
            y = np.where(y > 0, y, np.nan)
            ax.semilogy(
                df["cycle_index"],
                y,
                marker="o",
                linewidth=2.0,
            )
        else:
            ax.plot(
                df["cycle_index"],
                y,
                marker="o",
                linewidth=2.0,
            )

        ax.grid(True, which="both" if logy else "major", alpha=0.35)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel(col)
        ax.set_title(col)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title + (" [log]" if logy else " [linear]"), fontsize=16)
    savefig(FIG_DIR / filename)


def make_parameter_plots(tail_best: pd.DataFrame, good_best: pd.DataFrame) -> None:
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

    # All tail parameters.
    plot_parameter_dashboard(
        tail_best,
        core_cols,
        "S17 tail: all tail cycles core parameters",
        "05_tail_all_core_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        tail_best,
        core_cols,
        "S17 tail: all tail cycles core parameters",
        "06_tail_all_core_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        tail_best,
        k_cols,
        "S17 tail: all tail cycles independent k parameters",
        "07_tail_all_k_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        tail_best,
        k_cols,
        "S17 tail: all tail cycles independent k parameters",
        "08_tail_all_k_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        tail_best,
        k_edge_cols,
        "S17 tail: all tail cycles expanded k-edge parameters",
        "09_tail_all_k_edge_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        tail_best,
        k_edge_cols,
        "S17 tail: all tail cycles expanded k-edge parameters",
        "10_tail_all_k_edge_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        tail_best,
        beta_cols,
        "S17 tail: all tail cycles beta coefficients",
        "11_tail_all_beta_coefficients_linear.png",
        logy=False,
    )

    # Good-only tail parameters.
    if len(good_best) == 0:
        print("[skip] no good tail cycles for good-only parameter plots")
        return

    plot_parameter_dashboard(
        good_best,
        core_cols,
        "S17 tail: good cycles only core parameters",
        "12_tail_good_core_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        good_best,
        core_cols,
        "S17 tail: good cycles only core parameters",
        "13_tail_good_core_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        good_best,
        k_cols,
        "S17 tail: good cycles only independent k parameters",
        "14_tail_good_k_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        good_best,
        k_cols,
        "S17 tail: good cycles only independent k parameters",
        "15_tail_good_k_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        good_best,
        k_edge_cols,
        "S17 tail: good cycles only expanded k-edge parameters",
        "16_tail_good_k_edge_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        good_best,
        k_edge_cols,
        "S17 tail: good cycles only expanded k-edge parameters",
        "17_tail_good_k_edge_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        good_best,
        beta_cols,
        "S17 tail: good cycles only beta coefficients",
        "18_tail_good_beta_coefficients_linear.png",
        logy=False,
    )


# ============================================================
# Report
# ============================================================
def make_report(
    tail: pd.DataFrame,
    good: pd.DataFrame,
    excluded: pd.DataFrame,
) -> None:
    report_path = TAB_DIR / "s17_tail_good_fit_report.txt"

    lines = []
    lines.append("=" * 100)
    lines.append("S17_C4K TAIL-ONLY GOOD-FIT DIAGNOSTIC REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Tail cycle range analyzed: {int(tail['cycle_index'].min())} to {int(tail['cycle_index'].max())}")
    lines.append("")
    lines.append("Good-fit rule:")
    lines.append(f"  best_rmse <= {GOOD_RMSE_V:.6g} V ({1000*GOOD_RMSE_V:.3g} mV)")
    lines.append(f"  best_bfr_percent >= {GOOD_BFR_PERCENT:.3g} %")
    lines.append(f"  best_r2_percent >= {GOOD_R2_PERCENT:.5g} %")
    lines.append(f"  REQUIRE_RANK_FILTER = {REQUIRE_RANK_FILTER}")
    lines.append("")
    lines.append(f"Total tail cycles: {len(tail)}")
    lines.append(f"Good cycles kept: {len(good)}")
    lines.append(f"Excluded cycles: {len(excluded)}")
    lines.append("")

    if len(good):
        lines.append("Good cycle statistics:")
        cols_stats = [
            "best_rmse_mV",
            "median_rmse_mV",
            "best_bfr_percent",
            "best_r2_percent",
            "median_to_best_rmse_ratio",
            "rankX_fraction",
            "rankPhi_fraction",
        ]
        lines.append(simple_text_table(good[cols_stats].describe().reset_index()))
        lines.append("")

    lines.append("=" * 100)
    lines.append("Good tail cycles kept")
    lines.append("=" * 100)
    cols_good = [
        "cycle_index",
        "best_seed",
        "best_rmse_mV",
        "median_rmse_mV",
        "best_bfr_percent",
        "best_r2_percent",
        "median_to_best_rmse_ratio",
        "rankX_fraction",
        "rankPhi_fraction",
    ]
    cols_good = [c for c in cols_good if c in good.columns]
    lines.append(simple_text_table(good[cols_good]))
    lines.append("")

    lines.append("=" * 100)
    lines.append("Excluded tail cycles")
    lines.append("=" * 100)
    cols_excluded = [
        "cycle_index",
        "best_seed",
        "best_rmse_mV",
        "median_rmse_mV",
        "best_bfr_percent",
        "best_r2_percent",
        "rankX_fraction",
        "rankPhi_fraction",
        "excluded_reason",
    ]
    cols_excluded = [c for c in cols_excluded if c in excluded.columns]
    lines.append(simple_text_table(excluded[cols_excluded]))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 100)
    print("S17_C4K TAIL-ONLY GOOD-FIT DIAGNOSTICS")
    print("=" * 100)

    summary_all, best_all = load_tables()

    summary_all = add_diagnostics(summary_all)

    max_available = int(summary_all["cycle_index"].max())

    if TAIL_END_CYCLE_ENV:
        tail_end = int(TAIL_END_CYCLE_ENV)
    else:
        tail_end = max_available

    tail = summary_all[
        (summary_all["cycle_index"] >= TAIL_START_CYCLE)
        & (summary_all["cycle_index"] <= tail_end)
    ].copy()

    if len(tail) == 0:
        raise RuntimeError(
            f"No S17_C4K cycles found in requested tail range "
            f"{TAIL_START_CYCLE} to {tail_end}. "
            f"Current max available cycle is {max_available}."
        )

    good = tail[tail["is_good_fit_for_analysis"]].copy()
    excluded = tail[~tail["is_good_fit_for_analysis"]].copy()

    tail = tail.sort_values("cycle_index").reset_index(drop=True)
    good = good.sort_values("cycle_index").reset_index(drop=True)
    excluded = excluded.sort_values("cycle_index").reset_index(drop=True)

    tail_best = merge_summary_best(tail, best_all)
    good_best = merge_summary_best(good, best_all)

    # Save tables.
    tail.to_csv(TAB_DIR / "s17_tail_all_cycles_with_flags.csv", index=False)
    good.to_csv(TAB_DIR / "s17_tail_good_cycles_only_summary.csv", index=False)
    excluded.to_csv(TAB_DIR / "s17_tail_excluded_bad_or_weak_cycles.csv", index=False)
    tail_best.to_csv(TAB_DIR / "s17_tail_all_cycles_best_runs.csv", index=False)
    good_best.to_csv(TAB_DIR / "s17_tail_good_cycles_only_best_runs.csv", index=False)

    # Plots.
    plot_tail_rmse(tail, good, excluded)
    plot_tail_bfr_r2(tail)
    plot_tail_multistart_stability(tail)
    plot_tail_rank(tail)
    make_parameter_plots(tail_best, good_best)

    # Report.
    make_report(tail, good, excluded)

    print()
    print("=" * 100)
    print("S17_C4K TAIL-ONLY GOOD-FIT DIAGNOSTICS COMPLETE")
    print("=" * 100)
    print(f"Tail range analyzed: {int(tail['cycle_index'].min())} to {int(tail['cycle_index'].max())}")
    print(f"Total tail cycles: {len(tail)}")
    print(f"Good cycles kept: {len(good)}")
    print(f"Excluded cycles: {len(excluded)}")
    print()
    print("Figures saved to:")
    print(FIG_DIR)
    print()
    print("Tables saved to:")
    print(TAB_DIR)
    print()
    print("Most useful files:")
    print("  ", TAB_DIR / "s17_tail_good_fit_report.txt")
    print("  ", TAB_DIR / "s17_tail_good_cycles_only_summary.csv")
    print("  ", TAB_DIR / "s17_tail_excluded_bad_or_weak_cycles.csv")
    print("  ", FIG_DIR / "01_tail_rmse_mV_all_good_excluded.png")
    print("  ", FIG_DIR / "03_tail_multistart_stability.png")
    print("=" * 100)


if __name__ == "__main__":
    main()