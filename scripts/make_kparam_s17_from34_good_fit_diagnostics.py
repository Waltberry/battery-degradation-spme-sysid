#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_kparam_s17_from34_good_fit_diagnostics.py

S17-only good-fit diagnostics starting from cycle 34.

Purpose
-------
This script removes S7 completely and analyzes only S17_C4K cycles
from cycles 34-100, because cycle 34 is where the S17 fit becomes
more continuous and more useful for parameter trend interpretation.

Inputs
------
results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_summary.csv
results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_best_runs.csv

Outputs
-------
Figures:
results/figures/kparam_s17_from34_good_fit_diagnostics/

Tables:
results/tables/kparam_s17_from34_good_fit_diagnostics/
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

MODEL_ID = "S17_C4K"

TABLE_ROOT = PROJECT_DIR / "results/tables/real_warm_continuation_ctid" / MODEL_ID

FIG_DIR = PROJECT_DIR / "results/figures/kparam_s17_from34_good_fit_diagnostics"
TAB_DIR = PROJECT_DIR / "results/tables/kparam_s17_from34_good_fit_diagnostics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Start where S17 becomes more continuous.
START_CYCLE = 34

# Optional end cycle. Set to None to use all available cycles from START_CYCLE onward.
END_CYCLE = 100

# Good voltage-fit thresholds.
GOOD_RMSE_V = 0.002
GOOD_BFR_PERCENT = 98.0
GOOD_R2_PERCENT = 99.95

# Keep rank as a diagnostic, not a filter.
REQUIRE_RANK_FILTER = False
MIN_RANKX_FRACTION = 0.50
MIN_RANKPHI_FRACTION = 0.90


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


def cycle_region(cycle: int) -> str:
    if 34 <= cycle <= 100:
        return "34-100 continuous S17"
    if cycle > 100:
        return "100+ tail region"
    return "before 34 excluded"


def load_s17_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def filter_cycle_range(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["cycle_index"] >= START_CYCLE].copy()

    if END_CYCLE is not None:
        out = out[out["cycle_index"] <= END_CYCLE].copy()

    out = out.sort_values("cycle_index").reset_index(drop=True)
    return out


def add_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()

    df["cycle_index"] = df["cycle_index"].astype(int)

    df["best_rmse_mV"] = 1000.0 * df["best_rmse"]

    if "median_rmse" in df.columns:
        df["median_rmse_mV"] = 1000.0 * df["median_rmse"]
        df["median_to_best_rmse_ratio"] = safe_ratio(df["median_rmse"], df["best_rmse"])
    else:
        df["median_rmse_mV"] = np.nan
        df["median_to_best_rmse_ratio"] = np.nan

    if "best_mae" in df.columns:
        df["best_mae_mV"] = 1000.0 * df["best_mae"]
    else:
        df["best_mae_mV"] = np.nan

    if "n_success" in df.columns and "n_fail" in df.columns:
        total = df["n_success"] + df["n_fail"]
        df["fail_rate"] = df["n_fail"] / total.replace(0, np.nan)
        df["fail_rate_percent"] = 100.0 * df["fail_rate"]
    else:
        df["fail_rate"] = np.nan
        df["fail_rate_percent"] = np.nan

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

    df["region"] = df["cycle_index"].apply(cycle_region)

    return df


def merge_summary_best(good_summary: pd.DataFrame, best_all: pd.DataFrame) -> pd.DataFrame:
    keep_keys = ["model_id", "cycle_index"]
    good_keys = good_summary[keep_keys].drop_duplicates()

    out = best_all.merge(good_keys, on=keep_keys, how="inner")
    out = out.sort_values(["cycle_index"]).reset_index(drop=True)

    return out


# ============================================================
# Plotting
# ============================================================
def plot_good_mask(all_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 4.8))

    kept = all_summary[all_summary["is_good_fit_for_analysis"]]
    excluded = all_summary[~all_summary["is_good_fit_for_analysis"]]

    plt.scatter(
        kept["cycle_index"],
        np.ones(len(kept)),
        s=70,
        marker="s",
        label="kept good S17 cycles",
    )

    plt.scatter(
        excluded["cycle_index"],
        np.ones(len(excluded)),
        s=50,
        marker="x",
        label="excluded S17 cycles",
    )

    plt.grid(True, alpha=0.35)
    plt.yticks([1], [MODEL_ID])
    plt.xlabel("Cycle index")
    plt.title(f"{MODEL_ID}: good-fit mask from cycles {START_CYCLE}-{END_CYCLE}")
    plt.legend(loc="best")
    savefig(FIG_DIR / "01_s17_from34_good_fit_mask.png")


def plot_rmse(all_summary: pd.DataFrame, good_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6.2))

    plt.plot(
        all_summary["cycle_index"],
        all_summary["best_rmse_mV"],
        marker="o",
        linewidth=1.8,
        label="all S17 cycles from 34",
    )

    if len(good_summary):
        plt.scatter(
            good_summary["cycle_index"],
            good_summary["best_rmse_mV"],
            s=80,
            marker="s",
            label="kept good cycles",
        )

    plt.axhline(1000.0 * GOOD_RMSE_V, linestyle=":", linewidth=1.5, label="2 mV threshold")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [mV]")
    plt.title(f"{MODEL_ID}: best RMSE from cycles {START_CYCLE}-{END_CYCLE}")
    plt.legend(loc="best")
    savefig(FIG_DIR / "02_s17_from34_best_rmse_mV.png")


def plot_bfr_r2(all_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(
        all_summary["cycle_index"],
        all_summary["best_bfr_percent"],
        marker="o",
        linewidth=2.0,
    )
    axes[0].axhline(GOOD_BFR_PERCENT, linestyle=":", linewidth=1.5, label="98% threshold")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("BFR [%]")
    axes[0].set_title(f"{MODEL_ID}: BFR from cycles {START_CYCLE}-{END_CYCLE}")
    axes[0].legend(loc="best")

    axes[1].plot(
        all_summary["cycle_index"],
        all_summary["best_r2_percent"],
        marker="o",
        linewidth=2.0,
    )
    axes[1].axhline(GOOD_R2_PERCENT, linestyle=":", linewidth=1.5, label="99.95% threshold")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("R2 [%]")
    axes[1].set_title(f"{MODEL_ID}: R2 from cycles {START_CYCLE}-{END_CYCLE}")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "03_s17_from34_bfr_r2.png")


def plot_rank(all_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(
        all_summary["cycle_index"],
        all_summary["rankX_fraction"],
        marker="o",
        linewidth=2.0,
    )
    axes[0].axhline(MIN_RANKX_FRACTION, linestyle=":", linewidth=1.5, label="rankX reference")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("rank(X) / n_x")
    axes[0].set_title(f"{MODEL_ID}: state trajectory rank fraction")
    axes[0].legend(loc="best")

    axes[1].plot(
        all_summary["cycle_index"],
        all_summary["rankPhi_fraction"],
        marker="o",
        linewidth=2.0,
    )
    axes[1].axhline(MIN_RANKPHI_FRACTION, linestyle=":", linewidth=1.5, label="rankPhi reference")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("rank(Phi) / n_cols")
    axes[1].set_title(f"{MODEL_ID}: feature matrix rank fraction")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "04_s17_from34_rank_fractions.png")


def plot_parameter_dashboard(
    best_df: pd.DataFrame,
    cols: list[str],
    title: str,
    filename: str,
    logy: bool = False,
) -> None:
    available = [c for c in cols if c in best_df.columns]

    if not available:
        print(f"[skip] no available columns for {filename}")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, col in zip(axes, available):
        d = best_df.sort_values("cycle_index")
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)

        if logy:
            y = np.where(y > 0, y, np.nan)
            ax.semilogy(d["cycle_index"], y, marker="o", linewidth=2.0)
        else:
            ax.plot(d["cycle_index"], y, marker="o", linewidth=2.0)

        ax.grid(True, which="both" if logy else "major", alpha=0.35)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel(col)
        ax.set_title(col)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title + (" [log]" if logy else " [linear]"), fontsize=16)
    savefig(FIG_DIR / filename)


def make_parameter_plots(best_df: pd.DataFrame) -> None:
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
        best_df,
        core_cols,
        "S17 cycles 34-100: core dynamic and input gains",
        "05_s17_from34_core_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        best_df,
        core_cols,
        "S17 cycles 34-100: core dynamic and input gains",
        "06_s17_from34_core_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        best_df,
        k_cols,
        "S17 cycles 34-100: independent electrolyte k parameters",
        "07_s17_from34_k_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        best_df,
        k_cols,
        "S17 cycles 34-100: independent electrolyte k parameters",
        "08_s17_from34_k_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        best_df,
        k_edge_cols,
        "S17 cycles 34-100: expanded electrolyte edge couplings",
        "09_s17_from34_k_edge_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        best_df,
        k_edge_cols,
        "S17 cycles 34-100: expanded electrolyte edge couplings",
        "10_s17_from34_k_edge_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        best_df,
        beta_cols,
        "S17 cycles 34-100: voltage beta coefficients",
        "11_s17_from34_beta_coefficients_linear.png",
        logy=False,
    )


def make_region_stats(summary: pd.DataFrame) -> pd.DataFrame:
    stats = (
        summary.groupby("region")
        .agg(
            n_cycles=("cycle_index", "count"),
            n_good=("is_good_fit_for_analysis", "sum"),
            cycle_min=("cycle_index", "min"),
            cycle_max=("cycle_index", "max"),
            best_rmse_mV_mean=("best_rmse_mV", "mean"),
            best_rmse_mV_median=("best_rmse_mV", "median"),
            best_rmse_mV_max=("best_rmse_mV", "max"),
            best_bfr_percent_mean=("best_bfr_percent", "mean"),
            best_bfr_percent_median=("best_bfr_percent", "median"),
            best_r2_percent_mean=("best_r2_percent", "mean"),
            best_r2_percent_median=("best_r2_percent", "median"),
            fail_rate_percent_mean=("fail_rate_percent", "mean"),
            rankX_fraction_median=("rankX_fraction", "median"),
            rankPhi_fraction_median=("rankPhi_fraction", "median"),
        )
        .reset_index()
    )

    stats["good_percent"] = 100.0 * stats["n_good"] / stats["n_cycles"].replace(0, np.nan)
    return stats


def write_report(
    all_summary: pd.DataFrame,
    good_summary: pd.DataFrame,
    excluded_summary: pd.DataFrame,
    region_stats: pd.DataFrame,
) -> None:
    report_path = TAB_DIR / "s17_from34_good_fit_report.txt"

    lines = []
    lines.append("=" * 100)
    lines.append("S17-ONLY GOOD-FIT DIAGNOSTIC REPORT FROM CYCLE 34")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Model: {MODEL_ID}")
    lines.append(f"Start cycle: {START_CYCLE}")
    lines.append(f"End cycle: {END_CYCLE if END_CYCLE is not None else 'last available'}")
    lines.append("")
    lines.append("Good-fit rule:")
    lines.append(f"  best_rmse <= {GOOD_RMSE_V:.6g} V ({1000 * GOOD_RMSE_V:.3g} mV)")
    lines.append(f"  best_bfr_percent >= {GOOD_BFR_PERCENT:.3g} %")
    lines.append(f"  best_r2_percent >= {GOOD_R2_PERCENT:.5g} %")
    lines.append(f"  REQUIRE_RANK_FILTER = {REQUIRE_RANK_FILTER}")
    lines.append("")

    found = sorted(all_summary["cycle_index"].astype(int).unique())

    lines.append("Coverage:")
    lines.append(f"  cycles found: {len(found)}")
    if len(found):
        lines.append(f"  first cycle found: {found[0]}")
        lines.append(f"  last cycle found: {found[-1]}")
    lines.append(f"  good cycles kept: {len(good_summary)}")
    lines.append(f"  excluded cycles: {len(excluded_summary)}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("Region statistics")
    lines.append("=" * 100)
    lines.append(simple_text_table(region_stats))
    lines.append("")

    lines.append("=" * 100)
    lines.append("Good cycles kept")
    lines.append("=" * 100)
    cols_good = [
        "model_id",
        "cycle_index",
        "region",
        "best_seed",
        "best_rmse_mV",
        "best_bfr_percent",
        "best_r2_percent",
        "median_to_best_rmse_ratio",
        "rankX_fraction",
        "rankPhi_fraction",
    ]
    cols_good = [c for c in cols_good if c in good_summary.columns]
    lines.append(simple_text_table(good_summary[cols_good]))
    lines.append("")

    lines.append("=" * 100)
    lines.append("Excluded cycles")
    lines.append("=" * 100)
    cols_excl = [
        "model_id",
        "cycle_index",
        "region",
        "best_seed",
        "best_rmse_mV",
        "best_bfr_percent",
        "best_r2_percent",
        "rankX_fraction",
        "rankPhi_fraction",
        "excluded_reason",
    ]
    cols_excl = [c for c in cols_excl if c in excluded_summary.columns]
    lines.append(simple_text_table(excluded_summary[cols_excl]))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 100)
    print("S17-ONLY GOOD-FIT DIAGNOSTICS FROM CYCLE 34")
    print("=" * 100)

    summary_all, best_all = load_s17_tables()

    summary_all = filter_cycle_range(summary_all)
    best_all = filter_cycle_range(best_all)

    summary_all = standardize_best_columns(summary_all)
    best_all = standardize_best_columns(best_all)

    summary_all = add_diagnostics(summary_all)

    good_summary = summary_all[summary_all["is_good_fit_for_analysis"]].copy()
    excluded_summary = summary_all[~summary_all["is_good_fit_for_analysis"]].copy()

    good_summary = good_summary.sort_values("cycle_index").reset_index(drop=True)
    excluded_summary = excluded_summary.sort_values("cycle_index").reset_index(drop=True)

    good_best = merge_summary_best(good_summary, best_all)

    # Save tables.
    summary_all.to_csv(TAB_DIR / "s17_from34_all_cycles_with_good_fit_flags.csv", index=False)
    good_summary.to_csv(TAB_DIR / "s17_from34_good_cycles_only_summary.csv", index=False)
    excluded_summary.to_csv(TAB_DIR / "s17_from34_excluded_bad_or_weak_cycles.csv", index=False)
    good_best.to_csv(TAB_DIR / "s17_from34_good_cycles_only_best_runs.csv", index=False)

    region_stats = make_region_stats(summary_all)
    region_stats.to_csv(TAB_DIR / "s17_from34_region_stats.csv", index=False)

    # Plots.
    plot_good_mask(summary_all)
    plot_rmse(summary_all, good_summary)
    plot_bfr_r2(summary_all)
    plot_rank(summary_all)
    make_parameter_plots(good_best)

    # Report.
    write_report(summary_all, good_summary, excluded_summary, region_stats)

    print()
    print("=" * 100)
    print("S17-ONLY GOOD-FIT DIAGNOSTICS COMPLETE")
    print("=" * 100)
    print(f"Cycles analyzed: {len(summary_all)}")
    print(f"Good cycles kept: {len(good_summary)}")
    print(f"Excluded cycles: {len(excluded_summary)}")
    print()
    print("Figures saved to:")
    print(FIG_DIR)
    print()
    print("Tables saved to:")
    print(TAB_DIR)
    print()
    print("Most useful files:")
    print("  ", TAB_DIR / "s17_from34_good_cycles_only_summary.csv")
    print("  ", TAB_DIR / "s17_from34_excluded_bad_or_weak_cycles.csv")
    print("  ", TAB_DIR / "s17_from34_good_fit_report.txt")
    print("  ", FIG_DIR / "02_s17_from34_best_rmse_mV.png")
    print("=" * 100)


if __name__ == "__main__":
    main()