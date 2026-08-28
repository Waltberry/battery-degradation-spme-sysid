#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_kparam_s7_s17_good_fit_diagnostics.py

Good-fit-only diagnostics for completed S7_C4K and S17_C4K direct-K/B runs.

Purpose
-------
This script filters out bad/weak voltage fits and focuses only on cycles
where the model fit is trustworthy enough for parameter trend inspection.

It reads completed combined tables:

    results/tables/real_warm_continuation_ctid/S7_C4K/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/S7_C4K/all_cycles_best_runs.csv

    results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/S17_C4K/all_cycles_best_runs.csv

Outputs
-------
Figures:
    results/figures/kparam_s7_s17_good_fit_diagnostics/

Tables:
    results/tables/kparam_s7_s17_good_fit_diagnostics/

Good-fit rule
-------------
Default good-fit rule:

    best_rmse <= 0.002 V      # 2 mV
    best_bfr_percent >= 98 %
    best_r2_percent >= 99.95 %

Optional rank filtering can be enabled by setting:

    REQUIRE_RANK_FILTER = True

but by default it is False because a model can have excellent voltage fit
even if the fitted state trajectory has weak rank. Rank should be reported,
not automatically used to remove good voltage fits unless you explicitly want that.
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

FIG_DIR = PROJECT_DIR / "results/figures/kparam_s7_s17_good_fit_diagnostics"
TAB_DIR = PROJECT_DIR / "results/tables/kparam_s7_s17_good_fit_diagnostics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Good voltage-fit thresholds.
GOOD_RMSE_V = 0.002
GOOD_BFR_PERCENT = 98.0
GOOD_R2_PERCENT = 99.95

# Optional rank requirement.
# Keep False first. Use True only if you want "good fit + enough state excitation".
REQUIRE_RANK_FILTER = False
MIN_RANKX_FRACTION = 0.50
MIN_RANKPHI_FRACTION = 0.90

EXPECTED_CYCLES = set(range(100))


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


def load_model_tables(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_csv = TABLE_ROOT / model_id / "all_cycles_summary.csv"
    best_csv = TABLE_ROOT / model_id / "all_cycles_best_runs.csv"

    summary = standardize_best_columns(read_csv(summary_csv))
    best = standardize_best_columns(read_csv(best_csv))

    summary["model_id"] = model_id
    best["model_id"] = model_id

    summary["cycle_index"] = summary["cycle_index"].astype(int)
    best["cycle_index"] = best["cycle_index"].astype(int)

    summary = summary.sort_values("cycle_index").reset_index(drop=True)
    best = best.sort_values("cycle_index").reset_index(drop=True)

    return summary, best


def add_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()

    df["cycle_index"] = df["cycle_index"].astype(int)

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


def merge_summary_best(good_summary: pd.DataFrame, best_all: pd.DataFrame) -> pd.DataFrame:
    keep_keys = ["model_id", "cycle_index"]
    good_keys = good_summary[keep_keys].drop_duplicates()

    out = best_all.merge(good_keys, on=keep_keys, how="inner")
    out = out.sort_values(["model_id", "cycle_index"]).reset_index(drop=True)

    return out


def simple_text_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """
    Avoids pandas.to_markdown because cluster environment does not have tabulate.
    """
    if df is None or len(df) == 0:
        return "(empty)"

    d = df.copy()

    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows).copy()

    return d.to_string(index=False)


def window_label(cycle: int) -> str:
    if 0 <= cycle <= 4:
        return "0-4 smoke"
    if 5 <= cycle <= 33:
        return "5-33 difficult"
    if 34 <= cycle <= 99:
        return "34-99 later"
    return "outside"


# ============================================================
# Plotting
# ============================================================
def plot_good_rmse(good: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 6.5))

    for model_id in MODEL_IDS:
        d = good[good["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        plt.plot(
            d["cycle_index"],
            d["best_rmse_mV"],
            marker="o",
            linewidth=2.2,
            label=f"{model_id} good cycles",
        )

    plt.axhline(1000.0 * GOOD_RMSE_V, linestyle=":", linewidth=1.5, label="2 mV threshold")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [mV]")
    plt.title("Good-fit cycles only: best RMSE in mV")
    plt.legend(loc="best")
    savefig(FIG_DIR / "01_good_cycles_best_rmse_mV.png")


def plot_good_cycle_mask(all_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 4.8))

    y_map = {"S7_C4K": 1, "S17_C4K": 2}

    for model_id in MODEL_IDS:
        d = all_summary[all_summary["model_id"] == model_id].sort_values("cycle_index")

        kept = d[d["is_good_fit_for_analysis"]]
        excluded = d[~d["is_good_fit_for_analysis"]]

        plt.scatter(
            kept["cycle_index"],
            [y_map[model_id]] * len(kept),
            s=70,
            marker="s",
            label=f"{model_id} kept",
        )

        plt.scatter(
            excluded["cycle_index"],
            [y_map[model_id]] * len(excluded),
            s=45,
            marker="x",
            label=f"{model_id} excluded",
        )

    plt.yticks([1, 2], ["S7_C4K", "S17_C4K"])
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.title("Good-fit mask: kept vs excluded cycles")
    plt.legend(loc="best", fontsize=8, ncols=2)
    savefig(FIG_DIR / "02_good_fit_mask.png")


def plot_good_bfr_r2(good: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for model_id in MODEL_IDS:
        d = good[good["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        axes[0].plot(
            d["cycle_index"],
            d["best_bfr_percent"],
            marker="o",
            linewidth=2.2,
            label=model_id,
        )

        axes[1].plot(
            d["cycle_index"],
            d["best_r2_percent"],
            marker="o",
            linewidth=2.2,
            label=model_id,
        )

    axes[0].axhline(GOOD_BFR_PERCENT, linestyle=":", linewidth=1.5, label="98% threshold")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("BFR [%]")
    axes[0].set_title("Good-fit cycles only: BFR")
    axes[0].legend(loc="best")

    axes[1].axhline(GOOD_R2_PERCENT, linestyle=":", linewidth=1.5, label="99.95% threshold")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("R2 [%]")
    axes[1].set_title("Good-fit cycles only: R2")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "03_good_cycles_bfr_r2.png")


def plot_good_rank_diagnostics(good: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for model_id in MODEL_IDS:
        d = good[good["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        axes[0].plot(
            d["cycle_index"],
            d["rankX_fraction"],
            marker="o",
            linewidth=2.2,
            label=model_id,
        )

        axes[1].plot(
            d["cycle_index"],
            d["rankPhi_fraction"],
            marker="o",
            linewidth=2.2,
            label=model_id,
        )

    axes[0].axhline(MIN_RANKX_FRACTION, linestyle=":", linewidth=1.5, label="rankX reference")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("rank(X) / n_x")
    axes[0].set_title("Good voltage-fit cycles: state trajectory rank fraction")
    axes[0].legend(loc="best")

    axes[1].axhline(MIN_RANKPHI_FRACTION, linestyle=":", linewidth=1.5, label="rankPhi reference")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel("Cycle index")
    axes[1].set_ylabel("rank(Phi) / n_cols")
    axes[1].set_title("Good voltage-fit cycles: feature matrix rank fraction")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "04_good_cycles_rank_fractions.png")


def plot_parameter_dashboard(
    good_best: pd.DataFrame,
    cols: list[str],
    title: str,
    filename: str,
    logy: bool = False,
) -> None:
    available = [c for c in cols if c in good_best.columns]

    if not available:
        print(f"[skip] no available columns for {filename}")
        return

    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, col in zip(axes, available):
        for model_id in MODEL_IDS:
            d = good_best[good_best["model_id"] == model_id].sort_values("cycle_index")
            if len(d) == 0:
                continue

            y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)

            if logy:
                y = np.where(y > 0, y, np.nan)
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

        ax.grid(True, which="both" if logy else "major", alpha=0.35)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.legend(loc="best", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title + (" [log]" if logy else " [linear]"), fontsize=16)
    savefig(FIG_DIR / filename)


def make_good_parameter_plots(good_best: pd.DataFrame) -> None:
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
        good_best,
        core_cols,
        "Good-fit cycles only: core dynamic and input gains",
        "05_good_core_parameters_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        good_best,
        core_cols,
        "Good-fit cycles only: core dynamic and input gains",
        "06_good_core_parameters_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        good_best,
        k_cols,
        "Good-fit cycles only: independent electrolyte k parameters",
        "07_good_independent_k_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        good_best,
        k_cols,
        "Good-fit cycles only: independent electrolyte k parameters",
        "08_good_independent_k_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        good_best,
        k_edge_cols,
        "Good-fit cycles only: expanded electrolyte edge couplings",
        "09_good_k_edge_linear.png",
        logy=False,
    )

    plot_parameter_dashboard(
        good_best,
        k_edge_cols,
        "Good-fit cycles only: expanded electrolyte edge couplings",
        "10_good_k_edge_log.png",
        logy=True,
    )

    plot_parameter_dashboard(
        good_best,
        beta_cols,
        "Good-fit cycles only: voltage beta coefficients",
        "11_good_beta_coefficients_linear.png",
        logy=False,
    )


def plot_good_counts_by_window(good_summary: pd.DataFrame, excluded_summary: pd.DataFrame) -> None:
    rows = []

    for model_id in MODEL_IDS:
        for label in ["0-4 smoke", "5-33 difficult", "34-99 later"]:
            g = good_summary[
                (good_summary["model_id"] == model_id)
                & (good_summary["window"] == label)
            ]

            e = excluded_summary[
                (excluded_summary["model_id"] == model_id)
                & (excluded_summary["window"] == label)
            ]

            rows.append(
                {
                    "model_id": model_id,
                    "window": label,
                    "good_cycles": int(len(g)),
                    "excluded_cycles": int(len(e)),
                    "total_cycles": int(len(g) + len(e)),
                }
            )

    counts = pd.DataFrame(rows)
    counts["good_fraction"] = counts["good_cycles"] / counts["total_cycles"].replace(0, np.nan)
    counts["good_percent"] = 100.0 * counts["good_fraction"]

    counts.to_csv(TAB_DIR / "good_cycle_counts_by_window.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for model_id in MODEL_IDS:
        d = counts[counts["model_id"] == model_id]

        axes[0].plot(
            d["window"],
            d["good_cycles"],
            marker="o",
            linewidth=2.3,
            label=model_id,
        )

        axes[1].plot(
            d["window"],
            d["good_percent"],
            marker="o",
            linewidth=2.3,
            label=model_id,
        )

    axes[0].grid(True, alpha=0.35)
    axes[0].set_ylabel("Number of good cycles")
    axes[0].set_title("Good-fit cycles by region")
    axes[0].legend(loc="best")

    axes[1].grid(True, alpha=0.35)
    axes[1].set_ylabel("Good cycles [%]")
    axes[1].set_xlabel("Cycle region")
    axes[1].set_title("Good-fit fraction by region")
    axes[1].legend(loc="best")

    savefig(FIG_DIR / "12_good_cycle_counts_by_window.png")


# ============================================================
# Report
# ============================================================
def make_window_stats(good: pd.DataFrame) -> pd.DataFrame:
    if len(good) == 0:
        return pd.DataFrame()

    stats = (
        good.groupby(["model_id", "window"])
        .agg(
            n_good_cycles=("cycle_index", "count"),
            cycle_min=("cycle_index", "min"),
            cycle_max=("cycle_index", "max"),
            best_rmse_mV_mean=("best_rmse_mV", "mean"),
            best_rmse_mV_median=("best_rmse_mV", "median"),
            best_rmse_mV_max=("best_rmse_mV", "max"),
            best_bfr_percent_mean=("best_bfr_percent", "mean"),
            best_bfr_percent_median=("best_bfr_percent", "median"),
            best_r2_percent_mean=("best_r2_percent", "mean"),
            best_r2_percent_median=("best_r2_percent", "median"),
            median_to_best_rmse_ratio_median=("median_to_best_rmse_ratio", "median"),
            fail_rate_percent_mean=("fail_rate_percent", "mean"),
            rankX_fraction_median=("rankX_fraction", "median"),
            rankPhi_fraction_median=("rankPhi_fraction", "median"),
        )
        .reset_index()
    )

    order = {
        "0-4 smoke": 0,
        "5-33 difficult": 1,
        "34-99 later": 2,
    }

    stats["window_order"] = stats["window"].map(order).fillna(99)
    stats = stats.sort_values(["model_id", "window_order"]).drop(columns=["window_order"])

    return stats


def write_report(
    all_summary: pd.DataFrame,
    good_summary: pd.DataFrame,
    excluded_summary: pd.DataFrame,
    window_stats: pd.DataFrame,
) -> None:
    report_path = TAB_DIR / "good_fit_diagnostic_report.txt"

    lines = []
    lines.append("=" * 100)
    lines.append("GOOD-FIT-ONLY DIAGNOSTIC REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Good-fit rule:")
    lines.append(f"  best_rmse <= {GOOD_RMSE_V:.6g} V ({1000*GOOD_RMSE_V:.3g} mV)")
    lines.append(f"  best_bfr_percent >= {GOOD_BFR_PERCENT:.3g} %")
    lines.append(f"  best_r2_percent >= {GOOD_R2_PERCENT:.5g} %")
    lines.append(f"  REQUIRE_RANK_FILTER = {REQUIRE_RANK_FILTER}")
    lines.append("")

    lines.append("Coverage summary:")
    for model_id in MODEL_IDS:
        d_all = all_summary[all_summary["model_id"] == model_id]
        d_good = good_summary[good_summary["model_id"] == model_id]
        d_excl = excluded_summary[excluded_summary["model_id"] == model_id]

        found = set(d_all["cycle_index"].astype(int))
        missing = sorted(EXPECTED_CYCLES - found)

        lines.append("")
        lines.append(f"{model_id}:")
        lines.append(f"  all cycles found: {len(found)}")
        lines.append(f"  missing cycles: {missing}")
        lines.append(f"  good cycles kept: {len(d_good)}")
        lines.append(f"  excluded cycles: {len(d_excl)}")
        if len(d_good):
            lines.append(f"  first good cycle: {int(d_good['cycle_index'].min())}")
            lines.append(f"  last good cycle: {int(d_good['cycle_index'].max())}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("Window statistics for good cycles")
    lines.append("=" * 100)
    lines.append(simple_text_table(window_stats))
    lines.append("")

    lines.append("=" * 100)
    lines.append("Good cycles kept")
    lines.append("=" * 100)
    cols_good = [
        "model_id",
        "cycle_index",
        "window",
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
        "window",
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
    print("GOOD-FIT-ONLY DIAGNOSTICS: S7_C4K and S17_C4K")
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
        print("  cycles:", len(found))
        print("  missing:", missing)

    summary_all = pd.concat(summaries, ignore_index=True, sort=False)
    best_all = pd.concat(bests, ignore_index=True, sort=False)

    summary_all = standardize_best_columns(summary_all)
    best_all = standardize_best_columns(best_all)

    summary_all = add_diagnostics(summary_all)
    summary_all["window"] = summary_all["cycle_index"].apply(window_label)

    good_summary = summary_all[summary_all["is_good_fit_for_analysis"]].copy()
    excluded_summary = summary_all[~summary_all["is_good_fit_for_analysis"]].copy()

    good_summary = good_summary.sort_values(["model_id", "cycle_index"]).reset_index(drop=True)
    excluded_summary = excluded_summary.sort_values(["model_id", "cycle_index"]).reset_index(drop=True)

    good_best = merge_summary_best(good_summary, best_all)

    # Save tables.
    summary_all.to_csv(TAB_DIR / "all_cycles_with_good_fit_flags.csv", index=False)
    good_summary.to_csv(TAB_DIR / "good_cycles_only_summary.csv", index=False)
    excluded_summary.to_csv(TAB_DIR / "excluded_bad_or_weak_cycles.csv", index=False)
    good_best.to_csv(TAB_DIR / "good_cycles_only_best_runs.csv", index=False)

    window_stats = make_window_stats(good_summary)
    window_stats.to_csv(TAB_DIR / "good_cycles_window_stats.csv", index=False)

    # Plots.
    plot_good_cycle_mask(summary_all)
    plot_good_rmse(good_summary)
    plot_good_bfr_r2(good_summary)
    plot_good_rank_diagnostics(good_summary)
    make_good_parameter_plots(good_best)
    plot_good_counts_by_window(good_summary, excluded_summary)

    # Report.
    write_report(summary_all, good_summary, excluded_summary, window_stats)

    print()
    print("=" * 100)
    print("GOOD-FIT-ONLY DIAGNOSTICS COMPLETE")
    print("=" * 100)
    print("Good cycles kept:")
    for model_id in MODEL_IDS:
        d = good_summary[good_summary["model_id"] == model_id]
        print(f"  {model_id}: {len(d)} cycles")

    print()
    print("Excluded cycles:")
    for model_id in MODEL_IDS:
        d = excluded_summary[excluded_summary["model_id"] == model_id]
        print(f"  {model_id}: {len(d)} cycles")

    print()
    print("Figures saved to:")
    print(FIG_DIR)

    print()
    print("Tables saved to:")
    print(TAB_DIR)

    print()
    print("Most useful files:")
    print("  ", TAB_DIR / "good_cycles_only_summary.csv")
    print("  ", TAB_DIR / "excluded_bad_or_weak_cycles.csv")
    print("  ", TAB_DIR / "good_fit_diagnostic_report.txt")
    print("  ", FIG_DIR / "01_good_cycles_best_rmse_mV.png")
    print("  ", FIG_DIR / "02_good_fit_mask.png")
    print("  ", FIG_DIR / "12_good_cycle_counts_by_window.png")
    print("=" * 100)


if __name__ == "__main__":
    main()