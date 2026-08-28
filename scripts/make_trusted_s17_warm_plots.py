#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_trusted_s17_warm_plots.py

Trusted post-processing plots for S17_C4 warm-continuation run.

Reads:
    results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_summary.csv
    results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_best_runs.csv

Writes:
    results/figures/trusted_s17_warm_review/
    results/tables/trusted_s17_warm_review/

Purpose:
    Create cleaner, more interpretable plots than the automatic combined plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

SUMMARY_CSV = PROJECT_DIR / "results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_summary.csv"
BEST_CSV = PROJECT_DIR / "results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_best_runs.csv"

FIG_DIR = PROJECT_DIR / "results/figures/trusted_s17_warm_review"
TAB_DIR = PROJECT_DIR / "results/tables/trusted_s17_warm_review"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=280, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-300, a / b, np.nan)


def main():
    require_file(SUMMARY_CSV)
    require_file(BEST_CSV)

    df_sum = pd.read_csv(SUMMARY_CSV)
    df_best = pd.read_csv(BEST_CSV)

    df_sum = df_sum.sort_values("cycle_index").reset_index(drop=True)
    df_best = df_best.sort_values("cycle_index").reset_index(drop=True)

    # ------------------------------------------------------------
    # Coverage check
    # ------------------------------------------------------------
    cycles = set(df_sum["cycle_index"].astype(int))
    missing = sorted(set(range(100)) - cycles)

    coverage_lines = []
    coverage_lines.append("S17_C4 warm-continuation trusted review")
    coverage_lines.append("=" * 70)
    coverage_lines.append(f"Rows in summary: {len(df_sum)}")
    coverage_lines.append(f"Min cycle: {df_sum['cycle_index'].min()}")
    coverage_lines.append(f"Max cycle: {df_sum['cycle_index'].max()}")
    coverage_lines.append(f"Missing cycles: {missing}")
    coverage_lines.append("")
    coverage_lines.append("Important interpretation:")
    coverage_lines.append("- 100 rows and no missing cycles means cycles 0--99 are covered.")
    coverage_lines.append("- Low best RMSE means voltage fit is good.")
    coverage_lines.append("- Rank_X_raw tells whether the fitted state trajectory is numerically rich.")
    coverage_lines.append("- Large median RMSE means many starts failed even if the best start succeeded.")
    coverage_lines.append("- S17 has more states than S7, so rank behavior is especially important.")
    coverage_lines.append("=" * 70)

    coverage_txt = TAB_DIR / "coverage_check.txt"
    coverage_txt.write_text("\n".join(coverage_lines), encoding="utf-8")
    print(f"[saved] {coverage_txt}")

    # ------------------------------------------------------------
    # Save cleaned copies
    # ------------------------------------------------------------
    df_sum.to_csv(TAB_DIR / "s17_all_cycles_summary_clean.csv", index=False)
    df_best.to_csv(TAB_DIR / "s17_all_cycles_best_runs_clean.csv", index=False)

    # ------------------------------------------------------------
    # Derived diagnostics
    # ------------------------------------------------------------
    df_sum["fail_rate"] = df_sum["n_fail"] / (df_sum["n_success"] + df_sum["n_fail"]).replace(0, np.nan)
    df_sum["median_to_best_rmse_ratio"] = safe_ratio(df_sum["median_rmse"], df_sum["best_rmse"])

    suspicious = df_sum[
        (df_sum["best_rmse"] > 1e-2)
        | (df_sum["median_rmse"] > 1e-1)
        | (df_sum["fail_rate"] > 0.50)
        | (df_sum["best_rank_X_raw"] < 2)
        | (df_sum["median_to_best_rmse_ratio"] > 20)
    ].copy()

    suspicious_cols = [
        "cycle_index",
        "n_success",
        "n_fail",
        "fail_rate",
        "best_seed",
        "best_rmse",
        "median_rmse",
        "median_to_best_rmse_ratio",
        "best_r2_percent",
        "best_bfr_percent",
        "best_rank_phi_raw",
        "best_ncols_phi_raw",
        "best_rank_X_raw",
        "best_ncols_X_raw",
    ]

    suspicious_cols = [c for c in suspicious_cols if c in suspicious.columns]
    suspicious[suspicious_cols].to_csv(TAB_DIR / "suspicious_cycles.csv", index=False)

    # ------------------------------------------------------------
    # Plot 1: Best RMSE and median RMSE on log scale
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.semilogy(df_sum["cycle_index"], df_sum["best_rmse"], marker="o", linewidth=2.2, label="Best RMSE")
    plt.semilogy(df_sum["cycle_index"], df_sum["median_rmse"], marker="s", linewidth=2.0, label="Median RMSE")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("RMSE [V], log scale")
    plt.title("S17_C4 warm continuation: best vs median RMSE")
    plt.legend(loc="best")
    savefig(FIG_DIR / "01_rmse_best_vs_median_logscale.png")

    # ------------------------------------------------------------
    # Plot 2: Best RMSE only, clipped linear scale
    # ------------------------------------------------------------
    y = df_sum["best_rmse"].to_numpy(dtype=float)
    ymax = np.nanpercentile(y, 95) * 1.3
    ymax = max(ymax, 1e-4)

    plt.figure(figsize=(12, 6))
    plt.plot(df_sum["cycle_index"], df_sum["best_rmse"], marker="o", linewidth=2.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [V]")
    plt.ylim(0, ymax)
    plt.title("S17_C4 warm continuation: best RMSE only")
    savefig(FIG_DIR / "02_best_rmse_linear_clipped.png")

    # ------------------------------------------------------------
    # Plot 3: R2 and BFR
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_sum["cycle_index"], df_sum["best_r2_percent"], marker="o", linewidth=2.2, label="Best R2 [%]")
    plt.plot(df_sum["cycle_index"], df_sum["best_bfr_percent"], marker="s", linewidth=2.2, label="Best BFR [%]")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Percent [%]")
    plt.title("S17_C4 warm continuation: fit quality")
    plt.legend(loc="best")
    savefig(FIG_DIR / "03_fit_quality_r2_bfr.png")

    # ------------------------------------------------------------
    # Plot 4: Success and failure counts
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_sum["cycle_index"], df_sum["n_success"], marker="o", linewidth=2.2, label="Successful starts")
    plt.plot(df_sum["cycle_index"], df_sum["n_fail"], marker="s", linewidth=2.2, label="Failed starts")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Count out of 100 starts")
    plt.title("S17_C4 warm continuation: optimizer success/failure count")
    plt.legend(loc="best")
    savefig(FIG_DIR / "04_success_failure_counts.png")

    # ------------------------------------------------------------
    # Plot 5: Failure rate
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_sum["cycle_index"], 100 * df_sum["fail_rate"], marker="o", linewidth=2.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Failure rate [%]")
    plt.title("S17_C4 warm continuation: failed starts per cycle")
    savefig(FIG_DIR / "05_failure_rate.png")

    # ------------------------------------------------------------
    # Plot 6: Raw rank diagnostics
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_sum["cycle_index"], df_sum["best_rank_phi_raw"], marker="o", linewidth=2.2, label="rank(Phi)")
    plt.plot(df_sum["cycle_index"], df_sum["best_ncols_phi_raw"], linestyle="--", linewidth=1.8, label="columns(Phi)")
    plt.plot(df_sum["cycle_index"], df_sum["best_rank_X_raw"], marker="s", linewidth=2.2, label="rank(X)")
    plt.plot(df_sum["cycle_index"], df_sum["best_ncols_X_raw"], linestyle=":", linewidth=2.0, label="states")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Rank / dimension")
    plt.title("S17_C4 warm continuation: raw rank diagnostics")
    plt.legend(loc="best")
    savefig(FIG_DIR / "06_rank_diagnostics.png")

    # ------------------------------------------------------------
    # Plot 7: Absolute dynamic/gain parameters
    # ------------------------------------------------------------
    param_cols = [
        "alpha_n_hat",
        "alpha_p_hat",
        "K_e_hat",
        "g_n_hat",
        "g_p_hat",
        "g_e_hat",
    ]
    param_cols = [c for c in param_cols if c in df_best.columns]

    plt.figure(figsize=(12, 7))
    for c in param_cols:
        plt.plot(df_best["cycle_index"], df_best[c], marker="o", linewidth=2.0, label=c)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Parameter value")
    plt.title("S17_C4 warm continuation: absolute fitted dynamic/gain parameters")
    plt.legend(loc="best", fontsize=8)
    savefig(FIG_DIR / "07_absolute_dynamic_gain_parameters.png")

    # ------------------------------------------------------------
    # Plot 8: Dynamic/gain parameters on log scale
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    for c in param_cols:
        y = pd.to_numeric(df_best[c], errors="coerce").to_numpy(dtype=float)
        y = np.where(y > 0, y, np.nan)
        plt.semilogy(df_best["cycle_index"], y, marker="o", linewidth=2.0, label=c)
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Parameter value, log scale")
    plt.title("S17_C4 warm continuation: dynamic/gain parameters on log scale")
    plt.legend(loc="best", fontsize=8)
    savefig(FIG_DIR / "08_dynamic_gain_parameters_logscale.png")

    # ------------------------------------------------------------
    # Plot 9: Stoichiometry offsets
    # ------------------------------------------------------------
    theta_cols = ["theta_n0_hat", "theta_p0_hat"]
    theta_cols = [c for c in theta_cols if c in df_best.columns]

    plt.figure(figsize=(12, 6))
    for c in theta_cols:
        plt.plot(df_best["cycle_index"], df_best[c], marker="o", linewidth=2.2, label=c)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Theta estimate")
    plt.title("S17_C4 warm continuation: fitted theta offsets")
    plt.legend(loc="best")
    savefig(FIG_DIR / "09_theta_offsets.png")

    # ------------------------------------------------------------
    # Plot 10: Voltage-fit quality classification
    # ------------------------------------------------------------
    good_fit = df_sum["best_rmse"] < 0.002
    weak_fit = (df_sum["best_rmse"] >= 0.002) & (df_sum["best_rmse"] < 0.01)
    bad_fit = df_sum["best_rmse"] >= 0.01

    plt.figure(figsize=(12, 4.8))
    plt.scatter(df_sum.loc[good_fit, "cycle_index"], df_sum.loc[good_fit, "best_rmse"], marker="o", label="good: RMSE < 0.002")
    plt.scatter(df_sum.loc[weak_fit, "cycle_index"], df_sum.loc[weak_fit, "best_rmse"], marker="s", label="weak: 0.002 <= RMSE < 0.01")
    plt.scatter(df_sum.loc[bad_fit, "cycle_index"], df_sum.loc[bad_fit, "best_rmse"], marker="x", label="bad: RMSE >= 0.01")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [V]")
    plt.title("S17_C4 warm continuation: voltage-fit quality classification")
    plt.legend(loc="best")
    savefig(FIG_DIR / "10_voltage_fit_quality_classification.png")

    # ------------------------------------------------------------
    # Print concise report
    # ------------------------------------------------------------
    print()
    print("=" * 80)
    print("S17_C4 TRUSTED REVIEW COMPLETE")
    print("=" * 80)
    print(f"Summary rows: {len(df_sum)}")
    print(f"Cycle range: {df_sum['cycle_index'].min()} to {df_sum['cycle_index'].max()}")
    print(f"Missing cycles: {missing}")
    print()
    print("Best RMSE stats:")
    print(df_sum["best_rmse"].describe().to_string())
    print()
    print("Median RMSE stats:")
    print(df_sum["median_rmse"].describe().to_string())
    print()
    print("Failure-rate stats:")
    print((100 * df_sum["fail_rate"]).describe().to_string())
    print()
    print("Rank X counts:")
    print(df_sum["best_rank_X_raw"].value_counts(dropna=False).sort_index().to_string())
    print()
    print("Suspicious cycles saved to:")
    print(TAB_DIR / "suspicious_cycles.csv")
    print("Figures saved to:")
    print(FIG_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()