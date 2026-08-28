#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


MODEL_ID = "S17_C4"

PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

SUMMARY_CSV = PROJECT_DIR / "results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_summary.csv"
BEST_CSV = PROJECT_DIR / "results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_best_runs.csv"

OUT_DIR = PROJECT_DIR / "results/figures/real_warm_continuation_ctid/S17_C4/trusted_diagnostics"
TABLE_OUT_DIR = PROJECT_DIR / "results/tables/real_warm_continuation_ctid/S17_C4/trusted_diagnostics"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cycle_index"] = df["cycle_index"].astype(int)

    # Conservative quality flags.
    df["flag_good_voltage"] = (
        (df["best_rmse"] <= 0.002)
        & (df["best_bfr_percent"] >= 98.0)
        & (df["best_r2_percent"] >= 99.95)
    )

    df["flag_bad_voltage"] = (
        (df["best_rmse"] > 0.01)
        | (df["best_bfr_percent"] < 90.0)
        | (df["best_r2_percent"] < 99.0)
    )

    # S17 has 17 states. Rank 1 or 2 is a strong collapse.
    # Rank below about half the states is still weak.
    df["flag_rankX_collapsed"] = df["best_rank_X_raw"] <= 2
    df["flag_rankX_weak"] = df["best_rank_X_raw"] < 8

    df["flag_rankPhi_deficient"] = df["best_rank_phi_raw"] < df["best_ncols_phi_raw"]

    df["flag_many_failures"] = df["n_fail"] >= 30

    return df


def main() -> None:
    require_file(SUMMARY_CSV)
    require_file(BEST_CSV)

    summary = pd.read_csv(SUMMARY_CSV)
    best = pd.read_csv(BEST_CSV)

    # Normalize column names from best table if needed.
    if "rmse" in best.columns and "best_rmse" not in best.columns:
        best = best.rename(columns={"rmse": "best_rmse"})
    if "r2_percent" in best.columns and "best_r2_percent" not in best.columns:
        best = best.rename(columns={"r2_percent": "best_r2_percent"})
    if "bfr_percent" in best.columns and "best_bfr_percent" not in best.columns:
        best = best.rename(columns={"bfr_percent": "best_bfr_percent"})

    summary["cycle_index"] = summary["cycle_index"].astype(int)
    best["cycle_index"] = best["cycle_index"].astype(int)

    summary = summary.sort_values("cycle_index").reset_index(drop=True)
    best = best.sort_values("cycle_index").reset_index(drop=True)

    # Confirm cycle coverage.
    cycles_found = set(summary["cycle_index"].astype(int))
    missing = sorted(set(range(100)) - cycles_found)

    print("=" * 80)
    print(f"{MODEL_ID} trusted diagnostics")
    print("=" * 80)
    print("Summary rows:", len(summary))
    print("Min cycle:", summary["cycle_index"].min())
    print("Max cycle:", summary["cycle_index"].max())
    print("Missing cycles:", missing)
    print("=" * 80)

    summary = add_quality_flags(summary)

    summary.to_csv(TABLE_OUT_DIR / "s17_trusted_summary_with_flags.csv", index=False)
    best.to_csv(TABLE_OUT_DIR / "s17_best_runs_copy.csv", index=False)

    # ------------------------------------------------------------
    # Plot 1: Best RMSE only
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(summary["cycle_index"], summary["best_rmse"], marker="o", linewidth=2.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [V]")
    plt.title("S17_C4: best RMSE across 100 cycles")
    savefig("01_best_rmse_linear.png")

    # ------------------------------------------------------------
    # Plot 2: Best and median RMSE, log scale
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.semilogy(summary["cycle_index"], summary["best_rmse"], marker="o", linewidth=2.2, label="best RMSE")
    plt.semilogy(summary["cycle_index"], summary["median_rmse"], marker="s", linewidth=2.0, label="median RMSE")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("RMSE [V], log scale")
    plt.title("S17_C4: best vs median RMSE across cycles")
    plt.legend(loc="best")
    savefig("02_best_median_rmse_log.png")

    # ------------------------------------------------------------
    # Plot 3: BFR
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(summary["cycle_index"], summary["best_bfr_percent"], marker="o", linewidth=2.2)
    plt.axhline(99.0, linestyle="--", linewidth=1.5, label="99% BFR")
    plt.axhline(95.0, linestyle=":", linewidth=1.5, label="95% BFR")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best BFR [%]")
    plt.title("S17_C4: best BFR across cycles")
    plt.legend(loc="best")
    savefig("03_best_bfr.png")

    # ------------------------------------------------------------
    # Plot 4: R2
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(summary["cycle_index"], summary["best_r2_percent"], marker="o", linewidth=2.2)
    plt.axhline(99.95, linestyle="--", linewidth=1.5, label="99.95% R2")
    plt.axhline(99.0, linestyle=":", linewidth=1.5, label="99% R2")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best R2 [%]")
    plt.title("S17_C4: best R2 across cycles")
    plt.legend(loc="best")
    savefig("04_best_r2.png")

    # ------------------------------------------------------------
    # Plot 5: success/failure counts
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(summary["cycle_index"], summary["n_success"], marker="o", linewidth=2.2, label="successful fits")
    plt.plot(summary["cycle_index"], summary["n_fail"], marker="s", linewidth=2.2, label="failed fits")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Count out of 100 local starts")
    plt.title("S17_C4: successful and failed local starts per cycle")
    plt.legend(loc="best")
    savefig("05_success_failure_counts.png")

    # ------------------------------------------------------------
    # Plot 6: raw ranks
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(summary["cycle_index"], summary["best_rank_X_raw"], marker="o", linewidth=2.2, label="rank(X)")
    plt.plot(summary["cycle_index"], summary["best_rank_phi_raw"], marker="s", linewidth=2.2, label="rank(Phi)")
    plt.plot(summary["cycle_index"], summary["best_ncols_X_raw"], linestyle="--", linewidth=1.5, label="max rank X")
    plt.plot(summary["cycle_index"], summary["best_ncols_phi_raw"], linestyle=":", linewidth=1.5, label="max rank Phi")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Raw numerical rank")
    plt.title("S17_C4: raw state and output-feature ranks")
    plt.legend(loc="best")
    savefig("06_raw_ranks.png")

    # ------------------------------------------------------------
    # Plot 7: quality flags
    # ------------------------------------------------------------
    flag_cols = [
        "flag_good_voltage",
        "flag_bad_voltage",
        "flag_rankX_collapsed",
        "flag_rankX_weak",
        "flag_rankPhi_deficient",
        "flag_many_failures",
    ]

    flag_df = summary[["cycle_index"] + flag_cols].copy()

    plt.figure(figsize=(13, 6))
    offset = 0
    for col in flag_cols:
        y = np.where(flag_df[col].astype(bool), offset + 1, np.nan)
        plt.scatter(flag_df["cycle_index"], y, s=45, label=col)
        offset += 1

    plt.yticks(range(1, len(flag_cols) + 1), flag_cols)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.title("S17_C4: quality flags by cycle")
    plt.legend(loc="upper right", fontsize=8)
    savefig("07_quality_flags.png")

    # ------------------------------------------------------------
    # Plot 8: each physical parameter separately
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

    available = [c for c in param_cols if c in best.columns]

    for col in available:
        plt.figure(figsize=(12, 6))
        plt.plot(best["cycle_index"], best[col], marker="o", linewidth=2.2)
        plt.grid(True, alpha=0.35)
        plt.xlabel("Cycle index")
        plt.ylabel(col)
        plt.title(f"S17_C4: {col} across cycles")
        savefig(f"08_param_{col}.png")

    # ------------------------------------------------------------
    # Plot 9: positive parameters on log scale
    # ------------------------------------------------------------
    positive_params = [
        c for c in [
            "alpha_n_hat",
            "alpha_p_hat",
            "K_e_hat",
            "g_n_hat",
            "g_p_hat",
            "g_e_hat",
        ]
        if c in best.columns
    ]

    plt.figure(figsize=(12, 7))
    for col in positive_params:
        y = pd.to_numeric(best[col], errors="coerce").to_numpy(dtype=float)
        y = np.where(y > 0, y, np.nan)
        plt.semilogy(best["cycle_index"], y, marker="o", linewidth=2.0, label=col)

    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Parameter value, log scale")
    plt.title("S17_C4: positive physical parameters, log scale")
    plt.legend(loc="best", fontsize=8)
    savefig("09_positive_parameters_log_scale.png")

    # ------------------------------------------------------------
    # Plot 10: theta offsets
    # ------------------------------------------------------------
    theta_cols = [c for c in ["theta_n0_hat", "theta_p0_hat"] if c in best.columns]

    plt.figure(figsize=(12, 6))
    for col in theta_cols:
        plt.plot(best["cycle_index"], best[col], marker="o", linewidth=2.2, label=col)

    plt.axhline(0.02, linestyle=":", linewidth=1.2)
    plt.axhline(0.98, linestyle=":", linewidth=1.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Stoichiometry offset")
    plt.title("S17_C4: fitted theta offsets")
    plt.legend(loc="best")
    savefig("10_theta_offsets.png")

    # ------------------------------------------------------------
    # Plot 11: RMSE with flags
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.semilogy(summary["cycle_index"], summary["best_rmse"], marker="o", linewidth=2.2, label="best RMSE")

    bad = summary[summary["flag_bad_voltage"]]
    if len(bad):
        plt.scatter(bad["cycle_index"], bad["best_rmse"], s=90, marker="x", label="bad voltage cycles")

    collapsed = summary[summary["flag_rankX_collapsed"]]
    if len(collapsed):
        plt.scatter(collapsed["cycle_index"], collapsed["best_rmse"], s=50, marker="s", label="rankX collapsed")

    weak = summary[summary["flag_rankX_weak"] & (~summary["flag_rankX_collapsed"])]
    if len(weak):
        plt.scatter(weak["cycle_index"], weak["best_rmse"], s=35, marker="^", label="rankX weak")

    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Best RMSE [V], log scale")
    plt.title("S17_C4: RMSE with bad-voltage and rank flags")
    plt.legend(loc="best")
    savefig("11_rmse_with_flags.png")

    # ------------------------------------------------------------
    # Write text report
    # ------------------------------------------------------------
    bad_cycles = summary.loc[summary["flag_bad_voltage"], "cycle_index"].astype(int).tolist()
    rankx_collapsed_cycles = summary.loc[summary["flag_rankX_collapsed"], "cycle_index"].astype(int).tolist()
    rankx_weak_cycles = summary.loc[summary["flag_rankX_weak"], "cycle_index"].astype(int).tolist()
    many_fail_cycles = summary.loc[summary["flag_many_failures"], "cycle_index"].astype(int).tolist()

    report_path = TABLE_OUT_DIR / "s17_trusted_diagnostic_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("S17_C4 trusted diagnostic report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Rows: {len(summary)}\n")
        f.write(f"Min cycle: {summary['cycle_index'].min()}\n")
        f.write(f"Max cycle: {summary['cycle_index'].max()}\n")
        f.write(f"Missing cycles: {missing}\n\n")

        f.write("Best RMSE summary:\n")
        f.write(str(summary["best_rmse"].describe()) + "\n\n")

        f.write("Median RMSE summary:\n")
        f.write(str(summary["median_rmse"].describe()) + "\n\n")

        f.write("Failure-rate summary [%]:\n")
        total = summary["n_success"] + summary["n_fail"]
        fail_rate = 100.0 * summary["n_fail"] / total.replace(0, np.nan)
        f.write(str(fail_rate.describe()) + "\n\n")

        f.write("Rank-X value counts:\n")
        f.write(str(summary["best_rank_X_raw"].value_counts(dropna=False).sort_index()) + "\n\n")

        f.write(f"Bad voltage cycles: {bad_cycles}\n\n")
        f.write(f"Rank-X collapsed cycles: {rankx_collapsed_cycles}\n\n")
        f.write(f"Rank-X weak cycles: {rankx_weak_cycles}\n\n")
        f.write(f"Cycles with many failed starts: {many_fail_cycles}\n\n")

        f.write("Tail of summary:\n")
        f.write(
            summary[
                [
                    "cycle_index",
                    "n_success",
                    "n_fail",
                    "best_seed",
                    "best_rmse",
                    "median_rmse",
                    "best_bfr_percent",
                    "best_rank_phi_raw",
                    "best_ncols_phi_raw",
                    "best_rank_X_raw",
                    "best_ncols_X_raw",
                ]
            ].tail(20).to_string(index=False)
        )
        f.write("\n")

    print(f"[saved] {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()