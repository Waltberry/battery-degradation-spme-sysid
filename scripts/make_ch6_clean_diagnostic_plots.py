#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_clean_diagnostic_plots.py

Purpose
-------
Regenerate Chapter 6 diagnostic plots from saved CSV tables with clean
thesis-worthy titles.

This script does NOT paste titles onto PNGs.
It recreates the figures using Matplotlib, so the title is part of the plot.

Outputs:
    figures/chapter6/fig_ch6_good_fit_mask.png
    figures/chapter6/fig_ch6_best_rmse.png
    figures/chapter6/fig_ch6_bfr.png
    figures/chapter6/fig_ch6_core_parameters.png
    figures/chapter6/fig_ch6_electrolyte_k_parameters.png
    figures/chapter6/fig_ch6_beta_coefficients.png
"""

from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

REAL_PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

THESIS_FIG_DIR = REAL_PROJECT / "figures" / "chapter6"
BACKUP_FIG_DIR = REAL_PROJECT / "results" / "figures" / "chapter6_thesis_plots"
TABLE_OUT_DIR = REAL_PROJECT / "results" / "tables" / "chapter6_thesis_plots"

FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"
MIRROR_TO_FLOW_THESIS = True

SOURCE_TABLE_DIRS = [
    REAL_PROJECT / "results" / "tables" / "kparam_s17_from34_good_fit_diagnostics",
    REAL_PROJECT / "results" / "tables" / "chapter6_thesis_plots",
    REAL_PROJECT / "results" / "tables" / "real_warm_continuation_ctid" / "S17_C4K",
]

for p in [THESIS_FIG_DIR, BACKUP_FIG_DIR, TABLE_OUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

START_ORIGINAL_CYCLE = 34

TITLE_FONTSIZE = 15
AXIS_FONTSIZE = 11
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 9

LINEWIDTH = 1.8
MARKERSIZE = 4.0


# ============================================================
# Helpers
# ============================================================

def savefig(path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

    BACKUP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP_FIG_DIR / path.name)

    if MIRROR_TO_FLOW_THESIS:
        FLOW_THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)

    print("[saved]", path)


def find_first_existing(names: list[str]) -> Path | None:
    for d in SOURCE_TABLE_DIRS:
        for name in names:
            p = d / name
            if p.exists():
                return p
    return None


def read_first_existing(names: list[str]) -> pd.DataFrame | None:
    p = find_first_existing(names)
    if p is None:
        return None

    print("[reading]", p)
    return pd.read_csv(p)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    if "cycle_index" in df.columns:
        df["original_cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
        df["retained_cycle_index"] = df["original_cycle_index"] - START_ORIGINAL_CYCLE
    elif "original_cycle_index" in df.columns and "retained_cycle_index" not in df.columns:
        df["original_cycle_index"] = pd.to_numeric(df["original_cycle_index"], errors="coerce")
        df["retained_cycle_index"] = df["original_cycle_index"] - START_ORIGINAL_CYCLE

    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def get_cycle_x(df: pd.DataFrame) -> np.ndarray:
    if "retained_cycle_index" in df.columns:
        return make_numeric(df, "retained_cycle_index")
    if "cycle_index" in df.columns:
        return make_numeric(df, "cycle_index")
    if "original_cycle_index" in df.columns:
        return make_numeric(df, "original_cycle_index")
    return np.arange(len(df), dtype=float)


def plot_line(ax, x, y, label=None):
    ax.plot(
        x,
        y,
        marker="o",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        label=label,
    )


def clean_axis(ax):
    ax.grid(True, alpha=0.30)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


# ============================================================
# Load data
# ============================================================

def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = read_first_existing(
        [
            "s17_from34_all_cycles_with_good_fit_flags.csv",
            "s17_all_summary_reindexed.csv",
            "all_cycles_summary.csv",
        ]
    )

    best = read_first_existing(
        [
            "s17_from34_good_cycles_only_best_runs.csv",
            "s17_good_best_reindexed.csv",
            "all_cycles_best_runs.csv",
        ]
    )

    if summary is None:
        raise FileNotFoundError("Could not find a usable summary table.")

    if best is None:
        raise FileNotFoundError("Could not find a usable best-runs table.")

    summary = standardize_columns(summary)
    best = standardize_columns(best)

    if "retained_cycle_index" in summary.columns:
        summary = summary.sort_values("retained_cycle_index").reset_index(drop=True)
    elif "cycle_index" in summary.columns:
        summary = summary.sort_values("cycle_index").reset_index(drop=True)

    if "retained_cycle_index" in best.columns:
        best = best.sort_values("retained_cycle_index").reset_index(drop=True)
    elif "cycle_index" in best.columns:
        best = best.sort_values("cycle_index").reset_index(drop=True)

    summary.to_csv(TABLE_OUT_DIR / "clean_plot_summary_input.csv", index=False)
    best.to_csv(TABLE_OUT_DIR / "clean_plot_best_input.csv", index=False)

    print("[saved table]", TABLE_OUT_DIR / "clean_plot_summary_input.csv")
    print("[saved table]", TABLE_OUT_DIR / "clean_plot_best_input.csv")

    return summary, best


# ============================================================
# Plot 1 — Good Fit Mask
# ============================================================

def plot_good_fit_mask(summary: pd.DataFrame) -> None:
    df = summary.copy()
    x = get_cycle_x(df)

    flag_col = pick_col(
        df,
        [
            "is_good_fit",
            "good_fit",
            "good_fit_mask",
            "keep_good_fit",
            "is_good",
        ],
    )

    if flag_col is not None:
        y = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(float).to_numpy()
    else:
        rmse_col = pick_col(df, ["best_rmse", "rmse"])
        bfr_col = pick_col(df, ["best_bfr_percent", "bfr_percent"])
        r2_col = pick_col(df, ["best_r2_percent", "r2_percent"])

        if rmse_col is None or bfr_col is None or r2_col is None:
            raise RuntimeError("Cannot infer good-fit mask. Missing RMSE/BFR/R2 columns.")

        y = (
            (pd.to_numeric(df[rmse_col], errors="coerce") <= 0.002)
            & (pd.to_numeric(df[bfr_col], errors="coerce") >= 98.0)
            & (pd.to_numeric(df[r2_col], errors="coerce") >= 99.95)
        ).astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(11.8, 4.6))

    ax.step(x, y, where="mid", linewidth=2.0)
    ax.scatter(x, y, s=18)

    ax.set_title("Good Fit Mask", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("Good fit", fontsize=AXIS_FONTSIZE)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Rejected", "Kept"])
    clean_axis(ax)

    savefig(THESIS_FIG_DIR / "fig_ch6_good_fit_mask.png")


# ============================================================
# Plot 2 — Best RMSE
# ============================================================

def plot_best_rmse(summary: pd.DataFrame) -> None:
    df = summary.copy()
    x = get_cycle_x(df)

    rmse_col = pick_col(df, ["best_rmse_mV", "best_rmse", "rmse"])

    if rmse_col is None:
        raise RuntimeError("Could not find RMSE column.")

    y = make_numeric(df, rmse_col)

    if rmse_col in ["best_rmse", "rmse"]:
        y = 1000.0 * y

    fig, ax = plt.subplots(figsize=(11.8, 4.8))

    plot_line(ax, x, y)

    ax.set_title("Best RMSE", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("RMSE [mV]", fontsize=AXIS_FONTSIZE)
    clean_axis(ax)

    savefig(THESIS_FIG_DIR / "fig_ch6_best_rmse.png")


# ============================================================
# Plot 3 — BFR
# ============================================================

def plot_bfr(summary: pd.DataFrame) -> None:
    df = summary.copy()
    x = get_cycle_x(df)

    bfr_col = pick_col(df, ["best_bfr_percent", "bfr_percent", "best_bfr"])
    r2_col = pick_col(df, ["best_r2_percent", "r2_percent", "best_r2"])

    if bfr_col is None:
        raise RuntimeError("Could not find BFR column.")

    fig, ax = plt.subplots(figsize=(11.8, 4.8))

    plot_line(ax, x, make_numeric(df, bfr_col), label="BFR")

    if r2_col is not None:
        plot_line(ax, x, make_numeric(df, r2_col), label=r"$R^2$")

    ax.set_title("BFR", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel("Fit metric [%]", fontsize=AXIS_FONTSIZE)
    clean_axis(ax)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    savefig(THESIS_FIG_DIR / "fig_ch6_bfr.png")


# ============================================================
# Multi-panel helper
# ============================================================

def plot_parameter_panels(
    df: pd.DataFrame,
    plot_specs: list[tuple[str, str]],
    title: str,
    ylabel: str,
    out_name: str,
    ncols: int = 2,
) -> None:
    available = [(col, label) for col, label in plot_specs if col in df.columns]

    if not available:
        print("[skip] no columns found for", title)
        return

    x = get_cycle_x(df)

    n = len(available)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.5, max(3.0 * nrows, 4.5)),
        sharex=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, (col, label) in zip(axes, available):
        y = make_numeric(df, col)
        plot_line(ax, x, y)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        clean_axis(ax)

    for ax in axes[len(available):]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        if ax.has_data():
            ax.set_xlabel("Retained cycle index", fontsize=AXIS_FONTSIZE)

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=0.995)

    savefig(THESIS_FIG_DIR / out_name)


# ============================================================
# Plot 4 — Core Parameters
# ============================================================

def plot_core_parameters(best: pd.DataFrame) -> None:
    specs = [
        ("alpha_n", r"$\alpha_n$"),
        ("alpha_n_hat", r"$\alpha_n$"),
        ("best_alpha_n", r"$\alpha_n$"),

        ("alpha_p", r"$\alpha_p$"),
        ("alpha_p_hat", r"$\alpha_p$"),
        ("best_alpha_p", r"$\alpha_p$"),

        ("g_n", r"$g_n$"),
        ("g_n_hat", r"$g_n$"),
        ("best_g_n", r"$g_n$"),

        ("g_p", r"$g_p$"),
        ("g_p_hat", r"$g_p$"),
        ("best_g_p", r"$g_p$"),

        ("b_en", r"$b_{e,n}$"),
        ("b_en_hat", r"$b_{e,n}$"),
        ("best_b_en", r"$b_{e,n}$"),

        ("b_ep", r"$b_{e,p}$"),
        ("b_ep_hat", r"$b_{e,p}$"),
        ("best_b_ep", r"$b_{e,p}$"),
    ]

    # Remove duplicate labels by keeping the first available version.
    picked = []
    used_labels = set()

    for col, label in specs:
        if col in best.columns and label not in used_labels:
            picked.append((col, label))
            used_labels.add(label)

    plot_parameter_panels(
        best,
        picked,
        "Core Parameters",
        "Parameter value",
        "fig_ch6_core_parameters.png",
        ncols=2,
    )


# ============================================================
# Plot 5 — Electrolyte K Parameters
# ============================================================

def plot_k_parameters(best: pd.DataFrame) -> None:
    candidate_cols = []

    for j in range(1, 10):
        candidate_cols.extend(
            [
                (f"k{j}", rf"$k_{j}$"),
                (f"k{j}_hat", rf"$k_{j}$"),
                (f"best_k{j}", rf"$k_{j}$"),
            ]
        )

    picked = []
    used_labels = set()

    for col, label in candidate_cols:
        if col in best.columns and label not in used_labels:
            picked.append((col, label))
            used_labels.add(label)

    # Fallback for old names.
    fallback = [
        ("k12", r"$k_1$"),
        ("k12_hat", r"$k_1$"),
        ("k23", r"$k_2$"),
        ("k23_hat", r"$k_2$"),
    ]

    for col, label in fallback:
        if col in best.columns and label not in used_labels:
            picked.append((col, label))
            used_labels.add(label)

    plot_parameter_panels(
        best,
        picked,
        "Electrolyte K Parameters",
        "Parameter value",
        "fig_ch6_electrolyte_k_parameters.png",
        ncols=2,
    )


# ============================================================
# Plot 6 — Beta Coefficients
# ============================================================

def plot_beta_coefficients(best: pd.DataFrame) -> None:
    preferred = [
        ("beta_C", r"$C$"),
        ("beta_xppow2", r"$a_{p,2}$"),
        ("beta_xppow3", r"$a_{p,3}$"),
        ("beta_xppow4", r"$a_{p,4}$"),
        ("beta_minusxnpow2", r"$a_{n,2}$"),
        ("beta_minusxnpow3", r"$a_{n,3}$"),
        ("beta_minusxnpow4", r"$a_{n,4}$"),
        ("beta_I", r"$D_1$"),
        ("beta_zepow2", r"$E_2$"),
        ("beta_zepow3", r"$E_3$"),
        ("beta_zepow4", r"$E_4$"),
    ]

    picked = [(col, label) for col, label in preferred if col in best.columns]

    if not picked:
        beta_cols = [c for c in best.columns if str(c).startswith("beta_")]
        picked = [(c, c.replace("beta_", "").replace("_", r"\_")) for c in beta_cols]

    plot_parameter_panels(
        best,
        picked,
        "Beta Coefficients",
        "Coefficient value",
        "fig_ch6_beta_coefficients.png",
        ncols=2,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("CLEAN CHAPTER 6 DIAGNOSTIC PLOTS")
    print("=" * 100)
    print("REAL_PROJECT:", REAL_PROJECT)
    print("THESIS_FIG_DIR:", THESIS_FIG_DIR)
    print("=" * 100)

    summary, best = load_tables()

    plot_good_fit_mask(summary)
    plot_best_rmse(summary)
    plot_bfr(summary)

    plot_core_parameters(best)
    plot_k_parameters(best)
    plot_beta_coefficients(best)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    for p in sorted(THESIS_FIG_DIR.glob("fig_ch6_*.png")):
        print(" ", p)
    print("=" * 100)


if __name__ == "__main__":
    main()
