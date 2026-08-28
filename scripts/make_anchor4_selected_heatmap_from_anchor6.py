#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_anchor4_selected_heatmap_from_anchor6.py

Purpose
-------
Use the six-model anchor screening results but visualize only four cells:

    S7_C1
    best of S7_C4 and S7_C4K

    S17_C1
    best of S17_C4 and S17_C4K

The better quartic-family model is selected separately for S7 and S17 using
mean_best_rmse_mV by default.

Input:
    results/tables/anchor_model_screening_6models/model_complexity_summary_anchor.csv
    results/tables/anchor_model_screening_6models/anchor_cycle_metrics_long.csv

Outputs:
    results/tables/anchor_model_screening_6models/model_complexity_summary_selected4.csv
    results/tables/anchor_model_screening_6models/selected_c4_family_decision.csv

    figures/chapter6/heatmap_mean_rmse_anchor_selected4.png
    figures/chapter6/heatmap_median_rmse_anchor_selected4.png
    figures/chapter6/box_rmse_by_model_anchor_selected4.png
    figures/chapter6/line_best_rmse_by_cycle_anchor_selected4.png
"""

from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

TABLE_DIR = PROJECT / "results" / "tables" / "anchor_model_screening_6models"
FIG_DIR = PROJECT / "results" / "figures" / "anchor_model_screening_6models"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [TABLE_DIR, FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


STATE_ORDER = ["S7", "S17"]
SELECTED_COLS = ["C1", "Selected C4/C4K"]

SELECTION_METRIC = "mean_best_rmse_mV"


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[saved]", path)

    shutil.copy2(path, THESIS_FIG_DIR / path.name)
    shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = TABLE_DIR / "model_complexity_summary_anchor.csv"
    long_path = TABLE_DIR / "anchor_cycle_metrics_long.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run scripts/make_anchor6_model_complexity_summary.py first."
        )

    if not long_path.exists():
        raise FileNotFoundError(
            f"Missing {long_path}. Run scripts/make_anchor6_model_complexity_summary.py first."
        )

    summary = pd.read_csv(summary_path)
    long = pd.read_csv(long_path)

    return summary, long


def choose_selected_models(summary: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    selected_models = []
    decision_rows = []

    for state in STATE_ORDER:
        # Always keep C1.
        c1_model = f"{state}_C1"
        selected_models.append(c1_model)

        # Choose better between C4 and C4K.
        candidates = [f"{state}_C4", f"{state}_C4K"]

        d = summary[summary["model_id"].isin(candidates)].copy()

        if len(d) == 0:
            print(f"[warning] No C4/C4K candidates found for {state}")
            continue

        d = d.dropna(subset=[SELECTION_METRIC]).copy()

        if len(d) == 0:
            print(f"[warning] No finite {SELECTION_METRIC} for {state}")
            continue

        d = d.sort_values(SELECTION_METRIC).reset_index(drop=True)
        chosen = str(d.iloc[0]["model_id"])
        selected_models.append(chosen)

        decision_rows.append(
            {
                "state_id": state,
                "selected_model": chosen,
                "selection_metric": SELECTION_METRIC,
                "selected_metric_value": float(d.iloc[0][SELECTION_METRIC]),
                "other_candidate": str(d.iloc[1]["model_id"]) if len(d) > 1 else "",
                "other_metric_value": float(d.iloc[1][SELECTION_METRIC]) if len(d) > 1 else np.nan,
                "reason": f"Lowest {SELECTION_METRIC} among {state}_C4 and {state}_C4K.",
            }
        )

    decision = pd.DataFrame(decision_rows)
    return selected_models, decision


def make_selected_summary(summary: pd.DataFrame, selected_models: list[str]) -> pd.DataFrame:
    out = summary[summary["model_id"].isin(selected_models)].copy()

    selected_col = []

    for _, row in out.iterrows():
        if row["candidate_id"] == "C1":
            selected_col.append("C1")
        else:
            selected_col.append("Selected C4/C4K")

    out["selected_heatmap_col"] = selected_col
    out["selected_heatmap_row"] = out["state_id"].astype(str)

    out["row_sort"] = out["selected_heatmap_row"].map({s: i for i, s in enumerate(STATE_ORDER)})
    out["col_sort"] = out["selected_heatmap_col"].map({c: i for i, c in enumerate(SELECTED_COLS)})

    out = out.sort_values(["row_sort", "col_sort"]).reset_index(drop=True)
    return out


def plot_selected_heatmap(selected_summary: pd.DataFrame, metric_col: str, title: str, filename: str) -> None:
    mat = np.full((len(STATE_ORDER), len(SELECTED_COLS)), np.nan)
    labels = [["", ""], ["", ""]]

    for _, row in selected_summary.iterrows():
        i = STATE_ORDER.index(row["selected_heatmap_row"])
        j = SELECTED_COLS.index(row["selected_heatmap_col"])
        mat[i, j] = row[metric_col]
        labels[i][j] = str(row["model_id"])

    fig, ax = plt.subplots(figsize=(7.8, 5.6))

    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(SELECTED_COLS)))
    ax.set_yticks(np.arange(len(STATE_ORDER)))
    ax.set_xticklabels(SELECTED_COLS, fontsize=12)
    ax.set_yticklabels(STATE_ORDER, fontsize=12)

    ax.set_xlabel("Output/electrolyte formulation", fontsize=13)
    ax.set_ylabel("State model", fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE [mV]", fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{labels[i][j]}\n{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                )

    fig.tight_layout()
    savefig(FIG_DIR / filename)


def plot_selected_box(long: pd.DataFrame, selected_models: list[str]) -> None:
    d = long[long["model_id"].isin(selected_models)].copy()

    order = [m for m in ["S7_C1", "S7_C4", "S7_C4K", "S17_C1", "S17_C4", "S17_C4K"] if m in selected_models]

    data = []
    labels = []

    for m in order:
        vals = pd.to_numeric(d[d["model_id"] == m]["best_rmse_mV"], errors="coerce").dropna().to_numpy()
        if len(vals):
            data.append(vals)
            labels.append(m)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Selected Anchor Model RMSE Distributions", fontsize=15, pad=12)
    ax.set_xlabel("Selected anchor model", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    savefig(FIG_DIR / "box_rmse_by_model_anchor_selected4.png")


def plot_selected_lines(long: pd.DataFrame, selected_models: list[str]) -> None:
    d = long[long["model_id"].isin(selected_models)].copy()

    order = [m for m in ["S7_C1", "S7_C4", "S7_C4K", "S17_C1", "S17_C4", "S17_C4K"] if m in selected_models]

    fig, ax = plt.subplots(figsize=(12.5, 6.5))

    for m in order:
        g = d[d["model_id"] == m].sort_values("cycle_index")
        if len(g) == 0:
            continue

        ax.plot(
            g["cycle_index"],
            g["best_rmse_mV"],
            marker="o",
            linewidth=1.8,
            markersize=3.8,
            label=m,
        )

    ax.set_title("Selected Anchor Models Across Retained Cycles", fontsize=15, pad=12)
    ax.set_xlabel("Original cycle index", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9, ncols=2)

    fig.tight_layout()
    savefig(FIG_DIR / "line_best_rmse_by_cycle_anchor_selected4.png")


def main() -> None:
    print("=" * 100)
    print("SELECTED FOUR-CELL ANCHOR HEATMAP FROM SIX-MODEL SCREEN")
    print("=" * 100)
    print("TABLE_DIR:", TABLE_DIR)
    print("FIG_DIR:", FIG_DIR)
    print("SELECTION_METRIC:", SELECTION_METRIC)
    print("=" * 100)

    summary, long = load_inputs()

    selected_models, decision = choose_selected_models(summary)
    selected_summary = make_selected_summary(summary, selected_models)

    decision_path = TABLE_DIR / "selected_c4_family_decision.csv"
    selected_summary_path = TABLE_DIR / "model_complexity_summary_selected4.csv"

    decision.to_csv(decision_path, index=False)
    selected_summary.to_csv(selected_summary_path, index=False)

    print("[saved]", decision_path)
    print("[saved]", selected_summary_path)

    print()
    print("Decision:")
    print(decision.to_string(index=False))

    print()
    print("Selected models:")
    print(selected_summary[["model_id", "n_cycles", "mean_best_rmse_mV", "median_best_rmse_mV", "mean_bfr_percent", "mean_r2_percent"]].to_string(index=False))

    plot_selected_heatmap(
        selected_summary,
        metric_col="mean_best_rmse_mV",
        title="Mean Best RMSE Across Retained Cycles",
        filename="heatmap_mean_rmse_anchor_selected4.png",
    )

    plot_selected_heatmap(
        selected_summary,
        metric_col="median_best_rmse_mV",
        title="Median Best RMSE Across Retained Cycles",
        filename="heatmap_median_rmse_anchor_selected4.png",
    )

    plot_selected_box(long, selected_models)
    plot_selected_lines(long, selected_models)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Thesis figures:")
    print(" ", THESIS_FIG_DIR / "heatmap_mean_rmse_anchor_selected4.png")
    print(" ", THESIS_FIG_DIR / "heatmap_median_rmse_anchor_selected4.png")
    print(" ", THESIS_FIG_DIR / "box_rmse_by_model_anchor_selected4.png")
    print(" ", THESIS_FIG_DIR / "line_best_rmse_by_cycle_anchor_selected4.png")
    print("=" * 100)


if __name__ == "__main__":
    main()
