#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_full16_model_complexity_heatmaps.py

Purpose
-------
Create final Chapter 6 16-model heatmaps across cycles 34--99.

Final heatmap models:

    S7:  C1, C2, C3, C4K
    S12: C1, C2, C3, C4
    S14: C1, C2, C3, C4
    S17: C1, C2, C3, C4K

This uses C4K instead of C4 for S7 and S17 because the anchor screen showed
that C4K had lower mean RMSE than C4 for both S7 and S17.

Inputs:
    results/real_cycle_ctid_state_order_grid/anchor6_*/
    results/real_warm_continuation_ctid/S7_C4K/anchor6_*/
    results/real_warm_continuation_ctid/S17_C4K/anchor6_*/
    results/real_cycle_ctid_state_order_grid/full16rem_*/

Outputs:
    results/tables/full16_model_complexity/
    results/figures/full16_model_complexity/
    figures/chapter6/
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

GRID_ROOT = PROJECT / "results" / "real_cycle_ctid_state_order_grid"
KPARAM_ROOT = PROJECT / "results" / "real_warm_continuation_ctid"

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "full16_model_complexity"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "full16_model_complexity"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


CYCLES = list(range(34, 100))

STATE_ORDER = ["S7", "S12", "S14", "S17"]
DISPLAY_ORDER = ["C1", "C2", "C3", "C4/C4K"]

FINAL_MODEL_MAP = {
    ("S7", "C1"): "S7_C1",
    ("S7", "C2"): "S7_C2",
    ("S7", "C3"): "S7_C3",
    ("S7", "C4/C4K"): "S7_C4K",

    ("S12", "C1"): "S12_C1",
    ("S12", "C2"): "S12_C2",
    ("S12", "C3"): "S12_C3",
    ("S12", "C4/C4K"): "S12_C4",

    ("S14", "C1"): "S14_C1",
    ("S14", "C2"): "S14_C2",
    ("S14", "C3"): "S14_C3",
    ("S14", "C4/C4K"): "S14_C4",

    ("S17", "C1"): "S17_C1",
    ("S17", "C2"): "S17_C2",
    ("S17", "C3"): "S17_C3",
    ("S17", "C4/C4K"): "S17_C4K",
}

FINAL_MODEL_IDS = list(FINAL_MODEL_MAP.values())


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[saved]", path)

    shutil.copy2(path, THESIS_FIG_DIR / path.name)
    shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def parse_grid_tag(path: Path) -> dict | None:
    s = str(path)

    patterns = [
        r"anchor6_(S7|S12|S14|S17)_(C1|C2|C3|C4)_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
        r"full16rem_(S7|S12|S14|S17)_(C1|C2|C3|C4)_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if m:
            state_id, candidate_id, nseeds, cycle, seed0, dt = m.groups()
            return {
                "state_id": state_id,
                "candidate_id": candidate_id,
                "model_id": f"{state_id}_{candidate_id}",
                "n_multistart": int(nseeds),
                "cycle_index": int(cycle),
                "seed0": int(seed0),
                "id_downsample_dt": float(dt),
            }

    return None


def parse_kparam_tag(path: Path) -> dict | None:
    s = str(path)

    m = re.search(
        r"anchor6_(S7|S17)_C4K_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
        s,
    )

    if not m:
        return None

    state_id, nseeds, cycle, seed0, dt = m.groups()

    return {
        "state_id": state_id,
        "candidate_id": "C4K",
        "model_id": f"{state_id}_C4K",
        "n_multistart": int(nseeds),
        "cycle_index": int(cycle),
        "seed0": int(seed0),
        "id_downsample_dt": float(dt),
    }


def standardize_summary_row(df: pd.DataFrame, meta: dict, source_file: Path) -> dict | None:
    if df is None or len(df) == 0:
        return None

    row = df.iloc[0].copy()

    out = dict(meta)
    out["source_file"] = str(source_file)
    out["source_folder"] = str(source_file.parent)

    for col in ["model_id", "state_id", "candidate_id", "cycle_index"]:
        if col in row.index and pd.notna(row[col]):
            out[col] = row[col]

    out["state_id"] = str(out["state_id"])
    out["candidate_id"] = str(out["candidate_id"])
    out["model_id"] = f"{out['state_id']}_{out['candidate_id']}"
    out["cycle_index"] = int(out["cycle_index"])

    metric_map = {
        "best_rmse": ["best_rmse", "rmse"],
        "best_mae": ["best_mae", "mae"],
        "best_r2_percent": ["best_r2_percent", "r2_percent"],
        "best_bfr_percent": ["best_bfr_percent", "bfr_percent"],
        "median_rmse": ["median_rmse"],
        "mean_rmse": ["mean_rmse"],
        "std_rmse": ["std_rmse"],
        "n_success": ["n_success"],
        "n_fail": ["n_fail"],
    }

    for out_col, candidates in metric_map.items():
        out[out_col] = np.nan
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                out[out_col] = float(row[c])
                break

    out["best_rmse_mV"] = 1000.0 * out["best_rmse"] if np.isfinite(out["best_rmse"]) else np.nan
    out["best_mae_mV"] = 1000.0 * out["best_mae"] if np.isfinite(out["best_mae"]) else np.nan
    out["median_rmse_mV"] = 1000.0 * out["median_rmse"] if np.isfinite(out["median_rmse"]) else np.nan
    out["mean_rmse_mV"] = 1000.0 * out["mean_rmse"] if np.isfinite(out["mean_rmse"]) else np.nan

    return out


def collect_grid_rows() -> list[dict]:
    rows = []

    if not GRID_ROOT.exists():
        return rows

    for folder in sorted(list(GRID_ROOT.glob("anchor6_*")) + list(GRID_ROOT.glob("full16rem_*"))):
        if not folder.is_dir():
            continue

        meta = parse_grid_tag(folder)
        if meta is None:
            continue

        p = folder / "real_cycle_model_summary.csv"
        if p.exists():
            df = safe_read_csv(p)
            row = standardize_summary_row(df, meta, p)
            if row:
                rows.append(row)

    return rows


def collect_kparam_rows() -> list[dict]:
    rows = []

    if not KPARAM_ROOT.exists():
        return rows

    for folder in sorted(KPARAM_ROOT.glob("S*_C4K/anchor6_*")):
        if not folder.is_dir():
            continue

        meta = parse_kparam_tag(folder)
        if meta is None:
            continue

        p = folder / "summary.csv"
        if p.exists():
            df = safe_read_csv(p)
            row = standardize_summary_row(df, meta, p)
            if row:
                rows.append(row)

    return rows


def collect_all_rows() -> pd.DataFrame:
    rows = collect_grid_rows() + collect_kparam_rows()

    if not rows:
        raise RuntimeError("No result rows found.")

    df = pd.DataFrame(rows)

    df = df[df["model_id"].isin(FINAL_MODEL_IDS)].copy()

    # Keep best if accidentally rerun.
    df = df.sort_values(["model_id", "cycle_index", "best_rmse"], na_position="last")
    df = df.drop_duplicates(subset=["model_id", "cycle_index"], keep="first").reset_index(drop=True)

    df["cycle_index"] = df["cycle_index"].astype(int)

    return df


def make_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    expected = []

    for model_id in FINAL_MODEL_IDS:
        for cycle in CYCLES:
            expected.append({"model_id": model_id, "cycle_index": cycle})

    expected = pd.DataFrame(expected)

    have = df[["model_id", "cycle_index"]].copy()
    have["has_result"] = True

    merged = expected.merge(have, on=["model_id", "cycle_index"], how="left")
    merged["has_result"] = merged["has_result"].fillna(False)

    missing = merged[~merged["has_result"]].copy()
    return missing


def make_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["state_id", "candidate_id", "model_id"], as_index=False)
        .agg(
            n_cycles=("cycle_index", "nunique"),
            mean_best_rmse=("best_rmse", "mean"),
            median_best_rmse=("best_rmse", "median"),
            std_best_rmse=("best_rmse", "std"),
            min_best_rmse=("best_rmse", "min"),
            max_best_rmse=("best_rmse", "max"),
            q25_best_rmse=("best_rmse", lambda x: np.nanquantile(x, 0.25)),
            q75_best_rmse=("best_rmse", lambda x: np.nanquantile(x, 0.75)),
            mean_best_rmse_mV=("best_rmse_mV", "mean"),
            median_best_rmse_mV=("best_rmse_mV", "median"),
            std_best_rmse_mV=("best_rmse_mV", "std"),
            min_best_rmse_mV=("best_rmse_mV", "min"),
            max_best_rmse_mV=("best_rmse_mV", "max"),
            mean_bfr_percent=("best_bfr_percent", "mean"),
            median_bfr_percent=("best_bfr_percent", "median"),
            mean_r2_percent=("best_r2_percent", "mean"),
            median_r2_percent=("best_r2_percent", "median"),
        )
    )

    display_rows = []
    for state in STATE_ORDER:
        for col in DISPLAY_ORDER:
            model_id = FINAL_MODEL_MAP[(state, col)]
            d = summary[summary["model_id"] == model_id].copy()
            if len(d):
                row = d.iloc[0].to_dict()
                row["display_state"] = state
                row["display_order"] = col
                row["display_model_id"] = model_id
                row["state_sort"] = STATE_ORDER.index(state)
                row["order_sort"] = DISPLAY_ORDER.index(col)
                display_rows.append(row)

    out = pd.DataFrame(display_rows)
    out = out.sort_values(["state_sort", "order_sort"]).reset_index(drop=True)
    return out


def plot_heatmap(summary: pd.DataFrame, metric_col: str, title: str, cbar_label: str, filename: str) -> None:
    mat = np.full((len(STATE_ORDER), len(DISPLAY_ORDER)), np.nan)
    labels = [["" for _ in DISPLAY_ORDER] for _ in STATE_ORDER]

    for _, row in summary.iterrows():
        i = STATE_ORDER.index(row["display_state"])
        j = DISPLAY_ORDER.index(row["display_order"])

        mat[i, j] = row[metric_col]
        labels[i][j] = row["display_model_id"]

    fig, ax = plt.subplots(figsize=(9.6, 7.0))

    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(DISPLAY_ORDER)))
    ax.set_yticks(np.arange(len(STATE_ORDER)))
    ax.set_xticklabels(DISPLAY_ORDER, fontsize=12)
    ax.set_yticklabels(STATE_ORDER, fontsize=12)

    ax.set_xlabel("Voltage/output model order", fontsize=13)
    ax.set_ylabel("State model", fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{labels[i][j]}\n{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    fig.tight_layout()
    savefig(OUT_FIG_DIR / filename)


def plot_line_by_cycle(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.0))

    ordered_models = summary["display_model_id"].tolist()

    for model_id in ordered_models:
        d = df[df["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        ax.plot(
            d["cycle_index"],
            d["best_rmse_mV"],
            linewidth=1.5,
            marker="o",
            markersize=2.8,
            label=model_id,
        )

    ax.set_title("Final 16-Model Screening Across Retained Cycles", fontsize=15, pad=12)
    ax.set_xlabel("Original cycle index", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncols=4)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / "line_best_rmse_by_cycle_full16.png")


def plot_box_by_model(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ordered_models = summary["display_model_id"].tolist()

    data = []
    labels = []

    for model_id in ordered_models:
        vals = pd.to_numeric(df[df["model_id"] == model_id]["best_rmse_mV"], errors="coerce").dropna().to_numpy()
        if len(vals):
            data.append(vals)
            labels.append(model_id)

    fig, ax = plt.subplots(figsize=(14.0, 6.5))

    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)

    ax.set_title("Final 16-Model RMSE Distributions", fontsize=15, pad=12)
    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / "box_rmse_by_model_full16.png")


def main() -> None:
    print("=" * 100)
    print("FINAL 16-MODEL MODEL-COMPLEXITY HEATMAPS")
    print("=" * 100)
    print("PROJECT:", PROJECT)
    print("GRID_ROOT:", GRID_ROOT)
    print("KPARAM_ROOT:", KPARAM_ROOT)
    print("OUT_TABLE_DIR:", OUT_TABLE_DIR)
    print("OUT_FIG_DIR:", OUT_FIG_DIR)
    print("=" * 100)

    df = collect_all_rows()

    long_path = OUT_TABLE_DIR / "full16_cycle_metrics_long.csv"
    df.to_csv(long_path, index=False)
    print("[saved]", long_path)

    missing = make_missing_table(df)
    missing_path = OUT_TABLE_DIR / "full16_missing_models_cycles.csv"
    missing.to_csv(missing_path, index=False)
    print("[saved]", missing_path)

    summary = make_model_summary(df)
    summary_path = OUT_TABLE_DIR / "model_complexity_summary_full16.csv"
    summary.to_csv(summary_path, index=False)
    print("[saved]", summary_path)

    print()
    print("Model summary:")
    print(
        summary[
            [
                "display_model_id",
                "n_cycles",
                "mean_best_rmse_mV",
                "median_best_rmse_mV",
                "std_best_rmse_mV",
                "mean_bfr_percent",
                "mean_r2_percent",
            ]
        ].to_string(index=False)
    )

    print()
    print("Missing results:", len(missing))
    if len(missing):
        print(missing.head(40).to_string(index=False))

    plot_heatmap(
        summary,
        metric_col="mean_best_rmse_mV",
        title="Mean Best RMSE Across Retained Cycles",
        cbar_label="Mean best RMSE [mV]",
        filename="heatmap_mean_best_rmse_full16.png",
    )

    plot_heatmap(
        summary,
        metric_col="median_best_rmse_mV",
        title="Median Best RMSE Across Retained Cycles",
        cbar_label="Median best RMSE [mV]",
        filename="heatmap_median_best_rmse_full16.png",
    )

    plot_heatmap(
        summary,
        metric_col="std_best_rmse_mV",
        title="Standard Deviation of Best RMSE Across Retained Cycles",
        cbar_label="Std. best RMSE [mV]",
        filename="heatmap_std_best_rmse_full16.png",
    )

    plot_heatmap(
        summary,
        metric_col="mean_bfr_percent",
        title="Mean BFR Across Retained Cycles",
        cbar_label="Mean BFR [%]",
        filename="heatmap_mean_bfr_full16.png",
    )

    plot_heatmap(
        summary,
        metric_col="mean_r2_percent",
        title="Mean R2 Across Retained Cycles",
        cbar_label="Mean R2 [%]",
        filename="heatmap_mean_r2_full16.png",
    )

    plot_line_by_cycle(df, summary)
    plot_box_by_model(df, summary)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Main thesis figures copied to:")
    print(" ", THESIS_FIG_DIR)
    print(" ", FLOW_THESIS_FIG_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
