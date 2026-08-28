#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_anchor6_model_complexity_summary.py

Purpose
-------
Combine six-model anchor real-data CT-ID screening results over cycles 34--99.

Anchor models:
    S7_C1
    S7_C4
    S7_C4K
    S17_C1
    S17_C4
    S17_C4K

This script does NOT run identification.
It reads saved outputs from:
    results/real_cycle_ctid_state_order_grid/
    results/real_warm_continuation_ctid/

and creates summary CSVs plus thesis-ready visuals.
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

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "anchor_model_screening_6models"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "anchor_model_screening_6models"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


CYCLES = list(range(34, 100))

ANCHOR_MODELS = [
    ("S7", "C1", "S7_C1"),
    ("S7", "C4", "S7_C4"),
    ("S7", "C4K", "S7_C4K"),
    ("S17", "C1", "S17_C1"),
    ("S17", "C4", "S17_C4"),
    ("S17", "C4K", "S17_C4K"),
]

STATE_ORDER = ["S7", "S17"]
CANDIDATE_ORDER = ["C1", "C4", "C4K"]
MODEL_ORDER = [m[2] for m in ANCHOR_MODELS]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[saved]", path)

    # Copy to thesis folders
    shutil.copy2(path, THESIS_FIG_DIR / path.name)
    shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def parse_anchor_tag(path: Path) -> dict | None:
    s = str(path)

    m = re.search(
        r"anchor6_(S7|S17)_(C1|C4|C4K)_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
        s,
    )

    if not m:
        return None

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


def standardize_summary_row(df: pd.DataFrame, meta: dict, source_file: Path) -> dict | None:
    if df is None or len(df) == 0:
        return None

    row = df.iloc[0].copy()

    out = dict(meta)
    out["source_file"] = str(source_file)
    out["source_folder"] = str(source_file.parent)

    # Prefer explicit columns if available.
    for col in ["model_id", "state_id", "candidate_id", "cycle_index"]:
        if col in row.index and pd.notna(row[col]):
            out[col] = row[col]

    out["state_id"] = str(out["state_id"])
    out["candidate_id"] = str(out["candidate_id"])
    out["model_id"] = f"{out['state_id']}_{out['candidate_id']}"
    out["cycle_index"] = int(out["cycle_index"])

    # Metrics: tolerate different naming.
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

    for folder in sorted(GRID_ROOT.glob("anchor6_*")):
        if not folder.is_dir():
            continue

        meta = parse_anchor_tag(folder)
        if meta is None:
            continue

        # Grid script summary name.
        candidates = [
            folder / "real_cycle_model_summary.csv",
            folder / "summary.csv",
        ]

        for p in candidates:
            if p.exists():
                df = safe_read_csv(p)
                row = standardize_summary_row(df, meta, p)
                if row:
                    rows.append(row)
                break

    return rows


def collect_kparam_rows() -> list[dict]:
    rows = []

    if not KPARAM_ROOT.exists():
        return rows

    for folder in sorted(KPARAM_ROOT.glob("S*_C4K/anchor6_*")):
        if not folder.is_dir():
            continue

        meta = parse_anchor_tag(folder)
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
        raise RuntimeError("No anchor6 result rows found yet. Wait for sbatch jobs or check run tags.")

    df = pd.DataFrame(rows)

    # Drop duplicate reruns by keeping the best RMSE for each model-cycle.
    df = df.sort_values(["model_id", "cycle_index", "best_rmse"], na_position="last")
    df = df.drop_duplicates(subset=["model_id", "cycle_index"], keep="first").reset_index(drop=True)

    df["state_id"] = df["state_id"].astype(str)
    df["candidate_id"] = df["candidate_id"].astype(str)
    df["model_id"] = df["model_id"].astype(str)
    df["cycle_index"] = df["cycle_index"].astype(int)

    df["state_sort"] = df["state_id"].map({s: i for i, s in enumerate(STATE_ORDER)})
    df["candidate_sort"] = df["candidate_id"].map({c: i for i, c in enumerate(CANDIDATE_ORDER)})
    df["model_sort"] = df["model_id"].map({m: i for i, m in enumerate(MODEL_ORDER)})

    df = df.sort_values(["model_sort", "cycle_index"]).reset_index(drop=True)

    return df


def make_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    expected = []

    for state_id, candidate_id, model_id in ANCHOR_MODELS:
        for cycle in CYCLES:
            expected.append(
                {
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "cycle_index": cycle,
                }
            )

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
            q25_best_rmse=("best_rmse", lambda x: np.nanquantile(x, 0.25)),
            q75_best_rmse=("best_rmse", lambda x: np.nanquantile(x, 0.75)),
            mean_best_rmse_mV=("best_rmse_mV", "mean"),
            median_best_rmse_mV=("best_rmse_mV", "median"),
            std_best_rmse_mV=("best_rmse_mV", "std"),
            mean_bfr_percent=("best_bfr_percent", "mean"),
            median_bfr_percent=("best_bfr_percent", "median"),
            mean_r2_percent=("best_r2_percent", "mean"),
            median_r2_percent=("best_r2_percent", "median"),
        )
    )

    summary["state_sort"] = summary["state_id"].map({s: i for i, s in enumerate(STATE_ORDER)})
    summary["candidate_sort"] = summary["candidate_id"].map({c: i for i, c in enumerate(CANDIDATE_ORDER)})
    summary = summary.sort_values(["state_sort", "candidate_sort"]).reset_index(drop=True)

    return summary


def plot_heatmap(summary: pd.DataFrame, metric_col: str, title: str, filename: str) -> None:
    mat = np.full((len(STATE_ORDER), len(CANDIDATE_ORDER)), np.nan)

    for _, row in summary.iterrows():
        i = STATE_ORDER.index(row["state_id"])
        j = CANDIDATE_ORDER.index(row["candidate_id"])
        mat[i, j] = row[metric_col]

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(CANDIDATE_ORDER)))
    ax.set_yticks(np.arange(len(STATE_ORDER)))
    ax.set_xticklabels(CANDIDATE_ORDER, fontsize=12)
    ax.set_yticklabels(STATE_ORDER, fontsize=12)

    ax.set_xlabel("Output model", fontsize=13)
    ax.set_ylabel("State model", fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE [mV]", fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / filename)


def plot_line_by_cycle(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.5))

    for model_id in MODEL_ORDER:
        d = df[df["model_id"] == model_id].sort_values("cycle_index")
        if len(d) == 0:
            continue

        ax.plot(
            d["cycle_index"],
            d["best_rmse_mV"],
            marker="o",
            linewidth=1.8,
            markersize=3.8,
            label=model_id,
        )

    ax.set_title("Anchor Model Screening Across Retained Cycles", fontsize=15, pad=12)
    ax.set_xlabel("Original cycle index", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9, ncols=2)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / "line_best_rmse_by_cycle_anchor_6models.png")


def plot_box_by_model(df: pd.DataFrame) -> None:
    data = []
    labels = []

    for model_id in MODEL_ORDER:
        vals = pd.to_numeric(df[df["model_id"] == model_id]["best_rmse_mV"], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(model_id)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Anchor Model RMSE Distributions", fontsize=15, pad=12)
    ax.set_xlabel("Anchor model", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / "box_rmse_by_model_anchor_6models.png")


def main() -> None:
    print("=" * 100)
    print("ANCHOR SIX-MODEL MODEL-COMPLEXITY SUMMARY")
    print("=" * 100)
    print("PROJECT:", PROJECT)
    print("GRID_ROOT:", GRID_ROOT)
    print("KPARAM_ROOT:", KPARAM_ROOT)
    print("OUT_TABLE_DIR:", OUT_TABLE_DIR)
    print("OUT_FIG_DIR:", OUT_FIG_DIR)
    print("=" * 100)

    df = collect_all_rows()

    long_path = OUT_TABLE_DIR / "anchor_cycle_metrics_long.csv"
    df.to_csv(long_path, index=False)
    print("[saved]", long_path)

    missing = make_missing_table(df)
    missing_path = OUT_TABLE_DIR / "anchor_missing_models_cycles.csv"
    missing.to_csv(missing_path, index=False)
    print("[saved]", missing_path)

    summary = make_model_summary(df)
    summary_path = OUT_TABLE_DIR / "model_complexity_summary_anchor.csv"
    summary.to_csv(summary_path, index=False)
    print("[saved]", summary_path)

    print()
    print("Model summary:")
    print(summary[["model_id", "n_cycles", "mean_best_rmse_mV", "median_best_rmse_mV", "mean_bfr_percent", "mean_r2_percent"]].to_string(index=False))

    print()
    print("Missing results:", len(missing))
    if len(missing):
        print(missing.head(30).to_string(index=False))

    plot_heatmap(
        summary,
        metric_col="mean_best_rmse_mV",
        title="Mean Best RMSE Across Retained Cycles",
        filename="heatmap_mean_rmse_anchor_6models.png",
    )

    plot_heatmap(
        summary,
        metric_col="median_best_rmse_mV",
        title="Median Best RMSE Across Retained Cycles",
        filename="heatmap_median_rmse_anchor_6models.png",
    )

    plot_line_by_cycle(df)
    plot_box_by_model(df)

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
