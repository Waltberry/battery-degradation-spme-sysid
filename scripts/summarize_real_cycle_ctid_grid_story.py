#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% =====================================================
# CELL 0 — Imports
# =====================================================
"""
summarize_real_cycle_ctid_grid_story.py

Purpose
-------
Create combined story-level summaries and plots for the real-data CT-ID grid.

This script reads all completed real-cycle fitting outputs from:

    results/real_cycle_ctid_state_order_grid/

The fitting script is expected to have produced folders containing:

    real_cycle_all_runs.csv
    real_cycle_best_runs.csv
    real_cycle_model_summary.csv
    real_cycle_beta_coefficients.csv
    real_cycle_parameter_long.csv
    selected_real_cycle_id_data.csv
    real_cycle_config.json

This post-processing script creates:

    1. completion audit
    2. combined all-runs table
    3. combined best-run table
    4. combined model summary
    5. best RMSE heatmap: state variant x output order
    6. median RMSE heatmap: state variant x output order
    7. RMSE scatter plot for all 16 models
    8. RMSE boxplot by model
    9. RMSE boxplots grouped by state variant
    10. RMSE boxplots grouped by output order
    11. rank/condition heatmaps
    12. 1000-bin parameter histograms
    13. measured step response vs estimated step response for best models
    14. measured-vs-estimated overlay for the overall best model
    15. residual plots for the overall best model

Important
---------
There is no known true parameter for real data.
Therefore parameter histograms show:

    mean
    median
    best-RMSE estimate

No true-value line is drawn unless manually supplied.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# %% =====================================================
# CELL 1 — Paths and settings
# =====================================================
PROJECT_DIR = Path.cwd()

RESULT_ROOT = PROJECT_DIR / "results" / "real_cycle_ctid_state_order_grid"
LOG_ROOT = PROJECT_DIR / "results" / "logs"

OUT_DIR = RESULT_ROOT / "_combined_real_cycle_story"
FIG_DIR = PROJECT_DIR / "results" / "figures" / "real_cycle_ctid_state_order_grid" / "_combined_real_cycle_story"
TABLE_DIR = PROJECT_DIR / "results" / "tables" / "real_cycle_ctid_state_order_grid" / "_combined_real_cycle_story"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["S7", "S12", "S14", "S17"]
CANDIDATE_ORDER = ["C1", "C2", "C3", "C4"]

STATE_LABELS = {
    "S7": "S7\n2-2-3",
    "S12": "S12\n3-3-6",
    "S14": "S14\n4-4-6",
    "S17": "S17\n4-4-9",
}

CANDIDATE_LABELS = {
    "C1": "C1\nlinear",
    "C2": "C2\nquadratic",
    "C3": "C3\ncubic",
    "C4": "C4\nquartic",
}

HIST_BINS = int(float(__import__("os").environ.get("UN_HIST_BINS", "1000")))

PARAMETER_COLUMNS = [
    "alpha_n_hat",
    "alpha_p_hat",
    "K_e_hat",
    "g_n_hat",
    "g_p_hat",
    "g_e_hat",
    "theta_n0_hat",
    "theta_p0_hat",
]

RANK_COLUMNS = [
    "rank_phi_raw",
    "ncols_phi_raw",
    "cond_phi_raw",
    "rank_X_raw",
    "ncols_X_raw",
    "cond_X_raw",
]

print("=" * 100)
print("REAL-CYCLE CT-ID GRID STORY SUMMARY")
print("=" * 100)
print("PROJECT_DIR:", PROJECT_DIR)
print("RESULT_ROOT:", RESULT_ROOT)
print("LOG_ROOT:", LOG_ROOT)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)
print("TABLE_DIR:", TABLE_DIR)
print("HIST_BINS:", HIST_BINS)
print("=" * 100)


# %% =====================================================
# CELL 2 — Helper functions
# =====================================================
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=280, bbox_inches="tight")
    print("[saved figure]", path)
    plt.close()


def safe_filename(text: str) -> str:
    return (
        str(text)
        .replace("^", "pow")
        .replace("-", "minus")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("%", "percent")
    )


def state_sort_key(s: str) -> int:
    try:
        return STATE_ORDER.index(str(s))
    except ValueError:
        return 999


def candidate_sort_key(c: str) -> int:
    try:
        return CANDIDATE_ORDER.index(str(c))
    except ValueError:
        return 999


def model_sort_key(model_id: str) -> tuple[int, int]:
    state, cand = split_model_id(model_id)
    return state_sort_key(state), candidate_sort_key(cand)


def split_model_id(model_id: str) -> tuple[str, str]:
    parts = str(model_id).split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def infer_state_candidate_from_folder(name: str) -> tuple[str | None, str | None]:
    """
    Expected folder example:
        real_cycle0_S7_C1_100seeds_200_to_299_dt_1.0_bins_100
    """
    for s in STATE_ORDER:
        for c in CANDIDATE_ORDER:
            if f"_{s}_{c}_" in name or f"states_{s}_orders_{c}" in name:
                return s, c
    return None, None


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print("[warn] failed to read", path, repr(exc))
            return None
    return None


def finite_values(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def choose_zoom_range(x: np.ndarray, qlo: float = 0.01, qhi: float = 0.99) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    lo = float(np.quantile(x, qlo))
    hi = float(np.quantile(x, qhi))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(x))
        hi = float(np.max(x))

    if lo == hi:
        pad = abs(lo) * 0.01 + 1e-12
        lo -= pad
        hi += pad

    return lo, hi


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    colorbar_label: str,
    out_path: Path,
    annotate: bool = True,
    fmt: str = ".3e",
    log_transform: bool = False,
) -> None:
    data = pivot.values.astype(float)

    if log_transform:
        data_plot = np.log10(np.maximum(data, 1e-300))
        cbar_label = f"log10({colorbar_label})"
    else:
        data_plot = data
        cbar_label = colorbar_label

    plt.figure(figsize=(9.2, 6.2))
    im = plt.imshow(data_plot, aspect="auto")
    plt.colorbar(im, label=cbar_label)

    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    if annotate:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = data[i, j]
                if np.isfinite(val):
                    plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    plt.xlabel("Output candidate")
    plt.ylabel("State variant")
    plt.title(title)
    plt.tight_layout()
    savefig(out_path)


def make_hist_with_lines(
    values: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = HIST_BINS,
    best_value: float | None = None,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    mean_value = float(np.mean(values))
    median_value = float(np.median(values))

    plt.figure(figsize=(10.8, 6.2))
    plt.hist(
        values,
        bins=bins,
        density=False,
        edgecolor="black",
        linewidth=0.22,
        alpha=0.78,
        label=f"recovered values, n={len(values)}",
    )

    plt.axvline(mean_value, linestyle="--", linewidth=2.6, label=f"mean = {mean_value:.6g}")
    plt.axvline(median_value, linestyle=":", linewidth=3.0, label=f"median = {median_value:.6g}")

    if best_value is not None and np.isfinite(best_value):
        plt.axvline(best_value, linestyle="-.", linewidth=2.8, label=f"best-RMSE estimate = {best_value:.6g}")

    plt.grid(True, axis="y", alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel("Count of simulations")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    savefig(out_path)


# %% =====================================================
# CELL 3 — Discover and load all real-cycle result folders
# =====================================================
if not RESULT_ROOT.exists():
    raise RuntimeError(f"RESULT_ROOT does not exist: {RESULT_ROOT}")

result_dirs = sorted([
    p for p in RESULT_ROOT.glob("*")
    if p.is_dir()
    and not p.name.startswith("_")
    and (p / "real_cycle_all_runs.csv").exists()
])

if not result_dirs:
    raise RuntimeError(
        f"No real_cycle_all_runs.csv files found under {RESULT_ROOT}. "
        "Check whether the real-cycle jobs completed."
    )

print("\nFound result folders:", len(result_dirs))
for p in result_dirs[:10]:
    print(" ", p.name)
if len(result_dirs) > 10:
    print(" ...")

all_frames = []
best_frames = []
summary_frames = []
param_frames = []
beta_frames = []
config_rows = []
selected_cycle_files = []

for folder in result_dirs:
    state_id, candidate_id = infer_state_candidate_from_folder(folder.name)

    all_path = folder / "real_cycle_all_runs.csv"
    best_path = folder / "real_cycle_best_runs.csv"
    summary_path = folder / "real_cycle_model_summary.csv"
    param_path = folder / "real_cycle_parameter_long.csv"
    beta_path = folder / "real_cycle_beta_coefficients.csv"
    config_path = folder / "real_cycle_config.json"
    selected_cycle_path = folder / "selected_real_cycle_id_data.csv"

    df_all = read_csv_if_exists(all_path)
    if df_all is not None:
        if "state_id" not in df_all.columns or df_all["state_id"].isna().all():
            df_all["state_id"] = state_id
        if "candidate_id" not in df_all.columns or df_all["candidate_id"].isna().all():
            df_all["candidate_id"] = candidate_id
        if "model_id" not in df_all.columns:
            df_all["model_id"] = df_all["state_id"].astype(str) + "_" + df_all["candidate_id"].astype(str)

        df_all["source_folder"] = str(folder)
        df_all["run_tag"] = folder.name
        all_frames.append(df_all)

    df_best = read_csv_if_exists(best_path)
    if df_best is not None:
        if "state_id" not in df_best.columns or df_best["state_id"].isna().all():
            df_best["state_id"] = state_id
        if "candidate_id" not in df_best.columns or df_best["candidate_id"].isna().all():
            df_best["candidate_id"] = candidate_id
        if "model_id" not in df_best.columns:
            df_best["model_id"] = df_best["state_id"].astype(str) + "_" + df_best["candidate_id"].astype(str)

        df_best["source_folder"] = str(folder)
        df_best["run_tag"] = folder.name
        best_frames.append(df_best)

    df_summary = read_csv_if_exists(summary_path)
    if df_summary is not None:
        if "state_id" not in df_summary.columns or df_summary["state_id"].isna().all():
            df_summary["state_id"] = state_id
        if "candidate_id" not in df_summary.columns or df_summary["candidate_id"].isna().all():
            df_summary["candidate_id"] = candidate_id
        if "model_id" not in df_summary.columns:
            df_summary["model_id"] = df_summary["state_id"].astype(str) + "_" + df_summary["candidate_id"].astype(str)

        df_summary["source_folder"] = str(folder)
        df_summary["run_tag"] = folder.name
        summary_frames.append(df_summary)

    df_param = read_csv_if_exists(param_path)
    if df_param is not None:
        df_param["source_folder"] = str(folder)
        df_param["run_tag"] = folder.name
        param_frames.append(df_param)

    df_beta = read_csv_if_exists(beta_path)
    if df_beta is not None:
        df_beta["source_folder"] = str(folder)
        df_beta["run_tag"] = folder.name
        beta_frames.append(df_beta)

    if selected_cycle_path.exists():
        selected_cycle_files.append(selected_cycle_path)

    config_payload = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_payload = json.load(f)
        except Exception as exc:
            print("[warn] failed to read config:", config_path, repr(exc))

    real_cycle = config_payload.get("real_cycle", {})
    config_rows.append({
        "run_tag": folder.name,
        "source_folder": str(folder),
        "state_id": state_id,
        "candidate_id": candidate_id,
        "model_id": f"{state_id}_{candidate_id}",
        "config_path": str(config_path) if config_path.exists() else "",
        "cycle_index": real_cycle.get("cycle_index", np.nan),
        "id_samples": real_cycle.get("id_samples", np.nan),
        "id_Ts": real_cycle.get("id_Ts", np.nan),
        "current_flip_applied": real_cycle.get("current_flip_applied", np.nan),
        "voltage_start": real_cycle.get("voltage_start", np.nan),
        "voltage_end": real_cycle.get("voltage_end", np.nan),
        "voltage_min": real_cycle.get("voltage_min", np.nan),
        "voltage_max": real_cycle.get("voltage_max", np.nan),
    })

if not all_frames:
    raise RuntimeError("No real-cycle all-run tables could be loaded.")

all_runs = pd.concat(all_frames, ignore_index=True)
best_runs = pd.concat(best_frames, ignore_index=True) if best_frames else pd.DataFrame()
model_summary_raw = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
parameter_long = pd.concat(param_frames, ignore_index=True) if param_frames else pd.DataFrame()
beta_coefficients = pd.concat(beta_frames, ignore_index=True) if beta_frames else pd.DataFrame()
config_table = pd.DataFrame(config_rows)

all_runs["state_id"] = all_runs["state_id"].astype(str)
all_runs["candidate_id"] = all_runs["candidate_id"].astype(str)
all_runs["model_id"] = all_runs["state_id"] + "_" + all_runs["candidate_id"]
all_runs["state_sort"] = all_runs["state_id"].map(state_sort_key)
all_runs["candidate_sort"] = all_runs["candidate_id"].map(candidate_sort_key)

all_runs = all_runs.sort_values(["state_sort", "candidate_sort", "rmse"]).reset_index(drop=True)

all_runs.to_csv(OUT_DIR / "combined_real_cycle_all_runs.csv", index=False)
parameter_long.to_csv(OUT_DIR / "combined_real_cycle_parameter_long.csv", index=False)
beta_coefficients.to_csv(OUT_DIR / "combined_real_cycle_beta_coefficients.csv", index=False)
config_table.to_csv(OUT_DIR / "combined_real_cycle_config_table.csv", index=False)

all_runs.to_csv(TABLE_DIR / "combined_real_cycle_all_runs.csv", index=False)

print("\nCombined all_runs shape:", all_runs.shape)
print("Unique models:", sorted(all_runs["model_id"].unique(), key=model_sort_key))
print("Saved:", OUT_DIR / "combined_real_cycle_all_runs.csv")


# %% =====================================================
# CELL 4 — Completion audit
# =====================================================
completion_rows = []

for state_id in STATE_ORDER:
    for candidate_id in CANDIDATE_ORDER:
        model_id = f"{state_id}_{candidate_id}"
        g = all_runs[all_runs["model_id"] == model_id]

        completion_rows.append({
            "state_id": state_id,
            "candidate_id": candidate_id,
            "model_id": model_id,
            "found_runs": int(len(g)),
            "found": bool(len(g) > 0),
            "source_folders": ";".join(sorted(g["run_tag"].dropna().unique().tolist())),
        })

completion = pd.DataFrame(completion_rows)
completion.to_csv(OUT_DIR / "completion_audit.csv", index=False)
completion.to_csv(TABLE_DIR / "completion_audit.csv", index=False)

print("\nCompletion audit:")
print(completion.to_string(index=False))


# %% =====================================================
# CELL 5 — Build combined model summary
# =====================================================
summary_rows = []
best_rows = []

for model_id, g0 in all_runs.groupby("model_id"):
    g = g0.sort_values("rmse").reset_index(drop=True)
    best = g.iloc[0].copy()

    state_id = str(best["state_id"])
    candidate_id = str(best["candidate_id"])

    best_rows.append(best)

    row = {
        "model_id": model_id,
        "state_id": state_id,
        "candidate_id": candidate_id,
        "degree": int(best["degree"]) if "degree" in best and pd.notna(best["degree"]) else np.nan,
        "nx": int(best["nx"]) if "nx" in best and pd.notna(best["nx"]) else np.nan,
        "n_runs": int(len(g)),
        "best_seed": int(best["seed"]) if "seed" in best and pd.notna(best["seed"]) else np.nan,
        "best_rmse": float(best["rmse"]),
        "best_mae": float(best["mae"]) if "mae" in best else np.nan,
        "best_r2_percent": float(best["r2_percent"]) if "r2_percent" in best else np.nan,
        "best_bfr_percent": float(best["bfr_percent"]) if "bfr_percent" in best else np.nan,
        "median_rmse": float(g["rmse"].median()),
        "mean_rmse": float(g["rmse"].mean()),
        "std_rmse": float(g["rmse"].std(ddof=1)),
        "q05_rmse": float(g["rmse"].quantile(0.05)),
        "q25_rmse": float(g["rmse"].quantile(0.25)),
        "q75_rmse": float(g["rmse"].quantile(0.75)),
        "q95_rmse": float(g["rmse"].quantile(0.95)),
    }

    for col in RANK_COLUMNS + PARAMETER_COLUMNS:
        if col in best.index:
            row[f"best_{col}"] = best[col]

    summary_rows.append(row)

df_best = pd.DataFrame(best_rows).sort_values("rmse").reset_index(drop=True)
df_best.insert(0, "overall_best_rank", np.arange(1, len(df_best) + 1))

df_summary = pd.DataFrame(summary_rows)
df_summary["state_sort"] = df_summary["state_id"].map(state_sort_key)
df_summary["candidate_sort"] = df_summary["candidate_id"].map(candidate_sort_key)
df_summary = df_summary.sort_values(["state_sort", "candidate_sort"]).reset_index(drop=True)

df_best.to_csv(OUT_DIR / "combined_real_cycle_best_runs.csv", index=False)
df_summary.to_csv(OUT_DIR / "combined_real_cycle_model_summary.csv", index=False)

df_best.to_csv(TABLE_DIR / "combined_real_cycle_best_runs.csv", index=False)
df_summary.to_csv(TABLE_DIR / "combined_real_cycle_model_summary.csv", index=False)

print("\nTop models by best RMSE:")
show_cols = [
    "overall_best_rank",
    "model_id",
    "seed",
    "rmse",
    "mae",
    "r2_percent",
    "bfr_percent",
    "rank_phi_raw",
    "ncols_phi_raw",
    "rank_X_raw",
    "ncols_X_raw",
]
show_cols = [c for c in show_cols if c in df_best.columns]
print(df_best[show_cols].head(20).to_string(index=False))


# %% =====================================================
# CELL 6 — RMSE heatmaps: state variant x output order
# =====================================================
for metric, title, filename, log_flag in [
    ("best_rmse", "Best RMSE by state variant and output order", "heatmap_best_rmse.png", False),
    ("median_rmse", "Median RMSE by state variant and output order", "heatmap_median_rmse.png", False),
    ("mean_rmse", "Mean RMSE by state variant and output order", "heatmap_mean_rmse.png", False),
    ("std_rmse", "RMSE standard deviation by state variant and output order", "heatmap_std_rmse.png", False),
]:
    pivot = df_summary.pivot(index="state_id", columns="candidate_id", values=metric)
    pivot = pivot.reindex(index=STATE_ORDER, columns=CANDIDATE_ORDER)

    plot_heatmap(
        pivot=pivot,
        title=title,
        colorbar_label=metric,
        out_path=FIG_DIR / filename,
        annotate=True,
        fmt=".3e",
        log_transform=log_flag,
    )


# %% =====================================================
# CELL 7 — Rank and conditioning heatmaps
# =====================================================
rank_heatmap_specs = [
    ("best_rank_phi_raw", "Best-run Phi rank", "heatmap_best_phi_rank.png", ".0f", False),
    ("best_ncols_phi_raw", "Best-run Phi number of columns", "heatmap_best_phi_ncols.png", ".0f", False),
    ("best_cond_phi_raw", "Best-run Phi condition number", "heatmap_best_phi_condition_log.png", ".2e", True),
    ("best_rank_X_raw", "Best-run state-trajectory rank", "heatmap_best_state_rank.png", ".0f", False),
    ("best_ncols_X_raw", "Best-run number of states", "heatmap_best_state_ncols.png", ".0f", False),
    ("best_cond_X_raw", "Best-run state-trajectory condition number", "heatmap_best_state_condition_log.png", ".2e", True),
]

for metric, title, filename, fmt, log_flag in rank_heatmap_specs:
    if metric not in df_summary.columns:
        continue

    pivot = df_summary.pivot(index="state_id", columns="candidate_id", values=metric)
    pivot = pivot.reindex(index=STATE_ORDER, columns=CANDIDATE_ORDER)

    plot_heatmap(
        pivot=pivot,
        title=title,
        colorbar_label=metric,
        out_path=FIG_DIR / filename,
        annotate=True,
        fmt=fmt,
        log_transform=log_flag,
    )


# %% =====================================================
# CELL 8 — RMSE scatter and boxplots
# =====================================================
model_ids = sorted(all_runs["model_id"].unique(), key=model_sort_key)
model_to_x = {m: i for i, m in enumerate(model_ids)}

# Scatter plot across all runs.
plt.figure(figsize=(15.5, 6.8))
rng = np.random.default_rng(123)

for model_id in model_ids:
    g = all_runs[all_runs["model_id"] == model_id]
    x0 = model_to_x[model_id]
    jitter = rng.normal(0.0, 0.045, size=len(g))

    plt.scatter(
        np.full(len(g), x0) + jitter,
        g["rmse"].values,
        s=14,
        alpha=0.48,
    )

    best = g.sort_values("rmse").head(1)
    if len(best):
        plt.scatter(
            [x0],
            [float(best["rmse"].iloc[0])],
            marker="*",
            s=160,
            edgecolor="black",
            linewidth=0.8,
        )

plt.grid(True, axis="y", alpha=0.35)
plt.xticks(np.arange(len(model_ids)), model_ids, rotation=45, ha="right")
plt.xlabel("Model: state variant + output order")
plt.ylabel("RMSE [V]")
plt.title("Real-data RMSE scatter across all multistart runs")
plt.tight_layout()
savefig(FIG_DIR / "scatter_rmse_all_models.png")

# Boxplot across all models.
box_data = []
box_labels = []

for model_id in model_ids:
    g = all_runs[all_runs["model_id"] == model_id]
    box_data.append(g["rmse"].values)
    box_labels.append(model_id)

plt.figure(figsize=(15.5, 6.8))
plt.boxplot(box_data, labels=box_labels, showmeans=True)
plt.grid(True, axis="y", alpha=0.35)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Model: state variant + output order")
plt.ylabel("RMSE [V]")
plt.title("Real-data RMSE boxplot by model")
plt.tight_layout()
savefig(FIG_DIR / "boxplot_rmse_all_models.png")

# Boxplot grouped by state.
state_data = []
state_labels = []
for s in STATE_ORDER:
    g = all_runs[all_runs["state_id"] == s]
    if len(g):
        state_data.append(g["rmse"].values)
        state_labels.append(STATE_LABELS.get(s, s))

plt.figure(figsize=(9.2, 6.0))
plt.boxplot(state_data, labels=state_labels, showmeans=True)
plt.grid(True, axis="y", alpha=0.35)
plt.xlabel("State variant")
plt.ylabel("RMSE [V]")
plt.title("RMSE distribution grouped by state variant")
plt.tight_layout()
savefig(FIG_DIR / "boxplot_rmse_by_state_variant.png")

# Boxplot grouped by candidate.
cand_data = []
cand_labels = []
for c in CANDIDATE_ORDER:
    g = all_runs[all_runs["candidate_id"] == c]
    if len(g):
        cand_data.append(g["rmse"].values)
        cand_labels.append(CANDIDATE_LABELS.get(c, c))

plt.figure(figsize=(9.2, 6.0))
plt.boxplot(cand_data, labels=cand_labels, showmeans=True)
plt.grid(True, axis="y", alpha=0.35)
plt.xlabel("Output candidate")
plt.ylabel("RMSE [V]")
plt.title("RMSE distribution grouped by output polynomial order")
plt.tight_layout()
savefig(FIG_DIR / "boxplot_rmse_by_output_candidate.png")


# %% =====================================================
# CELL 9 — Best/median RMSE trends by state and candidate
# =====================================================
plt.figure(figsize=(10.8, 6.2))
for s in STATE_ORDER:
    sub = df_summary[df_summary["state_id"] == s].sort_values("candidate_sort")
    if len(sub):
        plt.plot(
            sub["candidate_id"],
            sub["best_rmse"],
            marker="o",
            linewidth=2.4,
            label=f"{s} best",
        )

plt.grid(True, alpha=0.35)
plt.xlabel("Output candidate")
plt.ylabel("Best RMSE [V]")
plt.title("Best RMSE trend with output order for each state variant")
plt.legend(loc="best")
plt.tight_layout()
savefig(FIG_DIR / "line_best_rmse_vs_output_order_by_state.png")

plt.figure(figsize=(10.8, 6.2))
for s in STATE_ORDER:
    sub = df_summary[df_summary["state_id"] == s].sort_values("candidate_sort")
    if len(sub):
        plt.plot(
            sub["candidate_id"],
            sub["median_rmse"],
            marker="s",
            linewidth=2.4,
            label=f"{s} median",
        )

plt.grid(True, alpha=0.35)
plt.xlabel("Output candidate")
plt.ylabel("Median RMSE [V]")
plt.title("Median RMSE trend with output order for each state variant")
plt.legend(loc="best")
plt.tight_layout()
savefig(FIG_DIR / "line_median_rmse_vs_output_order_by_state.png")


# %% =====================================================
# CELL 10 — Parameter summary and 1000-bin histograms
# =====================================================
param_summary_rows = []

for model_id, g in all_runs.groupby("model_id"):
    best = g.sort_values("rmse").iloc[0]

    for param in PARAMETER_COLUMNS:
        if param not in g.columns:
            continue

        values = finite_values(g[param])
        if len(values) == 0:
            continue

        best_value = float(best[param]) if param in best.index and pd.notna(best[param]) else np.nan

        param_summary_rows.append({
            "model_id": model_id,
            "state_id": best["state_id"],
            "candidate_id": best["candidate_id"],
            "parameter": param,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "q01": float(np.quantile(values, 0.01)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "best_rmse_estimate": best_value,
            "best_seed": int(best["seed"]) if "seed" in best and pd.notna(best["seed"]) else np.nan,
            "best_rmse": float(best["rmse"]),
        })

param_summary = pd.DataFrame(param_summary_rows)
param_summary.to_csv(OUT_DIR / "real_cycle_parameter_summary.csv", index=False)
param_summary.to_csv(TABLE_DIR / "real_cycle_parameter_summary.csv", index=False)

# Per-model histograms.
PARAM_FIG_DIR = FIG_DIR / "parameter_histograms_1000bins"
PARAM_FIG_DIR.mkdir(parents=True, exist_ok=True)

for model_id, g in all_runs.groupby("model_id"):
    model_dir = PARAM_FIG_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    best = g.sort_values("rmse").iloc[0]

    for param in PARAMETER_COLUMNS:
        if param not in g.columns:
            continue

        values = finite_values(g[param])
        if len(values) == 0:
            continue

        best_value = float(best[param]) if param in best.index and pd.notna(best[param]) else np.nan

        make_hist_with_lines(
            values=values,
            title=f"{model_id}: recovered {param} distribution ({HIST_BINS} bins)",
            xlabel=f"Recovered {param}",
            out_path=model_dir / f"{model_id}_hist_{safe_filename(param)}_1000bins.png",
            bins=HIST_BINS,
            best_value=best_value,
        )

# Combined small multiples for each parameter.
SMALL_DIR = FIG_DIR / "parameter_small_multiples_1000bins"
SMALL_DIR.mkdir(parents=True, exist_ok=True)

for param in PARAMETER_COLUMNS:
    if param not in all_runs.columns:
        continue

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=False, sharey=False)
    axes = axes.reshape(-1)

    for ax, model_id in zip(axes, model_ids):
        g = all_runs[all_runs["model_id"] == model_id]
        values = finite_values(g[param])

        if len(values) == 0:
            ax.set_title(f"{model_id}: no data")
            continue

        best = g.sort_values("rmse").iloc[0]
        best_value = float(best[param]) if param in best.index and pd.notna(best[param]) else np.nan

        mean_value = float(np.mean(values))
        median_value = float(np.median(values))

        lo, hi = choose_zoom_range(values, 0.01, 0.99)
        zoom_values = values[(values >= lo) & (values <= hi)]

        ax.hist(
            zoom_values,
            bins=HIST_BINS,
            density=False,
            edgecolor="black",
            linewidth=0.15,
            alpha=0.78,
            label=f"n={len(zoom_values)}",
        )
        ax.axvline(mean_value, linestyle="--", linewidth=1.8, label="mean")
        ax.axvline(median_value, linestyle=":", linewidth=2.0, label="median")

        if np.isfinite(best_value):
            ax.axvline(best_value, linestyle="-.", linewidth=1.8, label="best-RMSE")

        ax.grid(True, axis="y", alpha=0.30)
        ax.set_title(model_id, fontsize=9)
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(fontsize=6, loc="best")

    fig.suptitle(f"Real-data recovered {param} distributions by model, zoomed q01-q99", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(SMALL_DIR / f"small_multiple_{safe_filename(param)}_zoom_1000bins.png")


# %% =====================================================
# CELL 11 — Combined parameter overlays by state/candidate
# =====================================================
OVERLAY_DIR = FIG_DIR / "parameter_overlays_1000bins"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# Overlay candidates for each state.
for param in PARAMETER_COLUMNS:
    if param not in all_runs.columns:
        continue

    for s in STATE_ORDER:
        plt.figure(figsize=(11.5, 6.4))
        has_any = False

        sub_state = all_runs[all_runs["state_id"] == s]
        all_vals = finite_values(sub_state[param])

        if len(all_vals) == 0:
            plt.close()
            continue

        lo, hi = choose_zoom_range(all_vals, 0.01, 0.99)
        bins = np.linspace(lo, hi, HIST_BINS + 1)

        for c in CANDIDATE_ORDER:
            g = all_runs[(all_runs["state_id"] == s) & (all_runs["candidate_id"] == c)]
            values = finite_values(g[param])
            values = values[(values >= lo) & (values <= hi)]

            if len(values) == 0:
                continue

            has_any = True
            plt.hist(values, bins=bins, histtype="step", linewidth=1.8, density=False, label=f"{s}_{c}, n={len(values)}")

        if has_any:
            plt.grid(True, axis="y", alpha=0.35)
            plt.xlabel(f"Recovered {param}")
            plt.ylabel("Count of simulations")
            plt.title(f"{s}: candidate overlay histogram for {param}, zoomed q01-q99")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            savefig(OVERLAY_DIR / f"{s}_overlay_candidates_{safe_filename(param)}_1000bins.png")
        else:
            plt.close()

# Overlay states for each candidate.
for param in PARAMETER_COLUMNS:
    if param not in all_runs.columns:
        continue

    for c in CANDIDATE_ORDER:
        plt.figure(figsize=(11.5, 6.4))
        has_any = False

        sub_cand = all_runs[all_runs["candidate_id"] == c]
        all_vals = finite_values(sub_cand[param])

        if len(all_vals) == 0:
            plt.close()
            continue

        lo, hi = choose_zoom_range(all_vals, 0.01, 0.99)
        bins = np.linspace(lo, hi, HIST_BINS + 1)

        for s in STATE_ORDER:
            g = all_runs[(all_runs["state_id"] == s) & (all_runs["candidate_id"] == c)]
            values = finite_values(g[param])
            values = values[(values >= lo) & (values <= hi)]

            if len(values) == 0:
                continue

            has_any = True
            plt.hist(values, bins=bins, histtype="step", linewidth=1.8, density=False, label=f"{s}_{c}, n={len(values)}")

        if has_any:
            plt.grid(True, axis="y", alpha=0.35)
            plt.xlabel(f"Recovered {param}")
            plt.ylabel("Count of simulations")
            plt.title(f"{c}: state-variant overlay histogram for {param}, zoomed q01-q99")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            savefig(OVERLAY_DIR / f"{c}_overlay_states_{safe_filename(param)}_1000bins.png")
        else:
            plt.close()


# %% =====================================================
# CELL 12 — Measured step response vs estimated step response
# =====================================================
"""
The original fitter saved only run-level tables, not Yhat arrays for every run.
However, it saved best-fit voltage figures during fitting.

This cell does two things:

1. It creates a manifest of existing best-fit figures.
2. If per-run prediction CSV files exist in future runs, it will overlay measured
   and estimated voltage directly.

Recommended future improvement:
    Modify run_real_cycle_state_order_ctid_grid.py to save the best-run prediction
    as:
        best_prediction_timeseries.csv
    with columns:
        t_s, measured_voltage_V, estimated_voltage_V, current_A, residual_V
"""

FIT_MANIFEST_DIR = FIG_DIR / "best_fit_manifest"
FIT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

manifest_rows = []

# Existing per-run figures are usually under:
# results/figures/real_cycle_ctid_state_order_grid/<run_tag>/<model_id>/...
FIG_SEARCH_ROOT = PROJECT_DIR / "results" / "figures" / "real_cycle_ctid_state_order_grid"

for _, row in df_best.iterrows():
    model_id = str(row["model_id"])
    seed = int(row["seed"]) if "seed" in row.index and pd.notna(row["seed"]) else None

    if seed is None:
        continue

    patterns = [
        f"**/{model_id}_best_seed_{seed}_voltage_fit.png",
        f"**/*{model_id}*best*{seed}*voltage*.png",
        f"**/*best_seed_{seed}*voltage_fit.png",
    ]

    for pat in patterns:
        matches = sorted(FIG_SEARCH_ROOT.glob(pat))
        for m in matches:
            manifest_rows.append({
                "model_id": model_id,
                "seed": seed,
                "rmse": row["rmse"],
                "figure_path": str(m),
                "figure_name": m.name,
            })

manifest = pd.DataFrame(manifest_rows).drop_duplicates()
manifest.to_csv(OUT_DIR / "best_fit_voltage_figure_manifest.csv", index=False)
manifest.to_csv(TABLE_DIR / "best_fit_voltage_figure_manifest.csv", index=False)

print("\nBest-fit voltage figure manifest:")
if len(manifest):
    print(manifest.head(50).to_string(index=False))
else:
    print("No existing voltage-fit figures found. See note below.")

# Try to find saved prediction CSV files, if any exist.
prediction_files = sorted(RESULT_ROOT.glob("**/*prediction*.csv")) + sorted(RESULT_ROOT.glob("**/*timeseries*.csv"))

prediction_manifest_rows = []

PRED_FIG_DIR = FIG_DIR / "measured_vs_estimated_step_response"
PRED_FIG_DIR.mkdir(parents=True, exist_ok=True)

for pred_path in prediction_files:
    try:
        dfp = pd.read_csv(pred_path)
    except Exception:
        continue

    cols = {c.lower(): c for c in dfp.columns}

    t_col = None
    y_col = None
    yh_col = None
    i_col = None

    for possible in ["t_s", "time_s", "time", "t"]:
        if possible in cols:
            t_col = cols[possible]
            break

    for possible in ["measured_voltage_v", "voltage_v", "v_measured", "y", "y_true"]:
        if possible in cols:
            y_col = cols[possible]
            break

    for possible in ["estimated_voltage_v", "v_hat", "v_estimated", "yhat", "y_hat"]:
        if possible in cols:
            yh_col = cols[possible]
            break

    for possible in ["current_a", "current_discharge_positive_a", "i", "input"]:
        if possible in cols:
            i_col = cols[possible]
            break

    if t_col is None or y_col is None or yh_col is None:
        continue

    t = pd.to_numeric(dfp[t_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(dfp[y_col], errors="coerce").to_numpy(dtype=float)
    yh = pd.to_numeric(dfp[yh_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yh)
    t, y, yh = t[mask], y[mask], yh[mask]

    if len(t) == 0:
        continue

    residual = y - yh
    rmse_val = float(np.sqrt(np.mean(residual**2)))

    name_safe = safe_filename(pred_path.stem)

    plt.figure(figsize=(12.5, 6.2))
    plt.plot(t, y, linewidth=2.5, label="measured step response")
    plt.plot(t, yh, "--", linewidth=2.3, label="estimated step response")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"Measured vs estimated step response, RMSE={rmse_val:.6e}")
    plt.legend(loc="best")
    plt.tight_layout()
    fig_path = PRED_FIG_DIR / f"{name_safe}_measured_vs_estimated_voltage.png"
    savefig(fig_path)

    plt.figure(figsize=(12.5, 4.5))
    plt.plot(t, residual, linewidth=1.8)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Residual [V]")
    plt.title(f"Residual: measured minus estimated, RMSE={rmse_val:.6e}")
    plt.tight_layout()
    res_fig_path = PRED_FIG_DIR / f"{name_safe}_residual.png"
    savefig(res_fig_path)

    prediction_manifest_rows.append({
        "prediction_file": str(pred_path),
        "n_points": int(len(t)),
        "rmse_from_file": rmse_val,
        "voltage_fit_figure": str(fig_path),
        "residual_figure": str(res_fig_path),
    })

prediction_manifest = pd.DataFrame(prediction_manifest_rows)
prediction_manifest.to_csv(OUT_DIR / "prediction_timeseries_manifest.csv", index=False)
prediction_manifest.to_csv(TABLE_DIR / "prediction_timeseries_manifest.csv", index=False)

if len(prediction_manifest) == 0:
    print("\nNo prediction time-series CSV files found.")
    print("The script created a manifest of existing voltage-fit figures instead.")
    print("For future runs, save best_prediction_timeseries.csv from the fitter so this script can plot measured vs estimated directly.")


# %% =====================================================
# CELL 13 — Log summary
# =====================================================
log_rows = []

for path in sorted(LOG_ROOT.glob("real_ctid_grid_*.out")) + sorted(LOG_ROOT.glob("real_ctid_grid_*.err")):
    try:
        text = path.read_text(errors="replace")
    except Exception as exc:
        log_rows.append({
            "log_file": str(path),
            "read_error": repr(exc),
        })
        continue

    log_rows.append({
        "log_file": str(path),
        "log_name": path.name,
        "rmse_lines": len(re.findall(r"RMSE=", text)),
        "fail_lines": len(re.findall(r"\[FAIL\]", text)),
        "memory_error_detected": (
            "Cannot allocate memory" in text
            or "LLVM ERROR" in text
            or "Out Of Memory" in text
            or "oom-kill" in text.lower()
        ),
        "traceback_detected": "Traceback (most recent call last)" in text,
        "finished_marker": "Finished at:" in text,
    })

log_summary = pd.DataFrame(log_rows)
log_summary.to_csv(OUT_DIR / "log_summary.csv", index=False)
log_summary.to_csv(TABLE_DIR / "log_summary.csv", index=False)

if len(log_summary):
    print("\nLog summary:")
    print(log_summary.head(80).to_string(index=False))


# %% =====================================================
# CELL 14 — LaTeX-ready tables
# =====================================================
def to_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.6g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, escape=False, float_format=lambda x: float_format % x)
    path.write_text(latex, encoding="utf-8")
    print("[saved latex table]", path)

latex_summary_cols = [
    "model_id",
    "n_runs",
    "best_seed",
    "best_rmse",
    "median_rmse",
    "mean_rmse",
    "best_bfr_percent",
    "best_rank_phi_raw",
    "best_ncols_phi_raw",
    "best_rank_X_raw",
    "best_ncols_X_raw",
]
latex_summary_cols = [c for c in latex_summary_cols if c in df_summary.columns]

to_latex_table(
    df_summary.sort_values("best_rmse")[latex_summary_cols],
    TABLE_DIR / "latex_real_cycle_model_summary_ranked.tex",
)

top_best_cols = [
    "overall_best_rank",
    "model_id",
    "seed",
    "rmse",
    "mae",
    "r2_percent",
    "bfr_percent",
    "rank_phi_raw",
    "ncols_phi_raw",
    "rank_X_raw",
    "ncols_X_raw",
]
top_best_cols = [c for c in top_best_cols if c in df_best.columns]

to_latex_table(
    df_best[top_best_cols].head(20),
    TABLE_DIR / "latex_real_cycle_top20_best_runs.tex",
)

if len(param_summary):
    param_latex_cols = [
        "model_id",
        "parameter",
        "mean",
        "median",
        "std",
        "q05",
        "q95",
        "best_rmse_estimate",
        "best_rmse",
    ]
    param_latex_cols = [c for c in param_latex_cols if c in param_summary.columns]

    to_latex_table(
        param_summary[param_latex_cols],
        TABLE_DIR / "latex_real_cycle_parameter_summary.tex",
    )


# %% =====================================================
# CELL 15 — Final printout
# =====================================================
print("\n" + "=" * 100)
print("REAL-CYCLE CT-ID STORY SUMMARY COMPLETE")
print("=" * 100)
print("Combined outputs:")
print(" ", OUT_DIR)
print("Figures:")
print(" ", FIG_DIR)
print("Tables:")
print(" ", TABLE_DIR)

print("\nMost important tables:")
print(" ", TABLE_DIR / "combined_real_cycle_all_runs.csv")
print(" ", TABLE_DIR / "combined_real_cycle_model_summary.csv")
print(" ", TABLE_DIR / "combined_real_cycle_best_runs.csv")
print(" ", TABLE_DIR / "real_cycle_parameter_summary.csv")
print(" ", TABLE_DIR / "completion_audit.csv")
print(" ", TABLE_DIR / "best_fit_voltage_figure_manifest.csv")

print("\nMost important figures:")
print(" ", FIG_DIR / "heatmap_best_rmse.png")
print(" ", FIG_DIR / "heatmap_median_rmse.png")
print(" ", FIG_DIR / "scatter_rmse_all_models.png")
print(" ", FIG_DIR / "boxplot_rmse_all_models.png")
print(" ", FIG_DIR / "boxplot_rmse_by_state_variant.png")
print(" ", FIG_DIR / "boxplot_rmse_by_output_candidate.png")
print(" ", FIG_DIR / "line_best_rmse_vs_output_order_by_state.png")
print(" ", FIG_DIR / "line_median_rmse_vs_output_order_by_state.png")
print(" ", FIG_DIR / "parameter_small_multiples_1000bins")
print(" ", FIG_DIR / "parameter_overlays_1000bins")
print("=" * 100)