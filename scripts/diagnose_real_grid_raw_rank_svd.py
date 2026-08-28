#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% =====================================================
# CELL 0 — Imports
# =====================================================
"""
diagnose_real_grid_raw_rank_svd.py

Purpose
-------
Standalone raw-rank diagnostic script for real-data CT-ID results.

This script does NOT rerun JAX-SysID.
It only reads existing result folders and checks raw rank diagnostics.

It answers:

    1. What is the raw SVD rank of Phi for each S/C model?
    2. What is the raw SVD rank of Xhat for each S/C model?
    3. What are the singular values?
    4. What are the condition numbers?
    5. Which models are full-rank in Phi?
    6. Which models have weak state-trajectory rank?
    7. Does dynamic rank change after subtracting the initial row?

Definitions
-----------
Raw rank:
    rank(M) = number of singular values satisfying

        s_i > rel_tol * s_max

    No centering.
    No scaling.
    No column normalization.

Dynamic rank:
    First subtract the initial row:

        M_dyn(t_i,:) = M(t_i,:) - M(t_0,:)

    Then compute raw SVD rank on the moved trajectory.

    This is useful because a matrix can appear high-rank due to constants or
    offsets, while the actual movement over the cycle may be lower-dimensional.

Inputs
------
The script searches:

    results/real_cycle_ctid_state_order_grid/

for folders containing:

    real_cycle_all_runs.csv

Optional, if the step-response patch has been used:

    best_step_response_manifest.csv
    best_step_response_timeseries/*feature_matrix_phi.csv
    best_step_response_timeseries/*state_trajectory.csv

Outputs
-------
results/real_grid_raw_rank_diagnostics/
results/figures/real_grid_raw_rank_diagnostics/
results/tables/real_grid_raw_rank_diagnostics/
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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

OUT_DIR = PROJECT_DIR / "results" / "real_grid_raw_rank_diagnostics"
FIG_DIR = PROJECT_DIR / "results" / "figures" / "real_grid_raw_rank_diagnostics"
TABLE_DIR = PROJECT_DIR / "results" / "tables" / "real_grid_raw_rank_diagnostics"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["S7", "S12", "S14", "S17"]
CANDIDATE_ORDER = ["C1", "C2", "C3", "C4"]

# Raw SVD tolerance.
# This is the important rank rule:
#     s_i > SVD_REL_TOL * s_max
SVD_REL_TOL = float(__import__("os").environ.get("UN_RAW_RANK_TOL", "1e-10"))

# Which result folders to include.
# Default catches:
#   real_cycle0_S7_C1_100seeds...
#   real10_S7_C4_cycle...
#   real10_S17_C4_cycle...
#   quick_top10_response...
INCLUDE_PATTERNS = [
    "real_cycle*_S*_C*",
    "real10_S*_C*_cycle_*",
    "quick_top10_response_S*_C*_seed_*",
]

print("=" * 100)
print("RAW SVD RANK DIAGNOSTICS FOR REAL-DATA CT-ID")
print("=" * 100)
print("PROJECT_DIR:", PROJECT_DIR)
print("RESULT_ROOT:", RESULT_ROOT)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)
print("TABLE_DIR:", TABLE_DIR)
print("SVD_REL_TOL:", SVD_REL_TOL)
print("=" * 100)


# %% =====================================================
# CELL 2 — Helper functions
# =====================================================
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=280, bbox_inches="tight")
    print("[saved figure]", path)
    plt.close()


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("^", "pow")
        .replace("-", "minus")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(":", "_")
        .replace(";", "_")
        .replace(",", "_")
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
    state_id, candidate_id = split_model_id(model_id)
    return state_sort_key(state_id), candidate_sort_key(candidate_id)


def split_model_id(model_id: str) -> tuple[str, str]:
    m = re.search(r"(S\d+)_(C\d+)", str(model_id))
    if m:
        return m.group(1), m.group(2)
    return "", ""


def infer_state_candidate_cycle_from_folder(folder_name: str) -> dict:
    """
    Supports names like:

        real_cycle0_S17_C4_100seeds_...
        real10_S17_C4_cycle_0_chunk_...
        quick_top10_response_S17_C4_seed_265
    """
    out = {
        "state_id": None,
        "candidate_id": None,
        "model_id": None,
        "cycle_index": np.nan,
        "chunk_index": np.nan,
        "seed0": np.nan,
        "seed1": np.nan,
    }

    m = re.search(r"(S\d+)_(C\d+)", folder_name)
    if m:
        out["state_id"] = m.group(1)
        out["candidate_id"] = m.group(2)
        out["model_id"] = f"{out['state_id']}_{out['candidate_id']}"

    m = re.search(r"real_cycle(\d+)_", folder_name)
    if m:
        out["cycle_index"] = int(m.group(1))

    m = re.search(r"cycle_(\d+)_chunk_(\d+)", folder_name)
    if m:
        out["cycle_index"] = int(m.group(1))
        out["chunk_index"] = int(m.group(2))

    m = re.search(r"(\d+)seeds_(\d+)_to_(\d+)", folder_name)
    if m:
        out["seed0"] = int(m.group(2))
        out["seed1"] = int(m.group(3))

    return out


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print("[warn] failed to read", path, repr(exc))
        return None


def raw_svd_rank(M: np.ndarray, rel_tol: float = SVD_REL_TOL) -> dict:
    """
    Compute raw SVD rank.

    No centering.
    No scaling.
    No normalization.

    Rank rule:
        rank = number of singular values where

            s_i > rel_tol * s_max
    """
    M = np.asarray(M, dtype=float)

    if M.ndim != 2 or M.size == 0:
        return {
            "n_rows": int(M.shape[0]) if M.ndim == 2 else 0,
            "n_cols": int(M.shape[1]) if M.ndim == 2 else 0,
            "rank": 0,
            "nullity": np.nan,
            "condition_number": np.nan,
            "s_max": np.nan,
            "s_min": np.nan,
            "threshold": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    try:
        svals = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "n_rows": int(M.shape[0]),
            "n_cols": int(M.shape[1]),
            "rank": np.nan,
            "nullity": np.nan,
            "condition_number": np.nan,
            "s_max": np.nan,
            "s_min": np.nan,
            "threshold": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    if len(svals) == 0:
        return {
            "n_rows": int(M.shape[0]),
            "n_cols": int(M.shape[1]),
            "rank": 0,
            "nullity": int(M.shape[1]),
            "condition_number": np.nan,
            "s_max": np.nan,
            "s_min": np.nan,
            "threshold": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    smax = float(svals[0])
    smin = float(svals[-1])
    threshold = rel_tol * max(smax, 1e-300)

    rank = int(np.sum(svals > threshold))
    n_cols = int(M.shape[1])
    nullity = int(n_cols - rank)
    cond = float(smax / max(smin, 1e-300))

    return {
        "n_rows": int(M.shape[0]),
        "n_cols": n_cols,
        "rank": rank,
        "nullity": nullity,
        "condition_number": cond,
        "s_max": smax,
        "s_min": smin,
        "threshold": threshold,
        "singular_values": [float(v) for v in svals],
        "relative_singular_values": [float(v / max(smax, 1e-300)) for v in svals],
    }


def dynamic_raw_svd_rank(M: np.ndarray, rel_tol: float = SVD_REL_TOL) -> dict:
    """
    Compute raw SVD rank after subtracting the first row.

    This checks how many independent directions actually move during the cycle.

        M_dyn = M - M[0,:]

    Then zero-motion columns are removed before SVD.
    """
    M = np.asarray(M, dtype=float)

    if M.ndim != 2 or M.size == 0:
        ans = raw_svd_rank(M, rel_tol=rel_tol)
        ans["n_active_cols"] = 0
        return ans

    Md = M - M[0:1, :]

    col_norm = np.linalg.norm(Md, axis=0)
    active = col_norm > 1e-14

    if not np.any(active):
        return {
            "n_rows": int(M.shape[0]),
            "n_cols": int(M.shape[1]),
            "n_active_cols": 0,
            "rank": 0,
            "nullity": int(M.shape[1]),
            "condition_number": np.nan,
            "s_max": np.nan,
            "s_min": np.nan,
            "threshold": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    ans = raw_svd_rank(Md[:, active], rel_tol=rel_tol)
    ans["n_active_cols"] = int(np.sum(active))
    ans["original_n_cols"] = int(M.shape[1])
    return ans


def find_existing_file_from_manifest_value(value) -> Path | None:
    if value is None or pd.isna(value):
        return None

    p = Path(str(value))

    if p.exists():
        return p

    # Sometimes CSV stores relative path from project root.
    p2 = PROJECT_DIR / str(value)
    if p2.exists():
        return p2

    return None


def load_matrix_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)

    # Drop time column if present.
    drop_cols = [c for c in df.columns if c.lower() in ["t_s", "time", "time_s", "t"]]
    cols = [c for c in df.columns if c not in drop_cols]

    M = df[cols].to_numpy(dtype=float)
    return M, cols


def plot_singular_values(
    singular_values: list[float],
    threshold: float,
    title: str,
    out_path: Path,
) -> None:
    svals = np.asarray(singular_values, dtype=float)

    if len(svals) == 0:
        return

    plt.figure(figsize=(8.8, 5.5))
    plt.semilogy(np.arange(1, len(svals) + 1), svals, marker="o", linewidth=2.0)
    if np.isfinite(threshold):
        plt.axhline(threshold, linestyle="--", linewidth=1.6, label=f"threshold={threshold:.2e}")
        plt.legend(loc="best")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Singular-value index")
    plt.ylabel("Singular value")
    plt.title(title)
    plt.tight_layout()
    savefig(out_path)


# %% =====================================================
# CELL 3 — Discover result folders
# =====================================================
if not RESULT_ROOT.exists():
    raise RuntimeError(f"Result root does not exist: {RESULT_ROOT}")

folder_set = set()

for pattern in INCLUDE_PATTERNS:
    for p in RESULT_ROOT.glob(pattern):
        if p.is_dir() and (p / "real_cycle_all_runs.csv").exists():
            folder_set.add(p)

folders = sorted(folder_set)

if not folders:
    raise RuntimeError(
        f"No real_cycle_all_runs.csv folders found under {RESULT_ROOT}. "
        "Check result folder names."
    )

print("\nFound result folders:", len(folders))
for p in folders[:30]:
    print(" ", p.name)
if len(folders) > 30:
    print(" ...")


# %% =====================================================
# CELL 4 — Load all stored rank rows
# =====================================================
all_run_frames = []
folder_rows = []

for folder in folders:
    info = infer_state_candidate_cycle_from_folder(folder.name)

    df = read_csv_if_exists(folder / "real_cycle_all_runs.csv")
    if df is None:
        continue

    if "state_id" not in df.columns and info["state_id"] is not None:
        df["state_id"] = info["state_id"]

    if "candidate_id" not in df.columns and info["candidate_id"] is not None:
        df["candidate_id"] = info["candidate_id"]

    if "model_id" not in df.columns:
        df["model_id"] = df["state_id"].astype(str) + "_" + df["candidate_id"].astype(str)

    df["source_folder"] = str(folder)
    df["run_folder"] = folder.name
    df["cycle_index_from_folder"] = info["cycle_index"]
    df["chunk_index_from_folder"] = info["chunk_index"]

    # Standardize cycle and chunk.
    if "cycle_index" not in df.columns:
        df["cycle_index"] = info["cycle_index"]

    if "chunk_index" not in df.columns:
        df["chunk_index"] = info["chunk_index"]

    all_run_frames.append(df)

    folder_rows.append({
        "run_folder": folder.name,
        "source_folder": str(folder),
        **info,
        "n_rows": len(df),
    })

if not all_run_frames:
    raise RuntimeError("No all-run rows loaded.")

all_runs = pd.concat(all_run_frames, ignore_index=True)

# Force numeric fields.
for c in [
    "seed",
    "rmse",
    "mae",
    "r2_percent",
    "bfr_percent",
    "rank_phi_raw",
    "ncols_phi_raw",
    "cond_phi_raw",
    "rank_X_raw",
    "ncols_X_raw",
    "cond_X_raw",
    "cycle_index",
    "chunk_index",
]:
    if c in all_runs.columns:
        all_runs[c] = pd.to_numeric(all_runs[c], errors="coerce")

all_runs["state_id"] = all_runs["state_id"].astype(str)
all_runs["candidate_id"] = all_runs["candidate_id"].astype(str)
all_runs["model_id"] = all_runs["state_id"] + "_" + all_runs["candidate_id"]

all_runs["state_sort"] = all_runs["state_id"].map(state_sort_key)
all_runs["candidate_sort"] = all_runs["candidate_id"].map(candidate_sort_key)

all_runs = all_runs.sort_values(
    ["cycle_index", "state_sort", "candidate_sort", "rmse"]
).reset_index(drop=True)

folder_audit = pd.DataFrame(folder_rows)

all_runs.to_csv(OUT_DIR / "combined_all_runs_rank_columns.csv", index=False)
all_runs.to_csv(TABLE_DIR / "combined_all_runs_rank_columns.csv", index=False)

folder_audit.to_csv(OUT_DIR / "folder_audit.csv", index=False)
folder_audit.to_csv(TABLE_DIR / "folder_audit.csv", index=False)

print("\nCombined all-runs table:", all_runs.shape)
print("Models found:", sorted(all_runs["model_id"].dropna().unique(), key=model_sort_key))
print("Cycles found:", sorted(pd.to_numeric(all_runs["cycle_index"], errors="coerce").dropna().astype(int).unique()))


# %% =====================================================
# CELL 5 — Stored raw-rank summary from all_runs
# =====================================================
summary_rows = []
best_rows = []

group_cols = ["cycle_index", "model_id"]

for (cycle_index, model_id), g0 in all_runs.groupby(group_cols, dropna=False):
    g = g0.sort_values("rmse").reset_index(drop=True)
    best = g.iloc[0]
    best_rows.append(best)

    state_id, candidate_id = split_model_id(model_id)

    row = {
        "cycle_index": cycle_index,
        "model_id": model_id,
        "state_id": state_id,
        "candidate_id": candidate_id,
        "n_runs": int(len(g)),
        "best_seed": int(best["seed"]) if "seed" in best and pd.notna(best["seed"]) else np.nan,
        "best_rmse": float(best["rmse"]) if "rmse" in best and pd.notna(best["rmse"]) else np.nan,
        "median_rmse": float(g["rmse"].median()) if "rmse" in g.columns else np.nan,
    }

    for col in [
        "rank_phi_raw",
        "ncols_phi_raw",
        "cond_phi_raw",
        "rank_X_raw",
        "ncols_X_raw",
        "cond_X_raw",
    ]:
        if col in best.index:
            row[f"best_{col}"] = best[col]

    # Stability across seeds.
    if "rank_phi_raw" in g.columns:
        row["min_rank_phi_raw"] = float(g["rank_phi_raw"].min())
        row["median_rank_phi_raw"] = float(g["rank_phi_raw"].median())
        row["max_rank_phi_raw"] = float(g["rank_phi_raw"].max())

    if "rank_X_raw" in g.columns:
        row["min_rank_X_raw"] = float(g["rank_X_raw"].min())
        row["median_rank_X_raw"] = float(g["rank_X_raw"].median())
        row["max_rank_X_raw"] = float(g["rank_X_raw"].max())

    if "cond_phi_raw" in g.columns:
        row["median_cond_phi_raw"] = float(g["cond_phi_raw"].median())
        row["best_seed_cond_phi_raw"] = float(best["cond_phi_raw"])

    if "cond_X_raw" in g.columns:
        row["median_cond_X_raw"] = float(g["cond_X_raw"].median())
        row["best_seed_cond_X_raw"] = float(best["cond_X_raw"])

    summary_rows.append(row)

best_runs = pd.DataFrame(best_rows).reset_index(drop=True)
stored_summary = pd.DataFrame(summary_rows)

stored_summary["state_sort"] = stored_summary["state_id"].map(state_sort_key)
stored_summary["candidate_sort"] = stored_summary["candidate_id"].map(candidate_sort_key)

stored_summary = stored_summary.sort_values(
    ["cycle_index", "state_sort", "candidate_sort"]
).reset_index(drop=True)

best_runs.to_csv(OUT_DIR / "best_run_rank_rows.csv", index=False)
stored_summary.to_csv(OUT_DIR / "stored_raw_rank_summary.csv", index=False)

best_runs.to_csv(TABLE_DIR / "best_run_rank_rows.csv", index=False)
stored_summary.to_csv(TABLE_DIR / "stored_raw_rank_summary.csv", index=False)

print("\nStored raw-rank summary:")
display_cols = [
    "cycle_index",
    "model_id",
    "n_runs",
    "best_seed",
    "best_rmse",
    "best_rank_phi_raw",
    "best_ncols_phi_raw",
    "best_cond_phi_raw",
    "best_rank_X_raw",
    "best_ncols_X_raw",
    "best_cond_X_raw",
]
display_cols = [c for c in display_cols if c in stored_summary.columns]
print(stored_summary[display_cols].to_string(index=False))


# %% =====================================================
# CELL 6 — Recompute raw SVD from saved Phi/X CSV files if available
# =====================================================
recomputed_rows = []
singular_rows = []
column_rows = []

for _, best in best_runs.iterrows():
    folder = Path(str(best["source_folder"]))
    model_id = str(best["model_id"])
    seed = int(best["seed"]) if "seed" in best.index and pd.notna(best["seed"]) else None
    cycle_index = best.get("cycle_index", np.nan)

    if seed is None:
        continue

    manifest_path = folder / "best_step_response_manifest.csv"

    if not manifest_path.exists():
        continue

    manifest = read_csv_if_exists(manifest_path)
    if manifest is None or len(manifest) == 0:
        continue

    if "seed" in manifest.columns:
        manifest["seed"] = pd.to_numeric(manifest["seed"], errors="coerce")
        mrow_df = manifest[manifest["seed"] == seed]
        if len(mrow_df) == 0:
            mrow_df = manifest.sort_values("rmse").head(1) if "rmse" in manifest.columns else manifest.head(1)
    else:
        mrow_df = manifest.head(1)

    if len(mrow_df) == 0:
        continue

    mrow = mrow_df.iloc[0]

    phi_file = None
    x_file = None

    for possible in ["phi_csv", "feature_matrix_phi_csv"]:
        if possible in mrow.index:
            phi_file = find_existing_file_from_manifest_value(mrow[possible])
            if phi_file is not None:
                break

    for possible in ["state_csv", "state_trajectory_csv"]:
        if possible in mrow.index:
            x_file = find_existing_file_from_manifest_value(mrow[possible])
            if x_file is not None:
                break

    # If manifest paths are missing, search the folder.
    if phi_file is None:
        candidates = sorted(folder.glob("best_step_response_timeseries/*feature_matrix_phi.csv"))
        if candidates:
            phi_file = candidates[0]

    if x_file is None:
        candidates = sorted(folder.glob("best_step_response_timeseries/*state_trajectory.csv"))
        if candidates:
            x_file = candidates[0]

    base_row = {
        "cycle_index": cycle_index,
        "model_id": model_id,
        "state_id": str(best["state_id"]),
        "candidate_id": str(best["candidate_id"]),
        "seed": seed,
        "source_folder": str(folder),
        "phi_file": str(phi_file) if phi_file is not None else "",
        "x_file": str(x_file) if x_file is not None else "",
    }

    for matrix_name, file_path in [("Phi", phi_file), ("Xhat", x_file)]:
        if file_path is None or not file_path.exists():
            continue

        M, col_names = load_matrix_csv(file_path)

        raw = raw_svd_rank(M, rel_tol=SVD_REL_TOL)
        dyn = dynamic_raw_svd_rank(M, rel_tol=SVD_REL_TOL)

        row = dict(base_row)
        row.update({
            "matrix": matrix_name,
            "n_rows": raw["n_rows"],
            "n_cols": raw["n_cols"],
            "raw_rank": raw["rank"],
            "raw_nullity": raw["nullity"],
            "raw_condition_number": raw["condition_number"],
            "raw_s_max": raw["s_max"],
            "raw_s_min": raw["s_min"],
            "raw_threshold": raw["threshold"],
            "dynamic_rank": dyn["rank"],
            "dynamic_n_active_cols": dyn.get("n_active_cols", np.nan),
            "dynamic_condition_number": dyn["condition_number"],
            "file_path": str(file_path),
        })
        recomputed_rows.append(row)

        for i, sv in enumerate(raw["singular_values"], start=1):
            singular_rows.append({
                "cycle_index": cycle_index,
                "model_id": model_id,
                "seed": seed,
                "matrix": matrix_name,
                "singular_index": i,
                "singular_value": sv,
                "relative_singular_value": raw["relative_singular_values"][i - 1],
                "threshold": raw["threshold"],
                "below_threshold": bool(sv <= raw["threshold"]),
                "file_path": str(file_path),
            })

        for j, col in enumerate(col_names):
            arr = M[:, j]
            column_rows.append({
                "cycle_index": cycle_index,
                "model_id": model_id,
                "seed": seed,
                "matrix": matrix_name,
                "column_index": j,
                "column_name": col,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "span": float(np.ptp(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
                "norm": float(np.linalg.norm(arr)),
                "dynamic_norm": float(np.linalg.norm(arr - arr[0])),
                "file_path": str(file_path),
            })

        # Save singular-value plot.
        plot_singular_values(
            singular_values=raw["singular_values"],
            threshold=raw["threshold"],
            title=f"{model_id}, cycle {cycle_index}, seed {seed}: raw singular values of {matrix_name}",
            out_path=FIG_DIR / "singular_values" / f"{safe_name(model_id)}_cycle_{cycle_index}_seed_{seed}_{matrix_name}_singular_values.png",
        )

recomputed = pd.DataFrame(recomputed_rows)
singular_values = pd.DataFrame(singular_rows)
column_diagnostics = pd.DataFrame(column_rows)

recomputed.to_csv(OUT_DIR / "recomputed_raw_svd_from_saved_phi_x.csv", index=False)
singular_values.to_csv(OUT_DIR / "raw_singular_values_long.csv", index=False)
column_diagnostics.to_csv(OUT_DIR / "raw_column_diagnostics.csv", index=False)

recomputed.to_csv(TABLE_DIR / "recomputed_raw_svd_from_saved_phi_x.csv", index=False)
singular_values.to_csv(TABLE_DIR / "raw_singular_values_long.csv", index=False)
column_diagnostics.to_csv(TABLE_DIR / "raw_column_diagnostics.csv", index=False)

if len(recomputed):
    print("\nRecomputed raw SVD from saved Phi/X files:")
    print(recomputed.to_string(index=False))
else:
    print("\nNo saved Phi/X CSV files found for recomputation.")
    print("That is okay: stored rank columns from real_cycle_all_runs.csv were still summarized.")


# %% =====================================================
# CELL 7 — Heatmaps for first-cycle S/C grid
# =====================================================
def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    fmt: str = ".2e",
    log_scale: bool = False,
) -> None:
    if value_col not in df.columns:
        return

    # Prefer cycle 0 when available.
    d = df.copy()
    if "cycle_index" in d.columns:
        cycle0 = d[pd.to_numeric(d["cycle_index"], errors="coerce") == 0]
        if len(cycle0):
            d = cycle0

    pivot = d.pivot_table(
        index="state_id",
        columns="candidate_id",
        values=value_col,
        aggfunc="min" if "rank" not in value_col else "max",
    )

    pivot = pivot.reindex(index=STATE_ORDER, columns=CANDIDATE_ORDER)

    values = pivot.to_numpy(dtype=float)

    if log_scale:
        plot_values = np.log10(np.maximum(values, 1e-300))
        color_label = f"log10({value_col})"
    else:
        plot_values = values
        color_label = value_col

    plt.figure(figsize=(8.8, 6.2))
    im = plt.imshow(plot_values, aspect="auto")
    plt.colorbar(im, label=color_label)

    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = values[i, j]
            if np.isfinite(val):
                try:
                    txt = format(val, fmt)
                except Exception:
                    txt = str(val)
                plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.xlabel("Output candidate")
    plt.ylabel("State variant")
    plt.title(title)
    plt.tight_layout()
    savefig(out_path)


plot_heatmap(
    stored_summary,
    "best_rank_phi_raw",
    "Best-seed raw rank of Phi by S/C model",
    FIG_DIR / "heatmap_best_rank_phi_raw_cycle0.png",
    fmt=".0f",
)

plot_heatmap(
    stored_summary,
    "best_ncols_phi_raw",
    "Number of Phi columns by S/C model",
    FIG_DIR / "heatmap_ncols_phi_raw_cycle0.png",
    fmt=".0f",
)

plot_heatmap(
    stored_summary,
    "best_cond_phi_raw",
    "Best-seed raw condition number of Phi by S/C model",
    FIG_DIR / "heatmap_best_cond_phi_raw_log_cycle0.png",
    fmt=".1e",
    log_scale=True,
)

plot_heatmap(
    stored_summary,
    "best_rank_X_raw",
    "Best-seed raw rank of Xhat by S/C model",
    FIG_DIR / "heatmap_best_rank_X_raw_cycle0.png",
    fmt=".0f",
)

plot_heatmap(
    stored_summary,
    "best_ncols_X_raw",
    "Number of Xhat columns by S/C model",
    FIG_DIR / "heatmap_ncols_X_raw_cycle0.png",
    fmt=".0f",
)

plot_heatmap(
    stored_summary,
    "best_cond_X_raw",
    "Best-seed raw condition number of Xhat by S/C model",
    FIG_DIR / "heatmap_best_cond_X_raw_log_cycle0.png",
    fmt=".1e",
    log_scale=True,
)


# %% =====================================================
# CELL 8 — Rank ratio figures
# =====================================================
if {"best_rank_phi_raw", "best_ncols_phi_raw"}.issubset(stored_summary.columns):
    stored_summary["best_phi_rank_ratio"] = (
        pd.to_numeric(stored_summary["best_rank_phi_raw"], errors="coerce")
        / pd.to_numeric(stored_summary["best_ncols_phi_raw"], errors="coerce")
    )

if {"best_rank_X_raw", "best_ncols_X_raw"}.issubset(stored_summary.columns):
    stored_summary["best_X_rank_ratio"] = (
        pd.to_numeric(stored_summary["best_rank_X_raw"], errors="coerce")
        / pd.to_numeric(stored_summary["best_ncols_X_raw"], errors="coerce")
    )

stored_summary.to_csv(TABLE_DIR / "stored_raw_rank_summary_with_ratios.csv", index=False)

# Scatter: RMSE vs rank ratios.
if {"best_rmse", "best_phi_rank_ratio", "best_X_rank_ratio"}.issubset(stored_summary.columns):
    d = stored_summary.copy()
    if "cycle_index" in d.columns:
        cycle0 = d[pd.to_numeric(d["cycle_index"], errors="coerce") == 0]
        if len(cycle0):
            d = cycle0

    plt.figure(figsize=(9.5, 6.2))
    plt.scatter(d["best_phi_rank_ratio"], d["best_rmse"], s=90, label="Phi rank ratio")

    for _, row in d.iterrows():
        plt.annotate(
            str(row["model_id"]),
            xy=(row["best_phi_rank_ratio"], row["best_rmse"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.grid(True, alpha=0.35)
    plt.xlabel(r"Raw rank ratio $\mathrm{rank}(\Phi)/n_{\phi}$")
    plt.ylabel("Best RMSE [V]")
    plt.title("Best RMSE versus raw Phi-rank ratio")
    plt.tight_layout()
    savefig(FIG_DIR / "scatter_rmse_vs_phi_rank_ratio_cycle0.png")

    plt.figure(figsize=(9.5, 6.2))
    plt.scatter(d["best_X_rank_ratio"], d["best_rmse"], s=90, label="X rank ratio")

    for _, row in d.iterrows():
        plt.annotate(
            str(row["model_id"]),
            xy=(row["best_X_rank_ratio"], row["best_rmse"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.grid(True, alpha=0.35)
    plt.xlabel(r"Raw rank ratio $\mathrm{rank}(X)/n_x$")
    plt.ylabel("Best RMSE [V]")
    plt.title("Best RMSE versus raw state-trajectory rank ratio")
    plt.tight_layout()
    savefig(FIG_DIR / "scatter_rmse_vs_X_rank_ratio_cycle0.png")


# %% =====================================================
# CELL 9 — Rank distributions across seeds
# =====================================================
DIST_DIR = FIG_DIR / "rank_distributions_across_seeds"
DIST_DIR.mkdir(parents=True, exist_ok=True)

for col, ylabel, title, filename in [
    ("rank_phi_raw", r"Raw rank of $\Phi$", r"Raw $\Phi$ rank across seeds", "boxplot_rank_phi_raw_by_model.png"),
    ("rank_X_raw", r"Raw rank of $X$", r"Raw $X$ rank across seeds", "boxplot_rank_X_raw_by_model.png"),
    ("cond_phi_raw", r"Condition number of $\Phi$", r"Raw $\Phi$ condition number across seeds", "boxplot_cond_phi_raw_by_model_log.png"),
    ("cond_X_raw", r"Condition number of $X$", r"Raw $X$ condition number across seeds", "boxplot_cond_X_raw_by_model_log.png"),
]:
    if col not in all_runs.columns:
        continue

    # Prefer cycle 0 if multiple cycles are present.
    d = all_runs.copy()
    cycle0 = d[pd.to_numeric(d["cycle_index"], errors="coerce") == 0]
    if len(cycle0):
        d = cycle0

    model_ids = sorted(d["model_id"].dropna().unique(), key=model_sort_key)

    data = []
    labels = []

    for model_id in model_ids:
        vals = pd.to_numeric(d.loc[d["model_id"] == model_id, col], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            data.append(vals)
            labels.append(model_id)

    if not data:
        continue

    plt.figure(figsize=(14.5, 6.2))
    plt.boxplot(data, labels=labels, showmeans=True)

    if "cond_" in col:
        plt.yscale("log")

    plt.grid(True, axis="y", which="both", alpha=0.35)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    savefig(DIST_DIR / filename)


# %% =====================================================
# CELL 10 — LaTeX-ready tables
# =====================================================
def save_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.6g}")
    path.write_text(latex, encoding="utf-8")
    print("[saved latex table]", path)


latex_cols = [
    "cycle_index",
    "model_id",
    "n_runs",
    "best_seed",
    "best_rmse",
    "best_rank_phi_raw",
    "best_ncols_phi_raw",
    "best_cond_phi_raw",
    "best_rank_X_raw",
    "best_ncols_X_raw",
    "best_cond_X_raw",
    "min_rank_phi_raw",
    "median_rank_phi_raw",
    "max_rank_phi_raw",
    "min_rank_X_raw",
    "median_rank_X_raw",
    "max_rank_X_raw",
]
latex_cols = [c for c in latex_cols if c in stored_summary.columns]

save_latex_table(
    stored_summary[latex_cols],
    TABLE_DIR / "latex_stored_raw_rank_summary.tex",
)

if len(recomputed):
    latex_cols_recomputed = [
        "cycle_index",
        "model_id",
        "seed",
        "matrix",
        "n_rows",
        "n_cols",
        "raw_rank",
        "raw_nullity",
        "raw_condition_number",
        "dynamic_rank",
        "dynamic_n_active_cols",
        "dynamic_condition_number",
    ]
    latex_cols_recomputed = [c for c in latex_cols_recomputed if c in recomputed.columns]

    save_latex_table(
        recomputed[latex_cols_recomputed],
        TABLE_DIR / "latex_recomputed_raw_svd_from_saved_phi_x.tex",
    )


# %% =====================================================
# CELL 11 — Final printout
# =====================================================
print("\n" + "=" * 100)
print("RAW RANK DIAGNOSTIC COMPLETE")
print("=" * 100)
print("Tables:", TABLE_DIR)
print("Figures:", FIG_DIR)

print("\nImportant CSV files:")
print(" ", TABLE_DIR / "stored_raw_rank_summary.csv")
print(" ", TABLE_DIR / "stored_raw_rank_summary_with_ratios.csv")
print(" ", TABLE_DIR / "combined_all_runs_rank_columns.csv")
print(" ", TABLE_DIR / "recomputed_raw_svd_from_saved_phi_x.csv")
print(" ", TABLE_DIR / "raw_singular_values_long.csv")
print(" ", TABLE_DIR / "raw_column_diagnostics.csv")

print("\nImportant figures:")
print(" ", FIG_DIR / "heatmap_best_rank_phi_raw_cycle0.png")
print(" ", FIG_DIR / "heatmap_best_cond_phi_raw_log_cycle0.png")
print(" ", FIG_DIR / "heatmap_best_rank_X_raw_cycle0.png")
print(" ", FIG_DIR / "heatmap_best_cond_X_raw_log_cycle0.png")
print(" ", FIG_DIR / "scatter_rmse_vs_phi_rank_ratio_cycle0.png")
print(" ", FIG_DIR / "scatter_rmse_vs_X_rank_ratio_cycle0.png")
print(" ", FIG_DIR / "singular_values")
print(" ", FIG_DIR / "rank_distributions_across_seeds")
print("=" * 100)