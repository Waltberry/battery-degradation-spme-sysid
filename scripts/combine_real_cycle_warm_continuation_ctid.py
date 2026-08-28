#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROJECT_DIR = Path.cwd()

MODEL_ID = os.environ.get("UN_MODEL_ID", "S7_C4")
CYCLE_INDEX = int(os.environ.get("UN_REAL_CYCLE_INDEX", "0"))

ROOT = PROJECT_DIR / "results" / "real_warm_continuation_ctid" / MODEL_ID
FIG_ROOT = PROJECT_DIR / "results" / "figures" / "real_warm_continuation_ctid" / MODEL_ID

OUT_DIR = ROOT / "_combined"
FIG_DIR = FIG_ROOT / "_combined"

ANCHOR_DIR = ROOT / "anchors"
TABLE_DIR = PROJECT_DIR / "results" / "tables" / "real_warm_continuation_ctid" / MODEL_ID

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

prefix = f"warmseq_{MODEL_ID}_cycle_{CYCLE_INDEX}_"

folders = sorted([
    p for p in ROOT.glob(prefix + "*")
    if p.is_dir() and (p / "all_runs.csv").exists()
])

print("=" * 100)
print("COMBINE WARM-CONTINUATION CYCLE")
print("=" * 100)
print("MODEL_ID:", MODEL_ID)
print("CYCLE_INDEX:", CYCLE_INDEX)
print("ROOT:", ROOT)
print("folders found:", len(folders))
for p in folders:
    print(" ", p.name)
print("=" * 100)

if not folders:
    raise RuntimeError(f"No folders found for {prefix}")

all_frames = []
best_manifest_frames = []
summary_frames = []
failed_frames = []

for folder in folders:
    df = pd.read_csv(folder / "all_runs.csv")
    df["source_folder"] = str(folder)
    df["run_folder"] = folder.name
    all_frames.append(df)

    mpath = folder / "best_manifest.csv"
    if mpath.exists():
        dm = pd.read_csv(mpath)
        dm["source_folder"] = str(folder)
        dm["run_folder"] = folder.name
        best_manifest_frames.append(dm)

    spath = folder / "summary.csv"
    if spath.exists():
        ds = pd.read_csv(spath)
        ds["source_folder"] = str(folder)
        ds["run_folder"] = folder.name
        summary_frames.append(ds)



    fpath = folder / "failed_runs.csv"
    if fpath.exists() and fpath.stat().st_size > 0:
        try:
            dfail = pd.read_csv(fpath)

            if len(dfail):
                dfail["source_folder"] = str(folder)
                dfail["run_folder"] = folder.name
                failed_frames.append(dfail)

        except pd.errors.EmptyDataError:
            pass

df_all = pd.concat(all_frames, ignore_index=True)
df_all = df_all.sort_values("rmse").reset_index(drop=True)

df_all.to_csv(OUT_DIR / f"cycle_{CYCLE_INDEX:04d}_all_runs.csv", index=False)
df_all.to_csv(TABLE_DIR / f"cycle_{CYCLE_INDEX:04d}_all_runs.csv", index=False)

df_best = df_all.head(1).copy()
df_best.to_csv(OUT_DIR / f"cycle_{CYCLE_INDEX:04d}_best_run.csv", index=False)
df_best.to_csv(TABLE_DIR / f"cycle_{CYCLE_INDEX:04d}_best_run.csv", index=False)

if best_manifest_frames:
    df_manifest = pd.concat(best_manifest_frames, ignore_index=True)
else:
    df_manifest = pd.DataFrame()

if summary_frames:
    df_summary_chunks = pd.concat(summary_frames, ignore_index=True)
else:
    df_summary_chunks = pd.DataFrame()

if failed_frames:
    df_failed = pd.concat(failed_frames, ignore_index=True)
else:
    df_failed = pd.DataFrame()

df_failed.to_csv(OUT_DIR / f"cycle_{CYCLE_INDEX:04d}_failed_runs.csv", index=False)
df_failed.to_csv(TABLE_DIR / f"cycle_{CYCLE_INDEX:04d}_failed_runs.csv", index=False)

best_row = df_best.iloc[0]
best_seed = int(best_row["seed"])
best_rmse = float(best_row["rmse"])

# Find matching best manifest row.
best_param_path = None
best_response_csv = None
best_folder = None

if len(df_manifest):
    df_manifest["seed"] = pd.to_numeric(df_manifest["seed"], errors="coerce")
    candidates = df_manifest[df_manifest["seed"] == best_seed].copy()

    if len(candidates):
        candidates["rmse_diff"] = np.abs(pd.to_numeric(candidates["rmse"], errors="coerce") - best_rmse)
        m = candidates.sort_values("rmse_diff").iloc[0]
        best_param_path = Path(str(m["best_params_raw"]))
        best_response_csv = Path(str(m["response_csv"]))
        best_folder = Path(str(m["source_folder"]))

if best_param_path is None or not best_param_path.exists():
    # Fall back: find the run folder whose all_runs has the best seed/RMSE.
    for folder in folders:
        fbest = folder / "best_params_raw.npz"
        if fbest.exists():
            # Only use this fallback if its best_run matches.
            br = pd.read_csv(folder / "best_run.csv")
            if len(br) and int(br.iloc[0]["seed"]) == best_seed:
                best_param_path = fbest
                best_folder = folder
                break

if best_param_path is None or not best_param_path.exists():
    raise RuntimeError("Could not locate best_params_raw.npz for the best run.")

anchor_path = ANCHOR_DIR / f"cycle_{CYCLE_INDEX:04d}_best_params_raw.npz"
shutil.copy2(best_param_path, anchor_path)

print("[saved anchor]", anchor_path)

cycle_summary = {
    "model_id": MODEL_ID,
    "cycle_index": CYCLE_INDEX,
    "n_success": int(len(df_all)),
    "n_fail": int(len(df_failed)),
    "best_seed": best_seed,
    "best_rmse": best_rmse,
    "best_mae": float(best_row["mae"]),
    "best_r2_percent": float(best_row["r2_percent"]),
    "best_bfr_percent": float(best_row["bfr_percent"]),
    "median_rmse": float(df_all["rmse"].median()),
    "mean_rmse": float(df_all["rmse"].mean()),
    "std_rmse": float(df_all["rmse"].std(ddof=1)) if len(df_all) > 1 else np.nan,
    "best_rank_phi_raw": int(best_row["rank_phi_raw"]),
    "best_ncols_phi_raw": int(best_row["ncols_phi_raw"]),
    "best_cond_phi_raw": float(best_row["cond_phi_raw"]),
    "best_rank_X_raw": int(best_row["rank_X_raw"]),
    "best_ncols_X_raw": int(best_row["ncols_X_raw"]),
    "best_cond_X_raw": float(best_row["cond_X_raw"]),
    "anchor_path": str(anchor_path),
    "best_source_folder": str(best_folder) if best_folder is not None else "",
    "best_param_path_original": str(best_param_path),
    "best_response_csv": str(best_response_csv) if best_response_csv is not None else "",
}

df_cycle_summary = pd.DataFrame([cycle_summary])
df_cycle_summary.to_csv(OUT_DIR / f"cycle_{CYCLE_INDEX:04d}_summary.csv", index=False)
df_cycle_summary.to_csv(TABLE_DIR / f"cycle_{CYCLE_INDEX:04d}_summary.csv", index=False)

# Update cumulative summary.
all_summary_files = sorted(OUT_DIR.glob("cycle_*_summary.csv"))
summary_frames = [pd.read_csv(p) for p in all_summary_files]
df_cumulative = pd.concat(summary_frames, ignore_index=True)
df_cumulative = df_cumulative.sort_values("cycle_index").reset_index(drop=True)

df_cumulative.to_csv(OUT_DIR / "all_cycles_summary.csv", index=False)
df_cumulative.to_csv(TABLE_DIR / "all_cycles_summary.csv", index=False)

# Plot cumulative RMSE trend.
plt.figure(figsize=(11, 6))
plt.plot(df_cumulative["cycle_index"], df_cumulative["best_rmse"], marker="o", linewidth=2.4, label="best RMSE")
plt.plot(df_cumulative["cycle_index"], df_cumulative["median_rmse"], marker="s", linewidth=2.2, label="median RMSE")
plt.grid(True, alpha=0.35)
plt.xlabel("Cycle index")
plt.ylabel("RMSE [V]")
plt.title(f"{MODEL_ID}: warm-continuation RMSE trend")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "rmse_best_median_vs_cycle.png", dpi=280, bbox_inches="tight")
plt.close()

# Plot key parameters if available.
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

available_params = [c for c in param_cols if c in df_all.columns]

# Build best parameter table across cycles from best_run files.
best_files = sorted(TABLE_DIR.glob("cycle_*_best_run.csv"))
best_frames = []

for p in best_files:
    d = pd.read_csv(p)
    if len(d):
        best_frames.append(d.head(1))

if best_frames:
    df_best_all = pd.concat(best_frames, ignore_index=True)
    df_best_all = df_best_all.sort_values("cycle_index").reset_index(drop=True)
    df_best_all.to_csv(OUT_DIR / "all_cycles_best_runs.csv", index=False)
    df_best_all.to_csv(TABLE_DIR / "all_cycles_best_runs.csv", index=False)

    for c in param_cols:
        if c not in df_best_all.columns:
            continue

        plt.figure(figsize=(11, 6))
        plt.plot(df_best_all["cycle_index"], df_best_all[c], marker="o", linewidth=2.4)
        plt.grid(True, alpha=0.35)
        plt.xlabel("Cycle index")
        plt.ylabel(c)
        plt.title(f"{MODEL_ID}: best {c} vs cycle")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"best_{c}_vs_cycle.png", dpi=280, bbox_inches="tight")
        plt.close()

    # Normalized parameter plot.
    plt.figure(figsize=(12, 7))

    for c in param_cols:
        if c not in df_best_all.columns:
            continue

        y = pd.to_numeric(df_best_all[c], errors="coerce").to_numpy(dtype=float)
        if len(y) == 0 or not np.isfinite(y[0]) or abs(y[0]) < 1e-300:
            continue

        plt.plot(df_best_all["cycle_index"], y / y[0], marker="o", linewidth=2.0, label=c)

    plt.grid(True, alpha=0.35)
    plt.xlabel("Cycle index")
    plt.ylabel("Normalized value relative to cycle 0")
    plt.title(f"{MODEL_ID}: normalized best parameters vs cycle")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "normalized_best_parameters_vs_cycle.png", dpi=280, bbox_inches="tight")
    plt.close()

print("\nBest run:")
print(df_best.to_string(index=False))

print("\nCycle summary:")
print(df_cycle_summary.to_string(index=False))

print("\nCumulative summary saved:")
print(" ", OUT_DIR / "all_cycles_summary.csv")
print(" ", TABLE_DIR / "all_cycles_summary.csv")
print("=" * 100)
print("COMBINE COMPLETE")
print("=" * 100)