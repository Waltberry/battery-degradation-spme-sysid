#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
combine_real_cycle_warm_continuation_ctid_kparam.py

Minimal combine script for direct-K/B warm-continuation runs.

Purpose:
    For one model and one cycle:
        1. Find all chunk folders.
        2. Read all_runs.csv from each chunk.
        3. Pick the global best RMSE.
        4. Copy that chunk's best_params_raw.npz into anchors/.
        5. Save combined all_runs, best_run, failed_runs, and summary.

This is required because the next cycle warm-starts from:
    results/real_warm_continuation_ctid/<MODEL_ID>/anchors/cycle_XXXX_best_params_raw.npz
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

MODEL_ID = os.environ.get("UN_MODEL_ID", "").strip()
CYCLE_INDEX = int(os.environ.get("UN_REAL_CYCLE_INDEX", "0"))

if not MODEL_ID:
    raise RuntimeError("Need UN_MODEL_ID, for example S7_C4K or S17_C4K")

ROOT = PROJECT_DIR / "results/real_warm_continuation_ctid" / MODEL_ID
COMBINED_DIR = ROOT / "_combined"
ANCHOR_DIR = ROOT / "anchors"

TABLE_DIR = PROJECT_DIR / "results/tables/real_warm_continuation_ctid" / MODEL_ID

COMBINED_DIR.mkdir(parents=True, exist_ok=True)
ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_if_nonempty(path: Path):
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None

def safe_int_or_nan(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return int(x)
    except Exception:
        return np.nan


def safe_float_or_nan(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return x
    except Exception:
        return np.nan

def main() -> None:
    print("=" * 100)
    print("COMBINE DIRECT-K/B WARM-CONTINUATION CYCLE")
    print("=" * 100)
    print("MODEL_ID:", MODEL_ID)
    print("CYCLE_INDEX:", CYCLE_INDEX)
    print("ROOT:", ROOT)
    print("=" * 100)

    pattern = f"warmseq_{MODEL_ID}_cycle_{CYCLE_INDEX}_*"
    folders = sorted([p for p in ROOT.glob(pattern) if p.is_dir()])

    print(f"Found {len(folders)} folders matching:")
    print(" ", pattern)

    if len(folders) == 0:
        raise RuntimeError(f"No folders found for {MODEL_ID} cycle {CYCLE_INDEX}")

    run_frames = []
    failed_frames = []

    for folder in folders:
        all_runs_path = folder / "all_runs.csv"
        failed_path = folder / "failed_runs.csv"

        df = read_csv_if_nonempty(all_runs_path)
        if df is not None and len(df):
            df["source_folder"] = str(folder)
            df["run_folder"] = folder.name
            run_frames.append(df)

        dfail = read_csv_if_nonempty(failed_path)
        if dfail is not None and len(dfail):
            dfail["source_folder"] = str(folder)
            dfail["run_folder"] = folder.name
            failed_frames.append(dfail)

    if len(run_frames) == 0:
        raise RuntimeError(f"No successful all_runs.csv files found for {MODEL_ID} cycle {CYCLE_INDEX}")

    all_runs = pd.concat(run_frames, ignore_index=True)
    all_runs = all_runs.sort_values("rmse").reset_index(drop=True)

    if len(failed_frames):
        failed_runs = pd.concat(failed_frames, ignore_index=True)
    else:
        failed_runs = pd.DataFrame(
            columns=["model_id", "cycle_index", "seed", "error", "source_folder", "run_folder"]
        )

    best = all_runs.head(1).copy()
    best_row = best.iloc[0]

    best_source_folder = Path(str(best_row["source_folder"]))
    best_params_src = best_source_folder / "best_params_raw.npz"

    if not best_params_src.exists():
        print(f"[warn] Missing best params file for selected row: {best_params_src}")
        print("[warn] Re-selecting best row among folders that have best_params_raw.npz...")

        if "source_folder" not in all_runs.columns:
            raise RuntimeError("Cannot re-select best row because all_runs has no source_folder column.")

        has_params = []
        for _, r in all_runs.iterrows():
            sf = Path(str(r["source_folder"]))
            has_params.append((sf / "best_params_raw.npz").exists())

        valid_runs = all_runs.loc[has_params].copy()

        if len(valid_runs) == 0:
            raise FileNotFoundError(
                "No valid run folder has best_params_raw.npz for this cycle. "
                "At least one chunk must complete fully."
            )

        valid_runs = valid_runs.sort_values("rmse").reset_index(drop=True)

        best = valid_runs.head(1).copy()
        best_row = best.iloc[0]
        best_source_folder = Path(str(best_row["source_folder"]))
        best_params_src = best_source_folder / "best_params_raw.npz"

        print("[warn] New selected best row:")
        print(f"  seed: {best_row.get('seed', 'NA')}")
        print(f"  rmse: {best_row.get('rmse', 'NA')}")
        print(f"  source_folder: {best_row.get('source_folder', 'NA')}")
        print(f"  best_params_src: {best_params_src}")


    anchor_path = ANCHOR_DIR / f"cycle_{CYCLE_INDEX:04d}_best_params_raw.npz"
    shutil.copy2(best_params_src, anchor_path)

    print("[saved anchor]", anchor_path)

    # Save per-cycle combined outputs.
    cycle_all_runs = COMBINED_DIR / f"cycle_{CYCLE_INDEX:04d}_all_runs.csv"
    cycle_best_run = COMBINED_DIR / f"cycle_{CYCLE_INDEX:04d}_best_run.csv"
    cycle_failed_runs = COMBINED_DIR / f"cycle_{CYCLE_INDEX:04d}_failed_runs.csv"
    cycle_summary = COMBINED_DIR / f"cycle_{CYCLE_INDEX:04d}_summary.csv"

    all_runs.to_csv(cycle_all_runs, index=False)
    best.to_csv(cycle_best_run, index=False)
    failed_runs.to_csv(cycle_failed_runs, index=False)

    summary = {
        "model_id": MODEL_ID,
        "cycle_index": CYCLE_INDEX,
        "n_success": int(len(all_runs)),
        "n_fail": int(len(failed_runs)),
        "best_seed": int(best_row["seed"]),
        "best_rmse": float(best_row["rmse"]),
        "best_mae": float(best_row["mae"]),
        "best_r2_percent": float(best_row["r2_percent"]),
        "best_bfr_percent": float(best_row["bfr_percent"]),
        "median_rmse": float(all_runs["rmse"].median()),
        "mean_rmse": float(all_runs["rmse"].mean()),
        "std_rmse": float(all_runs["rmse"].std(ddof=1)) if len(all_runs) > 1 else np.nan,
        "best_rank_phi_raw": safe_int_or_nan(best_row["rank_phi_raw"]),
        "best_ncols_phi_raw": safe_int_or_nan(best_row["ncols_phi_raw"]),
        "best_cond_phi_raw": safe_float_or_nan(best_row["cond_phi_raw"]),
        "best_rank_X_raw": safe_int_or_nan(best_row["rank_X_raw"]),
        "best_ncols_X_raw": safe_int_or_nan(best_row["ncols_X_raw"]),
        "best_cond_X_raw": safe_float_or_nan(best_row["cond_X_raw"]),
        "best_source_folder": str(best_source_folder),
        "anchor_path": str(anchor_path),
    }

    pd.DataFrame([summary]).to_csv(cycle_summary, index=False)

    # Update cumulative tables.
    existing_summary_files = sorted(COMBINED_DIR.glob("cycle_*_summary.csv"))
    existing_best_files = sorted(COMBINED_DIR.glob("cycle_*_best_run.csv"))

    all_summary = pd.concat([pd.read_csv(p) for p in existing_summary_files], ignore_index=True)
    all_summary = all_summary.drop_duplicates(subset=["cycle_index"], keep="last")
    all_summary = all_summary.sort_values("cycle_index").reset_index(drop=True)

    all_best = pd.concat([pd.read_csv(p) for p in existing_best_files], ignore_index=True)
    all_best = all_best.drop_duplicates(subset=["cycle_index"], keep="last")
    all_best = all_best.sort_values("cycle_index").reset_index(drop=True)

    all_summary.to_csv(COMBINED_DIR / "all_cycles_summary.csv", index=False)
    all_best.to_csv(COMBINED_DIR / "all_cycles_best_runs.csv", index=False)

    all_summary.to_csv(TABLE_DIR / "all_cycles_summary.csv", index=False)
    all_best.to_csv(TABLE_DIR / "all_cycles_best_runs.csv", index=False)

    print()
    print("Best run:")
    print(best.to_string(index=False))
    print()
    print("Cycle summary:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print()
    print("[saved]", cycle_all_runs)
    print("[saved]", cycle_best_run)
    print("[saved]", cycle_summary)
    print("[saved cumulative]", TABLE_DIR / "all_cycles_summary.csv")
    print("[saved cumulative]", TABLE_DIR / "all_cycles_best_runs.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()