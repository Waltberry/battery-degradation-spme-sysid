#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% =====================================================
# CELL 0 — Imports
# =====================================================
"""
summarize_real_10cycles_s17c4_trends.py

Purpose
-------
Combine chunked S17_C4 real-data CT-ID results over 10 consecutive discharge cycles.

Expected input folders
----------------------
results/real_cycle_ctid_state_order_grid/
    real10_S17_C4_cycle_<cycle>_chunk_<chunk>_<n>seeds_<seed0>_to_<seed1>_dt_1.0_bins_100/

Each folder should contain:
    real_cycle_all_runs.csv
    real_cycle_best_runs.csv
    real_cycle_model_summary.csv
    real_cycle_parameter_long.csv
    real_cycle_beta_coefficients.csv
    selected_real_cycle_id_data.csv

If the step-response patch is active, each folder may also contain:
    best_step_response_manifest.csv
    best_step_response_timeseries/*.csv

Outputs
-------
results/real_10cycle_s17c4_summary/
results/figures/real_10cycle_s17c4_summary/
results/tables/real_10cycle_s17c4_summary/
"""

from __future__ import annotations

from pathlib import Path
import re
import json
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

OUT_DIR = PROJECT_DIR / "results" / "real_10cycle_s17c4_summary"
FIG_DIR = PROJECT_DIR / "results" / "figures" / "real_10cycle_s17c4_summary"
TABLE_DIR = PROJECT_DIR / "results" / "tables" / "real_10cycle_s17c4_summary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RUN_PREFIX = "real10_S17_C4_cycle_"

PARAMETER_COLS = [
    "alpha_n_hat",
    "alpha_p_hat",
    "K_e_hat",
    "g_n_hat",
    "g_p_hat",
    "g_e_hat",
    "theta_n0_hat",
    "theta_p0_hat",
]

RANK_COLS = [
    "rank_phi_raw",
    "ncols_phi_raw",
    "cond_phi_raw",
    "rank_X_raw",
    "ncols_X_raw",
    "cond_X_raw",
]

print("=" * 100)
print("REAL 10-CYCLE S17_C4 SUMMARY")
print("=" * 100)
print("PROJECT_DIR:", PROJECT_DIR)
print("RESULT_ROOT:", RESULT_ROOT)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)
print("TABLE_DIR:", TABLE_DIR)
print("=" * 100)


# %% =====================================================
# CELL 2 — Helpers
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
    )


def infer_cycle_chunk_seed_from_folder(folder_name: str) -> dict[str, int | None]:
    out = {
        "cycle_index": None,
        "chunk_index": None,
        "seed0": None,
        "seed1": None,
        "n_seeds": None,
    }

    m = re.search(r"cycle_(\d+)_chunk_(\d+)_([0-9]+)seeds_([0-9]+)_to_([0-9]+)", folder_name)
    if m:
        out["cycle_index"] = int(m.group(1))
        out["chunk_index"] = int(m.group(2))
        out["n_seeds"] = int(m.group(3))
        out["seed0"] = int(m.group(4))
        out["seed1"] = int(m.group(5))
        return out

    m = re.search(r"cycle(\d+).*chunk_(\d+)", folder_name)
    if m:
        out["cycle_index"] = int(m.group(1))
        out["chunk_index"] = int(m.group(2))

    return out


def finite_series(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def matrix_rank_condition_svd_np(M: np.ndarray, rel_tol: float = 1e-10) -> dict:
    M = np.asarray(M, dtype=float)

    if M.ndim != 2 or M.size == 0:
        return {
            "rank": 0,
            "n_rows": int(M.shape[0]) if M.ndim == 2 else 0,
            "n_cols": int(M.shape[1]) if M.ndim == 2 else 0,
            "condition_number": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    try:
        svals = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "rank": np.nan,
            "n_rows": int(M.shape[0]),
            "n_cols": int(M.shape[1]),
            "condition_number": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    if len(svals) == 0:
        return {
            "rank": 0,
            "n_rows": int(M.shape[0]),
            "n_cols": int(M.shape[1]),
            "condition_number": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    s0 = float(svals[0])
    threshold = rel_tol * max(s0, 1e-300)
    rank = int(np.sum(svals > threshold))
    cond = float(svals[0] / max(float(svals[-1]), 1e-300))

    return {
        "rank": rank,
        "n_rows": int(M.shape[0]),
        "n_cols": int(M.shape[1]),
        "condition_number": cond,
        "singular_values": [float(v) for v in svals],
        "relative_singular_values": [float(v / max(s0, 1e-300)) for v in svals],
    }


def dynamic_rank_diagnostics(M: np.ndarray, rel_tol: float = 1e-10) -> dict:
    """
    Dynamic rank removes the initial row from every column.

    This answers:
        During the cycle, how many independent directions actually change?

    It is not a replacement for raw rank. It is an added diagnostic.
    """
    M = np.asarray(M, dtype=float)

    if M.ndim != 2 or M.size == 0:
        return matrix_rank_condition_svd_np(M, rel_tol=rel_tol)

    Md = M - M[0:1, :]

    # Drop columns with almost zero norm after differencing.
    norms = np.linalg.norm(Md, axis=0)
    keep = norms > 1e-14

    if not np.any(keep):
        return {
            "rank": 0,
            "n_rows": int(M.shape[0]),
            "n_cols": int(M.shape[1]),
            "n_cols_active": 0,
            "condition_number": np.nan,
            "singular_values": [],
            "relative_singular_values": [],
        }

    ans = matrix_rank_condition_svd_np(Md[:, keep], rel_tol=rel_tol)
    ans["n_cols_active"] = int(np.sum(keep))
    return ans


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print("[warn] failed to read", path, repr(exc))
        return None


# %% =====================================================
# CELL 3 — Discover result folders
# =====================================================
folders = sorted([
    p for p in RESULT_ROOT.glob(f"{RUN_PREFIX}*")
    if p.is_dir() and (p / "real_cycle_all_runs.csv").exists()
])

if not folders:
    raise RuntimeError(f"No folders found matching {RUN_PREFIX}* under {RESULT_ROOT}")

print("Found folders:", len(folders))
for p in folders[:20]:
    print(" ", p.name)
if len(folders) > 20:
    print(" ...")


# %% =====================================================
# CELL 4 — Load all chunk tables
# =====================================================
all_frames = []
best_frames = []
summary_frames = []
param_frames = []
beta_frames = []
config_rows = []
response_manifest_frames = []

for folder in folders:
    info = infer_cycle_chunk_seed_from_folder(folder.name)

    cycle_index = info["cycle_index"]
    chunk_index = info["chunk_index"]

    all_path = folder / "real_cycle_all_runs.csv"
    best_path = folder / "real_cycle_best_runs.csv"
    summary_path = folder / "real_cycle_model_summary.csv"
    param_path = folder / "real_cycle_parameter_long.csv"
    beta_path = folder / "real_cycle_beta_coefficients.csv"
    config_path = folder / "real_cycle_config.json"
    response_manifest_path = folder / "best_step_response_manifest.csv"

    df_all = read_csv_if_exists(all_path)
    if df_all is not None:
        df_all["cycle_index"] = cycle_index
        df_all["chunk_index"] = chunk_index
        df_all["run_folder"] = folder.name
        df_all["source_folder"] = str(folder)
        all_frames.append(df_all)

    df_best = read_csv_if_exists(best_path)
    if df_best is not None:
        df_best["cycle_index"] = cycle_index
        df_best["chunk_index"] = chunk_index
        df_best["run_folder"] = folder.name
        df_best["source_folder"] = str(folder)
        best_frames.append(df_best)

    df_summary = read_csv_if_exists(summary_path)
    if df_summary is not None:
        df_summary["cycle_index"] = cycle_index
        df_summary["chunk_index"] = chunk_index
        df_summary["run_folder"] = folder.name
        df_summary["source_folder"] = str(folder)
        summary_frames.append(df_summary)

    df_param = read_csv_if_exists(param_path)
    if df_param is not None:
        df_param["cycle_index"] = cycle_index
        df_param["chunk_index"] = chunk_index
        df_param["run_folder"] = folder.name
        df_param["source_folder"] = str(folder)
        param_frames.append(df_param)

    df_beta = read_csv_if_exists(beta_path)
    if df_beta is not None:
        df_beta["cycle_index"] = cycle_index
        df_beta["chunk_index"] = chunk_index
        df_beta["run_folder"] = folder.name
        df_beta["source_folder"] = str(folder)
        beta_frames.append(df_beta)

    df_manifest = read_csv_if_exists(response_manifest_path)
    if df_manifest is not None:
        df_manifest["cycle_index"] = cycle_index
        df_manifest["chunk_index"] = chunk_index
        df_manifest["run_folder"] = folder.name
        df_manifest["source_folder"] = str(folder)
        response_manifest_frames.append(df_manifest)

    cfg_payload = {}
    if config_path.exists():
        try:
            cfg_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            cfg_payload = {}

    config_rows.append({
        "cycle_index": cycle_index,
        "chunk_index": chunk_index,
        "run_folder": folder.name,
        "source_folder": str(folder),
        "seed0": info["seed0"],
        "seed1": info["seed1"],
        "n_seeds": info["n_seeds"],
        "config_path": str(config_path) if config_path.exists() else "",
    })

if not all_frames:
    raise RuntimeError("No all-run CSVs loaded.")

df_all = pd.concat(all_frames, ignore_index=True)
df_best_chunks = pd.concat(best_frames, ignore_index=True) if best_frames else pd.DataFrame()
df_summary_chunks = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
df_params = pd.concat(param_frames, ignore_index=True) if param_frames else pd.DataFrame()
df_beta = pd.concat(beta_frames, ignore_index=True) if beta_frames else pd.DataFrame()
df_configs = pd.DataFrame(config_rows)
df_response_manifest = pd.concat(response_manifest_frames, ignore_index=True) if response_manifest_frames else pd.DataFrame()

# Make sure numeric fields are numeric.
for col in ["cycle_index", "chunk_index", "seed", "rmse", "mae", "r2_percent", "bfr_percent"]:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

df_all = df_all.sort_values(["cycle_index", "rmse"]).reset_index(drop=True)

df_all.to_csv(OUT_DIR / "combined_all_runs.csv", index=False)
df_all.to_csv(TABLE_DIR / "combined_all_runs.csv", index=False)

df_configs.to_csv(OUT_DIR / "run_folder_audit.csv", index=False)
df_configs.to_csv(TABLE_DIR / "run_folder_audit.csv", index=False)

if len(df_response_manifest):
    df_response_manifest.to_csv(OUT_DIR / "combined_step_response_manifest.csv", index=False)
    df_response_manifest.to_csv(TABLE_DIR / "combined_step_response_manifest.csv", index=False)

print("\nCombined all runs:", df_all.shape)
print("Cycles found:", sorted(df_all["cycle_index"].dropna().unique().astype(int).tolist()))
print("Total successful fits:", len(df_all))


# %% =====================================================
# CELL 5 — Completion audit
# =====================================================
audit_rows = []

for cycle in range(10):
    g = df_all[df_all["cycle_index"] == cycle]
    chunks = sorted(g["chunk_index"].dropna().astype(int).unique().tolist())
    seeds = sorted(g["seed"].dropna().astype(int).unique().tolist()) if "seed" in g.columns else []

    audit_rows.append({
        "cycle_index": cycle,
        "found_runs": int(len(g)),
        "expected_runs": 100,
        "missing_runs": int(100 - len(g)),
        "found_chunks": len(chunks),
        "chunk_indices": ",".join(map(str, chunks)),
        "min_seed": min(seeds) if seeds else np.nan,
        "max_seed": max(seeds) if seeds else np.nan,
        "complete": bool(len(g) == 100),
    })

df_audit = pd.DataFrame(audit_rows)
df_audit.to_csv(OUT_DIR / "completion_audit.csv", index=False)
df_audit.to_csv(TABLE_DIR / "completion_audit.csv", index=False)

print("\nCompletion audit:")
print(df_audit.to_string(index=False))


# %% =====================================================
# CELL 6 — Best run and summary per cycle
# =====================================================
best_rows = []
summary_rows = []

for cycle, g0 in df_all.groupby("cycle_index"):
    g = g0.sort_values("rmse").reset_index(drop=True)
    best = g.iloc[0].copy()
    best_rows.append(best)

    row = {
        "cycle_index": int(cycle),
        "n_runs": int(len(g)),
        "best_seed": int(best["seed"]) if "seed" in best and pd.notna(best["seed"]) else np.nan,
        "best_chunk_index": int(best["chunk_index"]) if "chunk_index" in best and pd.notna(best["chunk_index"]) else np.nan,
        "best_rmse": float(best["rmse"]),
        "best_mae": float(best["mae"]) if "mae" in best else np.nan,
        "best_r2_percent": float(best["r2_percent"]) if "r2_percent" in best else np.nan,
        "best_bfr_percent": float(best["bfr_percent"]) if "bfr_percent" in best else np.nan,
        "median_rmse": float(g["rmse"].median()),
        "mean_rmse": float(g["rmse"].mean()),
        "std_rmse": float(g["rmse"].std(ddof=1)),
        "q05_rmse": float(g["rmse"].quantile(0.05)),
        "q95_rmse": float(g["rmse"].quantile(0.95)),
    }

    for col in PARAMETER_COLS + RANK_COLS:
        if col in best.index:
            row[f"best_{col}"] = best[col]

    summary_rows.append(row)

df_best_by_cycle = pd.DataFrame(best_rows).sort_values("cycle_index").reset_index(drop=True)
df_summary = pd.DataFrame(summary_rows).sort_values("cycle_index").reset_index(drop=True)

df_best_by_cycle.to_csv(OUT_DIR / "best_run_per_cycle.csv", index=False)
df_summary.to_csv(OUT_DIR / "summary_per_cycle.csv", index=False)

df_best_by_cycle.to_csv(TABLE_DIR / "best_run_per_cycle.csv", index=False)
df_summary.to_csv(TABLE_DIR / "summary_per_cycle.csv", index=False)

print("\nBest run per cycle:")
show_cols = [
    "cycle_index",
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
show_cols = [c for c in show_cols if c in df_best_by_cycle.columns]
print(df_best_by_cycle[show_cols].to_string(index=False))


# %% =====================================================
# CELL 7 — Rank re-diagnosis from saved Phi and X CSVs if available
# =====================================================
rank_diag_rows = []

# The manifest usually points to the best response in each chunk.
# We only keep rows matching final best seed per cycle when possible.
if len(df_response_manifest):
    for _, best in df_best_by_cycle.iterrows():
        cycle = int(best["cycle_index"])
        best_seed = int(best["seed"])

        candidates = df_response_manifest[
            (df_response_manifest["cycle_index"] == cycle)
            & (pd.to_numeric(df_response_manifest["seed"], errors="coerce") == best_seed)
        ]

        if len(candidates) == 0:
            continue

        man = candidates.iloc[0]

        phi_path = Path(str(man.get("phi_csv", "")))
        state_path = Path(str(man.get("state_csv", man.get("state_trajectory_csv", ""))))

        row = {
            "cycle_index": cycle,
            "seed": best_seed,
            "phi_csv": str(phi_path) if str(phi_path) else "",
            "state_csv": str(state_path) if str(state_path) else "",
        }

        if phi_path.exists():
            df_phi = pd.read_csv(phi_path)
            phi_cols = [c for c in df_phi.columns if c != "t_s"]
            Phi = df_phi[phi_cols].to_numpy(dtype=float)

            raw = matrix_rank_condition_svd_np(Phi, rel_tol=1e-10)
            dyn = dynamic_rank_diagnostics(Phi, rel_tol=1e-10)

            row.update({
                "phi_rank_raw_recomputed": raw["rank"],
                "phi_ncols_raw_recomputed": raw["n_cols"],
                "phi_cond_raw_recomputed": raw["condition_number"],
                "phi_rank_dynamic": dyn["rank"],
                "phi_ncols_dynamic_active": dyn.get("n_cols_active", np.nan),
                "phi_cond_dynamic": dyn["condition_number"],
                "phi_singular_values_raw": json.dumps(raw["singular_values"]),
                "phi_relative_singular_values_raw": json.dumps(raw["relative_singular_values"]),
            })

        if state_path.exists():
            df_x = pd.read_csv(state_path)
            x_cols = [c for c in df_x.columns if c != "t_s"]
            X = df_x[x_cols].to_numpy(dtype=float)

            raw = matrix_rank_condition_svd_np(X, rel_tol=1e-10)
            dyn = dynamic_rank_diagnostics(X, rel_tol=1e-10)

            row.update({
                "X_rank_raw_recomputed": raw["rank"],
                "X_ncols_raw_recomputed": raw["n_cols"],
                "X_cond_raw_recomputed": raw["condition_number"],
                "X_rank_dynamic": dyn["rank"],
                "X_ncols_dynamic_active": dyn.get("n_cols_active", np.nan),
                "X_cond_dynamic": dyn["condition_number"],
                "X_singular_values_raw": json.dumps(raw["singular_values"]),
                "X_relative_singular_values_raw": json.dumps(raw["relative_singular_values"]),
            })

        rank_diag_rows.append(row)

df_rank_diag = pd.DataFrame(rank_diag_rows)
df_rank_diag.to_csv(OUT_DIR / "best_response_rank_rediagnosis.csv", index=False)
df_rank_diag.to_csv(TABLE_DIR / "best_response_rank_rediagnosis.csv", index=False)

if len(df_rank_diag):
    print("\nRecomputed rank diagnostics from saved best response files:")
    print(df_rank_diag.to_string(index=False))


# %% =====================================================
# CELL 8 — RMSE trend plots
# =====================================================
cycles = df_summary["cycle_index"].to_numpy(dtype=int)

plt.figure(figsize=(10.8, 6.2))
plt.plot(cycles, df_summary["best_rmse"], marker="o", linewidth=2.5, label="best RMSE")
plt.plot(cycles, df_summary["median_rmse"], marker="s", linewidth=2.5, label="median RMSE")
plt.fill_between(
    cycles,
    df_summary["q05_rmse"],
    df_summary["q95_rmse"],
    alpha=0.20,
    label="5-95% RMSE range",
)
plt.grid(True, alpha=0.35)
plt.xlabel("Discharge cycle index")
plt.ylabel("RMSE [V]")
plt.title("S17_C4 real-data fit quality across 10 discharge cycles")
plt.legend(loc="best")
plt.tight_layout()
savefig(FIG_DIR / "rmse_best_median_q05_q95_vs_cycle.png")

plt.figure(figsize=(10.8, 6.2))
plt.scatter(df_all["cycle_index"], df_all["rmse"], s=18, alpha=0.45, label="all seeds")
plt.plot(cycles, df_summary["best_rmse"], marker="o", linewidth=2.5, label="best per cycle")
plt.grid(True, alpha=0.35)
plt.xlabel("Discharge cycle index")
plt.ylabel("RMSE [V]")
plt.title("All S17_C4 RMSE values across cycles")
plt.legend(loc="best")
plt.tight_layout()
savefig(FIG_DIR / "rmse_scatter_all_seeds_vs_cycle.png")


# %% =====================================================
# CELL 9 — Best parameter trends vs cycle
# =====================================================
PARAM_FIG_DIR = FIG_DIR / "best_parameter_trends"
PARAM_FIG_DIR.mkdir(parents=True, exist_ok=True)

for param in PARAMETER_COLS:
    if param not in df_best_by_cycle.columns:
        continue

    plt.figure(figsize=(10.8, 6.2))
    plt.plot(
        df_best_by_cycle["cycle_index"],
        df_best_by_cycle[param],
        marker="o",
        linewidth=2.5,
    )
    plt.grid(True, alpha=0.35)
    plt.xlabel("Discharge cycle index")
    plt.ylabel(param)
    plt.title(f"Best-RMSE estimate of {param} vs cycle, S17_C4")
    plt.tight_layout()
    savefig(PARAM_FIG_DIR / f"best_{safe_name(param)}_vs_cycle.png")

# Combined physical parameters, normalized to first cycle.
plt.figure(figsize=(11.5, 6.8))
for param in PARAMETER_COLS:
    if param not in df_best_by_cycle.columns:
        continue

    y = pd.to_numeric(df_best_by_cycle[param], errors="coerce").to_numpy(dtype=float)
    if len(y) == 0 or not np.isfinite(y[0]) or abs(y[0]) < 1e-300:
        continue

    plt.plot(cycles, y / y[0], marker="o", linewidth=2.0, label=param)

plt.grid(True, alpha=0.35)
plt.xlabel("Discharge cycle index")
plt.ylabel("Normalized value relative to cycle 0")
plt.title("Normalized best-RMSE physical parameter trends, S17_C4")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
savefig(FIG_DIR / "normalized_best_physical_parameters_vs_cycle.png")


# %% =====================================================
# CELL 10 — Parameter boxplots across seeds by cycle
# =====================================================
BOX_DIR = FIG_DIR / "parameter_boxplots_by_cycle"
BOX_DIR.mkdir(parents=True, exist_ok=True)

for param in PARAMETER_COLS:
    if param not in df_all.columns:
        continue

    data = []
    labels = []

    for cycle in range(10):
        vals = finite_series(df_all.loc[df_all["cycle_index"] == cycle, param])
        if len(vals):
            data.append(vals)
            labels.append(str(cycle))

    if not data:
        continue

    plt.figure(figsize=(11.5, 6.5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.grid(True, axis="y", alpha=0.35)
    plt.xlabel("Discharge cycle index")
    plt.ylabel(param)
    plt.title(f"{param} distribution across multistart seeds by cycle, S17_C4")
    plt.tight_layout()
    savefig(BOX_DIR / f"boxplot_{safe_name(param)}_by_cycle.png")


# %% =====================================================
# CELL 11 — Output coefficient trends
# =====================================================
beta_cols = [c for c in df_best_by_cycle.columns if c.startswith("beta_")]

BETA_DIR = FIG_DIR / "best_beta_trends"
BETA_DIR.mkdir(parents=True, exist_ok=True)

for beta_col in beta_cols:
    plt.figure(figsize=(10.8, 6.2))
    plt.plot(
        df_best_by_cycle["cycle_index"],
        df_best_by_cycle[beta_col],
        marker="o",
        linewidth=2.5,
    )
    plt.grid(True, alpha=0.35)
    plt.xlabel("Discharge cycle index")
    plt.ylabel(beta_col)
    plt.title(f"Best-RMSE output coefficient {beta_col} vs cycle, S17_C4")
    plt.tight_layout()
    savefig(BETA_DIR / f"best_{safe_name(beta_col)}_vs_cycle.png")


# %% =====================================================
# CELL 12 — Rank and condition number trends
# =====================================================
if {"rank_phi_raw", "ncols_phi_raw"}.issubset(df_best_by_cycle.columns):
    plt.figure(figsize=(10.8, 6.2))
    plt.plot(cycles, df_best_by_cycle["rank_phi_raw"], marker="o", linewidth=2.5, label="rank(Phi)")
    plt.plot(cycles, df_best_by_cycle["ncols_phi_raw"], "--", linewidth=2.2, label="ncols(Phi)")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Discharge cycle index")
    plt.ylabel("Rank")
    plt.title("Best-run output feature rank vs cycle, S17_C4")
    plt.legend(loc="best")
    plt.tight_layout()
    savefig(FIG_DIR / "rank_phi_vs_cycle.png")

if {"rank_X_raw", "ncols_X_raw"}.issubset(df_best_by_cycle.columns):
    plt.figure(figsize=(10.8, 6.2))
    plt.plot(cycles, df_best_by_cycle["rank_X_raw"], marker="o", linewidth=2.5, label="rank(Xhat)")
    plt.plot(cycles, df_best_by_cycle["ncols_X_raw"], "--", linewidth=2.2, label="ncols(Xhat)")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Discharge cycle index")
    plt.ylabel("Rank")
    plt.title("Best-run fitted state-trajectory rank vs cycle, S17_C4")
    plt.legend(loc="best")
    plt.tight_layout()
    savefig(FIG_DIR / "rank_X_vs_cycle.png")

if "cond_phi_raw" in df_best_by_cycle.columns:
    plt.figure(figsize=(10.8, 6.2))
    plt.semilogy(cycles, df_best_by_cycle["cond_phi_raw"], marker="o", linewidth=2.5)
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Discharge cycle index")
    plt.ylabel("Condition number of Phi")
    plt.title("Best-run output feature condition number vs cycle, S17_C4")
    plt.tight_layout()
    savefig(FIG_DIR / "cond_phi_vs_cycle_log.png")

if "cond_X_raw" in df_best_by_cycle.columns:
    plt.figure(figsize=(10.8, 6.2))
    plt.semilogy(cycles, df_best_by_cycle["cond_X_raw"], marker="o", linewidth=2.5)
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Discharge cycle index")
    plt.ylabel("Condition number of Xhat")
    plt.title("Best-run state trajectory condition number vs cycle, S17_C4")
    plt.tight_layout()
    savefig(FIG_DIR / "cond_X_vs_cycle_log.png")


# %% =====================================================
# CELL 13 — Step-response figure manifest and combined overlays
# =====================================================
STEP_DIR = FIG_DIR / "best_step_response_overlays"
STEP_DIR.mkdir(parents=True, exist_ok=True)

step_rows = []

if len(df_response_manifest):
    for _, best in df_best_by_cycle.iterrows():
        cycle = int(best["cycle_index"])
        seed = int(best["seed"])

        cand = df_response_manifest[
            (df_response_manifest["cycle_index"] == cycle)
            & (pd.to_numeric(df_response_manifest["seed"], errors="coerce") == seed)
        ]

        if len(cand) == 0:
            continue

        row = cand.iloc[0].to_dict()
        response_csv = Path(str(row.get("response_csv", row.get("prediction_csv", ""))))

        if not response_csv.exists():
            continue

        df_resp = pd.read_csv(response_csv)

        required = {"t_s", "measured_voltage_V", "estimated_voltage_V", "residual_V"}
        if not required.issubset(df_resp.columns):
            continue

        t = df_resp["t_s"].to_numpy(dtype=float)
        y = df_resp["measured_voltage_V"].to_numpy(dtype=float)
        yh = df_resp["estimated_voltage_V"].to_numpy(dtype=float)
        e = df_resp["residual_V"].to_numpy(dtype=float)

        plt.figure(figsize=(11.5, 6.0))
        plt.plot(t, y, linewidth=2.6, label="measured")
        plt.plot(t, yh, "--", linewidth=2.4, label="estimated")
        plt.grid(True, alpha=0.35)
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title(f"S17_C4 cycle {cycle}: measured vs estimated, seed={seed}")
        plt.legend(loc="best")
        plt.tight_layout()
        fig_voltage = STEP_DIR / f"cycle_{cycle}_measured_vs_estimated.png"
        savefig(fig_voltage)

        plt.figure(figsize=(11.5, 4.8))
        plt.plot(t, e, linewidth=1.9)
        plt.axhline(0.0, linestyle="--", linewidth=1.2)
        plt.grid(True, alpha=0.35)
        plt.xlabel("Time [s]")
        plt.ylabel("Residual [V]")
        plt.title(f"S17_C4 cycle {cycle}: residual, seed={seed}")
        plt.tight_layout()
        fig_residual = STEP_DIR / f"cycle_{cycle}_residual.png"
        savefig(fig_residual)

        step_rows.append({
            "cycle_index": cycle,
            "seed": seed,
            "response_csv": str(response_csv),
            "voltage_overlay_figure": str(fig_voltage),
            "residual_figure": str(fig_residual),
        })

df_step_summary = pd.DataFrame(step_rows)
df_step_summary.to_csv(OUT_DIR / "best_step_response_summary.csv", index=False)
df_step_summary.to_csv(TABLE_DIR / "best_step_response_summary.csv", index=False)


# %% =====================================================
# CELL 14 — Save LaTeX tables
# =====================================================
def save_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.6g}")
    path.write_text(txt, encoding="utf-8")
    print("[saved latex table]", path)


latex_cols = [
    "cycle_index",
    "n_runs",
    "best_seed",
    "best_rmse",
    "best_mae",
    "best_r2_percent",
    "best_bfr_percent",
    "median_rmse",
    "best_rank_phi_raw",
    "best_ncols_phi_raw",
    "best_rank_X_raw",
    "best_ncols_X_raw",
]
latex_cols = [c for c in latex_cols if c in df_summary.columns]
save_latex_table(df_summary[latex_cols], TABLE_DIR / "latex_summary_per_cycle.tex")

best_cols = [
    "cycle_index",
    "seed",
    "rmse",
    "mae",
    "r2_percent",
    "bfr_percent",
    "alpha_n_hat",
    "alpha_p_hat",
    "K_e_hat",
    "g_n_hat",
    "g_p_hat",
    "g_e_hat",
    "theta_n0_hat",
    "theta_p0_hat",
    "rank_phi_raw",
    "ncols_phi_raw",
    "rank_X_raw",
    "ncols_X_raw",
]
best_cols = [c for c in best_cols if c in df_best_by_cycle.columns]
save_latex_table(df_best_by_cycle[best_cols], TABLE_DIR / "latex_best_parameters_per_cycle.tex")


# %% =====================================================
# CELL 15 — Final printout
# =====================================================
print("\n" + "=" * 100)
print("REAL 10-CYCLE S17_C4 SUMMARY COMPLETE")
print("=" * 100)
print("Tables:", TABLE_DIR)
print("Figures:", FIG_DIR)
print("Main files:")
print(" ", TABLE_DIR / "summary_per_cycle.csv")
print(" ", TABLE_DIR / "best_run_per_cycle.csv")
print(" ", TABLE_DIR / "completion_audit.csv")
print(" ", FIG_DIR / "rmse_best_median_q05_q95_vs_cycle.png")
print(" ", FIG_DIR / "normalized_best_physical_parameters_vs_cycle.png")
print(" ", FIG_DIR / "rank_phi_vs_cycle.png")
print(" ", FIG_DIR / "rank_X_vs_cycle.png")
print("=" * 100)