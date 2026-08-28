#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_s7_s17_final8_parameter_review.py

Purpose
-------
Final compact parameter review for Chapter 6 after supervisor feedback.

Focus only on the 8-model S7/S17 comparison:

    S7_C1,  S7_C2,  S7_C3,  S7_C4K
    S17_C1, S17_C2, S17_C3, S17_C4K

Why this script exists
----------------------
The full 16-model parameter comparison is too visually cluttered.
This script keeps only S7 and S17, and uses C4K as the final quartic model.

Important distinction
---------------------
C1/C2/C3 are from the old state-order grid script.
C4K is from the direct-k electrolyte-coupling script.

Therefore:
    - Core solid/gain parameters can be compared across all 8 models:
        alpha_n, alpha_p, g_n, g_p

    - Electrolyte parameters are model-family specific:
        Old grid C1/C2/C3:
            K_e, g_e

        C4K:
            S7_C4K:  k1, k2, b_e,n, b_e,p
            S17_C4K: k1, k2, k3, k4, k5, b_e,n, b_e,p

    - theta_n0 and theta_p0:
        Old grid models estimate them.
        C4K freezes them to nominal values.

Main outputs
------------
Tables:
    results/tables/chapter6_s7_s17_final8_parameter_review/
        final8_parameter_definitions.csv
        final8_core_parameters_long.csv
        final8_electrolyte_parameters_long.csv
        final8_parameter_summary.csv
        final8_s7_s17_core_similarity.csv

Figures:
    results/figures/chapter6_s7_s17_final8_parameter_review/
    figures/chapter6/s7_s17_final8_parameter_review/

    Per-model dashboards:
        fig_ch6_final8_core_params_S7_C1_raw.png
        ...
        fig_ch6_final8_core_params_S17_C4K_raw.png

    S7 vs S17 by candidate:
        fig_ch6_final8_core_compare_C1_raw.png
        fig_ch6_final8_core_compare_C1_normalized.png
        ...
        fig_ch6_final8_core_compare_C4K_raw.png
        fig_ch6_final8_core_compare_C4K_normalized.png

    Electrolyte family-specific:
        fig_ch6_final8_grid_electrolyte_C1.png
        fig_ch6_final8_grid_electrolyte_C2.png
        fig_ch6_final8_grid_electrolyte_C3.png
        fig_ch6_final8_c4k_electrolyte_S7_S17.png
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

GRID_ROOT = PROJECT / "results" / "real_cycle_ctid_state_order_grid"
KPARAM_ROOT = PROJECT / "results" / "real_warm_continuation_ctid"

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "chapter6_s7_s17_final8_parameter_review"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "chapter6_s7_s17_final8_parameter_review"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6" / "s7_s17_final8_parameter_review"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6" / "s7_s17_final8_parameter_review"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

CYCLE_START = 34
CYCLE_END = 99

FINAL8_MODELS = [
    "S7_C1",
    "S7_C2",
    "S7_C3",
    "S7_C4K",
    "S17_C1",
    "S17_C2",
    "S17_C3",
    "S17_C4K",
]

STATE_ORDER = ["S7", "S17"]
CANDIDATE_ORDER = ["C1", "C2", "C3", "C4K"]

CORE_PARAMS = ["alpha_n", "alpha_p", "g_n", "g_p"]

GRID_ELECTROLYTE_PARAMS = ["K_e", "g_e"]
C4K_ELECTROLYTE_PARAMS = ["b_e,n", "b_e,p", "k1", "k2", "k3", "k4", "k5"]

# optional old grid theta offsets
THETA_PARAMS = ["theta_n0", "theta_p0"]

FIG_DPI = 300
TITLE_SIZE = 15
AXIS_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 9
LINE_WIDTH = 1.9
MARKER_SIZE = 4.2


# ============================================================
# Helpers
# ============================================================

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("[saved]", path)

    for target in [THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / path.name)
        except Exception as exc:
            warnings.warn(f"Could not copy {path.name} to {target}: {exc}")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def normalize_param_name(name: str) -> str:
    s = str(name).strip()

    aliases = {
        "alpha_n_hat": "alpha_n",
        "alpha_p_hat": "alpha_p",
        "an": "alpha_n",
        "ap": "alpha_p",

        "K_e_hat": "K_e",
        "ke": "K_e",

        "g_n_hat": "g_n",
        "g_p_hat": "g_p",
        "g_e_hat": "g_e",
        "gn": "g_n",
        "gp": "g_p",
        "ge": "g_e",

        "theta_n0_hat": "theta_n0",
        "theta_p0_hat": "theta_p0",
        "theta_n_0": "theta_n0",
        "theta_p_0": "theta_p0",

        "b_en": "b_e,n",
        "b_ep": "b_e,p",
        "b_e_n": "b_e,n",
        "b_e_p": "b_e,p",
        "be_n": "b_e,n",
        "be_p": "b_e,p",
        "ben": "b_e,n",
        "bep": "b_e,p",

        "k_1": "k1",
        "k_2": "k2",
        "k_3": "k3",
        "k_4": "k4",
        "k_5": "k5",
    }

    return aliases.get(s, s)


def model_sort_key(model_id: str) -> tuple[int, int, str]:
    state, cand = model_id.split("_", 1)

    sidx = STATE_ORDER.index(state) if state in STATE_ORDER else 999
    cidx = CANDIDATE_ORDER.index(cand) if cand in CANDIDATE_ORDER else 999

    return sidx, cidx, model_id


def parse_model_cycle_from_path(path: Path) -> dict | None:
    s = str(path)

    # Grid: anchor6_S7_C1_... or full16rem_S17_C3_...
    m = re.search(
        r"(anchor6|full16rem)_(S7|S17)_(C1|C2|C3)_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
        s,
    )
    if m:
        prefix, state, cand, nseeds, cycle, seed0, dt = m.groups()
        cycle = int(cycle)
        return {
            "run_prefix": prefix,
            "state_id": state,
            "candidate_id": cand,
            "model_id": f"{state}_{cand}",
            "cycle_index": cycle,
            "retained_cycle_index": cycle - CYCLE_START,
            "n_multistart": int(nseeds),
            "seed0": int(seed0),
            "id_downsample_dt": float(dt),
            "model_family": "grid",
        }

    # K-param: anchor6_S7_C4K_...
    m = re.search(
        r"(anchor6|full16rem)_(S7|S17)_C4K_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
        s,
    )
    if m:
        prefix, state, nseeds, cycle, seed0, dt = m.groups()
        cycle = int(cycle)
        return {
            "run_prefix": prefix,
            "state_id": state,
            "candidate_id": "C4K",
            "model_id": f"{state}_C4K",
            "cycle_index": cycle,
            "retained_cycle_index": cycle - CYCLE_START,
            "n_multistart": int(nseeds),
            "seed0": int(seed0),
            "id_downsample_dt": float(dt),
            "model_family": "direct_k_c4k",
        }

    return None


def pick_name_value_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    name_candidates = ["parameter", "param", "parameter_name", "name", "variable", "parameter_id"]
    value_candidates = ["value", "estimate", "best_value", "parameter_value", "estimated_value", "param_value", "val"]

    name_col = None
    value_col = None

    for c in name_candidates:
        if c in df.columns:
            name_col = c
            break

    for c in value_candidates:
        if c in df.columns:
            value_col = c
            break

    return name_col, value_col


def param_category(parameter: str, model_family: str) -> str:
    if parameter in CORE_PARAMS:
        return "shared_core"
    if model_family == "grid" and parameter in GRID_ELECTROLYTE_PARAMS:
        return "grid_electrolyte"
    if model_family == "grid" and parameter in THETA_PARAMS:
        return "grid_theta_offsets"
    if model_family == "direct_k_c4k" and parameter in C4K_ELECTROLYTE_PARAMS:
        return "c4k_direct_electrolyte"
    return "other"


def long_table_to_rows(df: pd.DataFrame, meta: dict, source_file: Path, source_kind: str) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    name_col, value_col = pick_name_value_columns(df)

    if name_col is None or value_col is None:
        return wide_table_to_rows(df, meta, source_file, source_kind)

    rows = []

    for _, r in df.iterrows():
        pname = normalize_param_name(r[name_col])
        val = pd.to_numeric(pd.Series([r[value_col]]), errors="coerce").iloc[0]

        if not np.isfinite(val):
            continue

        cat = param_category(pname, meta["model_family"])

        if cat == "other":
            continue

        item = dict(meta)
        item.update(
            {
                "parameter": pname,
                "parameter_category": cat,
                "value": float(val),
                "source_file": str(source_file),
                "source_kind": source_kind,
            }
        )
        rows.append(item)

    return rows


def wide_table_to_rows(df: pd.DataFrame, meta: dict, source_file: Path, source_kind: str) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    row = df.iloc[0]
    rows = []

    for col in df.columns:
        pname = normalize_param_name(col)
        cat = param_category(pname, meta["model_family"])

        if cat == "other":
            continue

        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]

        if not np.isfinite(val):
            continue

        item = dict(meta)
        item.update(
            {
                "parameter": pname,
                "parameter_category": cat,
                "value": float(val),
                "source_file": str(source_file),
                "source_kind": source_kind,
            }
        )
        rows.append(item)

    return rows


def normalize_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    med = np.nanmedian(y)
    q25 = np.nanpercentile(y, 25)
    q75 = np.nanpercentile(y, 75)
    scale = q75 - q25

    if not np.isfinite(scale) or abs(scale) < 1e-15:
        scale = np.nanstd(y)

    if not np.isfinite(scale) or abs(scale) < 1e-15:
        return y * np.nan

    return (y - med) / scale


def slope_per_cycle(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    try:
        return float(np.polyfit(x[mask], y[mask], 1)[0])
    except Exception:
        return np.nan


def clean_axis(ax) -> None:
    ax.grid(True, alpha=0.30)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


# ============================================================
# Collection
# ============================================================

def collect_grid_rows() -> list[dict]:
    rows = []

    if not GRID_ROOT.exists():
        return rows

    folders = sorted(list(GRID_ROOT.glob("anchor6_*")) + list(GRID_ROOT.glob("full16rem_*")))

    for folder in folders:
        if not folder.is_dir():
            continue

        meta = parse_model_cycle_from_path(folder)

        if meta is None:
            continue

        if meta["model_id"] not in FINAL8_MODELS:
            continue

        if not (CYCLE_START <= meta["cycle_index"] <= CYCLE_END):
            continue

        found = False

        for p in [
            folder / "real_cycle_parameter_long.csv",
            folder / "parameter_long.csv",
        ]:
            df = safe_read_csv(p)
            if df is not None and len(df):
                rows.extend(long_table_to_rows(df, meta, p, "grid_parameter_long"))
                found = True
                break

        if not found:
            for p in [
                folder / "real_cycle_best_runs.csv",
                folder / "real_cycle_model_summary.csv",
                folder / "best_run.csv",
                folder / "summary.csv",
            ]:
                df = safe_read_csv(p)
                if df is not None and len(df):
                    new_rows = wide_table_to_rows(df, meta, p, "grid_wide_fallback")
                    if new_rows:
                        rows.extend(new_rows)
                        found = True
                        break

    return rows


def collect_kparam_rows() -> list[dict]:
    rows = []

    if not KPARAM_ROOT.exists():
        return rows

    folders = sorted(KPARAM_ROOT.glob("S*_C4K/anchor6_*")) + sorted(KPARAM_ROOT.glob("S*_C4K/full16rem_*"))

    for folder in folders:
        if not folder.is_dir():
            continue

        meta = parse_model_cycle_from_path(folder)

        if meta is None:
            continue

        if meta["model_id"] not in FINAL8_MODELS:
            continue

        if not (CYCLE_START <= meta["cycle_index"] <= CYCLE_END):
            continue

        found = False

        for p in [
            folder / "parameter_long.csv",
            folder / "real_cycle_parameter_long.csv",
        ]:
            df = safe_read_csv(p)
            if df is not None and len(df):
                rows.extend(long_table_to_rows(df, meta, p, "kparam_parameter_long"))
                found = True
                break

        if not found:
            for p in [
                folder / "best_run.csv",
                folder / "summary.csv",
                folder / "real_cycle_best_runs.csv",
                folder / "real_cycle_model_summary.csv",
            ]:
                df = safe_read_csv(p)
                if df is not None and len(df):
                    new_rows = wide_table_to_rows(df, meta, p, "kparam_wide_fallback")
                    if new_rows:
                        rows.extend(new_rows)
                        found = True
                        break

    return rows


def collect_all_parameters() -> pd.DataFrame:
    rows = collect_grid_rows() + collect_kparam_rows()

    if not rows:
        raise RuntimeError("No parameter rows found. Check result folder names and CSV files.")

    df = pd.DataFrame(rows)

    df["parameter"] = df["parameter"].map(normalize_param_name)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cycle_index", "value"]).copy()
    df["cycle_index"] = df["cycle_index"].astype(int)
    df["retained_cycle_index"] = df["cycle_index"] - CYCLE_START

    df = df[df["model_id"].isin(FINAL8_MODELS)].copy()
    df = df[df["cycle_index"].between(CYCLE_START, CYCLE_END)].copy()

    df = df.sort_values(
        ["model_id", "cycle_index", "parameter", "source_kind", "source_file"]
    ).drop_duplicates(
        subset=["model_id", "cycle_index", "parameter"],
        keep="first",
    )

    return df.reset_index(drop=True)


# ============================================================
# Tables
# ============================================================

def make_parameter_definitions() -> pd.DataFrame:
    rows = [
        {
            "model_family": "grid C1-C3",
            "model_ids": "S7_C1,S7_C2,S7_C3,S17_C1,S17_C2,S17_C3",
            "parameter": "alpha_n",
            "meaning": "negative-electrode solid diffusion/dynamic rate parameter",
            "comparison_note": "shared core parameter; comparable across S7/S17 and C1-C3/C4K",
        },
        {
            "model_family": "grid C1-C3",
            "model_ids": "S7_C1,S7_C2,S7_C3,S17_C1,S17_C2,S17_C3",
            "parameter": "alpha_p",
            "meaning": "positive-electrode solid diffusion/dynamic rate parameter",
            "comparison_note": "shared core parameter; comparable across S7/S17 and C1-C3/C4K",
        },
        {
            "model_family": "grid C1-C3",
            "model_ids": "S7_C1,S7_C2,S7_C3,S17_C1,S17_C2,S17_C3",
            "parameter": "g_n",
            "meaning": "negative-electrode solid input gain",
            "comparison_note": "shared core parameter; comparable across S7/S17 and C1-C3/C4K",
        },
        {
            "model_family": "grid C1-C3",
            "model_ids": "S7_C1,S7_C2,S7_C3,S17_C1,S17_C2,S17_C3",
            "parameter": "g_p",
            "meaning": "positive-electrode solid input gain",
            "comparison_note": "shared core parameter; comparable across S7/S17 and C1-C3/C4K",
        },
        {
            "model_family": "grid C1-C3",
            "model_ids": "S7_C1,S7_C2,S7_C3,S17_C1,S17_C2,S17_C3",
            "parameter": "K_e",
            "meaning": "single electrolyte diffusion/coupling scale converted to edge weights using geometry",
            "comparison_note": "not directly comparable to C4K k_i parameters",
        },
        {
            "model_family": "grid C1-C3",
            "model_ids": "S7_C1,S7_C2,S7_C3,S17_C1,S17_C2,S17_C3",
            "parameter": "g_e",
            "meaning": "single electrolyte input gain distributed over electrolyte nodes",
            "comparison_note": "not directly comparable to C4K b_e,n and b_e,p",
        },
        {
            "model_family": "direct-k C4K",
            "model_ids": "S7_C4K,S17_C4K",
            "parameter": "k_i",
            "meaning": "directly identified electrolyte edge coupling parameters",
            "comparison_note": "C4K-specific; do not compare directly with old K_e",
        },
        {
            "model_family": "direct-k C4K",
            "model_ids": "S7_C4K,S17_C4K",
            "parameter": "b_e,n",
            "meaning": "direct negative-side electrolyte input gain",
            "comparison_note": "C4K-specific; replaces old electrolyte source distribution",
        },
        {
            "model_family": "direct-k C4K",
            "model_ids": "S7_C4K,S17_C4K",
            "parameter": "b_e,p",
            "meaning": "direct positive-side electrolyte input gain",
            "comparison_note": "C4K-specific; replaces old electrolyte source distribution",
        },
        {
            "model_family": "direct-k C4K",
            "model_ids": "S7_C4K,S17_C4K",
            "parameter": "theta_n0, theta_p0",
            "meaning": "stoichiometry offsets",
            "comparison_note": "frozen in C4K, estimated in old grid models; not used in final trend comparison",
        },
    ]

    return pd.DataFrame(rows)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model_id, parameter), g in df.groupby(["model_id", "parameter"]):
        g = g.sort_values("retained_cycle_index")

        x = g["retained_cycle_index"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)

        n = int(np.isfinite(y).sum())

        if n == 0:
            continue

        med = float(np.nanmedian(y))
        mean = float(np.nanmean(y))
        std = float(np.nanstd(y, ddof=1)) if n > 1 else np.nan
        cv = abs(std) / max(abs(med), 1e-15) if np.isfinite(std) else np.nan

        rows.append(
            {
                "model_id": model_id,
                "state_id": g["state_id"].iloc[0],
                "candidate_id": g["candidate_id"].iloc[0],
                "model_family": g["model_family"].iloc[0],
                "parameter": parameter,
                "parameter_category": g["parameter_category"].iloc[0],
                "n_cycles": n,
                "mean_value": mean,
                "median_value": med,
                "std_value": std,
                "cv_abs": cv,
                "min_value": float(np.nanmin(y)),
                "max_value": float(np.nanmax(y)),
                "slope_per_retained_cycle": slope_per_cycle(x, y),
                "first_value": float(y[np.isfinite(y)][0]),
                "last_value": float(y[np.isfinite(y)][-1]),
            }
        )

    out = pd.DataFrame(rows)
    out["model_sort"] = out["model_id"].map(lambda m: model_sort_key(m)[0] * 10 + model_sort_key(m)[1])
    out["param_sort"] = out["parameter"].map(
        {p: i for i, p in enumerate(CORE_PARAMS + GRID_ELECTROLYTE_PARAMS + C4K_ELECTROLYTE_PARAMS + THETA_PARAMS)}
    ).fillna(999).astype(int)

    return out.sort_values(["model_sort", "param_sort"]).drop(columns=["model_sort", "param_sort"]).reset_index(drop=True)


def make_s7_s17_core_similarity(core: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cand in CANDIDATE_ORDER:
        m7 = f"S7_{cand}"
        m17 = f"S17_{cand}"

        for param in CORE_PARAMS:
            a = core[(core["model_id"] == m7) & (core["parameter"] == param)][
                ["retained_cycle_index", "value"]
            ].rename(columns={"value": "s7_value"})

            b = core[(core["model_id"] == m17) & (core["parameter"] == param)][
                ["retained_cycle_index", "value"]
            ].rename(columns={"value": "s17_value"})

            merged = a.merge(b, on="retained_cycle_index", how="inner").sort_values("retained_cycle_index")

            if len(merged) < 5:
                rows.append(
                    {
                        "candidate_id": cand,
                        "parameter": param,
                        "model_1": m7,
                        "model_2": m17,
                        "n_common_cycles": len(merged),
                        "raw_corr": np.nan,
                        "normalized_corr": np.nan,
                        "normalized_rmse": np.nan,
                        "median_rel_diff": np.nan,
                        "slope_abs_diff": np.nan,
                    }
                )
                continue

            x = merged["retained_cycle_index"].to_numpy(dtype=float)
            y7 = merged["s7_value"].to_numpy(dtype=float)
            y17 = merged["s17_value"].to_numpy(dtype=float)

            mask = np.isfinite(x) & np.isfinite(y7) & np.isfinite(y17)

            if mask.sum() < 5:
                continue

            x = x[mask]
            y7 = y7[mask]
            y17 = y17[mask]

            z7 = normalize_series(y7)
            z17 = normalize_series(y17)

            raw_corr = float(np.corrcoef(y7, y17)[0, 1]) if np.nanstd(y7) > 0 and np.nanstd(y17) > 0 else np.nan
            norm_corr = float(np.corrcoef(z7, z17)[0, 1]) if np.nanstd(z7) > 0 and np.nanstd(z17) > 0 else np.nan
            norm_rmse = float(np.sqrt(np.nanmean((z7 - z17) ** 2)))

            med7 = float(np.nanmedian(y7))
            med17 = float(np.nanmedian(y17))
            median_rel_diff = abs(med7 - med17) / max(abs(med7), abs(med17), 1e-15)

            slope7 = slope_per_cycle(x, y7)
            slope17 = slope_per_cycle(x, y17)
            slope_abs_diff = abs(slope7 - slope17) if np.isfinite(slope7) and np.isfinite(slope17) else np.nan

            rows.append(
                {
                    "candidate_id": cand,
                    "parameter": param,
                    "model_1": m7,
                    "model_2": m17,
                    "n_common_cycles": int(mask.sum()),
                    "raw_corr": raw_corr,
                    "normalized_corr": norm_corr,
                    "normalized_rmse": norm_rmse,
                    "median_rel_diff": median_rel_diff,
                    "slope_abs_diff": slope_abs_diff,
                }
            )

    out = pd.DataFrame(rows)
    out["similarity_score"] = (
        (1.0 - out["normalized_corr"].clip(-1, 1))
        + out["normalized_rmse"]
        + out["median_rel_diff"]
    )

    return out.sort_values(["candidate_id", "similarity_score"]).reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================

def plot_core_per_model(core: pd.DataFrame, model_id: str, normalized: bool = False) -> None:
    d = core[core["model_id"] == model_id].copy()

    if len(d) == 0:
        return

    fig, axes = plt.subplots(4, 1, figsize=(10.6, 10.4), sharex=True)

    for ax, param in zip(axes, CORE_PARAMS):
        g = d[d["parameter"] == param].sort_values("retained_cycle_index")

        if len(g) == 0:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylabel(param, fontsize=AXIS_SIZE)
            clean_axis(ax)
            continue

        x = g["retained_cycle_index"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)

        if normalized:
            y = normalize_series(y)

        ax.plot(x, y, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        ax.set_ylabel(param if not normalized else f"{param}\nnormalized", fontsize=AXIS_SIZE)
        clean_axis(ax)

    title = "normalized core trends" if normalized else "core parameter estimates"
    axes[0].set_title(f"{model_id}: {title}", fontsize=TITLE_SIZE, pad=12)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    fig.tight_layout()

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_final8_core_params_{model_id}_{suffix}.png")


def plot_s7_s17_core_by_candidate(core: pd.DataFrame, candidate_id: str, normalized: bool = False) -> None:
    models = [f"S7_{candidate_id}", f"S17_{candidate_id}"]

    d = core[core["model_id"].isin(models)].copy()

    if len(d) == 0:
        return

    fig, axes = plt.subplots(4, 1, figsize=(10.8, 10.5), sharex=True)

    for ax, param in zip(axes, CORE_PARAMS):
        dp = d[d["parameter"] == param].copy()

        for model_id in models:
            g = dp[dp["model_id"] == model_id].sort_values("retained_cycle_index")

            if len(g) == 0:
                continue

            x = g["retained_cycle_index"].to_numpy(dtype=float)
            y = g["value"].to_numpy(dtype=float)

            if normalized:
                y = normalize_series(y)

            ax.plot(
                x,
                y,
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=model_id,
            )

        ax.set_ylabel(param if not normalized else f"{param}\nnormalized", fontsize=AXIS_SIZE)
        clean_axis(ax)

    title = "normalized trend" if normalized else "raw value"
    axes[0].set_title(f"S7 vs S17 core parameters — {candidate_id} ({title})", fontsize=TITLE_SIZE, pad=12)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_final8_core_compare_{candidate_id}_{suffix}.png")


def plot_grid_electrolyte_by_candidate(elec: pd.DataFrame, candidate_id: str) -> None:
    models = [f"S7_{candidate_id}", f"S17_{candidate_id}"]

    d = elec[
        (elec["candidate_id"] == candidate_id)
        & (elec["parameter"].isin(GRID_ELECTROLYTE_PARAMS))
    ].copy()

    if len(d) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.7), sharex=True)

    for ax, param in zip(axes, GRID_ELECTROLYTE_PARAMS):
        dp = d[d["parameter"] == param].copy()

        for model_id in models:
            g = dp[dp["model_id"] == model_id].sort_values("retained_cycle_index")

            if len(g) == 0:
                continue

            ax.plot(
                g["retained_cycle_index"],
                g["value"],
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=model_id,
            )

        ax.set_ylabel(param, fontsize=AXIS_SIZE)
        clean_axis(ax)

    axes[0].set_title(f"Grid electrolyte parameters — {candidate_id}", fontsize=TITLE_SIZE, pad=12)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(OUT_FIG_DIR / f"fig_ch6_final8_grid_electrolyte_{candidate_id}.png")


def plot_c4k_electrolyte(elec: pd.DataFrame) -> None:
    d = elec[elec["candidate_id"] == "C4K"].copy()

    if len(d) == 0:
        return

    params = ["b_e,n", "b_e,p", "k1", "k2", "k3", "k4", "k5"]

    fig, axes = plt.subplots(len(params), 1, figsize=(11.0, 2.35 * len(params)), sharex=True)

    for ax, param in zip(axes, params):
        dp = d[d["parameter"] == param].copy()

        for model_id in ["S7_C4K", "S17_C4K"]:
            g = dp[dp["model_id"] == model_id].sort_values("retained_cycle_index")

            if len(g) == 0:
                continue

            ax.plot(
                g["retained_cycle_index"],
                g["value"],
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=model_id,
            )

        ax.set_ylabel(param, fontsize=AXIS_SIZE)
        clean_axis(ax)

    axes[0].set_title("C4K direct electrolyte parameters — S7 vs S17", fontsize=TITLE_SIZE, pad=12)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(OUT_FIG_DIR / "fig_ch6_final8_c4k_electrolyte_S7_S17.png")


def plot_summary_heatmap(summary: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    d = summary[summary["parameter_category"] == "shared_core"].copy()

    models = FINAL8_MODELS
    params = CORE_PARAMS

    mat = np.full((len(params), len(models)), np.nan)

    for i, p in enumerate(params):
        for j, m in enumerate(models):
            row = d[(d["parameter"] == p) & (d["model_id"] == m)]
            if len(row):
                mat[i, j] = pd.to_numeric(row.iloc[0][metric], errors="coerce")

    fig, ax = plt.subplots(figsize=(10.8, 4.8))

    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=TICK_SIZE)

    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels(params, fontsize=TICK_SIZE)

    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, fontsize=AXIS_SIZE)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / filename)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("CHAPTER 6 S7/S17 FINAL 8 PARAMETER REVIEW")
    print("=" * 100)
    print("PROJECT:", PROJECT)
    print("GRID_ROOT:", GRID_ROOT)
    print("KPARAM_ROOT:", KPARAM_ROOT)
    print("OUT_TABLE_DIR:", OUT_TABLE_DIR)
    print("OUT_FIG_DIR:", OUT_FIG_DIR)
    print("Cycles:", CYCLE_START, "to", CYCLE_END)
    print("Models:", ", ".join(FINAL8_MODELS))
    print("=" * 100)

    df = collect_all_parameters()

    definitions = make_parameter_definitions()
    definitions_path = OUT_TABLE_DIR / "final8_parameter_definitions.csv"
    definitions.to_csv(definitions_path, index=False)
    print("[saved]", definitions_path)

    all_path = OUT_TABLE_DIR / "final8_all_parameters_long.csv"
    df.to_csv(all_path, index=False)
    print("[saved]", all_path)

    core = df[df["parameter_category"] == "shared_core"].copy()
    electrolyte = df[df["parameter_category"].isin(["grid_electrolyte", "c4k_direct_electrolyte"])].copy()

    core_path = OUT_TABLE_DIR / "final8_core_parameters_long.csv"
    elec_path = OUT_TABLE_DIR / "final8_electrolyte_parameters_long.csv"

    core.to_csv(core_path, index=False)
    electrolyte.to_csv(elec_path, index=False)

    print("[saved]", core_path)
    print("[saved]", elec_path)

    summary = make_summary(df)

    summary_path = OUT_TABLE_DIR / "final8_parameter_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[saved]", summary_path)

    similarity = make_s7_s17_core_similarity(core)

    similarity_path = OUT_TABLE_DIR / "final8_s7_s17_core_similarity.csv"
    similarity.to_csv(similarity_path, index=False)
    print("[saved]", similarity_path)

    print()
    print("Parameter availability:")
    avail = (
        df.groupby(["model_id", "parameter"], as_index=False)
        .agg(n_cycles=("cycle_index", "nunique"))
        .pivot(index="model_id", columns="parameter", values="n_cycles")
        .fillna(0)
        .astype(int)
    )

    avail = avail.reindex(index=[m for m in FINAL8_MODELS if m in avail.index])
    print(avail.to_string())

    print()
    print("S7 vs S17 core similarity:")
    show_cols = [
        "candidate_id",
        "parameter",
        "normalized_corr",
        "normalized_rmse",
        "median_rel_diff",
        "slope_abs_diff",
        "similarity_score",
    ]
    print(similarity[show_cols].to_string(index=False))

    # Per-model core dashboards.
    for model_id in FINAL8_MODELS:
        plot_core_per_model(core, model_id, normalized=False)
        plot_core_per_model(core, model_id, normalized=True)

    # S7 vs S17 by candidate.
    for cand in CANDIDATE_ORDER:
        plot_s7_s17_core_by_candidate(core, cand, normalized=False)
        plot_s7_s17_core_by_candidate(core, cand, normalized=True)

    # Family-specific electrolyte plots.
    for cand in ["C1", "C2", "C3"]:
        plot_grid_electrolyte_by_candidate(electrolyte, cand)

    plot_c4k_electrolyte(electrolyte)

    # Summary heatmaps.
    plot_summary_heatmap(
        summary,
        metric="median_value",
        filename="fig_ch6_final8_core_median_heatmap.png",
        title="Median Core Parameters Across Retained Cycles",
    )

    plot_summary_heatmap(
        summary,
        metric="cv_abs",
        filename="fig_ch6_final8_core_cv_heatmap.png",
        title="Cycle-to-Cycle Relative Variability of Core Parameters",
    )

    plot_summary_heatmap(
        summary,
        metric="slope_per_retained_cycle",
        filename="fig_ch6_final8_core_slope_heatmap.png",
        title="Linear Trend of Core Parameters Across Retained Cycles",
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Main tables:")
    print(" ", definitions_path)
    print(" ", all_path)
    print(" ", core_path)
    print(" ", elec_path)
    print(" ", summary_path)
    print(" ", similarity_path)
    print()
    print("Main figures:")
    print(" ", OUT_FIG_DIR)
    print(" ", THESIS_FIG_DIR)
    print(" ", FLOW_THESIS_FIG_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
