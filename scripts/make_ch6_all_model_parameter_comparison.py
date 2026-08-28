#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_all_model_parameter_comparison.py

Purpose
-------
Compare estimated parameters across all retained-cycle model fits.

This script does NOT rerun CT-ID. It reads saved fitted result folders from:

    results/real_cycle_ctid_state_order_grid/
    results/real_warm_continuation_ctid/

and generates Chapter 6 parameter-comparison figures/tables.

Main visualization strategy
---------------------------
Avoid one unreadable 16-model plot.

Instead, produce:
    1. C1 model parameter comparison
    2. C2 model parameter comparison
    3. C3 model parameter comparison
    4. C4/C4K model parameter comparison
    5. C4K direct electrolyte parameter comparison
    6. Parameter median heatmaps
    7. Parameter coefficient-of-variation heatmaps

The C4/C4K group can include:
    S7_C4, S7_C4K
    S12_C4
    S14_C4
    S17_C4, S17_C4K

If S12_C4K or S14_C4K are generated later, the script will include them
automatically if their folders follow the same result layout.
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

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "chapter6_parameter_comparison"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "chapter6_parameter_comparison"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

CYCLE_START = 34
CYCLE_END = 99
RETAINED_CYCLES = list(range(CYCLE_START, CYCLE_END + 1))

STATE_ORDER = ["S7", "S12", "S14", "S17"]
ORDER_GROUPS = ["C1", "C2", "C3", "C4/C4K"]

# Final 16-model thesis comparison set.
# Includes C4K for S7/S17, plain C4 for S12/S14.
FINAL_THESIS_MODELS = [
    "S7_C1", "S7_C2", "S7_C3", "S7_C4K",
    "S12_C1", "S12_C2", "S12_C3", "S12_C4",
    "S14_C1", "S14_C2", "S14_C3", "S14_C4",
    "S17_C1", "S17_C2", "S17_C3", "S17_C4K",
]

# Extra models useful for C4 vs C4K comparison.
EXTRA_COMPARISON_MODELS = [
    "S7_C4",
    "S17_C4",
    "S12_C4K",
    "S14_C4K",
]

ALL_DESIRED_MODELS = FINAL_THESIS_MODELS + EXTRA_COMPARISON_MODELS

# Core parameters expected in plain grid models.
GRID_CORE_PARAMS = [
    "alpha_n",
    "alpha_p",
    "K_e",
    "g_n",
    "g_p",
    "g_e",
    "theta_n0",
    "theta_p0",
]

# Core parameters expected in C4K models.
KPARAM_CORE_PARAMS = [
    "alpha_n",
    "alpha_p",
    "g_n",
    "g_p",
    "b_e,n",
    "b_e,p",
]

KPARAM_DIRECT_K_PARAMS = ["k1", "k2", "k3", "k4", "k5"]

# Shared physical/core parameters that can be compared across most models.
SHARED_CORE_PARAMS = ["alpha_n", "alpha_p", "g_n", "g_p"]

# Optional extra comparison for grid-only electrolyte parameters.
GRID_ELECTROLYTE_PARAMS = ["K_e", "g_e"]

# Plot styling
FIG_DPI = 300
TITLE_SIZE = 15
AXIS_SIZE = 12
TICK_SIZE = 9
LEGEND_SIZE = 8
MARKER_SIZE = 3.5
LINE_WIDTH = 1.5


# ============================================================
# Helpers
# ============================================================

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("[saved]", path)

    try:
        shutil.copy2(path, THESIS_FIG_DIR / path.name)
    except Exception as exc:
        warnings.warn(f"Could not copy to thesis dir: {exc}")

    try:
        FLOW_THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)
    except Exception as exc:
        warnings.warn(f"Could not copy to Flow thesis dir: {exc}")


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

    replacements = {
        "alpha_n_hat": "alpha_n",
        "alpha_p_hat": "alpha_p",
        "an": "alpha_n",
        "ap": "alpha_p",
        "ke": "K_e",
        "K_e_hat": "K_e",
        "gn": "g_n",
        "gp": "g_p",
        "ge": "g_e",
        "g_n_hat": "g_n",
        "g_p_hat": "g_p",
        "g_e_hat": "g_e",
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

    return replacements.get(s, s)


def parse_model_cycle_from_path(path: Path) -> dict | None:
    s = str(path)

    patterns = [
        # Grid anchor/full16 folders
        r"(anchor6|full16rem)_(S7|S12|S14|S17)_(C1|C2|C3|C4)_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
        # K-param anchor folders
        r"(anchor6|full16rem)_(S7|S12|S14|S17)_C4K_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if not m:
            continue

        groups = m.groups()

        if len(groups) == 7:
            run_prefix, state_id, candidate_id, nseeds, cycle, seed0, dt = groups
        elif len(groups) == 6:
            run_prefix, state_id, nseeds, cycle, seed0, dt = groups
            candidate_id = "C4K"
        else:
            continue

        return {
            "run_prefix": run_prefix,
            "state_id": state_id,
            "candidate_id": candidate_id,
            "model_id": f"{state_id}_{candidate_id}",
            "cycle_index": int(cycle),
            "retained_cycle_index": int(cycle) - CYCLE_START,
            "n_multistart": int(nseeds),
            "seed0": int(seed0),
            "id_downsample_dt": float(dt),
        }

    return None


def model_order_group(model_id: str) -> str:
    if model_id.endswith("_C1"):
        return "C1"
    if model_id.endswith("_C2"):
        return "C2"
    if model_id.endswith("_C3"):
        return "C3"
    if model_id.endswith("_C4") or model_id.endswith("_C4K"):
        return "C4/C4K"
    return "other"


def model_sort_key(model_id: str) -> tuple[int, int, str]:
    state = model_id.split("_")[0]
    cand = "_".join(model_id.split("_")[1:])

    state_idx = STATE_ORDER.index(state) if state in STATE_ORDER else 999

    cand_order = {
        "C1": 0,
        "C2": 1,
        "C3": 2,
        "C4": 3,
        "C4K": 4,
    }.get(cand, 999)

    return state_idx, cand_order, model_id


def pick_name_value_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    name_candidates = [
        "parameter",
        "param",
        "parameter_name",
        "name",
        "variable",
        "parameter_id",
    ]

    value_candidates = [
        "value",
        "estimate",
        "best_value",
        "parameter_value",
        "estimated_value",
        "param_value",
        "val",
    ]

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


def wide_row_to_parameter_rows(
    df: pd.DataFrame,
    meta: dict,
    source_file: Path,
    source_kind: str,
) -> list[dict]:
    """
    Fallback when parameter table is in wide format or metrics table contains
    parameter columns.
    """
    if df is None or len(df) == 0:
        return []

    row = df.iloc[0]

    possible_params = set(
        GRID_CORE_PARAMS
        + KPARAM_CORE_PARAMS
        + KPARAM_DIRECT_K_PARAMS
        + ["C", "D1"]
    )

    # Include polynomial beta names too.
    for prefix in ["ap", "an", "E"]:
        for deg in range(1, 6):
            possible_params.add(f"{prefix}{deg}")

    # Also common variants.
    variants = {
        "alpha_n_hat": "alpha_n",
        "alpha_p_hat": "alpha_p",
        "K_e_hat": "K_e",
        "g_n_hat": "g_n",
        "g_p_hat": "g_p",
        "g_e_hat": "g_e",
        "theta_n0_hat": "theta_n0",
        "theta_p0_hat": "theta_p0",
        "b_en": "b_e,n",
        "b_ep": "b_e,p",
        "b_e_n": "b_e,n",
        "b_e_p": "b_e,p",
    }

    out = []

    for col in df.columns:
        norm = normalize_param_name(col)

        if norm not in possible_params and col not in variants:
            continue

        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]

        if not np.isfinite(val):
            continue

        pnorm = variants.get(col, norm)

        item = dict(meta)
        item.update(
            {
                "parameter": pnorm,
                "value": float(val),
                "source_file": str(source_file),
                "source_kind": source_kind,
            }
        )
        out.append(item)

    return out


def long_table_to_parameter_rows(
    df: pd.DataFrame,
    meta: dict,
    source_file: Path,
    source_kind: str,
) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    name_col, value_col = pick_name_value_columns(df)

    if name_col is None or value_col is None:
        return wide_row_to_parameter_rows(df, meta, source_file, source_kind)

    out = []

    for _, row in df.iterrows():
        pname = normalize_param_name(row[name_col])
        val = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]

        if not np.isfinite(val):
            continue

        item = dict(meta)
        item.update(
            {
                "parameter": pname,
                "value": float(val),
                "source_file": str(source_file),
                "source_kind": source_kind,
            }
        )
        out.append(item)

    return out


# ============================================================
# Result collection
# ============================================================

def collect_grid_parameter_rows() -> list[dict]:
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

        if meta["model_id"] not in ALL_DESIRED_MODELS:
            continue

        if not (CYCLE_START <= meta["cycle_index"] <= CYCLE_END):
            continue

        # Parameter-long file from grid script.
        parameter_files = [
            folder / "real_cycle_parameter_long.csv",
            folder / "parameter_long.csv",
        ]

        found_any = False

        for p in parameter_files:
            df = safe_read_csv(p)
            if df is not None and len(df):
                rows.extend(long_table_to_parameter_rows(df, meta, p, "grid_parameter_long"))
                found_any = True
                break

        # Fallback: best-runs or summary may contain wide parameter columns.
        if not found_any:
            for p in [
                folder / "real_cycle_best_runs.csv",
                folder / "real_cycle_model_summary.csv",
                folder / "best_run.csv",
                folder / "summary.csv",
            ]:
                df = safe_read_csv(p)
                if df is not None and len(df):
                    new_rows = wide_row_to_parameter_rows(df, meta, p, "grid_wide_fallback")
                    if new_rows:
                        rows.extend(new_rows)
                        found_any = True
                        break

        # Optional beta coefficients.
        for p in [
            folder / "real_cycle_beta_coefficients.csv",
            folder / "beta_coefficients.csv",
        ]:
            df = safe_read_csv(p)
            if df is not None and len(df):
                beta_rows = long_table_to_parameter_rows(df, meta, p, "grid_beta")
                for r in beta_rows:
                    r["parameter"] = normalize_param_name(r["parameter"])
                rows.extend(beta_rows)
                break

    return rows


def collect_kparam_parameter_rows() -> list[dict]:
    rows = []

    if not KPARAM_ROOT.exists():
        return rows

    # Search any C4K folder. Currently expected S7_C4K and S17_C4K.
    folders = sorted(KPARAM_ROOT.glob("S*_C4K/anchor6_*")) + sorted(KPARAM_ROOT.glob("S*_C4K/full16rem_*"))

    for folder in folders:
        if not folder.is_dir():
            continue

        meta = parse_model_cycle_from_path(folder)

        if meta is None:
            continue

        if meta["model_id"] not in ALL_DESIRED_MODELS:
            continue

        if not (CYCLE_START <= meta["cycle_index"] <= CYCLE_END):
            continue

        found_any = False

        for p in [
            folder / "parameter_long.csv",
            folder / "real_cycle_parameter_long.csv",
        ]:
            df = safe_read_csv(p)
            if df is not None and len(df):
                rows.extend(long_table_to_parameter_rows(df, meta, p, "kparam_parameter_long"))
                found_any = True
                break

        if not found_any:
            for p in [
                folder / "best_run.csv",
                folder / "summary.csv",
                folder / "real_cycle_best_runs.csv",
                folder / "real_cycle_model_summary.csv",
            ]:
                df = safe_read_csv(p)
                if df is not None and len(df):
                    new_rows = wide_row_to_parameter_rows(df, meta, p, "kparam_wide_fallback")
                    if new_rows:
                        rows.extend(new_rows)
                        found_any = True
                        break

        # Optional beta coefficients.
        for p in [
            folder / "beta_coefficients.csv",
            folder / "real_cycle_beta_coefficients.csv",
        ]:
            df = safe_read_csv(p)
            if df is not None and len(df):
                beta_rows = long_table_to_parameter_rows(df, meta, p, "kparam_beta")
                rows.extend(beta_rows)
                break

    return rows


def collect_all_parameters() -> pd.DataFrame:
    rows = collect_grid_parameter_rows() + collect_kparam_parameter_rows()

    if not rows:
        raise RuntimeError("No parameter rows were found. Check result folder names and parameter_long CSVs.")

    df = pd.DataFrame(rows)

    df["parameter"] = df["parameter"].map(normalize_param_name)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cycle_index", "value"]).copy()
    df["cycle_index"] = df["cycle_index"].astype(int)
    df["retained_cycle_index"] = df["cycle_index"] - CYCLE_START

    df["order_group"] = df["model_id"].map(model_order_group)
    df["state_sort"] = df["state_id"].map({s: i for i, s in enumerate(STATE_ORDER)}).fillna(999).astype(int)

    # Drop exact duplicate rows, keeping first.
    df = df.sort_values(
        ["model_id", "cycle_index", "parameter", "source_kind", "source_file"]
    ).drop_duplicates(
        subset=["model_id", "cycle_index", "parameter", "source_kind"],
        keep="first",
    )

    return df.reset_index(drop=True)


# ============================================================
# Summaries
# ============================================================

def make_parameter_summary(df: pd.DataFrame) -> pd.DataFrame:
    def slope_per_cycle(g: pd.DataFrame) -> float:
        x = pd.to_numeric(g["retained_cycle_index"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["value"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return np.nan
        try:
            return float(np.polyfit(x[mask], y[mask], 1)[0])
        except Exception:
            return np.nan

    out = (
        df.groupby(["model_id", "state_id", "candidate_id", "order_group", "parameter"], as_index=False)
        .agg(
            n_cycles=("cycle_index", "nunique"),
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            std_value=("value", "std"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            q25_value=("value", lambda x: np.nanquantile(x, 0.25)),
            q75_value=("value", lambda x: np.nanquantile(x, 0.75)),
        )
    )

    slope_rows = []
    for keys, g in df.groupby(["model_id", "parameter"]):
        model_id, parameter = keys
        slope_rows.append(
            {
                "model_id": model_id,
                "parameter": parameter,
                "slope_per_retained_cycle": slope_per_cycle(g),
            }
        )

    slopes = pd.DataFrame(slope_rows)

    out = out.merge(slopes, on=["model_id", "parameter"], how="left")

    out["cv_abs"] = out["std_value"].abs() / out["median_value"].abs().replace(0, np.nan)

    out["model_sort"] = out["model_id"].map(lambda m: model_sort_key(m)[0] * 10 + model_sort_key(m)[1])
    out = out.sort_values(["model_sort", "parameter"]).drop(columns=["model_sort"])

    return out


def make_missing_model_parameter_table(df: pd.DataFrame) -> pd.DataFrame:
    expected_rows = []

    for model_id in ALL_DESIRED_MODELS:
        state = model_id.split("_")[0]
        cand = "_".join(model_id.split("_")[1:])

        if cand == "C4K":
            expected_params = KPARAM_CORE_PARAMS
            if state == "S7":
                expected_params += ["k1", "k2"]
            elif state in ["S12", "S14"]:
                expected_params += ["k1", "k2", "k3"]
            elif state == "S17":
                expected_params += ["k1", "k2", "k3", "k4", "k5"]
        else:
            expected_params = GRID_CORE_PARAMS

        for p in expected_params:
            expected_rows.append({"model_id": model_id, "parameter": p})

    expected = pd.DataFrame(expected_rows)

    have = (
        df.groupby(["model_id", "parameter"], as_index=False)
        .agg(n_cycles=("cycle_index", "nunique"))
    )

    out = expected.merge(have, on=["model_id", "parameter"], how="left")
    out["n_cycles"] = out["n_cycles"].fillna(0).astype(int)
    out["missing"] = out["n_cycles"] == 0

    out = out.sort_values(["model_id", "parameter"]).reset_index(drop=True)

    return out


# ============================================================
# Plotting
# ============================================================

def clean_axis(ax):
    ax.grid(True, alpha=0.30)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def plot_order_group_core_parameters(df: pd.DataFrame, order_group: str) -> None:
    params = SHARED_CORE_PARAMS

    d = df[
        (df["order_group"] == order_group)
        & (df["parameter"].isin(params))
        & (~df["source_kind"].str.contains("beta", na=False))
    ].copy()

    if len(d) == 0:
        print(f"[skip] no data for {order_group}")
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    nrows = len(params)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12.5, 2.8 * nrows), sharex=True)

    if nrows == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
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

    axes[0].set_title(f"Core Parameter Comparison — {order_group}", fontsize=TITLE_SIZE, pad=10)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fname = f"fig_ch6_parameter_core_compare_{order_group.replace('/', '_').replace(' ', '_')}.png"
    savefig(OUT_FIG_DIR / fname)


def plot_c4_c4k_core_comparison(df: pd.DataFrame) -> None:
    params = SHARED_CORE_PARAMS

    d = df[
        (df["model_id"].str.endswith("_C4") | df["model_id"].str.endswith("_C4K"))
        & (df["parameter"].isin(params))
        & (~df["source_kind"].str.contains("beta", na=False))
    ].copy()

    if len(d) == 0:
        print("[skip] no C4/C4K core data")
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    nrows = len(params)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(13.5, 2.9 * nrows), sharex=True)

    if nrows == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
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

    axes[0].set_title("Core Parameter Comparison — C4 and C4K Models", fontsize=TITLE_SIZE, pad=10)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    savefig(OUT_FIG_DIR / "fig_ch6_parameter_core_compare_C4_C4K_all.png")


def plot_grid_electrolyte_comparison(df: pd.DataFrame) -> None:
    params = GRID_ELECTROLYTE_PARAMS

    d = df[
        (df["parameter"].isin(params))
        & (~df["model_id"].str.endswith("_C4K"))
        & (~df["source_kind"].str.contains("beta", na=False))
    ].copy()

    if len(d) == 0:
        print("[skip] no grid electrolyte data")
        return

    for order_group in ORDER_GROUPS:
        if order_group == "C4/C4K":
            dd = d[d["model_id"].str.endswith("_C4")].copy()
            title = "Grid Electrolyte Parameters — C4 Models"
            fname = "fig_ch6_parameter_grid_electrolyte_C4.png"
        else:
            dd = d[d["order_group"] == order_group].copy()
            title = f"Grid Electrolyte Parameters — {order_group}"
            fname = f"fig_ch6_parameter_grid_electrolyte_{order_group}.png"

        if len(dd) == 0:
            continue

        models = sorted(dd["model_id"].unique(), key=model_sort_key)

        fig, axes = plt.subplots(nrows=len(params), ncols=1, figsize=(12.5, 3.0 * len(params)), sharex=True)

        if len(params) == 1:
            axes = [axes]

        for ax, param in zip(axes, params):
            dp = dd[dd["parameter"] == param].copy()

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

        axes[0].set_title(title, fontsize=TITLE_SIZE, pad=10)
        axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=LEGEND_SIZE)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        savefig(OUT_FIG_DIR / fname)


def plot_c4k_direct_parameters(df: pd.DataFrame) -> None:
    params = ["b_e,n", "b_e,p", "k1", "k2", "k3", "k4", "k5"]

    d = df[
        (df["model_id"].str.endswith("_C4K"))
        & (df["parameter"].isin(params))
        & (~df["source_kind"].str.contains("beta", na=False))
    ].copy()

    if len(d) == 0:
        print("[skip] no C4K direct-k data")
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    nrows = len(params)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12.5, 2.45 * nrows), sharex=True)

    if nrows == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
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

    axes[0].set_title("Direct Electrolyte Parameters — C4K Models", fontsize=TITLE_SIZE, pad=10)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(OUT_FIG_DIR / "fig_ch6_parameter_c4k_direct_electrolyte.png")


def plot_parameter_summary_heatmap(summary: pd.DataFrame, metric: str, params: list[str], filename: str, title: str) -> None:
    d = summary[summary["parameter"].isin(params)].copy()

    if len(d) == 0:
        print(f"[skip] no data for heatmap {filename}")
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)
    params_present = [p for p in params if p in set(d["parameter"])]

    mat = np.full((len(params_present), len(models)), np.nan)

    for i, p in enumerate(params_present):
        for j, m in enumerate(models):
            row = d[(d["parameter"] == p) & (d["model_id"] == m)]
            if len(row):
                mat[i, j] = pd.to_numeric(row.iloc[0][metric], errors="coerce")

    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(models)), max(4.5, 0.55 * len(params_present))))

    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=TICK_SIZE)

    ax.set_yticks(np.arange(len(params_present)))
    ax.set_yticklabels(params_present, fontsize=TICK_SIZE)

    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, fontsize=AXIS_SIZE)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / filename)


def plot_beta_by_order(df: pd.DataFrame) -> None:
    d = df[df["source_kind"].str.contains("beta", na=False)].copy()

    if len(d) == 0:
        print("[skip] no beta coefficient rows found")
        return

    # Keep a reasonable set of beta-like coefficients.
    beta_params = sorted(d["parameter"].unique())

    for order_group in ORDER_GROUPS:
        dd = d[d["order_group"] == order_group].copy()

        if len(dd) == 0:
            continue

        models = sorted(dd["model_id"].unique(), key=model_sort_key)

        # Plot medians instead of all cycles to avoid extreme clutter.
        summary = (
            dd.groupby(["model_id", "parameter"], as_index=False)
            .agg(median_value=("value", "median"))
        )

        pivot = summary.pivot(index="parameter", columns="model_id", values="median_value")
        pivot = pivot.reindex(index=beta_params, columns=models)

        pivot = pivot.dropna(axis=0, how="all")

        if len(pivot) == 0:
            continue

        fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(models)), max(4, 0.35 * len(pivot))))

        mat = pivot.to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=TICK_SIZE)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=TICK_SIZE)

        ax.set_title(f"Median Output Coefficients — {order_group}", fontsize=TITLE_SIZE, pad=10)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("median coefficient", fontsize=AXIS_SIZE)

        fig.tight_layout()

        fname = f"fig_ch6_beta_median_heatmap_{order_group.replace('/', '_')}.png"
        savefig(OUT_FIG_DIR / fname)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("CHAPTER 6 PARAMETER COMPARISON ACROSS ALL FITTED MODELS")
    print("=" * 100)
    print("PROJECT:", PROJECT)
    print("GRID_ROOT:", GRID_ROOT)
    print("KPARAM_ROOT:", KPARAM_ROOT)
    print("OUT_TABLE_DIR:", OUT_TABLE_DIR)
    print("OUT_FIG_DIR:", OUT_FIG_DIR)
    print("Cycles:", CYCLE_START, "to", CYCLE_END)
    print("=" * 100)

    df = collect_all_parameters()

    all_path = OUT_TABLE_DIR / "all_model_parameters_long.csv"
    df.to_csv(all_path, index=False)
    print("[saved]", all_path)

    summary = make_parameter_summary(df)

    summary_path = OUT_TABLE_DIR / "model_parameter_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[saved]", summary_path)

    missing = make_missing_model_parameter_table(df)

    missing_path = OUT_TABLE_DIR / "missing_model_parameters.csv"
    missing.to_csv(missing_path, index=False)
    print("[saved]", missing_path)

    print()
    print("Models found:")
    for m in sorted(df["model_id"].unique(), key=model_sort_key):
        ncyc = df[df["model_id"] == m]["cycle_index"].nunique()
        npar = df[df["model_id"] == m]["parameter"].nunique()
        print(f"  {m:10s} cycles={ncyc:3d} parameters={npar:3d}")

    print()
    print("Missing model/parameter combinations with zero cycles:")
    miss0 = missing[missing["missing"]].copy()
    if len(miss0):
        print(miss0.to_string(index=False))
    else:
        print("  None")

    # Order-wise core parameter comparison.
    for order_group in ORDER_GROUPS:
        plot_order_group_core_parameters(df, order_group)

    # Main busy-but-contained C4/C4K comparison.
    plot_c4_c4k_core_comparison(df)

    # Electrolyte comparisons.
    plot_grid_electrolyte_comparison(df)
    plot_c4k_direct_parameters(df)

    # Summary heatmaps.
    plot_parameter_summary_heatmap(
        summary,
        metric="median_value",
        params=SHARED_CORE_PARAMS,
        filename="fig_ch6_parameter_median_heatmap_shared_core.png",
        title="Median Estimated Shared Core Parameters",
    )

    plot_parameter_summary_heatmap(
        summary,
        metric="cv_abs",
        params=SHARED_CORE_PARAMS,
        filename="fig_ch6_parameter_cv_heatmap_shared_core.png",
        title="Cycle-to-Cycle Relative Variability of Shared Core Parameters",
    )

    plot_parameter_summary_heatmap(
        summary,
        metric="slope_per_retained_cycle",
        params=SHARED_CORE_PARAMS,
        filename="fig_ch6_parameter_slope_heatmap_shared_core.png",
        title="Linear Trend of Shared Core Parameters Across Retained Cycles",
    )

    plot_parameter_summary_heatmap(
        summary,
        metric="median_value",
        params=GRID_ELECTROLYTE_PARAMS,
        filename="fig_ch6_parameter_median_heatmap_grid_electrolyte.png",
        title="Median Grid Electrolyte Parameters",
    )

    plot_parameter_summary_heatmap(
        summary,
        metric="median_value",
        params=["b_e,n", "b_e,p", "k1", "k2", "k3", "k4", "k5"],
        filename="fig_ch6_parameter_median_heatmap_c4k_direct_electrolyte.png",
        title="Median Direct Electrolyte Parameters in C4K Models",
    )

    # Optional beta coefficient heatmaps.
    plot_beta_by_order(df)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Main tables:")
    print(" ", all_path)
    print(" ", summary_path)
    print(" ", missing_path)
    print()
    print("Main figures copied to:")
    print(" ", OUT_FIG_DIR)
    print(" ", THESIS_FIG_DIR)
    print(" ", FLOW_THESIS_FIG_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
