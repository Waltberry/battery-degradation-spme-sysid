#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_core_parameter_comparison_only.py

Purpose
-------
Compare only the shared core fitted parameters across retained-cycle model fits.

Core parameters compared:
    alpha_n
    alpha_p
    g_n
    g_p

Excluded deliberately:
    K_e, g_e
    k1--k5
    b_e,n, b_e,p
    theta_n0, theta_p0
    beta/output coefficients

Reason:
-------
The electrolyte parameters are not comparable between the original grid models
and the C4K direct-k formulation. The core solid/gain parameters above are the
safest shared comparison across model structures.

Inputs:
-------
results/real_cycle_ctid_state_order_grid/anchor6_*/
results/real_cycle_ctid_state_order_grid/full16rem_*/
results/real_warm_continuation_ctid/S*_C4K/anchor6_*/
results/real_warm_continuation_ctid/S*_C4K/full16rem_*/

Outputs:
--------
results/tables/chapter6_core_parameter_comparison/
results/figures/chapter6_core_parameter_comparison/
figures/chapter6/
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

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "chapter6_core_parameter_comparison"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "chapter6_core_parameter_comparison"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

CYCLE_START = 34
CYCLE_END = 99

CORE_PARAMS = ["alpha_n", "alpha_p", "g_n", "g_p"]

STATE_ORDER = ["S7", "S12", "S14", "S17"]
ORDER_GROUPS = ["C1", "C2", "C3", "C4/C4K"]

# Thesis final model set.
# Uses C4K for S7/S17 and C4 for S12/S14.
FINAL_THESIS_MODELS = [
    "S7_C1", "S7_C2", "S7_C3", "S7_C4K",
    "S12_C1", "S12_C2", "S12_C3", "S12_C4",
    "S14_C1", "S14_C2", "S14_C3", "S14_C4",
    "S17_C1", "S17_C2", "S17_C3", "S17_C4K",
]

# Extra quartic models for comparing plain C4 with C4K where available.
EXTRA_MODELS = [
    "S7_C4",
    "S17_C4",
    "S12_C4K",
    "S14_C4K",
]

ALL_MODELS = FINAL_THESIS_MODELS + EXTRA_MODELS

FIG_DPI = 300
TITLE_SIZE = 15
AXIS_SIZE = 12
TICK_SIZE = 9
LEGEND_SIZE = 8
MARKER_SIZE = 3.4
LINE_WIDTH = 1.55


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
        warnings.warn(f"Could not copy to thesis figure dir: {exc}")

    try:
        FLOW_THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)
    except Exception as exc:
        warnings.warn(f"Could not copy to Flow thesis figure dir: {exc}")


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
        "g_n_hat": "g_n",
        "g_p_hat": "g_p",
        "gn": "g_n",
        "gp": "g_p",
    }

    return replacements.get(s, s)


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

    cand_idx = {
        "C1": 0,
        "C2": 1,
        "C3": 2,
        "C4": 3,
        "C4K": 4,
    }.get(cand, 999)

    return state_idx, cand_idx, model_id


def parse_model_cycle_from_path(path: Path) -> dict | None:
    s = str(path)

    patterns = [
        r"(anchor6|full16rem)_(S7|S12|S14|S17)_(C1|C2|C3|C4)_(\d+)seeds_cycle_(\d+)_seed_(\d+)_dt_([0-9.]+)",
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

        cycle = int(cycle)

        return {
            "run_prefix": run_prefix,
            "state_id": state_id,
            "candidate_id": candidate_id,
            "model_id": f"{state_id}_{candidate_id}",
            "cycle_index": cycle,
            "retained_cycle_index": cycle - CYCLE_START,
            "n_multistart": int(nseeds),
            "seed0": int(seed0),
            "id_downsample_dt": float(dt),
        }

    return None


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


def long_table_to_rows(df: pd.DataFrame, meta: dict, source_file: Path, source_kind: str) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    name_col, value_col = pick_name_value_columns(df)

    if name_col is None or value_col is None:
        return wide_table_to_rows(df, meta, source_file, source_kind)

    rows = []

    for _, r in df.iterrows():
        pname = normalize_param_name(r[name_col])

        if pname not in CORE_PARAMS:
            continue

        val = pd.to_numeric(pd.Series([r[value_col]]), errors="coerce").iloc[0]

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
        rows.append(item)

    return rows


def wide_table_to_rows(df: pd.DataFrame, meta: dict, source_file: Path, source_kind: str) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    row = df.iloc[0]
    rows = []

    aliases = {
        "alpha_n": "alpha_n",
        "alpha_n_hat": "alpha_n",
        "alpha_p": "alpha_p",
        "alpha_p_hat": "alpha_p",
        "g_n": "g_n",
        "g_n_hat": "g_n",
        "g_p": "g_p",
        "g_p_hat": "g_p",
        "gn": "g_n",
        "gp": "g_p",
    }

    for col in df.columns:
        pname = normalize_param_name(col)

        if col in aliases:
            pname = aliases[col]

        if pname not in CORE_PARAMS:
            continue

        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]

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
        rows.append(item)

    return rows


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

        if meta["model_id"] not in ALL_MODELS:
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

        if meta["model_id"] not in ALL_MODELS:
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


def collect_core_parameters() -> pd.DataFrame:
    rows = collect_grid_rows() + collect_kparam_rows()

    if not rows:
        raise RuntimeError("No core parameter rows found. Check result folder paths and CSV layouts.")

    df = pd.DataFrame(rows)

    df["parameter"] = df["parameter"].map(normalize_param_name)
    df = df[df["parameter"].isin(CORE_PARAMS)].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cycle_index", "value"]).copy()
    df["cycle_index"] = df["cycle_index"].astype(int)
    df["retained_cycle_index"] = df["cycle_index"] - CYCLE_START
    df["order_group"] = df["model_id"].map(model_order_group)

    df = df.sort_values(
        ["model_id", "cycle_index", "parameter", "source_kind", "source_file"]
    ).drop_duplicates(
        subset=["model_id", "cycle_index", "parameter"],
        keep="first",
    )

    return df.reset_index(drop=True)


# ============================================================
# Summary tables
# ============================================================

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


def make_core_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
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

    for (model_id, parameter), g in df.groupby(["model_id", "parameter"]):
        slope_rows.append(
            {
                "model_id": model_id,
                "parameter": parameter,
                "slope_per_retained_cycle": slope_per_cycle(g),
            }
        )

    slopes = pd.DataFrame(slope_rows)

    summary = summary.merge(slopes, on=["model_id", "parameter"], how="left")
    summary["cv_abs"] = summary["std_value"].abs() / summary["median_value"].abs().replace(0, np.nan)

    summary["model_sort"] = summary["model_id"].map(lambda m: model_sort_key(m)[0] * 10 + model_sort_key(m)[1])
    summary = summary.sort_values(["model_sort", "parameter"]).drop(columns=["model_sort"])

    return summary


def make_pairwise_trend_similarity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for order_group in ORDER_GROUPS:
        d_order = df[df["order_group"] == order_group].copy()

        for param in CORE_PARAMS:
            d = d_order[d_order["parameter"] == param].copy()

            models = sorted(d["model_id"].unique(), key=model_sort_key)

            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    m1, m2 = models[i], models[j]

                    a = d[d["model_id"] == m1][["retained_cycle_index", "value"]].rename(columns={"value": "v1"})
                    b = d[d["model_id"] == m2][["retained_cycle_index", "value"]].rename(columns={"value": "v2"})

                    merged = a.merge(b, on="retained_cycle_index", how="inner")

                    if len(merged) < 5:
                        corr_raw = np.nan
                        corr_norm = np.nan
                        n_common = len(merged)
                    else:
                        v1 = pd.to_numeric(merged["v1"], errors="coerce").to_numpy(dtype=float)
                        v2 = pd.to_numeric(merged["v2"], errors="coerce").to_numpy(dtype=float)

                        mask = np.isfinite(v1) & np.isfinite(v2)

                        if mask.sum() < 5:
                            corr_raw = np.nan
                            corr_norm = np.nan
                            n_common = int(mask.sum())
                        else:
                            v1 = v1[mask]
                            v2 = v2[mask]
                            n_common = len(v1)

                            corr_raw = float(np.corrcoef(v1, v2)[0, 1]) if np.std(v1) > 0 and np.std(v2) > 0 else np.nan

                            z1 = (v1 - np.mean(v1)) / np.std(v1) if np.std(v1) > 0 else v1 * np.nan
                            z2 = (v2 - np.mean(v2)) / np.std(v2) if np.std(v2) > 0 else v2 * np.nan

                            corr_norm = float(np.corrcoef(z1, z2)[0, 1]) if np.all(np.isfinite(z1)) and np.all(np.isfinite(z2)) else np.nan

                    rows.append(
                        {
                            "order_group": order_group,
                            "parameter": param,
                            "model_1": m1,
                            "model_2": m2,
                            "n_common_cycles": n_common,
                            "raw_correlation": corr_raw,
                            "normalized_trend_correlation": corr_norm,
                        }
                    )

    out = pd.DataFrame(rows)

    return out


# ============================================================
# Plotting
# ============================================================

def clean_axis(ax) -> None:
    ax.grid(True, alpha=0.30)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def normalize_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y)
    scale = np.nanpercentile(y, 75) - np.nanpercentile(y, 25)

    if not np.isfinite(scale) or abs(scale) < 1e-15:
        scale = np.nanstd(y)

    if not np.isfinite(scale) or abs(scale) < 1e-15:
        return y * np.nan

    return (y - med) / scale


def plot_core_by_order(df: pd.DataFrame, order_group: str, normalized: bool = False) -> None:
    d = df[df["order_group"] == order_group].copy()

    if len(d) == 0:
        print(f"[skip] no data for {order_group}")
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    fig, axes = plt.subplots(
        nrows=len(CORE_PARAMS),
        ncols=1,
        figsize=(12.8, 2.75 * len(CORE_PARAMS)),
        sharex=True,
    )

    if len(CORE_PARAMS) == 1:
        axes = [axes]

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

    title_suffix = "Normalized Trends" if normalized else "Raw Values"
    axes[0].set_title(f"Core Parameter Comparison — {order_group} ({title_suffix})", fontsize=TITLE_SIZE, pad=10)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    safe_group = order_group.replace("/", "_").replace(" ", "_")
    suffix = "normalized" if normalized else "raw"

    fname = f"fig_ch6_core_parameter_compare_{safe_group}_{suffix}.png"
    savefig(OUT_FIG_DIR / fname)


def plot_s14_s17_focus(df: pd.DataFrame, normalized: bool = False) -> None:
    d = df[
        df["model_id"].isin(["S14_C1", "S14_C2", "S14_C3", "S14_C4", "S17_C1", "S17_C2", "S17_C3", "S17_C4", "S17_C4K"])
    ].copy()

    if len(d) == 0:
        print("[skip] no S14/S17 data")
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    fig, axes = plt.subplots(
        nrows=len(CORE_PARAMS),
        ncols=1,
        figsize=(13.0, 2.85 * len(CORE_PARAMS)),
        sharex=True,
    )

    if len(CORE_PARAMS) == 1:
        axes = [axes]

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

    title_suffix = "Normalized Trends" if normalized else "Raw Values"
    axes[0].set_title(f"S14/S17 Core Parameter Comparison ({title_suffix})", fontsize=TITLE_SIZE, pad=10)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_parameter_compare_S14_S17_focus_{suffix}.png")


def plot_c4_c4k_focus(df: pd.DataFrame, normalized: bool = False) -> None:
    d = df[
        (df["model_id"].str.endswith("_C4")) | (df["model_id"].str.endswith("_C4K"))
    ].copy()

    if len(d) == 0:
        print("[skip] no C4/C4K data")
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    fig, axes = plt.subplots(
        nrows=len(CORE_PARAMS),
        ncols=1,
        figsize=(13.2, 2.85 * len(CORE_PARAMS)),
        sharex=True,
    )

    if len(CORE_PARAMS) == 1:
        axes = [axes]

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

    title_suffix = "Normalized Trends" if normalized else "Raw Values"
    axes[0].set_title(f"C4/C4K Core Parameter Comparison ({title_suffix})", fontsize=TITLE_SIZE, pad=10)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), fontsize=LEGEND_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_parameter_compare_C4_C4K_focus_{suffix}.png")


def plot_summary_heatmap(summary: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    d = summary.copy()

    models = sorted(d["model_id"].unique(), key=model_sort_key)
    params = CORE_PARAMS

    mat = np.full((len(params), len(models)), np.nan)

    for i, p in enumerate(params):
        for j, m in enumerate(models):
            row = d[(d["parameter"] == p) & (d["model_id"] == m)]

            if len(row):
                mat[i, j] = pd.to_numeric(row.iloc[0][metric], errors="coerce")

    fig, ax = plt.subplots(figsize=(max(9.5, 0.72 * len(models)), 4.8))

    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=TICK_SIZE)

    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels(params, fontsize=TICK_SIZE)

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


def plot_best_similarity_heatmap(similarity: pd.DataFrame) -> None:
    d = similarity.copy()

    if len(d) == 0:
        return

    # Focus on S14/S17 pairs because that is the key question.
    d = d[
        (
            d["model_1"].str.startswith("S14")
            & d["model_2"].str.startswith("S17")
        )
        |
        (
            d["model_1"].str.startswith("S17")
            & d["model_2"].str.startswith("S14")
        )
    ].copy()

    if len(d) == 0:
        print("[skip] no S14/S17 similarity pairs")
        return

    d["pair"] = d["model_1"] + " vs " + d["model_2"]

    pairs = sorted(d["pair"].unique())
    params = CORE_PARAMS

    mat = np.full((len(params), len(pairs)), np.nan)

    for i, p in enumerate(params):
        for j, pair in enumerate(pairs):
            row = d[(d["parameter"] == p) & (d["pair"] == pair)]

            if len(row):
                mat[i, j] = pd.to_numeric(row.iloc[0]["normalized_trend_correlation"], errors="coerce")

    fig, ax = plt.subplots(figsize=(max(10.0, 0.75 * len(pairs)), 4.8))

    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)

    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels(params, fontsize=TICK_SIZE)

    ax.set_title("S14/S17 Core-Parameter Trend Similarity", fontsize=TITLE_SIZE, pad=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("normalized trend correlation", fontsize=AXIS_SIZE)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    savefig(OUT_FIG_DIR / "fig_ch6_core_parameter_s14_s17_trend_similarity.png")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("CHAPTER 6 CORE PARAMETER COMPARISON ONLY")
    print("=" * 100)
    print("PROJECT:", PROJECT)
    print("GRID_ROOT:", GRID_ROOT)
    print("KPARAM_ROOT:", KPARAM_ROOT)
    print("OUT_TABLE_DIR:", OUT_TABLE_DIR)
    print("OUT_FIG_DIR:", OUT_FIG_DIR)
    print("Cycles:", CYCLE_START, "to", CYCLE_END)
    print("Core parameters:", ", ".join(CORE_PARAMS))
    print("=" * 100)

    df = collect_core_parameters()

    long_path = OUT_TABLE_DIR / "core_parameters_long.csv"
    df.to_csv(long_path, index=False)
    print("[saved]", long_path)

    summary = make_core_summary(df)

    summary_path = OUT_TABLE_DIR / "core_parameter_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[saved]", summary_path)

    similarity = make_pairwise_trend_similarity(df)

    similarity_path = OUT_TABLE_DIR / "core_parameter_pairwise_trend_similarity.csv"
    similarity.to_csv(similarity_path, index=False)
    print("[saved]", similarity_path)

    print()
    print("Models found:")
    for m in sorted(df["model_id"].unique(), key=model_sort_key):
        ncyc = df[df["model_id"] == m]["cycle_index"].nunique()
        npar = df[df["model_id"] == m]["parameter"].nunique()
        print(f"  {m:10s} cycles={ncyc:3d} core_parameters={npar:2d}")

    print()
    print("Parameter availability:")
    avail = (
        df.groupby(["model_id", "parameter"], as_index=False)
        .agg(n_cycles=("cycle_index", "nunique"))
        .pivot(index="model_id", columns="parameter", values="n_cycles")
        .fillna(0)
        .astype(int)
    )
    avail = avail.reindex(index=sorted(avail.index, key=model_sort_key))
    print(avail.to_string())

    # Raw and normalized plots by order group.
    for order_group in ORDER_GROUPS:
        plot_core_by_order(df, order_group, normalized=False)
        plot_core_by_order(df, order_group, normalized=True)

    # Focused plots for the key thesis question.
    plot_s14_s17_focus(df, normalized=False)
    plot_s14_s17_focus(df, normalized=True)

    # Focused C4/C4K plots.
    plot_c4_c4k_focus(df, normalized=False)
    plot_c4_c4k_focus(df, normalized=True)

    # Summary heatmaps.
    plot_summary_heatmap(
        summary,
        metric="median_value",
        filename="fig_ch6_core_parameter_median_heatmap.png",
        title="Median Estimated Core Parameters Across Retained Cycles",
    )

    plot_summary_heatmap(
        summary,
        metric="cv_abs",
        filename="fig_ch6_core_parameter_cv_heatmap.png",
        title="Cycle-to-Cycle Relative Variability of Core Parameters",
    )

    plot_summary_heatmap(
        summary,
        metric="slope_per_retained_cycle",
        filename="fig_ch6_core_parameter_slope_heatmap.png",
        title="Linear Trend of Core Parameters Across Retained Cycles",
    )

    plot_best_similarity_heatmap(similarity)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Tables:")
    print(" ", long_path)
    print(" ", summary_path)
    print(" ", similarity_path)
    print()
    print("Figures:")
    print(" ", OUT_FIG_DIR)
    print(" ", THESIS_FIG_DIR)
    print(" ", FLOW_THESIS_FIG_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
