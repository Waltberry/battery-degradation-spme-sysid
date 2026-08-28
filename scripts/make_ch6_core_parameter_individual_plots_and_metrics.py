#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_core_parameter_individual_plots_and_metrics.py

Purpose
-------
Create readable individual plots and quantitative similarity metrics for the
shared core parameters:

    alpha_n
    alpha_p
    g_n
    g_p

This script reads:
    results/tables/chapter6_core_parameter_comparison/core_parameters_long.csv

It does NOT rerun CT-ID.

Outputs:
--------
results/tables/chapter6_core_parameter_comparison/
    core_parameter_pairwise_similarity_detailed.csv
    core_parameter_s14_s17_similarity.csv
    core_parameter_best_matching_pairs.csv

results/figures/chapter6_core_parameter_individual/
figures/chapter6/
"""

from __future__ import annotations

from pathlib import Path
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

IN_TABLE = PROJECT / "results" / "tables" / "chapter6_core_parameter_comparison" / "core_parameters_long.csv"

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "chapter6_core_parameter_comparison"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "chapter6_core_parameter_individual"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

CORE_PARAMS = ["alpha_n", "alpha_p", "g_n", "g_p"]
STATE_ORDER = ["S7", "S12", "S14", "S17"]
ORDER_GROUPS = ["C1", "C2", "C3", "C4/C4K"]

# Keep thesis-final 16 set in the main figures.
FINAL_THESIS_MODELS = [
    "S7_C1", "S7_C2", "S7_C3", "S7_C4K",
    "S12_C1", "S12_C2", "S12_C3", "S12_C4",
    "S14_C1", "S14_C2", "S14_C3", "S14_C4",
    "S17_C1", "S17_C2", "S17_C3", "S17_C4K",
]

# Extra C4 models may exist from the anchor screen.
EXTRA_MODELS = ["S7_C4", "S17_C4"]

FIG_DPI = 300
TITLE_SIZE = 15
AXIS_SIZE = 12
TICK_SIZE = 9
LEGEND_SIZE = 8
LINE_WIDTH = 1.8
MARKER_SIZE = 4.2


# ============================================================
# Helpers
# ============================================================

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


def clean_axis(ax) -> None:
    ax.grid(True, alpha=0.30)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def normalize_series(y: np.ndarray) -> np.ndarray:
    """
    Robust normalization: median-center and divide by IQR.
    This compares trend shape rather than absolute magnitude.
    """
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def load_core_data() -> pd.DataFrame:
    if not IN_TABLE.exists():
        raise FileNotFoundError(
            f"Missing {IN_TABLE}. Run scripts/make_ch6_core_parameter_comparison_only.py first."
        )

    df = pd.read_csv(IN_TABLE)

    required = {"model_id", "parameter", "value", "retained_cycle_index", "order_group"}

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"{IN_TABLE} is missing required columns: {missing}")

    df = df[df["parameter"].isin(CORE_PARAMS)].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["retained_cycle_index"] = pd.to_numeric(df["retained_cycle_index"], errors="coerce")
    df = df.dropna(subset=["value", "retained_cycle_index"]).copy()

    return df


# ============================================================
# Individual plots
# ============================================================

def plot_parameter_all_models(df: pd.DataFrame, parameter: str, normalized: bool = False) -> None:
    d = df[df["parameter"] == parameter].copy()

    d = d[d["model_id"].isin(FINAL_THESIS_MODELS)].copy()

    if len(d) == 0:
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    fig, ax = plt.subplots(figsize=(13.5, 6.4))

    for model_id in models:
        g = d[d["model_id"] == model_id].sort_values("retained_cycle_index")

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

    suffix_title = "normalized trend" if normalized else "estimated value"

    ax.set_title(f"{parameter}: all final models ({suffix_title})", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)
    ax.set_ylabel(f"{parameter} {'[normalized]' if normalized else ''}", fontsize=AXIS_SIZE)
    clean_axis(ax)
    ax.legend(loc="best", fontsize=LEGEND_SIZE, ncol=4)

    fig.tight_layout()

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_{parameter}_all_models_{suffix}.png")


def plot_parameter_by_order(df: pd.DataFrame, parameter: str, order_group: str, normalized: bool = False) -> None:
    d = df[
        (df["parameter"] == parameter)
        & (df["order_group"] == order_group)
    ].copy()

    d = d[d["model_id"].isin(FINAL_THESIS_MODELS + EXTRA_MODELS)].copy()

    if len(d) == 0:
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    fig, ax = plt.subplots(figsize=(11.8, 5.6))

    for model_id in models:
        g = d[d["model_id"] == model_id].sort_values("retained_cycle_index")

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

    suffix_title = "normalized trend" if normalized else "estimated value"

    ax.set_title(f"{parameter}: {order_group} models ({suffix_title})", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)
    ax.set_ylabel(f"{parameter} {'[normalized]' if normalized else ''}", fontsize=AXIS_SIZE)
    clean_axis(ax)
    ax.legend(loc="best", fontsize=LEGEND_SIZE, ncol=2)

    fig.tight_layout()

    safe_group = order_group.replace("/", "_").replace(" ", "_")
    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_{parameter}_{safe_group}_{suffix}.png")


def plot_parameter_s14_s17_focus(df: pd.DataFrame, parameter: str, normalized: bool = False) -> None:
    focus_models = [
        "S14_C1", "S14_C2", "S14_C3", "S14_C4",
        "S17_C1", "S17_C2", "S17_C3", "S17_C4", "S17_C4K",
    ]

    d = df[
        (df["parameter"] == parameter)
        & (df["model_id"].isin(focus_models))
    ].copy()

    if len(d) == 0:
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    for model_id in models:
        g = d[d["model_id"] == model_id].sort_values("retained_cycle_index")

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

    suffix_title = "normalized trend" if normalized else "estimated value"

    ax.set_title(f"{parameter}: S14/S17 focus ({suffix_title})", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)
    ax.set_ylabel(f"{parameter} {'[normalized]' if normalized else ''}", fontsize=AXIS_SIZE)
    clean_axis(ax)
    ax.legend(loc="best", fontsize=LEGEND_SIZE, ncol=3)

    fig.tight_layout()

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_{parameter}_S14_S17_focus_{suffix}.png")


def plot_parameter_c4_c4k_focus(df: pd.DataFrame, parameter: str, normalized: bool = False) -> None:
    d = df[
        (df["parameter"] == parameter)
        & (
            df["model_id"].str.endswith("_C4")
            | df["model_id"].str.endswith("_C4K")
        )
    ].copy()

    if len(d) == 0:
        return

    models = sorted(d["model_id"].unique(), key=model_sort_key)

    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    for model_id in models:
        g = d[d["model_id"] == model_id].sort_values("retained_cycle_index")

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

    suffix_title = "normalized trend" if normalized else "estimated value"

    ax.set_title(f"{parameter}: C4/C4K focus ({suffix_title})", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)
    ax.set_ylabel(f"{parameter} {'[normalized]' if normalized else ''}", fontsize=AXIS_SIZE)
    clean_axis(ax)
    ax.legend(loc="best", fontsize=LEGEND_SIZE, ncol=3)

    fig.tight_layout()

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_{parameter}_C4_C4K_focus_{suffix}.png")


# ============================================================
# Metrics
# ============================================================

def make_pairwise_similarity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for parameter in CORE_PARAMS:
        d_param = df[df["parameter"] == parameter].copy()

        for order_group in ORDER_GROUPS:
            d = d_param[d_param["order_group"] == order_group].copy()

            models = sorted(d["model_id"].unique(), key=model_sort_key)

            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    m1 = models[i]
                    m2 = models[j]

                    a = (
                        d[d["model_id"] == m1][["retained_cycle_index", "value"]]
                        .rename(columns={"value": "v1"})
                    )
                    b = (
                        d[d["model_id"] == m2][["retained_cycle_index", "value"]]
                        .rename(columns={"value": "v2"})
                    )

                    merged = a.merge(b, on="retained_cycle_index", how="inner").sort_values("retained_cycle_index")

                    n_common = len(merged)

                    if n_common < 5:
                        rows.append(
                            {
                                "parameter": parameter,
                                "order_group": order_group,
                                "model_1": m1,
                                "model_2": m2,
                                "n_common_cycles": n_common,
                                "raw_corr": np.nan,
                                "normalized_corr": np.nan,
                                "normalized_rmse": np.nan,
                                "slope_1": np.nan,
                                "slope_2": np.nan,
                                "slope_abs_diff": np.nan,
                                "median_1": np.nan,
                                "median_2": np.nan,
                                "median_rel_diff": np.nan,
                            }
                        )
                        continue

                    x = merged["retained_cycle_index"].to_numpy(dtype=float)
                    v1 = merged["v1"].to_numpy(dtype=float)
                    v2 = merged["v2"].to_numpy(dtype=float)

                    mask = np.isfinite(x) & np.isfinite(v1) & np.isfinite(v2)

                    if mask.sum() < 5:
                        continue

                    x = x[mask]
                    v1 = v1[mask]
                    v2 = v2[mask]

                    z1 = normalize_series(v1)
                    z2 = normalize_series(v2)

                    raw_corr = float(np.corrcoef(v1, v2)[0, 1]) if np.std(v1) > 0 and np.std(v2) > 0 else np.nan
                    norm_corr = float(np.corrcoef(z1, z2)[0, 1]) if np.nanstd(z1) > 0 and np.nanstd(z2) > 0 else np.nan
                    norm_rmse = rmse(z1, z2)

                    slope_1 = slope_per_cycle(x, v1)
                    slope_2 = slope_per_cycle(x, v2)
                    slope_abs_diff = abs(slope_1 - slope_2) if np.isfinite(slope_1) and np.isfinite(slope_2) else np.nan

                    median_1 = float(np.nanmedian(v1))
                    median_2 = float(np.nanmedian(v2))
                    denom = max(abs(median_1), abs(median_2), 1e-15)
                    median_rel_diff = abs(median_1 - median_2) / denom

                    rows.append(
                        {
                            "parameter": parameter,
                            "order_group": order_group,
                            "model_1": m1,
                            "model_2": m2,
                            "n_common_cycles": int(mask.sum()),
                            "raw_corr": raw_corr,
                            "normalized_corr": norm_corr,
                            "normalized_rmse": norm_rmse,
                            "slope_1": slope_1,
                            "slope_2": slope_2,
                            "slope_abs_diff": slope_abs_diff,
                            "median_1": median_1,
                            "median_2": median_2,
                            "median_rel_diff": median_rel_diff,
                        }
                    )

    out = pd.DataFrame(rows)

    # Similarity score: high correlation, low normalized RMSE, low median difference.
    # Lower is better.
    out["similarity_score"] = (
        (1.0 - out["normalized_corr"].clip(-1, 1))
        + out["normalized_rmse"]
        + out["median_rel_diff"]
    )

    return out.sort_values(["parameter", "order_group", "similarity_score"]).reset_index(drop=True)


def make_s14_s17_similarity(sim: pd.DataFrame) -> pd.DataFrame:
    d = sim[
        (
            sim["model_1"].str.startswith("S14")
            & sim["model_2"].str.startswith("S17")
        )
        |
        (
            sim["model_1"].str.startswith("S17")
            & sim["model_2"].str.startswith("S14")
        )
    ].copy()

    return d.sort_values(["parameter", "similarity_score"]).reset_index(drop=True)


def make_best_matching_pairs(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for parameter in CORE_PARAMS:
        d = sim[sim["parameter"] == parameter].copy()
        d = d.dropna(subset=["similarity_score"]).copy()

        if len(d) == 0:
            continue

        rows.append(d.sort_values("similarity_score").head(10))

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("INDIVIDUAL CORE PARAMETER PLOTS AND SIMILARITY METRICS")
    print("=" * 100)
    print("Input:", IN_TABLE)
    print("Output tables:", OUT_TABLE_DIR)
    print("Output figures:", OUT_FIG_DIR)
    print("=" * 100)

    df = load_core_data()

    # Save clean input copy.
    clean_path = OUT_TABLE_DIR / "core_parameters_long_clean_for_individual_plots.csv"
    df.to_csv(clean_path, index=False)
    print("[saved]", clean_path)

    # Individual figures.
    for param in CORE_PARAMS:
        plot_parameter_all_models(df, param, normalized=False)
        plot_parameter_all_models(df, param, normalized=True)

        for order_group in ORDER_GROUPS:
            plot_parameter_by_order(df, param, order_group, normalized=False)
            plot_parameter_by_order(df, param, order_group, normalized=True)

        plot_parameter_s14_s17_focus(df, param, normalized=False)
        plot_parameter_s14_s17_focus(df, param, normalized=True)

        plot_parameter_c4_c4k_focus(df, param, normalized=False)
        plot_parameter_c4_c4k_focus(df, param, normalized=True)

    # Similarity metrics.
    sim = make_pairwise_similarity(df)

    sim_path = OUT_TABLE_DIR / "core_parameter_pairwise_similarity_detailed.csv"
    sim.to_csv(sim_path, index=False)
    print("[saved]", sim_path)

    s14s17 = make_s14_s17_similarity(sim)

    s14s17_path = OUT_TABLE_DIR / "core_parameter_s14_s17_similarity.csv"
    s14s17.to_csv(s14s17_path, index=False)
    print("[saved]", s14s17_path)

    best_pairs = make_best_matching_pairs(sim)

    best_path = OUT_TABLE_DIR / "core_parameter_best_matching_pairs.csv"
    best_pairs.to_csv(best_path, index=False)
    print("[saved]", best_path)

    print()
    print("Best S14/S17 similarities:")
    if len(s14s17):
        cols = [
            "parameter",
            "order_group",
            "model_1",
            "model_2",
            "normalized_corr",
            "normalized_rmse",
            "median_rel_diff",
            "similarity_score",
        ]
        print(s14s17[cols].head(30).to_string(index=False))
    else:
        print("No S14/S17 pairs found.")

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Most useful figures:")
    for param in CORE_PARAMS:
        print(" ", OUT_FIG_DIR / f"fig_ch6_core_{param}_S14_S17_focus_raw.png")
        print(" ", OUT_FIG_DIR / f"fig_ch6_core_{param}_S14_S17_focus_normalized.png")
    print()
    print("Most useful tables:")
    print(" ", sim_path)
    print(" ", s14s17_path)
    print(" ", best_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
