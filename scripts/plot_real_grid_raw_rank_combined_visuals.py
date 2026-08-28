#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% =====================================================
# CELL 0 — Imports
# =====================================================
"""
plot_real_grid_raw_rank_combined_visuals.py

Purpose
-------
Create combined raw-rank visuals for real-data CT-ID models.

This script does NOT rerun identification.

It reads the rank summary produced by:

    scripts/diagnose_real_grid_raw_rank_svd.py

Expected input:
    results/tables/real_grid_raw_rank_diagnostics/stored_raw_rank_summary_with_ratios.csv

or fallback:
    results/tables/real_grid_raw_rank_diagnostics/stored_raw_rank_summary.csv

Main visuals
------------
1. Heatmap of raw Phi rank:
       rows = state variant S7/S12/S14/S17
       cols = output candidate C1/C2/C3/C4

2. Heatmap of raw Phi rank ratio:
       rank(Phi) / n_phi

3. Heatmap of raw X rank:
       rank(Xhat) / nx

4. Heatmap of raw X rank ratio:
       rank(Xhat) / nx

5. Rank-deficit heatmaps:
       n_phi - rank(Phi)
       nx - rank(Xhat)

6. Rank evolution lines:
       x-axis = output candidate C1-C4
       y-axis = raw rank or rank ratio
       one line per state variant

7. RMSE vs rank-ratio scatter:
       shows whether lower RMSE is coming from full-rank models
       or from under-excited/high-flexibility models.

No centering.
No scaling.
No normalization.
Only raw stored rank values are visualized.
"""

from __future__ import annotations

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

IN_DIR = PROJECT_DIR / "results" / "tables" / "real_grid_raw_rank_diagnostics"
OUT_DIR = PROJECT_DIR / "results" / "real_grid_raw_rank_combined_visuals"
FIG_DIR = PROJECT_DIR / "results" / "figures" / "real_grid_raw_rank_combined_visuals"
TABLE_DIR = PROJECT_DIR / "results" / "tables" / "real_grid_raw_rank_combined_visuals"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["S7", "S12", "S14", "S17"]
CANDIDATE_ORDER = ["C1", "C2", "C3", "C4"]

STATE_LABELS = {
    "S7": "S7\n7 states",
    "S12": "S12\n12 states",
    "S14": "S14\n14 states",
    "S17": "S17\n17 states",
}

CANDIDATE_LABELS = {
    "C1": "C1\nlinear",
    "C2": "C2\nquadratic",
    "C3": "C3\ncubic",
    "C4": "C4\nquartic",
}

print("=" * 100)
print("COMBINED RAW-RANK VISUALS")
print("=" * 100)
print("PROJECT_DIR:", PROJECT_DIR)
print("IN_DIR:", IN_DIR)
print("FIG_DIR:", FIG_DIR)
print("TABLE_DIR:", TABLE_DIR)
print("=" * 100)


# %% =====================================================
# CELL 2 — Helper functions
# =====================================================
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("[saved figure]", path)
    plt.close()


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


def choose_cycle0_if_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple cycles exist, use cycle 0 for the S/C grid visual.
    If only one cycle exists, use everything.
    """
    if "cycle_index" not in df.columns:
        return df.copy()

    cycle_numeric = pd.to_numeric(df["cycle_index"], errors="coerce")
    d0 = df[cycle_numeric == 0].copy()

    if len(d0) > 0:
        return d0

    return df.copy()


def make_pivot(
    df: pd.DataFrame,
    value_col: str,
    aggfunc="first",
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="state_id",
        columns="candidate_id",
        values=value_col,
        aggfunc=aggfunc,
    )

    pivot = pivot.reindex(index=STATE_ORDER, columns=CANDIDATE_ORDER)
    return pivot


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    colorbar_label: str,
    out_path: Path,
    fmt: str = ".2f",
    log10: bool = False,
    cmap_label_values: bool = True,
) -> None:
    values = pivot.to_numpy(dtype=float)

    if log10:
        plot_values = np.log10(np.maximum(values, 1e-300))
        cbar_label = f"log10({colorbar_label})"
    else:
        plot_values = values
        cbar_label = colorbar_label

    plt.figure(figsize=(8.8, 6.4))
    ax = plt.gca()

    im = ax.imshow(plot_values, aspect="auto")
    plt.colorbar(im, label=cbar_label)

    ax.set_xticks(np.arange(len(CANDIDATE_ORDER)))
    ax.set_yticks(np.arange(len(STATE_ORDER)))

    ax.set_xticklabels([CANDIDATE_LABELS.get(c, c) for c in CANDIDATE_ORDER])
    ax.set_yticklabels([STATE_LABELS.get(s, s) for s in STATE_ORDER])

    if cmap_label_values:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )

    ax.set_xlabel("Output candidate")
    ax.set_ylabel("State variant")
    ax.set_title(title)
    plt.tight_layout()

    savefig(out_path)


def plot_rank_lines(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_path: Path,
    ylim=None,
) -> None:
    plt.figure(figsize=(9.8, 6.3))

    x = np.arange(len(CANDIDATE_ORDER))

    for state_id in STATE_ORDER:
        sub = df[df["state_id"] == state_id].copy()
        sub["candidate_sort"] = sub["candidate_id"].map(candidate_sort_key)
        sub = sub.sort_values("candidate_sort")

        y = []

        for c in CANDIDATE_ORDER:
            val = sub.loc[sub["candidate_id"] == c, y_col]
            if len(val):
                y.append(float(val.iloc[0]))
            else:
                y.append(np.nan)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            label=state_id,
        )

        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                plt.annotate(
                    f"{yi:.2f}" if "ratio" in y_col else f"{yi:.0f}",
                    xy=(xi, yi),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    plt.xticks(x, CANDIDATE_ORDER)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Output candidate order")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best")

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    savefig(out_path)


def plot_grouped_bars(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Grouped bar chart:
        x-axis = output candidate C1-C4
        bars = S7/S12/S14/S17
    """
    x = np.arange(len(CANDIDATE_ORDER))
    width = 0.18

    plt.figure(figsize=(10.5, 6.4))
    ax = plt.gca()

    for k, state_id in enumerate(STATE_ORDER):
        offsets = x + (k - 1.5) * width

        y = []
        for cand in CANDIDATE_ORDER:
            val = df.loc[
                (df["state_id"] == state_id) & (df["candidate_id"] == cand),
                y_col,
            ]

            if len(val):
                y.append(float(val.iloc[0]))
            else:
                y.append(np.nan)

        ax.bar(offsets, y, width=width, label=state_id)

    ax.set_xticks(x)
    ax.set_xticklabels(CANDIDATE_ORDER)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_xlabel("Output candidate")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")

    plt.tight_layout()
    savefig(out_path)


# %% =====================================================
# CELL 3 — Load rank summary
# =====================================================
path_with_ratios = IN_DIR / "stored_raw_rank_summary_with_ratios.csv"
path_plain = IN_DIR / "stored_raw_rank_summary.csv"

if path_with_ratios.exists():
    df = pd.read_csv(path_with_ratios)
    input_path = path_with_ratios
elif path_plain.exists():
    df = pd.read_csv(path_plain)
    input_path = path_plain
else:
    raise RuntimeError(
        "Could not find stored raw rank summary. Run this first:\n"
        "python -u scripts/diagnose_real_grid_raw_rank_svd.py"
    )

print("Loaded:", input_path)
print("Shape:", df.shape)

# Standardize numeric columns.
for col in df.columns:
    if (
        col.startswith("best_")
        or col.startswith("median_")
        or col.startswith("min_")
        or col.startswith("max_")
        or col in ["cycle_index", "n_runs"]
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["state_id"] = df["state_id"].astype(str)
df["candidate_id"] = df["candidate_id"].astype(str)

df["state_sort"] = df["state_id"].map(state_sort_key)
df["candidate_sort"] = df["candidate_id"].map(candidate_sort_key)

# Prefer cycle 0 for S/C grid visual.
d = choose_cycle0_if_available(df)
d = d.sort_values(["state_sort", "candidate_sort"]).reset_index(drop=True)

# Add ratios and deficits if missing.
if "best_phi_rank_ratio" not in d.columns:
    d["best_phi_rank_ratio"] = d["best_rank_phi_raw"] / d["best_ncols_phi_raw"]

if "best_X_rank_ratio" not in d.columns:
    d["best_X_rank_ratio"] = d["best_rank_X_raw"] / d["best_ncols_X_raw"]

d["best_phi_rank_deficit"] = d["best_ncols_phi_raw"] - d["best_rank_phi_raw"]
d["best_X_rank_deficit"] = d["best_ncols_X_raw"] - d["best_rank_X_raw"]

d.to_csv(OUT_DIR / "rank_grid_used_for_visuals.csv", index=False)
d.to_csv(TABLE_DIR / "rank_grid_used_for_visuals.csv", index=False)

print("\nRank grid used for visuals:")
print(
    d[
        [
            "cycle_index",
            "model_id",
            "best_rmse",
            "best_rank_phi_raw",
            "best_ncols_phi_raw",
            "best_phi_rank_ratio",
            "best_phi_rank_deficit",
            "best_rank_X_raw",
            "best_ncols_X_raw",
            "best_X_rank_ratio",
            "best_X_rank_deficit",
        ]
    ].to_string(index=False)
)


# %% =====================================================
# CELL 4 — Heatmaps: raw ranks
# =====================================================
plot_heatmap(
    make_pivot(d, "best_rank_phi_raw"),
    title=r"Raw output-feature rank $\mathrm{rank}(\Phi)$",
    colorbar_label=r"rank(Phi)",
    out_path=FIG_DIR / "combined_heatmap_raw_phi_rank.png",
    fmt=".0f",
)

plot_heatmap(
    make_pivot(d, "best_ncols_phi_raw"),
    title=r"Number of output-feature columns $n_{\phi}$",
    colorbar_label=r"n_phi",
    out_path=FIG_DIR / "combined_heatmap_phi_ncols.png",
    fmt=".0f",
)

plot_heatmap(
    make_pivot(d, "best_phi_rank_ratio"),
    title=r"Raw output-feature rank ratio $\mathrm{rank}(\Phi)/n_{\phi}$",
    colorbar_label=r"rank(Phi)/n_phi",
    out_path=FIG_DIR / "combined_heatmap_phi_rank_ratio.png",
    fmt=".2f",
)

plot_heatmap(
    make_pivot(d, "best_phi_rank_deficit"),
    title=r"Output-feature rank deficit $n_{\phi}-\mathrm{rank}(\Phi)$",
    colorbar_label=r"Phi rank deficit",
    out_path=FIG_DIR / "combined_heatmap_phi_rank_deficit.png",
    fmt=".0f",
)


# %% =====================================================
# CELL 5 — Heatmaps: state trajectory ranks
# =====================================================
plot_heatmap(
    make_pivot(d, "best_rank_X_raw"),
    title=r"Raw fitted-state trajectory rank $\mathrm{rank}(X_{\mathrm{hat}})$",
    colorbar_label=r"rank(Xhat)",
    out_path=FIG_DIR / "combined_heatmap_raw_X_rank.png",
    fmt=".0f",
)

plot_heatmap(
    make_pivot(d, "best_ncols_X_raw"),
    title=r"Number of fitted states $n_x$",
    colorbar_label=r"n_x",
    out_path=FIG_DIR / "combined_heatmap_X_ncols.png",
    fmt=".0f",
)

plot_heatmap(
    make_pivot(d, "best_X_rank_ratio"),
    title=r"Raw fitted-state rank ratio $\mathrm{rank}(X_{\mathrm{hat}})/n_x$",
    colorbar_label=r"rank(Xhat)/nx",
    out_path=FIG_DIR / "combined_heatmap_X_rank_ratio.png",
    fmt=".2f",
)

plot_heatmap(
    make_pivot(d, "best_X_rank_deficit"),
    title=r"Fitted-state rank deficit $n_x-\mathrm{rank}(X_{\mathrm{hat}})$",
    colorbar_label=r"X rank deficit",
    out_path=FIG_DIR / "combined_heatmap_X_rank_deficit.png",
    fmt=".0f",
)


# %% =====================================================
# CELL 6 — Heatmaps: condition numbers
# =====================================================
if "best_cond_phi_raw" in d.columns:
    plot_heatmap(
        make_pivot(d, "best_cond_phi_raw"),
        title=r"Raw condition number of $\Phi$",
        colorbar_label=r"cond(Phi)",
        out_path=FIG_DIR / "combined_heatmap_cond_phi_log10.png",
        fmt=".1e",
        log10=True,
    )

if "best_cond_X_raw" in d.columns:
    plot_heatmap(
        make_pivot(d, "best_cond_X_raw"),
        title=r"Raw condition number of $X_{\mathrm{hat}}$",
        colorbar_label=r"cond(Xhat)",
        out_path=FIG_DIR / "combined_heatmap_cond_X_log10.png",
        fmt=".1e",
        log10=True,
    )


# %% =====================================================
# CELL 7 — Rank evolution line plots
# =====================================================
plot_rank_lines(
    d,
    y_col="best_rank_phi_raw",
    y_label=r"Raw rank of $\Phi$",
    title=r"How raw $\Phi$ rank evolves as output order increases",
    out_path=FIG_DIR / "line_phi_rank_vs_candidate_by_state.png",
)

plot_rank_lines(
    d,
    y_col="best_ncols_phi_raw",
    y_label=r"Number of $\Phi$ columns",
    title=r"How output-feature dimension grows with output order",
    out_path=FIG_DIR / "line_phi_ncols_vs_candidate_by_state.png",
)

plot_rank_lines(
    d,
    y_col="best_phi_rank_ratio",
    y_label=r"Raw rank ratio $\mathrm{rank}(\Phi)/n_{\phi}$",
    title=r"Raw $\Phi$ rank ratio as model complexity increases",
    out_path=FIG_DIR / "line_phi_rank_ratio_vs_candidate_by_state.png",
    ylim=(-0.05, 1.08),
)

plot_rank_lines(
    d,
    y_col="best_phi_rank_deficit",
    y_label=r"$n_{\phi}-\mathrm{rank}(\Phi)$",
    title=r"Output-feature rank deficit as model complexity increases",
    out_path=FIG_DIR / "line_phi_rank_deficit_vs_candidate_by_state.png",
)

plot_rank_lines(
    d,
    y_col="best_rank_X_raw",
    y_label=r"Raw rank of $X_{\mathrm{hat}}$",
    title=r"How raw fitted-state rank changes with output order",
    out_path=FIG_DIR / "line_X_rank_vs_candidate_by_state.png",
)

plot_rank_lines(
    d,
    y_col="best_X_rank_ratio",
    y_label=r"Raw rank ratio $\mathrm{rank}(X_{\mathrm{hat}})/n_x$",
    title=r"Raw fitted-state rank ratio as model complexity increases",
    out_path=FIG_DIR / "line_X_rank_ratio_vs_candidate_by_state.png",
    ylim=(-0.05, 1.08),
)

plot_rank_lines(
    d,
    y_col="best_X_rank_deficit",
    y_label=r"$n_x-\mathrm{rank}(X_{\mathrm{hat}})$",
    title=r"Fitted-state rank deficit as model complexity increases",
    out_path=FIG_DIR / "line_X_rank_deficit_vs_candidate_by_state.png",
)


# %% =====================================================
# CELL 8 — Grouped bar charts
# =====================================================
plot_grouped_bars(
    d,
    y_col="best_phi_rank_deficit",
    y_label=r"$n_{\phi}-\mathrm{rank}(\Phi)$",
    title=r"Output-feature rank deficit grouped by state and output candidate",
    out_path=FIG_DIR / "bar_phi_rank_deficit_grouped.png",
)

plot_grouped_bars(
    d,
    y_col="best_X_rank_deficit",
    y_label=r"$n_x-\mathrm{rank}(X_{\mathrm{hat}})$",
    title=r"Fitted-state rank deficit grouped by state and output candidate",
    out_path=FIG_DIR / "bar_X_rank_deficit_grouped.png",
)

plot_grouped_bars(
    d,
    y_col="best_phi_rank_ratio",
    y_label=r"$\mathrm{rank}(\Phi)/n_{\phi}$",
    title=r"Output-feature rank ratio grouped by state and output candidate",
    out_path=FIG_DIR / "bar_phi_rank_ratio_grouped.png",
)

plot_grouped_bars(
    d,
    y_col="best_X_rank_ratio",
    y_label=r"$\mathrm{rank}(X_{\mathrm{hat}})/n_x$",
    title=r"Fitted-state rank ratio grouped by state and output candidate",
    out_path=FIG_DIR / "bar_X_rank_ratio_grouped.png",
)


# %% =====================================================
# CELL 9 — RMSE vs rank-ratio scatter
# =====================================================
if "best_rmse" in d.columns:
    plt.figure(figsize=(9.5, 6.3))
    plt.scatter(
        d["best_phi_rank_ratio"],
        d["best_rmse"],
        s=100,
    )

    for _, row in d.iterrows():
        plt.annotate(
            row["model_id"],
            xy=(row["best_phi_rank_ratio"], row["best_rmse"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.grid(True, alpha=0.35)
    plt.xlabel(r"Raw $\Phi$ rank ratio $\mathrm{rank}(\Phi)/n_{\phi}$")
    plt.ylabel("Best RMSE [V]")
    plt.title(r"Best RMSE versus raw output-feature rank ratio")
    plt.tight_layout()
    savefig(FIG_DIR / "scatter_best_rmse_vs_phi_rank_ratio.png")

    plt.figure(figsize=(9.5, 6.3))
    plt.scatter(
        d["best_X_rank_ratio"],
        d["best_rmse"],
        s=100,
    )

    for _, row in d.iterrows():
        plt.annotate(
            row["model_id"],
            xy=(row["best_X_rank_ratio"], row["best_rmse"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.grid(True, alpha=0.35)
    plt.xlabel(r"Raw $X_{\mathrm{hat}}$ rank ratio $\mathrm{rank}(X_{\mathrm{hat}})/n_x$")
    plt.ylabel("Best RMSE [V]")
    plt.title(r"Best RMSE versus raw fitted-state rank ratio")
    plt.tight_layout()
    savefig(FIG_DIR / "scatter_best_rmse_vs_X_rank_ratio.png")


# %% =====================================================
# CELL 10 — One combined dashboard figure
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

dashboard_specs = [
    (
        axes[0, 0],
        make_pivot(d, "best_phi_rank_ratio"),
        r"$\mathrm{rank}(\Phi)/n_{\phi}$",
        r"Raw $\Phi$ rank ratio",
        ".2f",
    ),
    (
        axes[0, 1],
        make_pivot(d, "best_phi_rank_deficit"),
        r"$n_{\phi}-\mathrm{rank}(\Phi)$",
        r"$\Phi$ rank deficit",
        ".0f",
    ),
    (
        axes[1, 0],
        make_pivot(d, "best_X_rank_ratio"),
        r"$\mathrm{rank}(X_{\mathrm{hat}})/n_x$",
        r"Raw $X_{\mathrm{hat}}$ rank ratio",
        ".2f",
    ),
    (
        axes[1, 1],
        make_pivot(d, "best_X_rank_deficit"),
        r"$n_x-\mathrm{rank}(X_{\mathrm{hat}})$",
        r"$X_{\mathrm{hat}}$ rank deficit",
        ".0f",
    ),
]

for ax, pivot, cbar_label, title, fmt in dashboard_specs:
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto")
    fig.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xticks(np.arange(len(CANDIDATE_ORDER)))
    ax.set_yticks(np.arange(len(STATE_ORDER)))
    ax.set_xticklabels(CANDIDATE_ORDER)
    ax.set_yticklabels(STATE_ORDER)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Output candidate")
    ax.set_ylabel("State variant")

fig.suptitle("Combined raw-rank dashboard for real-data S/C models", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])

savefig(FIG_DIR / "combined_raw_rank_dashboard.png")


# %% =====================================================
# CELL 11 — LaTeX-ready table
# =====================================================
latex_cols = [
    "cycle_index",
    "model_id",
    "best_rmse",
    "best_rank_phi_raw",
    "best_ncols_phi_raw",
    "best_phi_rank_ratio",
    "best_phi_rank_deficit",
    "best_rank_X_raw",
    "best_ncols_X_raw",
    "best_X_rank_ratio",
    "best_X_rank_deficit",
    "best_cond_phi_raw",
    "best_cond_X_raw",
]

latex_cols = [c for c in latex_cols if c in d.columns]

latex_table = d[latex_cols].copy()
latex_table.to_csv(TABLE_DIR / "combined_raw_rank_visual_table.csv", index=False)

latex_path = TABLE_DIR / "latex_combined_raw_rank_visual_table.tex"
latex_table.to_latex(
    latex_path,
    index=False,
    escape=False,
    float_format=lambda x: f"{x:.6g}",
)

print("[saved table]", TABLE_DIR / "combined_raw_rank_visual_table.csv")
print("[saved latex table]", latex_path)


# %% =====================================================
# CELL 12 — Final printout
# =====================================================
print("\n" + "=" * 100)
print("COMBINED RAW-RANK VISUALS COMPLETE")
print("=" * 100)
print("Figures saved to:", FIG_DIR)
print("Tables saved to:", TABLE_DIR)

print("\nMost important figures:")
print(" ", FIG_DIR / "combined_raw_rank_dashboard.png")
print(" ", FIG_DIR / "combined_heatmap_phi_rank_ratio.png")
print(" ", FIG_DIR / "combined_heatmap_X_rank_ratio.png")
print(" ", FIG_DIR / "line_phi_rank_ratio_vs_candidate_by_state.png")
print(" ", FIG_DIR / "line_X_rank_ratio_vs_candidate_by_state.png")
print(" ", FIG_DIR / "line_phi_rank_deficit_vs_candidate_by_state.png")
print(" ", FIG_DIR / "line_X_rank_deficit_vs_candidate_by_state.png")
print(" ", FIG_DIR / "scatter_best_rmse_vs_phi_rank_ratio.png")
print(" ", FIG_DIR / "scatter_best_rmse_vs_X_rank_ratio.png")
print("=" * 100)