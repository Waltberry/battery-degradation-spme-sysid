from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _finalize(save_path=None, show: bool = False):
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_parameter_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str | None = None,
    ylabel: str | None = None,
    save_path=None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[x_col], df[y_col], marker="o")
    ax.grid(True)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"{y_col} vs {x_col}")
    _finalize(save_path=save_path, show=show)


def plot_metric_vs_cycle(
    df: pd.DataFrame,
    metric_col: str,
    cycle_col: str = "cycle_idx",
    title: str | None = None,
    ylabel: str | None = None,
    save_path=None,
    show: bool = False,
):
    plot_parameter_trend(
        df=df,
        x_col=cycle_col,
        y_col=metric_col,
        title=title or f"{metric_col} vs cycle",
        ylabel=ylabel or metric_col,
        save_path=save_path,
        show=show,
    )


def plot_multi_parameter_trends(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    ylabel: str,
    save_path=None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(11, 5))

    for col in y_cols:
        if col in df.columns:
            ax.plot(df[x_col], df[col], marker="o", label=col)

    ax.grid(True)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    _finalize(save_path=save_path, show=show)


def plot_thetaA_vs_cycle(
    df: pd.DataFrame,
    cycle_col: str = "cycle_idx",
    save_path=None,
    show: bool = False,
):
    cols = [f"thetaA_hat_{k}" for k in range(1, 8) if f"thetaA_hat_{k}" in df.columns]
    plot_multi_parameter_trends(
        df=df,
        x_col=cycle_col,
        y_cols=cols,
        title="ThetaA parameters vs cycle",
        ylabel="thetaA",
        save_path=save_path,
        show=show,
    )


def plot_thetaB_vs_cycle(
    df: pd.DataFrame,
    cycle_col: str = "cycle_idx",
    save_path=None,
    show: bool = False,
):
    cols = [f"thetaB_hat_{k}" for k in [8, 9, 10, 11] if f"thetaB_hat_{k}" in df.columns]
    plot_multi_parameter_trends(
        df=df,
        x_col=cycle_col,
        y_cols=cols,
        title="ThetaB parameters vs cycle",
        ylabel="thetaB",
        save_path=save_path,
        show=show,
    )