from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finalize(save_path=None, show: bool = False):
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_selected_cycle(
    cycle_df,
    i_col: str,
    chosen: int,
    mode: str,
    save_path=None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cycle_df.index.values, cycle_df[i_col].values, label="Current")
    ax.grid(True)
    ax.set_title(f"Selected discharge cycle (index={chosen}, mode={mode})")
    ax.set_ylabel(i_col)
    ax.set_xlabel("time [s]")
    ax.legend()
    _finalize(save_path=save_path, show=show)


def plot_cycle_voltage_current(
    t,
    u,
    y,
    title: str = "Cycle Voltage and Current",
    save_path=None,
    show: bool = False,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, u)
    axes[0].set_ylabel("Current [A]")
    axes[0].set_title(title)
    axes[0].grid(True)

    axes[1].plot(t, y)
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True)

    _finalize(save_path=save_path, show=show)


def plot_cycle_boundaries(
    df,
    i_col: str,
    cycle_meta,
    title: str = "Detected discharge cycles",
    save_path=None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index.values, df[i_col].values, label="Current")

    for k, meta in enumerate(cycle_meta):
        ax.axvspan(
            meta["t_start"],
            meta["t_end"],
            alpha=0.15,
            label="Cycle" if k == 0 else None,
        )

    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(i_col)
    ax.grid(True)
    ax.legend()

    _finalize(save_path=save_path, show=show)