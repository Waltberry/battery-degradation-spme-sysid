from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _finalize(save_path=None, show: bool = False):
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_raw_signals(
    df,
    i_col: str,
    v_col: str,
    title: str = "Raw Signals",
    save_path=None,
    show: bool = False,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df.index.values, df[i_col].values)
    axes[0].grid(True)
    axes[0].set_title(f"{title} - Current")
    axes[0].set_ylabel(i_col)

    axes[1].plot(df.index.values, df[v_col].values)
    axes[1].grid(True)
    axes[1].set_title(f"{title} - Voltage")
    axes[1].set_ylabel(v_col)
    axes[1].set_xlabel("time [s]")

    _finalize(save_path=save_path, show=show)