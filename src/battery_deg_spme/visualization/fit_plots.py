from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _plain_voltage_axis(ax):
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def plot_voltage(
    t,
    y,
    yhat=None,
    title: str = "Voltage",
    measured_label: str = "Measured",
    pred_label: str = "Pred",
    save_path=None,
    show: bool = False,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y[:, 0], label=measured_label)

    if yhat is not None:
        yhat = np.asarray(yhat, dtype=np.float64).reshape(-1, 1)
        ax.plot(t, yhat[:, 0], "--", label=pred_label)

    ax.grid(True)
    ax.legend()
    ax.set_xlabel("t [s]")
    ax.set_ylabel("V [V]")
    ax.set_title(title)
    _plain_voltage_axis(ax)

    _finalize(save_path=save_path, show=show)


def plot_current_and_voltage(
    t,
    u,
    y,
    title: str = "Current and Voltage",
    save_path=None,
    show: bool = False,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(t, u)
    axes[0].grid(True)
    axes[0].set_ylabel("I [A]")
    axes[0].set_title(title)

    axes[1].plot(t, y)
    axes[1].grid(True)
    axes[1].set_ylabel("V [V]")
    axes[1].set_xlabel("t [s]")
    _plain_voltage_axis(axes[1])

    _finalize(save_path=save_path, show=show)


def plot_residuals(
    t,
    y,
    yhat,
    title: str = "Residuals",
    save_path=None,
    show: bool = False,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)

    err = yhat - y

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, err)
    ax.grid(True)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Pred - Meas [V]")
    ax.set_title(title)
    _plain_voltage_axis(ax)

    _finalize(save_path=save_path, show=show)


def plot_voltage_with_baselines(
    t,
    y,
    curves: dict,
    title: str = "Voltage Comparison",
    save_path=None,
    show: bool = False,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, label="Measured", linewidth=2)

    for name, vals in curves.items():
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        ax.plot(t, vals, "--", label=name)

    ax.grid(True)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("V [V]")
    ax.set_title(title)
    ax.legend()
    _plain_voltage_axis(ax)

    _finalize(save_path=save_path, show=show)


def plot_stage_fit_comparison(
    t,
    y,
    stage_curves: dict,
    title: str = "Stage Fit Comparison",
    save_path=None,
    show: bool = False,
):
    """
    stage_curves example:
        {
            "Stage 2": yhat_stage2,
            "Stage 3a": yhat_stage3a,
            "Stage 3b": yhat_stage3b,
        }
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, label="Measured", linewidth=2)

    for name, vals in stage_curves.items():
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        ax.plot(t, vals, "--", label=name)

    ax.grid(True)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("V [V]")
    ax.set_title(title)
    ax.legend()
    _plain_voltage_axis(ax)

    _finalize(save_path=save_path, show=show)