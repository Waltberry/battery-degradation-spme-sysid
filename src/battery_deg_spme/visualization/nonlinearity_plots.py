from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def _apply_plain_colorbar(fig, mappable, ax, label: str):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.8, label=label)
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    cb.formatter = fmt
    cb.update_ticks()
    return cb


def _finalize(save_path=None, show: bool = False):
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_nonlinearity_heatmap(
    XN: np.ndarray,
    XP: np.ndarray,
    Z: np.ndarray,
    title: str = "Learned Nonlinearity Heatmap",
    save_path=None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(XN, XP, Z, shading="auto")
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_p$")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    _apply_plain_colorbar(fig, im, ax, "Z [V]")
    _finalize(save_path=save_path, show=show)


def plot_nonlinearity_surface(
    XN: np.ndarray,
    XP: np.ndarray,
    Z: np.ndarray,
    title: str = "Learned Nonlinearity 3D Surface",
    save_path=None,
    show: bool = False,
):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XN, XP, Z, linewidth=0, antialiased=True)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_p$")
    ax.set_zlabel(r"$Z$ [V]")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.7, label="Z [V]")
    _finalize(save_path=save_path, show=show)


def plot_heatmap(
    XN: np.ndarray,
    XP: np.ndarray,
    Z: np.ndarray,
    title: str = "Heatmap",
    save_path=None,
    show: bool = False,
):
    plot_nonlinearity_heatmap(
        XN=XN,
        XP=XP,
        Z=Z,
        title=title,
        save_path=save_path,
        show=show,
    )


def plot_surface_3d(
    XN: np.ndarray,
    XP: np.ndarray,
    Z: np.ndarray,
    title: str = "3D Surface",
    save_path=None,
    show: bool = False,
):
    plot_nonlinearity_surface(
        XN=XN,
        XP=XP,
        Z=Z,
        title=title,
        save_path=save_path,
        show=show,
    )


def plot_surrogate_heatmap(
    xn_grid: np.ndarray,
    xp_grid: np.ndarray,
    z_grid: np.ndarray,
    title: str = "Surrogate Heatmap",
    save_path=None,
    show: bool = False,
):
    XN, XP = np.meshgrid(xn_grid, xp_grid, indexing="xy")
    plot_nonlinearity_heatmap(
        XN=XN,
        XP=XP,
        Z=z_grid,
        title=title,
        save_path=save_path,
        show=show,
    )


def plot_surrogate_surface_3d(
    xn_grid: np.ndarray,
    xp_grid: np.ndarray,
    z_grid: np.ndarray,
    title: str = "Surrogate 3D Surface",
    save_path=None,
    show: bool = False,
):
    XN, XP = np.meshgrid(xn_grid, xp_grid, indexing="xy")
    plot_nonlinearity_surface(
        XN=XN,
        XP=XP,
        Z=z_grid,
        title=title,
        save_path=save_path,
        show=show,
    )


def plot_surrogate_contours(
    xn_grid: np.ndarray,
    xp_grid: np.ndarray,
    z_grid: np.ndarray,
    title: str = "Surrogate Contours",
    save_path=None,
    show: bool = False,
):
    XN, XP = np.meshgrid(xn_grid, xp_grid, indexing="xy")

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(XN, XP, z_grid, levels=20)
    ax.contour(XN, XP, z_grid, levels=10, linewidths=0.5)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_p$")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    _apply_plain_colorbar(fig, cs, ax, "Z [V]")
    _finalize(save_path=save_path, show=show)


def plot_surface_3d_compare(
    XN: np.ndarray,
    XP: np.ndarray,
    Z_ref: np.ndarray,
    Z_hat: np.ndarray,
    title: str = "Reference vs Learned",
    save_path=None,
    show: bool = False,
):
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    s1 = ax1.plot_surface(XN, XP, Z_ref, linewidth=0, antialiased=True)
    ax1.set_title("Reference")
    ax1.set_xlabel(r"$x_n$")
    ax1.set_ylabel(r"$x_p$")
    ax1.set_zlabel("Z [V]")
    fig.colorbar(s1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    s2 = ax2.plot_surface(XN, XP, Z_hat, linewidth=0, antialiased=True)
    ax2.set_title("Learned")
    ax2.set_xlabel(r"$x_n$")
    ax2.set_ylabel(r"$x_p$")
    ax2.set_zlabel("Z_hat [V]")
    fig.colorbar(s2, ax=ax2, shrink=0.6)

    fig.suptitle(title)
    _finalize(save_path=save_path, show=show)


def plot_residual_heatmap(
    XN: np.ndarray,
    XP: np.ndarray,
    R: np.ndarray,
    title: str = "Residual Heatmap",
    save_path=None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(XN, XP, R, shading="auto")
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_p$")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    _apply_plain_colorbar(fig, im, ax, "Residual [V]")
    _finalize(save_path=save_path, show=show)


def plot_residual_surface_3d(
    XN: np.ndarray,
    XP: np.ndarray,
    R: np.ndarray,
    title: str = "Residual 3D Surface",
    save_path=None,
    show: bool = False,
):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XN, XP, R, linewidth=0, antialiased=True)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_p$")
    ax.set_zlabel("Residual [V]")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.7, label="Residual [V]")
    _finalize(save_path=save_path, show=show)