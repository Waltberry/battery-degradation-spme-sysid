# battery_deg_spme/analysis/nonlinearity.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from battery_deg_spme.visualization.nonlinearity_plots import (
    plot_heatmap,
    plot_nonlinearity_heatmap,
    plot_nonlinearity_surface,
    plot_residual_heatmap,
    plot_residual_surface_3d,
    plot_surface_3d,
    plot_surface_3d_compare,
    plot_surrogate_contours,
)


@dataclass
class NonlinearitySurfaceResult:
    xn_grid: np.ndarray
    xp_grid: np.ndarray
    XN: np.ndarray
    XP: np.ndarray
    Z: np.ndarray
    grad_xn: np.ndarray
    grad_xp: np.ndarray
    complexity: dict[str, float]
    title: str


@dataclass
class NonlinearityComparisonResult:
    xn_grid: np.ndarray
    xp_grid: np.ndarray
    XN: np.ndarray
    XP: np.ndarray
    Z_ref: np.ndarray
    Z_hat: np.ndarray
    residual: np.ndarray
    ref_complexity: dict[str, float]
    hat_complexity: dict[str, float]
    drift: dict[str, float]
    rmse: float
    mae: float
    max_abs: float
    title: str


def make_state_grid(
    xn_min: float,
    xn_max: float,
    xp_min: float,
    xp_max: float,
    n_xn: int = 80,
    n_xp: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xn_grid = np.linspace(float(xn_min), float(xn_max), int(n_xn))
    xp_grid = np.linspace(float(xp_min), float(xp_max), int(n_xp))
    XN, XP = np.meshgrid(xn_grid, xp_grid, indexing="xy")
    return xn_grid, xp_grid, XN, XP


def make_xn_xp_grid(
    n_per_axis: int = 60,
    guard: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(float(guard), 1.0 - float(guard), int(n_per_axis))
    XN, XP = np.meshgrid(xs, xs, indexing="xy")
    return XN, XP


def _default_state_template(n_states: int = 14) -> np.ndarray:
    return np.zeros(int(n_states), dtype=np.float64)


def _build_state_from_xn_xp(
    xn: float,
    xp: float,
    cfg,
    state_template: np.ndarray | None = None,
    ceL_value: float | None = None,
    ceR_value: float | None = None,
) -> np.ndarray:
    x = _default_state_template() if state_template is None else np.asarray(state_template, dtype=np.float64).copy()

    if x.ndim != 1:
        raise ValueError("state_template must be a 1D array.")

    if x.shape[0] < 14:
        raise ValueError("Expected state vector with at least 14 states.")

    x[3] = float(xn) * float(cfg.csn_max)
    x[7] = float(xp) * float(cfg.csp_max)

    if ceL_value is not None:
        if bool(getattr(cfg, "ce_is_deviation", True)):
            x[8] = float(ceL_value) - float(cfg.ce0)
        else:
            x[8] = float(ceL_value)

    if ceR_value is not None:
        if bool(getattr(cfg, "ce_is_deviation", True)):
            x[13] = float(ceR_value) - float(cfg.ce0)
        else:
            x[13] = float(ceR_value)

    return x


def summarize_surface_complexity(
    xn_grid: np.ndarray,
    xp_grid: np.ndarray,
    z_grid: np.ndarray,
) -> dict[str, float]:
    xn_grid = np.asarray(xn_grid, dtype=np.float64).reshape(-1)
    xp_grid = np.asarray(xp_grid, dtype=np.float64).reshape(-1)
    z_grid = np.asarray(z_grid, dtype=np.float64)

    if z_grid.shape != (xp_grid.size, xn_grid.size):
        raise ValueError(
            f"Expected z_grid shape {(xp_grid.size, xn_grid.size)}, got {z_grid.shape}."
        )

    dxn = float(xn_grid[1] - xn_grid[0]) if xn_grid.size > 1 else np.nan
    dxp = float(xp_grid[1] - xp_grid[0]) if xp_grid.size > 1 else np.nan

    grad_xp, grad_xn = np.gradient(z_grid, dxp, dxn)

    grad_mag = np.sqrt(grad_xn**2 + grad_xp**2)

    d2_xp = np.gradient(grad_xp, dxp, axis=0) if np.isfinite(dxp) else np.full_like(z_grid, np.nan)
    d2_xn = np.gradient(grad_xn, dxn, axis=1) if np.isfinite(dxn) else np.full_like(z_grid, np.nan)
    laplacian = d2_xp + d2_xn

    return {
        "z_min": float(np.min(z_grid)),
        "z_max": float(np.max(z_grid)),
        "z_mean": float(np.mean(z_grid)),
        "z_std": float(np.std(z_grid)),
        "z_range": float(np.max(z_grid) - np.min(z_grid)),
        "grad_xn_std": float(np.nanstd(grad_xn)),
        "grad_xp_std": float(np.nanstd(grad_xp)),
        "grad_mag_mean": float(np.nanmean(grad_mag)),
        "grad_mag_max": float(np.nanmax(grad_mag)),
        "laplacian_mean_abs": float(np.nanmean(np.abs(laplacian))),
        "laplacian_max_abs": float(np.nanmax(np.abs(laplacian))),
    }


def evaluate_surrogate_on_grid(
    theta_z: np.ndarray,
    surrogate_fn: Callable[[np.ndarray, np.ndarray], float],
    xn_grid: np.ndarray,
    xp_grid: np.ndarray,
    xp_ref: float,
    xn_ref: float,
    xp_scale: float,
    xn_scale: float,
    cfg=None,
    ceL_value: float | None = None,
    ceR_value: float | None = None,
    state_template: np.ndarray | None = None,
) -> NonlinearitySurfaceResult:
    theta_z = np.asarray(theta_z, dtype=np.float64).reshape(-1)
    xn_grid = np.asarray(xn_grid, dtype=np.float64).reshape(-1)
    xp_grid = np.asarray(xp_grid, dtype=np.float64).reshape(-1)

    XN, XP = np.meshgrid(xn_grid, xp_grid, indexing="xy")
    Z = np.zeros((xp_grid.size, xn_grid.size), dtype=np.float64)

    for i, xp in enumerate(xp_grid):
        for j, xn in enumerate(xn_grid):
            if cfg is not None:
                x = _build_state_from_xn_xp(
                    xn=float(xn),
                    xp=float(xp),
                    cfg=cfg,
                    state_template=state_template,
                    ceL_value=ceL_value,
                    ceR_value=ceR_value,
                )
            else:
                x = _default_state_template()
                x[3] = float(xn)
                x[7] = float(xp)

            Z[i, j] = float(surrogate_fn(x, theta_z))

    dxn = float(xn_grid[1] - xn_grid[0]) if xn_grid.size > 1 else np.nan
    dxp = float(xp_grid[1] - xp_grid[0]) if xp_grid.size > 1 else np.nan
    grad_xp, grad_xn = np.gradient(Z, dxp, dxn)

    complexity = summarize_surface_complexity(
        xn_grid=xn_grid,
        xp_grid=xp_grid,
        z_grid=Z,
    )

    return NonlinearitySurfaceResult(
        xn_grid=xn_grid,
        xp_grid=xp_grid,
        XN=XN,
        XP=XP,
        Z=Z,
        grad_xn=grad_xn,
        grad_xp=grad_xp,
        complexity=complexity,
        title="learned_nonlinearity",
    )


def compute_surrogate_surface(
    zhat_function: Callable[[np.ndarray, np.ndarray], float],
    thetaZ: np.ndarray,
    xn_range=(0.01, 0.99),
    xp_range=(0.01, 0.99),
    n_grid: int = 80,
    cfg=None,
    ceL_value: float | None = None,
    ceR_value: float | None = None,
    state_template: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xn_grid, xp_grid, XN, XP = make_state_grid(
        xn_min=float(xn_range[0]),
        xn_max=float(xn_range[1]),
        xp_min=float(xp_range[0]),
        xp_max=float(xp_range[1]),
        n_xn=int(n_grid),
        n_xp=int(n_grid),
    )

    result = evaluate_surrogate_on_grid(
        theta_z=np.asarray(thetaZ, dtype=np.float64).reshape(-1),
        surrogate_fn=zhat_function,
        xn_grid=xn_grid,
        xp_grid=xp_grid,
        xp_ref=0.0,
        xn_ref=0.0,
        xp_scale=1.0,
        xn_scale=1.0,
        cfg=cfg,
        ceL_value=ceL_value,
        ceR_value=ceR_value,
        state_template=state_template,
    )

    return result.XN, result.XP, result.Z


def evaluate_learned_nonlinearity_on_grid(
    cfg,
    thetaZ_hat: np.ndarray,
    zhat_from_thetaZ: Callable[[np.ndarray, np.ndarray], float],
    xp_ref: float,
    xn_ref: float,
    xp_scale: float,
    xn_scale: float,
    xp_min: float,
    xp_max: float,
    xn_min: float,
    xn_max: float,
    n_grid: int = 80,
    ceL_value: float | None = None,
    ceR_value: float | None = None,
    state_template: np.ndarray | None = None,
) -> dict[str, Any]:
    xn_grid, xp_grid, _, _ = make_state_grid(
        xn_min=xn_min,
        xn_max=xn_max,
        xp_min=xp_min,
        xp_max=xp_max,
        n_xn=n_grid,
        n_xp=n_grid,
    )

    result = evaluate_surrogate_on_grid(
        theta_z=np.asarray(thetaZ_hat, dtype=np.float64).reshape(-1),
        surrogate_fn=zhat_from_thetaZ,
        xn_grid=xn_grid,
        xp_grid=xp_grid,
        xp_ref=xp_ref,
        xn_ref=xn_ref,
        xp_scale=xp_scale,
        xn_scale=xn_scale,
        cfg=cfg,
        ceL_value=ceL_value,
        ceR_value=ceR_value,
        state_template=state_template,
    )

    return {
        "xp_grid": result.xp_grid,
        "xn_grid": result.xn_grid,
        "XN": result.XN,
        "XP": result.XP,
        "Z": result.Z,
        "grad_xp": result.grad_xp,
        "grad_xn": result.grad_xn,
        "smoothness": result.complexity,
    }


def evaluate_surface_on_grid(
    surface_fn: Callable[[float, float], float],
    n_per_axis: int = 60,
    guard: float = 1e-4,
) -> NonlinearitySurfaceResult:
    xs = np.linspace(float(guard), 1.0 - float(guard), int(n_per_axis))
    xn_grid = xs.copy()
    xp_grid = xs.copy()
    XN, XP = np.meshgrid(xn_grid, xp_grid, indexing="xy")

    Z = np.zeros((xp_grid.size, xn_grid.size), dtype=np.float64)
    for i, xp in enumerate(xp_grid):
        for j, xn in enumerate(xn_grid):
            Z[i, j] = float(surface_fn(float(xn), float(xp)))

    dxn = float(xn_grid[1] - xn_grid[0]) if xn_grid.size > 1 else np.nan
    dxp = float(xp_grid[1] - xp_grid[0]) if xp_grid.size > 1 else np.nan
    grad_xp, grad_xn = np.gradient(Z, dxp, dxn)

    return NonlinearitySurfaceResult(
        xn_grid=xn_grid,
        xp_grid=xp_grid,
        XN=XN,
        XP=XP,
        Z=Z,
        grad_xn=grad_xn,
        grad_xp=grad_xp,
        complexity=summarize_surface_complexity(xn_grid, xp_grid, Z),
        title="nonlinearity_surface",
    )


def compare_surfaces_on_grid(
    ref_surface_fn: Callable[[float, float], float],
    learned_surface_fn: Callable[[float, float], float],
    n_per_axis: int = 60,
    guard: float = 1e-4,
    title: str = "nonlinearity_comparison",
) -> NonlinearityComparisonResult:
    ref_result = evaluate_surface_on_grid(
        surface_fn=ref_surface_fn,
        n_per_axis=n_per_axis,
        guard=guard,
    )

    hat_result = evaluate_surface_on_grid(
        surface_fn=learned_surface_fn,
        n_per_axis=n_per_axis,
        guard=guard,
    )

    residual = hat_result.Z - ref_result.Z
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))

    return NonlinearityComparisonResult(
        xn_grid=ref_result.xn_grid,
        xp_grid=ref_result.xp_grid,
        XN=ref_result.XN,
        XP=ref_result.XP,
        Z_ref=ref_result.Z,
        Z_hat=hat_result.Z,
        residual=residual,
        ref_complexity=ref_result.complexity,
        hat_complexity=hat_result.complexity,
        drift=compute_shape_drift(ref_result.Z, hat_result.Z),
        rmse=rmse,
        mae=mae,
        max_abs=max_abs,
        title=title,
    )


def compute_shape_drift(
    reference_Z: np.ndarray,
    comparison_Z: np.ndarray,
) -> dict[str, float]:
    reference_Z = np.asarray(reference_Z, dtype=np.float64)
    comparison_Z = np.asarray(comparison_Z, dtype=np.float64)

    if reference_Z.shape != comparison_Z.shape:
        raise ValueError(
            f"Surface shape mismatch: {reference_Z.shape} vs {comparison_Z.shape}."
        )

    residual = comparison_Z - reference_Z
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "mean_signed": float(np.mean(residual)),
        "std_signed": float(np.std(residual)),
    }


def compute_shape_drift_against_reference(
    reference_surface_fn: Callable[[float, float], float],
    comparison_surface_fn: Callable[[float, float], float],
    n_per_axis: int = 60,
    guard: float = 1e-4,
) -> dict[str, float]:
    comp = compare_surfaces_on_grid(
        ref_surface_fn=reference_surface_fn,
        learned_surface_fn=comparison_surface_fn,
        n_per_axis=n_per_axis,
        guard=guard,
        title="shape_drift",
    )
    return comp.drift


def build_nonlinearity_analysis_result(
    surface_result: NonlinearitySurfaceResult,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "xn_grid": surface_result.xn_grid,
        "xp_grid": surface_result.xp_grid,
        "XN": surface_result.XN,
        "XP": surface_result.XP,
        "Z": surface_result.Z,
        "grad_xn": surface_result.grad_xn,
        "grad_xp": surface_result.grad_xp,
        "complexity": surface_result.complexity,
        "title": surface_result.title,
    }
    if metadata is not None:
        out["metadata"] = metadata
    return out


def save_surface_visuals(
    result: NonlinearitySurfaceResult,
    output_dir: str | Path,
    prefix: str,
    show: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_nonlinearity_heatmap(
        result.XN,
        result.XP,
        result.Z,
        title=f"{prefix}: heatmap",
        save_path=output_dir / f"{prefix}_heatmap.png",
        show=show,
    )

    plot_nonlinearity_surface(
        result.XN,
        result.XP,
        result.Z,
        title=f"{prefix}: 3D surface",
        save_path=output_dir / f"{prefix}_surface3d.png",
        show=show,
    )

    plot_surrogate_contours(
        result.xn_grid,
        result.xp_grid,
        result.Z,
        title=f"{prefix}: contours",
        save_path=output_dir / f"{prefix}_contours.png",
        show=show,
    )


def save_comparison_visuals(
    result: NonlinearityComparisonResult,
    output_dir: str | Path,
    prefix: str,
    show: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_surface_3d_compare(
        result.XN,
        result.XP,
        result.Z_ref,
        result.Z_hat,
        title=f"{prefix}: reference vs learned",
        save_path=output_dir / f"{prefix}_compare3d.png",
        show=show,
    )

    plot_heatmap(
        result.XN,
        result.XP,
        result.Z_ref,
        title=f"{prefix}: reference heatmap",
        save_path=output_dir / f"{prefix}_reference_heatmap.png",
        show=show,
    )

    plot_surface_3d(
        result.XN,
        result.XP,
        result.Z_ref,
        title=f"{prefix}: reference surface",
        save_path=output_dir / f"{prefix}_reference_surface3d.png",
        show=show,
    )

    plot_heatmap(
        result.XN,
        result.XP,
        result.Z_hat,
        title=f"{prefix}: learned heatmap",
        save_path=output_dir / f"{prefix}_learned_heatmap.png",
        show=show,
    )

    plot_surface_3d(
        result.XN,
        result.XP,
        result.Z_hat,
        title=f"{prefix}: learned surface",
        save_path=output_dir / f"{prefix}_learned_surface3d.png",
        show=show,
    )

    plot_residual_heatmap(
        result.XN,
        result.XP,
        result.residual,
        title=f"{prefix}: residual heatmap",
        save_path=output_dir / f"{prefix}_residual_heatmap.png",
        show=show,
    )

    plot_residual_surface_3d(
        result.XN,
        result.XP,
        result.residual,
        title=f"{prefix}: residual 3D surface",
        save_path=output_dir / f"{prefix}_residual_surface3d.png",
        show=show,
    )