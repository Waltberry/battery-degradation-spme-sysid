import numpy as np

from battery_deg_spme.analysis.nonlinearity import (
    compute_shape_drift,
    evaluate_surface_on_grid,
    summarize_surface_complexity,
)


def test_compute_shape_drift_zero():
    Z = np.ones((5, 5))
    out = compute_shape_drift(Z, Z)
    assert out["rmse"] == 0.0
    assert out["mae"] == 0.0
    assert out["max_abs"] == 0.0


def test_evaluate_surface_on_grid_shape():
    def surface_fn(xn, xp):
        return xn + xp

    out = evaluate_surface_on_grid(surface_fn=surface_fn, n_per_axis=7, guard=1e-3)
    assert out.Z.shape == (7, 7)
    assert out.XN.shape == (7, 7)
    assert out.XP.shape == (7, 7)


def test_summarize_surface_complexity_keys():
    xn = np.linspace(0.0, 1.0, 5)
    xp = np.linspace(0.0, 1.0, 5)
    XN, XP = np.meshgrid(xn, xp, indexing="xy")
    Z = XN + XP
    out = summarize_surface_complexity(xn, xp, Z)
    assert "z_min" in out
    assert "z_max" in out
    assert "grad_mag_mean" in out