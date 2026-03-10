import numpy as np

from battery_deg_spme.fitting.least_squares import (
    compare_ls_solvers,
    solve_ls_normal_eq,
    solve_ls_qr,
)


def test_solve_ls_qr_matches_linear_solution():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 2.0, 3.0])
    beta = solve_ls_qr(X, y)
    assert beta.shape == (2,)
    assert np.allclose(X @ beta, y)


def test_solve_ls_normal_eq_shape():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 2.0, 3.0])
    beta = solve_ls_normal_eq(X, y)
    assert beta.shape == (2,)


def test_compare_ls_solvers_small_difference():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = compare_ls_solvers(X, y)
    assert out["diff_qr_np"] < 1e-8
    assert out["diff_ne_np"] < 1e-8