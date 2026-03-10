import numpy as np

from battery_deg_spme.evaluation.metrics import (
    compute_scores,
    regression_metrics,
    report_fit,
    signal_span,
    summarize_err,
    vec_reshape,
)


def test_vec_reshape():
    x = np.array([1.0, 2.0, 3.0])
    y = vec_reshape(x)
    assert y.shape == (3, 1)


def test_compute_scores_rmse():
    y = np.array([[1.0], [2.0], [3.0]])
    yh = np.array([[1.0], [2.0], [3.0]])
    assert compute_scores(y, yh, fit="RMSE") == 0.0


def test_summarize_err_zero():
    y = np.array([[1.0], [2.0], [3.0]])
    out = summarize_err(y, y, name="zero")
    assert out["rmse"] == 0.0
    assert out["mae"] == 0.0
    assert out["max_abs"] == 0.0


def test_report_fit_keys():
    y = np.array([[1.0], [2.0], [3.0]])
    out = report_fit("fit", y, y)
    assert "rmse" in out
    assert "r2" in out
    assert "bfr" in out


def test_signal_span():
    out = signal_span([1.0, 4.0, 2.0])
    assert out["min"] == 1.0
    assert out["max"] == 4.0
    assert out["span"] == 3.0


def test_regression_metrics_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    out = regression_metrics(y, y)
    assert out["rmse"] == 0.0
    assert out["mae"] == 0.0
    assert out["max_abs"] == 0.0