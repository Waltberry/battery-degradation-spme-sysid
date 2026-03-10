import numpy as np

from battery_deg_spme.preprocessing.resampling import resample_uniform


def test_resample_uniform_shapes():
    t = np.array([0.0, 1.0, 2.0])
    u = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[4.0], [5.0], [6.0]])

    tg, ug, yg = resample_uniform(t, u, y, dt=1.0)

    assert tg.shape == (3,)
    assert ug.shape == (3, 1)
    assert yg.shape == (3, 1)
    assert np.allclose(tg, t)