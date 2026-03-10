import numpy as np

from battery_deg_spme.models.spme_proxy import Config
from battery_deg_spme.models.surrogate_polynomial import (
    build_feature_matrix,
    make_additive_poly_surrogate_fns,
)


def test_feature_matrix_shape():
    n = 10
    deg = 3
    xp = np.linspace(0.6, 0.61, n)
    xn = np.linspace(0.59, 0.60, n)
    ceL = np.ones(n) * 1000
    ceR = np.ones(n) * 1000.1

    Phi = build_feature_matrix(
        xp_sig=xp,
        xn_sig=xn,
        ceL_sig=ceL,
        ceR_sig=ceR,
        deg=deg,
        xp_ref=float(np.mean(xp)),
        xn_ref=float(np.mean(xn)),
        xp_scale=float(np.std(xp)),
        xn_scale=float(np.std(xn)),
        use_ln_feature=False,
    )
    assert Phi.shape == (n, 1 + deg + deg)


def test_feature_matrix_shape_with_ln():
    n = 8
    deg = 2
    xp = np.linspace(0.6, 0.61, n)
    xn = np.linspace(0.59, 0.60, n)
    ceL = np.ones(n) * 1000
    ceR = np.ones(n) * 1001

    Phi = build_feature_matrix(
        xp_sig=xp,
        xn_sig=xn,
        ceL_sig=ceL,
        ceR_sig=ceR,
        deg=deg,
        xp_ref=float(np.mean(xp)),
        xn_ref=float(np.mean(xn)),
        xp_scale=float(np.std(xp)),
        xn_scale=float(np.std(xn)),
        use_ln_feature=True,
    )
    assert Phi.shape == (n, 1 + deg + deg + 1)


def test_make_additive_poly_surrogate_fns_outputs():
    cfg = Config()
    deg = 3
    make_thetaZ0, zhat_from_thetaZ, meta = make_additive_poly_surrogate_fns(
        cfg=cfg,
        deg=deg,
        xp_ref=0.5,
        xn_ref=0.5,
        xp_scale=0.1,
        xn_scale=0.1,
        use_ln_feature=True,
    )

    y = np.array([[3.0], [3.2], [3.1]])
    theta0 = make_thetaZ0(y)
    assert theta0.shape[0] == meta["n_thetaZ"]

    x = np.zeros(14, dtype=float)
    x[3] = 0.5 * cfg.csn_max
    x[7] = 0.6 * cfg.csp_max
    z = zhat_from_thetaZ(x, theta0)
    assert np.isfinite(float(z))