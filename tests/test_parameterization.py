import numpy as np

from battery_deg_spme.models.parameterization import (
    pos,
    raw_from_pos,
    softplus_inv,
    thetaA_nom_from_cfg,
    thetaB_nom_from_cfg,
)
from battery_deg_spme.models.spme_proxy import Config


def test_pos_positive():
    assert float(pos(0.0)) > 0.0
    assert float(pos(-10.0)) > 0.0


def test_softplus_roundtrip_like():
    val = 0.5
    raw = raw_from_pos(val)
    recovered = float(pos(raw))
    assert np.isclose(recovered, val, atol=1e-6)


def test_softplus_inv_positive():
    x = softplus_inv(0.5)
    assert np.isfinite(x)


def test_thetaA_nom_from_cfg_shape():
    cfg = Config()
    thetaA = thetaA_nom_from_cfg(cfg)
    assert thetaA.shape == (7,)
    assert np.all(thetaA > 0)


def test_thetaB_nom_from_cfg_shape():
    cfg = Config()
    thetaB = thetaB_nom_from_cfg(cfg)
    assert thetaB.shape == (4,)