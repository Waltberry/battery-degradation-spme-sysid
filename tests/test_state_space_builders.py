from battery_deg_spme.models.spme_proxy import Config
from battery_deg_spme.models.state_space_builders import (
    build_Ae,
    build_An,
    build_Ap,
    build_Be,
    build_Bn,
    build_Bp,
)


def test_state_space_builder_shapes():
    cfg = Config()
    assert build_An(cfg).shape == (4, 4)
    assert build_Ap(cfg).shape == (4, 4)
    assert build_Ae(cfg).shape == (6, 6)
    assert build_Bn(cfg).shape == (4, 1)
    assert build_Bp(cfg).shape == (4, 1)
    assert build_Be(cfg).shape == (6, 1)


def test_state_space_builder_finite_values():
    cfg = Config()
    for M in [build_An(cfg), build_Ap(cfg), build_Ae(cfg), build_Bn(cfg), build_Bp(cfg), build_Be(cfg)]:
        assert M is not None
        assert (M == M).all()


def test_Bn_Bp_sign_change_with_discharge_flag():
    cfg1 = Config(discharge_positive=False)
    cfg2 = Config(discharge_positive=True)

    bn1 = build_Bn(cfg1)
    bn2 = build_Bn(cfg2)
    bp1 = build_Bp(cfg1)
    bp2 = build_Bp(cfg2)

    assert bn1[-1, 0] == -bn2[-1, 0]
    assert bp1[-1, 0] == -bp2[-1, 0]