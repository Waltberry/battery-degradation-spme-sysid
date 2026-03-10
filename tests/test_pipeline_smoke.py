from battery_deg_spme.config.settings import get_default_settings


def test_settings_smoke():
    settings = get_default_settings()
    assert settings is not None
    assert settings.data.mpr_path is not None
    assert settings.surrogate.poly_deg >= 1