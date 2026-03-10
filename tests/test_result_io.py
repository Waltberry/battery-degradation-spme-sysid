from pathlib import Path

import pandas as pd

from battery_deg_spme.io.result_io import ensure_parent, load_json, save_json


def test_save_and_load_json(tmp_path):
    path = tmp_path / "nested" / "file.json"
    data = {"a": 1, "b": "x"}
    save_json(data, path)
    out = load_json(path)
    assert out == data


def test_ensure_parent(tmp_path):
    path = tmp_path / "a" / "b" / "file.json"
    p = ensure_parent(path)
    assert isinstance(p, Path)
    assert p.parent.exists()