import json
from pathlib import Path

from camscan.settings import (
    SettingsStore,
    default_config_path,
    default_settings,
    normalize_settings,
)


def test_default_config_path_uses_absolute_xdg_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert default_config_path() == tmp_path / "viar-scanner" / "settings.json"


def test_missing_settings_return_defaults(tmp_path):
    store = SettingsStore(tmp_path / "missing" / "settings.json")

    assert store.load() == default_settings()


def test_valid_settings_round_trip(tmp_path):
    pdf_directory = tmp_path / "pdf"
    pdf_directory.mkdir()
    path = tmp_path / "config" / "settings.json"
    expected = {
        "appearance_mode": "Dark",
        "pdf_directory": str(pdf_directory),
        "scan_size": "A4 210х297 мм",
        "postprocessing": "Чёрно-белый",
        "bw_threshold": 73,
        "auto_portrait": False,
        "ui_scaling": "110%",
        "camera_index": 2,
        "camera_resolution": "1600x1200",
        "separate_file_type": "tiff",
    }

    store = SettingsStore(path)
    assert store.save(expected)
    assert store.load() == expected
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_invalid_values_fall_back_to_defaults(tmp_path):
    invalid = {
        "appearance_mode": "Neon",
        "pdf_directory": str(tmp_path / "does-not-exist"),
        "scan_size": "Letter",
        "postprocessing": "Destroy",
        "bw_threshold": 101,
        "auto_portrait": "yes",
        "ui_scaling": "500%",
        "camera_index": -1,
        "camera_resolution": "0x0",
        "separate_file_type": "exe",
        "unknown": "ignored",
    }

    assert normalize_settings(invalid) == default_settings()


def test_corrupt_json_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert SettingsStore(path).load() == default_settings()
