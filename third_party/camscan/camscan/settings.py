"""Persistent per-user settings for the VIAR scanner application."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import tempfile
import typing as t


DEFAULTS = {
    "appearance_mode": "System",
    "pdf_directory": "",
    "scan_size": "Авто (с коррекцией перспективы)",
    "postprocessing": "Без обработки",
    "bw_threshold": 50,
    "auto_portrait": True,
    "ui_scaling": "100%",
    "camera_index": 0,
    "camera_resolution": "2592x1944",
    "separate_file_type": "png",
}

APPEARANCE_MODES = ("System", "Light", "Dark")
UI_SCALINGS = ("80%", "90%", "100%", "110%", "120%")
SCAN_SIZES = (
    "Авто (с коррекцией перспективы)",
    "Авто (без коррекции перспективы)",
    "Пользовательский",
    "Множественные объекты",
    "A3 297х420 мм",
    "A4 210х297 мм",
    "A5 148х210 мм",
)
POSTPROCESSING_MODES = (
    "Без обработки",
    "Коррекция белого",
    "Повышенная резкость",
    "Оттенки серого",
    "Чёрно-белый",
)
SEPARATE_FILE_TYPES = (
    "png",
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
)
_RESOLUTION_PATTERN = re.compile(r"^([1-9]\d{1,4})x([1-9]\d{1,4})$")


def default_pdf_directory() -> Path:
    """Return a useful initial directory for the PDF save dialog."""
    documents = Path.home() / "Documents"
    return documents if documents.is_dir() else Path.home()


def default_config_path() -> Path:
    """Return the XDG-compatible per-user configuration path."""
    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_home and Path(xdg_home).is_absolute():
        base = Path(xdg_home)
    else:
        base = Path.home() / ".config"
    return base / "viar-scanner" / "settings.json"


def default_settings() -> dict:
    """Create defaults whose paths are evaluated for the current user."""
    result = dict(DEFAULTS)
    result["pdf_directory"] = str(default_pdf_directory())
    return result


def _valid_resolution(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _RESOLUTION_PATTERN.fullmatch(value)
    if match is None:
        return False
    width, height = (int(part) for part in match.groups())
    return 160 <= width <= 16384 and 120 <= height <= 16384


def normalize_settings(raw: object) -> dict:
    """Validate untrusted JSON data and replace invalid fields with defaults."""
    result = default_settings()
    if not isinstance(raw, dict):
        return result

    if raw.get("appearance_mode") in APPEARANCE_MODES:
        result["appearance_mode"] = raw["appearance_mode"]

    pdf_directory = raw.get("pdf_directory")
    if isinstance(pdf_directory, str) and pdf_directory.strip():
        candidate = Path(pdf_directory).expanduser()
        if candidate.is_dir():
            result["pdf_directory"] = str(candidate)

    if raw.get("scan_size") in SCAN_SIZES:
        result["scan_size"] = raw["scan_size"]

    if raw.get("postprocessing") in POSTPROCESSING_MODES:
        result["postprocessing"] = raw["postprocessing"]

    threshold = raw.get("bw_threshold")
    if (
        isinstance(threshold, int)
        and not isinstance(threshold, bool)
        and 0 <= threshold <= 100
    ):
        result["bw_threshold"] = threshold

    auto_portrait = raw.get("auto_portrait")
    if isinstance(auto_portrait, bool):
        result["auto_portrait"] = auto_portrait
    elif auto_portrait in (0, 1):
        result["auto_portrait"] = bool(auto_portrait)

    if raw.get("ui_scaling") in UI_SCALINGS:
        result["ui_scaling"] = raw["ui_scaling"]

    camera_index = raw.get("camera_index")
    if (
        isinstance(camera_index, int)
        and not isinstance(camera_index, bool)
        and 0 <= camera_index <= 99
    ):
        result["camera_index"] = camera_index

    if _valid_resolution(raw.get("camera_resolution")):
        result["camera_resolution"] = raw["camera_resolution"]

    if raw.get("separate_file_type") in SEPARATE_FILE_TYPES:
        result["separate_file_type"] = raw["separate_file_type"]

    return result


class SettingsStore:
    """Load and atomically save the application's per-user JSON settings."""

    def __init__(self, path: t.Optional[Path] = None):
        self.path = Path(path) if path is not None else default_config_path()

    def load(self) -> dict:
        try:
            with self.path.open("r", encoding="utf-8") as settings_file:
                raw = json.load(settings_file)
        except FileNotFoundError:
            return default_settings()
        except (OSError, ValueError) as error:
            logging.warning("Не удалось прочитать настройки %s: %s", self.path, error)
            return default_settings()
        return normalize_settings(raw)

    def save(self, settings: object) -> bool:
        normalized = normalize_settings(settings)
        temporary_path = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=".settings-",
                suffix=".tmp",
                delete=False,
            ) as settings_file:
                temporary_path = Path(settings_file.name)
                json.dump(
                    normalized,
                    settings_file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                settings_file.write("\n")
                settings_file.flush()
                os.fsync(settings_file.fileno())
            os.replace(str(temporary_path), str(self.path))
            try:
                os.chmod(str(self.path), 0o600)
            except OSError:
                pass
            return True
        except OSError as error:
            logging.warning("Не удалось сохранить настройки %s: %s", self.path, error)
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
            return False
