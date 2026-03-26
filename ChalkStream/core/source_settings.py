"""Persist lightweight per-source runtime preferences across app launches."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SETTINGS_PATH = REPO_ROOT / "artifacts" / "source_settings.json"


def ensure_source_settings_file(path: Path | None = None) -> Path:
    """Create the per-source settings file if it does not already exist."""
    settings_path = Path(path or SOURCE_SETTINGS_PATH)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if not settings_path.exists():
        settings_path.write_text("{}\n", encoding="utf-8")
    return settings_path


def source_settings_key(source_info) -> str | None:
    """Build a stable key for a camera, file, or saved-board source."""
    source = dict(source_info or {})
    kind = str(source.get("kind") or "").strip().lower()
    value = source.get("value")

    if kind == "camera":
        try:
            return f"camera::{int(value)}"
        except (TypeError, ValueError):
            return None

    if kind in {"file", "board"} and value:
        try:
            resolved = Path(str(value)).resolve()
        except OSError:
            resolved = Path(str(value))
        return f"{kind}::{str(resolved).lower()}"

    return None


def load_source_settings(path: Path | None = None) -> dict:
    """Read the full per-source settings registry."""
    settings_path = ensure_source_settings_file(path)
    try:
        with settings_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def get_source_settings(source_info, path: Path | None = None) -> dict:
    """Return persisted overrides for one source, if present."""
    key = source_settings_key(source_info)
    if not key:
        return {}
    registry = load_source_settings(path)
    payload = registry.get(key)
    return dict(payload) if isinstance(payload, dict) else {}


def save_source_settings(source_info, updates, path: Path | None = None) -> dict:
    """Persist selected overrides for one source and return the merged record."""
    key = source_settings_key(source_info)
    if not key:
        return {}

    settings_path = ensure_source_settings_file(path)
    registry = load_source_settings(settings_path)
    current = dict(registry.get(key) or {})
    current.update(dict(updates or {}))
    registry[key] = current

    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return current
