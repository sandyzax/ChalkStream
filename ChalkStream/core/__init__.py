"""Convenience exports for the core support layer."""

from .app_settings import APP_SETTINGS_PATH, ensure_app_settings_file, load_app_settings
from .board_storage import (
    SAVED_BOARDS_DIR,
    apply_interaction_state_snapshot,
    build_interaction_state_snapshot,
    discover_saved_boards,
    load_board_document,
    save_board_document,
)
from .camera_utils import (
    CAMERA_RESOLUTION_OPTIONS,
    camera_resolution_label,
    open_camera_capture,
    open_configured_camera_capture,
)
from .interaction_state import InteractionStateStore
from .run_console import RunConsole
from .runtime_settings import RuntimeSettingsSnapshot, RuntimeSettingsStore
from .source_settings import (
    SOURCE_SETTINGS_PATH,
    get_source_settings,
    save_source_settings,
)

__all__ = [
    "APP_SETTINGS_PATH",
    "CAMERA_RESOLUTION_OPTIONS",
    "SAVED_BOARDS_DIR",
    "apply_interaction_state_snapshot",
    "build_interaction_state_snapshot",
    "camera_resolution_label",
    "discover_saved_boards",
    "ensure_app_settings_file",
    "InteractionStateStore",
    "load_board_document",
    "load_app_settings",
    "open_camera_capture",
    "open_configured_camera_capture",
    "RunConsole",
    "RuntimeSettingsSnapshot",
    "RuntimeSettingsStore",
    "SOURCE_SETTINGS_PATH",
    "get_source_settings",
    "save_board_document",
    "save_source_settings",
]
