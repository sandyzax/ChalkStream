from dataclasses import asdict, dataclass
from threading import Lock


@dataclass
class RuntimeSettingsSnapshot:
    preprocess_history: int = 50
    preprocess_var_threshold: float = 25.0
    segment_kernel_width_frac: float = 0.025
    segment_kernel_height_frac: float = 0.025
    segment_max_area_pct: float = 0.50
    segment_occluder_area_frac: float = 0.45


class RuntimeSettingsStore:
    def __init__(self):
        self._lock = Lock()
        self._settings = RuntimeSettingsSnapshot()

    def snapshot(self):
        with self._lock:
            return RuntimeSettingsSnapshot(**asdict(self._settings))

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._settings, key):
                    raise AttributeError(f"Unknown runtime setting: {key}")
                setattr(self._settings, key, value)
            return RuntimeSettingsSnapshot(**asdict(self._settings))

    def reset_defaults(self):
        with self._lock:
            self._settings = RuntimeSettingsSnapshot()
            return RuntimeSettingsSnapshot(**asdict(self._settings))
