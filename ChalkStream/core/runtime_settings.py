"""Thread-safe runtime settings shared between the GUI and processing pipeline."""

from dataclasses import asdict, dataclass
from threading import Lock

from .app_settings import load_app_settings
from .camera_utils import DEFAULT_CAMERA_RESOLUTION_OPTION
from diagram_vectorization.vectorization import DEFAULT_VECTORIZATION_METHOD, normalize_vectorization_method


@dataclass
class RuntimeSettingsSnapshot:
    """Plain snapshot object used when the pipeline needs a stable settings view."""

    preprocess_history: int = 50
    preprocess_var_threshold: float = 25.0
    segment_kernel_width_frac: float = 0.025
    segment_kernel_height_frac: float = 0.025
    segment_match_iou_thresh: float = 0.35
    camera_resolution_option: str = DEFAULT_CAMERA_RESOLUTION_OPTION
    vectorization_method: str = DEFAULT_VECTORIZATION_METHOD
    equation_ocr_model: str = "Trained_V3"
    text_ocr_backend: str = "paddle"
    text_ocr_model: str = "TextOCR_V1"


class RuntimeSettingsStore:
    """Small synchronized store for live settings changes coming from the GUI."""

    def __init__(self):
        self._lock = Lock()
        self._settings = self._build_default_snapshot()

    @staticmethod
    def _build_default_snapshot():
        settings = load_app_settings()
        preprocessing = dict(settings.get("preprocessing") or {})
        segmentation = dict(settings.get("segmentation") or {})
        camera = dict(settings.get("camera") or {})
        orchestration = dict(settings.get("orchestration") or {})
        return RuntimeSettingsSnapshot(
            preprocess_history=int(preprocessing.get("history", 50)),
            preprocess_var_threshold=float(preprocessing.get("var_threshold", 25.0)),
            segment_kernel_width_frac=float(segmentation.get("kernel_width_frac", 0.025)),
            segment_kernel_height_frac=float(segmentation.get("kernel_height_frac", 0.025)),
            segment_match_iou_thresh=float(segmentation.get("match_iou_thresh", 0.35)),
            text_ocr_backend=str(segmentation.get("text_ocr_backend", "paddle")).strip().lower(),
            text_ocr_model=str(segmentation.get("text_ocr_model", "TextOCR_V1")),
            camera_resolution_option=str(camera.get("resolution_option", DEFAULT_CAMERA_RESOLUTION_OPTION)),
            vectorization_method=normalize_vectorization_method(
                orchestration.get("diagram_vectorization_method", DEFAULT_VECTORIZATION_METHOD)
            ),
            equation_ocr_model=str(orchestration.get("equation_ocr_model", "Trained_V3")),
        )

    def snapshot(self):
        """Return a detached copy so worker threads can read settings safely."""
        with self._lock:
            return RuntimeSettingsSnapshot(**asdict(self._settings))

    def update(self, **kwargs):
        """Apply one or more setting changes and return the updated snapshot."""
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._settings, key):
                    raise AttributeError(f"Unknown runtime setting: {key}")
                setattr(self._settings, key, value)
            return RuntimeSettingsSnapshot(**asdict(self._settings))

    def reset_defaults(self):
        """Restore the application defaults for all runtime-tunable values."""
        with self._lock:
            self._settings = self._build_default_snapshot()
            return RuntimeSettingsSnapshot(**asdict(self._settings))
