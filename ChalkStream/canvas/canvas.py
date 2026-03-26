"""Qt GUI for source selection, live board playback, and saved-board editing."""

import sys
import subprocess
import traceback
import json
import re
import os
import base64
from pathlib import Path
from threading import Event, Lock
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsPixmapItem,
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsTextItem,
    QLineEdit,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QMessageBox,
    QGridLayout,
    QMenu,
    QSizePolicy,
    QToolButton,
    QSplitter,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QStackedWidget,
)
import cv2
import numpy as np
from PySide6.QtGui import QPainter
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QFontMetrics, QAction, QActionGroup, QPen, QBrush, QKeySequence
from PySide6.QtCore import Qt, QTimer, QRectF
import matplotlib.mathtext as mathtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from core import (
    APP_SETTINGS_PATH,
    CAMERA_RESOLUTION_OPTIONS,
    InteractionStateStore,
    RunConsole,
    RuntimeSettingsStore,
    apply_interaction_state_snapshot,
    build_interaction_state_snapshot,
    camera_resolution_label,
    discover_saved_boards,
    ensure_app_settings_file,
    load_board_document,
    load_app_settings,
    open_camera_capture,
    open_configured_camera_capture,
    save_board_document,
)
from diagram_vectorization.vectorization import (
    DEFAULT_VECTORIZATION_METHOD,
    VECTORIZATION_METHOD_OPTIONS,
    normalize_vectorization_method,
)
from segmentation.board_isolation import estimate_board_size

APP_SETTINGS = load_app_settings()
CAMERA_SETTINGS = APP_SETTINGS.get("camera", {})
ORCHESTRATION_SETTINGS = APP_SETTINGS.get("orchestration", {})

MATRIX_ENV_RE = re.compile(
    r"\\begin\{(?P<env>matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix)\}(?P<body>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)
DEFAULT_SEGMENT_INTERVAL_S = float(ORCHESTRATION_SETTINGS.get("segment_interval_s", 0.5))
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_LIBRARY_DIRS = (
    REPO_ROOT / "pipeline" / "videos",
    REPO_ROOT / "pipeline",
)
MAX_CAMERA_DISCOVERY_INDEX = max(
    0,
    int(os.environ.get("CHALKSTREAM_MAX_CAMERA_DISCOVERY_INDEX", str(CAMERA_SETTINGS.get("max_discovery_index", 5)))),
)
CLASSIFICATION_MODE_OPTIONS = (
    ("normal", "Normal Function"),
    ("diagram", "Diagram Only"),
    ("text", "Text Only"),
    ("equation", "Equation Only"),
)
REGION_OVERRIDE_OPTIONS = (
    ("", "Auto"),
    ("diagram", "Force Diagram"),
    ("text", "Force Text"),
    ("equation", "Force Equation"),
)
DEFAULT_VECTORIZATION_SELECTION = normalize_vectorization_method(
    ORCHESTRATION_SETTINGS.get("diagram_vectorization_method", DEFAULT_VECTORIZATION_METHOD)
)
TEXT_OCR_BACKEND_OPTIONS = (
    ("paddle", "PaddlePaddle (Local)"),
    ("trocr_endpoint", "TrOCR Endpoint"),
)
TEXT_OCR_MODEL_OPTIONS = (
    ("TextOCR_V1", "TextOCR_V1"),
)
DEFAULT_TEXT_OCR_BACKEND = str(APP_SETTINGS.get("segmentation", {}).get("text_ocr_backend", "paddle")).strip().lower()
DEFAULT_TEXT_OCR_MODEL = str(APP_SETTINGS.get("segmentation", {}).get("text_ocr_model", "TextOCR_V1"))
EQUATION_OCR_MODEL_OPTIONS = (
    ("Trained_V1", "Trained_V1"),
    ("Trained_V2", "Trained_V2"),
    ("Trained_V3", "Trained_V3"),
    ("Trained_VX", "Trained_VX"),
)
DEFAULT_EQUATION_OCR_MODEL = str(ORCHESTRATION_SETTINGS.get("equation_ocr_model", "Trained_V3"))


def cv2_to_qpixmap(cv_img):
    """
    Convert OpenCV image (numpy array) to QPixmap
    """

    if cv_img is None:
        return QPixmap()

    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w

    qimage = QImage(
        rgb_img.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888
    )
    return QPixmap.fromImage(qimage)


def qpixmap_to_cv2(pixmap):
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
    rgb = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    return rgb.copy()


def encode_cv2_image_base64(cv_img):
    if cv_img is None:
        return None
    ok, encoded = cv2.imencode(".png", cv_img)
    if not ok:
        raise RuntimeError("Could not encode image override as PNG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_cv2_image_base64(encoded_text):
    if not encoded_text:
        return None
    raw = base64.b64decode(str(encoded_text).encode("ascii"))
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode image override payload.")
    return image


def clone_canvas_payload(data):
    cloned = dict(data or {})
    content = cloned.get("content")
    if isinstance(content, np.ndarray):
        cloned["content"] = content.copy()
    elif isinstance(content, dict):
        cloned["content"] = dict(content)
    cloned["metadata"] = dict(cloned.get("metadata", {}) or {})
    return cloned


parser = mathtext.MathTextParser("path")
def is_valid_latex(expr):
    try:
        parser.parse(f"${expr}$")
        return True
    except Exception:
        return False


def render_plain_text_pixmap(text, pixel_size=24):
    text = str(text or "")
    font = QFont("Cambria Math")
    font.setPixelSize(max(12, int(pixel_size)))
    metrics = QFontMetrics(font)
    width = max(1, metrics.horizontalAdvance(text) + 8)
    height = max(1, metrics.height() + 8)

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.setFont(font)
    painter.setPen(QColor(0, 0, 0))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
    painter.end()
    return pixmap


def render_mathtext_pixmap(expr, size=16):
    expr = str(expr or "").strip()
    if not expr:
        return QPixmap()

    fig = Figure(figsize=(1, 1), dpi=max(72, int(size) * 40))
    fig.patch.set_alpha(0.0)
    FigureCanvasAgg(fig)
    fig.text(0, 0, f"${expr}$")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True)

    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())
    return pixmap


def render_expression_pixmap(expr, size=16):
    expr = str(expr or "").strip()
    if not expr:
        return QPixmap()

    try:
        return render_mathtext_pixmap(expr, size)
    except Exception:
        return render_plain_text_pixmap(expr, pixel_size=max(16, int(size) * 2))


def parse_matrix_expression(expr):
    expr = str(expr or "").strip()
    match = MATRIX_ENV_RE.search(expr)
    if match is None:
        return None

    env = match.group("env")
    body = match.group("body")
    prefix = expr[:match.start()].strip()
    suffix = expr[match.end():].strip()

    left_delim = None
    right_delim = None

    if env == "pmatrix":
        left_delim, right_delim = "(", ")"
    elif env == "bmatrix":
        left_delim, right_delim = "[", "]"
    elif env == "Bmatrix":
        left_delim, right_delim = "{", "}"
    elif env == "vmatrix":
        left_delim, right_delim = "|", "|"
    elif env == "Vmatrix":
        left_delim, right_delim = "||", "||"

    if env == "matrix":
        delimiter_pairs = [
            (r"\left[", r"\right]", "[", "]"),
            (r"\left(", r"\right)", "(", ")"),
            (r"\left\{", r"\right\}", "{", "}"),
            ("[", "]", "[", "]"),
            ("(", ")", "(", ")"),
            (r"\{", r"\}", "{", "}"),
        ]
        for prefix_token, suffix_token, left_char, right_char in delimiter_pairs:
            if prefix.endswith(prefix_token) and suffix.startswith(suffix_token):
                prefix = prefix[:-len(prefix_token)].rstrip()
                suffix = suffix[len(suffix_token):].lstrip()
                left_delim, right_delim = left_char, right_char
                break

    rows = []
    for row in re.split(r"(?<!\\)\\\\", body):
        cells = [cell.strip() for cell in re.split(r"(?<!\\)&", row)]
        if any(cells):
            rows.append(cells)

    if not rows:
        return None

    return {
        "prefix": prefix,
        "suffix": suffix,
        "left_delim": left_delim,
        "right_delim": right_delim,
        "rows": rows,
    }


def render_matrix_expression_pixmap(expr, size=16):
    parsed = parse_matrix_expression(expr)
    if parsed is None:
        return None

    rows = parsed["rows"]
    col_count = max(len(row) for row in rows)
    cell_padding_x = max(8, int(size))
    cell_padding_y = max(4, int(size * 0.5))

    cell_pixmaps = []
    col_widths = [0] * col_count
    row_heights = [0] * len(rows)

    for row_index, row in enumerate(rows):
        rendered_row = []
        for col_index in range(col_count):
            cell_expr = row[col_index] if col_index < len(row) else ""
            pixmap = render_expression_pixmap(cell_expr, size)
            rendered_row.append(pixmap)
            col_widths[col_index] = max(col_widths[col_index], pixmap.width())
            row_heights[row_index] = max(row_heights[row_index], pixmap.height())
        cell_pixmaps.append(rendered_row)

    prefix_pixmap = render_expression_pixmap(parsed["prefix"], size) if parsed["prefix"] else QPixmap()
    suffix_pixmap = render_expression_pixmap(parsed["suffix"], size) if parsed["suffix"] else QPixmap()

    matrix_width = sum(col_widths) + max(0, col_count - 1) * cell_padding_x
    matrix_height = sum(row_heights) + max(0, len(rows) - 1) * cell_padding_y

    delimiter_height = max(matrix_height + max(8, int(size)), max(prefix_pixmap.height(), suffix_pixmap.height(), 1))
    left_delim_pixmap = render_plain_text_pixmap(parsed["left_delim"], pixel_size=max(24, matrix_height + 4)) if parsed["left_delim"] else QPixmap()
    right_delim_pixmap = render_plain_text_pixmap(parsed["right_delim"], pixel_size=max(24, matrix_height + 4)) if parsed["right_delim"] else QPixmap()

    gap = max(6, int(size * 0.5))
    total_width = (
        prefix_pixmap.width()
        + (gap if not prefix_pixmap.isNull() else 0)
        + left_delim_pixmap.width()
        + matrix_width
        + right_delim_pixmap.width()
        + (gap if not suffix_pixmap.isNull() else 0)
        + suffix_pixmap.width()
        + 8
    )
    total_height = max(
        delimiter_height,
        prefix_pixmap.height(),
        suffix_pixmap.height(),
        left_delim_pixmap.height(),
        right_delim_pixmap.height(),
        1,
    ) + 8

    pixmap = QPixmap(max(1, total_width), max(1, total_height))
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    painter.setRenderHint(QPainter.TextAntialiasing)

    x = 4
    center_y = pixmap.height() // 2

    if not prefix_pixmap.isNull():
        painter.drawPixmap(x, center_y - prefix_pixmap.height() // 2, prefix_pixmap)
        x += prefix_pixmap.width() + gap

    if not left_delim_pixmap.isNull():
        painter.drawPixmap(x, center_y - left_delim_pixmap.height() // 2, left_delim_pixmap)
        x += left_delim_pixmap.width()

    matrix_top = center_y - matrix_height // 2
    current_y = matrix_top
    for row_index, rendered_row in enumerate(cell_pixmaps):
        current_x = x
        for col_index, cell_pixmap in enumerate(rendered_row):
            cell_x = current_x + (col_widths[col_index] - cell_pixmap.width()) // 2
            cell_y = current_y + (row_heights[row_index] - cell_pixmap.height()) // 2
            if not cell_pixmap.isNull():
                painter.drawPixmap(cell_x, cell_y, cell_pixmap)
            current_x += col_widths[col_index] + cell_padding_x
        current_y += row_heights[row_index] + cell_padding_y

    x += matrix_width

    if not right_delim_pixmap.isNull():
        painter.drawPixmap(x, center_y - right_delim_pixmap.height() // 2, right_delim_pixmap)
        x += right_delim_pixmap.width()

    if not suffix_pixmap.isNull():
        x += gap
        painter.drawPixmap(x, center_y - suffix_pixmap.height() // 2, suffix_pixmap)

    painter.end()
    return pixmap


def fit_pixmap_to_box(pixmap, max_width=None, max_height=None):
    if pixmap.isNull():
        return pixmap

    if max_width is None or max_height is None:
        return pixmap

    max_width = max(1, int(max_width))
    max_height = max(1, int(max_height))
    return pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def format_duration(seconds):
    if seconds is None:
        return "N/A"

    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(rem)}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def format_bbox(bbox):
    if not bbox or len(bbox) != 4:
        return "N/A"
    x, y, w, h = map(int, bbox)
    return f"x={x}, y={y}, w={w}, h={h}"


def proximity_fraction_to_slider_value(fraction):
    fraction = float(fraction or 0.0)
    return max(2, min(30, int(round(fraction * 200.0))))


def proximity_slider_value_to_fraction(value):
    return max(0.01, min(0.15, float(value) / 200.0))


def proximity_slider_value_to_label(value):
    display_value = float(value) / 2.0
    if abs(display_value - round(display_value)) < 1e-9:
        return str(int(round(display_value)))
    return f"{display_value:.1f}"


def build_selection_metadata_text(metadata):
    if not metadata:
        return "Select a region to inspect metadata."

    segment_interval_s = float(metadata.get("segmentation_interval_s", DEFAULT_SEGMENT_INTERVAL_S) or DEFAULT_SEGMENT_INTERVAL_S)
    processing_time_s = metadata.get("processing_time_s")
    first_seen_processing_time_s = metadata.get("first_seen_processing_time_s")
    last_ocr_processing_time_s = metadata.get("last_ocr_processing_time_s")
    age_count = int(metadata.get("age", 0) or 0)

    lines = [
        f"Region {metadata.get('region_id', '?')}",
        f"Classification: {metadata.get('classification', 'unknown')}",
        f"Status: {metadata.get('status', 'unknown')}",
        f"Bounds: {format_bbox(metadata.get('bbox'))}",
    ]

    confidence = metadata.get("confidence")
    if confidence is not None:
        lines.append(f"Classifier confidence: {format_percent(confidence)}")

    classification_source = str(metadata.get("classification_source") or "").strip()
    if classification_source and classification_source != "lightweight_ocr":
        lines.append(f"Classification source: {classification_source}")

    classification_mode = str(metadata.get("classification_mode") or "normal").strip()
    if classification_mode and classification_mode != "normal":
        lines.append(f"Mode: {classification_mode}")

    region_override = str(metadata.get("region_label_override") or "").strip()
    if region_override:
        lines.append(f"Forced label: {region_override}")

    visual_edit_override = str(metadata.get("visual_edit_override") or "").strip()
    if visual_edit_override:
        lines.append(f"Visual edit: {visual_edit_override}")

    if processing_time_s is not None and first_seen_processing_time_s is not None:
        visible_for_s = float(processing_time_s) - float(first_seen_processing_time_s)
        lines.append(f"Visible for: {format_duration(visible_for_s)}")

    if age_count > 0:
        approx_age_s = age_count * segment_interval_s
        lines.append(f"Label stable for: {format_duration(approx_age_s)} ({age_count} updates)")

    if processing_time_s is not None and last_ocr_processing_time_s is not None:
        since_last_ocr_s = float(processing_time_s) - float(last_ocr_processing_time_s)
        lines.append(f"Since last OCR: {format_duration(since_last_ocr_s)}")
    else:
        lines.append("Since last OCR: Never")

    if metadata.get("ocr_time_s") is not None:
        lines.append(f"Last region OCR runtime: {format_duration(metadata.get('ocr_time_s'))}")

    equation_conf = metadata.get("equation_ocr_confidence")
    equation_latex = str(metadata.get("equation_ocr_prediction_latex") or "").strip()
    equation_error = str(metadata.get("equation_ocr_error") or "").strip()

    if metadata.get("equation_ocr_enabled"):
        lines.append(f"Equation OCR runs: {int(metadata.get('equation_ocr_attempt_count', 0) or 0)}")
        if metadata.get("equation_ocr_latency_ms") is not None:
            lines.append(f"Equation OCR latency: {float(metadata.get('equation_ocr_latency_ms')):.0f} ms")
        if equation_conf is not None:
            lines.append(f"Equation OCR confidence: {format_percent(equation_conf)}")
        lines.append(f"Equation render ready: {'Yes' if metadata.get('equation_ocr_rendered') else 'No'}")
        if equation_latex:
            lines.append(f"LaTeX: {equation_latex}")
        if equation_error:
            lines.append(f"Equation OCR error: {equation_error}")

    text_preview = str(metadata.get("text_preview") or "").strip()
    if text_preview:
        lines.append(f"Preview: {text_preview}")

    return "\n".join(lines)


def infer_board_dimensions(width, height, items=None, original_frame=None):
    inferred_width = max(1, int(width or 0))
    inferred_height = max(1, int(height or 0))

    if original_frame is not None:
        try:
            inferred_height = max(inferred_height, int(original_frame.shape[0]))
            inferred_width = max(inferred_width, int(original_frame.shape[1]))
        except Exception:
            pass

    for item in items or []:
        entry = dict(item or {})
        item_x = float(entry.get("x", 0.0) or 0.0)
        item_y = float(entry.get("y", 0.0) or 0.0)
        metadata = dict(entry.get("metadata", {}) or {})
        bbox = metadata.get("bbox")

        item_width = 0.0
        item_height = 0.0
        if bbox and len(bbox) == 4:
            item_x = float(bbox[0])
            item_y = float(bbox[1])
            item_width = float(bbox[2])
            item_height = float(bbox[3])
        elif entry.get("type") == "image" and isinstance(entry.get("content"), np.ndarray):
            item_height = float(entry["content"].shape[0])
            item_width = float(entry["content"].shape[1])
        else:
            item_width = float(entry.get("width", 0.0) or 0.0)
            item_height = float(entry.get("height", 0.0) or 0.0)

        inferred_width = max(inferred_width, int(round(item_x + item_width)))
        inferred_height = max(inferred_height, int(round(item_y + item_height)))

    return inferred_width, inferred_height


def humanize_bytes(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def discover_video_files():
    discovered = []
    seen_paths = set()

    for directory in VIDEO_LIBRARY_DIRS:
        if not directory.exists():
            continue

        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            resolved = path.resolve()
            if resolved in seen_paths:
                continue

            seen_paths.add(resolved)
            relative = path.relative_to(REPO_ROOT)
            discovered.append({
                "path": str(resolved),
                "relative_path": str(relative),
                "name": path.name,
                "size": path.stat().st_size,
                "preferred": directory == VIDEO_LIBRARY_DIRS[0],
            })

    discovered.sort(key=lambda item: (0 if item["preferred"] else 1, item["name"].lower(), item["relative_path"].lower()))
    return discovered


def make_camera_source(camera_index=0):
    return {
        "kind": "camera",
        "value": int(camera_index),
        "name": f"Camera {int(camera_index)}",
        "relative_path": f"Live camera feed (device {int(camera_index)})",
        "preferred": True,
        "size": None,
    }


def make_file_source(video):
    return {
        "kind": "file",
        "value": str(video["path"]),
        "name": str(video["name"]),
        "relative_path": str(video["relative_path"]),
        "preferred": bool(video["preferred"]),
        "size": int(video["size"]),
    }


def make_saved_board_source(board):
    return {
        "kind": "board",
        "value": str(board["value"]),
        "name": str(board["name"]),
        "relative_path": str(board["relative_path"]),
        "preferred": bool(board.get("preferred", True)),
        "size": int(board.get("size", 0) or 0),
        "saved_at": str(board.get("saved_at") or ""),
        "width": int(board.get("width", 0) or 0),
        "height": int(board.get("height", 0) or 0),
    }


def normalize_video_source(source):
    if source is None:
        return None

    if isinstance(source, dict):
        return dict(source)

    if isinstance(source, int):
        return make_camera_source(source)

    source_path = Path(str(source))
    resolved = source_path if source_path.is_absolute() else (REPO_ROOT / source_path)
    try:
        stat = resolved.stat()
        size = int(stat.st_size)
    except OSError:
        size = 0

    try:
        relative_path = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        relative_path = str(resolved)

    return {
        "kind": "file",
        "value": str(resolved),
        "name": resolved.name,
        "relative_path": relative_path,
        "preferred": False,
        "size": size,
    }


def with_default_run_options(source):
    normalized_source = dict(source or {})
    normalized_source["text_ocr_backend"] = str(
        normalized_source.get("text_ocr_backend", DEFAULT_TEXT_OCR_BACKEND)
    ).strip().lower()
    normalized_source["text_ocr_model"] = str(
        normalized_source.get("text_ocr_model", DEFAULT_TEXT_OCR_MODEL)
    )
    normalized_source["vectorization_method"] = normalize_vectorization_method(
        normalized_source.get("vectorization_method", DEFAULT_VECTORIZATION_SELECTION)
    )
    normalized_source["equation_ocr_model"] = str(
        normalized_source.get("equation_ocr_model", DEFAULT_EQUATION_OCR_MODEL)
    )
    return normalized_source


def discover_camera_sources():
    discovered = []
    for camera_index in range(MAX_CAMERA_DISCOVERY_INDEX + 1):
        capture = open_camera_capture(camera_index)
        try:
            if not capture.isOpened():
                continue
            ok, _frame = capture.read()
            if not ok:
                continue
            discovered.append(make_camera_source(camera_index))
        finally:
            capture.release()
    return discovered


def discover_source_options():
    sources = discover_camera_sources()
    sources.extend(make_file_source(video) for video in discover_video_files())
    sources.extend(make_saved_board_source(board) for board in discover_saved_boards())
    return sources


def source_capture_value(source):
    return int(source["value"]) if str(source.get("kind")) == "camera" else str(source["value"])


def source_display_name(source):
    source_kind = str(source.get("kind"))
    if source_kind == "camera":
        return f"Camera {int(source['value'])}"
    if source_kind == "board":
        return str(source.get("name") or "Saved Board")
    return str(source.get("name") or Path(str(source.get("value"))).name)


class VideoSelectionDialog(QDialog):
    def __init__(self, source_options, parent=None):
        super().__init__(parent)
        self.source_options = source_options
        self.selected_source = None
        self.video_sources = [dict(source) for source in self.source_options if source.get("kind") == "file"]
        self.board_sources = [dict(source) for source in self.source_options if source.get("kind") == "board"]
        self.camera_sources = [dict(source) for source in self.source_options if source.get("kind") == "camera"]
        self.current_category = None

        self.setWindowTitle("Open ChalkStream")
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 18, 20, 18)
        root_layout.setSpacing(14)

        self.header_label = QLabel("Choose How To Start")
        header_font = QFont(self.header_label.font())
        header_font.setPointSize(header_font.pointSize() + 6)
        header_font.setBold(True)
        self.header_label.setFont(header_font)
        root_layout.addWidget(self.header_label)

        self.subtitle_label = QLabel(
            "Start from a board recording, open a live session from your camera, or revisit a saved board."
        )
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("color: #4a5568;")
        root_layout.addWidget(self.subtitle_label)

        self.options_group = QGroupBox("Run Options")
        options_layout = QFormLayout(self.options_group)
        self.text_ocr_backend_combo = QComboBox()
        for option_key, label in TEXT_OCR_BACKEND_OPTIONS:
            self.text_ocr_backend_combo.addItem(label, option_key)
        text_backend_index = self.text_ocr_backend_combo.findData(DEFAULT_TEXT_OCR_BACKEND)
        if text_backend_index >= 0:
            self.text_ocr_backend_combo.setCurrentIndex(text_backend_index)
        self.text_ocr_backend_combo.currentIndexChanged.connect(self._refresh_selected_source_details)
        self.text_ocr_backend_combo.currentIndexChanged.connect(self._refresh_text_ocr_option_state)
        options_layout.addRow("Text OCR backend", self.text_ocr_backend_combo)

        self.text_ocr_model_combo = QComboBox()
        for option_key, label in TEXT_OCR_MODEL_OPTIONS:
            self.text_ocr_model_combo.addItem(label, option_key)
        text_model_index = self.text_ocr_model_combo.findData(DEFAULT_TEXT_OCR_MODEL)
        if text_model_index >= 0:
            self.text_ocr_model_combo.setCurrentIndex(text_model_index)
        self.text_ocr_model_combo.currentIndexChanged.connect(self._refresh_selected_source_details)
        options_layout.addRow("Text OCR model", self.text_ocr_model_combo)

        self.vectorization_method_combo = QComboBox()
        for option_key, label in VECTORIZATION_METHOD_OPTIONS:
            self.vectorization_method_combo.addItem(label, option_key)
        vectorization_index = self.vectorization_method_combo.findData(DEFAULT_VECTORIZATION_SELECTION)
        if vectorization_index >= 0:
            self.vectorization_method_combo.setCurrentIndex(vectorization_index)
        self.vectorization_method_combo.currentIndexChanged.connect(self._refresh_selected_source_details)
        options_layout.addRow("Diagram vectorization", self.vectorization_method_combo)

        self.equation_model_combo = QComboBox()
        for option_key, label in EQUATION_OCR_MODEL_OPTIONS:
            self.equation_model_combo.addItem(label, option_key)
        equation_model_index = self.equation_model_combo.findData(DEFAULT_EQUATION_OCR_MODEL)
        if equation_model_index >= 0:
            self.equation_model_combo.setCurrentIndex(equation_model_index)
        self.equation_model_combo.currentIndexChanged.connect(self._refresh_selected_source_details)
        options_layout.addRow("Equation OCR model", self.equation_model_combo)
        root_layout.addWidget(self.options_group)

        self.stacked_widget = QStackedWidget()
        root_layout.addWidget(self.stacked_widget, 1)

        self.home_page = QWidget()
        self.browser_page = QWidget()
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.browser_page)

        self._setup_home_page()
        self._setup_browser_page()
        self._show_home_page()
        self._refresh_text_ocr_option_state()

    def _current_run_options(self):
        return {
            "text_ocr_backend": str(self.text_ocr_backend_combo.currentData() or DEFAULT_TEXT_OCR_BACKEND).strip().lower(),
            "text_ocr_model": str(self.text_ocr_model_combo.currentData() or DEFAULT_TEXT_OCR_MODEL),
            "vectorization_method": normalize_vectorization_method(self.vectorization_method_combo.currentData()),
            "equation_ocr_model": str(self.equation_model_combo.currentData() or DEFAULT_EQUATION_OCR_MODEL),
        }

    def _apply_run_options_to_source(self, source):
        source_with_options = dict(source or {})
        source_with_options.update(self._current_run_options())
        return source_with_options

    def _refresh_selected_source_details(self):
        current_item = self.browser_list.currentItem() if hasattr(self, "browser_list") else None
        if current_item is not None:
            self.update_details(current_item, None)

    def _refresh_text_ocr_option_state(self):
        backend = str(self.text_ocr_backend_combo.currentData() or DEFAULT_TEXT_OCR_BACKEND).strip().lower()
        self.text_ocr_model_combo.setEnabled(backend == "trocr_endpoint")

    def _create_launch_button(self, title, description):
        button = QPushButton(f"{title}\n{description}")
        button.setMinimumHeight(120)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(
            """
            QPushButton {
                text-align: left;
                padding: 18px;
                border: 1px solid #cfd6df;
                border-radius: 12px;
                background: #f6f7fb;
                color: #1f2933;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #edf3ff;
                border-color: #5b8def;
            }
            QPushButton:disabled {
                color: #8b95a7;
                background: #f1f2f5;
                border-color: #d7dce3;
            }
            """
        )
        return button

    def _setup_home_page(self):
        layout = QGridLayout(self.home_page)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(14)

        self.open_video_button = self._create_launch_button(
            "Select Video",
            "Open an MP4, MOV, AVI, MKV, or WebM recording from the project video library.",
        )
        self.open_video_button.clicked.connect(lambda: self._show_source_browser("video"))
        self.open_video_button.setEnabled(bool(self.video_sources))
        layout.addWidget(self.open_video_button, 0, 0)

        self.start_recording_button = self._create_launch_button(
            "Start a Recording",
            "Use your camera for a live board session and freeze a frame to choose the board corners.",
        )
        self.start_recording_button.clicked.connect(self._accept_camera_source)
        self.start_recording_button.setEnabled(bool(self.camera_sources))
        layout.addWidget(self.start_recording_button, 0, 1)

        self.open_board_button = self._create_launch_button(
            "Select a Board",
            "Open a saved board directly in canvas editor mode without replaying a video.",
        )
        self.open_board_button.clicked.connect(lambda: self._show_source_browser("board"))
        self.open_board_button.setEnabled(bool(self.board_sources))
        layout.addWidget(self.open_board_button, 1, 0, 1, 2)

        if not self.video_sources:
            self.open_video_button.setToolTip("No supported video files were found in pipeline/videos or pipeline.")
        if not self.board_sources:
            self.open_board_button.setToolTip("No saved boards were found yet.")

    def _setup_browser_page(self):
        layout = QVBoxLayout(self.browser_page)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)

        self.browser_title_label = QLabel("Select Source")
        browser_title_font = QFont(self.browser_title_label.font())
        browser_title_font.setPointSize(browser_title_font.pointSize() + 2)
        browser_title_font.setBold(True)
        self.browser_title_label.setFont(browser_title_font)
        layout.addWidget(self.browser_title_label)

        self.browser_list = QListWidget()
        self.browser_list.currentItemChanged.connect(self.update_details)
        self.browser_list.itemDoubleClicked.connect(lambda _item: self.accept_selection())
        layout.addWidget(self.browser_list, 1)

        self.details_label = QLabel("Select an item to inspect its details.")
        self.details_label.setWordWrap(True)
        self.details_label.setMinimumHeight(72)
        self.details_label.setStyleSheet("padding: 8px; background: #f7f8fa; border: 1px solid #dde3ea; border-radius: 8px;")
        layout.addWidget(self.details_label)

        button_row = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._show_home_page)
        button_row.addWidget(self.back_button)
        button_row.addStretch()

        self.start_button = QPushButton("Open Selected")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.accept_selection)
        button_row.addWidget(self.start_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

    def _show_home_page(self):
        self.current_category = None
        self.selected_source = None
        self.header_label.setText("Choose How To Start")
        self.subtitle_label.setText(
            "Start from a board recording, open a live session from your camera, or revisit a saved board."
        )
        self.stacked_widget.setCurrentWidget(self.home_page)

    def _show_source_browser(self, category):
        if category not in {"video", "board", "camera"}:
            return

        self.current_category = category
        self.selected_source = None
        self.browser_list.clear()
        self.start_button.setEnabled(False)

        if category == "video":
            self.header_label.setText("Select Video")
            self.subtitle_label.setText("Choose a board recording to replay through the pipeline.")
            self.browser_title_label.setText("Available Videos")
            self.start_button.setText("Open Video")
            sources = self.video_sources
        elif category == "camera":
            self.header_label.setText("Select Camera")
            self.subtitle_label.setText("Choose which live camera device to use for the board session.")
            self.browser_title_label.setText("Available Cameras")
            self.start_button.setText("Start Recording")
            sources = self.camera_sources
        else:
            self.header_label.setText("Select a Board")
            self.subtitle_label.setText("Choose a saved board document to reopen in canvas editor mode.")
            self.browser_title_label.setText("Saved Boards")
            self.start_button.setText("Open Board")
            sources = self.board_sources

        for source in sources:
            if source["kind"] == "board":
                dimensions = ""
                if int(source.get("width", 0) or 0) > 0 and int(source.get("height", 0) or 0) > 0:
                    dimensions = f"  •  {int(source['width'])}x{int(source['height'])}"
                saved_at = f"  •  {source['saved_at']}" if source.get("saved_at") else ""
                label = f"{source['name']}{dimensions}{saved_at}"
            elif source["kind"] == "camera":
                label = f"{source['name']}  -  {source['relative_path']}"
            else:
                badge = "Preferred Library" if source["preferred"] else "Legacy Folder"
                label = f"{source['name']}  •  {humanize_bytes(source['size'])}  •  {badge}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, source)
            item.setToolTip(source["relative_path"])
            self.browser_list.addItem(item)

        if self.browser_list.count() > 0:
            self.browser_list.setCurrentRow(0)
        else:
            if category == "video":
                empty_message = "No videos found."
            elif category == "camera":
                empty_message = "No cameras found."
            else:
                empty_message = "No saved boards found."
            self.details_label.setText(empty_message)

        self.stacked_widget.setCurrentWidget(self.browser_page)

    def _accept_camera_source(self):
        if not self.camera_sources:
            QMessageBox.information(self, "Camera Unavailable", "No camera source is configured.")
            return
        if len(self.camera_sources) == 1:
            self.selected_source = self._apply_run_options_to_source(self.camera_sources[0])
            self.accept()
            return
        self._show_source_browser("camera")

    def update_details(self, current, _previous):
        source = current.data(Qt.UserRole) if current is not None else None
        self.start_button.setEnabled(source is not None)
        if source is None:
            self.details_label.setText("Select an item to inspect its details.")
            self.selected_source = None
            return

        self.selected_source = self._apply_run_options_to_source(source)
        options_text = (
            f"Text OCR backend: {self.text_ocr_backend_combo.currentText()}\n"
            f"Text OCR model: {self.text_ocr_model_combo.currentText()}\n"
            f"Diagram vectorization: {self.vectorization_method_combo.currentText()}\n"
            f"Equation OCR model: {self.equation_model_combo.currentText()}\n"
        )
        if source["kind"] == "board":
            dimensions = ""
            if int(source.get("width", 0) or 0) > 0 and int(source.get("height", 0) or 0) > 0:
                dimensions = f"{int(source['width'])} x {int(source['height'])}"
            self.details_label.setText(
                f"Path: {source['relative_path']}\n"
                f"Saved: {source.get('saved_at') or 'Unknown'}\n"
                f"Canvas Size: {dimensions or 'Unknown'}\n"
                f"{options_text}"
                "This opens directly into the canvas editor."
            )
        elif source["kind"] == "camera":
            self.details_label.setText(
                f"Device: Camera {int(source['value'])}\n"
                f"Path: {source['relative_path']}\n"
                f"{options_text}"
                "This starts a live board session and lets you freeze a frame for corner selection."
            )
        else:
            self.details_label.setText(
                f"Path: {source['relative_path']}\n"
                f"Size: {humanize_bytes(source['size'])}\n"
                f"Source: {'pipeline/videos' if source['preferred'] else 'pipeline'}\n"
                f"{options_text}"
            )

    def accept_selection(self):
        if self.selected_source is None:
            return
        self.selected_source = self._apply_run_options_to_source(self.selected_source)
        self.accept()


class VideoFrameView(QLabel):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.base_pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #101010; color: #d8d8d8; border: 1px solid #444;")
        self.setText(f"{self.title}\nWaiting for frames...")

    def set_frame(self, frame):
        self.base_pixmap = cv2_to_qpixmap(frame)
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self.base_pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(f"{self.title}\nWaiting for frames...")
            return

        scaled = self.base_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


class DiagramEditorCanvas(QLabel):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self._image = cv2_to_qpixmap(image).toImage().convertToFormat(QImage.Format_ARGB32)
        self._base_image = self._image.copy()
        self._draw_mode = "brush"
        self._pen_width = 3
        self._stroke_color = QColor(20, 20, 20)
        self._last_image_pos = None
        self._target_rect = QRectF()

        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #f3f3f3; border: 1px solid #999;")

    def set_draw_mode(self, mode):
        normalized = str(mode or "brush").strip().lower()
        self._draw_mode = "eraser" if normalized == "eraser" else "brush"

    def set_pen_width(self, width):
        self._pen_width = max(1, int(width))

    def revert(self):
        self._image = self._base_image.copy()
        self.update()

    def current_cv_image(self):
        return qpixmap_to_cv2(QPixmap.fromImage(self._image))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def _map_widget_to_image(self, pos):
        if self._image.isNull() or self._target_rect.isNull() or not self._target_rect.contains(QPointF(pos)):
            return None

        x = (float(pos.x()) - self._target_rect.x()) / max(self._target_rect.width(), 1.0)
        y = (float(pos.y()) - self._target_rect.y()) / max(self._target_rect.height(), 1.0)
        image_x = int(round(x * (self._image.width() - 1)))
        image_y = int(round(y * (self._image.height() - 1)))
        return QPointF(image_x, image_y)

    def _draw_line(self, start_point, end_point):
        if start_point is None or end_point is None:
            return

        painter = QPainter(self._image)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(255, 255, 255) if self._draw_mode == "eraser" else self._stroke_color
        pen = QPen(color, float(self._pen_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(start_point, end_point)
        painter.end()
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        self._last_image_pos = self._map_widget_to_image(event.position().toPoint())
        if self._last_image_pos is not None:
            self._draw_line(self._last_image_pos, self._last_image_pos)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return super().mouseMoveEvent(event)

        current_pos = self._map_widget_to_image(event.position().toPoint())
        if current_pos is None or self._last_image_pos is None:
            return super().mouseMoveEvent(event)

        self._draw_line(self._last_image_pos, current_pos)
        self._last_image_pos = current_pos
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_image_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(243, 243, 243))

        pixmap = QPixmap.fromImage(self._image)
        if pixmap.isNull():
            return

        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) / 2.0
        y = (self.height() - scaled.height()) / 2.0
        self._target_rect = QRectF(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(int(round(x)), int(round(y)), scaled)
        painter.end()


class DiagramEditorDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Diagram")
        self.resize(820, 620)

        layout = QVBoxLayout(self)

        self.editor_canvas = DiagramEditorCanvas(image, self)
        layout.addWidget(self.editor_canvas, 1)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        controls.addWidget(QLabel("Tool"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("Brush", "brush")
        self.tool_combo.addItem("Eraser", "eraser")
        self.tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        controls.addWidget(self.tool_combo)

        controls.addWidget(QLabel("Size"))
        self.pen_width_spin = QSpinBox()
        self.pen_width_spin.setRange(1, 24)
        self.pen_width_spin.setValue(3)
        self.pen_width_spin.valueChanged.connect(self.editor_canvas.set_pen_width)
        controls.addWidget(self.pen_width_spin)

        self.revert_button = QPushButton("Revert")
        self.revert_button.clicked.connect(self.editor_canvas.revert)
        controls.addWidget(self.revert_button)
        controls.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        controls.addWidget(self.cancel_button)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.accept)
        controls.addWidget(self.apply_button)

    def _on_tool_changed(self, _index):
        self.editor_canvas.set_draw_mode(self.tool_combo.currentData())

    def edited_image(self):
        return self.editor_canvas.current_cv_image()


class ExportCropCanvas(QLabel):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self._image = image.copy()
        self._target_rect = QRectF()
        self._selection_rect = QRectF(0, 0, float(self._image.width()), float(self._image.height()))
        self._drag_start = None

        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #999;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def reset_selection(self):
        self._selection_rect = QRectF(0, 0, float(self._image.width()), float(self._image.height()))
        self.update()

    def _map_widget_to_image(self, pos):
        if self._image.isNull() or self._target_rect.isNull() or not self._target_rect.contains(QPointF(pos)):
            return None

        x = (float(pos.x()) - self._target_rect.x()) / max(self._target_rect.width(), 1.0)
        y = (float(pos.y()) - self._target_rect.y()) / max(self._target_rect.height(), 1.0)
        image_x = max(0.0, min(float(self._image.width() - 1), x * (self._image.width() - 1)))
        image_y = max(0.0, min(float(self._image.height() - 1), y * (self._image.height() - 1)))
        return QPointF(image_x, image_y)

    def selected_rect(self):
        rect = self._selection_rect.normalized()
        left = int(round(max(0.0, min(float(self._image.width() - 1), rect.left()))))
        top = int(round(max(0.0, min(float(self._image.height() - 1), rect.top()))))
        right = int(round(max(left + 1.0, min(float(self._image.width()), rect.right()))))
        bottom = int(round(max(top + 1.0, min(float(self._image.height()), rect.bottom()))))
        return QRectF(float(left), float(top), float(max(1, right - left)), float(max(1, bottom - top)))

    def cropped_image(self):
        rect = self.selected_rect()
        return self._image.copy(
            int(round(rect.x())),
            int(round(rect.y())),
            int(round(rect.width())),
            int(round(rect.height())),
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        start = self._map_widget_to_image(event.position().toPoint())
        if start is None:
            return super().mousePressEvent(event)

        self._drag_start = start
        self._selection_rect = QRectF(start, start)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or self._drag_start is None:
            return super().mouseMoveEvent(event)

        current = self._map_widget_to_image(event.position().toPoint())
        if current is None:
            return super().mouseMoveEvent(event)

        self._selection_rect = QRectF(self._drag_start, current).normalized()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            current = self._map_widget_to_image(event.position().toPoint()) or self._drag_start
            self._selection_rect = QRectF(self._drag_start, current).normalized()
            self._drag_start = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        pixmap = QPixmap.fromImage(self._image)
        if pixmap.isNull():
            painter.end()
            return

        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) / 2.0
        y = (self.height() - scaled.height()) / 2.0
        self._target_rect = QRectF(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(int(round(x)), int(round(y)), scaled)

        selection = self.selected_rect()
        display_selection = QRectF(
            self._target_rect.x() + (selection.x() / max(1.0, float(self._image.width()))) * self._target_rect.width(),
            self._target_rect.y() + (selection.y() / max(1.0, float(self._image.height()))) * self._target_rect.height(),
            (selection.width() / max(1.0, float(self._image.width()))) * self._target_rect.width(),
            (selection.height() / max(1.0, float(self._image.height()))) * self._target_rect.height(),
        )

        painter.setPen(QPen(QColor(20, 120, 215), 2.0, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(display_selection)
        painter.fillRect(display_selection.adjusted(1, 1, -1, -1), QColor(20, 120, 215, 18))
        painter.end()


class ExportCropDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Export Area")
        self.resize(960, 760)

        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Drag across the board preview to define the export edges. "
            "The PNG will include only the selected area."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.crop_canvas = ExportCropCanvas(image, self)
        layout.addWidget(self.crop_canvas, 1)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self.reset_button = QPushButton("Use Full Board")
        self.reset_button.clicked.connect(self.crop_canvas.reset_selection)
        button_row.addWidget(self.reset_button)
        button_row.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)

        self.apply_button = QPushButton("Continue")
        self.apply_button.clicked.connect(self.accept)
        button_row.addWidget(self.apply_button)

    def cropped_image(self):
        return self.crop_canvas.cropped_image()


class ResizableGraphicsItemMixin:
    handle_size_px = 16.0
    min_scale_factor = 0.25

    def init_resize_behavior(self):
        self._resize_active = False
        self._resize_base_rect = QRectF()
        self._resize_original_scale = 1.0
        self._resize_anchor_scene_pos = QPointF()
        self._interaction_start_pos = QPointF()
        self._interaction_start_scale = 1.0
        self.setAcceptHoverEvents(True)
        self.setTransformOriginPoint(QPointF(0.0, 0.0))

    def _owning_canvas_view(self):
        scene = self.scene()
        if scene is None:
            return None
        views = scene.views()
        return views[0] if views else None

    def _begin_transform_tracking(self):
        self._interaction_start_pos = QPointF(self.pos())
        self._interaction_start_scale = float(self.scale())
        view = self._owning_canvas_view()
        if view is not None and hasattr(view, "set_item_interaction_active"):
            view.set_item_interaction_active(self.item_id, True)

    def _finish_transform_tracking(self):
        start_pos = QPointF(getattr(self, "_interaction_start_pos", self.pos()))
        start_scale = float(getattr(self, "_interaction_start_scale", self.scale()))
        end_pos = QPointF(self.pos())
        end_scale = float(self.scale())
        if (
            abs(start_pos.x() - end_pos.x()) < 1e-6
            and abs(start_pos.y() - end_pos.y()) < 1e-6
            and abs(start_scale - end_scale) < 1e-6
        ):
            return

        view = self._owning_canvas_view()
        if view is not None and hasattr(view, "record_item_transform"):
            view.record_item_transform(
                self.item_id,
                (start_pos.x(), start_pos.y()),
                start_scale,
                (end_pos.x(), end_pos.y()),
                end_scale,
            )

    def _end_transform_tracking(self):
        view = self._owning_canvas_view()
        if view is not None and hasattr(view, "set_item_interaction_active"):
            view.set_item_interaction_active(self.item_id, False)

    def _current_handle_size(self):
        return self.handle_size_px / max(self.scale(), 0.001)

    def _resize_handle_rect(self):
        rect = self.boundingRect()
        size = self._current_handle_size()
        return QRectF(rect.right() - size, rect.bottom() - size, size, size)

    def _draw_resize_overlay(self, painter):
        if not self.isSelected():
            return

        rect = self.boundingRect()
        pen_width = max(1.0, 1.0 / max(self.scale(), 0.001))
        painter.save()
        painter.setPen(QPen(QColor(28, 110, 200), pen_width, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.setPen(QPen(QColor(18, 82, 148), pen_width))
        painter.setBrush(QBrush(QColor(220, 238, 255)))
        painter.drawRect(self._resize_handle_rect())
        painter.restore()

    def _is_resize_hit(self, pos):
        return self.isSelected() and self._resize_handle_rect().contains(pos)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._begin_transform_tracking()
        if event.button() == Qt.LeftButton and self._is_resize_hit(event.pos()):
            self._resize_active = True
            self._resize_base_rect = self.boundingRect()
            self._resize_original_scale = self.scale()
            self._resize_anchor_scene_pos = self.mapToScene(self._resize_base_rect.topLeft())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, "_resize_active", False):
            base_rect = self._resize_base_rect
            if base_rect.width() <= 0 or base_rect.height() <= 0:
                return

            anchor_scene_pos = getattr(self, "_resize_anchor_scene_pos", QPointF())
            scene_pos = event.scenePos()
            scale_x = (scene_pos.x() - anchor_scene_pos.x()) / max(base_rect.width(), 1e-6)
            scale_y = (scene_pos.y() - anchor_scene_pos.y()) / max(base_rect.height(), 1e-6)
            new_scale = max(self.min_scale_factor, scale_x, scale_y)
            self.setScale(new_scale)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if getattr(self, "_resize_active", False) and event.button() == Qt.LeftButton:
            self._resize_active = False
            self._finish_transform_tracking()
            self._end_transform_tracking()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self._finish_transform_tracking()
            self._end_transform_tracking()

    def hoverMoveEvent(self, event):
        if self._is_resize_hit(event.pos()):
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.unsetCursor()
        super().hoverLeaveEvent(event)


class ImageItem(ResizableGraphicsItemMixin, QGraphicsPixmapItem):

    def __init__(self, item_id, image, scene):
        pixmap = cv2_to_qpixmap(image)
        super().__init__(pixmap)

        self.scene_ref = scene
        self.item_id = item_id
        self.item_type = "image"
        self.region_metadata = {}

        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable |
            QGraphicsPixmapItem.ItemSendsGeometryChanges
        )
        self.init_resize_behavior()

    
    def update_content(self, new_pixmap):
        new_pixmap = cv2_to_qpixmap(new_pixmap)
        old_pos = self.pos()
        self.setPixmap(new_pixmap)
        self.setPos(old_pos)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        self._draw_resize_overlay(painter)


class TextItem(ResizableGraphicsItemMixin, QGraphicsTextItem):
    def __init__(self, item_id, text, size=16, max_width=None, max_height=None) -> None:
        super().__init__(text)
        self.item_id = item_id
        self.item_type = "text"
        self.render_size = size
        self.target_width = None
        self.target_height = None
        self.region_metadata = {}

        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable 
        )
        self.setDefaultTextColor(QColor(0, 0, 0))
        self._fit_text_to_box(text, size=size, max_width=max_width, max_height=max_height)
        self.init_resize_behavior()

    def _apply_font_pixel_size(self, pixel_size):
        self.render_size = max(8, int(pixel_size))
        font = self.font()
        font.setPixelSize(self.render_size)
        self.setFont(font)

    def _set_text_width(self, max_width):
        self.target_width = None if max_width is None else float(max_width)
        if self.target_width is None or self.target_width <= 0:
            self.setTextWidth(-1)
            return

        # Let short labels keep a natural width so the selection box hugs the
        # text, and only expand to the full region width when wrapping is needed.
        self.setTextWidth(-1)
        ideal_width = float(self.document().idealWidth())
        if ideal_width <= 0:
            self.setTextWidth(self.target_width)
        else:
            self.setTextWidth(min(self.target_width, ideal_width))

    def _fit_text_to_box(self, text, size=None, max_width=None, max_height=None):
        self.target_width = None if max_width is None else float(max_width)
        self.target_height = None if max_height is None else float(max_height)
        self.setPlainText(str(text or ""))

        base_size = max(8, int(size or self.render_size or 16))
        if self.target_width is None or self.target_width <= 0 or self.target_height is None or self.target_height <= 0:
            self._apply_font_pixel_size(base_size)
            self._set_text_width(max_width)
            return

        upper_bound = max(base_size, int(self.target_height))
        lower_bound = 8
        best_size = lower_bound

        for candidate_size in range(upper_bound, lower_bound - 1, -1):
            self._apply_font_pixel_size(candidate_size)
            self._set_text_width(max_width)
            doc_size = self.document().size()
            if doc_size.width() <= (self.target_width + 0.5) and doc_size.height() <= (self.target_height + 0.5):
                best_size = candidate_size
                break

        self._apply_font_pixel_size(best_size)
        self._set_text_width(max_width)

    def update_content(self, new_text, size=None, max_width=None, max_height=None):

        old_pos = self.pos()
        self._fit_text_to_box(
            new_text,
            size=size,
            max_width=max_width if max_width is not None else self.target_width,
            max_height=max_height if max_height is not None else self.target_height,
        )
        self.setPos(old_pos)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        self._draw_resize_overlay(painter)


class LatexItem(ResizableGraphicsItemMixin, QGraphicsPixmapItem):
    def __init__(self, item_id, latex, size = 1, max_width=None, max_height=None, parent=None):
        self.target_width = max_width
        self.target_height = max_height
        pixmap = self.render_latex(latex, size)
        super().__init__(pixmap, parent)
        self.item_id = item_id
        self.latex_value = latex
        self.item_type = "latex"
        self.render_size = size
        self.region_metadata = {}

        # make it behave like QGraphicsTextItem
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable 
        )
        self.init_resize_behavior()

    def render_latex(self, latex,size = 16):
        latex = str(latex or "").strip()
        try:
            pixmap = render_mathtext_pixmap(latex, size)
        except Exception:
            matrix_pixmap = render_matrix_expression_pixmap(latex, size)
            if matrix_pixmap is not None:
                pixmap = matrix_pixmap
            else:
                pixmap = render_plain_text_pixmap(latex, pixel_size=max(18, int(size) * 2))
        return fit_pixmap_to_box(pixmap, self.target_width, self.target_height)

    def update_content(self, latex, size=None, max_width=None, max_height=None):
        old_pos = self.pos()
        self.latex_value = latex
        if size is not None:
            self.render_size = size
        if max_width is not None:
            self.target_width = max_width
        if max_height is not None:
            self.target_height = max_height
        new_pixmap = self.render_latex(latex, self.render_size)
        self.setPixmap(new_pixmap)
        self.setPos(old_pos)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        self._draw_resize_overlay(painter)

class CanvasView(QGraphicsView):

    def __init__(self,dimension_x,dimension_y, interaction_state=None):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.setSceneRect(0, 0, dimension_x, dimension_y)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.contents = {}
        self.paused = False
        self.queued_elements = []
        self.removed_elements_id = []
        self.next_manual_id = -1
        self.interaction_state = interaction_state or InteractionStateStore()
        self.undo_stack = []
        self._history_suspended = False
        self._active_interaction_item_ids = set()
        self._deferred_item_events = {}
        self.base_payloads = {}

        # The board scene is defined in board coordinates, which is often much
        # larger than the window. Keep the whole board fitted to the available
        # viewport so the user is not forced to pan left and right just to see it.
        self._fit_scene_to_viewport()

    def _fit_scene_to_viewport(self):
        scene_rect = self.scene.sceneRect()
        viewport_rect = self.viewport().rect()
        if scene_rect.isEmpty() or viewport_rect.isEmpty():
            return
        self.fitInView(scene_rect, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_scene_to_viewport()

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_scene_to_viewport()

    def allocate_manual_id(self):
        manual_id = self.next_manual_id
        self.next_manual_id -= 1
        return manual_id

    def _get_content_override(self, item_id):
        if self.interaction_state is None:
            return None
        return self.interaction_state.get_region_content_override(int(item_id))

    def _set_content_override_state(self, item_id, override):
        if self.interaction_state is None:
            return None
        if override is None:
            return self.interaction_state.remove_region_content_override(int(item_id))
        return self.interaction_state.set_region_content_override(int(item_id), override)

    def _payload_with_content_override(self, data):
        payload = clone_canvas_payload(data)
        item_id = int(payload["id"])
        override = self._get_content_override(item_id)
        if not override:
            return payload

        override_type = str(override.get("type") or "").strip().lower()
        metadata = dict(payload.get("metadata", {}) or {})
        metadata["visual_edit_override"] = override_type
        payload["metadata"] = metadata

        if override_type in {"text", "latex"}:
            payload["type"] = override_type
            payload["content"] = str(override.get("content") or "")
            if override.get("size") is not None:
                payload["size"] = int(override["size"])
            if override_type == "latex":
                if override.get("width") is not None:
                    payload["width"] = int(override["width"])
                if override.get("height") is not None:
                    payload["height"] = int(override["height"])
            return payload

        if override_type == "image":
            image = decode_cv2_image_base64(override.get("content"))
            payload["type"] = "image"
            payload["content"] = image
            return payload

        return payload

    def _refresh_item_from_base_payload(self, item_id, select=False):
        base_payload = self.base_payloads.get(int(item_id))
        if base_payload is None:
            item = self.contents.get(int(item_id))
            if item is None:
                return False
            base_payload = self._snapshot_item(item)
            self.base_payloads[int(item_id)] = clone_canvas_payload(base_payload)

        payload = self._payload_with_content_override(base_payload)
        was_paused = self.paused
        self.paused = False
        try:
            self._load_images_internal(payload, update_base_payload=False)
        finally:
            self.paused = was_paused

        refreshed_item = self.contents.get(int(item_id))
        if refreshed_item is not None and select:
            refreshed_item.setSelected(True)
        return refreshed_item is not None

    def set_item_content_override(self, item_id, override, select_after=True):
        item_id = int(item_id)
        current_override = self._get_content_override(item_id)
        new_override = dict(override or {}) if override is not None else None
        if current_override == new_override:
            return False

        self._set_content_override_state(item_id, new_override)
        self._push_undo_action({
            "type": "content_override",
            "item_id": item_id,
            "old_override": current_override,
            "new_override": new_override,
        })
        return self._refresh_item_from_base_payload(item_id, select=select_after)

    def _restore_content_override(self, item_id, override, select=False):
        self._set_content_override_state(item_id, override)
        return self._refresh_item_from_base_payload(item_id, select=select)

    def _push_undo_action(self, action):
        if self._history_suspended:
            return
        self.undo_stack.append(action)

    def record_item_transform(self, item_id, old_pos, old_scale, new_pos, new_scale):
        if self._history_suspended:
            return
        if item_id not in self.contents:
            return
        if (
            abs(float(old_pos[0]) - float(new_pos[0])) < 1e-6
            and abs(float(old_pos[1]) - float(new_pos[1])) < 1e-6
            and abs(float(old_scale) - float(new_scale)) < 1e-6
        ):
            return

        self._push_undo_action({
            "type": "transform",
            "item_id": int(item_id),
            "old_pos": (float(old_pos[0]), float(old_pos[1])),
            "old_scale": float(old_scale),
            "new_pos": (float(new_pos[0]), float(new_pos[1])),
            "new_scale": float(new_scale),
        })
        if self.interaction_state is not None:
            self.interaction_state.set_region_transform(
                item_id,
                float(new_pos[0]),
                float(new_pos[1]),
                float(new_scale),
            )

    def _apply_item_transform(self, item_id, pos, scale):
        item = self.contents.get(item_id)
        if item is None:
            return False

        item.setPos(float(pos[0]), float(pos[1]))
        item.setScale(float(scale))
        if self.interaction_state is not None:
            self.interaction_state.set_region_transform(
                item_id,
                float(pos[0]),
                float(pos[1]),
                float(scale),
            )
        return True

    def set_item_interaction_active(self, item_id, active):
        item_id = int(item_id)
        if active:
            self._active_interaction_item_ids.add(item_id)
            return

        self._active_interaction_item_ids.discard(item_id)
        deferred_event = self._deferred_item_events.pop(item_id, None)
        if deferred_event is None:
            return

        event_type, payload = deferred_event
        if event_type == "load":
            self.load_images(payload)
            return

        if event_type == "discard":
            remember_removal = bool(payload.get("remember", False))
            if self.paused:
                self.queued_elements.append({
                    "_canvas_cmd": "remove",
                    "id": item_id,
                    "remember": remember_removal,
                })
            else:
                self._discard_item(item_id, remember_removal=remember_removal)
            return

    def _snapshot_item(self, item):
        snapshot = {
            "id": int(item.item_id),
            "type": str(getattr(item, "item_type", "image")),
            "x": float(item.pos().x()),
            "y": float(item.pos().y()),
            "scale": float(item.scale()),
            "metadata": dict(getattr(item, "region_metadata", {}) or {}),
        }

        if snapshot["type"] == "image":
            snapshot["content"] = qpixmap_to_cv2(item.pixmap())
        elif snapshot["type"] == "text":
            snapshot["content"] = item.toPlainText()
            snapshot["size"] = int(getattr(item, "render_size", 16) or 16)
            snapshot["width"] = getattr(item, "target_width", None)
            snapshot["height"] = getattr(item, "target_height", None)
        elif snapshot["type"] == "latex":
            snapshot["content"] = str(getattr(item, "latex_value", ""))
            snapshot["size"] = int(getattr(item, "render_size", 16) or 16)
            snapshot["width"] = getattr(item, "target_width", None)
            snapshot["height"] = getattr(item, "target_height", None)
        else:
            raise ValueError(f"Unsupported item snapshot type: {snapshot['type']}")

        return snapshot

    def _restore_item_snapshot(self, snapshot, select=False):
        item_id = int(snapshot["id"])
        if item_id in self.contents:
            self._discard_item(item_id, remember_removal=False)

        self.load_images(dict(snapshot))
        item = self.contents.get(item_id)
        if item is None:
            return None

        item.setScale(float(snapshot.get("scale", 1.0) or 1.0))
        if select:
            item.setSelected(True)
        return item

    def snapshot_all_items(self):
        return [
            self._snapshot_item(item)
            for item in sorted(self.contents.values(), key=lambda current: int(getattr(current, "item_id", 0)))
        ]

    def clear_all_content(self):
        for item_id in list(self.contents.keys()):
            self._discard_item(item_id, remember_removal=False)
        self.queued_elements = []
        self.removed_elements_id = []
        self.undo_stack = []
        self.next_manual_id = -1
        self.base_payloads = {}

    def load_board_snapshots(self, snapshots, clear_existing=True):
        if clear_existing:
            self.clear_all_content()

        for snapshot in snapshots or []:
            self.load_images(dict(snapshot))

        negative_ids = [
            int(snapshot.get("id"))
            for snapshot in (snapshots or [])
            if int(snapshot.get("id", 0)) < 0
        ]
        if negative_ids:
            self.next_manual_id = min(negative_ids) - 1

    def _extract_source_region_ids(self, item):
        if item is None:
            return []

        metadata = dict(getattr(item, "region_metadata", {}) or {})
        source_ids = metadata.get("source_region_ids")
        if source_ids:
            return [int(source_id) for source_id in source_ids if int(source_id) > 0]

        if int(getattr(item, "item_id", -1)) > 0:
            return [int(item.item_id)]

        return []

    def _item_union_rect(self, item):
        metadata = dict(getattr(item, "region_metadata", {}) or {})
        bbox = metadata.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            return QRectF(float(x), float(y), float(w), float(h))
        return item.sceneBoundingRect()

    def _get_transform_override(self, item_id):
        if self.interaction_state is None:
            return None
        return self.interaction_state.get_region_transform(int(item_id))

    def _maybe_remove_merge_group(self, item_id):
        if self.interaction_state is None:
            return None
        return self.interaction_state.remove_merge(int(item_id))

    def _discard_item(self, item_id, remember_removal=False):
        item = self.contents.get(item_id)
        if item is None:
            return False

        self.scene.removeItem(item)
        item.setParentItem(None)
        del self.contents[item_id]
        del item

        if remember_removal and item_id not in self.removed_elements_id:
            self.removed_elements_id.append(item_id)

        return True

    def _create_item(self, data):
        item_id = int(data["id"])
        item_type = data.get("type")
        if item_type == "image":
            scene_rect = self.scene.sceneRect()
            item = ImageItem(item_id, data["content"], scene_rect)
        elif item_type == "text":
            item = TextItem(
                item_id,
                data["content"],
                data.get("size", 16) or 16,
                data.get("width"),
                data.get("height"),
            )
        elif item_type == "latex":
            item = LatexItem(
                item_id,
                data["content"],
                data.get("size", 16) or 16,
                data.get("width"),
                data.get("height"),
            )
        else:
            raise ValueError(f"Unsupported canvas item type: {item_type}")

        self._set_item_position(item, data)
        item.region_metadata = dict(data.get("metadata", {}))
        self.scene.addItem(item)
        self.contents[item_id] = item
        if self._get_transform_override(item_id) is None:
            item.setScale(float(data.get("scale", 1.0) or 1.0))

    def _set_item_position(self, item, data):
        transform_override = self._get_transform_override(getattr(item, "item_id", data.get("id")))
        if transform_override is not None:
            item.setPos(
                float(transform_override.get("x", 0.0)),
                float(transform_override.get("y", 0.0)),
            )
            item.setScale(float(transform_override.get("scale", 1.0)))
            return

        x = float(data["x"])
        y = float(data["y"])

        if getattr(item, "item_type", None) == "latex":
            target_width = data.get("width")
            target_height = data.get("height")
            pixmap = item.pixmap()
            if target_width is not None and target_height is not None and not pixmap.isNull():
                x += max(0.0, (float(target_width) - pixmap.width()) / 2.0)
                y += max(0.0, (float(target_height) - pixmap.height()) / 2.0)

        item.setPos(x, y)


    def replace_image(self, item_id, new_content):

        if item_id not in self.contents:
            print("Unknown ID:", item_id)
            return
        
        item = self.contents[item_id]

        item.update_content(new_content)
    
    def _load_images_internal(self, data:dict, update_base_payload=True):
        """
        Preload images into canvas
        """
        data = clone_canvas_payload(data)
        item_type = data.get("type")
        if item_type is None:
            raise KeyError("Canvas payload missing required key: 'type'")

        item_id = int(data["id"])
        if update_base_payload:
            self.base_payloads[item_id] = clone_canvas_payload(data)
        data = self._payload_with_content_override(data)
        item_type = data.get("type")

        if item_id in self.removed_elements_id:
            return
        if item_id in self._active_interaction_item_ids:
            self._deferred_item_events[item_id] = ("load", dict(data))
            return
        if self.paused:
            self.queued_elements.append(data)
            return

        
        if item_id not in self.contents: 
            self._create_item(data)

        else: 
            item = self.contents[item_id]
            if getattr(item, "item_type", None) != item_type:
                was_selected = item.isSelected()
                self._discard_item(item_id, remember_removal=False)
                self._create_item(data)
                if was_selected:
                    self.contents[item_id].setSelected(True)
                return

            if item_type == "image":
                item.update_content(data["content"])
            elif item_type == "text":
                item.update_content(
                    data["content"],
                    data.get("size", 16) or 16,
                    data.get("width"),
                    data.get("height"),
                )
            elif item_type == "latex":
                item.update_content(
                    data["content"],
                    data.get("size", 16) or 16,
                    data.get("width"),
                    data.get("height"),
                )

            item.region_metadata = dict(data.get("metadata", {}))
            self._set_item_position(item, data)

    def load_images(self,data:dict):
        self._load_images_internal(data, update_base_payload=True)

    def remove_image(self, item_id:int):
        """
        Remove an image item by ID
        """
        item_id = int(item_id)
        if self.paused:
            self.queued_elements.append({"_canvas_cmd": "remove", "id": item_id, "remember": True})
            return
        if item_id in self._active_interaction_item_ids:
            self._deferred_item_events[item_id] = ("discard", {"remember": True})
            return True

        if item_id not in self.contents:
            print(f"[remove_image] ID not found: {item_id}")
            return False

        return self._discard_item(item_id, remember_removal=True)

    def expire_content(self, item_id:int):
        item_id = int(item_id)
        if self.paused:
            self.queued_elements.append({"_canvas_cmd": "remove", "id": item_id, "remember": False})
            return
        if item_id in self._active_interaction_item_ids:
            self._deferred_item_events[item_id] = ("discard", {"remember": False})
            return True

        if item_id not in self.contents:
            return False

        return self._discard_item(item_id, remember_removal=False)
    
    def pause(self):
        self.paused = not self.paused
        if not self.paused:
            for item in self.queued_elements:
                if isinstance(item, dict) and item.get("_canvas_cmd") == "remove":
                    self._discard_item(item["id"], remember_removal=item.get("remember", False))
                elif isinstance(item,int):
                    self.remove_image(item)
                else:
                    self.load_images(item)
            self.queued_elements = []

    def remove_selected(self):
        selected_items = self.scene.selectedItems()
        if not selected_items:
            return

        removed_item_snapshots = [self._snapshot_item(item) for item in selected_items]
        removed_merge_groups = []
        for item in selected_items:
            if item.item_id >= 0:
                self.remove_image(item.item_id)
            else:
                existing_group = self._maybe_remove_merge_group(item.item_id)
                if existing_group is not None:
                    removed_merge_groups.append(existing_group)
                self._discard_item(item.item_id, remember_removal=False)

        self._push_undo_action({
            "type": "remove",
            "removed_item_snapshots": removed_item_snapshots,
            "removed_merge_groups": removed_merge_groups,
        })

    def merge_selected(self):
        selected_items = list(self.scene.selectedItems())
        if len(selected_items) < 2:
            return False

        removed_item_snapshots = [self._snapshot_item(item) for item in selected_items]
        removed_merge_groups = []
        source_region_ids = []
        for item in selected_items:
            source_region_ids.extend(self._extract_source_region_ids(item))
            if int(getattr(item, "item_id", -1)) < 0:
                existing_group = self._maybe_remove_merge_group(item.item_id)
                if existing_group is not None:
                    removed_merge_groups.append(existing_group)
        source_region_ids = [source_id for idx, source_id in enumerate(source_region_ids) if source_id not in source_region_ids[:idx]]

        union_rect = self._item_union_rect(selected_items[0])
        for item in selected_items[1:]:
            union_rect = union_rect.united(self._item_union_rect(item))

        source_rect = union_rect
        target_rect = QRectF(0, 0, source_rect.width(), source_rect.height())

        image = QImage(
            max(1, int(round(target_rect.width()))),
            max(1, int(round(target_rect.height()))),
            QImage.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(255, 255, 255, 255))

        hidden_states = []
        selected_states = []
        for item in self.scene.items():
            hidden_states.append((item, item.isVisible()))
            if item not in selected_items:
                item.setVisible(False)
            else:
                selected_states.append((item, item.isSelected()))
                item.setSelected(False)

        painter = QPainter(image)
        self.scene.render(painter, target=target_rect, source=source_rect)
        painter.end()

        for item, was_selected in selected_states:
            item.setSelected(was_selected)
        for item, was_visible in hidden_states:
            item.setVisible(was_visible)

        merged_ids = []
        for item in selected_items:
            merged_ids.append(getattr(item, "item_id", None))
            self._discard_item(item.item_id, remember_removal=False)

        merged_pixmap = QPixmap.fromImage(image)
        merged_id = self.allocate_manual_id()
        merged_bbox = [
            int(round(source_rect.x())),
            int(round(source_rect.y())),
            int(round(source_rect.width())),
            int(round(source_rect.height())),
        ]
        merge_group = None
        if source_region_ids:
            merge_group = self.interaction_state.upsert_merge(
                merged_id,
                source_region_ids,
                bbox=merged_bbox,
            )
        merged_data = {
            "id": merged_id,
            "type": "image",
            "content": qpixmap_to_cv2(merged_pixmap),
            "x": merged_bbox[0],
            "y": merged_bbox[1],
            "metadata": {
                "region_id": merged_id,
                "classification": "merged",
                "status": "manual",
                "bbox": merged_bbox,
                "source_region_ids": list(source_region_ids),
                "merged_source_ids": merged_ids,
                "merge_applied": bool(source_region_ids),
            },
        }
        self.load_images(merged_data)
        if merged_data["id"] in self.contents:
            self.contents[merged_data["id"]].setSelected(True)

        merged_snapshot = self._snapshot_item(self.contents[merged_id])
        self._push_undo_action({
            "type": "merge",
            "merged_item_id": merged_id,
            "merged_snapshot": merged_snapshot,
            "removed_item_snapshots": removed_item_snapshots,
            "removed_merge_groups": removed_merge_groups,
            "merge_group": merge_group,
        })
        return True

    def undo_last_action(self):
        if not self.undo_stack:
            return False

        action = self.undo_stack.pop()
        self._history_suspended = True
        try:
            if action["type"] == "transform":
                return self._apply_item_transform(
                    action["item_id"],
                    action["old_pos"],
                    action["old_scale"],
                )

            if action["type"] == "merge":
                merged_item_id = int(action["merged_item_id"])
                self._maybe_remove_merge_group(merged_item_id)
                self._discard_item(merged_item_id, remember_removal=False)

                for group in action.get("removed_merge_groups", []):
                    self.interaction_state.restore_merge(group)

                restored_items = []
                for snapshot in action.get("removed_item_snapshots", []):
                    restored = self._restore_item_snapshot(snapshot, select=True)
                    if restored is not None:
                        restored_items.append(restored)

                return bool(restored_items)

            if action["type"] == "remove":
                for group in action.get("removed_merge_groups", []):
                    self.interaction_state.restore_merge(group)

                restored_items = []
                for snapshot in action.get("removed_item_snapshots", []):
                    item_id = int(snapshot["id"])
                    if item_id in self.removed_elements_id:
                        self.removed_elements_id.remove(item_id)
                    restored = self._restore_item_snapshot(snapshot, select=True)
                    if restored is not None:
                        restored_items.append(restored)
                return bool(restored_items)

            if action["type"] == "content_override":
                return self._restore_content_override(
                    action["item_id"],
                    action.get("old_override"),
                    select=True,
                )
        finally:
            self._history_suspended = False

        return False

    def add_elements(self,element):
        if isinstance(element,str):
            item_id = self.allocate_manual_id()
            self.load_images({
                "id": item_id,
                "type": "text",
                "content": element,
                "size": 20,
                "x": 50,
                "y": 50,
                "metadata": {
                    "region_id": item_id,
                    "classification": "text",
                    "status": "manual",
                    "bbox": None,
                    "visual_edit_override": "text",
                },
            })
        

class MainWindow(QMainWindow):

    def __init__(self,dimension_x,dimension_y, runtime_settings=None, interaction_state=None, source_info=None, editor_only=False, original_frame=None):
        super().__init__()
        self.runtime_settings = runtime_settings or RuntimeSettingsStore()
        self.interaction_state = interaction_state or InteractionStateStore()
        self.runtime_settings_visible = False
        self.selection_metadata_visible = True
        self.source_info = dict(source_info or {})
        self.editor_only = bool(editor_only)
        self.playback_paused = False
        self.playback_started = True
        self.pause_toggle_callback = None
        self.current_board_path = None
        self.current_board_name = None

        self.setWindowTitle("Board")
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        #Main interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        central_widget.setLayout(main_layout)

        self.view = CanvasView(dimension_x, dimension_y, interaction_state=self.interaction_state)
        self.view.setMinimumSize(320, 240)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.scene.selectionChanged.connect(self.update_selection_metadata)

        self.original_frame_view = VideoFrameView("Original Board Feed")
        self.filtered_frame_view = VideoFrameView("Occlusion Removed Feed")

        self.display_container = QWidget()
        self.display_container.setMinimumSize(320, 240)
        self.display_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_layout = QVBoxLayout()
        self.display_layout.setContentsMargins(0, 0, 0, 0)
        self.display_layout.setSpacing(8)
        self.display_container.setLayout(self.display_layout)
        self.display_root_widget = None

        self.view_widgets = {
            "canvas": self.view,
            "original": self.original_frame_view,
            "filtered": self.filtered_frame_view,
        }
        self.view_labels = {
            "canvas": "Canvas",
            "original": "Original Video",
            "filtered": "Occlusion Removed",
        }
        self.selected_view_keys = ["canvas"]

        #Buttons 
        button_panel = QWidget()
        button_panel.setMinimumWidth(320)
        button_panel.setMaximumWidth(420)
        button_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        button_panel.setLayout(button_layout)

        view_mode_label = QLabel("View Mode")
        button_layout.addWidget(view_mode_label)

        self.view_mode_button = QToolButton()
        self.view_mode_button.setPopupMode(QToolButton.InstantPopup)
        self.view_mode_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.view_mode_menu = QMenu(self.view_mode_button)
        self.view_mode_actions = {}
        for key in ("canvas", "original", "filtered"):
            action = QAction(self.view_labels[key], self)
            action.setCheckable(True)
            action.setChecked(key == "canvas")
            action.toggled.connect(lambda checked, view_key=key: self.on_view_mode_toggled(view_key, checked))
            self.view_mode_menu.addAction(action)
            self.view_mode_actions[key] = action
        self.view_mode_button.setMenu(self.view_mode_menu)
        button_layout.addWidget(self.view_mode_button)
        self._setup_menu_bar()

        classification_mode_label = QLabel("Classification Mode")
        button_layout.addWidget(classification_mode_label)

        self.classification_mode_combo = QComboBox()
        for mode_key, label in CLASSIFICATION_MODE_OPTIONS:
            self.classification_mode_combo.addItem(label, mode_key)
        button_layout.addWidget(self.classification_mode_combo)

        self.video_library_button = QPushButton("Back to Video Library")
        button_layout.addWidget(self.video_library_button)

        self.reset_run_button = QPushButton("Reset Run")
        button_layout.addWidget(self.reset_run_button)

        self.save_board_button = QPushButton("Save Board")
        self.save_board_button.clicked.connect(self.save_board)
        button_layout.addWidget(self.save_board_button)

        self.export_board_button = QPushButton("Export Board Image")
        self.export_board_button.clicked.connect(self.export_board_image)
        button_layout.addWidget(self.export_board_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setMinimumHeight(44)
        self.pause_button.clicked.connect(self.pause)
        button_layout.addWidget(self.pause_button)

        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected)
        button_layout.addWidget(self.remove_button)

        self.merge_button = QPushButton("Merge Selected")
        self.merge_button.clicked.connect(self.merge_selected)
        button_layout.addWidget(self.merge_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_last_action)
        button_layout.addWidget(self.undo_button)

        self.add_word_textbox = QLineEdit()
        button_layout.addWidget(self.add_word_textbox)

        self.add_text_button = QPushButton("Add Text")
        self.add_text_button.clicked.connect(self.add_text)
        button_layout.addWidget(self.add_text_button)

        self.settings_group = QGroupBox("Runtime Settings")
        settings_layout = QFormLayout()
        self.settings_group.setLayout(settings_layout)
        button_layout.addWidget(self.settings_group)

        self.kernel_width_slider = QSlider(Qt.Horizontal)
        self.kernel_width_slider.setRange(2, 30)
        self.kernel_width_slider.setSingleStep(1)
        self.kernel_width_slider.setPageStep(1)
        self.kernel_width_slider.setTickInterval(2)
        self.kernel_width_slider.setTickPosition(QSlider.TicksBelow)
        self.kernel_width_value_label = QLabel("1")
        self.kernel_width_value_label.setMinimumWidth(32)
        kernel_width_widget = QWidget()
        kernel_width_layout = QHBoxLayout(kernel_width_widget)
        kernel_width_layout.setContentsMargins(0, 0, 0, 0)
        kernel_width_layout.setSpacing(8)
        kernel_width_layout.addWidget(self.kernel_width_slider, 1)
        kernel_width_layout.addWidget(self.kernel_width_value_label)
        settings_layout.addRow("Horizontal Handwriting Proximity", kernel_width_widget)

        self.kernel_height_slider = QSlider(Qt.Horizontal)
        self.kernel_height_slider.setRange(2, 30)
        self.kernel_height_slider.setSingleStep(1)
        self.kernel_height_slider.setPageStep(1)
        self.kernel_height_slider.setTickInterval(2)
        self.kernel_height_slider.setTickPosition(QSlider.TicksBelow)
        self.kernel_height_value_label = QLabel("1")
        self.kernel_height_value_label.setMinimumWidth(32)
        kernel_height_widget = QWidget()
        kernel_height_layout = QHBoxLayout(kernel_height_widget)
        kernel_height_layout.setContentsMargins(0, 0, 0, 0)
        kernel_height_layout.setSpacing(8)
        kernel_height_layout.addWidget(self.kernel_height_slider, 1)
        kernel_height_layout.addWidget(self.kernel_height_value_label)
        settings_layout.addRow("Vertical Handwriting Proximity", kernel_height_widget)

        self.segment_match_iou_spin = QDoubleSpinBox()
        self.segment_match_iou_spin.setRange(0.01, 0.95)
        self.segment_match_iou_spin.setDecimals(2)
        self.segment_match_iou_spin.setSingleStep(0.01)
        settings_layout.addRow("Match IoU", self.segment_match_iou_spin)

        self.preprocess_history_spin = QSpinBox()
        self.preprocess_history_spin.setRange(1, 5000)
        settings_layout.addRow("Preprocess History", self.preprocess_history_spin)

        self.preprocess_var_threshold_spin = QDoubleSpinBox()
        self.preprocess_var_threshold_spin.setRange(1.0, 200.0)
        self.preprocess_var_threshold_spin.setDecimals(1)
        self.preprocess_var_threshold_spin.setSingleStep(1.0)
        settings_layout.addRow("Var Threshold", self.preprocess_var_threshold_spin)

        self.settings_note = QLabel("Changes apply to newly processed frames.")
        self.settings_note.setWordWrap(True)
        settings_layout.addRow(self.settings_note)

        self.reset_settings_button = QPushButton("Reset Runtime Settings")
        settings_layout.addRow(self.reset_settings_button)
        self.set_runtime_settings_visible(False)

        self.region_override_label = QLabel("Region Label Override")
        button_layout.addWidget(self.region_override_label)

        self.region_override_combo = QComboBox()
        for override_key, label in REGION_OVERRIDE_OPTIONS:
            self.region_override_combo.addItem(label, override_key)
        button_layout.addWidget(self.region_override_combo)

        self.content_editor_title_label = QLabel("Selected Content Editor")
        button_layout.addWidget(self.content_editor_title_label)

        self.content_editor_group = QGroupBox()
        content_editor_layout = QVBoxLayout()
        self.content_editor_group.setLayout(content_editor_layout)
        button_layout.addWidget(self.content_editor_group)

        self.content_editor_status_label = QLabel("Pause the run or open a saved board to edit a selected region.")
        self.content_editor_status_label.setWordWrap(True)
        content_editor_layout.addWidget(self.content_editor_status_label)

        self.content_editor_stack = QStackedWidget()
        content_editor_layout.addWidget(self.content_editor_stack)

        self.content_editor_empty_page = QWidget()
        empty_layout = QVBoxLayout()
        self.content_editor_empty_page.setLayout(empty_layout)
        self.content_editor_empty_label = QLabel("Select a single region to edit.")
        self.content_editor_empty_label.setWordWrap(True)
        empty_layout.addWidget(self.content_editor_empty_label)
        empty_layout.addStretch()
        self.content_editor_stack.addWidget(self.content_editor_empty_page)

        self.content_editor_text_page = QWidget()
        text_layout = QVBoxLayout()
        self.content_editor_text_page.setLayout(text_layout)
        self.content_editor_mode_label = QLabel("Edit content")
        text_layout.addWidget(self.content_editor_mode_label)
        self.content_editor_textbox = QPlainTextEdit()
        self.content_editor_textbox.setMinimumHeight(120)
        text_layout.addWidget(self.content_editor_textbox)
        text_button_row = QHBoxLayout()
        self.apply_content_edit_button = QPushButton("Apply Edit")
        text_button_row.addWidget(self.apply_content_edit_button)
        self.reset_content_editor_button = QPushButton("Reset Field")
        text_button_row.addWidget(self.reset_content_editor_button)
        text_layout.addLayout(text_button_row)
        self.content_editor_stack.addWidget(self.content_editor_text_page)

        self.content_editor_image_page = QWidget()
        image_layout = QVBoxLayout()
        self.content_editor_image_page.setLayout(image_layout)
        self.content_editor_image_label = QLabel("Open the diagram editor to draw directly on the selected region.")
        self.content_editor_image_label.setWordWrap(True)
        image_layout.addWidget(self.content_editor_image_label)
        self.open_diagram_editor_button = QPushButton("Edit Diagram")
        image_layout.addWidget(self.open_diagram_editor_button)
        image_layout.addStretch()
        self.content_editor_stack.addWidget(self.content_editor_image_page)

        content_override_button_row = QHBoxLayout()
        self.clear_content_override_button = QPushButton("Clear Visual Edit")
        content_override_button_row.addWidget(self.clear_content_override_button)
        content_override_button_row.addStretch()
        content_editor_layout.addLayout(content_override_button_row)

        self.metadata_title_label = QLabel("Selection Metadata")
        button_layout.addWidget(self.metadata_title_label)

        self.metadata_view = QPlainTextEdit()
        self.metadata_view.setReadOnly(True)
        self.metadata_view.setMinimumWidth(320)
        self.metadata_view.setPlainText("Select a region to inspect metadata.")
        button_layout.addWidget(self.metadata_view)

        self._update_region_override_controls()
        self._content_editor_context = {}
        self._content_editor_dirty = False
        self._programmatic_content_editor_update = False
        self.region_refresh_callback = None
        self.set_selection_metadata_visible(self.selection_metadata_visible)

        button_layout.addStretch()

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setWidget(button_panel)
        sidebar_scroll.setMinimumWidth(340)
        sidebar_scroll.setMaximumWidth(440)
        sidebar_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.addWidget(self.display_container)
        self.main_splitter.addWidget(sidebar_scroll)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([
            max(720, int(dimension_x)),
            360,
        ])
        main_layout.addWidget(self.main_splitter, 1)

        self.metadata_refresh_timer = QTimer(self)
        self.metadata_refresh_timer.setInterval(250)
        self.metadata_refresh_timer.timeout.connect(self.update_selection_metadata)
        self.metadata_refresh_timer.start()
        self._connect_settings_controls()
        self._load_runtime_settings_into_controls()
        if original_frame is not None:
            self.update_original_frame(original_frame)
        self.apply_view_selection_layout()
        self._apply_editor_only_mode()
        self._update_playback_button_state()
        self.statusBar().showMessage("Ready")

    def _setup_menu_bar(self):
        file_menu = self.menuBar().addMenu("&File")
        self.save_board_action = QAction("Save Board", self)
        self.save_board_action.setShortcut(QKeySequence.Save)
        self.save_board_action.triggered.connect(self.save_board)
        file_menu.addAction(self.save_board_action)

        self.export_board_action = QAction("Export Board Image", self)
        self.export_board_action.setShortcut("Ctrl+Shift+E")
        self.export_board_action.triggered.connect(self.export_board_image)
        file_menu.addAction(self.export_board_action)

        file_menu.addSeparator()
        self.video_library_action = QAction("Back to Video Library", self)
        self.video_library_action.setShortcut("Ctrl+L")
        file_menu.addAction(self.video_library_action)

        self.reset_run_action = QAction("Reset Run", self)
        self.reset_run_action.setShortcut("Ctrl+R")
        file_menu.addAction(self.reset_run_action)

        file_menu.addSeparator()
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self.undo_last_action)
        edit_menu.addAction(self.undo_action)

        self.merge_action = QAction("Merge Selected", self)
        self.merge_action.setShortcut("Ctrl+M")
        self.merge_action.triggered.connect(self.merge_selected)
        edit_menu.addAction(self.merge_action)

        self.remove_action = QAction("Remove Selected", self)
        self.remove_action.setShortcut(QKeySequence.Delete)
        self.remove_action.triggered.connect(self.remove_selected)
        edit_menu.addAction(self.remove_action)

        self.edit_selected_content_action = QAction("Edit Selected Content", self)
        self.edit_selected_content_action.setShortcut("Ctrl+E")
        self.edit_selected_content_action.triggered.connect(self.edit_selected_content)
        edit_menu.addAction(self.edit_selected_content_action)

        view_menu = self.menuBar().addMenu("&View")
        for key in ("canvas", "original", "filtered"):
            view_menu.addAction(self.view_mode_actions[key])

        run_menu = self.menuBar().addMenu("&Run")
        self.pause_action = QAction("Pause", self)
        self.pause_action.setShortcut("Space")
        self.pause_action.triggered.connect(self.pause)
        run_menu.addAction(self.pause_action)
        run_menu.addAction(self.reset_run_action)

        settings_menu = self.menuBar().addMenu("&Settings")
        self.toggle_runtime_settings_action = QAction("Show Runtime Settings", self)
        self.toggle_runtime_settings_action.setCheckable(True)
        self.toggle_runtime_settings_action.toggled.connect(self.set_runtime_settings_visible)
        settings_menu.addAction(self.toggle_runtime_settings_action)
        self.camera_resolution_menu = settings_menu.addMenu("Live Camera Resolution")
        self.camera_resolution_action_group = QActionGroup(self)
        self.camera_resolution_action_group.setExclusive(True)
        self.camera_resolution_actions = {}
        for option_key, label, _dims in CAMERA_RESOLUTION_OPTIONS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, key=option_key: self.set_camera_resolution_option(key)
            )
            self.camera_resolution_action_group.addAction(action)
            self.camera_resolution_menu.addAction(action)
            self.camera_resolution_actions[option_key] = action
        self.text_ocr_backend_menu = settings_menu.addMenu("Text OCR Backend")
        self.text_ocr_backend_action_group = QActionGroup(self)
        self.text_ocr_backend_action_group.setExclusive(True)
        self.text_ocr_backend_actions = {}
        for option_key, label in TEXT_OCR_BACKEND_OPTIONS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, key=option_key: self.set_text_ocr_backend_option(key)
            )
            self.text_ocr_backend_action_group.addAction(action)
            self.text_ocr_backend_menu.addAction(action)
            self.text_ocr_backend_actions[option_key] = action
        self.text_ocr_model_menu = settings_menu.addMenu("Text OCR Model")
        self.text_ocr_model_action_group = QActionGroup(self)
        self.text_ocr_model_action_group.setExclusive(True)
        self.text_ocr_model_actions = {}
        for option_key, label in TEXT_OCR_MODEL_OPTIONS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, key=option_key: self.set_text_ocr_model_option(key)
            )
            self.text_ocr_model_action_group.addAction(action)
            self.text_ocr_model_menu.addAction(action)
            self.text_ocr_model_actions[option_key] = action
        self.vectorization_method_menu = settings_menu.addMenu("Diagram Vectorization")
        self.vectorization_method_action_group = QActionGroup(self)
        self.vectorization_method_action_group.setExclusive(True)
        self.vectorization_method_actions = {}
        for option_key, label in VECTORIZATION_METHOD_OPTIONS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, key=option_key: self.set_vectorization_method_option(key)
            )
            self.vectorization_method_action_group.addAction(action)
            self.vectorization_method_menu.addAction(action)
            self.vectorization_method_actions[option_key] = action
        self.equation_model_menu = settings_menu.addMenu("Equation OCR Model")
        self.equation_model_action_group = QActionGroup(self)
        self.equation_model_action_group.setExclusive(True)
        self.equation_model_actions = {}
        for option_key, label in EQUATION_OCR_MODEL_OPTIONS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, key=option_key: self.set_equation_ocr_model_option(key)
            )
            self.equation_model_action_group.addAction(action)
            self.equation_model_menu.addAction(action)
            self.equation_model_actions[option_key] = action
        self.toggle_selection_metadata_action = QAction("Show Selection Metadata", self)
        self.toggle_selection_metadata_action.setCheckable(True)
        self.toggle_selection_metadata_action.setChecked(True)
        self.toggle_selection_metadata_action.toggled.connect(self.set_selection_metadata_visible)
        settings_menu.addAction(self.toggle_selection_metadata_action)
        settings_menu.addSeparator()
        self.open_settings_file_action = QAction("Open Settings File", self)
        self.open_settings_file_action.triggered.connect(self.open_app_settings_file)
        settings_menu.addAction(self.open_settings_file_action)
        self.reset_runtime_settings_action = QAction("Reset Runtime Settings", self)
        self.reset_runtime_settings_action.triggered.connect(self.reset_runtime_settings)
        settings_menu.addAction(self.reset_runtime_settings_action)

        help_menu = self.menuBar().addMenu("&Help")
        self.shortcuts_action = QAction("Keyboard Shortcuts", self)
        self.shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        help_menu.addAction(self.shortcuts_action)

    def show_shortcuts_dialog(self):
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "\n".join([
                "Ctrl+Z: Undo",
                "Ctrl+M: Merge selected regions",
                "Ctrl+E: Edit the selected region",
                "Delete: Remove selected regions",
                "Space: Pause or resume",
                "Ctrl+R: Reset the current run",
                "Ctrl+L: Return to video library",
            ]),
        )

    def set_pause_toggle_callback(self, callback):
        self.pause_toggle_callback = callback

    def set_region_refresh_callback(self, callback):
        self.region_refresh_callback = callback

    def _update_playback_button_state(self):
        if self.playback_paused:
            button_label = "Start Playback" if not self.playback_started else "Resume Playback"
            action_label = button_label
            style = (
                "QPushButton {"
                "background-color: #1f8f4e; color: white; font-weight: 700; "
                "border: 1px solid #176c3a; border-radius: 6px; padding: 8px 14px;"
                "}"
                "QPushButton:hover { background-color: #249f58; }"
                "QPushButton:disabled { background-color: #8fbfa1; color: #f5f5f5; }"
            )
        else:
            button_label = "Pause Playback"
            action_label = button_label
            style = (
                "QPushButton {"
                "background-color: #d9841f; color: white; font-weight: 700; "
                "border: 1px solid #a96517; border-radius: 6px; padding: 8px 14px;"
                "}"
                "QPushButton:hover { background-color: #ea9530; }"
                "QPushButton:disabled { background-color: #d6b38b; color: #f5f5f5; }"
            )

        self.pause_button.setText(button_label)
        self.pause_button.setStyleSheet(style)
        self.pause_action.setText(action_label)

    def set_playback_paused(self, paused, initial_start=False, sync_canvas_queue=False):
        paused = bool(paused)
        if initial_start and paused:
            self.playback_started = False
        elif not paused:
            self.playback_started = True

        self.playback_paused = paused

        if sync_canvas_queue and self.view.paused != paused:
            self.view.pause()

        if self.pause_toggle_callback is not None:
            self.pause_toggle_callback(self.playback_paused)

        self._update_playback_button_state()
        self._refresh_content_editor_visibility()

    def pause(self):
        self.set_playback_paused(not self.playback_paused, sync_canvas_queue=True)
        self.update_selection_metadata()

    def remove_selected(self):
        self.view.remove_selected()

    def merge_selected(self):
        merged = self.view.merge_selected()
        if merged and self.region_refresh_callback is not None:
            region_id = self._selected_region_id()
            if region_id is not None:
                self.region_refresh_callback(region_id)
        if not merged:
            QMessageBox.information(
                self,
                "Merge Selected",
                "Select at least two regions on the canvas to merge them.",
            )

    def undo_last_action(self):
        self.view.undo_last_action()

    def add_text(self):
        text = self.add_word_textbox.text()
        self.view.add_elements(text)

    def set_board_document_context(self, board_path=None, board_name=None):
        self.current_board_path = str(board_path) if board_path else None
        self.current_board_name = str(board_name) if board_name else None

    def _default_board_name(self):
        if self.current_board_name:
            return self.current_board_name
        source_name = str(self.source_info.get("name") or self.windowTitle() or "board")
        return source_name.replace("Board - ", "").strip() or "board"

    def save_board(self):
        board_name = self._default_board_name()
        if not self.current_board_path:
            entered_name, accepted = QInputDialog.getText(
                self,
                "Save Board",
                "Board name:",
                text=board_name,
            )
            if not accepted:
                return
            board_name = str(entered_name).strip() or board_name

        board_path = save_board_document(
            board_name=board_name,
            width=int(self.view.scene.sceneRect().width()),
            height=int(self.view.scene.sceneRect().height()),
            item_snapshots=self.view.snapshot_all_items(),
            interaction_state_snapshot=build_interaction_state_snapshot(self.interaction_state),
            source_info=self.source_info,
            selected_view_keys=self.selected_view_keys,
            existing_board_path=self.current_board_path,
            original_frame=self.current_original_frame(),
        )
        self.set_board_document_context(board_path, board_name)
        self.statusBar().showMessage(f"Board saved: {board_path}", 5000)

    def current_original_frame(self):
        if self.original_frame_view.base_pixmap.isNull():
            return None
        return qpixmap_to_cv2(self.original_frame_view.base_pixmap)

    def _render_board_scene_image(self):
        scene_rect = self.view.scene.sceneRect()
        image = QImage(
            max(1, int(round(scene_rect.width()))),
            max(1, int(round(scene_rect.height()))),
            QImage.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(255, 255, 255, 255))

        selected_items = list(self.view.scene.selectedItems())
        for item in selected_items:
            item.setSelected(False)

        painter = QPainter(image)
        self.view.scene.render(painter, target=QRectF(image.rect()), source=scene_rect)
        painter.end()

        for item in selected_items:
            item.setSelected(True)

        return image

    def export_board_image(self):
        full_image = self._render_board_scene_image()
        crop_dialog = ExportCropDialog(full_image, self)
        if crop_dialog.exec() != QDialog.Accepted:
            return

        export_image = crop_dialog.cropped_image()

        default_name = f"{self._default_board_name()}.png"
        default_path = REPO_ROOT / default_name
        if self.current_board_path:
            default_path = Path(self.current_board_path).parent / default_name

        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Board Image",
            str(default_path),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not output_path:
            return

        if not export_image.save(output_path):
            QMessageBox.critical(self, "Export Failed", f"Could not export board image to:\n{output_path}")
            return

        self.statusBar().showMessage(f"Board exported: {output_path}", 5000)

    def _connect_settings_controls(self):
        self.classification_mode_combo.currentIndexChanged.connect(self.on_classification_mode_changed)
        self.kernel_width_slider.valueChanged.connect(
            lambda value: self.on_handwriting_proximity_changed("width", value)
        )
        self.kernel_height_slider.valueChanged.connect(
            lambda value: self.on_handwriting_proximity_changed("height", value)
        )
        self.segment_match_iou_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(segment_match_iou_thresh=float(value))
        )
        self.preprocess_history_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(preprocess_history=int(value))
        )
        self.preprocess_var_threshold_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(preprocess_var_threshold=float(value))
        )
        self.reset_settings_button.clicked.connect(self.reset_runtime_settings)
        self.region_override_combo.currentIndexChanged.connect(self.on_region_override_changed)
        self.apply_content_edit_button.clicked.connect(self.apply_selected_content_edit)
        self.reset_content_editor_button.clicked.connect(self.reset_selected_content_editor)
        self.clear_content_override_button.clicked.connect(self.clear_selected_content_edit)
        self.open_diagram_editor_button.clicked.connect(self.open_selected_diagram_editor)
        self.content_editor_textbox.textChanged.connect(self.on_content_editor_text_changed)

    def on_handwriting_proximity_changed(self, axis, slider_value):
        slider_value = int(slider_value)
        fraction = proximity_slider_value_to_fraction(slider_value)
        if axis == "width":
            self.kernel_width_value_label.setText(proximity_slider_value_to_label(slider_value))
            self.runtime_settings.update(segment_kernel_width_frac=fraction)
        else:
            self.kernel_height_value_label.setText(proximity_slider_value_to_label(slider_value))
            self.runtime_settings.update(segment_kernel_height_frac=fraction)

    def _load_runtime_settings_into_controls(self):
        snapshot = self.runtime_settings.snapshot()
        kernel_width_value = proximity_fraction_to_slider_value(snapshot.segment_kernel_width_frac)
        kernel_height_value = proximity_fraction_to_slider_value(snapshot.segment_kernel_height_frac)
        self.kernel_width_slider.blockSignals(True)
        self.kernel_width_slider.setValue(kernel_width_value)
        self.kernel_width_slider.blockSignals(False)
        self.kernel_width_value_label.setText(proximity_slider_value_to_label(kernel_width_value))
        self.kernel_height_slider.blockSignals(True)
        self.kernel_height_slider.setValue(kernel_height_value)
        self.kernel_height_slider.blockSignals(False)
        self.kernel_height_value_label.setText(proximity_slider_value_to_label(kernel_height_value))

        controls = (
            (self.segment_match_iou_spin, snapshot.segment_match_iou_thresh),
            (self.preprocess_history_spin, snapshot.preprocess_history),
            (self.preprocess_var_threshold_spin, snapshot.preprocess_var_threshold),
        )
        for widget, value in controls:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._load_camera_resolution_into_controls(snapshot.camera_resolution_option)
        self._load_text_ocr_backend_into_controls(snapshot.text_ocr_backend)
        self._load_text_ocr_model_into_controls(snapshot.text_ocr_model)
        self._load_vectorization_method_into_controls(snapshot.vectorization_method)
        self._load_equation_ocr_model_into_controls(snapshot.equation_ocr_model)
        self._load_classification_mode_into_controls()

    def reset_runtime_settings(self):
        self.runtime_settings.reset_defaults()
        self._load_runtime_settings_into_controls()
        self.statusBar().showMessage(
            f"Runtime settings reset from {APP_SETTINGS_PATH.name}.",
            4000,
        )

    def open_app_settings_file(self):
        settings_path = ensure_app_settings_file()
        try:
            os.startfile(str(settings_path))
        except AttributeError:
            subprocess.Popen(["open", str(settings_path)])
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Open Settings File",
                f"Could not open settings file:\n{settings_path}\n\n{exc}",
            )
            return
        self.statusBar().showMessage(f"Opened settings file: {settings_path}", 4000)

    def _load_camera_resolution_into_controls(self, option_key):
        normalized_option = str(option_key or "max").strip().lower()
        for candidate_key, action in getattr(self, "camera_resolution_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(candidate_key == normalized_option)
            action.blockSignals(False)

    def _load_text_ocr_backend_into_controls(self, option_key):
        normalized_option = str(option_key or DEFAULT_TEXT_OCR_BACKEND).strip().lower()
        for candidate_key, action in getattr(self, "text_ocr_backend_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(candidate_key == normalized_option)
            action.blockSignals(False)

    def _load_text_ocr_model_into_controls(self, option_key):
        normalized_option = str(option_key or DEFAULT_TEXT_OCR_MODEL).strip()
        for candidate_key, action in getattr(self, "text_ocr_model_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(candidate_key == normalized_option)
            action.blockSignals(False)

    def _load_vectorization_method_into_controls(self, option_key):
        normalized_option = normalize_vectorization_method(option_key)
        for candidate_key, action in getattr(self, "vectorization_method_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(candidate_key == normalized_option)
            action.blockSignals(False)

    def _load_equation_ocr_model_into_controls(self, option_key):
        normalized_option = str(option_key or DEFAULT_EQUATION_OCR_MODEL).strip()
        for candidate_key, action in getattr(self, "equation_model_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(candidate_key == normalized_option)
            action.blockSignals(False)

    def set_camera_resolution_option(self, option_key):
        normalized_option = str(option_key or "max").strip().lower()
        self.runtime_settings.update(camera_resolution_option=normalized_option)
        self._load_camera_resolution_into_controls(normalized_option)
        self.statusBar().showMessage(
            f"Live camera resolution set to {camera_resolution_label(normalized_option)}. Applies on the next camera session.",
            5000,
        )

    def set_text_ocr_backend_option(self, option_key):
        normalized_option = str(option_key or DEFAULT_TEXT_OCR_BACKEND).strip().lower()
        self.runtime_settings.update(text_ocr_backend=normalized_option)
        self._load_text_ocr_backend_into_controls(normalized_option)
        self.source_info["text_ocr_backend"] = normalized_option
        self.statusBar().showMessage(
            f"Text OCR backend set to {normalized_option}. Applies to future text OCR refreshes.",
            5000,
        )

    def set_text_ocr_model_option(self, option_key):
        normalized_option = str(option_key or DEFAULT_TEXT_OCR_MODEL).strip()
        self.runtime_settings.update(text_ocr_model=normalized_option)
        self._load_text_ocr_model_into_controls(normalized_option)
        self.source_info["text_ocr_model"] = normalized_option
        self.statusBar().showMessage(
            f"Text OCR model set to {normalized_option}. Applies to future TrOCR requests.",
            5000,
        )

    def set_vectorization_method_option(self, option_key):
        normalized_option = normalize_vectorization_method(option_key)
        self.runtime_settings.update(vectorization_method=normalized_option)
        self._load_vectorization_method_into_controls(normalized_option)
        self.source_info["vectorization_method"] = normalized_option
        self.statusBar().showMessage(
            f"Diagram vectorization set to {normalized_option.upper()}. Applies to newly processed regions.",
            5000,
        )

    def set_equation_ocr_model_option(self, option_key):
        normalized_option = str(option_key or DEFAULT_EQUATION_OCR_MODEL).strip()
        self.runtime_settings.update(equation_ocr_model=normalized_option)
        self._load_equation_ocr_model_into_controls(normalized_option)
        self.source_info["equation_ocr_model"] = normalized_option
        self.statusBar().showMessage(
            f"Equation OCR model set to {normalized_option}. Applies to future OCR requests.",
            5000,
        )

    def _load_classification_mode_into_controls(self):
        current_mode = self.interaction_state.get_classification_mode()
        index = self.classification_mode_combo.findData(current_mode)
        if index < 0:
            index = 0
        self.classification_mode_combo.blockSignals(True)
        self.classification_mode_combo.setCurrentIndex(index)
        self.classification_mode_combo.blockSignals(False)
        self._update_region_override_controls()

    def on_classification_mode_changed(self, _index):
        selected_mode = str(self.classification_mode_combo.currentData() or "normal")
        self.interaction_state.set_classification_mode(selected_mode)
        selected_mode_label = self.classification_mode_combo.currentText()
        self.statusBar().showMessage(f"Classification mode: {selected_mode_label}", 3000)
        self._update_region_override_controls()
        self.update_selection_metadata()

    def _selected_region_id(self):
        selected_items = self.view.scene.selectedItems()
        if len(selected_items) != 1:
            return None

        metadata = dict(getattr(selected_items[0], "region_metadata", {}) or {})
        region_id = metadata.get("region_id", getattr(selected_items[0], "item_id", None))
        try:
            return int(region_id)
        except (TypeError, ValueError):
            return None

    def _update_region_override_controls(self):
        selected_region_id = self._selected_region_id()
        normal_mode = self.interaction_state.get_classification_mode() == "normal"
        should_show = normal_mode and selected_region_id is not None

        self.region_override_label.setVisible(should_show)
        self.region_override_combo.setVisible(should_show)
        self.region_override_combo.setEnabled(should_show)

        override_value = ""
        if should_show:
            override_value = self.interaction_state.get_region_label_override(selected_region_id) or ""

        index = self.region_override_combo.findData(override_value)
        if index < 0:
            index = 0
        self.region_override_combo.blockSignals(True)
        self.region_override_combo.setCurrentIndex(index)
        self.region_override_combo.blockSignals(False)

    def on_region_override_changed(self, _index):
        region_id = self._selected_region_id()
        if region_id is None:
            return
        override_value = str(self.region_override_combo.currentData() or "")
        self.interaction_state.set_region_label_override(region_id, override_value)
        override_label = self.region_override_combo.currentText()
        if self.region_refresh_callback is not None:
            self.region_refresh_callback(region_id)
        if override_value:
            self.statusBar().showMessage(f"Region {region_id} override set to {override_label}", 3000)
        else:
            self.statusBar().showMessage(f"Region {region_id} override cleared", 3000)
        self.update_selection_metadata()

    def _selected_canvas_item(self):
        selected_items = self.view.scene.selectedItems()
        if len(selected_items) != 1:
            return None
        return selected_items[0]

    def _content_editing_allowed(self):
        return bool(self.editor_only or self.playback_paused)

    def _refresh_content_editor_visibility(self):
        canvas_selected = "canvas" in self.selected_view_keys
        editor_visible = canvas_selected and self._content_editing_allowed()
        if hasattr(self, "content_editor_title_label"):
            self.content_editor_title_label.setVisible(editor_visible)
        if hasattr(self, "content_editor_group"):
            self.content_editor_group.setVisible(editor_visible)

        selected_target = self._selected_editor_target() if canvas_selected else None
        if hasattr(self, "edit_selected_content_action"):
            self.edit_selected_content_action.setEnabled(editor_visible and selected_target is not None)

    def _selected_editor_target(self):
        item = self._selected_canvas_item()
        if item is None:
            return None

        metadata = dict(getattr(item, "region_metadata", {}) or {})
        item_id = int(getattr(item, "item_id", -1))
        classification = str(metadata.get("classification") or "").strip().lower()
        item_type = str(getattr(item, "item_type", "") or "").strip().lower()

        if item_type == "latex" or classification == "equation":
            editor_mode = "latex"
        elif item_type == "text" or classification == "text":
            editor_mode = "text"
        else:
            editor_mode = "image"

        content_value = ""
        if editor_mode == "latex":
            content_value = str(getattr(item, "latex_value", "") or metadata.get("equation_ocr_prediction_latex") or metadata.get("text_preview") or "")
        elif editor_mode == "text":
            if hasattr(item, "toPlainText"):
                content_value = str(item.toPlainText() or "")
            else:
                content_value = str(metadata.get("text_preview") or "")

        return {
            "item": item,
            "item_id": item_id,
            "editor_mode": editor_mode,
            "content_value": content_value,
            "metadata": metadata,
        }

    def _set_content_editor_text(self, text):
        self._programmatic_content_editor_update = True
        try:
            self.content_editor_textbox.setPlainText(str(text or ""))
        finally:
            self._programmatic_content_editor_update = False

    def _current_content_editor_text(self):
        return self.content_editor_textbox.toPlainText()

    def on_content_editor_text_changed(self):
        if self._programmatic_content_editor_update:
            return
        source_text = str(self._content_editor_context.get("source_text") or "")
        self._content_editor_dirty = self._current_content_editor_text() != source_text

    def update_content_editor(self):
        target = self._selected_editor_target()
        editing_allowed = self._content_editing_allowed()
        self._refresh_content_editor_visibility()

        if not editing_allowed:
            self._content_editor_context = {}
            self._content_editor_dirty = False
            return

        if target is None:
            self.content_editor_status_label.setText("Select a single region to edit.")
            self.content_editor_stack.setCurrentWidget(self.content_editor_empty_page)
            self.clear_content_override_button.setEnabled(False)
            self.apply_content_edit_button.setEnabled(False)
            self.reset_content_editor_button.setEnabled(False)
            self.open_diagram_editor_button.setEnabled(False)
            self._content_editor_context = {}
            self._content_editor_dirty = False
            return

        item_id = int(target["item_id"])
        editor_mode = target["editor_mode"]
        override_active = self.interaction_state.get_region_content_override(item_id) is not None
        context_key = (item_id, editor_mode)
        current_key = self._content_editor_context.get("key")

        if editor_mode == "image":
            self.content_editor_status_label.setText("Diagram edits are applied directly to the selected region image.")
        elif editor_mode == "latex":
            self.content_editor_status_label.setText("Edit the displayed equation as LaTeX.")
        else:
            self.content_editor_status_label.setText("Edit the displayed text content.")

        if editor_mode == "image":
            self.content_editor_stack.setCurrentWidget(self.content_editor_image_page)
            self.open_diagram_editor_button.setEnabled(editing_allowed)
            self.apply_content_edit_button.setEnabled(False)
            self.reset_content_editor_button.setEnabled(False)
        else:
            self.content_editor_stack.setCurrentWidget(self.content_editor_text_page)
            self.content_editor_mode_label.setText("Edit LaTeX" if editor_mode == "latex" else "Edit Text")
            target_text = str(target["content_value"] or "")
            should_refresh_text = (
                current_key != context_key
                or (
                    not self._content_editor_dirty
                    and self._current_content_editor_text() != target_text
                )
            )
            if should_refresh_text:
                self._set_content_editor_text(target["content_value"])
                self._content_editor_dirty = False
            self.apply_content_edit_button.setEnabled(editing_allowed)
            self.reset_content_editor_button.setEnabled(editing_allowed)
            self.open_diagram_editor_button.setEnabled(False)

        self.clear_content_override_button.setEnabled(editing_allowed and override_active)
        self.content_editor_textbox.setReadOnly(not editing_allowed)
        self._content_editor_context = {
            "key": context_key,
            "item_id": item_id,
            "editor_mode": editor_mode,
            "source_text": str(target["content_value"] or ""),
        }

    def reset_selected_content_editor(self):
        if not self._content_editor_context:
            return
        self._set_content_editor_text(self._content_editor_context.get("source_text", ""))
        self._content_editor_dirty = False

    def apply_selected_content_edit(self):
        target = self._selected_editor_target()
        if target is None:
            return
        if not self._content_editing_allowed():
            self.statusBar().showMessage("Pause the run or open a saved board to edit content.", 3000)
            return

        item_id = int(target["item_id"])
        editor_mode = target["editor_mode"]
        if editor_mode not in {"text", "latex"}:
            return

        edited_text = self._current_content_editor_text().rstrip("\n")
        bbox = list(target["metadata"].get("bbox") or [0, 0, 0, 0])
        bbox_height = int(bbox[3] or 0) if len(bbox) >= 4 else 0
        override = {
            "type": editor_mode,
            "content": edited_text,
        }
        if editor_mode == "text":
            override["size"] = max(12, min(30, int(round(max(bbox_height, 1) / 3.5))))
        if editor_mode == "latex":
            override["size"] = max(4, min(14, int(round(max(bbox_height, 1) / 12))))
            override["width"] = int(bbox[2] or 0) or None
            override["height"] = int(bbox[3] or 0) or None

        changed = self.view.set_item_content_override(item_id, override, select_after=True)
        if changed:
            self._content_editor_context["source_text"] = edited_text
            self._content_editor_dirty = False
            self.statusBar().showMessage(f"Applied visual edit to region {item_id}.", 3000)
            self.update_selection_metadata()

    def clear_selected_content_edit(self):
        target = self._selected_editor_target()
        if target is None:
            return
        if not self._content_editing_allowed():
            self.statusBar().showMessage("Pause the run or open a saved board to edit content.", 3000)
            return

        item_id = int(target["item_id"])
        changed = self.view.set_item_content_override(item_id, None, select_after=True)
        if changed:
            self.statusBar().showMessage(f"Cleared visual edit for region {item_id}.", 3000)
            self._content_editor_dirty = False
            self.update_selection_metadata()

    def open_selected_diagram_editor(self):
        target = self._selected_editor_target()
        if target is None or target["editor_mode"] != "image":
            return
        if not self._content_editing_allowed():
            self.statusBar().showMessage("Pause the run or open a saved board to edit content.", 3000)
            return

        current_item = target["item"]
        if not hasattr(current_item, "pixmap"):
            return
        image = qpixmap_to_cv2(current_item.pixmap())
        dialog = DiagramEditorDialog(image, self)
        if dialog.exec() != QDialog.Accepted:
            return

        override = {
            "type": "image",
            "content": encode_cv2_image_base64(dialog.edited_image()),
        }
        changed = self.view.set_item_content_override(int(target["item_id"]), override, select_after=True)
        if changed:
            self.statusBar().showMessage(f"Applied diagram edit to region {target['item_id']}.", 3000)
            self.update_selection_metadata()

    def edit_selected_content(self):
        target = self._selected_editor_target()
        if target is None:
            return
        if not self._content_editing_allowed():
            self.statusBar().showMessage("Pause the run or open a saved board to edit content.", 3000)
            return
        if target["editor_mode"] == "image":
            self.open_selected_diagram_editor()
            return
        self.content_editor_textbox.setFocus()
        self.content_editor_textbox.selectAll()

    def set_runtime_settings_visible(self, visible):
        self.runtime_settings_visible = bool(visible)
        if hasattr(self, "settings_group"):
            self.settings_group.setVisible(self.runtime_settings_visible)
        if hasattr(self, "toggle_runtime_settings_action"):
            self.toggle_runtime_settings_action.blockSignals(True)
            self.toggle_runtime_settings_action.setChecked(self.runtime_settings_visible)
            self.toggle_runtime_settings_action.blockSignals(False)

    def set_selection_metadata_visible(self, visible):
        self.selection_metadata_visible = bool(visible)
        if hasattr(self, "metadata_title_label"):
            self.metadata_title_label.setVisible(self.selection_metadata_visible)
        if hasattr(self, "metadata_view"):
            self.metadata_view.setVisible(self.selection_metadata_visible)
        if hasattr(self, "toggle_selection_metadata_action"):
            self.toggle_selection_metadata_action.blockSignals(True)
            self.toggle_selection_metadata_action.setChecked(self.selection_metadata_visible)
            self.toggle_selection_metadata_action.blockSignals(False)

    def _apply_editor_only_mode(self):
        if not self.editor_only:
            return

        allowed_view_keys = {"canvas"}
        if not self.original_frame_view.base_pixmap.isNull():
            allowed_view_keys.add("original")

        self.selected_view_keys = [
            key for key in self.selected_view_keys
            if key in allowed_view_keys
        ] or ["canvas"]

        for key, action in self.view_mode_actions.items():
            is_allowed = key in allowed_view_keys
            action.blockSignals(True)
            action.setChecked(key in self.selected_view_keys)
            action.setEnabled(is_allowed)
            action.blockSignals(False)

        self.view_mode_button.setEnabled(len(allowed_view_keys) > 1)
        self.reset_run_button.setEnabled(False)
        self.reset_run_action.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_action.setEnabled(False)
        self.filtered_frame_view.hide()
        self.apply_view_selection_layout()

    def on_view_mode_toggled(self, key, checked):
        selected = [view_key for view_key, action in self.view_mode_actions.items() if action.isChecked()]
        if not selected:
            action = self.view_mode_actions[key]
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
            selected = [key]

        self.selected_view_keys = [view_key for view_key in ("canvas", "original", "filtered") if self.view_mode_actions[view_key].isChecked()]
        self.apply_view_selection_layout()

    def _build_view_splitter(self, orientation, widgets):
        splitter = QSplitter(orientation)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        for index, widget in enumerate(widgets):
            splitter.addWidget(widget)
            splitter.setStretchFactor(index, 1)
        return splitter

    def apply_view_selection_layout(self):
        for widget in self.view_widgets.values():
            widget.hide()
            widget.setParent(None)

        if self.display_root_widget is not None:
            self.display_layout.removeWidget(self.display_root_widget)
            if self.display_root_widget not in self.view_widgets.values():
                self.display_root_widget.deleteLater()
            self.display_root_widget = None

        selected = list(self.selected_view_keys) or ["canvas"]
        widgets = [self.view_widgets[key] for key in selected]

        if len(widgets) == 1:
            root_widget = widgets[0]
        elif len(widgets) == 2:
            root_widget = self._build_view_splitter(Qt.Horizontal, widgets)
        else:
            top_splitter = self._build_view_splitter(Qt.Horizontal, widgets[:2])
            root_widget = self._build_view_splitter(Qt.Vertical, [top_splitter, widgets[2]])

        for widget in widgets:
            widget.show()

        self.display_root_widget = root_widget
        self.display_layout.addWidget(root_widget, 1)
        root_widget.show()

        self.update_view_mode_button_text()

        canvas_selected = "canvas" in selected
        controls_enabled = canvas_selected and not self.editor_only
        self.pause_button.setEnabled(canvas_selected)
        self.remove_button.setEnabled(canvas_selected)
        self.merge_button.setEnabled(canvas_selected)
        self.undo_button.setEnabled(canvas_selected)
        self.pause_action.setEnabled(controls_enabled)
        self.remove_action.setEnabled(canvas_selected)
        self.merge_action.setEnabled(canvas_selected)
        self.undo_action.setEnabled(canvas_selected)
        self.add_word_textbox.setEnabled(canvas_selected)
        self.add_text_button.setEnabled(canvas_selected)
        if self.editor_only:
            self.pause_button.setEnabled(False)
        self._refresh_content_editor_visibility()
        self.statusBar().showMessage(f"View: {self.view_mode_button.text()}")

    def update_view_mode_button_text(self):
        selected = [self.view_labels[key] for key in self.selected_view_keys]
        if len(selected) == 1:
            label = selected[0]
        elif len(selected) == 2:
            label = " + ".join(selected)
        else:
            label = f"{len(selected)} views"
        self.view_mode_button.setText(label)

    def update_original_frame(self, frame):
        self.original_frame_view.set_frame(frame)
        if self.editor_only:
            self._apply_editor_only_mode()

    def update_filtered_frame(self, frame):
        self.filtered_frame_view.set_frame(frame)

    def update_preview_frame(self, payload):
        mode = str((payload or {}).get("mode", ""))
        frame = (payload or {}).get("frame")
        if frame is None:
            return

        if mode == "original":
            self.update_original_frame(frame)
        elif mode == "filtered":
            self.update_filtered_frame(frame)

    def update_selection_metadata(self):
        selected_items = self.view.scene.selectedItems()
        self._update_region_override_controls()
        self.update_content_editor()
        if not selected_items:
            self.metadata_view.setPlainText("Select a region to inspect metadata.")
            return

        if len(selected_items) > 1:
            selected_ids = [str(getattr(item, "item_id", "?")) for item in selected_items[:6]]
            summary = (
                f"{len(selected_items)} regions selected: {', '.join(selected_ids)}\n"
                "Use Merge Selected to combine them into one manual region."
            )
            if self.metadata_view.toPlainText() != summary:
                self.metadata_view.setPlainText(summary)
            return

        item = selected_items[0]
        metadata = dict(getattr(item, "region_metadata", {}) or {})

        if not metadata:
            metadata = {
                "region_id": getattr(item, "item_id", None),
                "classification": getattr(item, "item_type", type(item).__name__),
                "status": "manual",
                "bbox": None,
            }

        current_mode = self.interaction_state.get_classification_mode()
        metadata["classification_mode"] = current_mode
        region_id = self._selected_region_id()
        region_override = self.interaction_state.get_region_label_override(region_id) if region_id is not None else None
        if region_override:
            metadata["region_label_override"] = region_override
        if current_mode in {"diagram", "text", "equation"}:
            metadata["classification"] = current_mode
            metadata["classification_source"] = "global_mode"
        elif region_override:
            metadata["classification"] = region_override
            metadata["classification_source"] = "manual_override"

        metadata_text = build_selection_metadata_text(metadata)
        if self.metadata_view.toPlainText() != metadata_text:
            self.metadata_view.setPlainText(metadata_text)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(1200, 800)
    sample_dir = REPO_ROOT / "assets" / "samples"
    image1 = cv2.imread(str(sample_dir / "input.jpg"))
    image2 = cv2.imread(str(sample_dir / "image2.png"))
    image3 = cv2.imread(str(sample_dir / "image3.png"))
    data1 = {"id":1,"type":"image","content":image1,"x":200,"y":300}
    data2 = {"id":2,"type":"image","content":image2,"x":200,"y":600}
    data3 = {"id":3,"type":"image","content":image3,"x":800,"y":300}
    data4 = {"id":4,"type":"text","content":"Hello World","size":16,"x":800,"y":500}
    data5 = {"id":5,"type":"latex","content":r"\frac{a}{b}","size":10,"x":800,"y":500}
    
    win.view.load_images(data1)
    win.view.load_images(data2)
    win.view.load_images(data3)
    win.view.load_images(data4)
    win.view.load_images(data5)
    win.show()
    sys.exit(app.exec())   


from PySide6.QtCore import QObject, Signal, Slot, QThread
from pipeline import run_orchestrator

class Worker(QObject):
    finished = Signal()
    load_content = Signal(object)
    remove_content = Signal(int)
    preview_frame = Signal(object)
    corners = None
    def __init__(self, corners, video_path, runtime_settings=None, interaction_state=None, debug_console=None, start_paused=False, source_info=None):
        super().__init__()
        self.corners = corners
        self.video_path = video_path
        self.source_info = dict(source_info or {})
        self.stop_event = Event()
        self.play_event = Event()
        self.play_event.set()
        if start_paused:
            self.play_event.clear()
        self.runtime_settings = runtime_settings
        self.interaction_state = interaction_state
        self.debug_console = debug_console
        self._refresh_lock = Lock()
        self._pending_region_refreshes = []

    def request_stop(self):
        self.stop_event.set()

    def set_paused(self, paused):
        if paused:
            self.play_event.clear()
        else:
            self.play_event.set()

    def request_region_refresh(self, region_id):
        try:
            region_id = int(region_id)
        except (TypeError, ValueError):
            return
        with self._refresh_lock:
            if region_id not in self._pending_region_refreshes:
                self._pending_region_refreshes.append(region_id)

    def drain_region_refreshes(self):
        with self._refresh_lock:
            pending = list(self._pending_region_refreshes)
            self._pending_region_refreshes.clear()
        return pending

    @Slot()
    def run(self):
        try:
            run_orchestrator(
                self.corners,
                self,
                video_path=self.video_path,
                stop_event=self.stop_event,
                show_debug=False,
                runtime_settings=self.runtime_settings,
                interaction_state=self.interaction_state,
                debug_console=self.debug_console,
                pause_event=self.play_event,
                vectorization_method=self.source_info.get("vectorization_method"),
                equation_ocr_model=self.source_info.get("equation_ocr_model"),
                refresh_request_getter=self.drain_region_refreshes,
            )
        except Exception:
            traceback.print_exc()
        finally:
            self.finished.emit()


from preprocessing.corner_selection import select_board_corners

def run_gui(video_path: str | int | dict | None = None,enable_gui = True):
    app = QApplication.instance()
    created_app = app is None
    if app is None:
        app = QApplication(sys.argv)

    pending_video_source = normalize_video_source(video_path)
    exit_code = 0
    runtime_settings = RuntimeSettingsStore()
    interaction_state = InteractionStateStore()
    debug_console = RunConsole(
        summary_interval_segments=max(
            1,
            int(ORCHESTRATION_SETTINGS.get("console_summary_interval_segments", 4)),
        ),
        verbose_ocr=bool(ORCHESTRATION_SETTINGS.get("verbose_region_ocr", False)),
    )
    pending_corners = None
    pending_dimensions = None

    while True:
        if pending_video_source is None:
            if enable_gui:
                source_options = discover_source_options()
                if not source_options:
                    QMessageBox.information(None, "No Sources Found", "No video files or camera sources were found.")
                    return 0
                picker = VideoSelectionDialog(source_options)
                picker.showMaximized()
                if picker.exec() != QDialog.Accepted or not picker.selected_source:
                    return 0
                pending_video_source = picker.selected_source
            else:
                legacy_default = REPO_ROOT / "pipeline" / "Test3.mp4"
                if legacy_default.exists():
                    pending_video_source = normalize_video_source(str(legacy_default))
                else:
                    video_files = discover_video_files()
                    if not video_files:
                        raise FileNotFoundError("No supported video files found in pipeline/videos or pipeline.")
                    pending_video_source = make_file_source(video_files[0])
            pending_corners = None
            pending_dimensions = None

        current_video_source = with_default_run_options(normalize_video_source(pending_video_source))
        current_source_name = source_display_name(current_video_source)

        if current_video_source["kind"] == "board":
            try:
                board_document = load_board_document(current_video_source["value"])
            except Exception as exc:
                if enable_gui:
                    QMessageBox.critical(None, "Open Saved Board Failed", f"Could not open saved board:\n{exc}")
                    pending_video_source = None
                    continue
                raise

            apply_interaction_state_snapshot(interaction_state, board_document.get("interaction_state"))
            board_width, board_height = infer_board_dimensions(
                board_document.get("width"),
                board_document.get("height"),
                items=board_document.get("items"),
                original_frame=board_document.get("original_frame"),
            )

            win = MainWindow(
                int(board_width),
                int(board_height),
                runtime_settings=runtime_settings,
                interaction_state=interaction_state,
                source_info=board_document.get("source_info") or current_video_source,
                editor_only=True,
                original_frame=board_document.get("original_frame"),
            )
            win.setWindowTitle(f"Board Editor - {board_document['board_name']}")
            win.set_board_document_context(board_document["board_path"], board_document["board_name"])
            win.selected_view_keys = list(board_document.get("selected_view_keys") or ["canvas"])
            win.view.load_board_snapshots(board_document.get("items", []), clear_existing=True)
            win._apply_editor_only_mode()
            win.apply_view_selection_layout()

            run_state = {
                "reselect_requested": False,
            }

            def return_to_video_library():
                run_state["reselect_requested"] = True
                win.close()

            win.video_library_button.clicked.connect(return_to_video_library)
            win.video_library_action.triggered.connect(return_to_video_library)

            if enable_gui:
                win.showMaximized()
                exit_code = app.exec()
            else:
                win.showMaximized()
                exit_code = app.exec()

            win.deleteLater()

            if run_state["reselect_requested"]:
                pending_video_source = None
                pending_corners = None
                pending_dimensions = None
                interaction_state.clear()
                continue

            break

        current_capture_value = source_capture_value(current_video_source)
        runtime_settings.update(
            text_ocr_backend=current_video_source.get("text_ocr_backend", DEFAULT_TEXT_OCR_BACKEND),
            text_ocr_model=current_video_source.get("text_ocr_model", DEFAULT_TEXT_OCR_MODEL),
            vectorization_method=current_video_source.get("vectorization_method", DEFAULT_VECTORIZATION_SELECTION),
            equation_ocr_model=current_video_source.get("equation_ocr_model", DEFAULT_EQUATION_OCR_MODEL),
        )
        if current_video_source["kind"] == "camera":
            runtime_settings.update(segment_kernel_width_frac=0.05)
        else:
            runtime_settings.update(segment_kernel_width_frac=0.025)

        if pending_corners is None or pending_dimensions is None:
            if current_video_source["kind"] == "camera":
                camera_resolution_option = runtime_settings.snapshot().camera_resolution_option
                cap, camera_capture_info, first_preview_frame = open_configured_camera_capture(
                    current_capture_value,
                    resolution_option=camera_resolution_option,
                )
                if not cap.isOpened():
                    cap.release()
                    if enable_gui:
                        QMessageBox.critical(None, "Camera Open Failed", f"Could not open {current_source_name}.")
                        pending_video_source = None
                        continue
                    raise RuntimeError(f"Could not open {current_source_name}.")

                actual_width = int(camera_capture_info.get("actual_width") or 0)
                actual_height = int(camera_capture_info.get("actual_height") or 0)
                if actual_width > 0 and actual_height > 0:
                    current_video_source["capture_width"] = actual_width
                    current_video_source["capture_height"] = actual_height
                    debug_console.info(
                        f"Live camera preview open at {actual_width}x{actual_height}. "
                        "Press Space or Enter to freeze a frame for corner selection. Press Esc to cancel."
                    )
                else:
                    debug_console.info(
                        "Live camera preview open. Press Space or Enter to freeze a frame for corner selection. Press Esc to cancel."
                    )
                preview_window = f"{current_source_name} Preview (Space to select corners)"
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(preview_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                first_frame = None
                while True:
                    if first_preview_frame is not None:
                        frame = first_preview_frame
                        ok = True
                        first_preview_frame = None
                    else:
                        ok, frame = cap.read()
                    if not ok:
                        break
                    cv2.imshow(preview_window, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (32, 13):
                        first_frame = frame.copy()
                        break
                    if key in (27, ord("q"), ord("Q")):
                        break
                cv2.destroyWindow(preview_window)
                cap.release()
                if first_frame is None:
                    debug_console.warn("Live preview cancelled before corner selection.")
                    pending_video_source = None if enable_gui else current_video_source
                    pending_corners = None
                    pending_dimensions = None
                    if enable_gui:
                        continue
                    return 0
            else:
                cap = cv2.VideoCapture(current_capture_value)
                ok, first_frame = cap.read()
                if not ok:
                    cap.release()
                    if enable_gui:
                        QMessageBox.critical(None, "Video Open Failed", f"Could not read source:\n{current_source_name}")
                        pending_video_source = None
                        continue
                    raise FileNotFoundError(f"Could not read source: {current_source_name}")
                cap.release()

            debug_console.info("Select the four board corners on the current source frame.")
            corners = select_board_corners(first_frame)
            if corners is None:
                debug_console.warn("Corner selection aborted.")
                pending_video_source = None if enable_gui else current_video_source
                pending_corners = None
                pending_dimensions = None
                if enable_gui:
                    continue
                return 0

            x, y = estimate_board_size(corners)
            pending_corners = corners
            pending_dimensions = (x, y)
            debug_console.info(f"Corners locked. Board space set to {x}x{y}.")

        x, y = pending_dimensions
        corners = pending_corners

        win = MainWindow(
            x,
            y,
            runtime_settings=runtime_settings,
            interaction_state=interaction_state,
            source_info=current_video_source,
        )
        win.setWindowTitle(f"Board - {current_source_name}")
        start_paused = current_video_source["kind"] == "file"

        thread = QThread()
        worker = Worker(
            corners,
            current_capture_value,
            runtime_settings=runtime_settings,
            interaction_state=interaction_state,
            debug_console=debug_console,
            start_paused=start_paused,
            source_info=current_video_source,
        )
        run_state = {
            "reselect_requested": False,
            "restart_requested": False,
            "close_pending": False,
        }

        worker.moveToThread(thread)

        def close_window_after_stop():
            if run_state["close_pending"] and win.isVisible():
                win.close()

        def return_to_video_library():
            if run_state["close_pending"]:
                return
            debug_console.info("Stopping current run and returning to the video library.")
            run_state["reselect_requested"] = True
            run_state["close_pending"] = True
            win.video_library_button.setEnabled(False)
            win.reset_run_button.setEnabled(False)
            worker.request_stop()
            if not thread.isRunning():
                close_window_after_stop()

        def reset_current_run():
            if run_state["close_pending"]:
                return
            debug_console.info("Restarting the current run from frame 0 with the existing board corners.")
            run_state["restart_requested"] = True
            run_state["close_pending"] = True
            win.video_library_button.setEnabled(False)
            win.reset_run_button.setEnabled(False)
            worker.request_stop()
            if not thread.isRunning():
                close_window_after_stop()

        thread.started.connect(worker.run)
        worker.load_content.connect(win.view.load_images)
        worker.remove_content.connect(win.view.expire_content)
        worker.preview_frame.connect(win.update_preview_frame)
        worker.finished.connect(thread.quit)
        thread.finished.connect(close_window_after_stop)
        win.video_library_button.clicked.connect(return_to_video_library)
        win.reset_run_button.clicked.connect(reset_current_run)
        win.video_library_action.triggered.connect(return_to_video_library)
        win.reset_run_action.triggered.connect(reset_current_run)
        win.set_pause_toggle_callback(worker.set_paused)
        win.set_region_refresh_callback(worker.request_region_refresh)
        if start_paused:
            win.set_playback_paused(True, initial_start=True, sync_canvas_queue=False)
        else:
            win.set_playback_paused(False, sync_canvas_queue=False)

        thread.start()
        if enable_gui:
            win.showMaximized()
            exit_code = app.exec()
        else:
            thread.finished.connect(app.quit)
            exit_code = app.exec()

        worker.request_stop()
        thread.quit()
        thread.wait(5000)
        win.deleteLater()

        if run_state["reselect_requested"]:
            pending_video_source = None
            pending_corners = None
            pending_dimensions = None
            interaction_state.clear()
            continue

        if run_state["restart_requested"]:
            continue

        break

    if created_app:
        sys.exit(exit_code)
    return exit_code




        
