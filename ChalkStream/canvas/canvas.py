import sys
import subprocess
import traceback
import json
import re
import os
from pathlib import Path
from threading import Event
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
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
)
import cv2
import numpy as np
from PySide6.QtGui import QPainter
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QFontMetrics, QAction, QPen, QBrush, QKeySequence
from PySide6.QtCore import Qt, QTimer, QRectF
import matplotlib.mathtext as mathtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from core import InteractionStateStore, RunConsole, RuntimeSettingsStore

MATRIX_ENV_RE = re.compile(
    r"\\begin\{(?P<env>matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix)\}(?P<body>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)
DEFAULT_SEGMENT_INTERVAL_S = 0.5
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_LIBRARY_DIRS = (
    REPO_ROOT / "pipeline" / "videos",
    REPO_ROOT / "pipeline",
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
        "name": f"Use My Camera (Camera {int(camera_index)})",
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


def discover_source_options():
    sources = [make_camera_source(0)]
    sources.extend(make_file_source(video) for video in discover_video_files())
    return sources


def source_capture_value(source):
    return int(source["value"]) if str(source.get("kind")) == "camera" else str(source["value"])


def source_display_name(source):
    if str(source.get("kind")) == "camera":
        return f"Camera {int(source['value'])}"
    return str(source.get("name") or Path(str(source.get("value"))).name)


class VideoSelectionDialog(QDialog):
    def __init__(self, source_options, parent=None):
        super().__init__(parent)
        self.source_options = source_options
        self.selected_source = None

        self.setWindowTitle("Select Source Video")
        self.resize(720, 420)

        layout = QVBoxLayout(self)

        intro = QLabel("Choose a board recording, or use your camera for a live run. Preferred file location: pipeline/videos")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.video_list = QListWidget()
        layout.addWidget(self.video_list, 1)

        self.details_label = QLabel("Select a video to inspect its path and size.")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.start_button = QPushButton("Start Selected Source")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.accept_selection)
        button_row.addWidget(self.start_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        layout.addLayout(button_row)

        self.video_list.currentItemChanged.connect(self.update_details)
        self.video_list.itemDoubleClicked.connect(lambda _item: self.accept_selection())

        for source in self.source_options:
            if source["kind"] == "camera":
                label = f"{source['name']}    [Live]"
            else:
                badge = "Preferred" if source["preferred"] else "Legacy"
                label = f"{source['name']}    {humanize_bytes(source['size'])}    [{badge}]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, source)
            item.setToolTip(source["relative_path"])
            self.video_list.addItem(item)

        if self.video_list.count() > 0:
            self.video_list.setCurrentRow(0)
        else:
            self.details_label.setText("No supported video files were found, and no live source is available.")

    def update_details(self, current, _previous):
        source = current.data(Qt.UserRole) if current is not None else None
        self.start_button.setEnabled(source is not None)
        if source is None:
            self.details_label.setText("Select a source to inspect its details.")
            self.selected_source = None
            return

        self.selected_source = dict(source)
        if source["kind"] == "camera":
            self.details_label.setText(
                f"Source: Live camera\n"
                f"Device: Camera {int(source['value'])}\n"
                "The live preview will open first so you can freeze a frame for corner selection."
            )
        else:
            self.details_label.setText(
                f"Path: {source['relative_path']}\n"
                f"Size: {humanize_bytes(source['size'])}\n"
                f"Source: {'pipeline/videos' if source['preferred'] else 'pipeline'}"
            )

    def accept_selection(self):
        if self.selected_source is None:
            return
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
    def __init__(self, item_id, text,size = 16) -> None:
        super().__init__(text)
        self.item_id = item_id
        self.item_type = "text"
        self.render_size = size
        self.region_metadata = {}

        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable 
        )
        self._set_font_size(size)
        self.init_resize_behavior()

    def _set_font_size(self, size):
        self.render_size = size
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

    def update_content(self, new_text, size=None):

        old_pos = self.pos()
        if size is not None and size != self.render_size:
            self._set_font_size(size)
        self.setPlainText(new_text)
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

    def allocate_manual_id(self):
        manual_id = self.next_manual_id
        self.next_manual_id -= 1
        return manual_id

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
            item = TextItem(item_id, data["content"], data.get("size", 16) or 16)
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
    
    def load_images(self,data:dict):
        """
        Preload images into canvas
        """
        item_type = data.get("type")
        if item_type is None:
            raise KeyError("Canvas payload missing required key: 'type'")

        item_id = int(data["id"])

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
                item.update_content(data["content"], data.get("size", 16) or 16)
            elif item_type == "latex":
                item.update_content(
                    data["content"],
                    data.get("size", 16) or 16,
                    data.get("width"),
                    data.get("height"),
                )

            item.region_metadata = dict(data.get("metadata", {}))
            self._set_item_position(item, data)

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
        finally:
            self._history_suspended = False

        return False

    def add_elements(self,element):
        if isinstance(element,str):
            text_item = TextItem(self.allocate_manual_id(), element, 20)
            text_item.setPos(50,50)
            self.scene.addItem(text_item)
            self.contents[text_item.item_id] = text_item
        

class MainWindow(QMainWindow):

    def __init__(self,dimension_x,dimension_y, runtime_settings=None, interaction_state=None):
        super().__init__()
        self.runtime_settings = runtime_settings or RuntimeSettingsStore()
        self.interaction_state = interaction_state or InteractionStateStore()
        self.runtime_settings_visible = False

        self.setWindowTitle("Board")

        #Main interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.view = CanvasView(dimension_x - 50,dimension_y - 50, interaction_state=self.interaction_state)
        self.view.setMinimumSize(320, 240)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.scene.selectionChanged.connect(self.update_selection_metadata)

        self.original_frame_view = VideoFrameView("Original Board Feed")
        self.filtered_frame_view = VideoFrameView("Occlusion Removed Feed")

        self.display_container = QWidget()
        self.display_container.setMinimumSize(dimension_x, dimension_y)
        self.display_layout = QVBoxLayout()
        self.display_layout.setContentsMargins(0, 0, 0, 0)
        self.display_layout.setSpacing(8)
        self.display_container.setLayout(self.display_layout)
        main_layout.addWidget(self.display_container, 1)
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
        button_layout = QVBoxLayout()
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

        self.pause_button = QPushButton("Pause")
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

        self.kernel_width_spin = QDoubleSpinBox()
        self.kernel_width_spin.setRange(0.1, 10.0)
        self.kernel_width_spin.setDecimals(2)
        self.kernel_width_spin.setSingleStep(0.1)
        self.kernel_width_spin.setSuffix("%")
        settings_layout.addRow("Segment Kernel H", self.kernel_width_spin)

        self.kernel_height_spin = QDoubleSpinBox()
        self.kernel_height_spin.setRange(0.1, 10.0)
        self.kernel_height_spin.setDecimals(2)
        self.kernel_height_spin.setSingleStep(0.1)
        self.kernel_height_spin.setSuffix("%")
        settings_layout.addRow("Segment Kernel V", self.kernel_height_spin)

        self.segment_max_area_spin = QDoubleSpinBox()
        self.segment_max_area_spin.setRange(1.0, 95.0)
        self.segment_max_area_spin.setDecimals(1)
        self.segment_max_area_spin.setSingleStep(1.0)
        self.segment_max_area_spin.setSuffix("%")
        settings_layout.addRow("Segment Max Area", self.segment_max_area_spin)

        self.segment_occluder_area_spin = QDoubleSpinBox()
        self.segment_occluder_area_spin.setRange(5.0, 95.0)
        self.segment_occluder_area_spin.setDecimals(1)
        self.segment_occluder_area_spin.setSingleStep(1.0)
        self.segment_occluder_area_spin.setSuffix("%")
        settings_layout.addRow("Occluder Area", self.segment_occluder_area_spin)

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

        metadata_title = QLabel("Selection Metadata")
        button_layout.addWidget(metadata_title)

        self.metadata_view = QPlainTextEdit()
        self.metadata_view.setReadOnly(True)
        self.metadata_view.setMinimumWidth(320)
        self.metadata_view.setPlainText("Select a region to inspect metadata.")
        button_layout.addWidget(self.metadata_view)

        self._update_region_override_controls()

        button_layout.addStretch()

        main_layout.addWidget(button_panel) 

        self.resize(dimension_x + 100, dimension_y)
        self.metadata_refresh_timer = QTimer(self)
        self.metadata_refresh_timer.setInterval(250)
        self.metadata_refresh_timer.timeout.connect(self.update_selection_metadata)
        self.metadata_refresh_timer.start()
        self._connect_settings_controls()
        self._load_runtime_settings_into_controls()
        self.apply_view_selection_layout()
        self.statusBar().showMessage("Ready")

    def _setup_menu_bar(self):
        file_menu = self.menuBar().addMenu("&File")
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
        settings_menu.addSeparator()
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
                "Delete: Remove selected regions",
                "Space: Pause or resume",
                "Ctrl+R: Reset the current run",
                "Ctrl+L: Return to video library",
            ]),
        )

    def pause(self):
        self.view.pause()
        if self.view.paused:
            self.pause_button.setText("Resume")
            self.pause_action.setText("Resume")
        else:
            self.pause_button.setText("Pause")
            self.pause_action.setText("Pause")

    def remove_selected(self):
        self.view.remove_selected()

    def merge_selected(self):
        merged = self.view.merge_selected()
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

    def _connect_settings_controls(self):
        self.classification_mode_combo.currentIndexChanged.connect(self.on_classification_mode_changed)
        self.kernel_width_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(segment_kernel_width_frac=float(value) / 100.0)
        )
        self.kernel_height_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(segment_kernel_height_frac=float(value) / 100.0)
        )
        self.segment_max_area_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(segment_max_area_pct=float(value) / 100.0)
        )
        self.segment_occluder_area_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(segment_occluder_area_frac=float(value) / 100.0)
        )
        self.preprocess_history_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(preprocess_history=int(value))
        )
        self.preprocess_var_threshold_spin.valueChanged.connect(
            lambda value: self.runtime_settings.update(preprocess_var_threshold=float(value))
        )
        self.reset_settings_button.clicked.connect(self.reset_runtime_settings)
        self.region_override_combo.currentIndexChanged.connect(self.on_region_override_changed)

    def _load_runtime_settings_into_controls(self):
        snapshot = self.runtime_settings.snapshot()
        controls = (
            (self.kernel_width_spin, snapshot.segment_kernel_width_frac * 100.0),
            (self.kernel_height_spin, snapshot.segment_kernel_height_frac * 100.0),
            (self.segment_max_area_spin, snapshot.segment_max_area_pct * 100.0),
            (self.segment_occluder_area_spin, snapshot.segment_occluder_area_frac * 100.0),
            (self.preprocess_history_spin, snapshot.preprocess_history),
            (self.preprocess_var_threshold_spin, snapshot.preprocess_var_threshold),
        )
        for widget, value in controls:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._load_classification_mode_into_controls()

    def reset_runtime_settings(self):
        self.runtime_settings.reset_defaults()
        self._load_runtime_settings_into_controls()

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
        if override_value:
            self.statusBar().showMessage(f"Region {region_id} override set to {override_label}", 3000)
        else:
            self.statusBar().showMessage(f"Region {region_id} override cleared", 3000)
        self.update_selection_metadata()

    def set_runtime_settings_visible(self, visible):
        self.runtime_settings_visible = bool(visible)
        if hasattr(self, "settings_group"):
            self.settings_group.setVisible(self.runtime_settings_visible)
        if hasattr(self, "toggle_runtime_settings_action"):
            self.toggle_runtime_settings_action.blockSignals(True)
            self.toggle_runtime_settings_action.setChecked(self.runtime_settings_visible)
            self.toggle_runtime_settings_action.blockSignals(False)

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
        self.pause_button.setEnabled(canvas_selected)
        self.remove_button.setEnabled(canvas_selected)
        self.merge_button.setEnabled(canvas_selected)
        self.undo_button.setEnabled(canvas_selected)
        self.pause_action.setEnabled(canvas_selected)
        self.remove_action.setEnabled(canvas_selected)
        self.merge_action.setEnabled(canvas_selected)
        self.undo_action.setEnabled(canvas_selected)
        self.add_word_textbox.setEnabled(canvas_selected)
        self.add_text_button.setEnabled(canvas_selected)
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
    def __init__(self, corners, video_path, runtime_settings=None, interaction_state=None, debug_console=None):
        super().__init__()
        self.corners = corners
        self.video_path = video_path
        self.stop_event = Event()
        self.runtime_settings = runtime_settings
        self.interaction_state = interaction_state
        self.debug_console = debug_console

    def request_stop(self):
        self.stop_event.set()

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
            int(os.environ.get("CHALKSTREAM_CONSOLE_SUMMARY_INTERVAL_SEGMENTS", "4")),
        ),
        verbose_ocr=os.environ.get("CHALKSTREAM_VERBOSE_REGION_OCR", "").strip().lower() in {"1", "true", "yes", "on"},
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

        current_video_source = normalize_video_source(pending_video_source)
        current_capture_value = source_capture_value(current_video_source)
        current_source_name = source_display_name(current_video_source)

        if pending_corners is None or pending_dimensions is None:
            if current_video_source["kind"] == "camera":
                cap = cv2.VideoCapture(current_capture_value)
                if not cap.isOpened():
                    cap.release()
                    if enable_gui:
                        QMessageBox.critical(None, "Camera Open Failed", f"Could not open {current_source_name}.")
                        pending_video_source = None
                        continue
                    raise RuntimeError(f"Could not open {current_source_name}.")

                debug_console.info("Live camera preview open. Press Space or Enter to freeze a frame for corner selection. Press Esc to cancel.")
                preview_window = f"{current_source_name} Preview (Space to select corners)"
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
                first_frame = None
                while True:
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

            tl, tr, br, bl = corners
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            x = int(max(widthA, widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            y = int(max(heightA, heightB))
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
        )
        win.setWindowTitle(f"Board - {current_source_name}")

        thread = QThread()
        worker = Worker(
            corners,
            current_capture_value,
            runtime_settings=runtime_settings,
            interaction_state=interaction_state,
            debug_console=debug_console,
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

        thread.start()
        if enable_gui:
            win.show()
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




        
