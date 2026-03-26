"""Compact terminal reporter for long-running ChalkStream sessions."""

from collections import Counter, deque
from datetime import datetime
from pathlib import Path


class RunConsole:
    """Emit readable summaries without flooding the terminal on every frame."""

    def __init__(self, enabled=True, summary_interval_segments=4, verbose_ocr=False):
        self.enabled = bool(enabled)
        self.summary_interval_segments = max(1, int(summary_interval_segments))
        self.verbose_ocr = bool(verbose_ocr)
        self._throughput_start_time = None
        self._last_throughput_emit_time = None
        self._throughput_history = deque()

    def _stamp(self):
        """Return a short wall-clock timestamp for log lines."""
        return datetime.now().strftime("%H:%M:%S")

    def emit(self, level, message):
        """Print a formatted log line if console output is enabled."""
        if not self.enabled:
            return
        print(f"[{self._stamp()}] [{level}] {message}")

    def info(self, message):
        self.emit("INFO", message)

    def warn(self, message):
        self.emit("WARN", message)

    def error(self, message):
        self.emit("ERROR", message)

    def startup(
        self,
        video_path,
        source_fps,
        board_size,
        segment_interval_s,
        preview_interval_s,
        equation_ocr_url,
        equation_ocr_model,
        vectorization_method=None,
    ):
        """Emit the run configuration once at startup."""
        self._throughput_start_time = datetime.now().timestamp()
        self._last_throughput_emit_time = None
        self._throughput_history.clear()
        video_name = Path(str(video_path)).name
        width, height = board_size
        self.emit(
            "RUN",
            f"Video {video_name} | source {source_fps:.2f} fps | board {int(width)}x{int(height)}",
        )
        self.emit(
            "RUN",
            f"Segmentation {segment_interval_s:.2f}s | preview {preview_interval_s:.2f}s | equation OCR {equation_ocr_model} @ {equation_ocr_url}",
        )
        if vectorization_method:
            self.emit("RUN", f"Diagram vectorization {vectorization_method}")

    def throughput(self, now_s, preprocess_count, segmentation_count):
        """Emit preprocessing and segmentation throughput once per second."""
        if not self.enabled:
            return

        now_s = float(now_s)
        preprocess_count = int(preprocess_count)
        segmentation_count = int(segmentation_count)

        if self._throughput_start_time is None:
            self._throughput_start_time = now_s

        self._throughput_history.append((now_s, preprocess_count, segmentation_count))
        cutoff_time = now_s - 10.0
        while len(self._throughput_history) > 1 and self._throughput_history[0][0] < cutoff_time:
            self._throughput_history.popleft()

        if self._last_throughput_emit_time is not None and (now_s - self._last_throughput_emit_time) < 1.0:
            return

        total_elapsed = max(now_s - float(self._throughput_start_time), 1e-6)
        overall_preprocess_fps = preprocess_count / total_elapsed
        overall_segmentation_fps = segmentation_count / total_elapsed

        if len(self._throughput_history) >= 2:
            window_start_time, window_preprocess_count, window_segmentation_count = self._throughput_history[0]
            window_elapsed = max(now_s - float(window_start_time), 1e-6)
            rolling_preprocess_fps = (preprocess_count - int(window_preprocess_count)) / window_elapsed
            rolling_segmentation_fps = (segmentation_count - int(window_segmentation_count)) / window_elapsed
        else:
            rolling_preprocess_fps = overall_preprocess_fps
            rolling_segmentation_fps = overall_segmentation_fps

        self.emit(
            "RATE",
            (
                f"preprocess overall={overall_preprocess_fps:.2f}/s last10s={rolling_preprocess_fps:.2f}/s | "
                f"segment overall={overall_segmentation_fps:.2f}/s last10s={rolling_segmentation_fps:.2f}/s"
            ),
        )
        self._last_throughput_emit_time = now_s

    def segment_summary(self, segmentation_index, frame_idx, source_fps, regions, equation_ocr_state=None, merge_groups=None, force=False):
        """Print a periodic summary of tracked regions and OCR state."""
        if not force and (segmentation_index % self.summary_interval_segments) != 0:
            return

        merge_groups = list(merge_groups or [])
        equation_ocr_state = equation_ocr_state or {}
        visible = 0
        missing = 0
        labels = Counter()
        for region in regions or []:
            status = str(region.get("status", "existing")).lower()
            label = str(region.get("display_label", region.get("label", "diagram"))).lower()
            if label.startswith("equation"):
                label = "equation"
            elif label.startswith("text"):
                label = "text"
            else:
                label = "diagram"
            labels[label] += 1
            if status == "missing":
                missing += 1
            else:
                visible += 1

        eq_ready = sum(1 for state in equation_ocr_state.values() if state.get("render_ready"))
        eq_attempts = sum(int(state.get("attempt_count", 0) or 0) for state in equation_ocr_state.values())
        processing_time_s = float(frame_idx) / max(float(source_fps), 1e-6)

        self.emit(
            "SEG",
            (
                f"#{int(segmentation_index):04d} | t={processing_time_s:6.1f}s | "
                f"visible={visible} missing={missing} | "
                f"text={labels.get('text', 0)} eq={labels.get('equation', 0)} diag={labels.get('diagram', 0)} | "
                f"merges={len(merge_groups)} eqReady={eq_ready} eqAttempts={eq_attempts}"
            ),
        )

    def equation_ocr_accept(self, region_id, confidence, latex, replaced_best=False):
        """Report when equation OCR produces a new accepted result."""
        verb = "updated" if replaced_best else "accepted"
        preview = str(latex or "").strip().replace("\n", " ")
        if len(preview) > 48:
            preview = preview[:45] + "..."
        self.emit(
            "EQOCR",
            f"region {region_id} {verb} at {float(confidence) * 100:.1f}% -> {preview}",
        )

    def equation_ocr_error(self, region_id, error_message):
        """Report an equation OCR failure without interrupting the run."""
        self.emit("EQOCR", f"region {region_id} failed: {error_message}")

    def ocr_trace(self, bbox, elapsed_s, label, confidence, text_preview):
        """Optionally print per-region OCR traces when verbose mode is enabled."""
        if not self.verbose_ocr:
            return
        x, y, w, h = map(int, bbox)
        preview = str(text_preview or "").replace("\n", " ")
        if len(preview) > 48:
            preview = preview[:45] + "..."
        self.emit(
            "OCR",
            f"({x},{y},{w},{h}) {label} {float(confidence) * 100:.1f}% {elapsed_s:.3f}s \"{preview}\"",
        )

    def done(self, frame_count, segmentation_index, source_fps):
        """Emit a final summary once processing stops."""
        elapsed_s = float(frame_count) / max(float(source_fps), 1e-6)
        self.emit(
            "DONE",
            f"Processed {int(frame_count)} frames | segmentations {int(segmentation_index)} | source time {elapsed_s:.1f}s",
        )
