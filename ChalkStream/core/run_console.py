from collections import Counter
from datetime import datetime
from pathlib import Path


class RunConsole:
    def __init__(self, enabled=True, summary_interval_segments=4, verbose_ocr=False):
        self.enabled = bool(enabled)
        self.summary_interval_segments = max(1, int(summary_interval_segments))
        self.verbose_ocr = bool(verbose_ocr)

    def _stamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def emit(self, level, message):
        if not self.enabled:
            return
        print(f"[{self._stamp()}] [{level}] {message}")

    def info(self, message):
        self.emit("INFO", message)

    def warn(self, message):
        self.emit("WARN", message)

    def error(self, message):
        self.emit("ERROR", message)

    def startup(self, video_path, source_fps, board_size, segment_interval_s, preview_interval_s, equation_ocr_url, equation_ocr_model):
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

    def segment_summary(self, segmentation_index, frame_idx, source_fps, regions, equation_ocr_state=None, merge_groups=None, force=False):
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
        verb = "updated" if replaced_best else "accepted"
        preview = str(latex or "").strip().replace("\n", " ")
        if len(preview) > 48:
            preview = preview[:45] + "..."
        self.emit(
            "EQOCR",
            f"region {region_id} {verb} at {float(confidence) * 100:.1f}% -> {preview}",
        )

    def equation_ocr_error(self, region_id, error_message):
        self.emit("EQOCR", f"region {region_id} failed: {error_message}")

    def ocr_trace(self, bbox, elapsed_s, label, confidence, text_preview):
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
        elapsed_s = float(frame_count) / max(float(source_fps), 1e-6)
        self.emit(
            "DONE",
            f"Processed {int(frame_count)} frames | segmentations {int(segmentation_index)} | source time {elapsed_s:.1f}s",
        )
