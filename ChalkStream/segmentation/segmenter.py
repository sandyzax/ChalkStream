"""Persistent region tracking, lightweight OCR, and heuristic classification."""

from datetime import datetime
import time
import cv2
import os
import json
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from paddleocr import TextRecognition, TextDetection, PaddleOCR
import numpy
from diagram_vectorization.content_presentation import Content, ContentType
from core import load_app_settings


SEGMENTATION_SETTINGS = load_app_settings().get("segmentation", {})

# Constants defining segmentation thresholds and morphological grouping rules

MIN_AREA_PCT = float(SEGMENTATION_SETTINGS.get("min_area_pct", 0.0005))
MAX_AREA_PCT = float(SEGMENTATION_SETTINGS.get("max_area_pct", 0.50))
#MORPH_KERNEL = (15, 15)     # Morphological kernel size for region grouping (now dynamic)
OCR_INTERVAL = float(SEGMENTATION_SETTINGS.get("ocr_interval_s", 5.0))
TEXT_OCR_BACKEND = str(SEGMENTATION_SETTINGS.get("text_ocr_backend", "paddle")).strip().lower()
TEXT_OCR_URL = str(SEGMENTATION_SETTINGS.get("text_ocr_url", "http://127.0.0.1:8010/infer_text"))
TEXT_OCR_MODEL = str(SEGMENTATION_SETTINGS.get("text_ocr_model", "TextOCR_V1"))
CHANGE_THRESHOLD = float(SEGMENTATION_SETTINGS.get("change_threshold", 25))
# Only OCR once region is persistent
NEW_REGION_MIN_AGE_S = float(SEGMENTATION_SETTINGS.get("new_region_min_age_s", 1.0))
NEW_REGION_MIN_HITS = int(SEGMENTATION_SETTINGS.get("new_region_min_hits", 2))

# Consecutive frames required before showing non-diagram label
STABLE_LABEL_MIN_AGE = int(SEGMENTATION_SETTINGS.get("stable_label_min_age", 5))

# Consecutive frames required before switching an already-displayed label (debounce/hysteresis)
FLIP_CONFIRM_FRAMES = int(SEGMENTATION_SETTINGS.get("flip_confirm_frames", 5))

# If a stable text/equation region temporarily disappears (e.g., screen blow-up), keep it alive
KEEPALIVE_MIN_AGE = int(SEGMENTATION_SETTINGS.get("keepalive_min_age", 5))
DIAGRAM_KEEPALIVE_MIN_AGE = int(SEGMENTATION_SETTINGS.get("diagram_keepalive_min_age", 2))
MISSING_CONFIRM_FRAMES = int(SEGMENTATION_SETTINGS.get("missing_confirm_frames", 5))
RECOVERY_GRACE_FRAMES = int(SEGMENTATION_SETTINGS.get("recovery_grace_frames", 4))
MAX_MISSING_CONFIRM_FRAMES = int(SEGMENTATION_SETTINGS.get("max_missing_confirm_frames", 12))
MISSING_CONFIRM_AGE_STEP = int(SEGMENTATION_SETTINGS.get("missing_confirm_age_step", 4))
MISSING_CONFIRM_SEEN_STEP = int(SEGMENTATION_SETTINGS.get("missing_confirm_seen_step", 8))

# Ignore "monster" regions that cover most of the screen (often transient camera/zoom artifacts)
MONSTER_AREA_FRAC = float(SEGMENTATION_SETTINGS.get("monster_area_frac", 0.85))

# Matching and occlusion handling
MATCH_IOU_THRESH = float(SEGMENTATION_SETTINGS.get("match_iou_thresh", 0.35))
RECOVERY_CENTER_DIST_FRAC = float(SEGMENTATION_SETTINGS.get("recovery_center_dist_frac", 0.75))
RECOVERY_AREA_RATIO_MIN = float(SEGMENTATION_SETTINGS.get("recovery_area_ratio_min", 0.18))
RECOVERY_ASPECT_RATIO_MIN = float(SEGMENTATION_SETTINGS.get("recovery_aspect_ratio_min", 0.20))
RECOVERY_SCORE_MIN = float(SEGMENTATION_SETTINGS.get("recovery_score_min", 0.45))

# Stable regions recover against a trusted anchor bbox so brief lighting/static artifacts
# do not replace the region reference or spawn duplicates nearby.
ANCHOR_UPDATE_IOU_MIN = float(SEGMENTATION_SETTINGS.get("anchor_update_iou_min", 0.45))
ANCHOR_UPDATE_CENTER_DIST_FRAC = float(SEGMENTATION_SETTINGS.get("anchor_update_center_dist_frac", 0.35))
ANCHOR_UPDATE_AREA_RATIO_MIN = float(SEGMENTATION_SETTINGS.get("anchor_update_area_ratio_min", 0.50))
ANCHOR_UPDATE_ASPECT_RATIO_MIN = float(SEGMENTATION_SETTINGS.get("anchor_update_aspect_ratio_min", 0.45))
ANCHOR_SMOOTHING = float(SEGMENTATION_SETTINGS.get("anchor_smoothing", 0.35))

# If an unmatched candidate strongly overlaps the anchor of an already tracked region,
# suppress it instead of creating a new region ID.
DUPLICATE_SUPPRESSION_IOU = float(SEGMENTATION_SETTINGS.get("duplicate_suppression_iou", 0.50))
DUPLICATE_SUPPRESSION_CENTER_DIST_FRAC = float(SEGMENTATION_SETTINGS.get("duplicate_suppression_center_dist_frac", 0.30))
DUPLICATE_SUPPRESSION_AREA_RATIO_MIN = float(SEGMENTATION_SETTINGS.get("duplicate_suppression_area_ratio_min", 0.55))
DUPLICATE_SUPPRESSION_ASPECT_RATIO_MIN = float(SEGMENTATION_SETTINGS.get("duplicate_suppression_aspect_ratio_min", 0.45))

# Merge highly-overlapping candidate boxes before tracking.
BOX_MERGE_OVERLAP_MIN = float(SEGMENTATION_SETTINGS.get("box_merge_overlap_min", 0.60))
BOX_MERGE_AREA_RATIO_MIN = float(SEGMENTATION_SETTINGS.get("box_merge_area_ratio_min", 0.18))
BOX_MERGE_CONTAINMENT_MIN = float(SEGMENTATION_SETTINGS.get("box_merge_containment_min", 0.98))

# Treat large/overlapping blobs as occluders during "screen blow-up" and do not treat as content
OCCLUDER_AREA_FRAC = float(SEGMENTATION_SETTINGS.get("occluder_area_frac", 0.45))
OCCLUDER_OVERLAP_COUNT = int(SEGMENTATION_SETTINGS.get("occluder_overlap_count", 3))
OCCLUDER_IOU_MIN = float(SEGMENTATION_SETTINGS.get("occluder_iou_min", 0.12))

# If an older region becomes almost fully contained by a larger active region,
# retire the stale track instead of keeping both references around.
ABSORB_CONTAINMENT_MIN = float(SEGMENTATION_SETTINGS.get("absorb_containment_min", 0.98))
from paddleocr import PaddleOCR





class Segmenter:
    def __init__(self, use_english_model =True, epsilon=MATCH_IOU_THRESH, logger=None, verbose_ocr=False, log_saved_results=False):
        # Initialize the segmenter and load OCR model; retains state between frames
        self.use_english_model = use_english_model
        self.match_iou_thresh = min(0.95, max(0.01, float(epsilon)))
        self.epsilon = self.match_iou_thresh
        self.prev_regions = []  # Cached regions from previous frame
        self.prev_img = None   # Cached image from previous frame
        self._next_region_id = 1
        self.recovery_frames_remaining = 0
        self.close_kernel_width_frac = 0.025
        self.close_kernel_height_frac = 0.025
        self.max_area_pct = MAX_AREA_PCT
        self.occluder_area_frac = OCCLUDER_AREA_FRAC
        self.logger = logger
        self.verbose_ocr = bool(verbose_ocr)
        self.log_saved_results = bool(log_saved_results)
        self.text_ocr_backend = TEXT_OCR_BACKEND
        self.text_ocr_url = TEXT_OCR_URL
        self.text_ocr_model = TEXT_OCR_MODEL

        # Model configuration and paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_base = os.path.join(BASE_DIR, "ocr", "models", "rec")

        if self.use_english_model:
            self.model_name = "en_PP-OCRv5_mobile_rec_infer"
        else:
            self.model_name = "PP-OCRv5_mobile_rec_infer"

        self.model_url = (
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
            f"paddle3.0.0/{self.model_name}.tar"
        )

        self.model_path = os.path.join(self.model_base, self.model_name)
        self.recognizer = None
        if self.text_ocr_backend == "paddle":
            self._ensure_local_recognizer()

    def _emit_log(self, level, message):
        if self.logger is not None and hasattr(self.logger, "emit"):
            self.logger.emit(level, message)
        else:
            print(f"[{level}] {message}")

    def _log_info(self, message):
        self._emit_log("INFO", message)

    def _log_error(self, message):
        self._emit_log("ERROR", message)

    def apply_runtime_settings(self, settings_snapshot=None):
        if settings_snapshot is None:
            self.close_kernel_width_frac = 0.025
            self.close_kernel_height_frac = 0.025
            self.match_iou_thresh = MATCH_IOU_THRESH
            self.epsilon = self.match_iou_thresh
            self.text_ocr_backend = TEXT_OCR_BACKEND
            self.text_ocr_model = TEXT_OCR_MODEL
            return

        self.close_kernel_width_frac = max(
            0.001,
            float(getattr(settings_snapshot, "segment_kernel_width_frac", 0.025)),
        )
        self.close_kernel_height_frac = max(
            0.001,
            float(getattr(settings_snapshot, "segment_kernel_height_frac", 0.025)),
        )
        self.match_iou_thresh = min(
            0.95,
            max(0.01, float(getattr(settings_snapshot, "segment_match_iou_thresh", MATCH_IOU_THRESH))),
        )
        self.epsilon = self.match_iou_thresh
        self.text_ocr_backend = str(getattr(settings_snapshot, "text_ocr_backend", TEXT_OCR_BACKEND)).strip().lower()
        self.text_ocr_model = str(getattr(settings_snapshot, "text_ocr_model", TEXT_OCR_MODEL))
        if self.text_ocr_backend == "paddle":
            self._ensure_local_recognizer()


    def segment_regions_only(self, img):
        """
        Return only bounding boxes for regions.
        No OCR. No cropping. No NumPy arrays.
        Output: list of (x, y, w, h)
        """
        binary = self._preprocess(img)
        regions = self._find_regions(binary, img.shape)
        return regions


    def _ensure_model(self):
        # Download and extract OCR model if not already cached
        if os.path.exists(os.path.join(self.model_path, "inference.pdiparams")):
            return
        os.makedirs(self.model_base, exist_ok=True)
        tar_path = os.path.join(self.model_base, os.path.basename(self.model_url))
        self._log_info(f"Downloading OCR model from {self.model_url} ...")
        urllib.request.urlretrieve(self.model_url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self.model_base)
        os.remove(tar_path)
        self._log_info(f"Model extracted to {self.model_base}")

    def _ensure_local_recognizer(self):
        if self.recognizer is not None:
            return self.recognizer
        self._ensure_model()
        self.recognizer = self._load_recognizer()
        return self.recognizer

    def _load_recognizer(self):
        self._log_info("Loading PaddleOCR model...")

        if self.use_english_model:
            model_type = "en_PP-OCRv5_mobile_rec"
            self.model_name = "en_PP-OCRv5_mobile_rec_infer"
        else:
            model_type = "PP-OCRv5_mobile_rec"
            self.model_name = "PP-OCRv5_mobile_rec_infer"

        self._log_info(f"Selected OCR model: {self.model_name}")

        recognizer = TextRecognition(
            model_name=model_type,
            device="gpu"
        )

        self._log_info("OCR model loaded successfully.")
        return recognizer

    def segment_frame(self, image_path, output_json=None, output_img=None):
        # Process one image frame: detect, classify, and track regions over time
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Preprocess image to binary form and find region bounding boxes
        binary = self._preprocess(img)
        boxes = self._find_regions(binary, img.shape)

        # If this is the first frame, mark all regions as new
        if not self.prev_regions:
            self._log_info(f"Initial frame: {len(boxes)} regions detected.")
            scored = self._score_regions(img, boxes, status="new")
        else:
            # For subsequent frames, compare each region against previous frame
            self._log_info(f"Comparing {len(boxes)} regions to previous frame...")
            scored = self._compare_and_score(img, boxes)

        # Cache the regions for use in the next frame comparison
        self.prev_regions = scored

        # Save outputs to disk
        if output_json:
            self._save_results(scored, output_json)
        if output_img:
            vis = self._visualize(img, scored)
            cv2.imwrite(output_img, vis)

        return scored

    def segment_dict_to_content_record(seg, content_id):
        return Content(
            id=content_id,
            bbox=seg["bbox"],
            content_type=ContentType.DIAGRAM,  # forced for now
            confidence=seg.get("confidence", 0.0),
            text_preview=seg.get("text_preview", ""),
            status=seg.get("status", "new"),
            timestamp=seg["timestamp"],
        )


    def segment_array(self, img, save_json=None, save_vis=None):
        if img is None:
            raise ValueError("segment_array received None image")

        binary = self._preprocess(img)
        boxes = self._find_regions(binary, img.shape)

        if not self.prev_regions:
            scored = self._score_regions(img, boxes, status="new")
        else:
            scored = self._compare_and_score(img, boxes)

        self.prev_regions = scored
        self.prev_img = img.copy()

        # Visualization
        vis = self._visualize(img, scored)

        if save_json:
            self._save_results(scored, save_json)
        if save_vis:
            cv2.imwrite(save_vis, vis)

        return vis


    def _preprocess(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove board shadows (illumination normalization)

        # Stretch contrast (normalizes whiteboard brightness)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Light smoothing
        blur = cv2.GaussianBlur(norm, (5, 5), 0)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 12   # slightly stronger constant helps eliminate shadows
        )

        # Dynamic morphological closing
        h, w = img.shape[:2]
        kernel_w = max(3, int(round(w * self.close_kernel_width_frac)))
        kernel_h = max(3, int(round(h * self.close_kernel_height_frac)))
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))

        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closed

    def _find_regions(self, binary, img_shape):
        # Detect contours representing candidate written areas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img_shape[:2]
        total_pixels = w * h

        # Compute dynamic thresholds
        min_area = total_pixels * MIN_AREA_PCT
        max_area = total_pixels * self.max_area_pct

        boxes = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh

            # Filter out noise and extreme bounding boxes
            if area < min_area:
                continue
            if area > max_area:
                continue

            boxes.append((x, y, bw, bh))

        merge_gap_x = max(3, int(round(w * self.close_kernel_width_frac)))
        merge_gap_y = max(3, int(round(h * self.close_kernel_height_frac)))
        return self._merge_candidate_boxes(
            boxes,
            proximity_x=merge_gap_x,
            proximity_y=merge_gap_y,
        )

    def _merge_candidate_boxes(self, boxes, proximity_x=0, proximity_y=0):
        if len(boxes) < 2:
            return boxes

        def intersection_area(box_a, box_b):
            ax, ay, aw, ah = map(int, box_a)
            bx, by, bw, bh = map(int, box_b)
            x1 = max(ax, bx)
            y1 = max(ay, by)
            x2 = min(ax + aw, bx + bw)
            y2 = min(ay + ah, by + bh)
            return max(0, x2 - x1) * max(0, y2 - y1)

        def union_box(box_a, box_b):
            ax, ay, aw, ah = map(int, box_a)
            bx, by, bw, bh = map(int, box_b)
            x1 = min(ax, bx)
            y1 = min(ay, by)
            x2 = max(ax + aw, bx + bw)
            y2 = max(ay + ah, by + bh)
            return (x1, y1, x2 - x1, y2 - y1)

        def should_merge(box_a, box_b):
            ax, ay, aw, ah = map(int, box_a)
            bx, by, bw, bh = map(int, box_b)
            inter = intersection_area(box_a, box_b)
            if inter > 0:
                area_a = max(1, int(box_a[2]) * int(box_a[3]))
                area_b = max(1, int(box_b[2]) * int(box_b[3]))
                overlap_over_min = inter / float(min(area_a, area_b))
                area_ratio = min(area_a, area_b) / float(max(area_a, area_b))

                if overlap_over_min >= BOX_MERGE_CONTAINMENT_MIN:
                    return True

                return (
                    overlap_over_min >= BOX_MERGE_OVERLAP_MIN
                    and area_ratio >= BOX_MERGE_AREA_RATIO_MIN
                )

            horizontal_gap = max(0, max(bx - (ax + aw), ax - (bx + bw)))
            vertical_gap = max(0, max(by - (ay + ah), ay - (by + bh)))
            overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
            min_w = max(1, min(aw, bw))
            min_h = max(1, min(ah, bh))

            horizontally_aligned = overlap_y >= int(round(0.20 * min_h))
            vertically_aligned = overlap_x >= int(round(0.20 * min_w))

            if proximity_x > 0 and horizontal_gap <= proximity_x and horizontally_aligned:
                return True
            if proximity_y > 0 and vertical_gap <= proximity_y and vertically_aligned:
                return True

            return False

        merged = [tuple(map(int, box)) for box in boxes]
        changed = True

        while changed:
            changed = False
            next_boxes = []
            used = [False] * len(merged)

            for i, box in enumerate(merged):
                if used[i]:
                    continue

                current = box
                used[i] = True
                merged_this_round = True

                while merged_this_round:
                    merged_this_round = False
                    for j, other in enumerate(merged):
                        if used[j]:
                            continue
                        if should_merge(current, other):
                            current = union_box(current, other)
                            used[j] = True
                            merged_this_round = True
                            changed = True

                next_boxes.append(current)

            merged = next_boxes

        return merged

    def _bbox_match_metrics(self, bbox_a, bbox_b):
        ax, ay, aw, ah = map(int, bbox_a)
        bx, by, bw, bh = map(int, bbox_b)

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax + aw, bx + bw)
        inter_y2 = min(ay + ah, by + bh)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(1, aw * ah)
        area_b = max(1, bw * bh)
        union_area = area_a + area_b - inter_area
        iou = inter_area / float(union_area + 1e-6)

        acx = ax + (aw / 2.0)
        acy = ay + (ah / 2.0)
        bcx = bx + (bw / 2.0)
        bcy = by + (bh / 2.0)
        center_dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
        diag_norm = max(
            1.0,
            ((aw ** 2 + ah ** 2) ** 0.5),
            ((bw ** 2 + bh ** 2) ** 0.5),
        )
        center_dist_frac = center_dist / diag_norm

        area_ratio = min(area_a, area_b) / float(max(area_a, area_b))

        aspect_a = aw / float(max(1, ah))
        aspect_b = bw / float(max(1, bh))
        aspect_ratio = min(aspect_a, aspect_b) / float(max(aspect_a, aspect_b, 1e-6))

        return iou, center_dist_frac, area_ratio, aspect_ratio

    def _containment_overlap(self, bbox_a, bbox_b):
        ax, ay, aw, ah = map(int, bbox_a)
        bx, by, bw, bh = map(int, bbox_b)
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax + aw, bx + bw)
        inter_y2 = min(ay + ah, by + bh)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        min_area = max(1, min(max(1, aw * ah), max(1, bw * bh)))
        return inter_area / float(min_area)

    def _reference_bbox(self, region):
        anchor_bbox = region.get("anchor_bbox")
        if anchor_bbox and len(anchor_bbox) == 4:
            return [int(v) for v in anchor_bbox]
        return [int(v) for v in region.get("bbox", (0, 0, 0, 0))]

    def _keepalive_min_age(self, region):
        display_label = str(region.get("display_label", region.get("label", "diagram")))
        return DIAGRAM_KEEPALIVE_MIN_AGE if display_label == "diagram" else KEEPALIVE_MIN_AGE

    def _missing_confirm_limit(self, region):
        base = MISSING_CONFIRM_FRAMES
        keepalive_min_age = self._keepalive_min_age(region)
        age = max(0, int(region.get("age", 1)) - keepalive_min_age)
        seen_count = max(0, int(region.get("seen_count", 1)) - keepalive_min_age)
        bonus = (age // MISSING_CONFIRM_AGE_STEP) + (seen_count // MISSING_CONFIRM_SEEN_STEP)
        return min(MAX_MISSING_CONFIRM_FRAMES, base + bonus)

    def _smoothed_bbox(self, prev_bbox, new_bbox, smoothing=ANCHOR_SMOOTHING):
        if not prev_bbox or len(prev_bbox) != 4:
            return [int(v) for v in new_bbox]

        smoothing = min(1.0, max(0.0, float(smoothing)))
        inv = 1.0 - smoothing
        return [
            int(round((float(prev_bbox[idx]) * inv) + (float(new_bbox[idx]) * smoothing)))
            for idx in range(4)
        ]

    def _should_refresh_anchor_bbox(self, prev_region, new_bbox):
        ref_bbox = self._reference_bbox(prev_region)
        iou, center_dist_frac, area_ratio, aspect_ratio = self._bbox_match_metrics(ref_bbox, new_bbox)
        return (
            iou >= ANCHOR_UPDATE_IOU_MIN
            or (
                center_dist_frac <= ANCHOR_UPDATE_CENTER_DIST_FRAC
                and area_ratio >= ANCHOR_UPDATE_AREA_RATIO_MIN
                and aspect_ratio >= ANCHOR_UPDATE_ASPECT_RATIO_MIN
            )
        )

    def _next_anchor_bbox(self, prev_region, new_bbox):
        if not self._should_refresh_anchor_bbox(prev_region, new_bbox):
            return self._reference_bbox(prev_region)
        return self._smoothed_bbox(self._reference_bbox(prev_region), new_bbox)

    def _is_duplicate_candidate(self, prev_region, new_bbox):
        ref_bbox = self._reference_bbox(prev_region)
        iou, center_dist_frac, area_ratio, aspect_ratio = self._bbox_match_metrics(ref_bbox, new_bbox)
        return (
            iou >= DUPLICATE_SUPPRESSION_IOU
            or (
                center_dist_frac <= DUPLICATE_SUPPRESSION_CENTER_DIST_FRAC
                and area_ratio >= DUPLICATE_SUPPRESSION_AREA_RATIO_MIN
                and aspect_ratio >= DUPLICATE_SUPPRESSION_ASPECT_RATIO_MIN
            )
        )

    def _should_suppress_duplicate_new_region(self, new_bbox, used_prev_idxs):
        for idx, prev in enumerate(self.prev_regions):
            if not self._is_recoverable_region(prev):
                continue

            missing_streak = int(prev.get("missing_streak", 0))
            if missing_streak <= 0 and idx not in used_prev_idxs:
                continue

            if self._is_duplicate_candidate(prev, new_bbox):
                return True

        return False

    def _is_absorbed_by_active_region(self, prev_region, active_regions):
        prev_bbox = self._reference_bbox(prev_region)
        prev_area = max(1, int(prev_bbox[2]) * int(prev_bbox[3]))

        for active in active_regions:
            active_bbox = active.get("bbox")
            if not active_bbox or len(active_bbox) != 4:
                continue

            active_area = max(1, int(active_bbox[2]) * int(active_bbox[3]))
            if active_area < prev_area:
                continue

            containment = self._containment_overlap(prev_bbox, active_bbox)
            if containment >= ABSORB_CONTAINMENT_MIN:
                return True

        return False

    def _is_recoverable_region(self, region):
        display_label = str(region.get("display_label", region.get("label", "diagram")))
        min_age = self._keepalive_min_age(region)
        return display_label in ("text", "equation", "diagram") and int(region.get("age", 1)) >= min_age

    def _recovery_score(self, prev_region, new_bbox):
        if not self._is_recoverable_region(prev_region):
            return None

        missing_streak = int(prev_region.get("missing_streak", 0))
        if missing_streak <= 0 and self.recovery_frames_remaining <= 0:
            return None

        iou, center_dist_frac, area_ratio, aspect_ratio = self._bbox_match_metrics(
            self._reference_bbox(prev_region),
            new_bbox,
        )
        if center_dist_frac > RECOVERY_CENTER_DIST_FRAC:
            return None
        if area_ratio < RECOVERY_AREA_RATIO_MIN:
            return None
        if aspect_ratio < RECOVERY_ASPECT_RATIO_MIN:
            return None

        score = (
            0.55 * area_ratio
            + 0.30 * aspect_ratio
            + 0.15 * max(0.0, 1.0 - (center_dist_frac / RECOVERY_CENTER_DIST_FRAC))
            + 0.50 * iou
        )
        if score < RECOVERY_SCORE_MIN:
            return None
        return score

    def _compare_and_score(self, img, new_boxes):
        results = []
        now = time.time()
        recovered_region_confirmed = False

        # Reset occluder flag for this segmentation pass
        self.occluder_present = False

        # Use previous frame for change detection (if available)
        prev_img = getattr(self, "prev_img", None)

        used_prev_idxs = set()

        for (x, y, w, h) in new_boxes:
            x, y, w, h = int(x), int(y), int(w), int(h)
            new_bbox = [x, y, w, h]

            # Ignore extremely large "monster" regions (often transient zoom/blow-up artifacts)
            try:
                img_h, img_w = img.shape[:2]
                if (w * h) > (MONSTER_AREA_FRAC * (img_w * img_h)):
                    continue
            except Exception:
                pass

            # Occluder suppression:
            # During "screen blow-up" events, segmentation can produce large blobs that overlap many existing regions.
            # Treat these as occluders (instructor/zoom artifact), not real content.
            try:
                img_h, img_w = img.shape[:2]
                area_frac = (w * h) / float(img_w * img_h + 1e-6)

                overlap_count = 0
                for prev in self.prev_regions:
                    px, py, pw, ph = map(int, prev.get("bbox", (0, 0, 0, 0)))
                    # IoU with prev
                    ix1 = max(x, px); iy1 = max(y, py)
                    ix2 = min(x + w, px + pw); iy2 = min(y + h, py + ph)
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    if inter <= 0:
                        continue
                    union = (w * h) + (pw * ph) - inter
                    iou_p = inter / float(union + 1e-6)
                    if iou_p >= OCCLUDER_IOU_MIN:
                        overlap_count += 1
                        if overlap_count >= OCCLUDER_OVERLAP_COUNT:
                            break

                if area_frac >= self.occluder_area_frac and overlap_count >= OCCLUDER_OVERLAP_COUNT:
                    self.occluder_present = True
                    # Skip creating a content region for this blob
                    continue
            except Exception:
                pass

            # -----------------------------
            # Match to previous region by best IoU
            # -----------------------------
            best_iou = 0.0
            best_prev = None
            best_prev_idx = None
            best_recovery_score = -1.0
            best_recovery_prev = None
            best_recovery_idx = None

            for idx, prev in enumerate(self.prev_regions):
                if idx in used_prev_idxs:
                    continue

                iou, _, _, _ = self._bbox_match_metrics(prev["bbox"], new_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_prev = prev
                    best_prev_idx = idx

                recovery_score = self._recovery_score(prev, new_bbox)
                if recovery_score is not None and recovery_score > best_recovery_score:
                    best_recovery_score = recovery_score
                    best_recovery_prev = prev
                    best_recovery_idx = idx

            matched_region = None
            matched_prev_idx = None
            if best_prev is not None and best_iou >= self.match_iou_thresh:
                matched_region = best_prev
                matched_prev_idx = best_prev_idx
            elif best_recovery_prev is not None:
                matched_region = best_recovery_prev
                matched_prev_idx = best_recovery_idx

            if matched_prev_idx is not None:
                used_prev_idxs.add(matched_prev_idx)

            # -----------------------------
            # Case 1: New region => DO NOT OCR YET
            # -----------------------------
            if matched_region is None:
                if self._should_suppress_duplicate_new_region(new_bbox, used_prev_idxs):
                    continue

                results.append({
                    "bbox": new_bbox,
                    "anchor_bbox": list(new_bbox),
                    "region_id": self._next_region_id,
                    "raw_label": "diagram",      # classifier label (ungated)
                    "label": "diagram",          # gated label shown downstream
                    "age": 1,                 # consecutive frames with same raw_label
                    "display_label": "diagram",
                    "flip_streak": 0,
                    "missing_streak": 0,
                    "confidence": 0.0,
                    "text_preview": "",
                    "status": "new_pending",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "ocr_time_s": None,
                    "last_ocr_time": 0.0,        # 0 means never
                    "first_seen_time": now,
                    "seen_count": 1,
                    "has_ocr": False,
                })
                self._next_region_id += 1
                continue

            # -----------------------------
            # Case 2: Existing region => update persistence + decide whether to OCR
            # -----------------------------
            result = matched_region.copy()
            result["bbox"] = new_bbox
            result["anchor_bbox"] = self._next_anchor_bbox(matched_region, new_bbox)
            result["status"] = "existing"
            result["missing_streak"] = 0
            result.setdefault("first_seen_time", now)    # in case old state lacks it
            result["seen_count"] = int(result.get("seen_count", 1)) + 1
            result.setdefault("has_ocr", bool(result.get("last_ocr_time", 0.0)))
            result.setdefault("raw_label", str(result.get("raw_label", result.get("label", "diagram"))))
            result.setdefault("age", 1)
            result.setdefault("display_label", str(result.get("display_label", result.get("label", "diagram"))))
            result.setdefault("flip_streak", 0)
            result.setdefault("missing_streak", 0)

            if int(matched_region.get("missing_streak", 0)) > 0:
                recovered_region_confirmed = True

            age = now - float(result.get("first_seen_time", now))
            has_ocr = bool(result.get("has_ocr", False))

            # Gate the FIRST OCR until region is persistent
            if not has_ocr:
                if age >= NEW_REGION_MIN_AGE_S or result["seen_count"] >= NEW_REGION_MIN_HITS:
                    ocr = self._ocr_single_region(img, new_bbox, status="new")
                    ocr["region_id"] = int(result["region_id"])
                    ocr["anchor_bbox"] = list(result["anchor_bbox"])
                    ocr["first_seen_time"] = float(result["first_seen_time"])
                    ocr["seen_count"] = int(result["seen_count"])
                    ocr["has_ocr"] = True
                    ocr["last_ocr_time"] = now
                    ocr = self._apply_stability(matched_region, ocr)
                    results.append(ocr)
                else:
                    # Still pending; keep it but don't OCR
                    result["status"] = "new_pending"
                    result = self._advance_stability_no_ocr(matched_region, result)
                    results.append(result)
                continue

            # If it already has OCR, keep your existing "changed or elapsed" behavior
            changed = False
            if prev_img is not None:
                try:
                    px, py, pw, ph = map(int, matched_region["bbox"])
                    old_crop = prev_img[py:py + ph, px:px + pw]
                    new_crop = img[y:y + h, x:x + w]
                    if old_crop.size > 0 and new_crop.size > 0:
                        old_resized = cv2.resize(old_crop, (w, h))
                        diff = cv2.absdiff(old_resized, new_crop)
                        if diff.mean() > CHANGE_THRESHOLD:
                            changed = True
                except Exception:
                    changed = True

            elapsed = now - float(matched_region.get("last_ocr_time", 0.0))
            if changed or elapsed >= OCR_INTERVAL:
                ocr = self._ocr_single_region(img, new_bbox, status="existing")
                ocr["region_id"] = int(result["region_id"])
                ocr["anchor_bbox"] = list(result["anchor_bbox"])
                ocr["first_seen_time"] = float(result["first_seen_time"])
                ocr["seen_count"] = int(result["seen_count"])
                ocr["has_ocr"] = True
                ocr["last_ocr_time"] = now
                ocr = self._apply_stability(matched_region, ocr)
                results.append(ocr)
            else:
                result["last_ocr_time"] = float(matched_region.get("last_ocr_time", now))
                result = self._advance_stability_no_ocr(matched_region, result)
                results.append(result)

        # ------------------------------------------------------------------
        # Keepalive for stable regions that temporarily disappear
        # (e.g., transient zoom/blow-up). We only drop them after they have
        # been missing for MISSING_CONFIRM_FRAMES consecutive segmentations.
        # ------------------------------------------------------------------
        try:
            prev_list = getattr(self, "prev_regions", []) or []
            for idx, prev in enumerate(prev_list):
                if idx in used_prev_idxs:
                    continue

                disp = str(prev.get("display_label", prev.get("label", "diagram")))
                min_age = self._keepalive_min_age(prev)
                is_keepalive = (
                    disp in ("text", "equation", "diagram")
                    and int(prev.get("age", 1)) >= min_age
                )

                if not is_keepalive:
                    continue

                if self._is_absorbed_by_active_region(prev, results):
                    continue

                ghost = prev.copy()
                ghost["missing_streak"] = int(prev.get("missing_streak", 0)) + 1
                ghost["status"] = "missing"
                ghost.setdefault("anchor_bbox", self._reference_bbox(prev))

                if ghost["missing_streak"] <= self._missing_confirm_limit(prev):
                    # Hold last known display label/content
                    ghost["label"] = disp
                    ghost["display_label"] = disp
                    ghost["flip_streak"] = 0
                    results.append(ghost)
        except Exception:
            pass

        if self.occluder_present or recovered_region_confirmed:
            self.recovery_frames_remaining = RECOVERY_GRACE_FRAMES
        elif self.recovery_frames_remaining > 0:
            self.recovery_frames_remaining -= 1


        return results




    
    def _update_region_state(self, prev_region: dict, cur_region: dict, raw_known: bool) -> dict:
        """Update persistence fields (raw_label age) and debounced display label.

        raw_known=True  -> cur_region contains a newly computed raw_label for this frame (from OCR/classifier)
        raw_known=False -> carry forward prev_region raw_label (no OCR this frame)

        Fields:
          - raw_label: instantaneous classifier output (or carried forward)
          - age: consecutive frames where raw_label stayed the same
          - display_label: what we currently show downstream
          - flip_streak: consecutive frames candidate_label disagrees with display_label
          - label: alias of display_label for downstream compatibility
        """
        prev_raw = str(prev_region.get("raw_label", prev_region.get("label", "diagram")))
        prev_age = int(prev_region.get("age", 1))

        if raw_known:
            cur_raw = str(cur_region.get("raw_label", cur_region.get("label", "diagram")))
        else:
            cur_raw = prev_raw

        # Update age based on raw_label stability
        if cur_raw == prev_raw:
            age = prev_age + 1
        else:
            age = 1

        cur_region["raw_label"] = cur_raw
        cur_region["age"] = age

        # Candidate label (promotion delay): force diagram until region is old enough
        candidate_label = "diagram" if age < STABLE_LABEL_MIN_AGE else cur_raw

               # Debounce label flips (including text->diagram and diagram->text)
        prev_display = str(prev_region.get("display_label", prev_region.get("label", "diagram")))
        prev_flip = int(prev_region.get("flip_streak", 0))

        # -----------------------------
        # NEW: equation stickiness
        # If we're currently displaying "equation", require more consecutive
        # evidence to demote away from equation.
        # -----------------------------
        promote_confirm = FLIP_CONFIRM_FRAMES
        demote_confirm = FLIP_CONFIRM_FRAMES

        if prev_display == "equation" and candidate_label != "equation":
            demote_confirm = FLIP_CONFIRM_FRAMES * 2   # e.g., 10 frames instead of 5

        # Optional micro-guard: ignore low-confidence demotions when raw is known
        # (keeps equations from flickering due to garbage OCR reads)
        if raw_known and prev_display == "equation" and candidate_label != "equation":
            cur_conf = float(cur_region.get("confidence", 0.0))
            if cur_conf < 0.70:
                candidate_label = "equation"

        if candidate_label == prev_display:
            display_label = prev_display
            flip_streak = 0
        else:
            flip_streak = prev_flip + 1
            needed = demote_confirm if (prev_display == "equation" and candidate_label != "equation") else promote_confirm
            if flip_streak >= needed:
                display_label = candidate_label
                flip_streak = 0
            else:
                display_label = prev_display


        
        # Persist stability state
        cur_region["display_label"] = display_label
        cur_region["flip_streak"] = flip_streak
        # What downstream sees
        cur_region["label"] = display_label
        return cur_region

    def _apply_stability(self, prev_region: dict, cur_region: dict) -> dict:
        # OCR/classifier ran this frame; raw_label is known
        return self._update_region_state(prev_region, cur_region, raw_known=True)

    def _advance_stability_no_ocr(self, prev_region: dict, cur_region: dict) -> dict:
        # No OCR/classifier this frame; carry forward raw_label
        return self._update_region_state(prev_region, cur_region, raw_known=False)


    def _ocr_single_region(self, img, box, status):
        # Run OCR inference on one cropped region and log timing details
        x, y, w, h = box
        crop = img[y:y+h, x:x+w]
        tmp_path = "_tmp_crop.png"
        cv2.imwrite(tmp_path, crop)

        start_time = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        try:
            if self.text_ocr_backend == "trocr_endpoint":
                response = self._call_text_ocr_endpoint(crop)
                elapsed = time.perf_counter() - start_time
                text = str(response.get("prediction_text", "") or "")
                conf = response.get("confidence", response.get("prediction_confidence", 0.0))
                try:
                    conf = float(conf) if conf is not None else 0.0
                except (TypeError, ValueError):
                    conf = 0.0
            else:
                output = self._ensure_local_recognizer().predict(tmp_path)
                elapsed = time.perf_counter() - start_time

                # Parse recognition result
                result = output[0] if isinstance(output[0], dict) else vars(output[0])
                text = result.get("rec_text", "")
                conf = result.get("rec_score", 0.0)

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            text, conf = "", 0.0
            self._log_error(
                f"{timestamp} OCR failed ({x},{y},{w},{h}) after {elapsed:.3f}s: {e}"
            )
            text, conf = "", 0.0

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        label = self._classify(text, conf, crop=crop, box=box)

        if self.verbose_ocr:
            if self.logger is not None and hasattr(self.logger, "ocr_trace"):
                self.logger.ocr_trace((x, y, w, h), elapsed, label, conf, text[:64])
            else:
                self._log_info(
                    f"{timestamp} OCR region ({x},{y},{w},{h}) took {elapsed:.3f}s | "
                    f"label={label} | conf={conf:.3f} | text=\"{text[:64]}\""
                )
        
        return {
            "bbox": [x, y, w, h],
            "raw_label": label,
            "label": label,
            "confidence": round(conf, 3),
            "text_preview": text,
            "text_ocr_backend": self.text_ocr_backend,
            "text_ocr_model": self.text_ocr_model,
            "status": status,
            "timestamp": timestamp,
            "ocr_time_s": round(elapsed, 3)
        }

    def _call_text_ocr_endpoint(self, crop):
        ok, buffer = cv2.imencode(".png", crop)
        if not ok:
            raise RuntimeError("Failed to encode text OCR crop as PNG.")

        params = urllib.parse.urlencode({
            "model": self.text_ocr_model,
        })
        request = urllib.request.Request(
            f"{self.text_ocr_url}?{params}",
            data=buffer.tobytes(),
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(request, timeout=30.0) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    def _score_regions(self, img, boxes, status="new"):
        # Run OCR on all regions with a uniform status (used for first frame)
        scored = []
        for box in boxes:
            region = self._ocr_single_region(img, box, status)
            region["region_id"] = self._next_region_id
            region["anchor_bbox"] = [int(v) for v in box]
            self._next_region_id += 1
            scored.append(region)
        return scored

    def _classify(self, text, conf, crop = None, box=None):
        # Lightweight content classifier based on recognized text and confidence\
        if box is not None:
            _, _, w, h = map(int, box)
            if w > 0 and h > 0:
                aspect = max(w / float(h), h / float(w))
                if aspect >= 10 and min(w, h) >= 6:
                    return "diagram"
        if crop is not None : # probable equation
            if any(sym in text for sym in ["=", "+", "-", "∫", "√", "π", "Σ", "\\", "$", "÷", "??"]):
                return "equation"
            if self._detect_fraction_bar(crop):
                return "equation"
        if (conf > 0.65) & (len(text) >= 2): # lower bound probable text, likely not equation but might be diagram 
            return "text"
        
        return "diagram"

    def _visualize(self, img, regions):
        # Draw bounding boxes and class labels on the image (unchanged)
        vis = img.copy()
        for r in regions:
            x, y, w, h = r["bbox"]
            label = r["label"]
            conf = r["confidence"]
            color = (
                (0, 255, 0) if label == "text" else
                (255, 0, 0) if label == "equation" else
                (255, 0, 255)
            )
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis, f"{label} ({conf:.2f})",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        return vis

    def _save_results(self, data, path):
        # Write detected regions and metadata to JSON
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        if self.log_saved_results:
            self._log_info(f"Saved {len(data)} regions to {path}")


    def _detect_fraction_bar(self, crop):
        """
        DEBUG VERSION
        Detects a long, near-continuous horizontal fraction bar.
        Prints detailed diagnostics for every rejection.
        """

        import cv2
        import numpy as np

        debug = False
        import builtins
        log = builtins.print if debug else (lambda *args, **kwargs: None)
        print = log
        log("\n================ FRACTION BAR DEBUG ================")
        log(f"[FB] Crop shape: {crop.shape}")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(blur, 60, 160)

        h, w = edges.shape
        log(f"[FB] Edge image size: w={w}, h={h}")

        # ---- Tunables ----
        MIN_LEN_FRAC = 0.60
        MAX_GAP_PX = 10
        MAX_SLOPE = 0.15
        CENTER_BAND = (0.20, 0.80)

        log(f"[FB] Params:")
        log(f"     MIN_LEN_FRAC={MIN_LEN_FRAC}")
        log(f"     MAX_GAP_PX={MAX_GAP_PX}")
        log(f"     MAX_SLOPE={MAX_SLOPE}")
        log(f"     CENTER_BAND={CENTER_BAND}")

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            minLineLength=int(w * MIN_LEN_FRAC),
            threshold=int((w * MIN_LEN_FRAC) * 0.3 - 0.5),
            maxLineGap=MAX_GAP_PX
        )

        if lines is None:
            log("[FB] [NO] No Hough lines detected")
            return False

        log(f"[FB] Hough returned {len(lines)} candidate lines")

        def continuity_score(edges_img, y, x1, x2, band=2):
            x1, x2 = sorted((int(x1), int(x2)))
            y = int(round(y))
            y0 = max(0, y - band)
            y1 = min(edges_img.shape[0], y + band + 1)
            strip = edges_img[y0:y1, x1:x2]

            if strip.size == 0:
                return 0.0

            col_has_edge = (strip > 0).any(axis=0)
            score = col_has_edge.mean()
            return score

        for idx, (x1, y1, x2, y2) in enumerate(lines[:, 0]):
            log(f"\n[FB] ---- Candidate {idx} ----")
            log(f"[FB] Raw line: ({x1},{y1}) -> ({x2},{y2})")

            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            slope = abs(dy / (dx + 1e-6))
            mid_y = (y1 + y2) / 2.0

            log(f"[FB] Length: {length:.1f}px (min {w * MIN_LEN_FRAC:.1f}px)")
            if length < w * MIN_LEN_FRAC:
                log("[FB] [NO] Rejected: too short")
                continue

            log(f"[FB] Slope: {slope:.3f} (max {MAX_SLOPE})")
            if slope > MAX_SLOPE:
                log("[FB] [NO] Rejected: too slanted")
                continue

            log(f"[FB] Mid Y: {mid_y:.1f}px (allowed {h * CENTER_BAND[0]:.1f}-{h * CENTER_BAND[1]:.1f})")
            if not (h * CENTER_BAND[0] <= mid_y <= h * CENTER_BAND[1]):
                log("[FB] [NO] Rejected: outside center band")
                continue

            cont = continuity_score(edges, mid_y, x1, x2, band=2)
            log(f"[FB] Continuity score: {cont:.3f} (min 0.85)")

            if cont < 0.85:
                log("[FB] [NO] Rejected: insufficient continuity")
                continue

            log("[FB] [YES] FRACTION BAR DETECTED")
            return True

        log("[FB] [NO] No candidates passed all checks")
        return False
