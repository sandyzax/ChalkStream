"""Load ChalkStream application defaults from a commented TOML settings file."""

from __future__ import annotations

import copy
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_SETTINGS_PATH = REPO_ROOT / "chalkstream_settings.toml"


DEFAULT_APP_SETTINGS_TEXT = """# ChalkStream settings
#
# This file holds the default processing settings used when the application starts.
# The values are grouped by module so you can see which part of the system each
# parameter belongs to. Edit the file, save it, and restart the app to apply the
# new defaults.

[preprocessing]
# Background subtractor history. Larger values make the model adapt more slowly.
history = 50

# Sensitivity of motion subtraction. Higher values make the subtractor less eager
# to mark motion.
var_threshold = 25.0

# MediaPipe person-mask threshold. Lower values include more pixels in the person mask.
person_mask_threshold = 0.4

# Erosion kernel used to shrink the person mask before compositing the board back in.
person_mask_erosion_kernel = 18

# Dilation kernel used to expand the motion mask before restoring the previous board state.
motion_mask_dilation_kernel = 18

[segmentation]
# Morphological close kernel width as a fraction of the board width.
kernel_width_frac = 0.025

# Morphological close kernel height as a fraction of the board height.
kernel_height_frac = 0.025

# Minimum candidate box area as a fraction of the board area.
min_area_pct = 0.0005

# Maximum candidate box area as a fraction of the board area.
max_area_pct = 0.50

# Seconds between lightweight OCR refreshes for an existing region.
ocr_interval_s = 5.0

# Text OCR backend for per-region recognition. "paddle" keeps recognition local,
# while "trocr_endpoint" sends the crop to the TrOCR server.
text_ocr_backend = "paddle"

# TrOCR endpoint used when text_ocr_backend = "trocr_endpoint".
text_ocr_url = "http://127.0.0.1:8010/infer_text"

# Text OCR model name sent to the TrOCR endpoint.
text_ocr_model = "TextOCR_V1"

# Pixel-difference threshold used when deciding whether a region changed enough to rerun OCR.
change_threshold = 25

# Minimum age in seconds before a newly found region is trusted.
new_region_min_age_s = 1.0

# Minimum number of segmentation hits before a newly found region is trusted.
new_region_min_hits = 2

# Number of segmentations a non-diagram label must persist before it is promoted.
stable_label_min_age = 5

# Number of confirmations needed before switching a displayed label.
flip_confirm_frames = 5

# Minimum age before a text or equation region qualifies for keepalive.
keepalive_min_age = 5

# Minimum age before a diagram region qualifies for keepalive.
diagram_keepalive_min_age = 2

# Base number of consecutive missing segmentations before a stable region is dropped.
missing_confirm_frames = 5

# Number of segmentation passes to keep recovery mode active after a reacquisition.
recovery_grace_frames = 4

# Hard ceiling for the missing confirmation window on very stable regions.
max_missing_confirm_frames = 12

# How quickly missing tolerance grows with region age.
missing_confirm_age_step = 4

# How quickly missing tolerance grows with repeated observations.
missing_confirm_seen_step = 8

# Ignore gigantic regions that cover this much of the board area.
monster_area_frac = 0.85

# IoU threshold used to match a new candidate box to a previously tracked region.
match_iou_thresh = 0.35

# Recovery path: maximum normalized center distance when reclaiming a region ID.
recovery_center_dist_frac = 0.75

# Recovery path: minimum area-ratio similarity when reclaiming a region ID.
recovery_area_ratio_min = 0.18

# Recovery path: minimum aspect-ratio similarity when reclaiming a region ID.
recovery_aspect_ratio_min = 0.20

# Recovery path: minimum blended recovery score needed to reclaim a region ID.
recovery_score_min = 0.45

# Minimum IoU before a recovered observation updates the trusted anchor box.
anchor_update_iou_min = 0.45

# Maximum normalized center distance before a recovered observation can update the anchor box.
anchor_update_center_dist_frac = 0.35

# Minimum area-ratio similarity before a recovered observation can update the anchor box.
anchor_update_area_ratio_min = 0.50

# Minimum aspect-ratio similarity before a recovered observation can update the anchor box.
anchor_update_aspect_ratio_min = 0.45

# Smoothing factor when blending a new observation into the anchor box.
anchor_smoothing = 0.35

# Minimum IoU before an unmatched candidate is treated as a duplicate near an existing anchor.
duplicate_suppression_iou = 0.50

# Maximum normalized center distance for duplicate suppression.
duplicate_suppression_center_dist_frac = 0.30

# Minimum area-ratio similarity for duplicate suppression.
duplicate_suppression_area_ratio_min = 0.55

# Minimum aspect-ratio similarity for duplicate suppression.
duplicate_suppression_aspect_ratio_min = 0.45

# Overlap threshold used when merging heavily overlapping candidate boxes before tracking.
box_merge_overlap_min = 0.60

# Minimum area-ratio similarity used during candidate box merging.
box_merge_area_ratio_min = 0.12

# If this much of the smaller box overlaps a larger one, absorb it even when the
# size ratio would normally block the merge.
box_merge_containment_min = 0.35

# Large-region area fraction used when deciding whether a blob is really an occluder.
occluder_area_frac = 0.45

# Number of previous regions a blob must overlap before it is treated as an occluder.
occluder_overlap_count = 3

# Minimum IoU used when counting overlap against previous regions for occlusion detection.
occluder_iou_min = 0.12

# If this much of an older region is covered by a larger active region, drop the
# stale missing track instead of keeping both references alive.
absorb_containment_min = 0.35

[orchestration]
# Equation OCR endpoint used for stable text and equation regions.
equation_ocr_url = "http://127.0.0.1:8010/infer"

# Equation OCR model name sent to the endpoint.
equation_ocr_model = "Trained_V3"

# Number of segmentation passes between equation OCR attempts on the same region.
equation_ocr_segment_interval = 4

# Minimum region age before equation OCR is allowed to run.
equation_ocr_min_region_age = 2

# Confidence threshold required before equation OCR is rendered as LaTeX.
equation_ocr_min_confidence = 0.70

# Confidence threshold required before equation OCR can override a text label to equation.
equation_ocr_label_override_confidence = 0.95

# HTTP timeout for the equation OCR request, in seconds.
equation_ocr_timeout_s = 120.0

# Maximum generation length sent to the equation OCR endpoint.
equation_ocr_max_gen_length = 128

# Beam count sent to the equation OCR endpoint.
equation_ocr_num_beams = 1

# Diagram vectorization method used for diagram and fallback math rendering.
diagram_vectorization_method = "v2"

# Minimum time between segmentation passes.
segment_interval_s = 0.5

# Minimum time between preview-frame updates to the GUI.
preview_interval_s = 0.1

# How many segmentation passes occur between summary log lines.
console_summary_interval_segments = 4

# If true, print per-region OCR traces in addition to the compact summaries.
verbose_region_ocr = false

[camera]
# Default live camera resolution. Use "max" to negotiate the highest working mode.
resolution_option = "max"

# Highest camera index the source picker will probe when listing available cameras.
max_discovery_index = 5
"""


def _default_settings_dict():
    return {
        "preprocessing": {
            "history": 50,
            "var_threshold": 25.0,
            "person_mask_threshold": 0.4,
            "person_mask_erosion_kernel": 18,
            "motion_mask_dilation_kernel": 18,
        },
        "segmentation": {
            "kernel_width_frac": 0.025,
            "kernel_height_frac": 0.025,
            "min_area_pct": 0.0005,
            "max_area_pct": 0.50,
            "ocr_interval_s": 5.0,
            "text_ocr_backend": "paddle",
            "text_ocr_url": "http://127.0.0.1:8010/infer_text",
            "text_ocr_model": "TextOCR_V1",
            "change_threshold": 25,
            "new_region_min_age_s": 1.0,
            "new_region_min_hits": 2,
            "stable_label_min_age": 5,
            "flip_confirm_frames": 5,
            "keepalive_min_age": 5,
            "diagram_keepalive_min_age": 2,
            "missing_confirm_frames": 5,
            "recovery_grace_frames": 4,
            "max_missing_confirm_frames": 12,
            "missing_confirm_age_step": 4,
            "missing_confirm_seen_step": 8,
            "monster_area_frac": 0.85,
            "match_iou_thresh": 0.35,
            "recovery_center_dist_frac": 0.75,
            "recovery_area_ratio_min": 0.18,
            "recovery_aspect_ratio_min": 0.20,
            "recovery_score_min": 0.45,
            "anchor_update_iou_min": 0.45,
            "anchor_update_center_dist_frac": 0.35,
            "anchor_update_area_ratio_min": 0.50,
            "anchor_update_aspect_ratio_min": 0.45,
            "anchor_smoothing": 0.35,
            "duplicate_suppression_iou": 0.50,
            "duplicate_suppression_center_dist_frac": 0.30,
            "duplicate_suppression_area_ratio_min": 0.55,
            "duplicate_suppression_aspect_ratio_min": 0.45,
            "box_merge_overlap_min": 0.60,
            "box_merge_area_ratio_min": 0.12,
            "box_merge_containment_min": 0.35,
            "occluder_area_frac": 0.45,
            "occluder_overlap_count": 3,
            "occluder_iou_min": 0.12,
            "absorb_containment_min": 0.35,
        },
        "orchestration": {
            "equation_ocr_url": "http://127.0.0.1:8010/infer",
            "equation_ocr_model": "Trained_V3",
            "equation_ocr_segment_interval": 4,
            "equation_ocr_min_region_age": 2,
            "equation_ocr_min_confidence": 0.70,
            "equation_ocr_label_override_confidence": 0.95,
            "equation_ocr_timeout_s": 120.0,
            "equation_ocr_max_gen_length": 128,
            "equation_ocr_num_beams": 1,
            "diagram_vectorization_method": "v2",
            "segment_interval_s": 0.5,
            "preview_interval_s": 0.1,
            "console_summary_interval_segments": 4,
            "verbose_region_ocr": False,
        },
        "camera": {
            "resolution_option": "max",
            "max_discovery_index": 5,
        },
    }


def ensure_app_settings_file(path: Path | None = None) -> Path:
    """Create the settings file with comments if it does not already exist."""
    settings_path = Path(path or APP_SETTINGS_PATH)
    if not settings_path.exists():
        settings_path.write_text(DEFAULT_APP_SETTINGS_TEXT, encoding="utf-8")
    return settings_path


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in dict(overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_app_settings(path: Path | None = None) -> dict:
    """Load the TOML settings file and merge it onto the built-in defaults."""
    settings_path = ensure_app_settings_file(path)
    defaults = _default_settings_dict()
    with settings_path.open("rb") as handle:
        file_settings = tomllib.load(handle)
    return _deep_merge(defaults, file_settings)
